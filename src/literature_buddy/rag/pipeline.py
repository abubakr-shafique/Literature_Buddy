"""The RAG pipeline: understand -> rewrite -> hybrid retrieve -> visuals (VLM) -> context -> answer."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..config.settings import AppConfig
from ..document.processor import ProcessedPaper
from ..document.schema import Figure, Table
from ..models.base import ChatMessage, prepare_image
from ..models.model_manager import ModelManager
from ..retrieval.multimodal import VisualEvidence, select_visuals
from ..retrieval.retriever import HybridRetriever
from . import prompts
from .citations import Source, unverified_quotes, validate_citations
from .context_builder import build_context
from .intent import classify
from .memory import ConversationMemory

log = logging.getLogger(__name__)
_FOLLOWUP = re.compile(r"\b(it|its|they|them|their|this|that|these|those|he|she|the same|former|latter|"
                       r"there|then|also|and)\b", re.I)


@dataclass
class AnswerResult:
    text: str
    sources: list[Source]
    used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    intent: str = "general"
    query: str = ""
    stopped: bool = False

    @property
    def display_sources(self) -> list[Source]:
        cited = [s for s in self.sources if s.sid in self.used]
        return cited or self.sources[:3]


@dataclass
class Event:
    kind: str  # status | token | done
    text: str = ""
    result: AnswerResult | None = None


class RagPipeline:
    def __init__(self, cfg: AppConfig, paper: ProcessedPaper, retriever: HybridRetriever,
                 models: ModelManager, memory: ConversationMemory | None = None) -> None:
        self.cfg, self.paper, self.retriever, self.models = cfg, paper, retriever, models
        self.memory = memory or ConversationMemory(cfg.conversation.history_tokens)

    # ---- helpers ------------------------------------------------------------------------------
    def _rewrite(self, question: str) -> str:
        if not (self.cfg.conversation.rewrite_queries and self.memory.has_turns):
            return question
        if len(question.split()) > 12 and not _FOLLOWUP.search(question):
            return question
        try:
            out = self.models.llm().generate(
                [ChatMessage("user", prompts.REWRITE_PROMPT.format(history=self.memory.as_text(), question=question))],
                max_new_tokens=80, temperature=0.0)
            out = out.strip().strip('"').splitlines()[0] if out.strip() else ""
            return out if 3 <= len(out) <= 300 else question
        except Exception as exc:  # noqa: BLE001
            log.info("Query rewrite skipped: %s", exc)
            return question

    def _summarizer(self, prev: str, turns: str) -> str:
        return self.models.llm().generate(
            [ChatMessage("user", prompts.SUMMARISE_HISTORY.format(summary=prev or "(none)", turns=turns))],
            max_new_tokens=200, temperature=0.0)

    def _analyse_visuals(self, question: str, visuals: list[VisualEvidence], stop: threading.Event | None,
                         warnings: list[str], intent, hits) -> Iterator[Event | tuple[str, str]]:
        vlm = self.models.vlm()
        if vlm is None:
            if any(v.item.image_path for v in visuals):
                warnings.append("Vision model unavailable: figures were interpreted from captions/text only.")
            return
        done = 0
        # VLM calls are expensive. If the user named a figure/table, analyse only those. Otherwise analyse
        # a retrieved figure only for visual questions or when it ranks in the top 2 (never merely
        # because the text mentions it).
        top = {h.chunk.label for h in hits[:2] if h.chunk.label}
        asked = [v for v in visuals if v.reason == "asked"]
        chosen = asked or [v for v in visuals if v.reason == "retrieved"
                           and (intent.kind == "visual" or v.item.label in top)]
        for v in chosen:
            item = v.item
            if not isinstance(item, (Figure, Table)) or not item.image_path or done >= self.cfg.vlm.max_images:
                continue
            asked = v.reason == "asked"
            key = f"vlm|{item.label}|{'q:' + hashlib.sha1(question.encode()).hexdigest()[:10] if asked else 'general'}|{self.cfg.vlm.model}"
            note = self.paper.store.get_meta(key)
            if note is None:
                yield Event("status", f"Looking at {item.label} with the vision model…")
                img = prepare_image(self.paper.directory / item.image_path, self.cfg.vlm.max_image_side)
                mentions = ""
                for h in self.retriever.store.chunks:
                    if h.kind == "text" and item.label in h.refs:
                        mentions = h.text[:600]
                        break
                ctx = f"Caption from the paper: {item.caption}" + (f"\nText that refers to it: {mentions}" if mentions else "")
                prompt = prompts.FIGURE_QUESTION.format(question=question) if asked else prompts.FIGURE_GENERAL
                try:
                    note = vlm.analyze_image(img, prompt, context=ctx, system=prompts.FIGURE_SYSTEM,
                                             max_new_tokens=450, stop_event=stop)
                    self.paper.store.set_meta(key, note)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"Vision analysis of {item.label} failed: {exc}")
                    continue
            done += 1
            yield (item.label, note)

    # ---- main entry ---------------------------------------------------------------------------
    def ask(self, question: str, highlights: list[dict] | None = None,
            stop_event: threading.Event | None = None) -> Iterator[Event]:
        highlights = highlights or []
        doc = self.paper.document
        intent = classify(question)
        yield Event("status", "Understanding the question…")
        self.memory.compress(self._summarizer if self.cfg.conversation.rolling_summary else None)
        query = self._rewrite(question)

        yield Event("status", "Searching the paper…")
        excl = None if intent.wants_references else {"reference"}
        hits = self.retriever.retrieve(query, exclude_kinds=excl, prefer_sections=intent.prefer_sections or None)
        if intent.kind == "summary":
            abstract = [c for c in self.retriever.store.chunks if c.kind == "abstract"]
            from ..retrieval.retriever import Hit
            hits = [Hit(c, 1.0) for c in abstract if all(h.chunk.chunk_id != c.chunk_id for h in hits)] + hits

        visuals = select_visuals(doc, f"{question} {query}", hits, self.cfg.retrieval.max_visuals)
        warnings: list[str] = []
        notes: dict[str, str] = {}
        if visuals:
            for out in self._analyse_visuals(question, visuals, stop_event, warnings, intent, hits):
                if isinstance(out, Event):
                    yield out
                else:
                    notes[out[0]] = out[1]

        context, sources = build_context(doc, hits, visuals, notes, [h for h in highlights if h.get("use_in_chat", True)],
                                         self.cfg.retrieval.max_context_tokens)
        if not sources:
            res = AnswerResult(prompts.NO_EVIDENCE, [], intent=intent.kind, query=query)
            self.memory.add("user", question)
            self.memory.add("assistant", res.text)
            yield Event("token", res.text)
            yield Event("done", result=res)
            return

        hint = {"critical": prompts.CRITICAL_HINT, "summary": prompts.SUMMARY_HINT}.get(intent.kind, "")
        title = f"Paper: {doc.title}\n" if doc.title else ""
        user = f"{title}SOURCES:\n{context}\n\nQUESTION: {question}" + (f"\n\n{hint}" if hint else "")
        messages = [ChatMessage("system", prompts.SYSTEM_PROMPT), *self.memory.messages(), ChatMessage("user", user)]

        yield Event("status", "Writing the answer…")
        parts: list[str] = []
        for piece in self.models.llm().stream(messages, stop_event=stop_event):
            parts.append(piece)
            yield Event("token", piece)
        raw = "".join(parts).strip()
        stopped = bool(stop_event is not None and stop_event.is_set())

        text, used, warn = validate_citations(raw, sources)
        warnings += warn
        for q in unverified_quotes(text, sources):
            warnings.append(f"Quoted text could not be verified in the paper: “{q[:80]}…”")
        res = AnswerResult(text, sources, used, warnings, intent.kind, query, stopped)
        self.memory.add("user", question)
        self.memory.add("assistant", text)
        yield Event("done", result=res)
