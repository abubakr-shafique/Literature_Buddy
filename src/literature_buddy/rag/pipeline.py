"""RAG pipeline (spec §8): route -> retrieve -> rerank -> context -> generate.

Visual questions ('What does Figure 2 show?') are routed to the VLM with the
retrieved figure image + caption as grounding context, minimizing visual
hallucination (spec §6).
"""

from __future__ import annotations

import re
from typing import Iterator

from literature_buddy.document.models import ParsedDocument
from literature_buddy.models.backends import ModelManager
from literature_buddy.rag import context_builder, prompts
from literature_buddy.rag.citations import Citation, citations_from_hits
from literature_buddy.rag.memory import ConversationMemory
from literature_buddy.retrieval.retriever import Retriever, RetrievedChunk

VISUAL_Q_RE = re.compile(
    r"\b(fig(?:ure)?|table|plot|graph|chart|image|panel)\b", re.IGNORECASE
)


class RAGPipeline:
    def __init__(
        self,
        doc: ParsedDocument,
        retriever: Retriever,
        manager: ModelManager,
        memory: ConversationMemory,
        context_tokens: int = 6_000,
    ):
        self.doc = doc
        self.retriever = retriever
        self.manager = manager
        self.memory = memory
        self.context_tokens = context_tokens

    # -- public API ---------------------------------------------------------
    def answer(
        self,
        question: str,
        extra_context: list[str] | None = None,
    ) -> tuple[str, list[Citation]]:
        """Answer a question using RAG over the current paper.

        - `extra_context`: optional additional strings (e.g. user notes or
          external excerpts) to be appended to the RAG context in text mode.
        """
        visual = bool(VISUAL_Q_RE.search(question))
        hits = self.retriever.retrieve(self.doc.slug, question)
        self.memory.add("user", question)

        # Prefer visual path only if we actually have a visual chunk
        use_visual = visual and any(h.is_visual for h in hits.chunks)

        if use_visual:
            text = self._answer_visual(question, hits.chunks)
        else:
            text = self._answer_text(question, hits.chunks, extra_context)

        citations = citations_from_hits(hits.chunks)
        self.memory.add("assistant", text)
        return text, citations

    def stream_answer(
        self,
        question: str,
        extra_context: list[str] | None = None,
    ) -> Iterator[tuple[str, list[Citation] | None]]:
        """Stream an LLM answer token-by-token, then yield citations at the end.

        Streaming currently uses the text-only path (LLM), even if the query
        mentions figures/tables. If you want streaming + VLM, that requires a
        different design (e.g. streaming caption + image tokens).
        """
        hits = self.retriever.retrieve(self.doc.slug, question)
        self.memory.add("user", question)

        context = self._format_context(hits.chunks, extra_context)
        messages = self._messages(context, question)

        llm = self.manager.llm()
        chunks: list[str] = []
        for tok in llm.stream(messages, max_tokens=2_048, temperature=0.1):
            chunks.append(tok)
            yield tok, None

        text = "".join(chunks)
        self.memory.add("assistant", text)
        yield "", citations_from_hits(hits.chunks)

    # -- internals ----------------------------------------------------------
    def _answer_text(
        self,
        question: str,
        hits: list[RetrievedChunk],
        extra_context: list[str] | None = None,
    ) -> str:
        context = self._format_context(hits, extra_context)
        llm = self.manager.llm()
        return llm.generate(
            self._messages(context, question),
            max_tokens=2_048,
            temperature=0.1,
        )

    def _answer_visual(self, question: str, hits: list[RetrievedChunk]) -> str:
        # Find the best visual chunk (highest score among visual hits)
        visual_hits = [h for h in hits if h.is_visual]
        if not visual_hits:
            # Fallback to text path if something went wrong
            return self._answer_text(question, hits)

        visual = max(visual_hits, key=lambda h: h.score)
        fig = visual.chunk

        if not fig.figure_image_path:
            # No image available; degrade gracefully to text-only with caption
            caption_context = [f"{fig.figure_label} (p.{fig.page}): {fig.caption}"]
            return self._answer_text(
                f"{question} (Focus on {fig.figure_label}.)",
                hits,
                extra_context=caption_context,
            )

        grounding = (
            f"You are shown {fig.figure_label} (page {fig.page}) from a scientific paper.\n"
            f"Author caption (ground truth): \"{fig.caption}\"\n"
            "Describe ONLY what is visible or explicitly supported by the caption. "
            "If the figure cannot be confidently interpreted, say so explicitly "
            "instead of guessing."
        )

        vlm = self.manager.vlm()
        return vlm.analyze_image(
            f"{grounding}\n\nQuestion: {question}",
            fig.figure_image_path,
        )

    def _format_context(
        self,
        hits: list[RetrievedChunk],
        extra_context: list[str] | None = None,
    ) -> str:
        ctx, used = context_builder.build_context(hits, self.context_tokens)

        if extra_context:
            extra_block = "\n\nADDITIONAL CONTEXT (user-provided):\n" + "\n".join(
                f"- {s}" for s in extra_context
            )
            ctx = ctx + extra_block if ctx else extra_block

        return f"{prompts.CONTEXT_HEADER}\n\n{ctx}"

    def _messages(self, context: str, question: str) -> list[dict]:
        return (
            [{"role": "system", "content": prompts.SYSTEM_RAG}]
            + self.memory.messages()
            + [
                {
                    "role": "user",
                    "content": prompts.PROMPT.format(
                        context=context, question=question
                    ),
                }
            ]
        )