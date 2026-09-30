"""Assemble the evidence block sent to the LLM (text + tables + figure analyses + user highlights)."""

from __future__ import annotations

from ..document.schema import Document, Figure, Table
from ..retrieval.multimodal import VisualEvidence
from ..retrieval.retriever import Hit
from .citations import Source
from .memory import est_tokens


def build_context(
    doc: Document,
    hits: list[Hit],
    visuals: list[VisualEvidence],
    visual_notes: dict[str, str],
    highlights: list[dict],
    max_tokens: int,
) -> tuple[str, list[Source]]:
    sources: list[Source] = []
    used = 0

    def fits(text: str) -> bool:
        nonlocal used
        cost = est_tokens(text)
        if used + cost > max_tokens and sources:
            return False
        used += cost
        return True

    for i, h in enumerate(highlights, 1):
        text = " ".join(h["text"].split())[:1500]
        src = Source(f"H{i}", "highlight", h["page"], "Highlight", text=text)
        if fits(text):
            sources.append(src)

    seen_visual: set[str] = set()
    for v in visuals:
        item = v.item
        if isinstance(item, (Figure, Table)):
            text = item.caption
            if isinstance(item, Table) and item.markdown:
                text += "\n" + item.markdown[:2500]
            note = visual_notes.get(item.label, "")
            src = Source(f"S{len(sources) + 1 - len(highlights)}", "figure" if isinstance(item, Figure) else "table",
                         item.page, item.label.split()[0], item.label, text, None, item.image_path, note)
        else:
            src = Source(f"S{len(sources) + 1 - len(highlights)}", "equation", item.page, "Equation",
                         item.label, f"{item.label}: {item.text}", None, item.image_path)
        if fits(src.text + src.note):
            sources.append(src)
            seen_visual.add(item.label)

    for h in hits:
        c = h.chunk
        if c.label and c.label in seen_visual:
            continue
        src = Source(f"S{len(sources) + 1 - len(highlights)}", c.kind, c.page, c.section, c.label, c.text, c.chunk_id)
        if fits(c.text):
            sources.append(src)

    blocks = []
    for s in sources:
        kind = "user highlight" if s.kind == "highlight" else s.kind
        note = f"\nMODEL-GENERATED VISUAL ANALYSIS (fallible, not part of the paper): {s.note}" if s.note else ""
        blocks.append(f"[{s.sid}] {s.title} ({kind})\n{s.text}{note}")
    return "\n\n".join(blocks), sources
