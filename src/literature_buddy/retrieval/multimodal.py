"""Multimodal evidence selection: decide which figures/tables/equations accompany the text.

Three sources, in priority order:
  1. explicit references in the question ("What does Figure 3 show?")
  2. figure/table chunks that the retriever itself returned
  3. figures/tables *mentioned by* the retrieved text chunks (text -> visual linking)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..document.schema import Document, Equation, Figure, Table, find_mentions
from .retriever import Hit


@dataclass
class VisualEvidence:
    item: Figure | Table | Equation
    reason: str  # "asked" | "retrieved" | "mentioned"


def select_visuals(
    doc: Document, question: str, hits: list[Hit], max_visuals: int = 3
) -> list[VisualEvidence]:
    out: list[VisualEvidence] = []
    seen: set[str] = set()

    def push(label: str, reason: str) -> None:
        if label in seen or len(out) >= max_visuals:
            return
        item = doc.visual(label)
        if item is not None:
            seen.add(label)
            out.append(VisualEvidence(item, reason))

    for label in find_mentions(question):
        push(label, "asked")
    for h in hits:
        if h.chunk.kind in ("figure", "table") and h.chunk.label:
            push(h.chunk.label, "retrieved")
    for h in hits:
        if h.chunk.kind == "text":
            for label in h.chunk.refs:
                if label.startswith(("Figure", "Table")):
                    push(label, "mentioned")
    return out
