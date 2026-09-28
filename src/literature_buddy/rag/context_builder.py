"""Budgeted context assembly: relevant text + figures + tables."""

from __future__ import annotations

from literature_buddy.retrieval.retriever import RetrievedChunk


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_context(hits: list[RetrievedChunk], token_budget: int = 6_000) -> tuple[str, list[RetrievedChunk]]:
    """Prioritize visual chunks (figure captions) if any present, then score order.
    Returns (formatted_context, used_hits)."""
    visual = [h for h in hits if h.is_visual]
    prose = [h for h in hits if not h.is_visual]
    ordered = visual + prose

    used, parts, budget = [], [], token_budget
    for h in ordered:
        loc = _loc(h)
        block = f"[{loc}] {h.chunk.text}"
        t = estimate_tokens(block)
        if t > budget and used:
            break
        parts.append(block); used.append(h); budget -= t
    return "\n\n".join(parts), used


def _loc(h: RetrievedChunk) -> str:
    c = h.chunk
    if c.figure_label:
        return f"{c.figure_label}, p.{c.page}"
    if c.table_label:
        return f"{c.table_label}, p.{c.page}"
    sec = c.section_title or c.section
    return f"p.{c.page}, {sec}" if sec else f"p.{c.page}"