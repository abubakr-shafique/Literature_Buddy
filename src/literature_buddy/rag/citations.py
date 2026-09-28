"""Citation cards for the GUI; links use lb://page/<n> anchors handled by MainWindow."""

from __future__ import annotations

from dataclasses import dataclass

from literature_buddy.retrieval.retriever import RetrievedChunk


@dataclass
class Citation:
    label: str   # e.g. "Figure 2, p.4"
    page: int
    snippet: str

    def as_html(self) -> str:
        return (f'<a href="lb://page/{self.page}">▸ {self.label}</a>'
                f'<br><small>{self.snippet[:160].strip()}…</small>')


def citations_from_hits(hits: list[RetrievedChunk], max_n: int = 4) -> list[Citation]:
    out, seen = [], set()
    for h in hits:
        c = h.chunk
        if c.figure_label:
            label = f"{c.figure_label}, p.{c.page}"
        elif c.table_label:
            label = f"{c.table_label}, p.{c.page}"
        else:
            sec = c.section_title or c.section or "Text"
            label = f"{sec}, p.{c.page}"
        if label in seen:
            continue
        seen.add(label)
        out.append(Citation(label=label, page=c.page, snippet=c.text))
        if len(out) >= max_n:
            break
    return out