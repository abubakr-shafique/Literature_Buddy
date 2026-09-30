"""Source records, citation validation and quote verification (the anti-hallucination layer)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ID_GROUP = re.compile(r"\[\s*((?:[SH]\d+)(?:\s*[,;]\s*[SH]\d+)*)\s*\]")
_ID = re.compile(r"[SH]\d+")
_QUOTE = re.compile(r"[\"\u201c]([^\"\u201c\u201d]{25,400})[\"\u201d]")


@dataclass
class Source:
    sid: str  # S1, S2, ... or H1 (user highlight)
    kind: str  # text | abstract | figure | table | equation | reference | highlight
    page: int  # 0-based
    section: str = ""
    label: str | None = None
    text: str = ""  # full text given to the model (also used for quote verification)
    chunk_id: str | None = None
    image_path: str | None = None  # relative to the paper directory
    note: str = ""  # e.g. MODEL-GENERATED visual analysis (not part of the paper)

    @property
    def title(self) -> str:
        if self.kind == "highlight":
            return f"Highlight, p.{self.page + 1}"
        if self.label:
            return f"{self.label}, p.{self.page + 1}"
        return f"{self.section or 'Text'}, p.{self.page + 1}"

    @property
    def excerpt(self) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= 220 else t[:217].rstrip() + "..."

    def to_dict(self) -> dict:
        return {"sid": self.sid, "kind": self.kind, "page": self.page, "section": self.section,
                "label": self.label, "excerpt": self.excerpt, "image_path": self.image_path,
                "title": self.title, "chunk_id": self.chunk_id}


def normalize_markers(text: str) -> str:
    """'[S1, S2]' -> '[S1][S2]' so every marker is a single id."""
    return _ID_GROUP.sub(lambda m: "".join(f"[{i}]" for i in _ID.findall(m.group(1))), text)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def validate_citations(answer: str, sources: list[Source]) -> tuple[str, list[str], list[str]]:
    """Strip markers that do not correspond to a real source. Returns (text, used_ids, warnings)."""
    valid = {s.sid for s in sources}
    used: list[str] = []
    bad: set[str] = set()
    text = normalize_markers(answer)

    def repl(m: re.Match[str]) -> str:
        sid = m.group(1)
        if sid in valid:
            if sid not in used:
                used.append(sid)
            return m.group(0)
        bad.add(sid)
        return ""

    text = re.sub(r"\[([SH]\d+)\]", repl, text)
    warnings = [f"Removed citation to non-existent source {', '.join(sorted(bad))}."] if bad else []
    return text, used, warnings


def unverified_quotes(answer: str, sources: list[Source]) -> list[str]:
    """Quoted passages (25+ chars) that do not appear verbatim (after normalisation) in any source."""
    haystack = " ".join(_norm(s.text) for s in sources if s.kind != "figure_note")
    out = []
    for m in _QUOTE.finditer(answer):
        q = _norm(m.group(1))
        if q and q not in haystack:
            out.append(m.group(1).strip())
    return out
