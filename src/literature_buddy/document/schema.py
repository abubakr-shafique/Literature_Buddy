"""Data model for a parsed paper. Everything here is JSON-serialisable (pydantic v2)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]
SCHEMA_VERSION = 1


class Section(BaseModel):
    idx: int
    title: str
    kind: str = "other"  # front|abstract|introduction|methods|results|discussion|conclusion|...
    level: int = 1
    page_start: int = 0  # 0-based page indices throughout the data model
    page_end: int = 0


class Paragraph(BaseModel):
    page: int
    bbox: BBox
    text: str
    section_idx: int


class Figure(BaseModel):
    label: str  # canonical, e.g. "Figure 3" or "Figure S1"
    number: str
    caption: str
    page: int
    bbox: BBox | None = None  # image region (None if no graphic could be located)
    caption_bbox: BBox | None = None
    image_path: str | None = None  # relative to the paper directory


class Table(BaseModel):
    label: str
    number: str
    caption: str
    page: int
    bbox: BBox | None = None
    caption_bbox: BBox | None = None
    markdown: str = ""
    n_rows: int = 0
    n_cols: int = 0
    image_path: str | None = None
    approximate: bool = False  # True when found with the low-confidence text strategy


class Equation(BaseModel):
    label: str  # "Equation 3"
    number: str
    text: str
    page: int
    bbox: BBox
    image_path: str | None = None


class Reference(BaseModel):
    index: int
    text: str
    page: int


class Document(BaseModel):
    schema_version: int = SCHEMA_VERSION
    build_key: str = ""  # parser version + options; used for cache invalidation
    doc_id: str
    source: str  # original path or URL
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    page_count: int = 0
    sections: list[Section] = Field(default_factory=list)
    paragraphs: list[Paragraph] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def visual(self, label: str) -> Figure | Table | Equation | None:
        """Look up a figure/table/equation by canonical label (case-insensitive)."""
        key = label.strip().lower()
        for item in (*self.figures, *self.tables, *self.equations):
            if item.label.lower() == key:
                return item
        return None

    def section_of(self, idx: int) -> Section | None:
        return self.sections[idx] if 0 <= idx < len(self.sections) else None

    def outline(self) -> str:
        lines = []
        for s in self.sections:
            if s.kind == "front":
                continue
            lines.append(f"{'  ' * max(s.level - 1, 0)}- {s.title} (p.{s.page_start + 1})")
        return "\n".join(lines)


class Chunk(BaseModel):
    """A retrievable unit. `kind` decides how it is displayed, boosted and cited."""

    chunk_id: str
    kind: str  # text | abstract | figure | table | equation | reference
    text: str
    page: int  # 0-based
    section: str = ""
    section_kind: str = "other"
    label: str | None = None  # "Figure 2", "Table 1", ...
    bbox: BBox | None = None
    refs: list[str] = Field(default_factory=list)  # labels mentioned in this chunk
    ord: int = 0

    def embedding_text(self, title: str = "") -> str:
        """Text sent to the embedder: a contextual header improves retrieval of short chunks."""
        head = " | ".join(x for x in (title[:120], self.section, self.label or "") if x)
        return f"{head}\n{self.text}" if head else self.text


_LABEL_KINDS = {"fig": "Figure", "figure": "Figure", "table": "Table", "eq": "Equation",
                "equation": "Equation"}
MENTION_RE = re.compile(
    r"\b(?P<kind>Fig(?:ure)?s?|Tables?|Eq(?:uation)?s?)\.?\s*\(?(?P<num>S?\d{1,3})(?P<sub>[A-Za-z])?\b",
    re.I,
)


def canonical_label(kind: str, number: str) -> str:
    key = kind.lower().rstrip("s").rstrip(".")
    return f"{_LABEL_KINDS.get(key, kind.title())} {number.upper()}"


def find_mentions(text: str) -> list[str]:
    """Canonical labels ('Figure 3', 'Table 2') mentioned in `text`, in order, de-duplicated."""
    seen: list[str] = []
    for m in MENTION_RE.finditer(text):
        label = canonical_label(m.group("kind"), m.group("num"))
        if label not in seen:
            seen.append(label)
    return seen
