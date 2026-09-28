# src/literature_buddy/document/models.py
"""Core document data model preserving scientific structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SectionKind(str, Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    SUPPLEMENTARY = "supplementary"
    OTHER = "other"


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class Figure:
    label: str            # e.g. "Figure 2"
    caption: str
    page: int             # 1-based
    image_path: str       # extracted image file, empty if unavailable
    bbox: Optional[BoundingBox] = None

    @property
    def key(self) -> str:
        return f"fig::{self.page}::{self.label}"


@dataclass
class Table:
    label: str            # e.g. "Table 2"
    caption: str
    text: str             # pipe-delimited render of the table content
    page: int


@dataclass
class Section:
    kind: SectionKind
    title: str            # as printed in the paper, e.g. "3. Methods"
    page_start: int
    page_end: int


@dataclass
class Chunk:
    """A retrieval unit with full citation metadata."""

    text: str
    page: int
    section: str = SectionKind.OTHER.value
    section_title: str = ""
    chunk_type: str = "text"   # text | figure_caption | table
    figure_label: str = ""
    table_label: str = ""
    figure_image_path: str = ""


@dataclass
class ParsedDocument:
    source_path: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    page_texts: list[str] = field(default_factory=list)
    num_pages: int = 0

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier for caches and indexes."""
        from hashlib import sha1

        return sha1(self.source_path.encode()).hexdigest()[:12]
