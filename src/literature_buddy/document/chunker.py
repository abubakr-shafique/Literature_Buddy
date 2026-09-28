# src/literature_buddy/document/chunker.py
"""Scientific-document-aware chunking.

Strategy (not naive fixed-size splitting):
1. Iterate pages, tracking which parsed Section each line belongs to via
   (page range -> section) mapping. References section text is excluded from
   retrieval chunks (low signal, huge token cost); references are still stored
   on the ParsedDocument.
2. Within a section, accumulate paragraph-ish lines until ~max_chunk_tokens,
   then flush. A 15% overlap paragraph is carried to the next chunk so
   sentence-level evidence is never cut mid-idea.
3. Figures and tables become their own dedicated chunks (caption text +
   image path metadata) so "What does Figure 2 show?" retrieval hits a
   visual chunk, not prose.
4. Every chunk carries citation metadata: page, section kind + title,
   chunk type, and figure/table labels.
"""

from __future__ import annotations

import re

from literature_buddy.document.models import Chunk, ParsedDocument, Section, SectionKind

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _section_for_page(sections: list[Section], page: int) -> Section | None:
    for s in sections:
        if s.page_start <= page <= s.page_end:
            return s
    return None


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_document(
    doc: ParsedDocument,
    max_chunk_tokens: int = 350,
    overlap_ratio: float = 0.15,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    skip_kinds = {SectionKind.REFERENCES}

    current_section = SectionKind.OTHER
    current_title = ""
    buffer: list[str] = []
    buffer_page = 1

    def flush() -> None:
        nonlocal buffer, buffer_page
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if len(text) < 40:
            buffer = []
            return
        overlap: list[str] = []
        if chunks and overlap_ratio > 0:
            sents = SENT_SPLIT_RE.split(chunks[-1].text)
            keep = max(1, int(len(sents) * overlap_ratio))
            overlap = sents[-keep:] if keep else []
        new = Chunk(text=(" ".join(overlap) + "\n" + text).strip() if overlap else text,
                    page=buffer_page,
                    section=current_section.value,
                    section_title=current_title)
        chunks.append(new)
        buffer = []

    for pno, page_text in enumerate(doc.page_texts, start=1):
        sec = _section_for_page(doc.sections, pno)
        if sec is not None:
            if sec.kind in skip_kinds:
                flush()
                continue
            sec_kind, sec_title = sec.kind, sec.title
        else:
            sec_kind, sec_title = SectionKind.OTHER, ""

        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            buffer.append(stripped)
            if not buffer_page and buffer:
                buffer_page = pno
            if _approx_tokens("\n".join(buffer)) >= max_chunk_tokens:
                flush()
                current_section, current_title = sec_kind, sec_title
        if current_section != sec_kind:
            current_section, current_title = sec_kind, sec_title
    flush()

    # Dedicated figure and table chunks.
    for fig in doc.figures:
        chunks.append(Chunk(
            text=f"{fig.label}: {fig.caption}",
            page=fig.page,
            section=current_section.value,
            section_title="",
            chunk_type="figure_caption",
            figure_label=fig.label,
            figure_image_path=fig.image_path,
        ))
    for tbl in doc.tables:
        chunks.append(Chunk(
            text=f"{tbl.label}: {tbl.caption}\n{tbl.text}",
            page=tbl.page,
            chunk_type="table",
            table_label=tbl.label,
        ))
    return chunks
