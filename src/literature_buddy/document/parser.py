"""PDF -> structured `Document` (sections, paragraphs, figures, tables, equations, references)."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

import pymupdf

from ..config.settings import DocumentConfig
from ..errors import Cancelled, ParseError
from .figure_extractor import Caption, find_captions, graphic_clusters, locate_figure
from .layout import (
    Block,
    body_font_size,
    extract_blocks,
    mark_furniture,
    reading_order,
    render_region,
)
from .metadata import (
    abstract_runin,
    guess_authors,
    guess_title,
    heading_info,
    parse_references,
)
from .schema import Document, Equation, Figure, Paragraph, Section, Table
from .table_extractor import detect_tables, match_caption, rows_to_markdown, text_strategy_fallback

log = logging.getLogger(__name__)
PARSER_VERSION = 1
ProgressFn = Callable[[float, str], None]

_MATH_CHARS = set("=<>\u2264\u2265\u2248\u223c~\u00b1\u00d7\u2211\u220f\u222b\u221a\u2202\u2207\u2208\u2200\u2203\u03bb\u03bc\u03c3\u03b1\u03b2\u03b3\u03b4\u03b8\u03c0\u03c6\u03c8\u03c9\u0394\u03a3\u03a9")
_EQ_RE = re.compile(r"^(?P<eq>.{3,}?)\s+\((?P<num>\d{1,3})\)\s*$")
_SKIP_ROLES = {"furniture", "caption", "figure_text", "table_text", "equation", "title", "front"}


def build_key(cfg: DocumentConfig) -> str:
    return (
        f"p{PARSER_VERSION}|ocr={cfg.ocr}|fig={cfg.extract_figures}|tab={cfg.extract_tables}"
        f"|eq={cfg.extract_equations}|dpi={cfg.figure_dpi}"
    )


def _center_inside(b: Block, box: tuple[float, float, float, float]) -> bool:
    cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def parse_pdf(
    pdf_path: Path,
    doc_id: str,
    source: str,
    out_dir: Path,
    cfg: DocumentConfig,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> Document:
    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    def check() -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled("Cancelled by user")

    try:
        pdf = pymupdf.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Could not open PDF: {exc}") from exc
    if pdf.needs_pass:
        raise ParseError("This PDF is password-protected.")
    if pdf.page_count == 0:
        raise ParseError("This PDF has no pages.")

    n = pdf.page_count
    warnings: list[str] = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: text blocks (+ OCR for pages without a text layer) ---------------------------
    pages: list[list[Block]] = []
    heights: list[float] = []
    widths: list[float] = []
    ocr_missing = 0
    for pno in range(n):
        check()
        page = pdf[pno]
        heights.append(page.rect.height)
        widths.append(page.rect.width)
        blocks = extract_blocks(page, pno)
        if not blocks and cfg.ocr and page.get_images():
            try:
                tp = page.get_textpage_ocr(dpi=200, full=True)
                blocks = extract_blocks(page, pno, textpage=tp)
            except Exception as exc:  # noqa: BLE001 - Tesseract not installed / no language data
                ocr_missing += 1
                log.debug("OCR unavailable on page %d: %s", pno, exc)
        pages.append(blocks)
        report(0.25 * (pno + 1) / n, f"Reading page {pno + 1}/{n}")
    if ocr_missing:
        warnings.append(f"{ocr_missing} page(s) have no text layer and OCR (Tesseract) is unavailable.")
    if not any(pages):
        raise ParseError("No extractable text found (scanned PDF without OCR?).")

    body = body_font_size(pages)
    mark_furniture(pages, heights)

    # ---- pass 2: per-page structure ------------------------------------------------------------
    figures: dict[str, Figure] = {}
    tables: dict[str, Table] = {}
    equations: list[Equation] = []
    ordered_pages: list[list[Block]] = []
    for pno in range(n):
        check()
        page = pdf[pno]
        pw = widths[pno]
        live = [b for b in pages[pno] if b.role != "furniture"]
        order = reading_order(live, pw)
        caps = find_captions(order, body)
        ordered_pages.append(order)

        table_boxes: list[tuple[float, float, float, float]] = []
        used_caps: set[int] = set()
        if cfg.extract_tables:
            for bbox, rows in detect_tables(page):
                cap = match_caption(bbox, caps, used_caps)
                if cap is None:
                    continue
                used_caps.add(id(cap.block))
                _add_table(tables, page, pno, cap, bbox, rows, fig_dir, cfg, approximate=False)
                table_boxes.append(bbox)
            for cap in caps:
                if cap.kind == "Table" and id(cap.block) not in used_caps:
                    found = text_strategy_fallback(page, cap)
                    used_caps.add(id(cap.block))
                    if found:
                        _add_table(tables, page, pno, cap, found[0], found[1], fig_dir, cfg, approximate=True)
                        table_boxes.append(found[0])
                    elif cap.label not in tables:
                        tables[cap.label] = Table(
                            label=cap.label, number=cap.number, caption=cap.text, page=pno,
                            caption_bbox=cap.bbox,
                        )
            for b in order:
                if b.role == "body" and any(_center_inside(b, tb) for tb in table_boxes):
                    b.role = "table_text"

        fig_caps = sorted((c for c in caps if c.kind == "Figure"), key=lambda c: c.bbox[1])
        if fig_caps:
            clusters = graphic_clusters(page, exclude=table_boxes) if cfg.extract_figures else []
            used: set[int] = set()
            for cap in fig_caps:
                box, absorbed = locate_figure(cap, clusters, order, used, page.rect) if clusters else (None, [])
                for tb in absorbed:
                    tb.role = "figure_text"
                if box:
                    for b in order:
                        if b.role == "body" and _center_inside(b, box) and len(b.text.split()) < 40:
                            b.role = "figure_text"
                fig = Figure(label=cap.label, number=cap.number, caption=cap.text, page=pno,
                             bbox=box, caption_bbox=cap.bbox)
                if box:
                    rel = f"figures/{_slug(cap.label)}_p{pno + 1}.png"
                    render_region(page, box, str(out_dir / rel), cfg.figure_dpi)
                    fig.image_path = rel
                prev = figures.get(cap.label)
                if prev is None or (prev.bbox is None and fig.bbox is not None):
                    figures[cap.label] = fig

        if cfg.extract_equations:
            for b in order:
                if b.role != "body":
                    continue
                m = _EQ_RE.match(b.text)
                if m and len(b.text.split()) <= 40 and any(ch in _MATH_CHARS for ch in m.group("eq")):
                    b.role = "equation"
                    label = f"Equation {m.group('num')}"
                    rel = f"figures/{_slug(label)}_p{pno + 1}.png"
                    render_region(page, b.bbox, str(out_dir / rel), cfg.figure_dpi, pad=4)
                    equations.append(Equation(label=label, number=m.group("num"), text=m.group("eq"),
                                              page=pno, bbox=b.bbox, image_path=rel))
        report(0.25 + 0.45 * (pno + 1) / n, f"Analysing page {pno + 1}/{n}")

    # ---- title / authors / sections -----------------------------------------------------------
    meta_title = (pdf.metadata or {}).get("title", "") or ""
    first = ordered_pages[0]
    title, title_blocks = guess_title(first, body, heights[0], meta_title.strip() or Path(source).stem)
    for tb in title_blocks:
        tb.role = "title"
    authors = guess_authors(first, title_blocks, max((b.size for b in title_blocks), default=0.0))
    for b in first:
        if b.role == "body" and authors and any(a in b.text for a in authors[:2]):
            b.role = "front"

    sections: list[Section] = [Section(idx=0, title="Front matter", kind="front", level=0)]
    paragraphs: list[Paragraph] = []
    ref_blocks: list[Block] = []
    cur, parent_kind = 0, "other"
    for pno, order in enumerate(ordered_pages):
        for b in order:
            if b.role in _SKIP_ROLES:
                continue
            runin = abstract_runin(b)
            if runin is not None and sections[cur].kind in ("front",):
                sections.append(Section(idx=len(sections), title="Abstract", kind="abstract", level=1,
                                        page_start=pno, page_end=pno))
                cur = len(sections) - 1
                paragraphs.append(Paragraph(page=pno, bbox=b.bbox, text=runin, section_idx=cur))
                continue
            info = heading_info(b, body)
            if info is not None:
                level, htitle, kind = info
                b.role = "heading"
                if kind == "other" and level > 1:
                    kind = parent_kind
                if level == 1:
                    parent_kind = kind
                sections.append(Section(idx=len(sections), title=htitle, kind=kind, level=level,
                                        page_start=pno, page_end=pno))
                cur = len(sections) - 1
                continue
            sec = sections[cur]
            if sec.kind == "references":
                ref_blocks.append(b)
                continue
            if len(b.text) < 25 and len(b.text.split()) < 4:
                continue
            paragraphs.append(Paragraph(page=pno, bbox=b.bbox, text=b.text, section_idx=cur))
            sec.page_end = max(sec.page_end, pno)

    paragraphs = _merge_continuations(paragraphs)
    abstract = " ".join(p.text for p in paragraphs if sections[p.section_idx].kind == "abstract")
    if not abstract:  # no heading: first long paragraph in the front matter
        for p in paragraphs:
            if sections[p.section_idx].kind == "front" and len(p.text.split()) >= 60:
                abstract = p.text
                break
    references = parse_references(ref_blocks)
    if not references:
        warnings.append("No reference list detected.")
    if cfg.extract_figures and not figures:
        warnings.append("No figures with captions were detected.")
    report(0.72, "Structure extracted")

    return Document(
        doc_id=doc_id, source=source, build_key=build_key(cfg), title=title, authors=authors,
        abstract=abstract, page_count=n, sections=sections, paragraphs=paragraphs,
        figures=list(figures.values()), tables=list(tables.values()), equations=equations,
        references=references, warnings=warnings,
    )


def _add_table(
    tables: dict[str, Table], page: pymupdf.Page, pno: int, cap: Caption,
    bbox: tuple[float, float, float, float], rows: list[list[str | None]], fig_dir: Path,
    cfg: DocumentConfig, approximate: bool,
) -> None:
    rel = f"figures/{_slug(cap.label)}_p{pno + 1}.png"
    render_region(page, bbox, str(fig_dir.parent / rel), cfg.figure_dpi)
    tables[cap.label] = Table(
        label=cap.label, number=cap.number, caption=cap.text, page=pno, bbox=bbox,
        caption_bbox=cap.bbox, markdown=rows_to_markdown(rows), n_rows=len(rows),
        n_cols=max(len(r) for r in rows), image_path=rel, approximate=approximate,
    )


def _merge_continuations(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """Join paragraphs split by a column or page break (previous text unfinished, next lower-case)."""
    out: list[Paragraph] = []
    for p in paragraphs:
        if (
            out
            and out[-1].section_idx == p.section_idx
            and not out[-1].text.rstrip().endswith((".", "?", "!", ":", ";", ")", '"', "\u201d"))
            and p.text[:1].islower()
        ):
            out[-1] = out[-1].model_copy(update={"text": out[-1].text + " " + p.text})
        else:
            out.append(p)
    return out
