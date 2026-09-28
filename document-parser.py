# src/literature_buddy/document/parser.py
"""Scientific PDF parser built on PyMuPDF.

Why PyMuPDF as the core: it is fast, dependency-light, gives per-block geometry
(bboxes, font sizes) which we use for heading detection and caption association,
and extracts embedded raster images with xref handles. GROBID (TEI header/refs)
and Docling (table structure) are documented upgrade paths behind the same
ParsedDocument data model; this module keeps Phase 1-3 fully self-contained.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from literature_buddy.config.settings import AppSettings
from literature_buddy.document.models import (
    BoundingBox,
    Figure,
    ParsedDocument,
    Section,
    SectionKind,
    Table,
)

FIGURE_RE = re.compile(r"\b(Fig(?:ure)?\.?\s*\d+[A-Za-z]?)", re.IGNORECASE)
TABLE_RE = re.compile(r"\b(Table\s*\d+[A-Za-z]?)", re.IGNORECASE)
REF_ITEM_RE = re.compile(r"^\s*\[\d+\]|^\s*\d+\.\s+\S")

_SECTION_KEYWORDS: dict[SectionKind, tuple[str, ...]] = {
    SectionKind.ABSTRACT: ("abstract",),
    SectionKind.INTRODUCTION: ("introduction",),
    SectionKind.BACKGROUND: ("background", "related work", "preliminaries"),
    SectionKind.METHODS: ("methods", "materials and methods", "methodology",
                          "experimental setup", "experiments"),
    SectionKind.RESULTS: ("results", "findings", "evaluation"),
    SectionKind.DISCUSSION: ("discussion",),
    SectionKind.CONCLUSION: ("conclusion", "conclusions", "concluding remarks"),
    SectionKind.REFERENCES: ("references", "bibliography"),
    SectionKind.SUPPLEMENTARY: ("supplementary", "appendix", "supplement"),
}


def _classify_heading(text: str) -> SectionKind | None:
    norm = re.sub(r"^[0-9A-Z.\-ivx]+\s*[.)]?\s*", "", text.strip().lower())
    for kind, keys in _SECTION_KEYWORDS.items():
        if any(norm == k or norm.startswith(k + " ") or norm.startswith(k + ":") for k in keys):
            return kind
    return None


def _looks_like_heading(span: dict, body_median: float) -> bool:
    text = span.get("text", "").strip()
    if not text or len(text) > 120 or text.endswith((".", ";")):
        return False
    if not _classify_heading(text):
        return False
    size = span.get("size", body_median)
    is_bold = "Bold" in span.get("font", "") or (span.get("flags", 0) & 16)
    return size >= body_median or is_bold


def parse_pdf(pdf_path: str | Path, settings: AppSettings) -> ParsedDocument:
    doc = fitz.open(str(pdf_path))
    out = ParsedDocument(source_path=str(pdf_path), num_pages=len(doc))

    # --- First pass: collect body font size median + all blocks -------------
    sizes: list[float] = []
    pages_blocks: list[list[dict]] = []
    for page in doc:
        d = page.get_text("dict")
        blocks: list[dict] = []
        for b in d.get("blocks", []):
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text.strip():
                        sizes.append(span.get("size", 10))
                        blocks.append({
                            "text": text,
                            "size": span["size"],
                            "font": span.get("font", ""),
                            "flags": span.get("flags", 0),
                            "bbox": b["bbox"],
                        })
        pages_blocks.append(blocks)
    body_median = sorted(sizes)[len(sizes) // 2] if sizes else 10.0

    # --- Sections --------------------------------------------------------------
    headings: list[tuple[str, int, SectionKind]] = []  # (title, page, kind)
    for pno, blocks in enumerate(pages_blocks, start=1):
        out.page_texts.append("\n".join(b["text"] for b in blocks))
        for b in blocks:
            if _looks_like_heading(b, body_median):
                kind = _classify_heading(b["text"])
                if kind:
                    headings.append((b["text"].strip(), pno, kind))

    # Drop single-word artifacts: require plausible heading text.
    headings = [h for h in headings if len(h[0]) >= 5 or h[2] is SectionKind.ABSTRACT]
    for i, (title, page, kind) in enumerate(headings):
        end = headings[i + 1][1] if i + 1 < len(headings) else out.num_pages
        out.sections.append(Section(kind=kind, title=title, page_start=page, page_end=end))

    # --- Title / abstract -------------------------------------------------------
    meta = doc.metadata
    out.title = (meta or {}).get("title", "").strip()
    if not out.title and out.page_texts:
        first_nonempty = next((l for l in out.page_texts[0].splitlines()
                               if len(l.strip()) > 20), "")
        out.title = first_nonempty.strip()
    authors_raw = (meta or {}).get("author", "")
    out.authors = [a.strip() for a in re.split(r"[;,]", authors_raw) if a.strip()]

    abstract_sec = next((s for s in out.sections if s.kind is SectionKind.ABSTRACT), None)
    if abstract_sec:
        page_idx = abstract_sec.page_start - 1
        after = out.page_texts[page_idx].split(abstract_sec.title, 1)
        if len(after) > 1:
            nxt = next((s.title for s in out.sections
                        if s.page_start >= abstract_sec.page_start
                        and s is not abstract_sec), None)
            body = after[1].split(nxt)[0] if nxt and nxt in after[1] else after[1]
            out.abstract = " ".join(body.split())[:2000]

    # --- Figures and captions -----------------------------------------------------
    if settings.document.extract_figures:
        out.figures = _extract_figures_with_captions(doc, pdf_path, settings)

    # --- Tables ---------------------------------------------------------------------
    if settings.document.extract_tables:
        out.tables = _extract_tables(out)

    # --- References -------------------------------------------------------------------
    refs_sec = next((s for s in out.sections if s.kind is SectionKind.REFERENCES), None)
    if refs_sec:
        joined = "\n".join(out.page_texts[refs_sec.page_start - 1:refs_sec.page_end])
        after = joined.split(refs_sec.title, 1)
        if len(after) > 1:
            cands = [ln.strip() for ln in after[1].splitlines() if ln.strip()]
            items, buf = [], ""
            for ln in cands:
                if REF_ITEM_RE.match(ln):
                    if buf:
                        items.append(buf.strip())
                    buf = ln
                else:
                    buf += " " + ln
            if buf:
                items.append(buf.strip())
            out.references = items[:500]

    doc.close()
    return out


def _extract_figures_with_captions(
    doc: fitz.Document, pdf_path: Path | str, settings: AppSettings
) -> list[Figure]:
    figures: list[Figure] = []
    fig_dir = settings.storage.resolved("figures_dir")
    slug_fig_dir = fig_dir / Path(str(pdf_path)).stem[:40]
    slug_fig_dir.mkdir(parents=True, exist_ok=True)

    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text("text")
        cap_matches = list(FIGURE_RE.finditer(text))
        images = page.get_images(full=True)
        if not cap_matches or not images:
            continue

        seen_labels: set[str] = set()
        raster_imgs: list[tuple[int, int, int]] = []  # (xref, w, h)
        for img in images:
            xref, w, h = img[0], img[2], img[3]
            if w * h >= settings.document.min_figure_area_px:
                raster_imgs.append((xref, w, h))

        for m in cap_matches:
            raw = m.group(1)
            label = raw.replace("Fig.", "Figure").replace("Fig ", "Figure ")
            if not label.lower().startswith("figure"):
                label = "Figure " + label
            label = label.split(".")[0].strip()
            if label in seen_labels:
                continue
            seen_labels.add(label)

            caption = _extract_caption(text, m.start())

            image_path = ""
            if raster_imgs:
                xref, _, _ = raster_imgs.pop(0)
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:  # CMSKY etc.
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    out_path = slug_fig_dir / f"p{pno + 1}_{label.replace(' ', '_')}.png"
                    pix.save(str(out_path))
                    image_path = str(out_path)
                except Exception:
                    image_path = ""

            bbox = None
            rects = page.search_for(label)
            if rects:
                r = rects[0]
                bbox = BoundingBox(r.x0, r.y0, r.x1, r.y1)
            figures.append(Figure(label=label, caption=caption, page=pno + 1,
                                  image_path=image_path, bbox=bbox))
    return figures


def _extract_caption(page_text: str, label_start: int) -> str:
    """Caption text runs from the figure label until the next sentence-ending
    boundary that is followed by a capital + ~20 chars (heuristic), capped at 600 chars."""
    snippet = page_text[label_start:label_start + 900].replace("\n", " ")
    snippet = " ".join(snippet.split())
    for end_re in (r"\.\s+(?=[A-Z][A-Za-z]{2})", r"\.\s*$"):
        m = re.search(end_re, snippet)
        if m and m.end() > 30:
            return snippet[: m.end()].strip()
    return snippet[:600].strip()


def _extract_tables(doc: ParsedDocument) -> list[Table]:
    tables: list[Table] = []
    for pno, text in enumerate(doc.page_texts, start=1):
        for m in TABLE_RE.finditer(text):
            label = m.group(1).strip()
            if any(t.label == label for t in tables):
                continue
            start = m.start()
            window = text[start:start + 1200].replace("\n", " | ")
            window = " ".join(window.split())
            caption_end = min(len(window), 300)
            tables.append(Table(label=label,
                                caption=window[:caption_end],
                                text=window[:1200],
                                page=pno))
    return tables
