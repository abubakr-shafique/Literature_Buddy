"""Low-level layout analysis on top of PyMuPDF: blocks, reading order, headers/footers."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

Rect4 = tuple[float, float, float, float]
_WS = re.compile(r"[ \t\u00a0\u2009\u200a\u202f]+")


@dataclass
class Block:
    page: int
    bbox: Rect4
    text: str
    size: float  # dominant font size
    bold: bool  # majority of characters are bold
    first_bold: bool
    first_span: str
    n_lines: int
    role: str = "body"  # body | heading | caption | furniture | figure_text | table_text | equation
    extra: dict = field(default_factory=dict)

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines, repairing end-of-line hyphenation ('repre-\\nsentation')."""
    out = ""
    for raw in lines:
        t = _WS.sub(" ", raw).strip()
        if not t:
            continue
        if not out:
            out = t
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t
        else:
            out += " " + t
    return out


def extract_blocks(page: pymupdf.Page, pno: int, textpage: pymupdf.TextPage | None = None) -> list[Block]:
    """Text blocks with font statistics. Rotated (e.g. arXiv margin stamp) lines are skipped."""
    data = page.get_text("dict", flags=pymupdf.TEXT_MEDIABOX_CLIP, textpage=textpage)
    blocks: list[Block] = []
    for b in data.get("blocks", []):
        if b.get("type") != 0:
            continue
        lines: list[str] = []
        sizes: Counter[float] = Counter()
        bold_chars = total_chars = 0
        first_bold = False
        first_span = ""
        for ln in b.get("lines", []):
            dx, dy = ln.get("dir", (1.0, 0.0))
            if abs(dy) > 0.5 or dx < 0:
                continue
            parts = []
            for sp in ln.get("spans", []):
                txt = sp.get("text", "")
                if not txt.strip():
                    parts.append(txt)
                    continue
                n = len(txt.strip())
                is_bold = bool(sp.get("flags", 0) & 16) or "bold" in sp.get("font", "").lower()
                sizes[round(sp.get("size", 0) * 2) / 2] += n
                total_chars += n
                bold_chars += n if is_bold else 0
                if not first_span:
                    first_bold, first_span = is_bold, txt.strip()
                parts.append(txt)
            lines.append("".join(parts))
        text = join_lines(lines)
        if not text or total_chars == 0:
            continue
        blocks.append(
            Block(
                page=pno,
                bbox=tuple(b["bbox"]),
                text=text,
                size=sizes.most_common(1)[0][0],
                bold=bold_chars / total_chars > 0.6,
                first_bold=first_bold,
                first_span=first_span,
                n_lines=len(lines),
            )
        )
    return blocks


def body_font_size(pages: list[list[Block]]) -> float:
    counts: Counter[float] = Counter()
    for blocks in pages:
        for b in blocks:
            counts[b.size] += len(b.text)
    return counts.most_common(1)[0][0] if counts else 10.0


def mark_furniture(pages: list[list[Block]], page_heights: list[float]) -> None:
    """Flag running headers/footers and bare page numbers (role='furniture')."""
    n = len(pages)
    seen: Counter[str] = Counter()
    keys: dict[int, str] = {}

    def in_margin(b: Block, h: float) -> bool:
        return b.y1 < h * 0.08 or b.y0 > h * 0.92

    for pno, blocks in enumerate(pages):
        h = page_heights[pno]
        for b in blocks:
            if in_margin(b, h):
                key = re.sub(r"\d+", "#", b.text.lower())[:80]
                keys[id(b)] = key
                seen[key] += 1
    for pno, blocks in enumerate(pages):
        h = page_heights[pno]
        for b in blocks:
            if not in_margin(b, h):
                continue
            if re.fullmatch(r"\W*\d{1,4}\W*", b.text):
                b.role = "furniture"
            elif n >= 3 and seen[keys[id(b)]] >= max(2, 0.4 * n):
                b.role = "furniture"


def _column_of(b: Block, mid: float, pw: float) -> str:
    if b.width > 0.58 * pw or (b.x0 < mid - 0.06 * pw and b.x1 > mid + 0.06 * pw):
        return "full"
    return "left" if (b.x0 + b.x1) / 2 < mid else "right"


def reading_order(blocks: list[Block], page_width: float) -> list[Block]:
    """Column-aware ordering: full-width blocks split the page into bands; within a band the
    left column is read before the right one. Single-column pages fall back to top-to-bottom."""
    if not blocks:
        return []
    mid = page_width / 2
    cols = {id(b): _column_of(b, mid, page_width) for b in blocks}
    left = sum(len(b.text) for b in blocks if cols[id(b)] == "left")
    right = sum(len(b.text) for b in blocks if cols[id(b)] == "right")
    total = sum(len(b.text) for b in blocks) or 1
    ordered = sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))
    if not (left / total > 0.2 and right / total > 0.2):
        return ordered
    out: list[Block] = []
    pend_l: list[Block] = []
    pend_r: list[Block] = []
    for b in ordered:
        c = cols[id(b)]
        if c == "full":
            out += pend_l + pend_r
            pend_l, pend_r = [], []
            out.append(b)
        elif c == "left":
            pend_l.append(b)
        else:
            pend_r.append(b)
    return out + pend_l + pend_r


def union(rects: list[Rect4]) -> Rect4:
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def overlap_area(a: Rect4, b: Rect4) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def area(a: Rect4) -> float:
    return max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)


def render_region(
    page: pymupdf.Page, bbox: Rect4, out_path: str, dpi: int = 170, pad: float = 3.0,
    max_side: int = 2000,
) -> None:
    """Render a page region to PNG (scaled down if larger than `max_side` px)."""
    r = pymupdf.Rect(bbox) + (-pad, -pad, pad, pad)
    r = r & page.rect
    if r.is_empty:
        r = page.rect
    zoom = dpi / 72
    zoom = min(zoom, max_side / max(r.width, r.height, 1))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=r, alpha=False)
    pix.save(out_path)
