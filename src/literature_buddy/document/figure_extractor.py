"""Figure detection: caption discovery + graphic-region clustering + caption association.

Strategy (works for raster *and* vector figures):
  1. find caption blocks ("Figure 3.", "Fig. 3:", "Fig. 3 |")
  2. collect graphic primitives on the page (images + vector drawings)
  3. merge primitives into connected clusters using an occupancy grid (flood fill)
  4. pick the cluster(s) directly above each caption (or below, for caption-above layouts)
  5. pull in nearby short text blocks (axis labels, panel letters), render the union to PNG
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pymupdf

from .layout import Block, Rect4, area, overlap_area, union
from .schema import canonical_label

CAPTION_RE = re.compile(
    r"^\s*(?P<supp>Supplementary\s+|Extended\s+Data\s+)?(?P<kind>Fig(?:ure)?|Table)s?\.?\s*"
    r"(?P<num>S?\d{1,3})(?![A-Za-z0-9])\s*(?P<sep>[.:|\u2013\u2014\-])?",
    re.I,
)
CELL = 3.0  # grid resolution in PDF points


@dataclass
class Caption:
    block: Block
    kind: str  # "Figure" | "Table"
    number: str
    text: str
    bbox: Rect4

    @property
    def label(self) -> str:
        return canonical_label(self.kind, self.number)


def find_captions(blocks: list[Block], body_size: float) -> list[Caption]:
    """Identify caption blocks (and merge label-only / continuation blocks). Marks role='caption'."""
    out: list[Caption] = []
    consumed: set[int] = set()
    ordered = sorted(blocks, key=lambda b: (round(b.y0), b.x0))
    for i, b in enumerate(ordered):
        if id(b) in consumed or b.role not in ("body", "heading"):
            continue
        m = CAPTION_RE.match(b.text)
        if not m:
            continue
        small = b.size < body_size - 0.3
        if not (m.group("sep") or b.first_bold or small):
            continue  # "Figure 3 shows ..." in running text is a mention, not a caption
        kind = "Figure" if m.group("kind").lower().startswith("fig") else "Table"
        num = m.group("num").upper()
        if m.group("supp") and not num.startswith("S"):
            num = "S" + num
        text, bbox = b.text, b.bbox
        b.role = "caption"
        # merge continuation blocks directly below (same size/left edge) while caption is unfinished
        for nb in ordered[i + 1: i + 4]:
            if id(nb) in consumed or nb.role != "body":
                continue
            gap = nb.y0 - bbox[3]
            same_col = abs(nb.x0 - b.x0) < 8 and 0 <= gap < 5
            label_only = len(b.text.split()) <= 3 and 0 <= gap < 12
            unfinished = not text.rstrip().endswith((".", "!", "?"))
            if same_col and nb.size <= b.size + 0.3 and (unfinished or label_only):
                text += " " + nb.text
                bbox = union([bbox, nb.bbox])
                nb.role = "caption"
                consumed.add(id(nb))
            else:
                break
        out.append(Caption(b, kind, num, " ".join(text.split()), bbox))
    return out


def _grid_clusters(
    rects: list[Rect4], page_rect: pymupdf.Rect, gap: float = 6.0
) -> list[Rect4]:
    """Connected components of dilated rectangles -> union boxes of the original rectangles."""
    if not rects:
        return []
    w = int(math.ceil(page_rect.width / CELL)) + 3
    h = int(math.ceil(page_rect.height / CELL)) + 3
    grid = np.zeros((h, w), dtype=bool)
    d = int(math.ceil(gap / CELL))

    def cells(r: Rect4) -> tuple[int, int, int, int]:
        x0 = max(int((r[0] - page_rect.x0) / CELL) - d, 0)
        y0 = max(int((r[1] - page_rect.y0) / CELL) - d, 0)
        x1 = min(int((r[2] - page_rect.x0) / CELL) + d + 1, w)
        y1 = min(int((r[3] - page_rect.y0) / CELL) + d + 1, h)
        return x0, y0, x1, y1

    for r in rects:
        x0, y0, x1, y1 = cells(r)
        grid[y0:y1, x0:x1] = True

    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    ys, xs = np.nonzero(grid)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        if labels[y, x]:
            continue
        n += 1
        labels[y, x] = n
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = n
                    stack.append((ny, nx))
    groups: dict[int, list[Rect4]] = {}
    for r in rects:
        cx = min(max(int(((r[0] + r[2]) / 2 - page_rect.x0) / CELL), 0), w - 1)
        cy = min(max(int(((r[1] + r[3]) / 2 - page_rect.y0) / CELL), 0), h - 1)
        lab = int(labels[cy, cx])
        if lab:
            groups.setdefault(lab, []).append(r)
    return [union(g) for g in groups.values()]


def graphic_clusters(page: pymupdf.Page, exclude: list[Rect4] | None = None) -> list[Rect4]:
    """Bounding boxes of contiguous graphics (images + vector art) on the page."""
    pr = page.rect
    prim: list[Rect4] = []
    for info in page.get_image_info():
        r = tuple(info["bbox"])
        if area(r) > 0.85 * pr.width * pr.height:
            continue  # full-page scan / background
        if area(r) > 25:
            prim.append(r)  # type: ignore[arg-type]
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 - malformed content streams
        drawings = []
    for d in drawings:
        r = d["rect"]
        rw, rh = r.width, r.height
        if rw > 0.9 * pr.width and rh < 4:
            continue  # page-wide rules
        if rw * rh > 0.7 * pr.width * pr.height:
            continue  # page background
        prim.append((r.x0, r.y0, r.x1 + 0.5, r.y1 + 0.5))
    if exclude:
        prim = [p for p in prim if not any(overlap_area(p, e) > 0.5 * max(area(p), 1) for e in exclude)]
    clusters = _grid_clusters(prim, pr)
    # drop tiny clusters (bullets, underlines, logos)
    return [c for c in clusters if (c[2] - c[0]) >= 40 and (c[3] - c[1]) >= 30]


def locate_figure(
    cap: Caption, clusters: list[Rect4], text_blocks: list[Block], used: set[int], page_rect: pymupdf.Rect
) -> tuple[Rect4 | None, list[Block]]:
    """Choose the graphic region belonging to `cap`. Returns (bbox, absorbed short text blocks)."""
    cb = cap.bbox
    lo, hi = cb[0] - 30, cb[2] + 30

    def h_ok(c: Rect4) -> bool:
        ov = min(c[2], hi) - max(c[0], lo)
        return ov > 0.4 * min(c[2] - c[0], hi - lo)

    above = [(i, c) for i, c in enumerate(clusters) if i not in used and c[3] <= cb[1] + 6 and h_ok(c)]
    below = [(i, c) for i, c in enumerate(clusters) if i not in used and c[1] >= cb[3] - 6 and h_ok(c)]
    pool, direction = (above, "up") if above else (below, "down")
    if not pool:
        return None, []
    if direction == "up":
        idx, first = min(pool, key=lambda ic: cb[1] - ic[1][3])
        if cb[1] - first[3] > 140:
            return None, []
    else:
        idx, first = min(pool, key=lambda ic: ic[1][1] - cb[3])
        if first[1] - cb[3] > 60:
            return None, []
    members = [idx]
    box = first
    changed = True
    while changed:  # absorb neighbouring panels (multi-panel figures)
        changed = False
        for i, c in pool:
            if i in members:
                continue
            near_v = (c[1] - box[3] < 30 and c[3] > box[1] - 30)
            near_h = (c[0] - box[2] < 30 and c[2] > box[0] - 30)
            if near_v and near_h:
                members.append(i)
                box = union([box, c])
                changed = True
    used.update(members)
    absorbed: list[Block] = []
    for _ in range(2):  # short text near the graphic: axis labels, panel letters, legends
        for tb in text_blocks:
            if tb in absorbed or tb.role not in ("body",):
                continue
            grown = (box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4)
            if overlap_area(tb.bbox, grown) > 0 and len(tb.text.split()) < 25 and tb.height < 60:
                if tb.bbox[3] <= cb[1] + 2 or direction == "down":
                    absorbed.append(tb)
                    box = union([box, tb.bbox])
    box = (
        max(box[0], page_rect.x0), max(box[1], page_rect.y0),
        min(box[2], page_rect.x1), min(box[3], page_rect.y1),
    )
    return box, absorbed
