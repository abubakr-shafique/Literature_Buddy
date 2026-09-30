"""Table detection with PyMuPDF's `find_tables`, associated to 'Table N' captions."""

from __future__ import annotations

import logging

import pymupdf

from .figure_extractor import Caption
from .layout import Rect4, area

log = logging.getLogger(__name__)


def rows_to_markdown(rows: list[list[str | None]], max_rows: int = 80) -> str:
    clean = [[(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r] for r in rows[:max_rows]]
    if not clean:
        return ""
    ncol = max(len(r) for r in clean)
    clean = [r + [""] * (ncol - len(r)) for r in clean]
    lines = ["| " + " | ".join(clean[0]) + " |", "| " + " | ".join(["---"] * ncol) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    if len(rows) > max_rows:
        lines.append(f"| ... {len(rows) - max_rows} more rows omitted ... |" + " |" * (ncol - 1))
    return "\n".join(lines)


def _fill_ratio(rows: list[list[str | None]]) -> float:
    cells = [c for r in rows for c in r]
    return sum(1 for c in cells if c and c.strip()) / max(len(cells), 1)


def _valid(rows: list[list[str | None]], bbox: Rect4, page_rect: pymupdf.Rect) -> bool:
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    if _fill_ratio(rows) < 0.35:
        return False
    if area(bbox) > 0.75 * page_rect.width * page_rect.height:
        return False
    return not any(len(c or "") > 400 for r in rows for c in r)


def detect_tables(page: pymupdf.Page) -> list[tuple[Rect4, list[list[str | None]]]]:
    """Ruled tables detected on the page: [(bbox, rows)]."""
    out: list[tuple[Rect4, list[list[str | None]]]] = []
    try:
        finder = page.find_tables()
    except Exception as exc:  # noqa: BLE001
        log.debug("find_tables failed on page %s: %s", page.number, exc)
        return out
    for t in finder.tables:
        try:
            rows = t.extract()
        except Exception:  # noqa: BLE001
            continue
        bbox = tuple(t.bbox)
        if _valid(rows, bbox, page.rect):  # type: ignore[arg-type]
            out.append((bbox, rows))  # type: ignore[arg-type]
    return out


def match_caption(bbox: Rect4, captions: list[Caption], used: set[int]) -> Caption | None:
    """Nearest unused 'Table N' caption directly above (preferred) or below the table."""
    best: tuple[float, Caption] | None = None
    for cap in captions:
        if cap.kind != "Table" or id(cap.block) in used:
            continue
        cb = cap.bbox
        if min(cb[2], bbox[2]) - max(cb[0], bbox[0]) < 20:
            continue
        gap_above = bbox[1] - cb[3]  # caption above table
        gap_below = cb[1] - bbox[3]  # caption below table
        if -6 <= gap_above <= 70:
            d = abs(gap_above)
        elif -6 <= gap_below <= 50:
            d = abs(gap_below) + 30
        else:
            continue
        if best is None or d < best[0]:
            best = (d, cap)
    return best[1] if best else None


def text_strategy_fallback(
    page: pymupdf.Page, cap: Caption
) -> tuple[Rect4, list[list[str | None]]] | None:
    """Low-confidence detection of borderless (booktabs-style) tables below a caption."""
    pr = page.rect
    x0 = max(cap.bbox[0] - 10, pr.x0)
    x1 = min(cap.bbox[2] + 10, pr.x1)
    clip = pymupdf.Rect(x0, cap.bbox[3], x1, min(cap.bbox[3] + 260, pr.y1 - 30))
    try:
        finder = page.find_tables(clip=clip, strategy="text")
    except Exception:  # noqa: BLE001
        return None
    for t in finder.tables:
        bbox = tuple(t.bbox)
        if bbox[1] - cap.bbox[3] > 40:
            continue
        rows = t.extract()
        if _valid(rows, bbox, pr) and len(rows) >= 3:  # type: ignore[arg-type]
            return bbox, rows  # type: ignore[return-value]
    return None
