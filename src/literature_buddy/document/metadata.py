"""Scientific-paper structure heuristics: headings, section kinds, title/authors, references."""

from __future__ import annotations

import re

from .layout import Block
from .schema import Reference

# --- section kinds -----------------------------------------------------------------------
_KIND_PATTERNS: list[tuple[str, str]] = [
    ("abstract", r"abstract|summary|significance statement|author summary"),
    ("introduction", r"introduction|background|overview|motivation"),
    ("related_work", r"related work|literature review|prior work|previous work"),
    ("methods", r"(materials? and )?methods?( and materials?)?|methodology|experimental (setup|procedures?|design|methods?|section)|"
                r"approach|study design|patients? and methods|proposed method|implementation( details)?|"
                r"data( collection| sets?)?|datasets?|model|models"),
    ("results", r"results?( and discussion)?|experiments?|experimental results|evaluation|findings"),
    ("discussion", r"discussion|analysis"),
    ("limitations", r"limitations?( and future work)?|threats to validity"),
    ("conclusion", r"conclusions?( and future (work|directions))?|concluding remarks|future work|summary and outlook"),
    ("references", r"references?|bibliography|literature cited|works cited"),
    ("acknowledgments", r"acknowledge?ments?|funding|author contributions?|conflicts? of interest|"
                        r"competing interests?|data availability( statement)?|code availability|ethics statement"),
    ("supplementary", r"supplement(ary|al)( materials?| information| data)?|appendix|appendices|supporting information"),
]
_KIND_RES = [(k, re.compile(p, re.I)) for k, p in _KIND_PATTERNS]

_NUM_PREFIX = re.compile(
    r"^\s*(?:(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?)|(?:[IVX]{1,4}\.)|(?:[A-Z][.)]))\s+"
)
_NUM_HEAD = re.compile(
    r"^(?P<num>(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]{1,4}\.|[A-Z][.)]))\s+(?P<title>[A-Z\u00C0-\u00DE][^\n]{1,110})$"
)
_ABSTRACT_RUNIN = re.compile(r"^\s*abstract\b[\s\-\u2013\u2014:.]*", re.I)


def strip_numbering(title: str) -> str:
    return _NUM_PREFIX.sub("", title).strip().rstrip(":").strip()


def canonical_kind(title: str) -> str | None:
    t = strip_numbering(title).lower().strip(" .:")
    for kind, rx in _KIND_RES:
        if rx.fullmatch(t):
            return kind
    return None


def heading_info(b: Block, body_size: float) -> tuple[int, str, str] | None:
    """Return (level, title, kind) if the block looks like a section heading, else None."""
    t = b.text.strip()
    words = t.split()
    if not t or len(words) > 14 or b.n_lines > 2 or len(t) < 3:
        return None
    stripped = strip_numbering(t)
    kind = canonical_kind(t)
    emphasised = b.bold or b.size >= body_size * 1.08 or (t.isupper() and len(words) <= 6)
    m = _NUM_HEAD.match(t)

    if kind and emphasised and len(stripped.split()) <= 7 and not t.endswith((",", ";")):
        level = t.split()[0].count(".") + 1 if m and t[0].isdigit() else 1
        return min(level, 3), stripped.title() if stripped.isupper() else stripped, kind
    if m and emphasised and not t.endswith((".", ",", ";")) and len(words) <= 12:
        num = m.group("num").rstrip(".)")
        level = num.count(".") + 1 if num[0].isdigit() else 1
        return min(level, 3), stripped, kind or "other"
    if b.size >= body_size * 1.2 and b.bold and len(words) <= 10 and not t.endswith((".", ",")):
        return 1, stripped, kind or "other"
    return None


def abstract_runin(b: Block) -> str | None:
    """'Abstract— text ...' run-in paragraph: returns the text after the label."""
    m = _ABSTRACT_RUNIN.match(b.text)
    if m and len(b.text) - m.end() > 80:
        return b.text[m.end():].strip()
    return None


# --- title & authors ---------------------------------------------------------------------
_AFFIL = re.compile(
    r"univers|institut|department|school of|hospital|laborator|college|centre|center|@|"
    r"faculty|academy|corresponding|received|accepted|copyright|doi\b|http",
    re.I,
)
_NAME = re.compile(r"^[A-Z\u00C0-\u00DE][\w.'\u2019\-\u00C0-\u017F]*(?:\s+[A-Za-z\u00C0-\u017F.'\u2019\-]+){0,4}$")


def guess_title(first_page: list[Block], body_size: float, page_height: float, fallback: str) -> tuple[str, list[Block]]:
    """Largest-font text in the top half of page 1. Returns (title, title_blocks)."""
    cands = [b for b in first_page if b.role == "body" and b.y0 < page_height * 0.5 and 2 <= len(b.text.split()) <= 45]
    if not cands:
        return fallback, []
    top = max(b.size for b in cands)
    if top < body_size * 1.2 and fallback:
        return fallback, []
    title_blocks = sorted((b for b in cands if b.size >= top - 0.6), key=lambda b: (b.y0, b.x0))
    text = " ".join(b.text for b in title_blocks).strip()
    return (text or fallback), title_blocks


def guess_authors(ordered_first_page: list[Block], title_blocks: list[Block], title_size: float) -> list[str]:
    """Names from the 1-3 blocks following the title (best effort; GROBID is more reliable)."""
    if not title_blocks:
        return []
    last = title_blocks[-1]
    idx = next((i for i, b in enumerate(ordered_first_page) if b is last), -1)
    names: list[str] = []
    for b in ordered_first_page[idx + 1: idx + 4]:
        if b.size >= title_size - 0.6 or len(b.text.split()) > 70:
            continue
        if re.match(r"\s*abstract", b.text, re.I) or _AFFIL.search(b.text):
            continue
        cleaned = re.sub(r"[\d*\u2020\u2021\u00a7\u00b6\u2217]+", " ", b.text)
        for part in re.split(r",|;|\band\b|\u00b7|\u2022|&", cleaned):
            part = " ".join(part.split())
            if part and _NAME.match(part) and 2 <= len(part.split()) <= 5:
                names.append(part)
        if names:
            break
    return names[:40]


# --- references --------------------------------------------------------------------------
_REF_START = re.compile(r"(?:(?<=\s)|^)\[?(\d{1,3})\]?[.)]?\s+(?=[A-Z\u00C0-\u00DE])")


def parse_references(blocks: list[Block]) -> list[Reference]:
    """Split the reference list into entries. Numbered lists are split on their numbers;
    otherwise every text block is treated as one entry."""
    refs: list[Reference] = []
    if not blocks:
        return refs
    joined = "\n".join(b.text for b in blocks)
    starts = [m for m in _REF_START.finditer(joined)]
    if len(starts) >= 3 and starts[0].group(1) in {"1", "0"}:
        for i, m in enumerate(starts):
            end = starts[i + 1].start() if i + 1 < len(starts) else len(joined)
            text = " ".join(joined[m.end():end].split())
            if len(text) > 15:
                page = _page_at(blocks, joined, m.start())
                refs.append(Reference(index=len(refs) + 1, text=text, page=page))
        return refs
    for b in blocks:
        text = " ".join(b.text.split())
        if len(text) > 25:
            refs.append(Reference(index=len(refs) + 1, text=text, page=b.page))
    return refs


def _page_at(blocks: list[Block], joined: str, offset: int) -> int:
    pos = 0
    for b in blocks:
        pos += len(b.text) + 1
        if offset < pos:
            return b.page
    return blocks[-1].page
