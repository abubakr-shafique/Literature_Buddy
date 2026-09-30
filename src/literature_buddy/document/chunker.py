"""Scientific-document-aware chunking.

Rules:
  * never cross section boundaries or page boundaries (so a citation always has one exact page)
  * paragraphs are the atomic unit; long paragraphs are split on sentence boundaries with 1 sentence overlap
  * figures, tables and equations are separate chunks that carry their label (Figure 2, Table 1, ...)
  * figure/table chunks also carry the sentences in the text that refer to them
  * references are grouped, and are excluded from retrieval unless the question is about them
"""

from __future__ import annotations

import re

from ..config.settings import DocumentConfig
from .schema import Chunk, Document, find_mentions

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def _words(text: str) -> int:
    return len(text.split())


def split_long(text: str, max_words: int) -> list[str]:
    if _words(text) <= max_words:
        return [text]
    sents = _SENT_SPLIT.split(text)
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for s in sents:
        w = _words(s)
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = cur[-1:], _words(cur[-1])  # 1-sentence overlap
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def _mention_sentences(doc: Document, label: str, limit: int = 3) -> list[str]:
    out: list[str] = []
    for p in doc.paragraphs:
        if doc.sections[p.section_idx].kind in ("front", "references"):
            continue
        for s in _SENT_SPLIT.split(p.text):
            if label in find_mentions(s) and _words(s) <= 70 and not s.lower().startswith(label.lower() + "."):
                out.append(s.strip())
                if len(out) >= limit:
                    return out
    return out


def build_chunks(doc: Document, cfg: DocumentConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    prefix = doc.doc_id[:8]

    def add(kind: str, text: str, page: int, section: str, skind: str, label: str | None = None,
            bbox: tuple[float, float, float, float] | None = None) -> Chunk:
        c = Chunk(chunk_id=f"{prefix}-{len(chunks):04d}", kind=kind, text=text.strip(), page=page,
                  section=section, section_kind=skind, label=label, bbox=bbox,
                  refs=find_mentions(text), ord=len(chunks))
        chunks.append(c)
        return c

    if doc.abstract:
        page = next((s.page_start for s in doc.sections if s.kind == "abstract"), 0)
        add("abstract", doc.abstract, page, "Abstract", "abstract")

    # ---- body text --------------------------------------------------------------------------
    cur: list[str] = []
    cur_boxes: list[tuple[float, float, float, float]] = []
    key: tuple[int, int] | None = None
    last_key: tuple[int, int] | None = None
    last_chunk: Chunk | None = None

    def flush() -> None:
        nonlocal cur, cur_boxes, last_key, last_chunk
        if not cur or key is None:
            cur, cur_boxes = [], []
            return
        text = " ".join(cur)
        sec = doc.sections[key[0]]
        if last_chunk is not None and last_key == key and _words(text) < cfg.min_chunk_words // 2:
            last_chunk.text += " " + text  # tiny tail: fold into the previous chunk
            last_chunk.refs = find_mentions(last_chunk.text)
        else:
            box = (min(b[0] for b in cur_boxes), min(b[1] for b in cur_boxes),
                   max(b[2] for b in cur_boxes), max(b[3] for b in cur_boxes)) if cur_boxes else None
            last_chunk = add("text", text, key[1], sec.title, sec.kind, bbox=box)
            last_key = key
        cur, cur_boxes = [], []

    for p in doc.paragraphs:
        sec = doc.sections[p.section_idx]
        if sec.kind in ("front", "abstract", "references"):
            continue
        k = (p.section_idx, p.page)
        if k != key:
            flush()
            key = k
        for piece in split_long(p.text, cfg.max_chunk_words):
            if cur and _words(" ".join(cur)) + _words(piece) > cfg.max_chunk_words:
                flush()
            cur.append(piece)
            cur_boxes.append(p.bbox)
    flush()

    # ---- visuals ----------------------------------------------------------------------------
    for f in doc.figures:
        text = f.caption
        mentions = _mention_sentences(doc, f.label)
        if mentions:
            text += "\nReferred to in the text: " + " ".join(mentions)
        add("figure", text, f.page, "Figure", "figure", label=f.label, bbox=f.bbox or f.caption_bbox)
    for t in doc.tables:
        text = t.caption
        if t.markdown:
            text += "\n" + t.markdown[:3000]
        mentions = _mention_sentences(doc, t.label)
        if mentions:
            text += "\nReferred to in the text: " + " ".join(mentions)
        add("table", text, t.page, "Table", "table", label=t.label, bbox=t.bbox or t.caption_bbox)
    for e in doc.equations:
        add("equation", f"{e.label}: {e.text}", e.page, "Equation", "equation", label=e.label, bbox=e.bbox)

    # ---- references (grouped) ---------------------------------------------------------------
    for i in range(0, len(doc.references), 5):
        grp = doc.references[i: i + 5]
        add("reference", "\n".join(f"[{r.index}] {r.text}" for r in grp), grp[0].page, "References",
            "references")
    return chunks
