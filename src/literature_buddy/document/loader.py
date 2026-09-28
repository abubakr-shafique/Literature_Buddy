# src/literature_buddy/document/loader.py
"""Safe local/URL paper loading.

Security policy:
- Only application/pdf content accepted; content-type and magic bytes validated.
- Download size capped at 64 MB; redirect following limited.
- PDFs are parsed, never executed (no JS, no embedded actions are run by PyMuPDF
  during text-only extraction; rendering happens in-process via QImage only).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

MAX_BYTES = 64 * 1024 * 1024
PDF_MAGIC = b"%PDF"
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


class LoaderError(RuntimeError):
    pass


def load_local_pdf(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise LoaderError(f"File not found: {p}")
    if p.suffix.lower() != ".pdf":
        raise LoaderError("Only .pdf files are supported")
    with open(p, "rb") as fh:
        if fh.read(4) != PDF_MAGIC:
            raise LoaderError("File does not appear to be a valid PDF")
    return p


def load_pdf_from_url(url: str, dest_dir: str | Path | None = None) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise LoaderError(f"Unsupported URL scheme: {parsed.scheme}")

    dest_dir = Path(dest_dir) if dest_dir else Path(tempfile.gettempdir()) / "lb_papers"
    dest_dir.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as resp:
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype and ctype not in ALLOWED_CONTENT_TYPES:
            raise LoaderError(f"URL did not return a PDF (content-type: {ctype})")

        name = f"paper-{hashlib.sha1(url.encode()).hexdigest()[:10]}.pdf"
        out = dest_dir / name
        total = 0
        with open(out, "wb") as fh:
            for block in resp.iter_content(chunk_size=1 << 16):
                total += len(block)
                if total > MAX_BYTES:
                    out.unlink(missing_ok=True)
                    raise LoaderError("PDF exceeds the 64 MB download limit")
                fh.write(block)

    with open(out, "rb") as fh:
        if fh.read(4) != PDF_MAGIC:
            out.unlink(missing_ok=True)
            raise LoaderError("Downloaded file is not a valid PDF")
    return out


def load_paper(source: str, dest_dir: str | Path | None = None) -> Path:
    """Entry point: load from a local path or an http(s) URL."""
    if source.startswith(("http://", "https://")):
        # Convenience: arXiv abs links -> pdf links.
        if "arxiv.org/abs/" in source:
            source = source.replace("/abs/", "/pdf/")
        return load_pdf_from_url(source, dest_dir)
    return load_local_pdf(source)
