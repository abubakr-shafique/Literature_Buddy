"""Open a paper from a local file or a URL, safely.

Safety properties (see docs/security):
  * only http/https; hosts resolving to private/loopback/link-local addresses are refused
  * redirects are followed manually and re-validated at every hop
  * download size is capped; the payload must start with %PDF- and must open in PyMuPDF
  * nothing from the PDF is ever executed (we only render/extract with MuPDF)
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import pymupdf

from ..errors import Cancelled, DownloadError

log = logging.getLogger(__name__)
USER_AGENT = "LiteratureBuddy/0.1 (+https://github.com/your-org/literature-buddy)"
ProgressFn = Callable[[float, str], None]


# ---- URL normalisation ---------------------------------------------------------------------
def candidate_urls(url: str) -> list[str]:
    """Best-guess direct-PDF URLs for common open-access hosts (original URL is always last)."""
    u = url.strip()
    p = urlparse(u)
    host = p.netloc.lower().removeprefix("www.")
    path = p.path
    out: list[str] = []
    if host == "arxiv.org":
        m = re.match(r"^/(?:abs|html|pdf)/(.+?)(?:\.pdf)?/?$", path)
        if m:
            out.append(f"https://arxiv.org/pdf/{m.group(1)}")
    elif host.endswith(("biorxiv.org", "medrxiv.org")) and "/content/" in path:
        base = re.sub(r"(\.full(\.pdf)?|\.pdf|/)$", "", path)
        base = re.sub(r"\.(full|abstract|article-info)$", "", base)
        out.append(f"{p.scheme}://{p.netloc}{base}.full.pdf")
    elif host in ("ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov") and (m := re.search(r"(PMC\d+)", path)):
        pmc = m.group(1)
        out.append(f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmc}&blobtype=pdf")
        out.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/pdf/")
    elif host == "europepmc.org" and (m := re.search(r"(PMC\d+)", u)):
        out.append(f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={m.group(1)}&blobtype=pdf")
    out.append(u)
    return list(dict.fromkeys(out))


_META_PDF = [
    re.compile(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]*content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*name=["\']citation_pdf_url["\']', re.I),
    re.compile(r'<link[^>]+type=["\']application/pdf["\'][^>]*href=["\']([^"\']+)', re.I),
]


def find_pdf_link(html: str, base_url: str) -> str | None:
    for rx in _META_PDF:
        m = rx.search(html)
        if m:
            return urljoin(base_url, m.group(1))
    return None


# ---- host safety ---------------------------------------------------------------------------
def assert_public_http_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise DownloadError("Only http(s) URLs are supported.")
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror as exc:
        raise DownloadError(f"Cannot resolve host '{p.hostname}'. Are you offline?") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise DownloadError("Refusing to download from a private or local network address.")


# ---- local files ---------------------------------------------------------------------------
def validate_pdf(path: Path) -> None:
    """Cheap structural validation. Raises DownloadError if `path` is not a readable PDF."""
    try:
        with path.open("rb") as fh:
            head = fh.read(1024)
    except OSError as exc:
        raise DownloadError(f"Cannot read file: {exc}") from exc
    if b"%PDF-" not in head:
        raise DownloadError("The file is not a PDF (missing %PDF header).")
    try:
        with pymupdf.open(path) as doc:
            if doc.page_count < 1:
                raise DownloadError("The PDF has no pages.")
    except DownloadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DownloadError(f"The PDF could not be opened: {exc}") from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- download ------------------------------------------------------------------------------
def _fetch(
    client: httpx.Client, url: str, max_bytes: int, dest: Path, progress: ProgressFn | None,
    cancel: threading.Event | None,
) -> tuple[str, str | None]:
    """GET with manual, validated redirects. Returns ("pdf", None) or ("html", body)."""
    for _hop in range(6):
        assert_public_http_url(url)
        with client.stream("GET", url, follow_redirects=False) as r:
            if r.status_code in (301, 302, 303, 307, 308) and "location" in r.headers:
                url = urljoin(url, r.headers["location"])
                continue
            if r.status_code >= 400:
                raise DownloadError(f"Server returned HTTP {r.status_code} for {url}")
            ctype = r.headers.get("content-type", "").lower()
            total = int(r.headers.get("content-length", 0) or 0)
            if total and total > max_bytes:
                raise DownloadError(f"File is larger than the {max_bytes // 1_000_000} MB limit.")
            buf = bytearray()
            first = True
            is_pdf = "pdf" in ctype
            with dest.open("wb") as out:
                for chunk in r.iter_bytes(64 * 1024):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled("Download cancelled")
                    if first:
                        first = False
                        is_pdf = is_pdf or chunk.startswith(b"%PDF-")
                    if is_pdf:
                        out.write(chunk)
                    else:
                        buf += chunk
                    got = out.tell() if is_pdf else len(buf)
                    if got > max_bytes:
                        raise DownloadError(f"Download exceeded the {max_bytes // 1_000_000} MB limit.")
                    if progress and total:
                        progress(min(got / total, 1.0), "Downloading PDF")
            if is_pdf:
                return "pdf", None
            return "html", buf.decode("utf-8", errors="replace")
    raise DownloadError("Too many redirects.")


def download_pdf(
    url: str, dest_dir: Path, max_mb: int = 100, progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Download an open-access PDF (resolving common landing pages) into `dest_dir`."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.pdf"  # noqa: S324 - not security
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,text/html;q=0.8,*/*;q=0.5"}
    last_error = "unknown error"
    with httpx.Client(headers=headers, timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        for cand in candidate_urls(url):
            try:
                kind, body = _fetch(client, cand, max_mb * 1_000_000, dest, progress, cancel)
                if kind == "html" and body:
                    link = find_pdf_link(body, cand)
                    if link and link != cand:
                        kind, body = _fetch(client, link, max_mb * 1_000_000, dest, progress, cancel)
                if kind == "pdf":
                    validate_pdf(dest)
                    return dest
                last_error = "the page did not link to a PDF"
            except (DownloadError, httpx.HTTPError) as exc:
                last_error = str(exc) or exc.__class__.__name__
                log.info("Candidate %s failed: %s", cand, last_error)
    dest.unlink(missing_ok=True)
    raise DownloadError(
        f"Could not download a PDF ({last_error}). Many publishers block automated downloads - "
        "download the PDF in your browser and open it from disk instead."
    )


def is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip(), re.I))
