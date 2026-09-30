import json

import httpx
import pytest

from literature_buddy.config.settings import ModelConfig
from literature_buddy.document import loader
from literature_buddy.errors import DownloadError, ModelUnavailableError
from literature_buddy.models.ollama_backend import OllamaBackend


def test_candidate_urls():
    assert loader.candidate_urls("https://arxiv.org/abs/2401.01234")[0] == "https://arxiv.org/pdf/2401.01234"
    assert loader.candidate_urls("https://www.biorxiv.org/content/10.1101/2020.01.01.123456v1")[0].endswith("v1.full.pdf")
    pmc = loader.candidate_urls("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/")
    assert "europepmc.org" in pmc[0] and pmc[-1].endswith("PMC123456/")


def test_find_pdf_link():
    html = '<meta name="citation_pdf_url" content="/files/a.pdf">'
    assert loader.find_pdf_link(html, "https://x.org/p/1") == "https://x.org/files/a.pdf"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x.org/a.pdf", "http://127.0.0.1/a.pdf",
                                 "http://192.168.1.5/a.pdf", "http://[::1]/a.pdf"])
def test_unsafe_urls_rejected(url):
    with pytest.raises(DownloadError):
        loader.assert_public_http_url(url)


def test_validate_pdf(tmp_path, sample_pdf):
    loader.validate_pdf(sample_pdf)
    bad = tmp_path / "x.pdf"
    bad.write_text("<html>not a pdf</html>")
    with pytest.raises(DownloadError):
        loader.validate_pdf(bad)


def _ollama(handler):
    b = OllamaBackend(ModelConfig(model="m", think=False))
    b._client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def test_ollama_stream_and_think_removal():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        lines = [{"message": {"content": "<think>hmm</think>Hel"}}, {"message": {"content": "lo"}}, {"done": True}]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines))

    out = _ollama(handler).generate([])
    assert out == "Hello" and seen["think"] is False and seen["stream"] is True


def test_ollama_missing_model_message():
    with pytest.raises(ModelUnavailableError, match="ollama pull m"):
        _ollama(lambda r: httpx.Response(404)).generate([])
