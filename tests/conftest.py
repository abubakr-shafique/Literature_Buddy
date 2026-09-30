import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_pdf import make_sample_pdf, make_two_column_pdf  # noqa: E402

from literature_buddy.config import load_config  # noqa: E402
from literature_buddy.document.processor import DocumentProcessor  # noqa: E402
from literature_buddy.retrieval.embeddings import HashingEmbedder  # noqa: E402
from literature_buddy.storage import PaperCache  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    return make_sample_pdf(tmp_path_factory.mktemp("pdf") / "sample.pdf")


@pytest.fixture(scope="session")
def two_col_pdf(tmp_path_factory) -> Path:
    return make_two_column_pdf(tmp_path_factory.mktemp("pdf2") / "two.pdf")


@pytest.fixture()
def cfg():
    return load_config(profile="lite", environ={})


@pytest.fixture()
def paper(sample_pdf, cfg, tmp_path):
    proc = DocumentProcessor(cfg, HashingEmbedder(), PaperCache(tmp_path))
    return proc.process(sample_pdf)
