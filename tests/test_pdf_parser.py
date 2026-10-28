"""PDF parser selection: pypdf by default, Docling when enabled and installed."""

import config
from src.tools import pdf_parser


def test_default_backend_is_pypdf(monkeypatch):
    monkeypatch.setattr(config, "USE_DOCLING", False)
    assert pdf_parser.backend() == "pypdf"


def test_docling_flag_falls_back_when_missing(monkeypatch):
    monkeypatch.setattr(config, "USE_DOCLING", True)
    backend = pdf_parser.backend()
    assert backend.startswith("pypdf")           # docling is not installed here


def test_missing_file_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        pdf_parser.parse("does-not-exist.pdf")
