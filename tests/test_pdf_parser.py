"""PDF parsing: pypdf by default, Docling when installed and enabled."""

import os

import pymupdf
import pytest

import config
from src.tools import pdf_parser


@pytest.fixture
def sample_pdf(tmp_path):
    """A small two-paragraph PDF with a text layer."""
    path = tmp_path / "paper.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "QueryMesh test paper", fontsize=18)
    page.insert_text((72, 110), "Graph RAG summarises community reports.", fontsize=11)
    page.insert_text((72, 130), "Hybrid retrieval fuses BM25 with dense vectors.", fontsize=11)
    document.save(str(path))
    document.close()
    return path


def test_default_backend_is_pypdf(monkeypatch):
    monkeypatch.setattr(config, "USE_DOCLING", False)
    assert pdf_parser.backend() == "pypdf"


def test_pypdf_extracts_the_text(monkeypatch, sample_pdf):
    monkeypatch.setattr(config, "USE_DOCLING", False)
    text = pdf_parser.parse(sample_pdf)
    assert "Graph RAG summarises community reports" in text


def test_docling_extracts_the_text(monkeypatch, sample_pdf):
    """Opt-in: Docling loads layout models and needs ~1.5 GB of RAM."""
    monkeypatch.setattr(config, "USE_DOCLING", True)
    if pdf_parser.backend() != "docling":
        pytest.skip("docling is not installed")
    if not os.getenv("DOCLING_TEST"):
        pytest.skip("set DOCLING_TEST=1 to run the Docling parse (memory heavy)")
    assert pdf_parser.backend() == "docling"
    text = pdf_parser.parse(sample_pdf)
    assert "QueryMesh test paper" in text
    assert "BM25" in text


def test_sections_split(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "USE_DOCLING", False)
    path = tmp_path / "paper.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Abstract", fontsize=14)
    page.insert_text((72, 96), "We study retrieval augmented generation systems in depth.", fontsize=11)
    page.insert_text((72, 130), "Introduction", fontsize=14)
    page.insert_text((72, 154), "Prior work covers BM25 and dense retrieval baselines.", fontsize=11)
    page.insert_text((72, 190), "References", fontsize=14)
    page.insert_text((72, 214), "[1] Lewis et al. 2020.", fontsize=11)
    document.save(str(path))
    document.close()

    sections = pdf_parser.parse_sections(path)
    assert "retrieval augmented generation" in sections["abstract"].lower()
    assert "References" not in sections["body"]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pdf_parser.parse("does-not-exist.pdf")
