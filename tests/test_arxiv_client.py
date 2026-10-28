"""arXiv ingestion: Atom parsing, PDF link extraction and persistence.

The HTTP layer is replaced with the recorded feed in data/fixtures, so the parser
and the ingest path are tested without touching the network.
"""

import json
from pathlib import Path

import config
from src.tools import arxiv_client

FIXTURE = Path("data/fixtures/arxiv_sample.xml")


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


def test_search_parses_the_atom_feed(monkeypatch):
    monkeypatch.setattr(arxiv_client.requests, "get",
                        lambda *a, **k: FakeResponse(FIXTURE.read_text()))
    papers = arxiv_client.search("retrieval", max_results=2)

    assert len(papers) == 2
    first = papers[0]
    assert first["id"] == "2404.16130v2"
    assert first["title"].startswith("From Local to Global")
    assert first["published"] == "2024-04-24"
    assert first["authors"] == ["Darren Edge", "Ha Trinh"]
    assert first["pdf_url"] == "http://arxiv.org/pdf/2404.16130v2"
    assert first["url"].endswith("2404.16130v2")


def test_search_returns_empty_when_offline(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("blocked")

    monkeypatch.setattr(arxiv_client.requests, "get", boom)
    assert arxiv_client.search("anything") == []


def test_ingest_writes_papers_to_the_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(arxiv_client.requests, "get",
                        lambda *a, **k: FakeResponse(FIXTURE.read_text()))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    papers = arxiv_client.ingest("retrieval", limit=2)

    assert len(papers) == 2
    written = sorted((tmp_path / "papers").glob("*.json"))
    assert len(written) == 2
    stored = json.loads(written[0].read_text())
    assert stored["source"] == "arxiv"


def test_download_pdf_writes_the_file(monkeypatch, tmp_path):
    class PdfResponse:
        content = b"%PDF-1.4 fake"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(arxiv_client.requests, "get", lambda *a, **k: PdfResponse())
    path = arxiv_client.download_pdf("2404.16130", destination=tmp_path / "p.pdf")

    assert path.exists() and path.read_bytes().startswith(b"%PDF")
