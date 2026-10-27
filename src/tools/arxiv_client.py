"""arXiv API client: search, parse the Atom feed, download PDFs.

No key needed. If the network is unavailable the caller falls back to the
bundled sample corpus, so the assistant still answers.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

import config

API_URL = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _parse_entry(entry) -> dict:
    def text(tag):
        node = entry.find(f"a:{tag}", ATOM)
        return (node.text or "").strip() if node is not None else ""

    arxiv_id = text("id").rsplit("/", 1)[-1]
    pdf_url = ""
    for link in entry.findall("a:link", ATOM):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
    return {
        "id": arxiv_id,
        "title": " ".join(text("title").split()),
        "abstract": " ".join(text("summary").split()),
        "authors": [a.findtext("a:name", default="", namespaces=ATOM)
                    for a in entry.findall("a:author", ATOM)],
        "published": text("published")[:10],
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
        "source": "arxiv",
    }


def search(query: str, max_results: int = 5) -> list:
    """Live arXiv search. Returns [] when the network is unavailable."""
    try:
        response = requests.get(
            API_URL,
            params={"search_query": f"all:{query}", "start": 0,
                    "max_results": max_results, "sortBy": "relevance"},
            timeout=25,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        return [_parse_entry(entry) for entry in root.findall("a:entry", ATOM)]
    except Exception as exc:                      # noqa: BLE001
        print(f"[arxiv] live search unavailable ({type(exc).__name__}); "
              "using the bundled corpus")
        return []


def download_pdf(arxiv_id: str, destination: Path = None) -> Path:
    """Fetch one PDF into data/pdfs/<id>.pdf."""
    destination = destination or (config.DATA_DIR / "pdfs" / f"{arxiv_id}.pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def ingest(query: str, limit: int = 5) -> list:
    """Search arXiv and persist the results into data/papers/ for the index."""
    papers = search(query, max_results=limit)
    if not papers:
        return []
    out_dir = config.DATA_DIR / "papers"
    out_dir.mkdir(parents=True, exist_ok=True)
    for paper in papers:
        safe_id = paper["id"].replace("/", "_")
        with open(out_dir / f"{safe_id}.json", "w", encoding="utf-8") as handle:
            json.dump(paper, handle, indent=2)
    return papers
