"""Hybrid index: corpus loading, chunking, search results."""

import config
from src.search.index import HybridIndex, chunk_text, get_index


def test_corpus_loads():
    index = get_index()
    assert len(index.papers) >= 5
    assert len(index.chunks) >= len(index.papers)


def test_chunking_respects_size():
    chunks = chunk_text(" ".join(["word"] * 250), size=90)
    assert len(chunks) == 3
    assert all(len(c.split()) <= 90 for c in chunks)


def test_search_returns_ranked_hits():
    index = get_index()
    hits = index.search("retrieval augmented generation", k=3)
    assert hits
    assert hits[0]["score"] >= hits[-1]["score"]
    assert "chunk" in hits[0] and hits[0]["chunk"]["paper_id"]


def test_hybrid_uses_both_rankers():
    index = get_index()
    hits = index.search("graph rag community summaries", k=5)
    assert any(hit["bm25_rank"] for hit in hits)


def test_empty_index_is_safe():
    assert HybridIndex([]).search("anything") == []
