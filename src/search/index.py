"""Hybrid index: BM25 + dense vectors + RRF over paper chunks.

Chunks are the retrieval unit; every chunk keeps its paper id so the citation
agent can point back to a real arXiv entry.
"""

import json

import config
from src.search import embeddings, rrf
from src.search.bm25 import BM25


def chunk_text(text: str, size: int) -> list:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [text]


def load_corpus() -> list:
    """Sample corpus bundled with the repo, plus anything ingested into data/papers."""
    papers = []
    for directory in (config.PAPERS_DIR, config.DATA_DIR / "papers"):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            papers.extend(record if isinstance(record, list) else [record])
    return papers


class HybridIndex:
    def __init__(self, papers: list):
        self.papers = papers
        self.chunks = []
        for paper in papers:
            body = paper.get("abstract", "")
            for number, text in enumerate(chunk_text(body, config.CHUNK_WORDS)):
                self.chunks.append({
                    "chunk_id": f"{paper.get('id', 'unknown')}#{number}",
                    "paper_id": paper.get("id", "unknown"),
                    "title": paper.get("title", ""),
                    "published": paper.get("published", ""),
                    "url": paper.get("url", ""),
                    "text": text,
                })

        # The search index sees title + abstract (titles carry a lot of signal);
        # the retrieved chunk text stays abstract-only so quotes read cleanly.
        self.texts = [f"{c['title']} {c['text']}" for c in self.chunks]
        self.bm25 = BM25(self.texts)
        self.vectors = embeddings.embed_many(self.texts) if self.texts else []

    def search(self, query: str, k: int = None) -> list:
        """Returns [{"chunk": ..., "score": ..., "bm25_rank": ..., "dense_rank": ...}]."""
        k = k or config.TOP_K
        if not self.chunks:
            return []

        keyword_hits = self.bm25.top_k(query, k=min(len(self.chunks), k * 3))

        query_vector = embeddings.embed(query)
        dense_scored = [
            (i, embeddings.cosine(query_vector, vector))
            for i, vector in enumerate(self.vectors)
        ]
        dense_scored.sort(key=lambda pair: pair[1], reverse=True)
        dense_hits = [(i, s) for i, s in dense_scored[:min(len(self.chunks), k * 3)] if s > 0]

        fused = rrf.fuse([keyword_hits, dense_hits])[:k]

        keyword_rank = {i: r for r, (i, _) in enumerate(keyword_hits, start=1)}
        dense_rank = {i: r for r, (i, _) in enumerate(dense_hits, start=1)}

        return [
            {
                "chunk": self.chunks[i],
                "score": round(score, 6),
                "bm25_rank": keyword_rank.get(i),
                "dense_rank": dense_rank.get(i),
            }
            for i, score in fused
        ]


_CACHE = {}


def get_index() -> HybridIndex:
    """Build once per process."""
    if "index" not in _CACHE:
        _CACHE["index"] = HybridIndex(load_corpus())
    return _CACHE["index"]


def rebuild() -> HybridIndex:
    _CACHE.pop("index", None)
    return get_index()
