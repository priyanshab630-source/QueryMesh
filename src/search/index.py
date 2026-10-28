"""Hybrid index: BM25 + dense vectors + RRF over paper chunks.

Chunks are the retrieval unit; every chunk keeps its paper id so the citation
agent can point back to a real arXiv entry.
"""

import json
import sys

import config
from src.search import embeddings, opensearch_backend, rrf
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

        # Optional: real BM25 from OpenSearch instead of the in-process index.
        self.os_client = opensearch_backend.client()
        self.os_index = None
        if self.os_client is not None:
            try:
                self.os_index = opensearch_backend.ensure_index(self.os_client)
                opensearch_backend.bulk_index(self.os_client, self.chunks, self.os_index)
            except Exception as exc:              # noqa: BLE001 - degrade, don't fail
                print(f"[index] OpenSearch unavailable ({type(exc).__name__}); "
                      "using the in-process BM25", file=sys.stderr)
                self.os_client, self.os_index = None, None

    def _opensearch_hits(self, query: str, k: int) -> list:
        """Map OpenSearch chunk ids back to local chunk indices."""
        by_id = {chunk["chunk_id"]: index for index, chunk in enumerate(self.chunks)}
        hits = opensearch_backend.search(self.os_client, query,
                                         min(len(self.chunks), k * 3), self.os_index)
        return [(by_id[chunk_id], score) for chunk_id, score in hits
                if chunk_id in by_id]

    def search(self, query: str, k: int = None) -> list:
        """Returns [{"chunk": ..., "score": ..., "bm25_rank": ..., "dense_rank": ...}]."""
        k = k or config.TOP_K
        if not self.chunks:
            return []

        if self.os_client is not None:
            keyword_hits = self._opensearch_hits(query, k)
        else:
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
