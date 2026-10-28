"""OpenSearch BM25 backend.

The in-process BM25 in `src/search/bm25.py` is the default because it needs no
infrastructure. When `SEARCH_BACKEND=opensearch` and `OPENSEARCH_URL` is set, the
keyword half of the hybrid retriever comes from a real cluster instead - same
interface, so `HybridIndex` does not care which one answered.

Index layout: one document per chunk, `text` analysed with the standard analyser,
`paper_id`/`title` as keyword+text fields for filtering and display.
"""

import config


def available() -> bool:
    try:
        import opensearchpy  # noqa: F401

        return True
    except ImportError:
        return False


def client():
    """An OpenSearch client, or None when not configured/installed."""
    if not (config.SEARCH_BACKEND == "opensearch" and config.OPENSEARCH_URL
            and available()):
        return None
    from opensearchpy import OpenSearch

    return OpenSearch(hosts=[config.OPENSEARCH_URL])


MAPPING = {
    "settings": {"index": {"k1": 1.5, "b": 0.75}},
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "paper_id": {"type": "keyword"},
            "title": {"type": "text"},
            "published": {"type": "keyword"},
            "url": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "standard"},
        }
    },
}


def ensure_index(search_client, index: str = None) -> str:
    """Create the index if it does not exist yet."""
    index = index or config.OPENSEARCH_INDEX
    if not search_client.indices.exists(index=index):
        search_client.indices.create(index=index, body=MAPPING)
    return index


def bulk_index(search_client, chunks: list, index: str = None) -> int:
    """Index every chunk. Returns how many documents were accepted."""
    index = index or config.OPENSEARCH_INDEX
    if not chunks:
        return 0

    body = []
    for chunk in chunks:
        body.append({"index": {"_index": index, "_id": chunk["chunk_id"]}})
        body.append({
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "title": chunk.get("title", ""),
            "published": chunk.get("published", ""),
            "url": chunk.get("url", ""),
            "text": chunk["text"],
        })
    response = search_client.bulk(body=body, refresh=True)
    if response.get("errors"):
        failed = sum(1 for item in response.get("items", [])
                     if item.get("index", {}).get("error"))
        raise RuntimeError(f"{failed} chunk(s) failed to index")
    return len(response.get("items", []))


def query_dsl(query: str, k: int) -> dict:
    """The BM25 query: title boosted over body text."""
    return {
        "size": k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "text"],
                "type": "best_fields",
            }
        },
        "_source": ["chunk_id", "paper_id", "title"],
    }


def search(search_client, query: str, k: int, index: str = None) -> list:
    """Returns [(chunk_id, score)] ranked by BM25."""
    index = index or config.OPENSEARCH_INDEX
    response = search_client.search(index=index, body=query_dsl(query, k))
    return [(hit["_id"], float(hit["_score"]))
            for hit in response.get("hits", {}).get("hits", [])]


def backend_name() -> str:
    return "opensearch" if client() is not None else "builtin-bm25"
