"""Dense retrieval: Jina AI embeddings when a key exists, hashed vectors otherwise.

The fallback is the "hashing trick": every token is hashed into a fixed-width
vector with a sign. It is not semantic, but it is deterministic, needs no model
download and keeps the hybrid pipeline fully runnable offline.
"""

import hashlib
import math

import requests

import config
from src.search.bm25 import tokenize


def _hashed_vector(text: str) -> list:
    vector = [0.0] * config.EMBEDDING_DIM
    tokens = tokenize(text)
    if not tokens:
        return vector
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    for token, count in counts.items():
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % config.EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log(count))
    return _normalise(vector)


def _normalise(vector: list) -> list:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


def _jina_vectors(texts: list) -> list:
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {config.JINA_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": config.EMBEDDING_MODEL, "input": texts,
              "task": "retrieval.passage"},
        timeout=40,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return [_normalise(item["embedding"]) for item in data]


def embed_many(texts: list) -> list:
    """Embed a batch, falling back silently to hashed vectors."""
    if config.JINA_API_KEY:
        try:
            return _jina_vectors(texts)
        except Exception as exc:                  # noqa: BLE001
            print(f"[embeddings] Jina unavailable ({type(exc).__name__}); "
                  "using hashed vectors")
    return [_hashed_vector(text) for text in texts]


def embed(text: str) -> list:
    return embed_many([text])[0]


def cosine(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def backend_name() -> str:
    return "jina" if config.JINA_API_KEY else "hashed-vectors"
