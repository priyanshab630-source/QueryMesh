"""Embedding backends: hashed vectors, and the Jina API path with mocked HTTP."""

import math

import pytest

import config
from src.search import embeddings


def test_hashed_vectors_are_deterministic_and_normalised():
    a = embeddings._hashed_vector("retrieval augmented generation")
    b = embeddings._hashed_vector("retrieval augmented generation")
    assert a == b
    assert len(a) == config.EMBEDDING_DIM
    assert math.isclose(sum(v * v for v in a), 1.0, rel_tol=1e-6)


def test_cosine_of_identical_text_is_one():
    vector = embeddings._hashed_vector("graph rag")
    assert embeddings.cosine(vector, vector) == pytest.approx(1.0)


def test_jina_path_is_used_when_a_key_exists(monkeypatch):
    monkeypatch.setattr(config, "JINA_API_KEY", "test-key")

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.6, 0.8]}, {"embedding": [0.0, 1.0]}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    vectors = embeddings.embed_many(["first text", "second text"])

    assert captured["url"].endswith("/v1/embeddings")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["input"] == ["first text", "second text"]
    assert vectors == [[0.6, 0.8], [0.0, 1.0]]


def test_jina_failure_falls_back_to_hashed(monkeypatch):
    monkeypatch.setattr(config, "JINA_API_KEY", "test-key")

    def boom(*args, **kwargs):
        raise ConnectionError("api down")

    monkeypatch.setattr(embeddings.requests, "post", boom)
    vectors = embeddings.embed_many(["fallback text"])
    assert len(vectors[0]) == config.EMBEDDING_DIM


def test_backend_name(monkeypatch):
    monkeypatch.setattr(config, "JINA_API_KEY", "")
    assert embeddings.backend_name() == "hashed-vectors"
    monkeypatch.setattr(config, "JINA_API_KEY", "key")
    assert embeddings.backend_name() == "jina"
