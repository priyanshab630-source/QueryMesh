"""Exact-match cache: keys, TTL, backend reporting."""

from src.tools import cache


def test_key_is_stable_and_distinct():
    assert cache.key_for("a", "b") == cache.key_for("a", "b")
    assert cache.key_for("a", "b") != cache.key_for("a", "c")


def test_set_then_get():
    key = cache.key_for("unit-test")
    cache.set(key, {"answer": "cached"}, ttl=60)
    assert cache.get(key) == {"answer": "cached"}


def test_missing_key_returns_none():
    assert cache.get(cache.key_for("never-stored")) is None


def test_backend_name():
    assert cache.backend_name() in ("memory", "redis")
