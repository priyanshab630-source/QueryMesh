"""The Redis cache path, exercised against a real redis-py client (fakeredis)."""

import pytest

fakeredis = pytest.importorskip("fakeredis", reason="fakeredis is a dev extra")
redis = pytest.importorskip("redis", reason="redis-py is an optional extra")

import config
from src.tools import cache


@pytest.fixture
def redis_cache(monkeypatch):
    server = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(config, "REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(redis, "from_url", lambda *a, **k: server)
    monkeypatch.setattr(cache, "_redis_client", None)
    monkeypatch.setattr(cache, "_redis_failed", False)
    yield server
    cache._redis_client = None
    cache._redis_failed = False


def test_roundtrip_through_redis(redis_cache):
    key = cache.key_for("redis-test", 1)
    cache.set(key, {"answer": "cached in redis"}, ttl=60)

    stored = redis_cache.get(key)
    assert stored is not None                     # actually written to the server
    assert cache.get(key) == {"answer": "cached in redis"}
    assert cache.backend_name() == "redis"


def test_miss_returns_none(redis_cache):
    assert cache.get(cache.key_for("never-written")) is None


def test_ttl_is_applied(redis_cache):
    key = cache.key_for("ttl-test")
    cache.set(key, "value", ttl=30)
    assert 0 < redis_cache.ttl(key) <= 30


def test_falls_back_to_memory_when_redis_breaks(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(cache, "_redis_client", None)
    monkeypatch.setattr(cache, "_redis_failed", False)

    def boom(*args, **kwargs):
        raise ConnectionError("no server")

    monkeypatch.setattr(redis, "from_url", boom)
    key = cache.key_for("fallback")
    cache.set(key, {"v": 1}, ttl=30)

    assert cache.get(key) == {"v": 1}
    assert cache.backend_name() == "memory"
    cache._redis_failed = False
