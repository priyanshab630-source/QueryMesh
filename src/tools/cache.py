"""Exact-match response cache.

Key = SHA256(model + prompt). Upstash/Redis when REDIS_URL is set, otherwise an
in-process TTL dict - same interface, so the agents never branch on backend.
Repeated identical questions return in microseconds instead of re-running the
whole graph.
"""

import hashlib
import json
import time

import config

_memory = {}
_redis_client = None
_redis_failed = False


def key_for(*parts) -> str:
    digest = hashlib.sha256("||".join(str(p) for p in parts).encode())
    return f"querymesh:{digest.hexdigest()}"


def _redis():
    """Lazily connect; disable the backend after one failure."""
    global _redis_client, _redis_failed
    if _redis_failed or not config.REDIS_URL:
        return None
    if _redis_client is None:
        try:
            import redis

            _redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
            _redis_client.ping()
        except Exception as exc:                 # noqa: BLE001
            print(f"[cache] Redis unavailable ({type(exc).__name__}); using memory")
            _redis_failed = True
            return None
    return _redis_client


def get(key: str):
    client = _redis()
    if client is not None:
        raw = client.get(key)
        return json.loads(raw) if raw else None

    entry = _memory.get(key)
    if not entry:
        return None
    value, expires_at = entry
    if expires_at < time.time():
        _memory.pop(key, None)
        return None
    return value


def set(key: str, value, ttl: int = None) -> None:
    ttl = ttl or config.CACHE_TTL_SECONDS
    client = _redis()
    if client is not None:
        client.set(key, json.dumps(value), ex=ttl)     # setex is deprecated
        return
    _memory[key] = (value, time.time() + ttl)


def backend_name() -> str:
    return "redis" if _redis() is not None else "memory"


def stats() -> dict:
    return {"backend": backend_name(), "memory_entries": len(_memory)}
