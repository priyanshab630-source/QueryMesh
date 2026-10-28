"""Minimal HS256 JWT auth - no PyJWT dependency.

Two endpoints worth knowing:
  POST /token   -> exchange client_id/client_secret for a bearer token
  any protected route requires `Authorization: Bearer <token>`

The implementation is deliberately small and readable: base64url(header) +
base64url(payload) + HMAC-SHA256 signature, `exp` enforced on verify. For a
production deployment, swap in PyJWT (or your identity provider) - the shape of
`require_token` stays the same.
"""

import base64
import hashlib
import hmac
import json
import time

from fastapi import Header, HTTPException

import config


class TokenError(Exception):
    pass


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def create_token(subject: str, scopes: list = None, ttl: int = None) -> str:
    ttl = ttl or config.JWT_TTL_SECONDS
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "scope": " ".join(scopes or ["ask", "ingest"]),
        "iat": int(time.time()),
        "exp": int(time.time()) + ttl,
    }
    signing_input = f"{_b64(json.dumps(header).encode())}." \
                    f"{_b64(json.dumps(payload).encode())}"
    signature = hmac.new(config.JWT_SECRET.encode(), signing_input.encode(),
                         hashlib.sha256).digest()
    return f"{signing_input}.{_b64(signature)}"


def verify_token(token: str) -> dict:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise TokenError("malformed token") from exc

    expected = hmac.new(config.JWT_SECRET.encode(),
                        f"{header_b64}.{payload_b64}".encode(),
                        hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _unb64(signature_b64)):
        raise TokenError("signature mismatch")

    payload = json.loads(_unb64(payload_b64))
    if payload.get("exp", 0) < time.time():
        raise TokenError("token expired")
    return payload


def require_token(authorization: str = Header(default="")) -> dict:
    """FastAPI dependency: validates the bearer token when auth is enabled."""
    if not config.AUTH_ENABLED:
        return {"sub": "anonymous", "scope": "ask ingest"}

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        return verify_token(authorization.split(" ", 1)[1].strip())
    except TokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def check_scope(claims: dict, scope: str) -> None:
    if scope not in claims.get("scope", "").split():
        raise HTTPException(status_code=403, detail=f"missing scope: {scope}")
