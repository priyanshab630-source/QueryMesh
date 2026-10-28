"""JWT issuing, verification and route protection."""

import time

import pytest

pytest.importorskip("fastapi")

import config
from api.auth import TokenError, create_token, verify_token
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_token_roundtrip():
    token = create_token("tester")
    claims = verify_token(token)
    assert claims["sub"] == "tester"
    assert "ask" in claims["scope"]


def test_tampered_token_is_rejected():
    token = create_token("tester")
    with pytest.raises(TokenError):
        verify_token(token[:-4] + "aaaa")


def test_expired_token_is_rejected(monkeypatch):
    token = create_token("tester", ttl=-5)
    with pytest.raises(TokenError):
        verify_token(token)


def test_token_endpoint_issues_a_jwt(monkeypatch):
    monkeypatch.setattr(config, "API_CLIENT_ID", "demo-client")
    monkeypatch.setattr(config, "API_CLIENT_SECRET", "demo-secret")

    ok = client.post("/token", json={"client_id": "demo-client",
                                     "client_secret": "demo-secret"})
    assert ok.status_code == 200
    assert verify_token(ok.json()["access_token"])["sub"] == "demo-client"

    bad = client.post("/token", json={"client_id": "demo-client",
                                      "client_secret": "wrong"})
    assert bad.status_code == 401


def test_routes_require_a_token_when_auth_is_on(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", True)

    assert client.post("/ask", json={"question": "anything"}).status_code == 401

    token = create_token("tester")
    response = client.post("/ask", json={"question":
                                         "What did the graph RAG paper find?"},
                           headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_scope_is_enforced(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    token = create_token("read-only", scopes=["ask"])
    response = client.post("/ingest", json={"query": "rag"},
                           headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_open_by_default():
    assert client.post("/ask", json={"question": "What is ReAct?"}).status_code == 200
