"""A2A agent cards and JSON-RPC dispatch."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from api.app import app
from src.a2a import agent_cards
from src.a2a.server import handle

client = TestClient(app)


def test_every_agent_has_a_card():
    cards = agent_cards.AGENT_CARDS
    assert set(cards) == {"querymesh", "supervisor", "retriever", "grader",
                          "synthesizer", "citation"}
    for card in cards.values():
        assert card["skills"] and card["url"].endswith("/a2a")
        assert card["protocolVersion"]


def test_well_known_card_is_served():
    card = client.get("/.well-known/agent.json").json()
    assert card["name"] == "QueryMesh orchestrator"


def test_get_card_via_jsonrpc():
    response = handle({"jsonrpc": "2.0", "id": 1, "method": "agent/getCard",
                       "params": {"agent": "grader"}})
    assert response["result"]["skills"][0]["id"] == "grade"


def test_message_send_to_orchestrator():
    response = handle({
        "jsonrpc": "2.0", "id": 2, "method": "message/send",
        "params": {"agent": "orchestrator", "message": {"parts": [
            {"kind": "text",
             "text": "What did the graph RAG paper find about global questions?"}]}},
    })
    task = response["result"]
    assert task["status"]["state"] == "completed"
    assert task["artifacts"][0]["parts"][0]["text"]


def test_each_sub_agent_answers():
    for agent in ("supervisor", "retriever", "grader", "synthesizer", "citation"):
        response = handle({
            "jsonrpc": "2.0", "id": 3, "method": "message/send",
            "params": {"agent": agent, "message": {"parts": [
                {"kind": "text", "text": "How does Self-RAG decide when to retrieve?"}]}},
        })
        assert "result" in response, agent
        assert response["result"]["artifacts"][0]["parts"][0]["text"], agent


def test_unknown_agent_and_method():
    unknown_agent = handle({"jsonrpc": "2.0", "id": 4, "method": "message/send",
                            "params": {"agent": "nope", "message": {"parts": []}}})
    assert unknown_agent["error"]["code"] == -32602

    unknown_method = handle({"jsonrpc": "2.0", "id": 5, "method": "nope"})
    assert unknown_method["error"]["code"] == -32601


def test_http_endpoint_accepts_jsonrpc():
    response = client.post("/a2a", json={
        "jsonrpc": "2.0", "id": 6, "method": "agent/getCard",
        "params": {"agent": "retriever"},
    })
    assert response.status_code == 200
    assert response.json()["result"]["name"] == "Retriever agent"
