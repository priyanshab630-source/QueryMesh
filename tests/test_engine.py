"""Both orchestration engines answer the same question the same way."""

import pytest

import config
from src.agents.workflow import ask
from src.core import engine

QUESTION = "What did the graph RAG paper find about global questions?"


def _run(use_langgraph: bool, monkeypatch):
    monkeypatch.setattr(config, "USE_LANGGRAPH", use_langgraph)
    return ask(QUESTION, trace=False)


def test_builtin_engine_is_reported(monkeypatch):
    assert _run(False, monkeypatch)["meta"]["engine"] == "builtin"


def test_langgraph_matches_builtin(monkeypatch):
    pytest.importorskip("langgraph", reason="langgraph is an optional extra")

    builtin = _run(False, monkeypatch)
    langgraph_run = _run(True, monkeypatch)

    assert langgraph_run["meta"]["engine"] == "langgraph"
    assert langgraph_run["answer"] == builtin["answer"]
    assert [c["paper_id"] for c in langgraph_run["citations"]] == \
           [c["paper_id"] for c in builtin["citations"]]


def test_langgraph_survives_the_retrieval_loop(monkeypatch):
    pytest.importorskip("langgraph", reason="langgraph is an optional extra")

    monkeypatch.setattr(config, "USE_LANGGRAPH", True)
    result = ask("What is the airspeed velocity of an unladen swallow?", trace=False)
    assert result["answer"]
    assert "does not directly answer" in result["answer"] or not result["citations"]


def test_engine_name():
    assert engine.engine_name()
