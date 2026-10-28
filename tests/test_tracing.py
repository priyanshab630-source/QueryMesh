"""Local JSONL traces and the optional remote sinks."""

import json

import config
from src.core import tracing
from src.core.tracing import Tracer


def test_trace_is_written_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(config, "TRACE_ENABLED", True)

    tracer = Tracer("qm.test", {"question": "why"})
    tracer.event("supervisor", {"message": "decomposed"})
    record = tracer.end({"citations": 1})

    stored = json.loads((tmp_path / "runs.jsonl").read_text().strip())
    assert stored["run_id"] == record["run_id"]
    assert stored["events"][0]["name"] == "supervisor"
    assert stored["outputs"]["citations"] == 1


def test_sinks_without_keys(monkeypatch):
    monkeypatch.setattr(config, "LANGSMITH_API_KEY", "")
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "")
    assert tracing.sinks() == ["local-jsonl"]


def test_sinks_with_keys(monkeypatch):
    monkeypatch.setattr(config, "LANGSMITH_API_KEY", "k")
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "p")
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "s")
    assert tracing.sinks() == ["local-jsonl", "langsmith", "langfuse"]


def test_ask_writes_one_trace(tmp_path, monkeypatch):
    from src.agents.workflow import ask

    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    ask("How does Self-RAG decide when retrieval is needed?")

    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["name"] == "querymesh.ask"
