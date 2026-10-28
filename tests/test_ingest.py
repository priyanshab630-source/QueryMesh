"""Local ingestion runner (the logic the Airflow DAG calls)."""

from dags import run_ingest


def test_ingest_topics_uses_the_client(monkeypatch):
    calls = []

    def fake_ingest(query, limit=5):
        calls.append((query, limit))
        return [{"id": "2401.00001"}]

    monkeypatch.setattr("src.tools.arxiv_client.ingest", fake_ingest)
    summary = run_ingest.ingest_topics(["graph rag"], limit=2)

    assert calls == [("graph rag", 2)]
    assert summary["ingested"] == 1
    assert summary["topics"] == {"graph rag": 1}
    assert summary["papers"] is not None


def test_ingest_with_no_results(monkeypatch):
    monkeypatch.setattr("src.tools.arxiv_client.ingest", lambda q, limit=5: [])
    summary = run_ingest.ingest_topics(["nothing"], limit=1)
    assert summary["ingested"] == 0
    assert summary["papers"] is None


def test_default_topics_exist():
    assert len(run_ingest.TOPICS) >= 3
