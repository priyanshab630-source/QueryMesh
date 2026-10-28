"""OpenSearch BM25 backend, exercised through the real client's transport layer.

No cluster is required: `perform_request` is replaced with a recorder, so the
test asserts the index mapping, the bulk payload and the query DSL that would be
sent, and then feeds a realistic response back through the parser.
"""

import json

import pytest

pytest.importorskip("opensearchpy", reason="opensearch-py is an optional extra")

from opensearchpy import OpenSearch          # noqa: E402

from src.search import opensearch_backend as osb  # noqa: E402

CHUNKS = [
    {"chunk_id": "2404.16130#0", "paper_id": "2404.16130",
     "title": "From Local to Global", "published": "2024-04-24",
     "url": "https://arxiv.org/abs/2404.16130",
     "text": "Chunk-based retrieval fails on global questions."},
    {"chunk_id": "2310.11511#0", "paper_id": "2310.11511",
     "title": "Self-RAG", "published": "2023-10-17",
     "url": "https://arxiv.org/abs/2310.11511",
     "text": "Reflection tokens mark whether a passage is relevant."},
]


class Recorder:
    """Captures every request the client would send."""

    def __init__(self, responses):
        self.calls = []
        self.responses = responses

    def __call__(self, method, url, params=None, body=None, **kwargs):
        self.calls.append({"method": method, "url": url, "params": params, "body": body})
        return self.responses.pop(0) if self.responses else {}


@pytest.fixture
def client_with(monkeypatch):
    def _make(responses):
        client = OpenSearch(hosts=["http://localhost:9200"])
        recorder = Recorder(responses)
        monkeypatch.setattr(client.transport, "perform_request", recorder)
        return client, recorder
    return _make


def _body(payload):
    """opensearch-py serialises bodies; parse them back for assertions."""
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


def test_ensure_index_creates_when_missing(client_with):
    # first call is HEAD /index (exists -> False), second is PUT /index
    client, recorder = client_with([{}, {"acknowledged": True}])
    index = osb.ensure_index(client, "test-index")
    assert index == "test-index"
    created = [c for c in recorder.calls if c["method"] == "PUT"]
    assert created
    mapping = _body(created[0]["body"])
    assert mapping["mappings"]["properties"]["text"]["type"] == "text"


def test_bulk_index_sends_one_action_per_chunk(client_with):
    client, recorder = client_with([{"errors": False, "items": [{"index": {}}, {"index": {}}]}])
    count = osb.bulk_index(client, CHUNKS, "test-index")

    assert count == 2
    raw = recorder.calls[0]["body"]
    lines = [json.loads(line) for line in raw.strip().splitlines()]
    assert lines[0] == {"index": {"_index": "test-index", "_id": "2404.16130#0"}}
    assert lines[1]["paper_id"] == "2404.16130"
    assert "global questions" in lines[1]["text"]


def test_bulk_index_raises_on_partial_failure(client_with):
    client, _ = client_with([{"errors": True,
                              "items": [{"index": {"error": "mapper_parsing"}}]}])
    with pytest.raises(RuntimeError):
        osb.bulk_index(client, CHUNKS[:1], "test-index")


def test_search_parses_hits(client_with):
    client, recorder = client_with([{
        "hits": {"hits": [
            {"_id": "2404.16130#0", "_score": 4.2},
            {"_id": "2310.11511#0", "_score": 2.1},
        ]}
    }])
    hits = osb.search(client, "global questions", 3, "test-index")

    assert hits == [("2404.16130#0", 4.2), ("2310.11511#0", 2.1)]
    sent = _body(recorder.calls[0]["body"])
    assert sent["query"]["multi_match"]["fields"] == ["title^2", "text"]
    assert sent["size"] == 3


def test_query_dsl_shape():
    dsl = osb.query_dsl("rag", 5)
    assert dsl["size"] == 5
    assert dsl["query"]["multi_match"]["query"] == "rag"


def test_backend_name_reflects_configuration(monkeypatch):
    import config

    monkeypatch.setattr(config, "SEARCH_BACKEND", "builtin")
    assert osb.backend_name() == "builtin-bm25"


def test_hybrid_index_uses_opensearch_when_configured(monkeypatch):
    """The index swaps the keyword ranker when a cluster is configured."""
    import config
    from src.search import index as index_module

    client, recorder = None, None

    def fake_client():
        nonlocal client, recorder
        client, recorder = OpenSearch(hosts=["http://localhost:9200"]), None
        recorder = Recorder([
            {},                                           # HEAD: index missing
            {"acknowledged": True},                       # create index
            {"errors": False, "items": [{"index": {}}] * 6},   # bulk
            {"hits": {"hits": [{"_id": "2404.16130#0", "_score": 9.0}]}},  # search
        ])
        monkeypatch.setattr(client.transport, "perform_request", recorder)
        return client

    monkeypatch.setattr(index_module.opensearch_backend, "client", fake_client)
    hybrid = index_module.HybridIndex(index_module.load_corpus())

    assert hybrid.os_client is not None
    hits = hybrid.search("global questions", k=2)
    assert hits and hits[0]["chunk"]["paper_id"] == "2404.16130"
    assert hits[0]["bm25_rank"] == 1


def test_live_cluster_roundtrip():
    """Runs only against a real cluster: OPENSEARCH_URL=http://localhost:9200 pytest -k live."""
    import config

    if not config.OPENSEARCH_URL:
        pytest.skip("set OPENSEARCH_URL to run the live OpenSearch test")

    client = osb.client()
    index = osb.ensure_index(client)
    osb.bulk_index(client, CHUNKS, index)
    hits = osb.search(client, "global questions", 2, index)
    assert hits and hits[0][1] > 0
