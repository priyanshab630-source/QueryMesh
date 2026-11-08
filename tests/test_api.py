"""API surface: health, papers, ask."""

import pytest

pytest.importorskip("fastapi")

try:                                            # starlette's TestClient needs httpx2
    from fastapi.testclient import TestClient
except Exception as exc:                        # noqa: BLE001
    pytest.skip(f"TestClient unavailable: {exc}", allow_module_level=True)

from api.app import app                         # noqa: E402

client = TestClient(app)


def test_static_dir_resolves_portably():
    """`__file__` uses backslashes on Windows - the static dir must still resolve."""
    from api import app as app_module

    static = app_module.STATIC_DIR
    assert hasattr(static, "is_dir"), "STATIC_DIR should be a pathlib.Path"
    assert static.is_dir(), f"{static} is not a directory"
    assert "app.py" not in str(static)
    assert (static / "index.html").is_file()


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["agents"] == 5


def test_papers():
    papers = client.get("/papers").json()
    assert isinstance(papers, list) and papers


def test_ask():
    response = client.post("/ask", json={"question":
                                         "What did the graph RAG paper find?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body and "citations" in body and "trace" in body
