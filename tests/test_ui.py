"""Gradio and Telegram interfaces (thin wrappers over answer())."""

import pytest


def test_gradio_answer_returns_three_parts():
    pytest.importorskip("gradio", reason="gradio is an optional extra")
    from ui.gradio_app import answer_question

    answer, sources, trace = answer_question(
        "What did the graph RAG paper find about global questions?")
    assert answer and "grounding" in sources
    assert "supervisor" in trace


def test_gradio_handles_empty_question():
    pytest.importorskip("gradio", reason="gradio is an optional extra")
    from ui.gradio_app import answer_question

    assert answer_question("")[0].startswith("Ask a question")


def test_gradio_demo_builds():
    pytest.importorskip("gradio", reason="gradio is an optional extra")
    from ui.gradio_app import build_demo

    assert build_demo() is not None


def test_telegram_reply_contains_sources():
    from ui.telegram_bot import format_reply

    text = format_reply({
        "answer": "Graph RAG summarises communities.",
        "citations": [{"number": 1, "title": "From Local to Global",
                       "paper_id": "2404.16130",
                       "url": "https://arxiv.org/abs/2404.16130"}],
    })
    assert "Sources" in text and "2404.16130" in text


def test_telegram_send_message_posts(monkeypatch):
    import ui.telegram_bot as bot

    captured = {}

    class FakeResponse:
        content = b"{}"

        def json(self):
            return {"ok": True}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(bot.requests, "post", fake_post)
    bot.send_message(12345, "hello", token="test-token")

    assert captured["url"].endswith("/sendMessage")
    assert captured["json"]["chat_id"] == 12345
    assert "test-token" in captured["url"]
