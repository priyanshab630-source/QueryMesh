"""Telegram interface (long polling, no framework).

    export TELEGRAM_BOT_TOKEN=...
    python ui/telegram_bot.py

Answers are chunked to Telegram's 4096-character limit and HTML-escaped. The
same `answer()` used by the CLI and the API handles the actual work.
"""

import html
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

API_ROOT = "https://api.telegram.org/bot{token}/{method}"
MAX_LENGTH = 4000


def _token(token: str = None) -> str:
    token = token or config.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    return token


def send_message(chat_id, text: str, token: str = None) -> dict:
    token = _token(token)
    response = requests.post(
        API_ROOT.format(token=token, method="sendMessage"),
        json={"chat_id": chat_id, "text": text[:MAX_LENGTH],
              "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=30,
    )
    return response.json() if response.content else {}


def format_reply(result: dict) -> str:
    """Answer + a compact source list, escaped for Telegram HTML."""
    body = html.escape(result["answer"])
    if result["citations"]:
        lines = ["\n\n<b>Sources</b>"]
        for citation in result["citations"]:
            lines.append(
                f'{citation["number"]}. <a href="{citation["url"]}">'
                f'{html.escape(citation["title"][:80])}</a> '
                f'(arXiv {citation["paper_id"]})'
            )
        body += "\n".join(lines)
    return body[:MAX_LENGTH]


def handle_update(update: dict) -> str:
    """Process one Telegram update; returns the reply text that was sent."""
    message = update.get("message") or update.get("edited_message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = (message.get("text") or "").strip()
    if not chat_id or not text:
        return ""

    from main import answer

    if text.startswith("/start"):
        reply = ("QueryMesh: ask me about the indexed arXiv papers. "
                 "e.g. \"How does Self-RAG decide when retrieval is needed?\"")
    else:
        reply = format_reply(answer(text))

    send_message(chat_id, reply)
    return reply


def poll_forever(token: str = None, timeout: int = 30) -> None:
    token = _token(token)
    offset = None
    print("QueryMesh Telegram bot polling…")
    while True:
        params = {"timeout": timeout}
        if offset:
            params["offset"] = offset
        response = requests.get(
            API_ROOT.format(token=token, method="getUpdates"),
            params=params, timeout=timeout + 10,
        )
        for update in response.json().get("result", []):
            offset = update["update_id"] + 1
            try:
                handle_update(update)
            except Exception as exc:              # noqa: BLE001 - keep polling
                print(f"[telegram] update failed ({type(exc).__name__}): {exc}")
        time.sleep(0.5)


if __name__ == "__main__":
    poll_forever()
