"""Shared state for the supervisor-worker graph."""

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    question: str            # what the user asked
    query: str               # current retrieval query (rewritten on retries)
    sub_questions: list      # what the supervisor decided to look for
    chunks: list             # retrieved + graded chunks
    rejected: list           # chunks the grader dropped (kept for the trace)
    answer: str              # draft answer from the synthesizer
    citations: list          # verified citations
    loops: int               # retrieval retries used
    next_action: str         # "retrieve" | "write"
    trace: list              # agent-by-agent audit trail
    meta: dict               # timings, cache hit, backend names


def log(state: ResearchState, agent: str, message: str) -> ResearchState:
    state.setdefault("trace", []).append({"agent": agent, "message": message})
    return state
