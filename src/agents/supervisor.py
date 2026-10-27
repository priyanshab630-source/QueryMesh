"""Agent 1 - Supervisor.

Two jobs, one node: decompose the question on the first pass, and rewrite the
query on a retry pass. Deciding *what to search for* is the supervisor's call;
the retriever never invents a query of its own.
"""

from src.core.state import log
from src.tools import llm

SYSTEM = (
    "You are the supervisor of a research assistant. Decompose the user's "
    "question into at most 3 focused search queries for an academic corpus. "
    "Return one query per line, no numbering, no commentary."
)

FILLER = {"what", "which", "how", "does", "do", "is", "are", "the", "a", "an",
          "of", "for", "in", "on", "to", "and", "with", "by", "that", "this"}


def _decompose_rule_based(question: str) -> list:
    """Split on conjunctions and drop filler words."""
    parts = [p.strip(" ?.") for p in question.replace(";", ",").split(",") if p.strip()]
    if len(parts) == 1:
        parts = [p.strip(" ?.") for p in question.split(" and ") if p.strip()]
    queries = []
    for part in parts:
        words = [w for w in part.split() if w.lower() not in FILLER]
        if words:
            queries.append(" ".join(words))
    return queries[:3] or [question]


def run(state):
    question = state["question"]
    loops = state.get("loops", 0)

    if loops > 0:
        # Retry pass: widen the query with the terms the grader found missing.
        missing = state.get("meta", {}).get("missing_terms", [])
        base = state.get("query") or question
        query = " ".join([base] + missing[:4]).strip()
        state["query"] = query
        state["next_action"] = "retrieve"
        return log(state, "supervisor",
                   f"loop {loops}: rewrote query -> '{query[:70]}'")

    sub_questions = _decompose_rule_based(question)
    text = llm.chat(SYSTEM, question)
    if text:
        lines = [line.strip("-*0123456789. ") for line in text.splitlines() if line.strip()]
        if lines:
            sub_questions = lines[:3]

    state["sub_questions"] = sub_questions
    state["query"] = " ".join(sub_questions)
    state["loops"] = 0
    state["next_action"] = "retrieve"
    state["meta"] = state.get("meta", {})
    return log(state, "supervisor",
               f"decomposed into {len(sub_questions)} sub-question(s): "
               f"{'; '.join(sub_questions)[:80]}")
