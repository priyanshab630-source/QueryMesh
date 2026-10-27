"""Agent 3 - Grader.

Scores each retrieved chunk against the question and decides whether the
evidence is good enough to answer. If not, it tells the supervisor which terms
were missing so the retry is targeted rather than a blind re-query.
"""

import config
from src.core.state import log
from src.search.bm25 import tokenize


def _coverage(question_terms: set, text: str) -> float:
    if not question_terms:
        return 0.0
    present = question_terms & set(tokenize(text))
    return len(present) / len(question_terms)


def run(state):
    question_terms = set(tokenize(state["question"]))
    chunks = state.get("chunks", [])
    if not chunks:
        state["next_action"] = "write"
        state["meta"]["missing_terms"] = sorted(question_terms)[:5]
        return log(state, "grader", "nothing retrieved - handing over to the writer")

    top_score = max(hit["score"] for hit in chunks) or 1.0
    kept, rejected = [], []
    for hit in chunks:
        text = hit["chunk"]["text"]
        coverage = _coverage(question_terms, text)
        normalised = hit["score"] / top_score
        relevance = round(0.65 * coverage + 0.35 * normalised, 4)
        hit["relevance"] = relevance
        hit["coverage"] = round(coverage, 4)
        if relevance >= config.RELEVANCE_THRESHOLD:
            kept.append(hit)
        else:
            rejected.append(hit)

    kept.sort(key=lambda h: h["relevance"], reverse=True)
    state["chunks"] = kept
    state["rejected"] = rejected

    best = kept[0]["relevance"] if kept else 0.0
    loops = state.get("loops", 0)

    if best < config.RELEVANCE_THRESHOLD and loops < config.MAX_RETRIEVAL_LOOPS:
        covered = set()
        for hit in chunks[:3]:
            covered |= set(tokenize(hit["chunk"]["text"]))
        state["meta"]["missing_terms"] = sorted(question_terms - covered)[:5]
        state["loops"] = loops + 1
        state["next_action"] = "retrieve"
        return log(state, "grader",
                   f"best relevance {best:.2f} < {config.RELEVANCE_THRESHOLD:.2f} "
                   f"-> retry {loops + 1}/{config.MAX_RETRIEVAL_LOOPS}")

    state["next_action"] = "write"
    return log(state, "grader",
               f"kept {len(kept)} chunk(s), rejected {len(rejected)}; "
               f"best relevance {best:.2f}")
