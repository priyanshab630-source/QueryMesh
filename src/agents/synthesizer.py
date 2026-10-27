"""Agent 4 - Synthesizer.

Writes the answer from the graded chunks only. Every sentence that makes a
claim carries a [n] marker pointing at the chunk it came from; the Citation
agent then checks those markers, so the writer cannot quietly invent support.
"""

import config
from src.core.state import log
from src.search.bm25 import tokenize
from src.tools import llm

SYSTEM = (
    "You are a research assistant. Answer ONLY from the numbered context. "
    "Cite every claim with its bracketed number, e.g. [1]. If the context does "
    "not answer the question, say so plainly instead of guessing."
)


def _context_block(chunks: list) -> str:
    lines = []
    for number, hit in enumerate(chunks, start=1):
        chunk = hit["chunk"]
        lines.append(f"[{number}] {chunk['title']} (arXiv {chunk['paper_id']}, "
                     f"{chunk['published']}): {chunk['text']}")
    return "\n\n".join(lines)


def _extractive_answer(question: str, chunks: list) -> str:
    """No-LLM fallback: quote the most question-relevant sentences of one chunk.

    Quoting a single chunk keeps the answer coherent; every sentence carries the
    same [n] marker, which the Citation agent then verifies.
    """
    terms = set(tokenize(question))
    best_number, best_sentences = None, []

    for number, hit in enumerate(chunks[:4], start=1):
        text = hit["chunk"]["text"]
        sentences = [s.strip() for s in text.split(". ") if len(s) > 40] or [text.strip()]
        scored = sorted(sentences, key=lambda s: len(terms & set(tokenize(s))), reverse=True)
        selected = [s for s in scored[:2] if terms & set(tokenize(s))]
        if len(selected) > len(best_sentences):
            best_number, best_sentences = number, selected

    if not best_number:
        chunk = chunks[0]["chunk"]
        return f"{chunk['text'].strip()} [1]"

    body = " ".join(s.rstrip(".") + "." for s in best_sentences)
    return f"{body} [{best_number}]"


def run(state):
    question = state["question"]
    chunks = state.get("chunks", [])

    if not chunks:
        state["answer"] = ("I could not find relevant material in the corpus for this "
                           "question. Try a broader query or ingest more papers.")
        state["citations"] = []
        return log(state, "synthesizer", "no chunks - returned a refusal")

    coverage = max((hit.get("coverage", 0.0) for hit in chunks), default=0.0)
    if coverage < config.MIN_COVERAGE:
        titles = "; ".join(hit["chunk"]["title"] for hit in chunks[:2])
        state["answer"] = (
            "The corpus does not directly answer this question - only "
            f"{coverage:.0%} of its key terms appear in the retrieved passages. "
            f"Closest material found: {titles}. Ingest more papers on this topic "
            "before relying on an answer."
        )
        state["citations"] = []
        state["meta"]["writer"] = "coverage-gate"
        return log(state, "synthesizer",
                   f"coverage {coverage:.2f} < {config.MIN_COVERAGE:.2f} -> refused "
                   "instead of guessing")

    answer = llm.chat(SYSTEM, f"Question: {question}\n\nContext:\n{_context_block(chunks)}")
    mode = "llm"
    if not answer:
        answer = _extractive_answer(question, chunks)
        mode = "extractive-fallback"

    state["answer"] = answer
    state["meta"]["writer"] = mode
    return log(state, "synthesizer",
               f"drafted {len(answer.split())} words ({mode}) from {len(chunks)} chunk(s)")
