"""Agent 5 - Citation.

Checks every [n] marker in the draft against the chunk it points at. A marker
whose sentence is not actually supported by that chunk is flagged and removed,
so the final answer never carries decorative citations.
"""

import re

from src.core.state import log
from src.search.bm25 import tokenize

MARKER = re.compile(r"\[(\d+)\]")


def _supported(sentence: str, text: str, threshold: float = 0.34) -> bool:
    """Content-word overlap between the claim and the cited chunk."""
    claim = MARKER.sub(" ", sentence)          # the marker itself is not a word
    claim_terms = set(tokenize(claim)) - set(tokenize(text))
    if not claim_terms:
        return True                       # nothing beyond the chunk's own words
    cited_terms = set(tokenize(text))
    overlap = len(claim_terms & cited_terms) / len(claim_terms)
    return overlap >= threshold


def run(state):
    answer = state.get("answer", "")
    chunks = state.get("chunks", [])
    by_number = {number: hit for number, hit in enumerate(chunks, start=1)}

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    citations, removed, verified = [], 0, 0
    cleaned = []

    for sentence in sentences:
        numbers = [int(n) for n in MARKER.findall(sentence)]
        if not numbers:
            cleaned.append(sentence)
            continue

        good = []
        for number in numbers:
            hit = by_number.get(number)
            if hit is None:
                removed += 1
                continue
            if _supported(sentence, hit["chunk"]["text"]):
                good.append(number)
                verified += 1
            else:
                removed += 1

        if not good:
            # Keep the sentence but strip the unsupported markers.
            cleaned.append(MARKER.sub("", sentence).strip())
            continue

        for number in good:
            hit = by_number[number]
            entry = {
                "number": number,
                "paper_id": hit["chunk"]["paper_id"],
                "title": hit["chunk"]["title"],
                "url": hit["chunk"]["url"],
                "published": hit["chunk"]["published"],
            }
            if entry not in citations:
                citations.append(entry)
        cleaned.append(sentence)

    state["answer"] = " ".join(cleaned).strip()
    state["citations"] = citations
    total = verified + removed
    state["meta"]["grounding"] = round(verified / total, 3) if total else 0.0
    return log(state, "citation",
               f"{len(citations)} source(s) cited, {removed} unsupported marker(s) "
               f"removed, grounding {state['meta']['grounding']:.2f}")
