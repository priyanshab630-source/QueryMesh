"""Agent 2 - Retriever.

Runs the hybrid search (BM25 + dense vectors, fused with RRF) for each
sub-question and merges the results, keeping the best rank per chunk.
"""

from src.core.state import log
from src.search.index import get_index


def run(state):
    index = get_index()
    sub_questions = state.get("sub_questions") or [state.get("query", state["question"])]

    best = {}
    for query in sub_questions:
        for hit in index.search(query):
            chunk_id = hit["chunk"]["chunk_id"]
            previous = best.get(chunk_id)
            if previous is None or hit["score"] > previous["score"]:
                best[chunk_id] = hit

    ranked = sorted(best.values(), key=lambda h: h["score"], reverse=True)
    state["chunks"] = ranked
    state["meta"] = state.get("meta", {})
    state["meta"]["retrieval_backend"] = "hybrid: bm25+dense+rrf"
    return log(state, "retriever",
               f"{len(ranked)} unique chunk(s) from {len(sub_questions)} quer(ies)")
