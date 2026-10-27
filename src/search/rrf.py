"""Reciprocal Rank Fusion - merges the keyword and dense rankings.

score(d) = sum over rankers of weight / (k + rank(d))

RRF is used because BM25 scores and cosine similarities live on completely
different scales; fusing *ranks* avoids inventing a normalisation.
"""

import config


def fuse(ranked_lists: list, k: int = None, weights: list = None) -> list:
    """ranked_lists: [[(doc_index, score), ...], ...] -> [(doc_index, rrf_score)]."""
    k = k or config.RRF_K
    weights = weights or [1.0] * len(ranked_lists)

    fused = {}
    for weight, ranking in zip(weights, ranked_lists):
        for rank, (doc_index, _score) in enumerate(ranking, start=1):
            fused[doc_index] = fused.get(doc_index, 0.0) + weight / (k + rank)

    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
