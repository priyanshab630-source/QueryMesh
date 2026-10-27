"""Keyword retriever: tokenisation, ranking, empty-query safety."""

from src.search.bm25 import BM25, tokenize


def test_tokenize_drops_stopwords():
    assert tokenize("The retriever is used for the paper") == ["retriever", "paper"]


def test_ranks_matching_document_first():
    docs = ["graph retrieval augmented generation",
            "protein docking and molecular dynamics",
            "citation verification for generated answers"]
    index = BM25(docs)
    top = index.top_k("retrieval augmented", k=2)
    assert top and top[0][0] == 0


def test_unseen_terms_return_nothing():
    index = BM25(["alpha beta", "gamma delta"])
    assert index.top_k("zzz", k=3) == []


def test_scores_are_finite():
    index = BM25(["a b c", "b c d"])
    assert all(isinstance(s, float) for s in index.scores("b"))
