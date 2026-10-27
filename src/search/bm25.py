"""Okapi BM25 in pure Python - the keyword half of the hybrid retriever."""

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = set("""
a an and are as at be been being between by can could do does for from had has
have how i if in into is it its of on or our than that the their them then there
these they this to was we were what when where which who why will with would you
your using use used based via et al
""".split())


def tokenize(text: str) -> list:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, documents: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in documents]
        self.lengths = [len(d) for d in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        self.frequencies = [Counter(d) for d in self.docs]
        document_frequency = Counter()
        for frequencies in self.frequencies:
            document_frequency.update(set(frequencies))
        total = len(self.docs) or 1
        self.idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> list:
        terms = tokenize(query)
        results = []
        for i, frequencies in enumerate(self.frequencies):
            score = 0.0
            length = self.lengths[i] or 1
            for term in terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_length or 1)
                )
                score += idf * tf * (self.k1 + 1) / denominator
            results.append(score)
        return results

    def top_k(self, query: str, k: int = 10) -> list:
        """[(doc_index, score)] sorted by score, zeros dropped."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
