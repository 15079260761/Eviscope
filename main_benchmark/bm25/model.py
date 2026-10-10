from __future__ import annotations

import math
from collections import Counter

from _shared.metrics import tokenize


class BM25Retriever:
    """Whole-catalog-IDF BM25; no labels and no training."""

    def __init__(self, facts: list[dict], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tokens = [tokenize(x.get("canonical_fact_text") or x.get("fact_text", "")) for x in facts]
        self.lengths = [len(x) for x in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter(token for doc in self.tokens for token in set(doc))
        n = len(self.tokens)
        self.idf = {token: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
                    for token, count in df.items()}

    def score(self, question: str, candidates: list[int]) -> dict[int, float]:
        query = Counter(tokenize(question))
        scores = {}
        for index in candidates:
            counts = Counter(self.tokens[index])
            norm = self.k1 * (1.0 - self.b + self.b * self.lengths[index] / max(self.avgdl, 1e-9))
            scores[index] = sum(
                qtf * self.idf.get(token, 0.0) * counts[token] * (self.k1 + 1.0) /
                (counts[token] + norm)
                for token, qtf in query.items() if counts[token]
            )
        return scores
