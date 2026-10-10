"""Dependency-free BM25 over fact_text for candidate recall and scoring.

Construction-time similarity is used ONLY for candidate recall and ranking
(never as a gold label): all role labels (anchor / substitute / distractor)
come from the deterministic executors (construction document section 2.6).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from .common import iter_jsonl

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class Bm25Stats:
    """Document frequencies + average length over the frozen fact corpus."""

    def __init__(self, facts_path, max_docs: int | None = None) -> None:
        self.doc_count = 0
        self.total_len = 0
        self.df: Counter[str] = Counter()
        for fact in iter_jsonl(facts_path):
            if max_docs is not None and self.doc_count >= max_docs:
                break
            tokens = tokenize(fact.get("fact_text", ""))
            self.doc_count += 1
            self.total_len += len(tokens)
            for token in set(tokens):
                self.df[token] += 1

    @property
    def avg_len(self) -> float:
        return self.total_len / self.doc_count if self.doc_count else 0.0

    def score(self, query_text: str, doc_text: str, k1: float = 1.5, b: float = 0.75) -> float:
        query = tokenize(query_text)
        doc = tokenize(doc_text)
        if not query or not doc:
            return 0.0
        tf: Counter[str] = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in set(query):
            if term not in self.df:
                continue
            n = self.df[term]
            idf = math.log(1.0 + (self.doc_count - n + 0.5) / (n + 0.5))
            f = tf.get(term, 0)
            score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / max(1.0, self.avg_len)))
        return score

    def score_cached(self, query_text: str, doc_tokens: list, doc_tf: Counter,
                     k1: float = 1.5, b: float = 0.75) -> float:
        """Score against pre-tokenized document data (avoids re-tokenizing
        every candidate on every anchor)."""
        query = tokenize(query_text)
        if not query or not doc_tokens:
            return 0.0
        dl = len(doc_tokens)
        score = 0.0
        for term in set(query):
            if term not in self.df:
                continue
            n = self.df[term]
            idf = math.log(1.0 + (self.doc_count - n + 0.5) / (n + 0.5))
            f = doc_tf.get(term, 0)
            if f:
                score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / max(1.0, self.avg_len)))
        return score


def lexical_jaccard(text_a: str, text_b: str) -> float:
    a, b = set(tokenize(text_a)), set(tokenize(text_b))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rank_by_bm25(stats: Bm25Stats, query_text: str, candidates: Iterable[tuple], top_n: int):
    """candidates: iterable of (key, text); returns top_n sorted (score desc, key asc)."""
    scored = [(stats.score(query_text, text), key) for key, text in candidates]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[:top_n]
