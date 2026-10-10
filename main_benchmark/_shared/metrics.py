from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .io import fact_text, iter_jsonl

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


class FrozenHashEncoder:
    """The release-compatible hashing-token-IDF encoder used by BS/Cov."""

    def __init__(self, facts_path: Path, dim: int = 4096, max_tokens: int = 32) -> None:
        self.dim, self.max_tokens = dim, max_tokens
        self.n = 0
        self.df: Counter[str] = Counter()
        for fact in iter_jsonl(facts_path):
            self.n += 1
            self.df.update(set(tokenize(fact_text(fact))))

    def embed(self, text: str) -> dict[int, float]:
        vector: dict[int, float] = {}
        for token in tokenize(text)[:self.max_tokens]:
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % self.dim
            value = math.log((self.n + 1.0) / (self.df.get(token, 0) + 1.0)) + 1.0
            vector[index] = vector.get(index, 0.0) + value
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {key: value / norm for key, value in vector.items()} if norm else {}

    @staticmethod
    def cosine(left: dict[int, float], right: dict[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(key, 0.0) for key, value in left.items())


def _hungarian_max(cost: list[list[float]]) -> tuple[float, list[int]]:
    nr, nc = len(cost), len(cost[0]) if cost else 0
    if not nr or not nc:
        return 0.0, []
    size = max(nr, nc)
    matrix = [[-cost[i][j] if i < nr and j < nc else 0.0
               for j in range(size)] for i in range(size)]
    u, v, p, way = [0.0] * (size + 1), [0.0] * (size + 1), [0] * (size + 1), [0] * (size + 1)
    for i in range(1, size + 1):
        p[0], j0 = i, 0
        minv, used = [float("inf")] * (size + 1), [False] * (size + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment = [-1] * size
    for j in range(1, size + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    total = sum(cost[i][j] for i, j in enumerate(assignment[:nr]) if 0 <= j < nc)
    return total, assignment[:nr]


def score_question(ranking: list[int], label: dict[str, Any], k: int,
                   facts: list[dict[str, Any]], encoder: FrozenHashEncoder,
                   tau: float) -> dict[str, float]:
    anchors = [int(x) for x in label["anchors"]]
    valid = set(anchors) | {int(x) for x in label["substitutes"]}
    retrieved = list(dict.fromkeys(int(x) for x in ranking))[:k]
    hits = float(any(index in valid for index in retrieved))
    anchor_embeddings = [encoder.embed(fact_text(facts[index])) for index in anchors]
    retrieved_embeddings = [encoder.embed(fact_text(facts[index])) for index in retrieved]
    cost = []
    for left in anchor_embeddings:
        row = []
        for right in retrieved_embeddings:
            cosine = encoder.cosine(left, right)
            row.append(cosine if cosine > tau else 0.0)
        cost.append(row)
    similarity, assignment = _hungarian_max(cost)
    matched = sum(1 for i, j in enumerate(assignment)
                  if 0 <= j < len(retrieved) and cost[i][j] > 0.0)
    denominator = len(anchors)
    return {
        "hits": hits,
        "bs": similarity / denominator if denominator else 0.0,
        "coverage": matched / denominator if denominator else 0.0,
        "nbs": similarity / min(k, denominator) if denominator else 0.0,
    }


def evaluate(predictions: list[dict[str, Any]], labels: list[dict[str, Any]],
             facts: list[dict[str, Any]], encoder: FrozenHashEncoder,
             tau: float, k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred = {x["question_id"]: x["ranked_fact_indices"] for x in predictions}
    gold = {x["question_id"]: x for x in labels}
    if set(pred) != set(gold):
        raise ValueError(f"prediction/gold ID mismatch: pred={len(pred)} gold={len(gold)}")
    per_question = []
    for question_id in sorted(gold):
        scores = score_question(pred[question_id], gold[question_id], k, facts, encoder, tau)
        per_question.append({"question_id": question_id, **scores})
    n = len(per_question)
    summary = {"k": k, "questions": n}
    for name in ("hits", "bs", "coverage", "nbs"):
        summary[name] = sum(row[name] for row in per_question) / n if n else 0.0
    return summary, per_question
