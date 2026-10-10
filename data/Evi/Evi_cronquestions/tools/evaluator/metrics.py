"""Structure-aware evaluation per spec section 17 (paper definitions
verbatim):

  M_ij = cos(e(f_i), e(r_j)) if cos(e(f_i), e(r_j)) > tau else 0
  Hits@K: the Top-K retrieved facts contain a fact that derives the correct
          answer (verified anchor or substitute — never mere answer-string
          presence in arbitrary positions)
  BS@K(q) = (1/|F*(q)|) * max_pi sum_(i,j) M_ij   (max-weight one-to-one)
  Coverage@K(q) = #anchors matched / |F*(q)|

NO rank discount, no double credit (section 17.3/21).
"""

from __future__ import annotations

import math
from typing import Any

from src.common import iter_jsonl
from src.encoder import FrozenEncoder


def _dp_max_matching(n_groups: int, weights: list[tuple[int, float]]) -> float:
    states = {0: 0.0}
    for group_index, weight in weights:
        bit = 1 << group_index
        next_states = dict(states)
        for mask, score in states.items():
            if not mask & bit:
                candidate = score + weight
                new_mask = mask | bit
                if candidate > next_states.get(new_mask, -1.0):
                    next_states[new_mask] = candidate
        states = next_states
    return max(states.values(), default=0.0)


def _hungarian_max_matching_with_assignment(
        cost: list[list[float]]) -> tuple[float, list[int]]:
    n_rows = len(cost)
    n_cols = len(cost[0]) if cost else 0
    if n_rows == 0 or n_cols == 0:
        return 0.0, []
    size = max(n_rows, n_cols)
    matrix = [[-cost[i][j] if i < n_rows and j < n_cols else 0.0
               for j in range(size)] for i in range(size)]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = -1
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
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
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    total = 0.0
    for i in range(n_rows):
        j = assignment[i]
        if j is not None and 0 <= j < n_cols:
            total += cost[i][j]
    return total, assignment[:n_rows]


def _hungarian_max_matching(cost: list[list[float]]) -> float:
    total, _ = _hungarian_max_matching_with_assignment(cost)
    return total


def _fact(facts: dict[Any, dict[str, Any]], reference: Any) -> dict[str, Any] | None:
    return facts.get(reference) or facts.get(str(reference))


def _fact_id(facts: dict[Any, dict[str, Any]], reference: Any) -> str | None:
    row = _fact(facts, reference)
    return str(row["fact_id"]) if row else None


def score_question(question: dict[str, Any], ranked_facts: list[Any], k: int,
                   encoder: FrozenEncoder, tau: float,
                   facts: dict[Any, dict[str, Any]]) -> dict[str, float]:
    anchors = list(question.get("anchor_fact_indices",
                                question.get("anchor_fact_ids", [])))
    if "substitute_fact_indices" in question:
        substitutes = set(question.get("substitute_fact_indices", []))
    else:
        substitutes = {d["fact_id"] for d in question.get("substitutable_evidence", [])}
    retrieved = list(dict.fromkeys(ranked_facts))[:k]
    anchor_ids = {_fact_id(facts, ref) for ref in anchors}
    substitute_ids = {_fact_id(facts, ref) for ref in substitutes}
    retrieved_ids = [_fact_id(facts, ref) for ref in retrieved]
    # Hits@K: a retrieved fact that derives the correct answer
    hits = 1.0 if any(fid in anchor_ids | substitute_ids
                      for fid in retrieved_ids if fid is not None) else 0.0
    # M_ij with the frozen encoder (17.2)
    anchor_embs = []
    for fid in anchors:
        fact = _fact(facts, fid)
        text = fact.get("canonical_fact_text") or fact.get("fact_text", "") if fact else ""
        anchor_embs.append(encoder.embed(text))
    retrieved_embs = []
    for fid in retrieved:
        fact = _fact(facts, fid)
        text = fact.get("canonical_fact_text") or fact.get("fact_text", "") if fact else ""
        retrieved_embs.append(encoder.embed(text))
    cost_rows: list[list[float]] = []
    for i, emb_a in enumerate(anchor_embs):
        row = []
        for emb_r in retrieved_embs:
            cos = encoder.cosine(emb_a, emb_r)
            row.append(cos if cos > tau else 0.0)
        cost_rows.append(row)
    # exact max-weight bipartite matching: one-to-one on BOTH sides
    # (each retrieved fact matches at most one anchor and vice versa)
    best, assignment = _hungarian_max_matching_with_assignment(cost_rows)
    bs = best / len(anchors) if anchors else 0.0
    matched = {
        i for i, j in enumerate(assignment)
        if 0 <= j < len(retrieved_embs) and cost_rows[i][j] > 0.0
    }
    coverage = len(matched) / len(anchors) if anchors else 0.0
    return {"hits": hits, "coverage": coverage, "bs": bs}


def evaluate_all(questions, submissions: list[dict[str, Any]], k: int,
                 encoder: FrozenEncoder, tau: float,
                 facts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = {q["question_id"]: q for q in questions}
    per_question = []
    seen = set()
    for row in submissions:
        qid = row.get("question_id")
        if qid in seen:
            raise ValueError(f"Duplicate question_id in submission: {qid}")
        if qid not in labels:
            raise ValueError(f"Unknown question_id in submission: {qid}")
        seen.add(qid)
        ranking = row.get("ranked_fact_indices")
        ranking_field = "ranked_fact_indices"
        if ranking is None:
            ranking = row.get("ranked_fact_ids")
            ranking_field = "ranked_fact_ids"
        if not isinstance(ranking, list):
            raise ValueError(
                f"ranked_fact_indices or ranked_fact_ids must be a list for {qid}")
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"Duplicate fact reference in {ranking_field} for {qid}")
        scores = score_question(labels[qid], ranking, k, encoder, tau, facts)
        per_question.append({"question_id": qid, **scores})
    if seen != set(labels):
        raise ValueError(f"Submission is missing {len(set(labels) - seen)} questions")
    n = len(per_question)
    summary = {
        "k": k,
        "questions": n,
        "hits": sum(x["hits"] for x in per_question) / n if n else 0.0,
        "coverage": sum(x["coverage"] for x in per_question) / n if n else 0.0,
        "bs": sum(x["bs"] for x in per_question) / n if n else 0.0,
        "per_question": per_question,
    }
    return summary


def stratified_summary(questions, per_question: list[dict[str, Any]]) -> dict[str, Any]:
    by_q = {x["question_id"]: x for x in per_question}
    buckets: dict[str, list[dict[str, Any]]] = {}

    def key_of(q: dict[str, Any]) -> str:
        level = (q.get("hybrid_difficulty") or {}).get("difficulty_level", "unset")
        anchors = q.get("anchor_fact_indices", q.get("anchor_fact_ids", []))
        return f"type:{q.get('question_type')}|difficulty:{level}|anchors:{len(anchors)}"

    for q in questions:
        buckets.setdefault(key_of(q), []).append(by_q[q["question_id"]])
    out = {}
    for key, rows in sorted(buckets.items()):
        n = len(rows)
        out[key] = {
            "questions": n,
            "hits": sum(x["hits"] for x in rows) / n,
            "coverage": sum(x["coverage"] for x in rows) / n,
            "bs": sum(x["bs"] for x in rows) / n,
        }
    return out
