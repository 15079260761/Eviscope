"""Frozen text encoder for S_sem, HDM neighborhoods and BS@K (spec 7/12/14/15/17).

`hashing_token_idf_v1` revision 1.0.0: word tokens of canonical_fact_text are
hashed to one dimension of a fixed-size space, weighted by corpus IDF,
stored as a sparse vector and L2-normalized.  Deterministic across runs and
machines (sha256-based hashing, no randomness, no downloads).  The corpus IDF
statistics are frozen together with the facts file hash and recorded in the
resolved construction config (section 14).

The module also provides an exact cosine-neighborhood index over the frozen
fact space — required verbatim by HDM (section 15.2) — with a numpy fast path
and a pure-Python fallback.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import iter_jsonl, sha256_text
from .bm25 import tokenize

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def _hash_dim(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dim


class FrozenEncoder:
    """hashing_token_idf_v1: sparse IDF-weighted token-hash embeddings."""

    name = "hashing_token_idf_v1"
    base_revision = "1.0.0"

    def __init__(self, config: dict[str, Any], facts_path: Path) -> None:
        enc = config.get("encoder", {})
        self.dim = int(enc.get("dim", 4096))
        self.max_tokens = int(enc.get("max_tokens_per_fact", 32))
        self.facts_path = facts_path
        # corpus IDF statistics (frozen with the facts file)
        self.doc_count = 0
        self.df: Counter[str] = Counter()
        for fact in iter_jsonl(facts_path):
            text = fact.get("canonical_fact_text") or fact.get("fact_text", "")
            self.doc_count += 1
            for token in set(tokenize(text)):
                self.df[token] += 1
        self.corpus_sha256 = sha256_text(f"{self.doc_count}:{len(self.df)}")
        self.revision = f"{self.base_revision}:dim={self.dim}:corpus={self.corpus_sha256[:16]}"

    def _idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        return math.log((self.doc_count + 1.0) / (df + 1.0)) + 1.0

    def embed(self, text: str) -> dict[int, float]:
        """Sparse, L2-normalized embedding {dim: value}."""
        vector: dict[int, float] = {}
        for token in tokenize(text)[: self.max_tokens]:
            dim = _hash_dim(token, self.dim)
            vector[dim] = vector.get(dim, 0.0) + self._idf(token)
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm == 0.0:
            return {}
        return {d: v / norm for d, v in vector.items()}

    def cosine(self, emb_a: dict[int, float], emb_b: dict[int, float]) -> float:
        if not emb_a or not emb_b:
            return 0.0
        if len(emb_a) > len(emb_b):
            emb_a, emb_b = emb_b, emb_a
        return sum(v * emb_b.get(d, 0.0) for d, v in emb_a.items())


class NeighborIndex:
    """Exact cosine-neighborhood index over the frozen fact space.

    For a query embedding, scores[fact] = dot(query, fact) is exact (both
    sides L2-normalized).  Postings are stored per dimension as
    position/value arrays; with numpy the accumulation is fully vectorized,
    with a pure-Python fallback."""

    def __init__(self, encoder: FrozenEncoder, facts_path: Path,
                 fact_ids: Iterable[str], facts: dict[str, dict[str, Any]]) -> None:
        self.encoder = encoder
        self.fact_ids = list(fact_ids)
        self._pos = {fid: i for i, fid in enumerate(self.fact_ids)}
        dim_ids: dict[int, list[int]] = {}
        dim_vals: dict[int, list[float]] = {}
        for fid in self.fact_ids:
            fact = facts[fid]
            text = fact.get("canonical_fact_text") or fact.get("fact_text", "")
            emb = encoder.embed(text)
            for d, v in emb.items():
                dim_ids.setdefault(d, []).append(self._pos[fid])
                dim_vals.setdefault(d, []).append(v)
        self._np = None
        try:
            import numpy  # type: ignore
            self._np = numpy
        except ImportError:
            self._np = None
        if self._np is not None:
            import numpy as np
            self.dim_pos: dict[int, Any] = {
                d: np.asarray(pos, dtype=np.int64) for d, pos in dim_ids.items()}
            self.dim_vals: dict[int, Any] = {
                d: np.asarray(vals, dtype=np.float32) for d, vals in dim_vals.items()}
        else:
            self.dim_pos = dim_ids
            self.dim_vals = dim_vals
        self.n_facts = len(self.fact_ids)

    def _scores_array(self, anchor_emb: dict[int, float]):
        """Exact corpus-wide dot products for one anchor (numpy path)."""
        import numpy as np
        scores = np.zeros(self.n_facts, dtype=np.float32)
        for d, w in anchor_emb.items():
            pos = self.dim_pos.get(d)
            if pos is None:
                continue
            scores[pos] += w * self.dim_vals[d]
        return scores

    def _scores_dict(self, anchor_emb: dict[int, float]) -> dict[int, float]:
        """Exact corpus-wide dot products for one anchor (stdlib path)."""
        scores: dict[int, float] = {}
        for d, w in anchor_emb.items():
            for pos, v in zip(self.dim_pos.get(d, []), self.dim_vals.get(d, [])):
                scores[pos] = scores.get(pos, 0.0) + w * v
        return scores

    def neighborhood_size(self, anchor_emb: dict[int, float], tau: float) -> int:
        if not anchor_emb:
            return 0
        if self._np is not None:
            return int((self._scores_array(anchor_emb) >= tau).sum())
        return sum(1 for s in self._scores_dict(anchor_emb).values() if s >= tau)

    def top_neighbor_cosine(self, anchor_emb: dict[int, float],
                            self_fid: str) -> float:
        """Cosine to the nearest NON-SELF fact (for train fitting of tau_amb)."""
        self_pos = self._pos.get(self_fid)
        if self._np is not None:
            scores = self._scores_array(anchor_emb)
            if self_pos is not None:
                scores[self_pos] = -1.0
            if scores.size:
                return float(scores.max())
            return 0.0
        best = -1.0
        for pos, s in self._scores_dict(anchor_emb).items():
            if pos != self_pos and s > best:
                best = s
        return best if best >= 0 else 0.0
