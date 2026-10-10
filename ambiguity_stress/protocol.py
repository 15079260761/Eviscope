from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "main_benchmark"
sys.path.insert(0, str(EXP1))

from _shared.config import POOL_SIZE, SEED, TEST_TARGET, TRAIN_TARGET, dataset_spec
from _shared.data import _load_runtime, _load_split, _stratified
from _shared.io import fact_text, iter_jsonl, qid, sha256, write_json, write_jsonl
from _shared.metrics import FrozenHashEncoder

LEVELS = ("low", "medium", "high")
METHODS = ("bm25", "dual_encoder", "cross_encoder", "eviret")
DISTRACTORS_PER_QUESTION = 5


def stable_key(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    """Linear empirical quantile (deterministic; equivalent to NumPy default)."""
    if not values:
        raise ValueError("cannot fit an ambiguity quantile from no observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def level_of(score: float, q33: float, q66: float) -> str:
    if score < q33:
        return "low"
    if score < q66:
        return "medium"
    return "high"


def source_label_path(dataset: str, split: str) -> Path:
    spec = dataset_spec(dataset)
    return spec.private_test if split == "test" else spec.supervision_dir / f"{split}.jsonl"


def load_source_labels(dataset: str, split: str, ids: set[str] | None = None) -> dict[str, dict]:
    rows = {}
    for row in iter_jsonl(source_label_path(dataset, split)):
        key = qid(row)
        if ids is None or key in ids:
            rows[key] = row
    if ids is not None and set(rows) != ids:
        raise ValueError(f"{dataset}/{split}: missing source labels for {len(ids - set(rows))} IDs")
    return rows


class SemanticSimilarity:
    """Read or exactly reconstruct the benchmark's frozen S_sem score."""

    def __init__(self, dataset: str, facts: list[dict]) -> None:
        self.dataset = dataset
        self.facts = facts
        self.encoder = FrozenHashEncoder(dataset_spec(dataset).facts) if dataset == "cron" else None
        self.cache: dict[int, dict[int, float]] = {}

    def embedding(self, index: int) -> dict[int, float]:
        if index not in self.cache:
            assert self.encoder is not None
            self.cache[index] = self.encoder.embed(fact_text(self.facts[index]))
        return self.cache[index]

    def scored_distractors(self, label: dict) -> list[tuple[int, float]]:
        if self.dataset == "icews":
            return [(int(item["fact_index"]), float(item["similarity_score"]))
                    for item in label.get("distracting_evidence", [])]
        anchors = list(dict.fromkeys(int(x) for x in label.get("anchor_fact_indices", [])))
        if not anchors:
            return []
        anchor_embeddings = [self.embedding(index) for index in anchors]
        result = []
        assert self.encoder is not None
        for raw in dict.fromkeys(int(x) for x in label.get("distractor_fact_indices", [])):
            candidate = self.embedding(raw)
            score = max(self.encoder.cosine(candidate, anchor) for anchor in anchor_embeddings)
            result.append((raw, score))
        return result


def fit_train_quantiles(dataset: str, facts: list[dict]) -> tuple[dict[str, Any], set[str]]:
    """Fit Q33/Q66 on the same deterministic 8k full-training cohort as Exp.1."""
    eligible = _load_split(dataset_spec(dataset), "train")
    cohort = _stratified(eligible, TRAIN_TARGET, SEED)
    ids = {row["question_id"] for row in cohort}
    labels = load_source_labels(dataset, "train", ids)
    scorer = SemanticSimilarity(dataset, facts)
    scores: list[float] = []
    for position, key in enumerate(sorted(ids), 1):
        scores.extend(score for _, score in scorer.scored_distractors(labels[key]))
        if position % 500 == 0 or position == len(ids):
            print(f"[quantiles:{dataset}] train questions {position}/{len(ids)} scores={len(scores)}", flush=True)
    q33, q66 = quantile(scores, 0.33), quantile(scores, 0.66)
    if not 0.0 <= q33 < q66 <= 1.0:
        raise ValueError(f"invalid fitted quantiles: q33={q33}, q66={q66}")
    return {
        "fit_split": "train", "fit_questions": len(ids), "fit_scores": len(scores),
        "q33": q33, "q66": q66,
        "intervals": {"low": [0.0, q33], "medium": [q33, q66], "high": [q66, 1.0]},
        "boundary_policy": "low < Q33; Q33 <= medium < Q66; high >= Q66",
        "score_source": ("published distracting_evidence.similarity_score" if dataset == "icews"
                         else "exact reconstruction with release-compatible frozen hashing-token-IDF encoder"),
    }, ids


def release_fact_index(dataset: str, runtime, spec):
    fact_index = runtime.ReleaseFactIndex(spec.facts)
    if dataset == "icews":
        descriptor = next(iter_jsonl(spec.descriptors_dir / "test.jsonl"))
        declared = descriptor["fact_catalog_sha256"]
        if fact_index.catalog_sha256 != declared:
            lineage = json.loads((spec.release_dir / "audit" / "source_lineage.json").read_text(encoding="utf-8"))
            semantic = json.loads((spec.release_dir / "audit" / "full_semantic_audit.json").read_text(encoding="utf-8"))
            if (lineage.get("compact_facts_sha256") != declared or semantic.get("status") != "pass"
                    or int(semantic.get("facts", -1)) != len(fact_index.ids)):
                raise ValueError("ICEWS fact catalog is not the attested release")
            fact_index.catalog_sha256 = declared
    return fact_index


def prepare_dataset(dataset: str, output: Path) -> dict[str, Any]:
    spec = dataset_spec(dataset)
    facts = list(iter_jsonl(spec.facts))
    thresholds, train_ids = fit_train_quantiles(dataset, facts)
    q33, q66 = thresholds["q33"], thresholds["q66"]
    test_population = _load_split(spec, "test")
    source = load_source_labels(dataset, "test", {row["question_id"] for row in test_population})
    scorer = SemanticSimilarity(dataset, facts)
    scored: dict[str, dict[str, list[tuple[int, float]]]] = {}
    eligible = []
    availability = {"at_least_15_total": 0, **{f"fewer_than_5_{level}": 0 for level in LEVELS}}
    for position, row in enumerate(test_population, 1):
        bins = {level: [] for level in LEVELS}
        for index, score in scorer.scored_distractors(source[row["question_id"]]):
            bins[level_of(score, q33, q66)].append((index, score))
        if sum(len(values) for values in bins.values()) >= 3 * DISTRACTORS_PER_QUESTION:
            availability["at_least_15_total"] += 1
        for level in LEVELS:
            if len(bins[level]) < DISTRACTORS_PER_QUESTION:
                availability[f"fewer_than_5_{level}"] += 1
        if all(len(bins[level]) >= DISTRACTORS_PER_QUESTION for level in LEVELS):
            scored[row["question_id"]] = bins
            eligible.append(row)
        if position % 500 == 0 or position == len(test_population):
            print(f"[eligibility:{dataset}] {position}/{len(test_population)} strict={len(eligible)}", flush=True)
    selected = _stratified(eligible, TEST_TARGET, SEED + 2)
    if not selected:
        raise ValueError(f"{dataset}: no test questions satisfy the strict 5/5/5 intervention")

    runtime = _load_runtime(spec)
    fact_index = release_fact_index(dataset, runtime, spec)
    descriptor_by_id = {qid(row): row for row in iter_jsonl(spec.descriptors_dir / "test.jsonl")}
    public, trusted = [], []
    pool_rows = {level: [] for level in LEVELS}
    assignment_rows = []
    for position, row in enumerate(selected, 1):
        key = row["question_id"]
        label, descriptor = source[key], descriptor_by_id[key]
        anchors = list(dict.fromkeys(int(x) for x in label["anchor_fact_indices"]))
        if dataset == "cron":
            clean = runtime.materialize_pool(descriptor, label, "clean", fact_index)
        else:
            clean = runtime.materialize_all(descriptor, label, fact_index)["clean_control"]
        anchor_set = set(anchors)
        neutral = [int(x) for x in clean["candidate_fact_indices"] if int(x) not in anchor_set]
        common_neutral_count = POOL_SIZE - len(anchors) - DISTRACTORS_PER_QUESTION
        common_neutral = neutral[:common_neutral_count]
        if len(common_neutral) != common_neutral_count:
            raise ValueError(f"{dataset}/{key}: insufficient common neutral facts")
        selections: dict[str, list[dict[str, Any]]] = {}
        for level in LEVELS:
            candidates = sorted(scored[key][level], key=lambda item: stable_key(SEED, dataset, key, level, item[0]))
            chosen = candidates[:DISTRACTORS_PER_QUESTION]
            chosen_indices = [item[0] for item in chosen]
            pool = anchors + common_neutral + chosen_indices
            if len(pool) != POOL_SIZE or len(set(pool)) != POOL_SIZE:
                raise ValueError(f"{dataset}/{key}/{level}: invalid controlled pool")
            pool.sort(key=lambda index: stable_key(SEED, dataset, key, index))
            pool_rows[level].append({"question_id": key, "candidate_fact_indices": pool})
            selections[level] = [{"fact_index": index, "similarity_score": score} for index, score in chosen]
        public.append({k: row[k] for k in ("question_id", "question", "question_type", "stratum")})
        trusted.append({"question_id": key, "anchors": anchors, "substitutes": [],
                        "distractors": sorted({item["fact_index"] for values in selections.values() for item in values})})
        assignment_rows.append({"question_id": key, "selected_distractors": selections})
        if position % 100 == 0 or position == len(selected):
            print(f"[prepare:{dataset}] controlled pools {position}/{len(selected)}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "model_input" / "test_questions.jsonl", public)
    for level in LEVELS:
        write_jsonl(output / "model_input" / "test_pools" / f"{level}.jsonl", pool_rows[level])
    write_jsonl(output / "trusted" / "test_labels.jsonl", trusted)
    write_jsonl(output / "trusted" / "distractor_assignments.jsonl", assignment_rows)
    audit = {
        "experiment": "controlled_ambiguity_stress", "dataset": dataset, "seed": SEED,
        "pool_size": POOL_SIZE, "top_k": [1, 10], "distractors_per_question": DISTRACTORS_PER_QUESTION,
        "train_quantiles": thresholds, "train_cohort_ids": len(train_ids),
        "full_eligible_test_population": len(test_population), "strict_5_per_bin_eligible": len(eligible),
        "selected_test_questions": len(selected), "selection_target": TEST_TARGET, "levels": list(LEVELS),
        "test_distractor_availability": availability,
        "controls": {"same_question_ids": True, "same_anchor_count": True, "same_neutral_candidates": True,
                     "same_pool_size": True, "same_top_k": True, "same_checkpoint_required": True,
                     "only_changed_component": "five curated distractor facts selected from the requested similarity tercile"},
        "facts_sha256": sha256(spec.facts),
        "test_questions_sha256": sha256(output / "model_input" / "test_questions.jsonl"),
        "no_private_labels_in_model_input": True,
    }
    write_json(output / "audit.json", audit)
    return audit


def load_facts(path: Path) -> list[dict]:
    facts = list(iter_jsonl(path))
    if any(int(row.get("fact_index", -1)) != position for position, row in enumerate(facts)):
        raise ValueError("fact catalog is not dense")
    return facts


def top10(scores: dict[int, float], candidates: Iterable[int]) -> list[int]:
    return sorted(candidates, key=lambda index: (-scores[index], index))[:10]


def recursive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted((x for x in path.rglob("*") if x.is_file()), key=lambda x: x.as_posix()):
        digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
