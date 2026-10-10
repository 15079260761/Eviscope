from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import (DEV_TARGET, POOL_SIZE, SEED, TEST_TARGET, TRAIN_TARGET,
                     DatasetSpec, dataset_spec, experiment_root)
from .io import fact_text, iter_jsonl, qid, sha256, write_json, write_jsonl


def roles(dataset: str, row: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    anchors = list(dict.fromkeys(int(x) for x in row.get("anchor_fact_indices", [])))
    if dataset == "cron":
        substitutes = list(dict.fromkeys(int(x) for x in row.get("substitute_fact_indices", [])))
        distractors = list(dict.fromkeys(int(x) for x in row.get("distractor_fact_indices", [])))
    else:
        substitutes = list(dict.fromkeys(
            int(x["fact_index"]) for x in row.get("substitutable_evidence", [])))
        distractors = list(dict.fromkeys(
            int(x["fact_index"]) for x in row.get("distracting_evidence", [])))
    return anchors, substitutes, distractors


def substitute_groups(dataset: str, row: dict[str, Any]) -> dict[int, list[int]]:
    anchors, substitutes, _ = roles(dataset, row)
    groups = {anchor: [] for anchor in anchors}
    if dataset == "icews":
        for item in row.get("substitutable_evidence", []):
            anchor = int(item["anchor_fact_index"])
            if anchor in groups:
                groups[anchor].append(int(item["fact_index"]))
    return groups


def _resolve_cron_groups(rows: list[dict[str, Any]], facts_path: Path) -> None:
    """Reconstruct missing Cron substitute→anchor training edges semantically.

    The compact Cron release publishes substitute roles but not their generating
    anchor edge.  EviRet needs that edge only for training.  We deterministically
    attach each substitute to its most similar anchor using the exact frozen
    benchmark encoder; no test information is involved.
    """
    from .metrics import FrozenHashEncoder
    encoder = FrozenHashEncoder(facts_path)
    facts = list(iter_jsonl(facts_path))
    embedding_cache: dict[int, dict[int, float]] = {}

    def embedding(index: int) -> dict[int, float]:
        if index not in embedding_cache:
            embedding_cache[index] = encoder.embed(fact_text(facts[index]))
        return embedding_cache[index]

    for position, row in enumerate(rows, 1):
        anchors = [int(x) for x in row["anchors"]]
        groups = {anchor: [] for anchor in anchors}
        for substitute in row["substitutes"]:
            sub_embedding = embedding(int(substitute))
            anchor = max(anchors, key=lambda item: (encoder.cosine(
                sub_embedding, embedding(item)), -item))
            groups[anchor].append(int(substitute))
        row["substitute_groups"] = groups
        if position % 500 == 0 or position == len(rows):
            print(f"[prepare:cron] semantic substitute groups {position}/{len(rows)}", flush=True)


def _eligible(required_modes: dict[str, str], descriptor: dict[str, Any]) -> bool:
    expected = set(required_modes.values())
    eligibility = descriptor.get("mode_eligibility", {})
    return expected.issubset({key for key, value in eligibility.items() if value})


def _bin(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 16:
        return "5-16"
    return "17+"


def _stratum(question: dict[str, Any], label: dict[str, Any], dataset: str) -> str:
    anchors, substitutes, distractors = roles(dataset, label)
    difficulty = (label.get("hybrid_difficulty") or {}).get("difficulty_level", "unknown")
    return "|".join((str(question.get("question_type", "unknown")), str(difficulty),
                     _bin(len(anchors)), _bin(len(substitutes)), _bin(len(distractors))))


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _stratified(rows: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= target:
        return sorted(rows, key=lambda x: _stable_key(seed, x["question_id"]))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["stratum"]].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda x: _stable_key(seed, x["question_id"]))
    exact = {key: target * len(values) / len(rows) for key, values in buckets.items()}
    quota = {key: min(len(buckets[key]), int(math.floor(value))) for key, value in exact.items()}
    # Preserve rare strata whenever the target permits it.
    if target >= len(buckets):
        for key in buckets:
            quota[key] = max(1, quota[key])
    while sum(quota.values()) < target:
        candidates = [key for key in buckets if quota[key] < len(buckets[key])]
        key = max(candidates, key=lambda k: (exact[k] - quota[k], _stable_key(seed, k)))
        quota[key] += 1
    while sum(quota.values()) > target:
        candidates = [key for key in buckets if quota[key] > 1]
        key = min(candidates, key=lambda k: (exact[k] - quota[k], _stable_key(seed, k)))
        quota[key] -= 1
    selected = [row for key, values in buckets.items() for row in values[:quota[key]]]
    return sorted(selected, key=lambda x: _stable_key(seed, x["question_id"]))


def _load_split(spec: DatasetSpec, split: str) -> list[dict[str, Any]]:
    label_path = spec.private_test if split == "test" else spec.supervision_dir / f"{split}.jsonl"
    questions = {qid(x): x for x in iter_jsonl(spec.questions_dir / f"{split}.jsonl")}
    labels = {qid(x): x for x in iter_jsonl(label_path)}
    descriptors = {qid(x): x for x in iter_jsonl(spec.descriptors_dir / f"{split}.jsonl")}
    common = set(questions) & set(labels) & set(descriptors)
    rows = []
    for key in common:
        if not _eligible(spec.modes, descriptors[key]):
            continue
        row = {
            "question_id": key,
            "question": questions[key]["question"],
            "question_type": questions[key].get("question_type"),
            "stratum": _stratum(questions[key], labels[key], spec.key),
            "anchors": roles(spec.key, labels[key])[0],
            "substitutes": roles(spec.key, labels[key])[1],
            "distractors": roles(spec.key, labels[key])[2],
            "substitute_groups": substitute_groups(spec.key, labels[key]),
            "difficulty": (labels[key].get("hybrid_difficulty") or {}).get("difficulty_level"),
            "hdm_difficulty": float((labels[key].get("hybrid_difficulty") or {}).get("D_q", 0.0)),
            "ambiguity": float((labels[key].get("ambiguity_pressure") or {}).get("A_q", 0.0)),
            "descriptor": descriptors[key],
        }
        rows.append(row)
    return rows


def _load_runtime(spec: DatasetSpec):
    path = (spec.release_dir / "tools" / "candidate_space" / "runtime" / "pool_views.py"
            if spec.key == "cron" else spec.release_dir / "runtime" / "pool_views.py")
    name = f"eviscope_{spec.key}_pool_views"
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load pool runtime: {path}")
    module = importlib.util.module_from_spec(module_spec)
    if spec.key == "cron":
        # The Cron module uses a package-relative import.  Its runtime directory
        # is imported as a temporary package without modifying release files.
        import sys
        package_name = f"eviscope_{spec.key}_runtime"
        package_spec = importlib.util.spec_from_file_location(
            package_name, path.parent / "__init__.py",
            submodule_search_locations=[str(path.parent)])
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package
        assert package_spec and package_spec.loader
        package_spec.loader.exec_module(package)
        module_spec = importlib.util.spec_from_file_location(f"{package_name}.pool_views", path)
        assert module_spec and module_spec.loader
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def prepare_dataset(dataset: str, output: Path,
                    modes: dict[str, str] | None = None) -> dict[str, Any]:
    spec = dataset_spec(dataset)
    if modes is not None:
        spec = replace(spec, modes=modes)
    output.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[dict[str, Any]]] = {}
    targets = {"train": TRAIN_TARGET, "dev": DEV_TARGET, "test": TEST_TARGET}
    for offset, split in enumerate(("train", "dev", "test")):
        eligible = _load_split(spec, split)
        selected[split] = _stratified(eligible, targets[split], SEED + offset)
        print(f"[prepare:{dataset}] split={split} full_eligible={len(eligible)} selected={len(selected[split])}", flush=True)
        if dataset == "cron" and split == "train":
            _resolve_cron_groups(selected[split], spec.facts)
        public = [{k: row[k] for k in ("question_id", "question", "question_type", "stratum")}
                  for row in selected[split]]
        trusted = [{k: v for k, v in row.items() if k != "descriptor"}
                   for row in selected[split]]
        write_jsonl(output / "model_input" / f"{split}_questions.jsonl", public)
        write_jsonl(output / "trusted" / f"{split}_labels.jsonl", trusted)

    runtime = _load_runtime(spec)
    fact_index = runtime.ReleaseFactIndex(spec.facts)
    catalog_compatibility: dict[str, Any] = {"used": False}
    if dataset == "icews" and selected["test"]:
        declared = selected["test"][0]["descriptor"]["fact_catalog_sha256"]
        if fact_index.catalog_sha256 != declared:
            # The downloaded lightweight package expands `facts.jsonl.gz` to
            # `facts.jsonl`; descriptors intentionally retain the hash of the
            # immutable compressed release artifact.  Accept this packaging
            # transformation only when both frozen lineage and the semantic
            # release audit attest the same catalog and dense row count.
            lineage = json.loads((spec.release_dir / "audit" / "source_lineage.json").read_text(
                encoding="utf-8"))
            semantic = json.loads((spec.release_dir / "audit" / "full_semantic_audit.json").read_text(
                encoding="utf-8"))
            if (lineage.get("compact_facts_sha256") != declared or
                    semantic.get("status") != "pass" or
                    int(semantic.get("facts", -1)) != len(fact_index.ids)):
                raise ValueError("ICEWS fact catalog differs from its attested packaged release")
            catalog_compatibility = {
                "used": True,
                "reason": "downloaded gzip artifact was transparently expanded",
                "descriptor_and_lineage_sha256": declared,
                "local_uncompressed_sha256": fact_index.catalog_sha256,
                "semantic_audit_status": semantic["status"],
            }
            fact_index.catalog_sha256 = declared
    # Private labels are used only here, by trusted preprocessing.
    original_labels = {qid(x): x for x in iter_jsonl(spec.private_test)}
    for canonical_mode, release_mode in spec.modes.items():
        pool_rows = []
        for position, row in enumerate(selected["test"], 1):
            label = original_labels[row["question_id"]]
            if dataset == "cron":
                pool = runtime.materialize_pool(row["descriptor"], label, release_mode, fact_index)
            else:
                pool = runtime.materialize_all(row["descriptor"], label, fact_index)[release_mode]
            if pool["pool_size"] != POOL_SIZE:
                raise ValueError(f"{dataset}/{canonical_mode}/{row['question_id']}: pool != {POOL_SIZE}")
            pool_rows.append({"question_id": row["question_id"],
                              "candidate_fact_indices": pool["candidate_fact_indices"]})
            if position % 250 == 0 or position == len(selected["test"]):
                print(f"[prepare:{dataset}] mode={canonical_mode} pools={position}/{len(selected['test'])}", flush=True)
        write_jsonl(output / "model_input" / "test_pools" / f"{canonical_mode}.jsonl", pool_rows)

    audit = {
        "dataset": dataset, "seed": SEED, "pool_size": POOL_SIZE,
        "eligible_and_selected": {split: len(rows) for split, rows in selected.items()},
        "test_ids_sha256": sha256(output / "model_input" / "test_questions.jsonl"),
        "facts_sha256": sha256(spec.facts), "modes": list(spec.modes),
        "no_test_labels_in_model_input": True,
        "catalog_packaging_compatibility": catalog_compatibility,
        "strata": {
            split: {
                "distinct": len({row["stratum"] for row in rows}),
                "counts": dict(sorted((key, sum(1 for row in rows if row["stratum"] == key))
                                      for key in {row["stratum"] for row in rows})),
            }
            for split, rows in selected.items()
        },
    }
    write_json(output / "audit.json", audit)
    return audit


def prepared_root() -> Path:
    return experiment_root() / "prepared"
