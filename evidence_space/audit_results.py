#!/usr/bin/env python3
"""Fail-closed audit for Experiment 2 raw artifacts before table generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ("bm25", "dual_encoder", "cross_encoder", "eviret")
DATASETS = {"cron": 1691, "icews": 2000}
CONDITIONS = ("original", "sub", "dis", "full")
VARIANTS = ("anchor_only", "sub", "dis", "sub_dis", "full_eviret")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def check_predictions(path: Path, expected: int) -> list[dict]:
    rows = read_jsonl(path)
    ids = [row["question_id"] for row in rows]
    if len(rows) != expected or len(set(ids)) != expected:
        raise ValueError(f"{path}: expected {expected} unique questions")
    for row in rows:
        ranking = row.get("ranked_fact_indices")
        if not isinstance(ranking, list) or len(ranking) != 10 or len(set(ranking)) != 10:
            raise ValueError(f"{path}: invalid Top-10 ranking for {row['question_id']}")
    return rows


def saturation(path: Path, expected: int) -> dict:
    rows = read_jsonl(path)
    if len(rows) != expected:
        raise ValueError(f"{path}: per-question metric count mismatch")
    return {
        name: sum(abs(float(row[name]) - 1.0) <= 1e-12 for row in rows) / expected
        for name in ("hits", "bs", "coverage", "nbs")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp2_v2")
    args = parser.parse_args()
    report = {"run_id": args.run_id, "benchmark": {}, "training_ablation": {}}

    for method in METHODS:
        for dataset, expected in DATASETS.items():
            root = HERE / "benchmark_side_test_settings" / "results" / args.run_id / method / dataset
            manifest = read_json(root / "run_manifest.json")
            if manifest.get("protocol_version") != "exp2-section-2.1-2.6-v2":
                raise ValueError(f"bad benchmark protocol manifest: {root}")
            predictions = {condition: check_predictions(root / condition / "predictions.jsonl", expected)
                           for condition in CONDITIONS}
            if predictions["original"] != predictions["sub"]:
                raise ValueError(f"{method}/{dataset}: Original/+Sub rankings differ")
            if predictions["dis"] != predictions["full"]:
                raise ValueError(f"{method}/{dataset}: +Dis/Full rankings differ")
            summary = read_json(root / "summary.json")
            for condition in CONDITIONS:
                for k in (1, 10):
                    values = summary[condition][f"k{k}"]
                    if values.get("questions") != expected:
                        raise ValueError(f"{root}: metric question count mismatch")
                    if any(not 0.0 <= float(values[name]) <= 1.0
                           for name in ("hits", "bs", "coverage", "nbs")):
                        raise ValueError(f"{root}: metric outside [0,1]")
            report["benchmark"][f"{method}/{dataset}"] = {
                "status": "pass",
                "full_fraction_equal_to_1_at_k1": saturation(
                    root / "full" / "per_question_k1.jsonl", expected),
                "full_fraction_equal_to_1_at_k10": saturation(
                    root / "full" / "per_question_k10.jsonl", expected),
            }

    expected_flags = {
        "anchor_only": (False, False, False), "sub": (True, False, False),
        "dis": (False, True, False), "sub_dis": (True, True, False),
        "full_eviret": (True, True, True),
    }
    for dataset, expected in DATASETS.items():
        for variant in VARIANTS:
            root = HERE / "training_side_train_settings" / "results" / args.run_id / "eviret" / dataset / variant
            manifest = read_json(root / "run_manifest.json")
            if manifest.get("protocol_version") != "exp2-section-2.7-v2":
                raise ValueError(f"bad training protocol manifest: {root}")
            flags = (manifest.get("use_substitutes"), manifest.get("use_distractors"),
                     manifest.get("use_hdm"))
            if flags != expected_flags[variant]:
                raise ValueError(f"{root}: variant flags mismatch")
            if manifest.get("training_questions") != 8000 or manifest.get("training_unique_question_ids") != 8000:
                raise ValueError(f"{root}: training rows are not 8000 unique questions")
            metadata = manifest.get("checkpoint_training_metadata", {})
            if metadata.get("examples") != 8000 or metadata.get("optimizer_steps") != manifest.get("expected_optimizer_steps"):
                raise ValueError(f"{root}: checkpoint does not attest expected training")
            check_predictions(root / "predictions.jsonl", expected)
            report["training_ablation"][f"eviret/{dataset}/{variant}"] = {
                "status": "pass", "training_questions": 8000,
                "optimizer_steps": metadata["optimizer_steps"],
                "training_question_ids_sha256": manifest["training_question_ids_sha256"],
                "training_labels_sha256": manifest["training_labels_sha256"],
                "fraction_equal_to_1_at_k1": saturation(root / "per_question_k1.jsonl", expected),
                "fraction_equal_to_1_at_k10": saturation(root / "per_question_k10.jsonl", expected),
            }

    output = HERE / "results" / args.run_id / "integrity_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"[SUCCESS] strict raw-result audit passed -> {output}")


if __name__ == "__main__":
    main()
