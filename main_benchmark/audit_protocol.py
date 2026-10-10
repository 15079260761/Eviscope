#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _shared.config import POOL_SIZE, model_paths
from _shared.io import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=HERE / "prepared")
    args = parser.parse_args()
    for name, path in model_paths().items():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"missing local {name} model: {path}")
    report = {}
    for dataset in ("cron", "icews"):
        root = args.prepared / dataset
        questions = list(iter_jsonl(root / "model_input" / "test_questions.jsonl"))
        ids = {x["question_id"] for x in questions}
        if any(key in row for row in questions for key in ("anchors", "substitutes", "distractors")):
            raise ValueError(f"{dataset}: test labels leaked into model input")
        mode_counts = {}
        for mode in ("full",):
            rows = list(iter_jsonl(root / "model_input" / "test_pools" / f"{mode}.jsonl"))
            if {x["question_id"] for x in rows} != ids:
                raise ValueError(f"{dataset}/{mode}: IDs differ from the frozen test cohort")
            if any(len(x["candidate_fact_indices"]) != POOL_SIZE or
                   len(set(x["candidate_fact_indices"])) != POOL_SIZE for x in rows):
                raise ValueError(f"{dataset}/{mode}: invalid candidate pool")
            mode_counts[mode] = len(rows)
        report[dataset] = {"test_questions": len(ids), "mode_counts": mode_counts,
                           "label_boundary": "pass"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[SUCCESS] protocol audit passed")


if __name__ == "__main__":
    main()
