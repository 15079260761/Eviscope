#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from protocol import DISTRACTORS_PER_QUESTION, LEVELS

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "main_benchmark"
sys.path.insert(0, str(EXP1))
from _shared.io import iter_jsonl

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cron", "icews", "all"), default="all")
    args = parser.parse_args()
    for dataset in (("cron", "icews") if args.dataset == "all" else (args.dataset,)):
        root = HERE / "prepared" / dataset
        audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
        questions = {x["question_id"] for x in iter_jsonl(root / "model_input" / "test_questions.jsonl")}
        labels = {x["question_id"]: x for x in iter_jsonl(root / "trusted" / "test_labels.jsonl")}
        assignments = {x["question_id"]: x for x in iter_jsonl(root / "trusted" / "distractor_assignments.jsonl")}
        pools = {level: {x["question_id"]: x["candidate_fact_indices"]
                         for x in iter_jsonl(root / "model_input" / "test_pools" / f"{level}.jsonl")}
                 for level in LEVELS}
        if set(labels) != questions or set(assignments) != questions or any(set(x) != questions for x in pools.values()):
            raise ValueError(f"{dataset}: paired cohort IDs differ")
        for key in questions:
            anchors = set(labels[key]["anchors"])
            condition_sets = {level: set(pools[level][key]) for level in LEVELS}
            if any(len(pools[level][key]) != audit["pool_size"] or len(condition_sets[level]) != audit["pool_size"] for level in LEVELS):
                raise ValueError(f"{dataset}/{key}: pool size or uniqueness failure")
            chosen = {level: {x["fact_index"] for x in assignments[key]["selected_distractors"][level]} for level in LEVELS}
            if any(len(chosen[level]) != DISTRACTORS_PER_QUESTION for level in LEVELS):
                raise ValueError(f"{dataset}/{key}: not exactly five distractors")
            if any(chosen[left] & chosen[right] for left, right in (("low", "medium"), ("low", "high"), ("medium", "high"))):
                raise ValueError(f"{dataset}/{key}: distractor bins overlap")
            neutral = {level: condition_sets[level] - anchors - chosen[level] for level in LEVELS}
            if not (neutral["low"] == neutral["medium"] == neutral["high"]):
                raise ValueError(f"{dataset}/{key}: neutral background differs")
            for level in LEVELS:
                if not chosen[level].issubset(condition_sets[level]):
                    raise ValueError(f"{dataset}/{key}/{level}: assigned distractor absent")
            if any(len(condition_sets[left] ^ condition_sets[right]) != 10
                   for left, right in (("low", "medium"), ("low", "high"), ("medium", "high"))):
                raise ValueError(f"{dataset}/{key}: a pair does not replace exactly five facts")
            q33, q66 = audit["train_quantiles"]["q33"], audit["train_quantiles"]["q66"]
            scores = {level: [float(x["similarity_score"])
                              for x in assignments[key]["selected_distractors"][level]] for level in LEVELS}
            if (not all(score < q33 for score in scores["low"])
                    or not all(q33 <= score < q66 for score in scores["medium"])
                    or not all(score >= q66 for score in scores["high"])):
                raise ValueError(f"{dataset}/{key}: a selected score violates its train-fitted bin")
        print(f"[PASS] {dataset}: N={len(questions)}, paired pools=1000, anchors/neutrals fixed, distractors=5", flush=True)

if __name__ == "__main__":
    main()
