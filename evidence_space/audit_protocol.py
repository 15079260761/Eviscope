#!/usr/bin/env python3
from __future__ import annotations
import json
from protocol import load_prepared
def main() -> None:
    report = {}
    for dataset, expected in (("cron", 1691), ("icews", 2000)):
        questions, labels, train, pools = load_prepared(dataset)
        if len(questions) != expected or len(labels) != expected: raise ValueError(f"{dataset}: unexpected paired test size")
        for mode, rows in pools.items():
            if any(len(v) != 1000 or len(set(v)) != 1000 for v in rows.values()): raise ValueError(f"{dataset}/{mode}: invalid pool")
        train_ids = [row["question_id"] for row in train]
        if len(train) != 8000 or len(set(train_ids)) != len(train_ids): raise ValueError(f"{dataset}: training cohort is not 8000 unique questions")
        if not all(row["substitutes"] and row["distractors"] for row in train): raise ValueError(f"{dataset}: selected training cohort lacks required evidence roles")
        train_texts = [row["question"].strip().casefold() for row in train]
        test_texts = [row["question"].strip().casefold() for row in labels]
        if len(set(train_texts)) != len(train_texts) or len(set(test_texts)) != len(test_texts): raise ValueError(f"{dataset}: duplicate question text in selected cohort")
        if set(train_texts) & set(test_texts): raise ValueError(f"{dataset}: exact train/test question-text leakage")
        anchors = [len(row["anchors"]) for row in labels]
        report[dataset] = {"test": len(questions), "train": len(train), "unique_train_ids": len(set(train_ids)), "unique_train_texts": len(set(train_texts)), "unique_test_texts": len(set(test_texts)), "exact_train_test_text_overlap": 0, "train_rows_with_substitutes": sum(bool(row["substitutes"]) for row in train), "train_rows_with_distractors": sum(bool(row["distractors"]) for row in train), "anchor_count": {"mean": sum(anchors)/len(anchors), "median": sorted(anchors)[len(anchors)//2], "min": min(anchors), "max": max(anchors)}, "bs_ceiling": {"k1": sum(min(1,n)/n for n in anchors)/len(anchors), "k10": sum(min(10,n)/n for n in anchors)/len(anchors)}, "pool_rows": {k: len(v) for k,v in pools.items()}, "factorial": {"original": ["substitute", False], "sub": ["substitute", True], "dis": ["full", False], "full": ["full", True]}, "training_test_condition": "full", "training_ablation_variants": ["anchor_only", "sub", "dis", "sub_dis", "full_eviret"]}
    print(json.dumps(report, ensure_ascii=False, indent=2)); print("[SUCCESS] Experiment 2 protocol audit passed")
if __name__ == "__main__": main()
