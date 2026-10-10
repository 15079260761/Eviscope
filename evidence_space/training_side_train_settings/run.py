#!/usr/bin/env python3
"""Section 2.7 training ablation: five EviRet variants on a fixed Full test set."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent
EXP1 = EXP2.parent / "main_benchmark"
sys.path[:0] = [str(EXP2), str(EXP1)]

from protocol import (FrozenHashEncoder, SEED, dataset_spec, load_facts,
                      load_prepared, make_retriever, model_paths,
                      save_and_evaluate, score_union, top10, write_json)
from _shared.io import sha256
from _shared.neural import seed_everything


VARIANTS = {
    "anchor_only": {"use_substitutes": False, "use_distractors": False, "use_hdm": False},
    "sub": {"use_substitutes": True, "use_distractors": False, "use_hdm": False},
    "dis": {"use_substitutes": False, "use_distractors": True, "use_hdm": False},
    "sub_dis": {"use_substitutes": True, "use_distractors": True, "use_hdm": False},
    "full_eviret": {"use_substitutes": True, "use_distractors": True, "use_hdm": True},
}


def ids_sha256(rows: list[dict]) -> str:
    payload = "\n".join(sorted(str(row["question_id"]) for row in rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cron", "icews"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "exp2_v2"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--resume-verified", action="store_true",
                        help="resume only from a checkpoint/result whose metadata passes all checks")
    args = parser.parse_args()

    if args.epochs <= 0 or args.train_batch_size <= 0:
        raise ValueError("epochs and train batch size must be positive")

    flags = VARIANTS[args.variant]
    root = HERE / "results" / args.run_id / "eviret" / args.dataset / args.variant
    checkpoint = HERE / "checkpoints" / args.run_id / "eviret" / args.dataset / args.variant
    if root.exists():
        manifest_path = root / "run_manifest.json"
        if args.resume_verified and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("protocol_version") == "exp2-section-2.7-v2" and
                    manifest.get("dataset") == args.dataset and
                    manifest.get("variant") == args.variant and
                    manifest.get("training_questions") == 8000 and
                    (root / "summary.json").is_file() and
                    (root / "predictions.jsonl").is_file()):
                print(f"[SKIP VERIFIED] Section 2.7 EviRet/{args.dataset}/{args.variant}",
                      flush=True)
                return
        raise FileExistsError(f"unverified or incomplete result directory: {root}")
    if checkpoint.exists() and not args.resume_verified:
        raise FileExistsError(
            f"refusing silent reuse/overwrite for {args.run_id}/{args.dataset}/{args.variant}; "
            "use a new run ID or explicit --resume-verified"
        )

    seed_everything(SEED)
    spec = dataset_spec(args.dataset)
    facts = load_facts(spec.facts)
    questions, labels, train, pools = load_prepared(args.dataset)
    if len(train) != 8000 or len({row["question_id"] for row in train}) != len(train):
        raise ValueError(f"{args.dataset}: expected 8000 unique training questions")
    if flags["use_substitutes"] and not all(row["substitutes"] for row in train):
        raise ValueError(f"{args.dataset}/{args.variant}: missing substitute supervision")
    if flags["use_distractors"] and not all(row["distractors"] for row in train):
        raise ValueError(f"{args.dataset}/{args.variant}: missing distractor supervision")

    qmap = {row["question_id"]: row["question"] for row in questions}
    expected_steps = math.ceil(len(train) / args.train_batch_size) * args.epochs
    metadata_path = checkpoint / "training_metadata.json"
    if checkpoint.exists():
        print(f"[resume] validating completed checkpoint {checkpoint}", flush=True)
    else:
        print(json.dumps({
            "stage": "training_start",
            "dataset": args.dataset,
            "variant": args.variant,
            "training_questions": len(train),
            "training_question_ids_sha256": ids_sha256(train),
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "expected_optimizer_steps": expected_steps,
            **flags,
        }, ensure_ascii=False, sort_keys=True), flush=True)
        retriever = make_retriever("eviret", model_paths()["dual"], facts, args.device)
        completed_steps = retriever.train_model(
            train, facts, checkpoint, args.epochs, args.train_batch_size, SEED,
            eviret=True,
            use_substitutes=flags["use_substitutes"],
            use_distractors=flags["use_distractors"],
            use_hdm=flags["use_hdm"],
        )
        if completed_steps != expected_steps:
            raise RuntimeError(f"trainer returned {completed_steps} steps, expected {expected_steps}")
    if not metadata_path.is_file():
        raise RuntimeError("training finished without checkpoint metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("examples", -1)) != len(train):
        raise RuntimeError("checkpoint metadata does not attest all training examples")
    if int(metadata.get("optimizer_steps", -1)) != expected_steps:
        raise RuntimeError(
            f"checkpoint attests {metadata.get('optimizer_steps')} optimizer steps, "
            f"expected {expected_steps}"
        )
    for key, value in flags.items():
        if metadata.get(key) is not value:
            raise RuntimeError(f"checkpoint metadata mismatch: {key}={metadata.get(key)!r}, expected {value!r}")
    if int(metadata.get("epochs", -1)) != args.epochs:
        raise RuntimeError("checkpoint epoch count mismatch")
    retriever = make_retriever("eviret", checkpoint, facts, args.device)

    indices = sorted({index for values in pools["full"].values() for index in values})
    cache = retriever.encode_corpus(facts, indices)
    predictions = []
    for position, question_id in enumerate(sorted(qmap), 1):
        candidates = pools["full"][question_id]
        scores = score_union("eviret", retriever, qmap[question_id], candidates,
                             facts, cache, args.score_batch_size)
        predictions.append({"question_id": question_id,
                            "ranked_fact_indices": top10(scores, candidates)})
        if position % 100 == 0 or position == len(qmap):
            print(f"[training-ablation] eviret/{args.dataset}/{args.variant} "
                  f"{position}/{len(qmap)}", flush=True)

    encoder = FrozenHashEncoder(spec.facts)
    summary = save_and_evaluate(root, predictions, labels, facts, encoder, spec.bs_tau)
    write_json(root / "summary.json", summary)
    write_json(root / "run_manifest.json", {
        "protocol_version": "exp2-section-2.7-v2",
        "direction": "training_ablation",
        "method": "eviret",
        "dataset": args.dataset,
        "variant": args.variant,
        "test_condition": "full",
        "training_questions": len(train),
        "training_unique_question_ids": len({row["question_id"] for row in train}),
        "training_question_ids_sha256": ids_sha256(train),
        "training_labels_sha256": sha256(EXP2 / "prepared" / args.dataset /
                                         "trusted" / "train_labels.jsonl"),
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "expected_optimizer_steps": expected_steps,
        "checkpoint_training_metadata": metadata,
        **flags,
    })
    print(f"[SUCCESS] Section 2.7 EviRet/{args.dataset}/{args.variant}", flush=True)


if __name__ == "__main__":
    main()
