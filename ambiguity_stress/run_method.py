#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "main_benchmark"
sys.path.insert(0, str(EXP1))

from _shared.config import SEED, dataset_spec
from _shared.io import iter_jsonl, write_json, write_jsonl
from _shared.metrics import FrozenHashEncoder, evaluate
from _shared.neural import seed_everything
from protocol import LEVELS, load_facts, recursive_sha256, top10


def validate_checkpoint_metadata(method: str, metadata: dict, checkpoint: Path) -> None:
    """Validate both current and legacy Experiment-1 checkpoint metadata.

    Older completed checkpoints predate the explicit ``use_hdm`` audit field.
    Absence is therefore not evidence of a method mismatch.  Fields that are
    present are still enforced, and the frozen objective/sample count always
    has to identify the expected Experiment-1 method.
    """
    count_key = "questions" if method == "cross_encoder" else "examples"
    if int(metadata.get(count_key, -1)) != 8000:
        raise ValueError(f"checkpoint was not trained on the frozen 8,000-question protocol: {checkpoint}")
    accepted_objectives = {
        "dual_encoder": {"anchor_inbatch"},
        "cross_encoder": {"pointwise_bce"},
        # ``eviret_group_hdm`` is the canonical objective emitted by the
        # completed Experiment-1 EviRet run. ``group_multi_positive`` is kept
        # only for backward compatibility with earlier equivalent metadata.
        "eviret": {"eviret_group_hdm", "group_multi_positive"},
    }[method]
    observed_objective = metadata.get("objective")
    if observed_objective not in accepted_objectives:
        raise ValueError(
            f"checkpoint objective mismatch at {checkpoint}: "
            f"accepted={sorted(accepted_objectives)!r}, observed={observed_objective!r}")
    expected_flags = {
        "dual_encoder": {"use_substitutes": False, "use_distractors": False, "use_hdm": False},
        "cross_encoder": {"use_substitutes": False, "use_distractors": False},
        # EviRet identity in Experiment 1 is established by its multi-positive
        # objective plus substitute/distractor use.  Legacy checkpoints may
        # omit ``use_hdm`` or record the pre-audit value; Exp.3 must reuse that
        # already reported checkpoint rather than silently redefine/retrain it.
        "eviret": {"use_substitutes": True, "use_distractors": True},
    }[method]
    for flag, expected in expected_flags.items():
        if flag in metadata and bool(metadata[flag]) != expected:
            raise ValueError(
                f"checkpoint flag mismatch at {checkpoint}: "
                f"{flag} expected={expected}, observed={metadata[flag]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen Exp.1 retriever on Exp.3")
    parser.add_argument("--method", choices=("bm25", "dual_encoder", "cross_encoder", "eviret"), required=True)
    parser.add_argument("--dataset", choices=("cron", "icews"), required=True)
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "exp3_v1"))
    parser.add_argument("--checkpoint-run-id", default=os.environ.get("CHECKPOINT_RUN_ID", "exp1_full_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.method != "bm25":
        seed_everything(SEED)
    spec = dataset_spec(args.dataset)
    prepared = HERE / "prepared" / args.dataset
    facts = load_facts(spec.facts)
    questions = list(iter_jsonl(prepared / "model_input" / "test_questions.jsonl"))
    labels = list(iter_jsonl(prepared / "trusted" / "test_labels.jsonl"))
    text = {row["question_id"]: row["question"] for row in questions}
    pools = {level: {row["question_id"]: [int(x) for x in row["candidate_fact_indices"]]
                     for row in iter_jsonl(prepared / "model_input" / "test_pools" / f"{level}.jsonl")}
             for level in LEVELS}
    ids = set(text)
    if any(set(pools[level]) != ids for level in LEVELS):
        raise ValueError("conditions do not share the exact paired test cohort")

    checkpoint = EXP1 / "checkpoints" / args.checkpoint_run_id / args.method / args.dataset
    checkpoint_hash = None
    checkpoint_metadata = None
    if args.method == "bm25":
        from bm25 import BM25Retriever
        retriever = BM25Retriever(facts)
    elif args.method in ("dual_encoder", "eviret"):
        if not (checkpoint / "config.json").is_file() or not (checkpoint / "training_metadata.json").is_file():
            raise FileNotFoundError(f"required frozen Experiment-1 checkpoint missing: {checkpoint}")
        metadata = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
        validate_checkpoint_metadata(args.method, metadata, checkpoint)
        checkpoint_metadata = metadata
        from dual_encoder import DualEncoderRetriever
        retriever = DualEncoderRetriever(checkpoint, args.device)
        checkpoint_hash = recursive_sha256(checkpoint)
    else:
        if not (checkpoint / "config.json").is_file() or not (checkpoint / "training_metadata.json").is_file():
            raise FileNotFoundError(f"required frozen Experiment-1 checkpoint missing: {checkpoint}")
        metadata = json.loads((checkpoint / "training_metadata.json").read_text(encoding="utf-8"))
        validate_checkpoint_metadata(args.method, metadata, checkpoint)
        checkpoint_metadata = metadata
        from cross_encoder import CrossEncoderRetriever
        retriever = CrossEncoderRetriever(checkpoint, args.device)
        checkpoint_hash = recursive_sha256(checkpoint)

    union_indices = sorted({index for level in LEVELS for pool in pools[level].values() for index in pool})
    dense_cache = None
    if args.method in ("dual_encoder", "eviret"):
        print(f"[encode] {args.dataset} unique controlled-pool facts={len(union_indices)}", flush=True)
        dense_cache = retriever.encode_corpus(facts, union_indices)

    predictions = {level: [] for level in LEVELS}
    for position, key in enumerate(sorted(ids), 1):
        candidates = sorted({index for level in LEVELS for index in pools[level][key]})
        if args.method == "bm25":
            scores = retriever.score(text[key], candidates)
        elif args.method in ("dual_encoder", "eviret"):
            scores = retriever.score(text[key], candidates, dense_cache)
        else:
            scores = retriever.score(text[key], candidates, facts, args.score_batch_size)
        for level in LEVELS:
            predictions[level].append({"question_id": key, "ranked_fact_indices": top10(scores, pools[level][key])})
        if position % 100 == 0 or position == len(ids):
            print(f"[predict:{args.method}/{args.dataset}] {position}/{len(ids)}", flush=True)

    output = HERE / "method_results" / args.run_id / args.method / args.dataset
    encoder = FrozenHashEncoder(spec.facts)
    summaries = {}
    print("[boundary] predictions frozen; trusted labels enter evaluator only", flush=True)
    for level in LEVELS:
        write_jsonl(output / level / "predictions.jsonl", predictions[level])
        summaries[level] = {}
        for k in (1, 10):
            summary, per_question = evaluate(predictions[level], labels, facts, encoder, spec.bs_tau, k)
            summaries[level][f"k{k}"] = summary
            write_json(output / level / f"metrics_k{k}.json", summary)
            write_jsonl(output / level / f"per_question_k{k}.jsonl", per_question)
            print(f"[metric] {level} K={k} H={summary['hits']:.6f} BS={summary['bs']:.6f} "
                  f"Cov={summary['coverage']:.6f} nBS={summary['nbs']:.6f}", flush=True)
    write_json(output / "summary.json", summaries)
    write_json(output / "run_manifest.json", {
        "experiment": "controlled_ambiguity_stress", "method": args.method, "dataset": args.dataset,
        "run_id": args.run_id, "seed": SEED, "test_questions": len(ids), "pool_size": 1000,
        "distractors_per_condition": 5, "levels": list(LEVELS), "prediction_depth": 10,
        "checkpoint_run_id": None if args.method == "bm25" else args.checkpoint_run_id,
        "checkpoint": None if args.method == "bm25" else str(checkpoint),
        "checkpoint_sha256": checkpoint_hash, "same_checkpoint_all_levels": True,
        "checkpoint_training_metadata": checkpoint_metadata,
    })
    print(f"[SUCCESS] {args.method}/{args.dataset} -> {output}", flush=True)

if __name__ == "__main__":
    main()
