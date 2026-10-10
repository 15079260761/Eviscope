#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _shared.config import SEED, dataset_spec, model_paths
from _shared.io import iter_jsonl, write_json, write_jsonl
from _shared.metrics import FrozenHashEncoder, evaluate
from _shared.neural import seed_everything


def load_facts(path: Path) -> list[dict]:
    facts = list(iter_jsonl(path))
    for expected, row in enumerate(facts):
        if int(row.get("fact_index", -1)) != expected:
            raise ValueError(f"non-dense fact catalog at {expected}")
    return facts


def rank(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda index: (-scores[index], index))[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one method/dataset on Experiment 1 full condition")
    parser.add_argument("--method", choices=("bm25", "dual_encoder", "cross_encoder", "eviret"), required=True)
    parser.add_argument("--dataset", choices=("cron", "icews"), required=True)
    parser.add_argument("--prepared", type=Path, default=HERE / "prepared")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", "exp1_v1"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    if args.method != "bm25":
        seed_everything(SEED)
    spec = dataset_spec(args.dataset)
    prepared = args.prepared / args.dataset
    if not (prepared / "audit.json").exists():
        raise FileNotFoundError(f"run prepare_protocol.py first: {prepared}")
    facts = load_facts(spec.facts)
    train_rows = list(iter_jsonl(prepared / "trusted" / "train_labels.jsonl"))
    questions = list(iter_jsonl(prepared / "model_input" / "test_questions.jsonl"))
    question_by_id = {x["question_id"]: x["question"] for x in questions}
    modes = ("full",)
    pools = {mode: {x["question_id"]: x["candidate_fact_indices"]
                    for x in iter_jsonl(prepared / "model_input" / "test_pools" / f"{mode}.jsonl")}
             for mode in modes}
    expected_ids = set(question_by_id)
    if set(pools["full"]) != expected_ids:
        raise ValueError("full view does not contain exactly the frozen test cohort")

    checkpoint = HERE / "checkpoints" / args.run_id / args.method / args.dataset
    model_source = model_paths()
    retriever = None
    dense_cache = None
    print(f"[protocol] method={args.method} dataset={args.dataset} train={len(train_rows)} test={len(questions)} condition=full", flush=True)
    if args.method == "bm25":
        from bm25 import BM25Retriever
        retriever = BM25Retriever(facts)
    elif args.method in ("dual_encoder", "eviret"):
        from dual_encoder import DualEncoderRetriever
        source = checkpoint if (checkpoint / "config.json").exists() and not args.force_train else model_source["dual"]
        retriever = DualEncoderRetriever(source, args.device)
        if source == model_source["dual"]:
            retriever.train_model(train_rows, facts, checkpoint, args.epochs,
                                  args.train_batch_size, SEED, eviret=args.method == "eviret")
        union = sorted({index for mode in modes for pool in pools[mode].values() for index in pool})
        print(f"[encode] unique test facts={len(union)}", flush=True)
        dense_cache = retriever.encode_corpus(facts, union)
    else:
        from cross_encoder import CrossEncoderRetriever
        source = checkpoint if (checkpoint / "config.json").exists() and not args.force_train else model_source["cross"]
        retriever = CrossEncoderRetriever(source, args.device)
        if source == model_source["cross"]:
            retriever.train_model(train_rows, facts, checkpoint, args.epochs,
                                  args.train_batch_size, SEED)

    # Prediction phase receives only public questions and unlabeled pools.
    prediction_by_mode = {mode: [] for mode in modes}
    for position, question_id in enumerate(sorted(expected_ids), 1):
        union = sorted({index for mode in modes for index in pools[mode][question_id]})
        question = question_by_id[question_id]
        if args.method == "bm25":
            union_scores = retriever.score(question, union)
        elif args.method in ("dual_encoder", "eviret"):
            union_scores = retriever.score(question, union, dense_cache)
        else:
            union_scores = retriever.score(question, union, facts, args.score_batch_size)
        for mode in modes:
            subset = {index: union_scores[index] for index in pools[mode][question_id]}
            prediction_by_mode[mode].append({"question_id": question_id,
                                             "ranked_fact_indices": rank(subset)})
        if position % 100 == 0 or position == len(expected_ids):
            print(f"[predict] {position}/{len(expected_ids)}", flush=True)

    output = args.results / args.run_id / args.method / args.dataset
    for mode in modes:
        write_jsonl(output / mode / "predictions.jsonl", prediction_by_mode[mode])
    print("[boundary] predictions frozen; private test labels now enter evaluator only", flush=True)

    labels = list(iter_jsonl(prepared / "trusted" / "test_labels.jsonl"))
    encoder = FrozenHashEncoder(spec.facts)
    summaries = {}
    for mode in modes:
        summaries[mode] = {}
        for k in (1, 10):
            summary, per_question = evaluate(prediction_by_mode[mode], labels, facts,
                                             encoder, spec.bs_tau, k)
            summaries[mode][f"k{k}"] = summary
            write_json(output / mode / f"metrics_k{k}.json", summary)
            write_jsonl(output / mode / f"per_question_k{k}.jsonl", per_question)
            print(f"[metric] {mode} K={k} Hits={summary['hits']:.6f} BS={summary['bs']:.6f} "
                  f"Cov={summary['coverage']:.6f} nBS={summary['nbs']:.6f}", flush=True)
    write_json(output / "summary.json", summaries)
    write_json(output / "run_manifest.json", {
        "method": args.method, "dataset": args.dataset, "run_id": args.run_id,
        "seed": SEED, "train_questions": len(train_rows), "test_questions": len(questions),
        "checkpoint": None if args.method == "bm25" else str(checkpoint),
        "condition": "full",
        "prediction_depth": 10,
    })
    print(f"[SUCCESS] {args.method}/{args.dataset} -> {output}", flush=True)


if __name__ == "__main__":
    main()
