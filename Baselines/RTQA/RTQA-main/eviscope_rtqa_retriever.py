"""RTQA BGE-M3 retrieval-stage adapter for EviScope CronQuestions.

RTQA is an inference-time retrieval/reasoning system, not a supervised QA
trainer.  EviScope evaluates ranked evidence facts, so this adapter uses the
method's published BGE-M3 dense retrieval stage and deliberately excludes its
answer-generation LLM (which neither emits evidence ranks nor has reproducible
weights/API settings in the released code).
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eviscope-root", type=Path, required=True)
    parser.add_argument("--view", choices=("clean", "full"), required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(args.eviscope_root))
    from tools.candidate_space.runtime.pool_views import ReleaseFactIndex, materialize_pool

    try:
        import torch
        from FlagEmbedding import BGEM3FlagModel
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "FlagEmbedding is required; install the experiment requirements first") from exc
    if not torch.cuda.is_available():
        raise SystemExit("RTQA BGE-M3 retrieval requires the assigned CUDA GPU")

    facts_path = args.eviscope_root / "data" / "facts" / "facts.jsonl"
    fact_runtime = ReleaseFactIndex(facts_path)
    questions = {row["paired_id"]: row for row in iter_jsonl(
        args.eviscope_root / "data" / "questions" / "test.jsonl")}
    descriptors = iter_jsonl(args.eviscope_root / "data" / "descriptors" / "test.jsonl")
    labels = iter_jsonl(args.eviscope_root / "data" / "private_labels" / "test.jsonl")

    examples = []
    unique_indices = set()
    for descriptor, label in zip(descriptors, labels, strict=True):
        if descriptor.get("paired_id") != label.get("paired_id"):
            raise RuntimeError("Descriptor/private-label pairing mismatch")
        # Use one fixed paired cohort for both conditions.
        if not descriptor.get("mode_eligibility", {}).get("full"):
            continue
        paired_id = descriptor["paired_id"]
        if paired_id not in questions:
            raise RuntimeError(f"Missing public question: {paired_id}")
        pool = materialize_pool(descriptor, label, args.view, fact_runtime)
        candidates = [int(value) for value in pool["candidate_fact_indices"]]
        if len(candidates) != 1000:
            raise RuntimeError(f"Unexpected pool size for {paired_id}: {len(candidates)}")
        examples.append((questions[paired_id], candidates))
        unique_indices.update(candidates)
    if len(examples) != 1691:
        raise RuntimeError(f"Expected 1691 paired Test questions, found {len(examples)}")

    fact_text_by_index = {}
    for ordinal, fact in enumerate(iter_jsonl(facts_path)):
        index = int(fact.get("fact_index", ordinal))
        if index in unique_indices:
            fact_text_by_index[index] = fact.get("fact_text") or fact["canonical_fact_text"]
    if set(fact_text_by_index) != unique_indices:
        raise RuntimeError("Candidate fact catalog lookup is incomplete")
    ordered_indices = sorted(unique_indices)
    index_to_row = {index: row for row, index in enumerate(ordered_indices)}
    corpus = [fact_text_by_index[index] for index in ordered_indices]
    query_texts = [question["question"] for question, _ in examples]

    print(json.dumps({
        "method": "RTQA", "component": "published_bge_m3_retrieval_stage",
        "view": args.view, "questions": len(examples), "pool_size": 1000,
        "unique_candidate_facts": len(corpus), "model": args.model,
        "test_labels_exposed_to_model": False,
    }, ensure_ascii=False))
    # CUDA_VISIBLE_DEVICES makes the assigned physical card local cuda:0.
    # Resolve the already downloaded snapshot ourselves. Passing a Hub repo ID
    # to FlagEmbedding makes it call snapshot_download again without our ignore
    # rules, which fails on some mirrors for the irrelevant imgs/.DS_Store.
    model_path = snapshot_download(
        args.model,
        local_files_only=True,
        ignore_patterns=["*.DS_Store", "**/.DS_Store", "imgs/*", "imgs/**"],
    )
    print(f"[model] offline snapshot: {model_path}")
    # FlagEmbedding 1.3 uses the new Transformers `dtype` loading keyword.
    # Transformers 4.44 (already used by the other baselines) forwards that
    # keyword into XLMRobertaModel.__init__, which rejects it.  Remove only
    # that loader-only keyword on old Transformers instead of upgrading the
    # shared environment and risking the completed baseline experiments.
    from transformers import XLMRobertaModel
    if "dtype" not in inspect.signature(XLMRobertaModel.__init__).parameters:
        original_init = XLMRobertaModel.__init__

        def compatible_init(self, config, *model_args, **model_kwargs):
            model_kwargs.pop("dtype", None)
            return original_init(self, config, *model_args, **model_kwargs)

        XLMRobertaModel.__init__ = compatible_init
        print("[compat] removed unsupported XLMRobertaModel dtype loader keyword")
    model = BGEM3FlagModel(model_path, use_fp16=True, devices=["cuda:0"])
    corpus_encoded = model.encode_corpus(
        corpus, batch_size=args.batch_size, max_length=args.max_length,
        convert_to_numpy=True, return_dense=True, return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"].astype(np.float32, copy=False)
    query_encoded = model.encode_queries(
        query_texts, batch_size=args.batch_size, max_length=args.max_length,
        convert_to_numpy=True, return_dense=True, return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"].astype(np.float32, copy=False)

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_out.open("w", encoding="utf-8", newline="\n") as stream:
        for (question, candidates), query in zip(examples, query_encoded, strict=True):
            rows = np.fromiter((index_to_row[index] for index in candidates),
                               dtype=np.int64, count=len(candidates))
            scores = corpus_encoded[rows] @ query
            # Stable sorting makes candidate-pool order the deterministic tie break.
            order = np.argsort(-scores, kind="stable")[:args.top_k]
            ranked = [candidates[int(position)] for position in order]
            output = {
                "question_id": question["question_id"],
                "paired_id": question["paired_id"],
                "view": args.view,
                "materialized_mode": args.view,
                "ranked_fact_indices": ranked,
            }
            stream.write(json.dumps(output, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    manifest = {
        "method": "RTQA", "adapted_component": "BGE-M3 evidence retriever",
        "model": args.model, "view": args.view, "questions": len(examples),
        "pool_size": 1000, "top_k": args.top_k,
        "full_recursive_answer_generator_used": False,
        "reason": "EviScope scores ranked evidence, not generated answers",
        "output": str(args.predictions_out),
    }
    args.predictions_out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
