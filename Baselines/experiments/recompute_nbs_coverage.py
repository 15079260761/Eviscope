"""Recompute capacity-normalized BS and corrected one-to-one Coverage.

This script never runs a model.  It consumes the frozen EviScope labels,
facts/configuration, and already emitted ``predictions.jsonl`` files.  For
each result directory it writes two new, auditable JSON files without
overwriting the original ``scores_k*.json`` outputs:

* ``nbs_scores.json``
* ``coverage_corrected_scores.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_result_dirs(root: Path) -> list[Path]:
    directories = []
    for score_path in root.rglob("scores_k10.json"):
        directory = score_path.parent
        required = [
            directory / "scores_k1.json",
            directory / "scores_k10.json",
            directory / "predictions.jsonl",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete result directory {directory}; missing: {missing}")
        directories.append(directory)
    if not directories:
        raise FileNotFoundError(f"No scores_k10.json found under {root}")
    return sorted(set(directories))


def normalize_reference(reference: Any) -> tuple[str, Any]:
    if isinstance(reference, bool):
        raise TypeError("Boolean is not a valid fact reference")
    if isinstance(reference, int):
        return ("index", reference)
    if isinstance(reference, str):
        try:
            return ("index", int(reference))
        except ValueError:
            return ("id", reference)
    raise TypeError(f"Unsupported fact reference: {reference!r}")


def load_required_fact_texts(
    facts_path: Path, required: set[tuple[str, Any]]
) -> dict[tuple[str, Any], str]:
    texts: dict[tuple[str, Any], str] = {}
    for ordinal, fact in enumerate(iter_jsonl(facts_path)):
        index_key = ("index", int(fact.get("fact_index", ordinal)))
        id_key = ("id", str(fact["fact_id"]))
        text = fact.get("canonical_fact_text") or fact.get("fact_text", "")
        if index_key in required:
            texts[index_key] = text
        if id_key in required:
            texts[id_key] = text
    missing = required - set(texts)
    if missing:
        preview = sorted(map(str, missing))[:10]
        raise KeyError(f"Missing {len(missing)} required facts; examples: {preview}")
    return texts


def load_predictions(path: Path, expected_ids: set[str]) -> dict[str, list[Any]]:
    predictions: dict[str, list[Any]] = {}
    for row in iter_jsonl(path):
        qid = str(row["question_id"])
        if qid in predictions:
            raise ValueError(f"Duplicate question_id in {path}: {qid}")
        ranking = row.get("ranked_fact_indices")
        if ranking is None:
            ranking = row.get("ranked_fact_ids")
        if not isinstance(ranking, list):
            raise ValueError(f"Missing ranking for {qid} in {path}")
        if len(ranking) != len({str(item) for item in ranking}):
            raise ValueError(f"Duplicate ranking entry for {qid} in {path}")
        predictions[qid] = ranking
    actual_ids = set(predictions)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Question IDs differ in {path}: missing={len(expected_ids-actual_ids)}, "
            f"extra={len(actual_ids-expected_ids)}")
    return predictions


def score_matching(
    anchor_refs: list[Any],
    retrieved_refs: list[Any],
    embed,
    cosine,
    tau: float,
) -> tuple[float, int]:
    if not anchor_refs or not retrieved_refs:
        return 0.0, 0
    anchor_embeddings = [embed(normalize_reference(ref)) for ref in anchor_refs]
    retrieved_embeddings = [embed(normalize_reference(ref)) for ref in retrieved_refs]
    weights = np.zeros((len(anchor_refs), len(retrieved_refs)), dtype=np.float64)
    for i, anchor_embedding in enumerate(anchor_embeddings):
        for j, retrieved_embedding in enumerate(retrieved_embeddings):
            similarity = cosine(anchor_embedding, retrieved_embedding)
            if similarity > tau:
                weights[i, j] = similarity
    row_indices, column_indices = linear_sum_assignment(weights, maximize=True)
    selected = weights[row_indices, column_indices]
    return float(selected.sum()), int(np.count_nonzero(selected > 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--evi-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    experiments_root = args.experiments_root.resolve()
    evi_root = args.evi_root.resolve()
    labels_path = args.labels.resolve()
    facts_path = evi_root / "data" / "facts" / "facts.jsonl"
    config_path = evi_root / "construction_config.json"
    tools_path = evi_root / "tools"
    for path in (experiments_root, facts_path, config_path, labels_path, tools_path):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(tools_path))
    from src.encoder import FrozenEncoder  # pylint: disable=import-error

    labels = list(iter_jsonl(labels_path))
    label_by_id = {str(row["question_id"]): row for row in labels}
    if len(label_by_id) != len(labels):
        raise ValueError("Duplicate question_id in labels")
    expected_ids = set(label_by_id)
    result_dirs = discover_result_dirs(experiments_root)

    predictions_by_dir: dict[Path, dict[str, list[Any]]] = {}
    required_references: set[tuple[str, Any]] = set()
    for label in labels:
        anchors = list(label.get("anchor_fact_indices", label.get("anchor_fact_ids", [])))
        if not anchors:
            raise ValueError(f"No anchors for {label['question_id']}")
        required_references.update(normalize_reference(ref) for ref in anchors)
    for directory in result_dirs:
        predictions = load_predictions(directory / "predictions.jsonl", expected_ids)
        predictions_by_dir[directory] = predictions
        for ranking in predictions.values():
            required_references.update(normalize_reference(ref) for ref in ranking[:10])

    config = read_json(config_path)
    tau = float(config.get("evaluation", {}).get("bs_tau", 0.0))
    encoder = FrozenEncoder(
        {"encoder": config.get("hdm", {}).get("encoder", {})}, facts_path)
    fact_texts = load_required_fact_texts(facts_path, required_references)
    embedding_cache: dict[tuple[str, Any], dict[int, float]] = {}

    def embed(reference: tuple[str, Any]) -> dict[int, float]:
        if reference not in embedding_cache:
            embedding_cache[reference] = encoder.embed(fact_texts[reference])
        return embedding_cache[reference]

    labels_sha256 = sha256_file(labels_path)
    print(
        f"[ready] questions={len(labels)} result_dirs={len(result_dirs)} "
        f"required_facts={len(required_references)} tau={tau}",
        flush=True,
    )

    for position, directory in enumerate(result_dirs, 1):
        relative_directory = directory.relative_to(experiments_root).as_posix()
        predictions = predictions_by_dir[directory]
        existing = {
            1: read_json(directory / "scores_k1.json"),
            10: read_json(directory / "scores_k10.json"),
        }
        existing_per_question = {
            k: {str(row["question_id"]): row for row in existing[k]["per_question"]}
            for k in (1, 10)
        }
        nbs_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        nbs_totals = {1: 0.0, 10: 0.0}
        coverage_totals = {1: 0.0, 10: 0.0}
        max_bs_error = 0.0
        max_bound_error = 0.0

        for label in labels:
            qid = str(label["question_id"])
            anchors = list(label.get("anchor_fact_indices", label.get("anchor_fact_ids", [])))
            anchor_count = len(anchors)
            ranking = predictions[qid]
            nbs_row: dict[str, Any] = {"question_id": qid, "anchor_count": anchor_count}
            coverage_row: dict[str, Any] = {
                "question_id": qid,
                "anchor_count": anchor_count,
            }
            for k in (1, 10):
                retrieved = list(dict.fromkeys(ranking))[:k]
                matching_sum, matched_count = score_matching(
                    anchors, retrieved, embed, encoder.cosine, tau)
                bs = matching_sum / anchor_count
                capacity = min(k, anchor_count)
                nbs = matching_sum / capacity if capacity else 0.0
                corrected_coverage = matched_count / anchor_count
                old_row = existing_per_question[k][qid]
                old_bs = float(old_row["bs"])
                old_coverage = float(old_row["coverage"])
                max_bs_error = max(max_bs_error, abs(bs - old_bs))
                bound_error = max(
                    0.0,
                    old_bs - corrected_coverage,
                    corrected_coverage - old_coverage,
                )
                max_bound_error = max(max_bound_error, bound_error)
                nbs_row[f"nbs_at_{k}"] = nbs
                nbs_row[f"matching_sum_at_{k}"] = matching_sum
                coverage_row[f"coverage_at_{k}"] = corrected_coverage
                coverage_row[f"matched_anchor_count_at_{k}"] = matched_count
                coverage_row[f"previous_coverage_at_{k}"] = old_coverage
                nbs_totals[k] += nbs
                coverage_totals[k] += corrected_coverage

            nbs_rows.append(nbs_row)
            coverage_rows.append(coverage_row)

        if max_bs_error > 1e-9:
            raise AssertionError(
                f"BS replay mismatch in {directory}: max_error={max_bs_error}")
        if max_bound_error > 1e-12:
            raise AssertionError(
                f"Coverage bound violation in {directory}: {max_bound_error}")

        common_metadata = {
            "result_directory": relative_directory,
            "questions": len(labels),
            "labels_sha256": labels_sha256,
            "encoder_revision": encoder.revision,
            "bs_tau": tau,
            "matching": "scipy_linear_sum_assignment_max_weight_one_to_one",
            "source_predictions": "predictions.jsonl",
            "source_scores": {"1": "scores_k1.json", "10": "scores_k10.json"},
        }
        nbs_output = {
            "metric": "capacity_normalized_bs",
            "definition": "nBS@K(q) = S_K(q) / min(K, |F*(q)|)",
            **common_metadata,
            "summary": {
                "nbs_at_1": nbs_totals[1] / len(labels),
                "nbs_at_10": nbs_totals[10] / len(labels),
            },
            "audit": {"max_replayed_bs_absolute_error": max_bs_error},
            "per_question": nbs_rows,
        }
        coverage_output = {
            "metric": "coverage_corrected",
            "definition": (
                "Cov@K(q) = positive edges selected by the optimal one-to-one "
                "matching / |F*(q)|"
            ),
            **common_metadata,
            "summary": {
                "coverage_at_1": coverage_totals[1] / len(labels),
                "coverage_at_10": coverage_totals[10] / len(labels),
                "previous_coverage_at_1": float(existing[1]["coverage"]),
                "previous_coverage_at_10": float(existing[10]["coverage"]),
                "delta_at_1": (
                    coverage_totals[1] / len(labels) - float(existing[1]["coverage"])
                ),
                "delta_at_10": (
                    coverage_totals[10] / len(labels) - float(existing[10]["coverage"])
                ),
            },
            "audit": {
                "max_replayed_bs_absolute_error": max_bs_error,
                "max_coverage_bound_violation": max_bound_error,
            },
            "per_question": coverage_rows,
        }
        write_json_atomic(directory / "nbs_scores.json", nbs_output)
        write_json_atomic(
            directory / "coverage_corrected_scores.json", coverage_output)
        print(
            f"[{position:02d}/{len(result_dirs):02d}] {relative_directory} "
            f"nBS@1={nbs_output['summary']['nbs_at_1']:.6f} "
            f"nBS@10={nbs_output['summary']['nbs_at_10']:.6f} "
            f"Cov@1={coverage_output['summary']['coverage_at_1']:.6f} "
            f"Cov@10={coverage_output['summary']['coverage_at_10']:.6f}",
            flush=True,
        )

    print(f"[success] wrote 2 files in each of {len(result_dirs)} result directories")


if __name__ == "__main__":
    main()
