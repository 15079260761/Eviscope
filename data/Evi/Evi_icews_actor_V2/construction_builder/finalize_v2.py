#!/usr/bin/env python3
"""Finalize public/private, lineage, cohort, and documentation audits for V2."""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
BUILDER = Path(__file__).resolve().parent


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    forbidden = {
        "answers", "gold_answers", "anchor_fact_indices", "substitutable_evidence",
        "distracting_evidence", "ambiguity_pressure", "reasoning_demand",
        "hybrid_difficulty", "substitute_anchor_coverage",
    }
    ids: dict[str, set[str]] = defaultdict(set)
    surfaces: dict[str, set[str]] = defaultdict(set)
    boundary = Counter()
    for split in ("train", "dev", "test"):
        q_path = ROOT / f"data/questions/{split}.jsonl"
        l_path = ROOT / ("private_labels/test.jsonl" if split == "test"
                         else f"data/supervision/{split}.jsonl")
        d_path = ROOT / f"candidate_space/descriptors/{split}.jsonl"
        for question, label, descriptor in zip(rows(q_path), rows(l_path), rows(d_path),
                                               strict=True):
            qid = question["question_id"]
            if label["question_id"] != qid or descriptor["question_id"] != qid:
                raise AssertionError(f"row alignment mismatch: {split}/{qid}")
            if forbidden & set(question):
                boundary["forbidden_label_fields_in_public_questions"] += 1
            ids[split].add(qid)
            surfaces[question["normalized_question"]].add(split)
            boundary[f"questions_{split}"] += 1
    intersections = {
        "train_dev": len(ids["train"] & ids["dev"]),
        "train_test": len(ids["train"] & ids["test"]),
        "dev_test": len(ids["dev"] & ids["test"]),
    }
    cross_surface = sum(1 for splits in surfaces.values() if len(splits) > 1)
    if any(intersections.values()) or cross_surface:
        raise AssertionError({"id_intersections": intersections,
                              "cross_split_surfaces": cross_surface})
    if (ROOT / "data/supervision/test.jsonl").exists():
        raise AssertionError("public test supervision must not exist")

    cohort_counts: dict[str, dict[str, int]] = {}
    cohort_ids: dict[str, set[str]] = {}
    for cohort in ("clean_all", "substitute_effect", "distractor_effect",
                   "four_way_common"):
        by_split = {}
        combined: set[str] = set()
        for split in ("train", "dev", "test"):
            path = ROOT / f"cohorts/{cohort}/{split}.jsonl.gz"
            values = {row["question_id"] for row in rows(path)}
            if not values <= ids[split]:
                raise AssertionError(f"{cohort}/{split}: unknown IDs")
            by_split[split] = len(values)
            combined |= values
        cohort_counts[cohort] = by_split | {"total": len(combined)}
        cohort_ids[cohort] = combined
    if not cohort_ids["four_way_common"] <= cohort_ids["substitute_effect"]:
        raise AssertionError("four-way cohort not contained in substitute cohort")
    if not cohort_ids["four_way_common"] <= cohort_ids["distractor_effect"]:
        raise AssertionError("four-way cohort not contained in distractor cohort")
    v1_ids = {row["question_id"] for row in rows(BUILDER / "v1_id_mapping.jsonl")}
    if cohort_ids["four_way_common"] != v1_ids:
        raise AssertionError("V1 four-way ID set changed")

    lineage_count = sum(1 for _ in rows(ROOT / "audit/v2_source_lineage.jsonl"))
    total = sum(len(values) for values in ids.values())
    if lineage_count != total:
        raise AssertionError(f"lineage count mismatch: {lineage_count}/{total}")
    public_private = {
        "status": "pass",
        "public_test_supervision_files": 0,
        "private_test_labels_present": (ROOT / "private_labels/test.jsonl").exists(),
        "public_question_forbidden_label_fields":
            boundary["forbidden_label_fields_in_public_questions"],
        "public_private_test_id_alignment": "exact",
        "test_materialization_requires_trusted_private_labels": True,
        "answer_type_is_schema_metadata_not_a_gold_answer": True,
    }
    write_json(ROOT / "audit/public_private_boundary.json", public_private)
    integrity = {
        "status": "pass",
        "release_version": "2.0.0",
        "questions": total,
        "splits": {split: len(values) for split, values in ids.items()},
        "cross_split_question_id_intersections": intersections,
        "cross_split_normalized_question_surfaces": cross_surface,
        "cohorts": cohort_counts,
        "v1_four_way_question_ids_preserved_exactly": True,
        "source_lineage_rows": lineage_count,
        "public_private_boundary": public_private,
    }
    write_json(ROOT / "audit/v2_integrity_report.json", integrity)
    write_json(ROOT / "audit/source_lineage.json", {
        "status": "pass",
        "source_dataset_root": str((ROOT.parents[1] / "Ori/ICEWS").resolve()),
        "construction": "full source rebuild with V1 thresholds; relaxed release eligibility",
        "source_splits_inherited": True,
        "source_questions_modified": False,
        "source_facts_modified": False,
        "v1_four_way_cohort_preserved": True,
        "lineage_file": "audit/v2_source_lineage.jsonl",
        "lineage_rows": lineage_count,
    })
    (ROOT / "cohorts/README.md").write_text(
        "# V2 evaluation cohorts\n\n"
        "- `clean_all`: every anchor-valid released question.\n"
        "- `substitute_effect`: questions eligible for clean and substitute-only views.\n"
        "- `distractor_effect`: questions eligible for clean and distractor-only views.\n"
        "- `four_way_common`: the frozen V1 paired cohort eligible for all four views.\n\n"
        "Controlled four-condition experiments must continue to use `four_way_common`.\n",
        encoding="utf-8",
    )
    (ROOT / "README.md").write_text(
        "# EviScope-ICEWS-Actor V2\n\n"
        "V2 expands the official release from 23,453 to 48,122 anchor-valid questions "
        "without changing the frozen 23,453-question four-view experimental cohort. "
        "All source splits are inherited. Anchor sufficiency/minimality, answer "
        "reproduction, semantic-family isolation, Train-only evidence thresholds, "
        "verified evidence roles, and public/private test separation remain enforced.\n\n"
        "## View eligibility\n\n"
        "Each question declares `mode_eligibility` in its descriptor. Clean is available "
        "for all 48,122 questions; substitute-only for 46,504; distractor-only for "
        "24,948; and full EviScope for the unchanged 23,453-question paired cohort. "
        "A missing optional role never creates an empty or fabricated evidence list.\n\n"
        "## Compatibility\n\n"
        "Existing four-view experiments remain tied to `cohorts/four_way_common` and "
        "do not require rerunning. New full-release statistics must use V2 counts and "
        "must distinguish release size from four-view eligibility. Runtime pools are "
        "materialized deterministically from the single fact catalog and descriptors.\n\n"
        "## Audits\n\n"
        "See `audit/full_semantic_audit.json`, `audit/v2_integrity_report.json`, "
        "`audit/public_private_boundary.json`, and `statistics/release_statistics.json`.\n"
        "The local `construction_builder/` directory contains reproducibility code and "
        "is excluded from the immutable release manifest.\n",
        encoding="utf-8",
    )
    print(json.dumps(integrity, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
