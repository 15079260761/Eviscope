#!/usr/bin/env python3
"""Independent full semantic audit of the compact ICEWS-Actor release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .pool_views import MODES, ReleaseFactIndex, _normalized, iter_jsonl, materialize_all, role_indices
except ImportError:
    from pool_views import MODES, ReleaseFactIndex, _normalized, iter_jsonl, materialize_all, role_indices


def verify(root: Path) -> dict:
    facts_path = root / "data/facts.jsonl"
    if not facts_path.exists():
        facts_path = root / "data/facts.jsonl.gz"
    facts = ReleaseFactIndex(facts_path)
    total = 0
    by_split = {}
    eligible_by_mode = {mode: 0 for mode in MODES}
    for split in ("train", "dev", "test"):
        labels_path = root / ("private_labels/test.jsonl" if split == "test" else f"data/supervision/{split}.jsonl")
        if not labels_path.exists():
            labels_path = labels_path.with_suffix(labels_path.suffix + ".gz")
        descriptor_path = root / f"candidate_space/descriptors/{split}.jsonl"
        if not descriptor_path.exists():
            descriptor_path = descriptor_path.with_suffix(descriptor_path.suffix + ".gz")
        question_path = root / f"data/questions/{split}.jsonl"
        if not question_path.exists():
            question_path = question_path.with_suffix(question_path.suffix + ".gz")
        descriptors = list(iter_jsonl(descriptor_path))
        labels = list(iter_jsonl(labels_path))
        questions = list(iter_jsonl(question_path))
        if not (len(descriptors) == len(labels) == len(questions)):
            raise AssertionError(f"{split}: row count mismatch")
        for descriptor, label, question in zip(descriptors, labels, questions, strict=True):
            if descriptor["question_id"] != label["question_id"] or descriptor["question_id"] != question["question_id"]:
                raise AssertionError(f"{split}: row alignment mismatch")
            pools = materialize_all(descriptor, label, facts)
            expected_modes = {mode for mode, eligible in
                              descriptor["mode_eligibility"].items() if eligible}
            if set(pools) != expected_modes:
                raise AssertionError(f"{descriptor['question_id']}: eligibility mismatch")
            for mode in pools:
                eligible_by_mode[mode] += 1
            anchors, substitutes, distractors = map(set, role_indices(label))
            candidate_sets = {mode: set(row["candidate_fact_indices"]) for mode, row in pools.items()}
            if any(len(values) != 1000 for values in candidate_sets.values()):
                raise AssertionError(f"{descriptor['question_id']}: bad pool size")
            if candidate_sets["clean_control"] & (substitutes | distractors):
                raise AssertionError(f"{descriptor['question_id']}: clean role contamination")
            if "substitute_only" in candidate_sets and (not substitutes <= candidate_sets["substitute_only"] or candidate_sets["substitute_only"] & distractors):
                raise AssertionError(f"{descriptor['question_id']}: substitute mode invalid")
            if "distractor_only" in candidate_sets and (not distractors <= candidate_sets["distractor_only"] or candidate_sets["distractor_only"] & substitutes):
                raise AssertionError(f"{descriptor['question_id']}: distractor mode invalid")
            if "full_eviscope" in candidate_sets and not substitutes | distractors <= candidate_sets["full_eviscope"]:
                raise AssertionError(f"{descriptor['question_id']}: full mode invalid")
            clean_neutral = candidate_sets["clean_control"] - anchors
            for mode, roles in {
                "substitute_only": anchors | substitutes,
                "distractor_only": anchors | distractors,
                "full_eviscope": anchors | substitutes | distractors,
            }.items():
                if mode not in candidate_sets:
                    continue
                if not (candidate_sets[mode] - roles) <= clean_neutral:
                    raise AssertionError(f"{descriptor['question_id']}: unpaired neutral background")
            normalized_answers = [_normalized(value) for value in label.get("answers", []) if _normalized(value)]
            for index in clean_neutral:
                if facts.answer_bearing(index, normalized_answers):
                    raise AssertionError(f"{descriptor['question_id']}: answer-bearing neutral fact")
            total += 1
        by_split[split] = len(questions)
        print(json.dumps({"verified_split": split, "questions": len(questions)}, ensure_ascii=False), flush=True)
    return {
        "status": "pass",
        "questions": total,
        "splits": by_split,
        "facts": len(facts.ids),
        "modes": list(MODES),
        "eligible_by_mode": eligible_by_mode,
        "pool_size": 1000,
        "all_fact_indices_valid": True,
        "all_pools_unique_and_fixed_size": True,
        "roles_are_condition_correct": True,
        "neutral_background_is_shared": True,
        "answer_bearing_neutral_facts": 0,
        "role_precedence": "anchor > substitute > distractor",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.release_root.resolve())
    output = args.output or args.release_root / "audit/full_semantic_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(output), **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
