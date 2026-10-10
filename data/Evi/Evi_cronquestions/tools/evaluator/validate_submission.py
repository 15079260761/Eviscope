"""Submission validator (spec section 17/22): fixed input format, no
duplicate or unknown fact ids, every question present.  Rankings longer than
K are allowed — the evaluator auto-truncates to K."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import iter_jsonl


def validate(labels_path: Path, submission_path: Path, facts_path: Path | None = None) -> dict:
    labels = {q["question_id"]: q for q in iter_jsonl(labels_path)}
    fact_ids = None
    fact_indices = None
    if facts_path is not None:
        fact_ids, fact_indices = set(), set()
        for index, fact in enumerate(iter_jsonl(facts_path)):
            fact_ids.add(fact["fact_id"])
            fact_indices.add(int(fact.get("fact_index", index)))
    errors = []
    seen = set()
    for line_number, row in enumerate(iter_jsonl(submission_path), 1):
        qid = row.get("question_id")
        if qid is None:
            errors.append(f"line {line_number}: missing question_id")
            continue
        if qid in seen:
            errors.append(f"line {line_number}: duplicate question_id {qid}")
        seen.add(qid)
        if qid not in labels:
            errors.append(f"line {line_number}: unknown question_id {qid}")
            continue
        ranking = row.get("ranked_fact_indices")
        reference_kind = "index"
        if ranking is None:
            ranking = row.get("ranked_fact_ids")
            reference_kind = "id"
        if not isinstance(ranking, list):
            errors.append(
                f"line {line_number}: ranked_fact_indices or ranked_fact_ids must be a list")
            continue
        if len(ranking) != len(set(ranking)):
            errors.append(f"line {line_number}: duplicate fact reference in ranking")
        if fact_ids is not None:
            universe = fact_indices if reference_kind == "index" else fact_ids
            for reference in ranking:
                if reference not in universe:
                    errors.append(
                        f"line {line_number}: unknown fact {reference_kind} {reference}")
    missing = set(labels) - seen
    if missing:
        errors.append(f"submission is missing {len(missing)} questions")
    return {"valid": not errors, "errors": errors[:100], "error_count": len(errors)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="EviScope submission validator")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--facts", type=Path, default=None)
    args = parser.parse_args(argv)
    result = validate(args.labels, args.submission, args.facts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
