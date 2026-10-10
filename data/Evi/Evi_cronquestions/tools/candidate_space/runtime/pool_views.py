"""Deterministic lazy candidate pools for compact EviScope-CronQuestions.

The release never stores per-question neutral fact lists. A descriptor,
frozen fact catalog and versioned algorithm reconstruct the same neutral
reservoir on every machine. Trusted orchestration combines that reservoir
with private evidence roles and exposes only an unlabeled candidate list.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from .common import canonical_json, iter_jsonl, sha256_file, sha256_text


MODES = ("clean", "substitute_only", "distractor_only", "full")
POLICY_VERSION = "lazy_neutral_v1"


def _unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _year(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None


class ReleaseFactIndex:
    """Small runtime index over the single released fact catalog."""

    def __init__(self, facts_path: Path) -> None:
        self.facts_path = facts_path
        self.ids: list[str] = []
        self.by_year: dict[int, list[int]] = {}
        for ordinal, fact in enumerate(iter_jsonl(facts_path)):
            index = int(fact.get("fact_index", ordinal))
            if index != ordinal:
                raise ValueError(
                    f"fact_index must be dense and ordered: row={ordinal}, value={index}")
            self.ids.append(str(fact["fact_id"]))
            start, end = _year(fact.get("time_start")), _year(fact.get("time_end"))
            if start is None or end is None or end < start or end - start > 500:
                continue
            for year in range(start, end + 1):
                self.by_year.setdefault(year, []).append(index)
        self.catalog_sha256 = sha256_file(facts_path)


def role_indices(labels: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    """Read the compact release supervision schema."""
    return (
        _unique(int(x) for x in labels.get("anchor_fact_indices", [])),
        _unique(int(x) for x in labels.get("substitute_fact_indices", [])),
        _unique(int(x) for x in labels.get("distractor_fact_indices", [])),
    )


def _neutral_reservoir(descriptor: dict[str, Any],
                       roles: tuple[list[int], list[int], list[int]],
                       facts: ReleaseFactIndex) -> list[int]:
    anchors, substitutes, distractors = roles
    blocked = set(anchors) | set(substitutes) | set(distractors)
    target = max(0, min(int(descriptor["pool_size"]), len(facts.ids)) - len(anchors))
    years = sorted(set(int(y) for y in descriptor.get("background_years", [])))
    background = _unique(index for year in years for index in facts.by_year.get(year, []))
    universe = background if background else list(range(len(facts.ids)))
    rng = random.Random(int(descriptor["sampling_seed"]))
    selected: list[int] = []
    seen: set[int] = set()
    attempts = 0
    while len(selected) < target and universe and attempts < target * 50:
        index = universe[rng.randrange(len(universe))]
        attempts += 1
        if index not in blocked and index not in seen:
            selected.append(index)
            seen.add(index)
    if len(selected) < target:
        offset = rng.randrange(len(facts.ids)) if facts.ids else 0
        for step in range(len(facts.ids)):
            index = (offset + step) % len(facts.ids)
            if index not in blocked and index not in seen:
                selected.append(index)
                seen.add(index)
                if len(selected) == target:
                    break
    if len(selected) != target:
        raise ValueError(
            f"{descriptor.get('paired_id')}: only {len(selected)} neutral facts, "
            f"expected {target}")
    return selected


def materialize_pool(descriptor: dict[str, Any], labels: dict[str, Any], mode: str,
                     facts: ReleaseFactIndex, emit_fact_ids: bool = False) -> dict[str, Any]:
    """Materialize one fixed-size, unlabeled pool from compact release files."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
    if descriptor.get("policy_version") != POLICY_VERSION:
        raise ValueError(f"Unsupported policy: {descriptor.get('policy_version')}")
    if descriptor.get("paired_id") != labels.get("paired_id"):
        raise ValueError("Descriptor/supervision paired_id mismatch")
    if descriptor.get("fact_catalog_sha256") != facts.catalog_sha256:
        raise ValueError("Fact catalog hash mismatch; refusing non-reproducible pool")
    if int(descriptor.get("fact_universe_size", -1)) != len(facts.ids):
        raise ValueError("Fact universe size mismatch")

    anchors, all_substitutes, all_distractors = role_indices(labels)
    for index in anchors + all_substitutes + all_distractors:
        if index < 0 or index >= len(facts.ids):
            raise IndexError(f"Fact index outside released catalog: {index}")
    eligibility = {
        "clean": bool(anchors),
        "substitute_only": bool(anchors and all_substitutes),
        "distractor_only": bool(anchors and all_distractors),
        "full": bool(anchors and all_substitutes and all_distractors),
    }
    if descriptor.get("mode_eligibility") != eligibility:
        raise ValueError("Descriptor/supervision mode eligibility mismatch")
    if not eligibility[mode]:
        raise ValueError(f"{descriptor.get('paired_id')}: mode {mode!r} is not eligible")
    max_subs = int(descriptor.get("max_substitutes", len(all_substitutes)))
    max_dis = int(descriptor.get("max_distractors", len(all_distractors)))
    substitutes = all_substitutes[:max_subs] \
        if mode in ("substitute_only", "full") else []
    distractors = all_distractors[:max_dis] \
        if mode in ("distractor_only", "full") else []
    roles = _unique(anchors + substitutes + distractors)
    pool_size = min(int(descriptor["pool_size"]), len(facts.ids))
    roles = roles[:pool_size]
    neutral = _neutral_reservoir(
        descriptor, (anchors, all_substitutes, all_distractors), facts)
    chosen = roles + neutral[:pool_size - len(roles)]
    if len(chosen) != pool_size or len(set(chosen)) != pool_size:
        raise ValueError(f"{descriptor.get('paired_id')}: invalid materialized pool")
    seed = int(descriptor["sampling_seed"])
    qid = str(descriptor["paired_id"])
    chosen.sort(key=lambda index: sha256_text(f"{seed}|{qid}|{index}"))
    result = {
        "candidate_pool_id": f"{descriptor['pool_descriptor_id']}:{mode}",
        "pool_descriptor_id": descriptor["pool_descriptor_id"],
        "paired_id": qid,
        "benchmark_split": descriptor["benchmark_split"],
        "pool_mode": mode,
        "pool_size": len(chosen),
        "candidate_fact_indices": chosen,
        "policy_version": POLICY_VERSION,
    }
    if emit_fact_ids:
        result["candidate_fact_ids"] = [facts.ids[index] for index in chosen]
    return result


def _paired_rows(descriptors: Path, labels: Path) \
        -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    left, right = iter_jsonl(descriptors), iter_jsonl(labels)
    while True:
        try:
            descriptor = next(left)
        except StopIteration:
            try:
                next(right)
            except StopIteration:
                return
            raise ValueError("Supervision contains more rows than descriptors")
        try:
            label = next(right)
        except StopIteration as exc:
            raise ValueError("Descriptors contain more rows than supervision") from exc
        yield descriptor, label


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Materialize compact CronQuestions pools")
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True,
                        help="trusted supervision/private-label file")
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-fact-ids", action="store_true")
    args = parser.parse_args(argv)

    facts = ReleaseFactIndex(args.facts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = skipped = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for descriptor, labels in _paired_rows(args.descriptors, args.labels):
            anchors, substitutes, distractors = role_indices(labels)
            available = {
                "clean": bool(anchors),
                "substitute_only": bool(anchors and substitutes),
                "distractor_only": bool(anchors and distractors),
                "full": bool(anchors and substitutes and distractors),
            }
            if not available[args.mode]:
                skipped += 1
                continue
            row = materialize_pool(descriptor, labels, args.mode, facts,
                                   emit_fact_ids=args.emit_fact_ids)
            handle.write(canonical_json(row) + "\n")
            count += 1
    print(json.dumps({"mode": args.mode, "materialized": count,
                      "ineligible_skipped": skipped, "output": str(args.output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
