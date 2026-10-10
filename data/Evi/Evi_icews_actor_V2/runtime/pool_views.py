#!/usr/bin/env python3
"""Deterministic lazy candidate pools for compact EviScope-ICEWS-Actor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator


MODES = ("clean_control", "substitute_only", "distractor_only", "full_eviscope")
POLICY_VERSION = "icews_lazy_neutral_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from exc


def _year(value: Any) -> int | None:
    match = re.match(r"^\s*(-?\d{1,4})", str(value or ""))
    return int(match.group(1)) if match else None


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


class ReleaseFactIndex:
    """Runtime index over the release's single, dense fact catalog."""

    def __init__(self, facts_path: Path) -> None:
        self.path = facts_path
        self.ids: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self.answer_fields: list[set[str]] = []
        self.answer_texts: list[str] = []
        self.by_year: dict[int, list[int]] = {}
        for ordinal, fact in enumerate(iter_jsonl(facts_path)):
            index = int(fact.get("fact_index", -1))
            if index != ordinal:
                raise ValueError(f"fact_index is not dense at row {ordinal}: {index}")
            self.ids.append(str(fact["fact_id"]))
            self.rows.append(fact)
            self.answer_fields.append({
                _normalized(fact.get(key)) for key in (
                    "subject_id", "subject_name", "object_id", "object_name",
                    "relation_id", "relation_name", "time_start", "time_end",
                    "raw_time_start", "raw_time_end",
                )
            })
            self.answer_texts.append(_normalized(
                f"{fact.get('fact_text', '')} {fact.get('canonical_fact_text', '')}"
            ))
            years = {
                year for year in (
                    _year(fact.get("time_start")), _year(fact.get("time_end")),
                    _year(fact.get("raw_time_start")), _year(fact.get("raw_time_end")),
                ) if year is not None
            }
            for year in years:
                self.by_year.setdefault(year, []).append(index)
        self.catalog_sha256 = sha256_file(facts_path)

    def answer_bearing(self, index: int, normalized_answers: list[str]) -> bool:
        fields = self.answer_fields[index]
        text = self.answer_texts[index]
        for answer in normalized_answers:
            if answer in fields:
                return True
            if len(answer) >= 4 and answer in text:
                if re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text):
                    return True
        return False


def role_indices(labels: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    unique = lambda xs: list(dict.fromkeys(int(x) for x in xs))
    anchors = unique(labels.get("anchor_fact_indices", []))
    substitutes = unique(item["fact_index"] for item in labels.get("substitutable_evidence", []))
    distractors = unique(item["fact_index"] for item in labels.get("distracting_evidence", []))
    return anchors, substitutes, distractors


def answer_bearing(fact: dict[str, Any], answers: Iterable[Any]) -> bool:
    fields = {
        _normalized(fact.get(key)) for key in (
            "subject_id", "subject_name", "object_id", "object_name", "relation_id",
            "relation_name", "time_start", "time_end", "raw_time_start", "raw_time_end",
        )
    }
    text = _normalized(f"{fact.get('fact_text', '')} {fact.get('canonical_fact_text', '')}")
    for raw in answers:
        answer = _normalized(raw)
        if not answer:
            continue
        if answer in fields:
            return True
        if len(answer) >= 4 and re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text):
            return True
    return False


def _permutation(size: int, seed: int) -> Iterator[int]:
    if size <= 0:
        return
    start = seed % size
    step = ((seed >> 17) % size) or 1
    while math.gcd(step, size) != 1:
        step = (step + 1) % size or 1
    for offset in range(size):
        yield (start + offset * step) % size


def neutral_reservoir(descriptor: dict[str, Any], labels: dict[str, Any],
                      facts: ReleaseFactIndex) -> list[int]:
    """Build one shared neutral sequence; every mode consumes a prefix of it."""
    anchors, substitutes, distractors = role_indices(labels)
    blocked = set(anchors) | set(substitutes) | set(distractors)
    target = min(int(descriptor["pool_size"]), len(facts.ids)) - len(anchors)
    answers = [_normalized(value) for value in labels.get("answers", []) if _normalized(value)]
    selected: list[int] = []
    seen: set[int] = set()

    def accept(index: int) -> None:
        if index in seen or index in blocked:
            return
        if facts.answer_bearing(index, answers):
            return
        seen.add(index)
        selected.append(index)

    years = sorted(set(int(year) for year in descriptor.get("background_years", [])))
    local = list(dict.fromkeys(index for year in years for index in facts.by_year.get(year, [])))
    for position in _permutation(len(local), stable_int(descriptor["sampling_seed"], "year")):
        accept(local[position])
        if len(selected) == target:
            return selected
    for index in _permutation(len(facts.ids), stable_int(descriptor["sampling_seed"], "global")):
        accept(index)
        if len(selected) == target:
            return selected
    raise ValueError(f"{descriptor.get('paired_id')}: only {len(selected)}/{target} neutral facts")


def _validate(descriptor: dict[str, Any], labels: dict[str, Any],
              facts: ReleaseFactIndex) -> tuple[list[int], list[int], list[int]]:
    if descriptor.get("policy_version") != POLICY_VERSION:
        raise ValueError(f"Unsupported policy: {descriptor.get('policy_version')}")
    if descriptor.get("paired_id") != labels.get("paired_id"):
        raise ValueError("Descriptor/labels paired_id mismatch")
    if descriptor.get("fact_catalog_sha256") != facts.catalog_sha256:
        raise ValueError("Fact catalog checksum mismatch")
    if int(descriptor.get("fact_universe_size", -1)) != len(facts.ids):
        raise ValueError("Fact universe size mismatch")
    roles = role_indices(labels)
    anchors, substitutes, distractors = roles
    if set(anchors) & set(substitutes) or set(anchors) & set(distractors) or set(substitutes) & set(distractors):
        raise ValueError("Evidence roles are not disjoint")
    if any(index < 0 or index >= len(facts.ids) for group in roles for index in group):
        raise IndexError("Evidence role references a fact outside the catalog")
    expected = {
        "clean_control": bool(anchors),
        "substitute_only": bool(anchors and substitutes),
        "distractor_only": bool(anchors and distractors),
        "full_eviscope": bool(anchors and substitutes and distractors),
    }
    if descriptor.get("mode_eligibility") != expected:
        raise ValueError("Descriptor mode_eligibility mismatch")
    return roles


def materialize_all(descriptor: dict[str, Any], labels: dict[str, Any],
                    facts: ReleaseFactIndex, emit_fact_ids: bool = False) -> dict[str, dict[str, Any]]:
    anchors, substitutes, distractors = _validate(descriptor, labels, facts)
    neutral = neutral_reservoir(descriptor, labels, facts)
    additions = {
        "clean_control": [],
        "substitute_only": substitutes,
        "distractor_only": distractors,
        "full_eviscope": substitutes + distractors,
    }
    pool_size = min(int(descriptor["pool_size"]), len(facts.ids))
    result: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        if not descriptor["mode_eligibility"][mode]:
            continue
        roles = list(dict.fromkeys(anchors + additions[mode]))
        if len(roles) > pool_size:
            raise ValueError(f"{descriptor['paired_id']}/{mode}: roles exceed pool size")
        candidates = roles + neutral[:pool_size - len(roles)]
        if len(candidates) != pool_size or len(set(candidates)) != pool_size:
            raise ValueError(f"{descriptor['paired_id']}/{mode}: invalid fixed-size pool")
        candidates.sort(key=lambda index: stable_int(descriptor["sampling_seed"], index))
        row = {
            "candidate_pool_id": f"{descriptor['pool_descriptor_id']}:{mode}",
            "pool_descriptor_id": descriptor["pool_descriptor_id"],
            "paired_id": descriptor["paired_id"],
            "benchmark_split": descriptor["benchmark_split"],
            "pool_mode": mode,
            "pool_size": len(candidates),
            "candidate_fact_indices": candidates,
            "policy_version": POLICY_VERSION,
        }
        if emit_fact_ids:
            row["candidate_fact_ids"] = [facts.ids[index] for index in candidates]
        result[mode] = row
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Materialize a compact ICEWS-Actor pool view")
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-fact-ids", action="store_true")
    args = parser.parse_args(argv)
    facts = ReleaseFactIndex(args.facts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    descriptors = iter_jsonl(args.descriptors)
    labels = iter_jsonl(args.labels)
    count = 0
    with opener(args.output, "wt", encoding="utf-8", newline="\n") as handle:
        for descriptor, label in zip(descriptors, labels, strict=True):
            rows = materialize_all(descriptor, label, facts, args.emit_fact_ids)
            if args.mode in rows:
                handle.write(canonical_json(rows[args.mode]) + "\n")
                count += 1
    print(canonical_json({"mode": args.mode, "materialized": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
