"""Prepare the two published SubGTR configurations for paired evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eviscope_adapter import prepare_questions


def deterministic(value: Any) -> Any:
    if isinstance(value, set):
        return sorted((deterministic(x) for x in value), key=str)
    if isinstance(value, list):
        return [deterministic(x) for x in value]
    if isinstance(value, tuple):
        return tuple(deterministic(x) for x in value)
    if isinstance(value, dict):
        return {key: deterministic(item) for key, item in value.items()}
    return value


def surface(row: dict[str, Any]) -> str:
    text = row.get("question") or (row.get("paraphrases") or [""])[0]
    return " ".join(str(text).lower().split())


def family(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(sorted(map(str, row.get("entities", [])))),
        tuple(sorted(map(str, row.get("relations", [])))),
        tuple(sorted(map(str, row.get("times", [])))),
        str(row.get("type")), str(row.get("answer_type")),
    )


def stable_key(row: dict[str, Any], seed: int, split: str) -> str:
    raw = f"{seed}|{split}|{row.get('uniq_id')}|{surface(row)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sample(rows: list[dict[str, Any]], limit: int, seed: int,
           split: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("type")), str(row.get("answer_type")))].append(row)
    for group in groups.values():
        group.sort(key=lambda row: stable_key(row, seed, split))
    exact = {key: len(group) * limit / len(rows) for key, group in groups.items()}
    quota = {key: int(value) for key, value in exact.items()}
    remainder = limit - sum(quota.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - quota[key]), key))
    for key in order[:remainder]:
        quota[key] += 1
    selected = [row for key, group in groups.items() for row in group[:quota[key]]]
    selected.sort(key=lambda row: stable_key(row, seed, split))
    if len(selected) != limit:
        raise ValueError(f"{split}: expected {limit}, selected {len(selected)}")
    return selected


def load(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        return pickle.load(stream)


def dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(rows, stream, protocol=4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--orig-root", type=Path, required=True)
    p.add_argument("--evi-root", type=Path, required=True)
    p.add_argument("--complex-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("eviscope_data"))
    p.add_argument("--train-limit", type=int, default=10_000)
    p.add_argument("--dev-limit", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=20260927)
    args = p.parse_args()

    standard = prepare_questions(args.orig_root, args.evi_root, args.output_root,
                                 args.train_limit, args.dev_limit, args.seed)
    standard_test = load(args.output_root / "wikidata_big/questions/test_eviscope.pickle")
    test_surfaces = {surface(row) for row in standard_test}
    test_families = {family(row) for row in standard_test}

    source_train = load(args.complex_root / "train.pickle")
    source_valid = load(args.complex_root / "valid.pickle")
    eligible: dict[str, list[dict[str, Any]]] = {}
    excluded: dict[str, dict[str, int]] = {}
    for split, source in (("train", source_train), ("dev", source_valid)):
        kept, surface_hits, family_hits = [], 0, 0
        for original in source:
            s_hit = surface(original) in test_surfaces
            f_hit = family(original) in test_families
            surface_hits += int(s_hit)
            family_hits += int(f_hit)
            if s_hit or f_hit:
                continue
            row = deterministic(dict(original))
            row["_eviscope_source_split"] = split
            row["_eviscope_complex_source_key"] = stable_key(row, args.seed, split)
            kept.append(row)
        eligible[split] = kept
        excluded[split] = {"surface_overlap": surface_hits,
                           "semantic_family_overlap": family_hits,
                           "eligible": len(kept)}
    if len(eligible["train"]) < args.train_limit or len(eligible["dev"]) < args.dev_limit:
        raise SystemExit(f"Insufficient leakage-safe complex data: {excluded}")
    complex_train = sample(eligible["train"], args.train_limit, args.seed, "complex_train")
    train_surface = {surface(row) for row in complex_train}
    train_family = {family(row) for row in complex_train}
    safe_dev = [row for row in eligible["dev"]
                if surface(row) not in train_surface and family(row) not in train_family]
    if len(safe_dev) < args.dev_limit:
        raise SystemExit(f"Only {len(safe_dev)} complex dev rows remain after train isolation")
    complex_dev = sample(safe_dev, args.dev_limit, args.seed, "complex_dev")
    # Both published SubGTR configurations face exactly the same unlabeled Test cohort.
    complex_test = [deterministic(dict(row)) for row in standard_test]
    out = args.output_root / "wikidata_big_complex/questions"
    dump(out / "train_10k.pickle", complex_train)
    dump(out / "valid_2k.pickle", complex_dev)
    dump(out / "test_eviscope.pickle", complex_test)

    complex_report = {
        "configuration": "wikidata_big_complex",
        "training_source": "SubGTR ComplexCronQuestions/train.pickle",
        "validation_source": "SubGTR ComplexCronQuestions/valid.pickle",
        "test_source": "shared paired EviScope CronQuestions Test cohort",
        "test_gold_in_model_input": False,
        "source_counts": {"train": len(source_train), "dev": len(source_valid)},
        "excluded_against_formal_test": excluded,
        "splits": {
            "train": {"prepared": len(complex_train),
                      "type_counts": dict(Counter(str(x.get("type")) for x in complex_train))},
            "dev": {"prepared": len(complex_dev),
                    "type_counts": dict(Counter(str(x.get("type")) for x in complex_dev))},
            "test": {"prepared": len(complex_test)},
        },
    }
    manifest = {"protocol": "subgtr_standard_and_complex_paired_clean_full_v1",
                "seed": args.seed, "standard": standard, "complex": complex_report}
    (args.output_root / "subgtr_preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
