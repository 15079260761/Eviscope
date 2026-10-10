"""Shared, leakage-safe Table-2 protocol for CronKGQA and CTRN.

This module does not replace either original model.  It prepares a deterministic
10k training subset from the retained CronQuestions split and lifts each
model's native entity/time answer scores to the released candidate facts.
Only unlabeled candidate indices are visible while ranking; gold evidence is
read later by the official evaluator.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL {path}:{line_number}") from exc


def _stable_key(seed: int, split: str, row: dict[str, Any]) -> str:
    identity = row.get("paired_id") or row.get("question_id") or row.get("source_question_id")
    return hashlib.sha256(f"{seed}|{split}|{identity}".encode()).hexdigest()


def _deterministic_containers(value: Any) -> Any:
    """Remove hash-order dependence from legacy set-valued pickle fields."""
    if isinstance(value, set):
        return sorted((_deterministic_containers(item) for item in value), key=str)
    if isinstance(value, list):
        return [_deterministic_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deterministic_containers(item) for item in value)
    if isinstance(value, dict):
        return {key: _deterministic_containers(item) for key, item in value.items()}
    return value


def stratified_sample(rows: list[dict[str, Any]], limit: int, seed: int,
                      split: str) -> list[dict[str, Any]]:
    """Exact-size deterministic proportional sample by type and answer type."""
    if limit >= len(rows):
        return list(rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("type")), str(row.get("answer_type")))].append(row)
    for group in groups.values():
        group.sort(key=lambda row: _stable_key(seed, split, row))
    exact = {key: len(group) * limit / len(rows) for key, group in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = limit - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    selected = [row for key, group in groups.items() for row in group[:quotas[key]]]
    selected.sort(key=lambda row: _stable_key(seed, split, row))
    if len(selected) != limit:
        raise AssertionError(f"Expected {limit} sampled rows, got {len(selected)}")
    return selected


def _paired_release_rows(evi_root: Path, split: str):
    question_path = evi_root / "data" / "questions" / f"{split}.jsonl"
    labels_path = (evi_root / "data" / "private_labels" / "test.jsonl") \
        if split == "test" else (evi_root / "data" / "supervision" / f"{split}.jsonl")
    questions, labels = iter_jsonl(question_path), iter_jsonl(labels_path)
    while True:
        try:
            question = next(questions)
        except StopIteration:
            try:
                next(labels)
            except StopIteration:
                return
            raise ValueError(f"{split}: labels contain extra rows")
        try:
            label = next(labels)
        except StopIteration as exc:
            raise ValueError(f"{split}: questions contain extra rows") from exc
        if question.get("paired_id") != label.get("paired_id"):
            raise ValueError(f"{split}: question/label pairing mismatch")
        yield question, label


def _full_intersection_rows(evi_root: Path):
    """Yield the paired Test cohort eligible for both clean and full views."""
    descriptors = iter_jsonl(evi_root / "data" / "descriptors" / "test.jsonl")
    labels = iter_jsonl(evi_root / "data" / "private_labels" / "test.jsonl")
    for descriptor, label in zip(descriptors, labels, strict=True):
        if descriptor.get("paired_id") != label.get("paired_id"):
            raise ValueError("Test descriptor/private-label pairing mismatch")
        if descriptor.get("mode_eligibility", {}).get("full"):
            yield descriptor, label


def prepare_questions(orig_root: Path, evi_root: Path, output_root: Path,
                      train_limit: int = 10_000, dev_limit: int = 2_000,
                      seed: int = 20260926) -> dict[str, Any]:
    """Create paired model inputs without moving examples across source splits."""
    source_dir = orig_root / "wikidata_big" / "questions"
    output_dir = output_root / "wikidata_big" / "questions"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_names = {"train": "train", "dev": "valid", "test": "test"}
    limits = {"train": train_limit, "dev": dev_limit, "test": None}
    full_rows = list(_full_intersection_rows(evi_root))
    full_ids = {str(descriptor["paired_id"]) for descriptor, _ in full_rows}
    if not full_ids:
        raise ValueError("No Test questions are eligible for the paired clean/full protocol")
    manifest: dict[str, Any] = {
        "seed": seed, "train_limit": train_limit, "dev_limit": dev_limit,
        "source_split_inherited": True,
        "evaluation_protocol": "paired_clean_full_v1",
        "evaluation_modes": ["clean", "full"],
        "evaluation_cohort_size": len(full_ids),
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        with (source_dir / f"{source_names[split]}.pickle").open("rb") as stream:
            source = pickle.load(stream)
        by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source:
            key = str(row.get("uniq_id"))
            by_id[key].append(row)
        adapted: list[dict[str, Any]] = []
        mismatch_answers = 0
        for public, labels in _paired_release_rows(evi_root, split):
            if split == "test" and str(public["paired_id"]) not in full_ids:
                continue
            source_id = str(public["source_question_id"])
            if source_id not in by_id:
                raise KeyError(f"Missing source question {split}:{source_id}")
            released_answers = set(str(x) for x in labels.get("gold_answers", []))
            candidates = by_id[source_id]
            exact = [candidate for candidate in candidates
                     if public["question"] == candidate.get("question") or
                     public["question"] in candidate.get("paraphrases", [])]
            if exact:
                candidates = exact
            answer_matched = [candidate for candidate in candidates
                              if set(str(x) for x in candidate.get("answers", [])) == released_answers]
            if answer_matched:
                candidates = answer_matched
            type_matched = [candidate for candidate in candidates
                            if str(candidate.get("type")) == str(public.get("question_type")) and
                            str(candidate.get("answer_type")) == str(public.get("answer_type"))]
            if type_matched:
                candidates = type_matched
            if not candidates:
                raise KeyError(f"No source variant for {split}:{source_id}")
            # Duplicate uniq_id values denote question/template variants in
            # CronQuestions.  Exact surface text, gold answer set and type are
            # used in that order; remaining copies are semantically identical.
            row = _deterministic_containers(dict(candidates[0]))
            if set(str(x) for x in row.get("answers", [])) != released_answers:
                mismatch_answers += 1
            # The model input artifact carries no Test answer.  Test labels are
            # consumed only inside trusted pool assembly and official scoring.
            row["answers"] = [] if split == "test" else sorted(released_answers)
            # Both Orig and Evi views consume identical question text.  Only
            # the evidence candidate space changes at evaluation time.
            row["paraphrases"] = [public["question"]]
            row["_eviscope_question_id"] = public["question_id"]
            row["_eviscope_paired_id"] = public["paired_id"]
            row["_eviscope_source_split"] = split
            adapted.append(row)
        available = len(adapted)
        if limits[split] is not None:
            adapted = stratified_sample(adapted, int(limits[split]), seed, split)
        output_name = {"train": "train_10k", "dev": "valid_2k",
                       "test": "test_eviscope"}[split]
        with (output_dir / f"{output_name}.pickle").open("wb") as stream:
            pickle.dump(adapted, stream, protocol=4)
        manifest["splits"][split] = {
            "release_available": available,
            "prepared": len(adapted),
            "output": f"{output_name}.pickle",
            "answer_set_mismatches_reconciled_to_release": mismatch_answers,
            "type_counts": dict(Counter(str(row.get("type")) for row in adapted)),
            "answer_type_counts": dict(Counter(str(row.get("answer_type")) for row in adapted)),
        }
    manifest_path = output_root / "preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    eval_dir = output_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    labels_path = eval_dir / "test_labels_full_intersection.jsonl"
    with labels_path.open("w", encoding="utf-8", newline="\n") as stream:
        for _, label in full_rows:
            stream.write(json.dumps(label, ensure_ascii=False, separators=(",", ":")) + "\n")
    return manifest


class EvidenceRanker:
    """Lift native entity/time scores to a role-blind candidate-fact ranking."""

    def __init__(self, evi_root: Path, view: str, all_dicts: dict[str, Any],
                 output_path: Path, top_k: int = 10) -> None:
        modes = ("clean", "full")
        if view not in modes:
            raise ValueError(f"view must be one of {modes}")
        self.view = view
        self.top_k = top_k
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = output_path.open("w", encoding="utf-8", newline="\n")
        self.count = 0
        self.mode_counts: Counter[str] = Counter()
        self.evi_root = evi_root
        sys.path.insert(0, str(evi_root))
        from tools.candidate_space.runtime.pool_views import (  # type: ignore
            ReleaseFactIndex, materialize_pool,
        )
        self.materialize_pool = materialize_pool
        facts_path = evi_root / "data" / "facts" / "facts.jsonl"
        self.fact_runtime = ReleaseFactIndex(facts_path)
        self.pool_rows = iter(_full_intersection_rows(evi_root))
        ent2id = all_dicts["ent2id"]
        ts2id = all_dicts["ts2id"]
        num_entities = len(ent2id)
        endpoints: list[tuple[int, int, int, int]] = []
        for fact in iter_jsonl(facts_path):
            values: list[int] = []
            for entity in (fact.get("subject_id"), fact.get("object_id")):
                if entity in ent2id:
                    values.append(int(ent2id[entity]))
            for raw in (fact.get("time_start"), fact.get("time_end")):
                try:
                    year = int(str(raw)[:4])
                except (TypeError, ValueError):
                    continue
                key = (year, 0, 0)
                if key in ts2id:
                    values.append(num_entities + int(ts2id[key]))
            values = list(dict.fromkeys(values))[:4]
            values += [-1] * (4 - len(values))
            endpoints.append(tuple(values))
        import torch
        self.endpoint_tensor = torch.tensor(endpoints, dtype=torch.long)

    def write(self, scores, question: dict[str, Any]) -> None:
        import torch
        descriptor, labels = next(self.pool_rows)
        expected = question.get("_eviscope_paired_id")
        if descriptor.get("paired_id") != expected or labels.get("paired_id") != expected:
            raise ValueError(
                f"Model/release order mismatch: {expected} vs {descriptor.get('paired_id')}")
        mode = self.view
        pool = self.materialize_pool(descriptor, labels, mode, self.fact_runtime)
        candidates = pool["candidate_fact_indices"]
        endpoint = self.endpoint_tensor[candidates].to(scores.device)
        valid = endpoint >= 0
        safe = endpoint.clamp_min(0)
        endpoint_scores = scores[safe]
        endpoint_scores = endpoint_scores.masked_fill(~valid, float("-inf"))
        primary = endpoint_scores.max(dim=1).values
        finite = torch.where(valid, endpoint_scores, torch.zeros_like(endpoint_scores))
        secondary = finite.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        fact_scores = primary + secondary * 1e-6
        # Stable CPU sort gives deterministic candidate-order tie breaking.
        order = torch.argsort(fact_scores.detach().cpu(), descending=True,
                              stable=True)[:self.top_k].tolist()
        ranked = [int(candidates[index]) for index in order]
        row = {
            "question_id": labels["question_id"],
            "paired_id": labels["paired_id"],
            "view": self.view,
            "materialized_mode": mode,
            "ranked_fact_indices": ranked,
        }
        self.handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.count += 1
        self.mode_counts[mode] += 1

    def close(self) -> dict[str, Any]:
        self.handle.close()
        try:
            next(self.pool_rows)
            raise ValueError("Unused full-intersection pool rows remain after inference")
        except StopIteration:
            pass
        report = {"view": self.view, "questions": self.count,
                  "materialized_modes": dict(self.mode_counts),
                  "output": str(self.output_path)}
        self.output_path.with_suffix(".manifest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
