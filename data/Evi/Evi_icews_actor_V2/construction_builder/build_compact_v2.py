#!/usr/bin/env python3
"""Build the relaxed ICEWS-Actor V2 compact release from verified stage 11.

The existing V1 four-way cohort is preserved byte-for-byte at the row level.
Only anchor-valid questions that pass the same public-question quality filters
are added.  Missing substitute/distractor roles affect view eligibility rather
than causing exclusion.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
BUILDER = Path(__file__).resolve().parent
INTERIM = BUILDER / "data/interim/icews_actor"
STAGING = BUILDER / "staging_compact_v2"
V1_MAPPING = BUILDER / "v1_id_mapping.jsonl"
FACTS = ROOT / "data/facts.jsonl"
POOL_SIZE = 1000
POLICY = "icews_lazy_neutral_v1"
SEED = 20260923


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON: {path}:{number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def quality_reasons(question: dict[str, Any]) -> list[str]:
    text = normalized(question.get("question"))
    reasons = []
    if any(normalized(answer) and normalized(answer) in text
           for answer in question.get("gold_answers", [])):
        reasons.append("answer_literal_in_question")
    if any(character in str(question.get("question", "")) for character in "{}[]"):
        reasons.append("serialized_template_object_in_question")
    return reasons


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}{hashlib.sha256(payload).hexdigest()[:24]}"


def stable_int(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def event_years(question: dict[str, Any]) -> list[int]:
    years: set[int] = set()
    for event in question.get("source_events", []):
        parts = str(event).split("|")
        for raw in parts[3:5] if len(parts) == 5 else []:
            match = re.match(r"^\s*(\d{1,4})", raw)
            if match:
                years.add(int(match.group(1)))
    return sorted(years)


def load_fact_index() -> tuple[dict[str, int], str, int]:
    mapping: dict[str, int] = {}
    for ordinal, fact in enumerate(rows(FACTS)):
        if int(fact["fact_index"]) != ordinal:
            raise AssertionError(f"fact index not dense at {ordinal}")
        mapping[str(fact["fact_id"])] = ordinal
    return mapping, sha256_file(FACTS), len(mapping)


def build_old_index(database: Path) -> tuple[sqlite3.Connection, dict[str, str]]:
    if database.exists():
        database.unlink()
    qid_to_source: dict[str, str] = {}
    source_to_qid: dict[str, str] = {}
    for item in rows(V1_MAPPING):
        qid_to_source[item["question_id"]] = item["source_paired_id"]
        source_to_qid[item["source_paired_id"]] = item["question_id"]
    connection = sqlite3.connect(database)
    connection.execute(
        "create table old_rows(source_pid text primary key, qid text unique, split text, "
        "question text, label text, descriptor text)"
    )
    for split in ("train", "dev", "test"):
        question_path = ROOT / f"data/questions/{split}.jsonl"
        label_path = ROOT / ("private_labels/test.jsonl" if split == "test"
                             else f"data/supervision/{split}.jsonl")
        descriptor_path = ROOT / f"candidate_space/descriptors/{split}.jsonl"
        for question, label, descriptor in zip(rows(question_path), rows(label_path),
                                                rows(descriptor_path), strict=True):
            qid = question["question_id"]
            if label["question_id"] != qid or descriptor["question_id"] != qid:
                raise AssertionError(f"V1 row alignment mismatch: {split}/{qid}")
            source_pid = qid_to_source.get(qid)
            if source_pid is None:
                # A repeatable V2 rebuild may be run against an already
                # expanded release.  Only the frozen V1 IDs seed this table;
                # V2 additions are deterministically reconstructed below.
                continue
            connection.execute(
                "insert into old_rows values(?,?,?,?,?,?)",
                (source_pid, qid, split, canonical(question), canonical(label),
                 canonical(descriptor)),
            )
    connection.commit()
    count = connection.execute("select count(*) from old_rows").fetchone()[0]
    if count != 23453 or len(source_to_qid) != 23453:
        raise AssertionError(f"unexpected V1 cohort size: db={count}, map={len(source_to_qid)}")
    return connection, source_to_qid


def converted_rows(question: dict[str, Any], fact_index: dict[str, int],
                   facts_sha: str, fact_count: int) -> tuple[dict, dict, dict]:
    split = question["benchmark_split"]
    source_pid = question["paired_id"]
    qid = stable_id(f"icews_actor_v2:q-{split}:", "icews_actor_v2", source_pid)
    descriptor_id = stable_id("icews_actor_v2:descriptor:", qid, POLICY, SEED)

    anchors = list(dict.fromkeys(fact_index[fid]
                                 for fid in question.get("anchor_fact_ids", [])))
    anchor_set = set(anchors)
    substitutes = []
    substitute_indices: set[int] = set()
    for item in question.get("substitutable_evidence", []):
        index = fact_index[item["fact_id"]]
        if index in anchor_set or index in substitute_indices:
            continue
        anchor_index = fact_index[item["anchor_fact_id"]]
        converted = {
            "anchor_fact_index": anchor_index,
            "answer_preservation": bool(item["answer_preservation"]),
            "fact_index": index,
            "relation_score": item["relation_score"],
            "semantic_score": item["semantic_score"],
            "substitutability_score": item["substitutability_score"],
            "temporal_score": item["temporal_score"],
            "threshold": item["threshold"],
            "valid_gate": item["valid_gate"],
            "verification": item["verification"],
        }
        substitutes.append(converted)
        substitute_indices.add(index)
    distractors = []
    distractor_indices: set[int] = set()
    for item in question.get("distracting_evidence", []):
        index = fact_index[item["fact_id"]]
        if index in anchor_set or index in substitute_indices or index in distractor_indices:
            continue
        distractors.append({
            "fact_index": index,
            "similarity_score": item["similarity_score"],
            "subtype": item["subtype"],
            "support_check": item["support_check"],
            "top_level_type": item["top_level_type"],
            "verification": item["verification"],
        })
        distractor_indices.add(index)
    if not anchors:
        raise AssertionError(f"{source_pid}: empty anchor set")
    eligibility = {
        "clean_control": True,
        "substitute_only": bool(substitutes),
        "distractor_only": bool(distractors),
        "full_eviscope": bool(substitutes and distractors),
    }
    public_question = {
        key: question.get(key) for key in (
            "answer_type", "benchmark_split", "normalized_question", "question",
            "question_level", "question_type", "temporal_operator"
        )
    }
    public_question.update({
        "paired_id": qid, "pool_descriptor_id": descriptor_id, "question_id": qid,
    })
    label = {
        "ambiguity_pressure": question["ambiguity_pressure"],
        "anchor_fact_indices": anchors,
        "answers": question["gold_answers"],
        "benchmark_split": split,
        "distracting_evidence": distractors,
        "hybrid_difficulty": question["hybrid_difficulty"],
        "paired_id": qid,
        "question_id": qid,
        "reasoning_demand": question["reasoning_demand"],
        "substitutable_evidence": substitutes,
        "substitute_anchor_coverage": question.get("substitute_anchor_coverage", 0.0),
    }
    descriptor = {
        "background_years": event_years(question),
        "benchmark_split": split,
        "fact_catalog_sha256": facts_sha,
        "fact_universe_size": fact_count,
        "mode_eligibility": eligibility,
        "paired_id": qid,
        "policy_version": POLICY,
        "pool_descriptor_id": descriptor_id,
        "pool_size": POOL_SIZE,
        "question_id": qid,
        "sampling_seed": stable_int(SEED, qid),
    }
    return public_question, label, descriptor


def open_outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    handles: dict[str, Any] = {}
    cohort_handles: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        for category, relative in (
            ("q", f"data/questions/{split}.jsonl"),
            ("l", "private_labels/test.jsonl" if split == "test"
             else f"data/supervision/{split}.jsonl"),
            ("d", f"candidate_space/descriptors/{split}.jsonl"),
        ):
            path = STAGING / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[f"{category}:{split}"] = path.open("w", encoding="utf-8", newline="\n")
        for mode in ("clean_all", "substitute_effect", "distractor_effect", "four_way_common"):
            path = STAGING / f"cohorts/{mode}/{split}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            cohort_handles[f"{mode}:{split}"] = gzip.open(
                path, "wt", encoding="utf-8", newline="\n"
            )
    return handles, cohort_handles


def main() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    fact_index, facts_sha, fact_count = load_fact_index()
    database = BUILDER / "v1_rows.sqlite"
    connection, source_to_qid = build_old_index(database)
    handles, cohort_handles = open_outputs()
    counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    quality: dict[str, Counter[str]] = defaultdict(Counter)
    difficulty: Counter[str] = Counter()
    source_lineage = (STAGING / "audit/v2_source_lineage.jsonl")
    source_lineage.parent.mkdir(parents=True, exist_ok=True)
    lineage_handle = source_lineage.open("w", encoding="utf-8", newline="\n")
    old_seen: set[str] = set()
    ids_by_split: dict[str, set[str]] = defaultdict(set)
    sum_c = sum_a = sum_d = 0.0

    try:
        for question in rows(INTERIM / "11_hdm.jsonl"):
            split = question.get("benchmark_split")
            if not question.get("eviscope_retained"):
                continue
            reasons = quality_reasons(question)
            if reasons:
                quality[split].update(reasons)
                counts["quality_excluded"] += 1
                continue
            source_pid = question["paired_id"]
            old_row = connection.execute(
                "select qid,split,question,label,descriptor from old_rows where source_pid=?",
                (source_pid,),
            ).fetchone()
            if old_row:
                qid, old_split, q_text, l_text, d_text = old_row
                if split != old_split:
                    raise AssertionError(f"V1 split changed: {source_pid}")
                public_question, label, descriptor = map(json.loads, (q_text, l_text, d_text))
                # V1 metadata recorded the compressed catalog checksum.  V2
                # ships the decompressed JSONL catalog, so bind descriptors to
                # the actual released bytes without changing pools or labels.
                descriptor["fact_catalog_sha256"] = facts_sha
                descriptor["fact_universe_size"] = fact_count
                if not (label.get("substitutable_evidence") and label.get("distracting_evidence")):
                    raise AssertionError(f"V1 row is not four-way eligible: {qid}")
                old_seen.add(source_pid)
                origin = "v1_four_way_preserved"
                counts["v1_preserved"] += 1
            else:
                if question.get("substitutable_evidence") and question.get("distracting_evidence"):
                    raise AssertionError(
                        f"strict four-way quality-passing row absent from V1 mapping: {source_pid}"
                    )
                public_question, label, descriptor = converted_rows(
                    question, fact_index, facts_sha, fact_count
                )
                qid = public_question["question_id"]
                origin = "v2_relaxed_addition"
                counts["v2_added"] += 1
            if qid in ids_by_split[split]:
                raise AssertionError(f"duplicate question ID: {qid}")
            ids_by_split[split].add(qid)
            if not set(label["anchor_fact_indices"]).isdisjoint(
                    item["fact_index"] for item in label["substitutable_evidence"]):
                raise AssertionError(f"anchor/substitute overlap: {qid}")
            if not set(label["anchor_fact_indices"]).isdisjoint(
                    item["fact_index"] for item in label["distracting_evidence"]):
                raise AssertionError(f"anchor/distractor overlap: {qid}")
            eligibility = descriptor["mode_eligibility"]
            expected = {
                "clean_control": bool(label["anchor_fact_indices"]),
                "substitute_only": bool(label["anchor_fact_indices"] and label["substitutable_evidence"]),
                "distractor_only": bool(label["anchor_fact_indices"] and label["distracting_evidence"]),
                "full_eviscope": bool(label["anchor_fact_indices"] and label["substitutable_evidence"]
                                      and label["distracting_evidence"]),
            }
            if eligibility != expected:
                raise AssertionError(f"mode eligibility mismatch: {qid}")
            handles[f"q:{split}"].write(canonical(public_question) + "\n")
            handles[f"l:{split}"].write(canonical(label) + "\n")
            handles[f"d:{split}"].write(canonical(descriptor) + "\n")
            cohort_row = canonical({"benchmark_split": split, "paired_id": qid,
                                    "question_id": qid}) + "\n"
            cohort_handles[f"clean_all:{split}"].write(cohort_row)
            if eligibility["substitute_only"]:
                cohort_handles[f"substitute_effect:{split}"].write(cohort_row)
            if eligibility["distractor_only"]:
                cohort_handles[f"distractor_effect:{split}"].write(cohort_row)
            if eligibility["full_eviscope"]:
                cohort_handles[f"four_way_common:{split}"].write(cohort_row)
            lineage_handle.write(canonical({
                "benchmark_split": split,
                "question_id": qid,
                "release_origin": origin,
                "source_paired_id": source_pid,
                "source_question_id": question.get("source_question_id"),
                "source_record": question.get("source_provenance"),
            }) + "\n")
            split_counts[split]["questions"] += 1
            for mode, eligible in eligibility.items():
                split_counts[split][f"eligible:{mode}"] += int(eligible)
            split_counts[split]["anchors"] += len(label["anchor_fact_indices"])
            split_counts[split]["substitutes"] += len(label["substitutable_evidence"])
            split_counts[split]["distractors"] += len(label["distracting_evidence"])
            level = label["hybrid_difficulty"]["difficulty_level"]
            difficulty[level] += 1
            sum_c += float(label["reasoning_demand"]["C_q"])
            sum_a += float(label["ambiguity_pressure"]["A_q"])
            sum_d += float(label["hybrid_difficulty"]["D_q"])
    finally:
        for handle in handles.values():
            handle.close()
        for handle in cohort_handles.values():
            handle.close()
        lineage_handle.close()
        connection.close()

    if old_seen != set(source_to_qid):
        missing = set(source_to_qid) - old_seen
        raise AssertionError(f"V1 cohort was not fully preserved; missing={len(missing)}")
    if any(ids_by_split[a] & ids_by_split[b]
           for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise AssertionError("cross-split question ID collision")
    total = sum(item["questions"] for item in split_counts.values())
    release_stats = {
        "release_version": "2.0.0",
        "policy_version": POLICY,
        "fact_count": fact_count,
        "pool_size": POOL_SIZE,
        "release_policy": "anchor-valid questions; optional verified roles control view eligibility",
        "v1_four_way_questions_preserved": counts["v1_preserved"],
        "v2_relaxed_questions_added": counts["v2_added"],
        "quality_excluded": counts["quality_excluded"],
        "splits": {split: dict(values) for split, values in sorted(split_counts.items())},
        "totals": {
            key: sum(values.get(key, 0) for values in split_counts.values())
            for key in ("questions", "eligible:clean_control", "eligible:substitute_only",
                        "eligible:distractor_only", "eligible:full_eviscope", "anchors",
                        "substitutes", "distractors")
        },
        "hdm": {
            "frozen_tau_amb": 0.740344,
            "frozen_boundaries": {"simple_max": 3.0, "medium_max": 3.5},
            "level_counts": dict(difficulty),
            "mean_C_q": round(sum_c / total, 6),
            "mean_A_q": round(sum_a / total, 6),
            "mean_D_q": round(sum_d / total, 6),
        },
        "quality_filter": {split: dict(values) for split, values in sorted(quality.items())},
    }
    stats_path = STAGING / "statistics/release_statistics.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(release_stats, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "status": "pass",
        "source": str((ROOT.parents[1] / "Ori/ICEWS").resolve()),
        "stage_11_sha256": sha256_file(INTERIM / "11_hdm.jsonl"),
        "fact_catalog_sha256": facts_sha,
        "v1_preserved": counts["v1_preserved"],
        "v2_added": counts["v2_added"],
        "total": total,
        "old_four_way_set_preserved": True,
        "cross_split_question_id_intersections": 0,
        "quality_filter_applied": True,
        "thresholds_and_hdm_boundaries_frozen_from_v1": True,
    }
    audit_path = STAGING / "audit/v2_extension_report.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n", encoding="utf-8")

    for relative in (
        "data/questions/train.jsonl", "data/questions/dev.jsonl", "data/questions/test.jsonl",
        "data/supervision/train.jsonl", "data/supervision/dev.jsonl",
        "private_labels/test.jsonl",
        "candidate_space/descriptors/train.jsonl",
        "candidate_space/descriptors/dev.jsonl",
        "candidate_space/descriptors/test.jsonl",
        "cohorts/clean_all/train.jsonl.gz", "cohorts/clean_all/dev.jsonl.gz",
        "cohorts/clean_all/test.jsonl.gz",
        "cohorts/substitute_effect/train.jsonl.gz", "cohorts/substitute_effect/dev.jsonl.gz",
        "cohorts/substitute_effect/test.jsonl.gz",
        "cohorts/distractor_effect/train.jsonl.gz", "cohorts/distractor_effect/dev.jsonl.gz",
        "cohorts/distractor_effect/test.jsonl.gz",
        "cohorts/four_way_common/train.jsonl.gz", "cohorts/four_way_common/dev.jsonl.gz",
        "cohorts/four_way_common/test.jsonl.gz",
        "statistics/release_statistics.json", "audit/v2_extension_report.json",
        "audit/v2_source_lineage.jsonl",
    ):
        source = STAGING / relative
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    (ROOT / "VERSION").write_text("2.0.0\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
