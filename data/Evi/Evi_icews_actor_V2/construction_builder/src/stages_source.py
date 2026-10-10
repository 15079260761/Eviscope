"""Source stages per spec sections 4-8:
00_verify_sources, 01_lock_source_splits, 02_normalize_facts,
03_normalize_questions.

Section 5 is enforced structurally: benchmark_split := source_split for every
question; no re-splitting code exists.  All split information flows through
the frozen manifests (section 6); later stages read only the manifests.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import (
    canonical_json, dump_yaml, iter_jsonl, normalize_question, sha256_file,
    sha256_text, utc_now, write_json, write_jsonl,
)
from .sources import (
    SourceBundle, iter_cron_questions, iter_icews_questions,
    normalize_facts, verify_source,
)
from .timeutils import interval_overlap


def _manifest_dir(dataset: str, root: Path) -> Path:
    return root / "manifests" / dataset


def append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(row) + "\n")


# --------------------------------------------------------------------------- #
# 00_verify_sources
# --------------------------------------------------------------------------- #

def verify_sources_stage(config: dict[str, Any], bundle: SourceBundle,
                         output: Path, root: Path, log: Path) -> dict[str, Any]:
    report = verify_source(config, bundle, output)
    # section 4.2: the source split counts must be re-verified from the file
    actual = Counter()
    rows = iter_icews_questions(config, bundle, None) if config["dataset"] == "icews_actor" \
        else iter_cron_questions(config, bundle, None)
    for row in rows:
        actual[row["source_split"]] += 1
    expected = config.get("source_counts", {})
    mapping = config.get("source_split_names", {})
    mapped_actual = Counter()
    for key, count in actual.items():
        mapped_actual[mapping.get(key, key)] += count
    expected_mapped = {mapping.get(key, key): value for key, value in expected.items()}
    report["source_split_counts_raw"] = dict(actual)
    report["source_split_counts_mapped"] = dict(mapped_actual)
    report["source_split_counts_expected"] = dict(expected_mapped)
    mismatch = {split: mapped_actual.get(split, 0) != expected_mapped.get(split)
                for split in expected_mapped}
    if report["status"] == "pass" and any(mismatch.values()):
        report["status"] = "fail"
        report["source_count_mismatch"] = mismatch
    # section 6: frozen source manifest (yaml + sha recorded)
    mdir = _manifest_dir(config["dataset"], root)
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_version": config["rule_version"],
        "dataset": config["dataset"],
        "source_release": config["source_release"],
        "license": config["license"],
        "temporal_coverage": config["temporal_coverage"],
        "archive_path": config["archive_path"],
        "source_path": report.get("source_path"),
        "source_kind": report.get("source_kind"),
        "member_sha256": report.get("member_sha256", {}),
        "checked_at": report.get("checked_at"),
        "source_split_counts_raw": report.get("source_split_counts_raw", {}),
        "source_split_counts_mapped": report.get("source_split_counts_mapped", {}),
    }
    manifest_path = mdir / "source_manifest.yaml"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(dump_yaml(manifest) + "\n")
    report["source_manifest_path"] = str(manifest_path)
    report["source_manifest_sha256"] = sha256_file(manifest_path)
    write_json(output, report)
    if report["status"] == "fail":
        raise ValueError(f"Source verification failed; see {output}")
    return report


# --------------------------------------------------------------------------- #
# 01_lock_source_splits
# --------------------------------------------------------------------------- #

def _question_row_hashes(row: dict[str, Any]) -> dict[str, str]:
    question = " ".join(str(row.get("question", "")).split())
    answers = sorted(set(str(x).strip() for x in row.get("answers", []) if str(x).strip()))
    return {
        "question_sha256": sha256_text(question),
        "answer_sha256": sha256_text(canonical_json(answers)),
        "question": question,
        "answers": answers,
    }


def lock_source_splits(config: dict[str, Any], bundle: SourceBundle,
                       output: Path, root: Path, log: Path,
                       limit: int | None = None) -> dict[str, Any]:
    """Read every source question, lock its source split and paired ids, and
    freeze the split manifests (section 6).

    Two streaming passes over the source readers (the source question files
    are large): pass 1 counts source ids for the disambiguation rule, pass 2
    writes rows and manifests.  Peak memory is O(unique source ids)."""
    dataset = config["dataset"]
    mapping = config.get("source_split_names", {})
    stats: Counter[str] = Counter()

    def read_rows():
        return iter_icews_questions(config, bundle, limit) if dataset == "icews_actor" \
            else iter_cron_questions(config, bundle, limit)

    def normalize(row: dict[str, Any], count_rejections: bool = False):
        raw_split = str(row.get("source_split") or "")
        split = mapping.get(raw_split)
        if split is None:
            if count_rejections:
                stats[f"rejected:unmappable_source_split:{raw_split}"] += 1
            return None
        hashes = _question_row_hashes(row)
        if not hashes["question"] or not hashes["answers"]:
            if count_rejections:
                stats["rejected:empty_question_or_answers"] += 1
            return None
        row["source_split_raw"] = raw_split
        row["source_split"] = split       # section 5: benchmark_split == source_split
        row["benchmark_split"] = split
        row.update(hashes)
        return row

    # pass 1: count occurrences of each source id (CronQuestions reuses
    # uniq_id for unrelated rows, so the id needs a stable occurrence suffix)
    id_counts: Counter[str] = Counter()
    for row in read_rows():
        if normalize(row) is None:
            continue
        id_counts[str(row["source_question_id"])] += 1
    stats["locked_questions"] = sum(id_counts.values())

    mdir = _manifest_dir(dataset, root)
    split_path = mdir / f"{dataset}_source_split.jsonl"
    paired_path = mdir / "paired_question_manifest.jsonl"
    exclusion_path = mdir / "exclusion_manifest.jsonl"
    quarantine_path = mdir / "quarantine_manifest.jsonl"
    for path in (split_path, paired_path, exclusion_path, quarantine_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    # pass 2: stream rows to the locked output + manifests
    occurrence: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    def gen():
        for row in read_rows():
            if normalize(row, count_rejections=True) is None:
                continue
            key = str(row["source_question_id"])
            index = occurrence[key]
            occurrence[key] += 1
            base = f"{dataset}:{key}"
            if id_counts[key] > 1:
                paired_id = f"{base}:{row['benchmark_split']}:{index:04d}"
                stats["paired_id_disambiguated"] += 1
            else:
                paired_id = base
            row["paired_id"] = paired_id
            row["question_id"] = paired_id
            row["source_provenance"] = {
                "source_file": row.get("_source_file", ""),
                "source_row": row.get("_source_row", 0),
            }
            split_counts[row["benchmark_split"]] += 1
            append_manifest(split_path, {
                "source_question_id": row["source_question_id"],
                "paired_id": row["paired_id"],
                "source_split": row["benchmark_split"],
                "question_sha256": row["question_sha256"],
                "answer_sha256": row["answer_sha256"],
                "source_file": row["source_provenance"]["source_file"],
                "source_row": row["source_provenance"]["source_row"],
            })
            append_manifest(paired_path, {
                "paired_id": row["paired_id"],
                "question_id": row["paired_id"],
                "source_dataset": dataset,
                "source_question_id": row["source_question_id"],
                "source_split": row["benchmark_split"],
                "benchmark_split": row["benchmark_split"],
            })
            yield row

    write_jsonl(output, gen())
    manifest_sha = sha256_file(split_path)
    write_json(mdir / "manifest_hashes.json", {
        "source_split_manifest_sha256": manifest_sha,
        "locked_at": utc_now(),
        "split_counts": dict(split_counts),
        "paired_id_disambiguated": stats["paired_id_disambiguated"],
    })
    stats["manifest_sha256"] = manifest_sha
    stats.update({f"split:{split}": split_counts.get(split, 0)
                  for split in ("train", "dev", "test")})
    return dict(stats)


# --------------------------------------------------------------------------- #
# 02_normalize_facts
# --------------------------------------------------------------------------- #

def normalize_facts_stage(config: dict[str, Any], bundle: SourceBundle,
                          facts_path: Path, quarantine_path: Path,
                          statistics_path: Path,
                          max_records: int | None) -> dict[str, Any]:
    """Facts in the section-7 schema: nested time object with granularity and
    raw time, canonical_fact_text, granularity-aware fact_id."""
    stats = normalize_facts(config, bundle, facts_path, quarantine_path,
                            statistics_path, max_records=max_records)
    return stats


# --------------------------------------------------------------------------- #
# 03_normalize_questions
# --------------------------------------------------------------------------- #

def normalize_questions_stage(config: dict[str, Any], locked_path: Path,
                              output: Path, report_path: Path,
                              log: Path) -> dict[str, Any]:
    dataset = config["dataset"]
    stats: Counter[str] = Counter()

    def gen():
        for row in iter_jsonl(locked_path):
            stats["input_questions"] += 1
            normalized = normalize_question(row["question"])
            row["normalized_question"] = normalized
            row["source_dataset"] = dataset
            # section 8: normalization is surface-only; semantics unchanged
            row["topic_entity_ids"] = sorted(set(str(x) for x in row.get("entities", [])))
            row["time_constraints"] = sorted(set(str(x) for x in row.get("times", [])))
            row["relations"] = sorted(set(str(x) for x in row.get("relations", [])))
            row["gold_answers"] = sorted(set(str(x) for x in row.get("answers", [])))
            row.pop("entities", None)
            row.pop("times", None)
            yield row

    write_jsonl(output, gen())
    report = {"dataset": dataset, **dict(stats)}
    write_json(report_path, report)
    return report
