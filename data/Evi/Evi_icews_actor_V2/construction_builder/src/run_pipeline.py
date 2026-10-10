"""EviScope benchmark construction pipeline (spec sections 23/25).

14 stages: 00_verify_sources, 01_lock_source_splits, 02_normalize_facts,
03_normalize_questions, 04_extract_anchor_facts, 05_build_family_audit,
06_recall_candidates, 07_score_substitutability, 08_build_distractors,
09_build_evidence_spaces, 10_build_paired_pools, 11_compute_hdm,
12_run_audits, 13_export_release.

Splits are inherited from the source (section 5) — there is no re-splitting
script.  Stages 04/06/07/08 run per split with frozen parameters (section 19).
Thresholds are fitted on Train only, confirmed on Dev once, consumed frozen
by Test (section 14).  Any unresolved LOCK_BEFORE_BUILD aborts (section 14).

Usage:
  python src/run_pipeline.py --dataset icews_actor|cronquestions [--stage all|00..13]
                            [--split train|dev|test] [--limit-questions N] [--limit-facts N]
                            [--archive PATH] [--fresh]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (
    append_log, builder_root, environment_info, load_config, sha256_file, utc_now,
)
from src.executors import FactIndex
from src.bm25 import Bm25Stats
from src.encoder import FrozenEncoder
from src.sources import resolve_source
from src.stages_source import (
    lock_source_splits, normalize_facts_stage, normalize_questions_stage,
    verify_sources_stage,
)
from src.stages_evidence import (
    build_distractors_stage, build_evidence_spaces, build_family_audit,
    extract_anchor_facts, recall_candidates_stage, score_substitutability_stage,
)
from src.stages_release import (
    build_paired_pools, compute_hdm, export_release, run_audits,
)

DATASETS = ("icews_actor", "cronquestions")


def _check_locks(config: dict[str, Any]) -> None:
    """Section 14: unresolved LOCK_BEFORE_BUILD aborts the formal pipeline."""
    for section in ("evidence_scoring", "hdm", "evaluation"):
        block = config.get(section, {}) or {}
        for key, value in block.items():
            if isinstance(value, str) and "LOCK_BEFORE_BUILD" in value:
                raise SystemExit(f"Unresolved LOCK_BEFORE_BUILD: {section}.{key}")
    for key, value in config.items():
        if isinstance(value, str) and "LOCK_BEFORE_BUILD" in value:
            raise SystemExit(f"Unresolved LOCK_BEFORE_BUILD: {key}")


def interim_dir(dataset: str, root: Path) -> Path:
    return root / "data" / "interim" / dataset


def _paths(dataset: str, root: Path) -> dict[str, Path]:
    d = interim_dir(dataset, root)
    return {
        "00_report": d / "00_verify_report.json",
        "01_locked": d / "01_questions_locked.jsonl",
        "02_facts": d / "02_facts.jsonl",
        "02_facts_quarantine": d / "02_facts_quarantine.jsonl",
        "02_facts_stats": d / "02_facts_stats.json",
        "03_questions": d / "03_questions_normalized.jsonl",
        "03_report": d / "03_normalize_report.json",
        "07_thresholds": d / "07_thresholds.jsonl",
        "11_tau_amb": d / "11_tau_amb.jsonl",
        "11_boundaries": d / "11_hdm_boundaries.jsonl",
        "10_pools": d / "10_pools",
        "12_leakage": d / "12_leakage_report.json",
        "12_split": d / "12_split_consistency_report.json",
        "12_evidence": d / "12_evidence_validation_report.json",
        "12_hdm": d / "12_hdm_validation_report.json",
        "log": d / "construction_log.jsonl",
    }


def _chain(dataset: str, root: Path) -> dict[str, Path]:
    """Frozen questions-file chain across per-split stages 04/06/07/08."""
    d = interim_dir(dataset, root)
    return {
        "start": d / "03_questions_normalized.jsonl",
        "04_train": d / "04_anchors_train.jsonl",
        "04_dev": d / "04_anchors_dev.jsonl",
        "04_test": d / "04_anchors_test.jsonl",
        "05": d / "05_family_audit.jsonl",
        "06_train": d / "06_recall_train.jsonl",
        "06_dev": d / "06_recall_dev.jsonl",
        "06_test": d / "06_recall_test.jsonl",
        "07_train": d / "07_scores_train.jsonl",
        "07_dev": d / "07_scores_dev.jsonl",
        "07_test": d / "07_scores_test.jsonl",
        "08_train": d / "08_distractors_train.jsonl",
        "08_dev": d / "08_distractors_dev.jsonl",
        "08_test": d / "08_distractors_test.jsonl",
        "09": d / "09_evidence_spaces.jsonl",
        "10": d / "10_pools_questions.jsonl",
        "11": d / "11_hdm.jsonl",
    }


def run_stage(config: dict[str, Any], stage: str, split: str | None,
              limit_questions: int | None, limit_facts: int | None,
              archive_override: str | None, root: Path) -> None:
    dataset = config["dataset"]
    paths = _paths(dataset, root)
    chain = _chain(dataset, root)
    started = utc_now()

    def qfile(name: str) -> Path:
        return paths[name] if name in paths else (interim_dir(dataset, root) / f"{name}.jsonl")

    def finish(stage_name: str, out_files: list[Path], stats: dict[str, Any] | None) -> None:
        entry = {
            "stage": stage_name,
            "dataset": dataset,
            "split": split,
            "started_at": started,
            "finished_at": utc_now(),
            "config_sha256": sha256_file(Path(config["_config_path"])),
            "stats": stats or {},
            "outputs": {str(p.name): (sha256_file(p) if p.exists() else None) for p in out_files},
            "environment": environment_info(),
        }
        append_log(paths["log"], entry)

    def index() -> FactIndex:
        return FactIndex(paths["02_facts"])

    def encoder() -> FrozenEncoder:
        return FrozenEncoder(config["hdm"], paths["02_facts"])

    if stage == "00":
        bundle = resolve_source(config, archive_override, root)
        report = verify_sources_stage(config, bundle, paths["00_report"], root, paths["log"])
        finish("00_verify_sources", [paths["00_report"]], None)
        print(f"[00] verify: status={report['status']} kind={report['source_kind']} "
              f"splits={report.get('source_split_counts_mapped')}")
        return

    if stage == "01":
        bundle = resolve_source(config, archive_override, root)
        stats = lock_source_splits(config, bundle, paths["01_locked"], root, paths["log"],
                                   limit=limit_questions)
        finish("01_lock_source_splits", [paths["01_locked"]], stats)
        print(f"[01] lock splits: {stats.get('locked_questions')} locked, "
              f"{stats.get('split:train')}/{stats.get('split:dev')}/{stats.get('split:test')}")
        return

    if stage == "02":
        bundle = resolve_source(config, archive_override, root)
        stats = normalize_facts_stage(config, bundle, paths["02_facts"],
                                      paths["02_facts_quarantine"], paths["02_facts_stats"],
                                      max_records=limit_facts)
        finish("02_normalize_facts", [paths["02_facts"], paths["02_facts_quarantine"],
                                      paths["02_facts_stats"]], stats)
        print(f"[02] facts: accepted={stats['accepted_facts']} "
              f"duplicates={stats['duplicate_facts']} quarantined={stats['quarantined_facts']}")
        return

    if stage == "03":
        stats = normalize_questions_stage(config, paths["01_locked"], paths["03_questions"],
                                          paths["03_report"], paths["log"])
        finish("03_normalize_questions", [paths["03_questions"], paths["03_report"]], stats)
        print(f"[03] normalize questions: {stats['input_questions']}")
        return

    if stage == "04":
        if split is None:
            raise SystemExit("--split is required for stage 04")
        # chain order train->dev->test
        prev_name = {"train": "start", "dev": "04_train", "test": "04_dev"}[split]
        prev_file = chain[prev_name]
        out_file = chain[f"04_{split}"]
        stats = extract_anchor_facts(config, prev_file, split, index(), out_file, root,
                                     qfile(f"04_anchors_{split}_report.json"))
        finish(f"04_extract_anchor_facts_{split}", [out_file], stats)
        print(f"[04:{split}] anchors: verified={stats.get('verified', 0)} "
              f"excluded={stats.get('excluded', 0)}")
        return

    if stage == "05":
        stats = build_family_audit(config, chain["04_test"], chain["05"], root,
                                   qfile("05_family_audit_report.json"))
        finish("05_build_family_audit", [chain["05"]], stats)
        print(f"[05] family audit: families={stats['families']} "
              f"excluded={stats['excluded']}")
        return

    if stage == "06":
        if split is None:
            raise SystemExit("--split is required for stage 06")
        prev_name = {"train": "05", "dev": "06_train", "test": "06_dev"}[split]
        out_file = chain[f"06_{split}"]
        stats = recall_candidates_stage(config, chain[prev_name], split, index(),
                                        Bm25Stats(paths["02_facts"]), out_file,
                                        qfile(f"06_recall_{split}_report.json"))
        finish(f"06_recall_candidates_{split}", [out_file], stats)
        print(f"[06:{split}] recall: {stats}")
        return

    if stage == "07":
        if split is None:
            raise SystemExit("--split is required for stage 07")
        prev_name = {"train": "06_test", "dev": "07_train", "test": "07_dev"}[split]
        out_file = chain[f"07_{split}"]
        thresholds_path = paths["07_thresholds"] if split != "train" else None
        fit_path = paths["07_thresholds"] if split == "train" else None
        stats = score_substitutability_stage(config, chain[prev_name], split, index(),
                                             encoder(), out_file, thresholds_path, fit_path,
                                             qfile(f"07_scores_{split}_report.json"))
        finish(f"07_score_substitutability_{split}", [out_file], stats)
        print(f"[07:{split}] scoring: {stats}")
        return

    if stage == "08":
        if split is None:
            raise SystemExit("--split is required for stage 08")
        prev_name = {"train": "07_test", "dev": "08_train", "test": "08_dev"}[split]
        out_file = chain[f"08_{split}"]
        stats = build_distractors_stage(config, chain[prev_name], split, index(), out_file,
                                        root, qfile(f"08_distractors_{split}_report.json"))
        finish(f"08_build_distractors_{split}", [out_file], stats)
        print(f"[08:{split}] distractors: {stats}")
        return

    if stage == "09":
        stats = build_evidence_spaces(config, chain["08_test"], chain["09"], root,
                                      qfile("09_evidence_spaces_report.json"))
        finish("09_build_evidence_spaces", [chain["09"]], stats)
        print(f"[09] evidence spaces: retained={stats['retained']} excluded={stats['excluded']} "
              f"reconciled={stats['train_role_labels_reconciled']}")
        return

    if stage == "10":
        stats = build_paired_pools(config, chain["09"], index(), paths["10_pools"],
                                   qfile("10_pools_report.json"), chain["10"])
        finish("10_build_paired_pools", [paths["10_pools"] / "source_control" / "candidate_pools.jsonl",
                                         paths["10_pools"] / "eviscope" / "candidate_pools.jsonl",
                                         chain["10"]], stats)
        print(f"[10] paired pools: {stats}")
        return

    if stage == "11":
        stats = compute_hdm(config, chain["10"], index(), encoder(), chain["11"],
                            paths["11_tau_amb"], paths["11_boundaries"], root,
                            qfile("11_hdm_report.json"))
        finish("11_compute_hdm", [chain["11"], paths["11_tau_amb"], paths["11_boundaries"]],
               stats)
        print(f"[11] hdm: tau_amb={stats['tau_amb']} levels={stats['level_counts']}")
        return

    if stage == "12":
        source_report = {}
        try:
            from src.common import read_json
            source_report = read_json(paths["00_report"]) or {}
        except Exception:
            pass
        report = run_audits(config, chain["11"], paths["02_facts"], paths["10_pools"],
                            source_report, encoder(), paths["12_leakage"],
                            paths["12_hdm"], paths["12_evidence"], paths["12_split"])
        finish("12_run_audits", [paths["12_leakage"], paths["12_split"],
                                 paths["12_evidence"], paths["12_hdm"]],
               report.get("mandatory_checks"))
        print(f"[12] audits: status={report['status']}")
        if report["status"] == "fail":
            print("  mandatory failures:", report["mandatory_checks"])
        return

    if stage == "13":
        source_report = {}
        try:
            from src.common import read_json
            source_report = read_json(paths["00_report"]) or {}
        except Exception:
            pass
        report = export_release(config, chain["11"], paths["02_facts"], paths["10_pools"],
                                root / "data" / "release", source_report, encoder(),
                                config["rule_version"], root)
        finish("13_export_release", [], report)
        print(f"[13] export: retained={report['total_retained']} "
              f"splits={report['question_counts']} -> {report['release_dir']}")
        return

    raise SystemExit(f"unknown stage: {stage}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="EviScope benchmark construction pipeline")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--stage", default="all")
    parser.add_argument("--split", choices=["train", "dev", "test"], default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--limit-facts", type=int, default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)

    root = builder_root()
    config = load_config(args.dataset, root)
    # Full ICEWS builds enforce the configured publication-size floor.  An
    # explicitly limited run is a diagnostic pilot and must never be mistaken
    # for a releasable build; allow it to complete so every stage can be tested.
    config["_allow_undersized_pilot"] = (
        args.limit_questions is not None or args.limit_facts is not None
    )
    _check_locks(config)

    if args.fresh:
        import shutil
        for target in (root / "data" / "interim" / args.dataset,
                       root / "data" / "release" / args.dataset,
                       root / "manifests" / args.dataset):
            if target.exists():
                shutil.rmtree(target)
                print(f"[fresh] removed {target}")

    if args.stage == "all":
        stages = ["00", "01", "02", "03"]
        for s in ("train", "dev", "test"):
            stages.append(f"04:{s}")
        stages.append("05")
        for s in ("train", "dev", "test"):
            stages += [f"06:{s}"]
        for s in ("train", "dev", "test"):
            stages += [f"07:{s}"]
        for s in ("train", "dev", "test"):
            stages += [f"08:{s}"]
        stages += ["09", "10", "11", "12", "13"]
        for item in stages:
            stage, _, split = item.partition(":")
            run_stage(config, stage, split or None, args.limit_questions,
                      args.limit_facts, args.archive, root)
    else:
        run_stage(config, args.stage, args.split, args.limit_questions,
                  args.limit_facts, args.archive, root)


if __name__ == "__main__":
    main()
