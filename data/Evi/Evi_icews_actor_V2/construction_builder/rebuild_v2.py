#!/usr/bin/env python3
"""One-command reproducible rebuild of the ICEWS-Actor V2 release."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


BUILDER = Path(__file__).resolve().parent
RELEASE = BUILDER.parent
SOURCE = RELEASE.parents[1] / "Ori/ICEWS"
INTERIM = BUILDER / "data/interim/icews_actor"


def run(*arguments: str, cwd: Path = BUILDER) -> None:
    print("[run]", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=cwd, check=True)


def main() -> None:
    python = sys.executable
    common = [python, "-B", "src/run_pipeline.py", "--dataset", "icews_actor",
              "--archive", str(SOURCE)]
    run(*common, "--stage", "00", "--fresh")
    for stage in ("01", "02", "03"):
        run(*common, "--stage", stage)
    for split in ("train", "dev", "test"):
        run(*common, "--stage", "04", "--split", split)
    run(*common, "--stage", "05")
    for stage in ("06", "07", "08"):
        for split in ("train", "dev", "test"):
            run(*common, "--stage", stage, "--split", split)
    run(*common, "--stage", "09")

    (INTERIM / "11_tau_amb.jsonl").write_text(
        json.dumps({"encoder_revision": "1.0.0:dim=4096:corpus=f122f027755c1aa9",
                    "fit_n_anchors": 36542, "fit_source": "train_v1_frozen",
                    "tau_amb": 0.740344}, sort_keys=True) + "\n", encoding="utf-8")
    (INTERIM / "11_hdm_boundaries.jsonl").write_text(
        json.dumps({"b1": 3.0, "b2": 3.5, "boundary_version": "hdm_v1",
                    "fit_source": "train_v1_frozen", "n_train": 15225},
                   sort_keys=True) + "\n", encoding="utf-8")
    code = (
        "from pathlib import Path; from src.common import load_config; "
        "from src.executors import FactIndex; from src.encoder import FrozenEncoder; "
        "from src.stages_release import compute_hdm; root=Path('.').resolve(); "
        "d=root/'data/interim/icews_actor'; cfg=load_config('icews_actor',root); "
        "facts=d/'02_facts.jsonl'; compute_hdm(cfg,d/'09_evidence_spaces.jsonl',"
        "FactIndex(facts),FrozenEncoder(cfg['hdm'],facts),d/'11_hdm.jsonl',"
        "d/'11_tau_amb.jsonl',d/'11_hdm_boundaries.jsonl',root,d/'11_hdm_report.json')"
    )
    run(python, "-B", "-c", code)
    run(python, "-B", "build_compact_v2.py")
    run(python, "-B", "runtime/verify_release.py", "--release-root", ".",
        "--output", "audit/full_semantic_audit.json", cwd=RELEASE)
    run(python, "-B", "finalize_v2.py")
    run(python, "-B", "runtime/refresh_manifest.py", "--release-root", ".",
        cwd=RELEASE)

    audit_dir = RELEASE / "audit/v2_builder"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for path in INTERIM.iterdir():
        if path.is_file() and ("report" in path.name or path.name in {
                "02_facts_stats.json", "07_thresholds.jsonl",
                "11_hdm_boundaries.jsonl", "11_tau_amb.jsonl",
                "construction_log.jsonl"}) and path.stat().st_size < 1024 * 1024:
            shutil.copy2(path, audit_dir / path.name)
    for target in (BUILDER / "data", BUILDER / "staging_compact_v2",
                   BUILDER / "v1_rows.sqlite"):
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    print("[success] ICEWS-Actor V2 rebuilt and audited", flush=True)


if __name__ == "__main__":
    main()
