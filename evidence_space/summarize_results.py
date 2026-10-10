#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ("bm25", "dual_encoder", "cross_encoder", "eviret")
DATASETS = ("cron", "icews")
CONDITIONS = ("original", "sub", "dis", "full")
VARIANTS = ("anchor_only", "sub", "dis", "sub_dis", "full_eviret")


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing required raw result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp2_v2")
    args = parser.parse_args()
    benchmark_rows = []
    training_rows = []

    for method in METHODS:
        for dataset in DATASETS:
            root = HERE / "benchmark_side_test_settings" / "results" / args.run_id / method / dataset
            manifest = load(root / "run_manifest.json")
            if manifest.get("protocol_version") != "exp2-section-2.1-2.6-v2":
                raise ValueError(f"unattested benchmark result: {root}")
            values = load(root / "summary.json")
            row = {"method": method, "dataset": dataset}
            for condition in CONDITIONS:
                for k in (1, 10):
                    for metric in ("hits", "bs", "coverage", "nbs"):
                        row[f"{condition}_{metric}@{k}"] = values[condition][f"k{k}"][metric]
            for k in (1, 10):
                for metric in ("hits", "bs", "coverage", "nbs"):
                    row[f"delta_sub_{metric}@{k}"] = row[f"sub_{metric}@{k}"] - row[f"original_{metric}@{k}"]
                    row[f"delta_dis_{metric}@{k}"] = row[f"original_{metric}@{k}"] - row[f"dis_{metric}@{k}"]
            # Under the released paper metric, label-only substitute activation
            # can change Hits but cannot change anchor-only BS/Cov/nBS.
            for k in (1, 10):
                for metric in ("bs", "coverage", "nbs"):
                    if abs(row[f"delta_sub_{metric}@{k}"]) > 1e-12:
                        raise ValueError(f"unexpected +Sub {metric}@{k} change for fixed ranking")
            benchmark_rows.append(row)

    for dataset in DATASETS:
        for variant in VARIANTS:
            root = HERE / "training_side_train_settings" / "results" / args.run_id / "eviret" / dataset / variant
            manifest = load(root / "run_manifest.json")
            if manifest.get("protocol_version") != "exp2-section-2.7-v2":
                raise ValueError(f"unattested training result: {root}")
            if manifest.get("training_questions") != 8000 or manifest.get("training_unique_question_ids") != 8000:
                raise ValueError(f"training count is not attested: {root}")
            values = load(root / "summary.json")
            row = {"method": "eviret", "dataset": dataset, "variant": variant,
                   "use_substitutes": manifest["use_substitutes"],
                   "use_distractors": manifest["use_distractors"],
                   "use_hdm": manifest["use_hdm"],
                   "training_questions": manifest["training_questions"],
                   "expected_optimizer_steps": manifest["expected_optimizer_steps"]}
            for k in (1, 10):
                for metric in ("hits", "bs", "coverage", "nbs"):
                    row[f"{metric}@{k}"] = values[f"k{k}"][metric]
            training_rows.append(row)

    output = HERE / "results" / args.run_id
    write_csv(output / "table_2a_benchmark_side.csv", benchmark_rows)
    write_csv(output / "table_2b_training_ablation.csv", training_rows)

    lines = [
        "| Method | Dataset | Original H@10/BS@10/nBS@10 | +Sub H@10/BS@10/nBS@10 | +Dis H@10/BS@10/nBS@10 | Full H@10/BS@10/nBS@10 | ΔSub H@10 | ΔDis H@10/BS@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in benchmark_rows:
        cell = lambda condition: (f"{row[condition + '_hits@10']:.4f}/"
                                  f"{row[condition + '_bs@10']:.4f}/"
                                  f"{row[condition + '_nbs@10']:.4f}")
        lines.append(
            f"| {row['method']} | {row['dataset']} | {cell('original')} | {cell('sub')} | "
            f"{cell('dis')} | {cell('full')} | {row['delta_sub_hits@10']:+.4f} | "
            f"{row['delta_dis_hits@10']:+.4f}/{row['delta_dis_bs@10']:+.4f} |"
        )
    lines.extend([
        "",
        "> Original/+Sub and +Dis/Full deliberately share rankings. Under the released",
        "> anchor-only BS/Cov/nBS definitions, substitute-label activation can change Hits",
        "> but cannot change BS/Cov/nBS. Repeated values are structural, not independent runs.",
    ])
    (output / "table_2a_benchmark_side.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    detailed = [
        "| Method | Dataset | Setting | H@1 | H@10 | BS@1 | BS@10 | Cov@1 | Cov@10 | nBS@1 | nBS@10 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in benchmark_rows:
        for condition in CONDITIONS:
            values = [row[f"{condition}_{metric}@{k}"]
                      for metric, k in (("hits", 1), ("hits", 10), ("bs", 1),
                                        ("bs", 10), ("coverage", 1), ("coverage", 10),
                                        ("nbs", 1), ("nbs", 10))]
            detailed.append(
                f"| {row['method']} | {row['dataset']} | {condition} | "
                + " | ".join(f"{value:.4f}" for value in values) + " |"
            )
    (output / "table_2a_benchmark_side_all_metrics.md").write_text(
        "\n".join(detailed) + "\n", encoding="utf-8")

    lines = [
        "| Dataset | EviRet variant | Sub | Dis | HDM | Train N | Steps | H@10 | BS@10 | Cov@10 | nBS@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in training_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {int(row['use_substitutes'])} | "
            f"{int(row['use_distractors'])} | {int(row['use_hdm'])} | "
            f"{row['training_questions']} | {row['expected_optimizer_steps']} | "
            f"{row['hits@10']:.4f} | {row['bs@10']:.4f} | "
            f"{row['coverage@10']:.4f} | {row['nbs@10']:.4f} |"
        )
    (output / "table_2b_training_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SUCCESS] audited Experiment 2 tables -> {output}")


if __name__ == "__main__":
    main()
