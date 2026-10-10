#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from protocol import LEVELS, METHODS

HERE = Path(__file__).resolve().parent
DISPLAY = {"bm25": "BM25", "dual_encoder": "Dual Encoder", "cross_encoder": "Cross-Encoder", "eviret": "EviRet"}
DELTA_H = "ΔH (High−Low)"
DELTA_BS = "ΔBS (High−Low)"


def fixed(value: float) -> str:
    return f"{value:.4f}"


def signed(value: float) -> str:
    # Eliminate the misleading textual form ``-0.0000`` after rounding.
    if round(value, 4) == 0:
        return "0.0000"
    return f"{value:+.4f}"


def write_main_table(destination: Path, dataset: str, rows: list[dict]) -> None:
    fields = ("Method", "N", "Low H@1", "Med H@1", "High H@1", DELTA_H,
              "Low BS@1", "Med BS@1", "High BS@1", DELTA_BS)
    formatted = []
    for row in rows:
        formatted.append({
            "Method": row["Method"], "N": str(row["N"]),
            "Low H@1": fixed(float(row["Low H@1"])),
            "Med H@1": fixed(float(row["Med H@1"])),
            "High H@1": fixed(float(row["High H@1"])),
            DELTA_H: signed(float(row["High H@1"]) - float(row["Low H@1"])),
            "Low BS@1": fixed(float(row["Low BS@1"])),
            "Med BS@1": fixed(float(row["Med BS@1"])),
            "High BS@1": fixed(float(row["High BS@1"])),
            DELTA_BS: signed(float(row["High BS@1"]) - float(row["Low BS@1"])),
        })
    with (destination / f"table_exp3_{dataset}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(formatted)
    header = "| " + " | ".join(fields) + " |"
    align = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(row[name] for name in fields) + " |" for row in formatted]
    note = ("Δ = High − Low. ‘+’ 表示 High 条件相对 Low 条件升高，‘−’ 表示降低；"
            "四舍五入后绝对值小于 0.00005 记为 0.0000。")
    (destination / f"table_exp3_{dataset}.md").write_text(
        "\n".join([note, "", header, align, *body]) + "\n", encoding="utf-8")


def reformat_existing(destination: Path) -> None:
    for dataset in ("cron", "icews"):
        path = destination / f"table_exp3_{dataset}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            old_rows = list(csv.DictReader(handle))
        required = {"Method", "N", "Low H@1", "Med H@1", "High H@1",
                    "Low BS@1", "Med BS@1", "High BS@1"}
        if not old_rows or not required.issubset(old_rows[0]):
            raise ValueError(f"cannot reformat malformed table: {path}")
        write_main_table(destination, dataset, old_rows)
        print(f"[reformatted] {path}", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp3_v1")
    parser.add_argument("--reformat-existing", action="store_true",
                        help="repair table direction/signs without requiring raw method outputs")
    args = parser.parse_args()
    source = HERE / "method_results" / args.run_id
    destination = HERE / "results" / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    if args.reformat_existing:
        reformat_existing(destination)
        return
    all_rows = []
    for dataset in ("cron", "icews"):
        rows = []
        checkpoint_hashes = {}
        for method in METHODS:
            root = source / method / dataset
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest["test_questions"] <= 0 or not manifest["same_checkpoint_all_levels"]:
                raise ValueError(f"invalid manifest: {root}")
            checkpoint_hashes[method] = manifest["checkpoint_sha256"]
            metrics = {level: {k: json.loads((root / level / f"metrics_k{k}.json").read_text(encoding="utf-8"))
                               for k in (1, 10)} for level in LEVELS}
            if len({metrics[level][1]["questions"] for level in LEVELS}) != 1:
                raise ValueError(f"unpaired metric counts: {method}/{dataset}")
            row = {
                "Method": DISPLAY[method], "N": metrics["low"][1]["questions"],
                "Low H@1": metrics["low"][1]["hits"], "Med H@1": metrics["medium"][1]["hits"],
                "High H@1": metrics["high"][1]["hits"],
                DELTA_H: metrics["high"][1]["hits"] - metrics["low"][1]["hits"],
                "Low BS@1": metrics["low"][1]["bs"], "Med BS@1": metrics["medium"][1]["bs"],
                "High BS@1": metrics["high"][1]["bs"],
                DELTA_BS: metrics["high"][1]["bs"] - metrics["low"][1]["bs"],
            }
            rows.append(row)
            for level in LEVELS:
                for k in (1, 10):
                    metric = metrics[level][k]
                    all_rows.append({"method": DISPLAY[method], "dataset": dataset, "level": level, "k": k,
                                     "N": metric["questions"], "hits": metric["hits"], "bs": metric["bs"],
                                     "coverage": metric["coverage"], "nbs": metric["nbs"]})
        write_main_table(destination, dataset, rows)
        (destination / f"checkpoint_audit_{dataset}.json").write_text(json.dumps(checkpoint_hashes, indent=2) + "\n", encoding="utf-8")
        print(f"[table] {destination / f'table_exp3_{dataset}.md'}", flush=True)
    with (destination / "all_metrics_k1_k10.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    print(f"[SUCCESS] exactly two main tables plus auxiliary complete metrics -> {destination}", flush=True)

if __name__ == "__main__":
    main()
