#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp1_v1")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    rows = []
    for method in ("bm25", "dual_encoder", "cross_encoder", "eviret"):
        for dataset in ("cron", "icews"):
            path = args.results / args.run_id / method / dataset / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            summary = json.loads(path.read_text(encoding="utf-8"))
            for mode in ("full",):
                row = {"method": method, "dataset": dataset, "condition": mode}
                for k in (1, 10):
                    values = summary[mode][f"k{k}"]
                    for metric in ("hits", "bs", "coverage", "nbs"):
                        row[f"{metric}@{k}"] = values[metric]
                rows.append(row)
    output = args.results / args.run_id
    fields = list(rows[0])
    with (output / "table_exp1_full.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["| Method | Dataset | Condition | Hits@1 | Hits@10 | BS@1 | BS@10 | Cov@1 | Cov@10 | nBS@1 | nBS@10 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join(
            [row["method"], row["dataset"], row["condition"]] +
            [f"{row[key]:.4f}" for key in ("hits@1", "hits@10", "bs@1", "bs@10",
                                             "coverage@1", "coverage@10", "nbs@1", "nbs@10")]
        ) + " |")
    (output / "table_exp1_full.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SUCCESS] {output / 'table_exp1_full.csv'}")


if __name__ == "__main__":
    main()
