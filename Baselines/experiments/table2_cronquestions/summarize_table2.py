"""Summarize the two strictly paired candidate-pool conditions."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results-root", type=Path, required=True)
args = parser.parse_args()

methods = ("cronkgqa", "ctrn")
modes = ("clean", "full")
rows = []
expected_n = None
for method in methods:
    for mode in modes:
        scores = {}
        for k in (1, 10):
            path = args.results_root / method / mode / f"scores_k{k}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            n = int(payload["questions"])
            if expected_n is None:
                expected_n = n
            if n != expected_n:
                raise SystemExit(f"Unpaired result size: {path}: {n} != {expected_n}")
            scores[k] = payload
        rows.append((method, mode, scores))

lines = [
    "# CronQuestions paired Clean vs Full-EviScope results", "",
    f"All cells use the identical {expected_n}-question full-intersection Test cohort.", "",
    "| Method | Pool mode | N | Hits@1 | Hits@10 | Coverage@10 | BS@10 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for method, mode, scores in rows:
    one, ten = scores[1], scores[10]
    lines.append(
        f"| {method} | {mode} | {expected_n} | {one['hits']:.6f} | "
        f"{ten['hits']:.6f} | {ten['coverage']:.6f} | {ten['bs']:.6f} |")
out = args.results_root / "table2.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out.read_text(encoding="utf-8"))
