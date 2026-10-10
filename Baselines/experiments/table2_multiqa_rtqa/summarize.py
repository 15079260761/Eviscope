import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results-root", type=Path, required=True)
args = parser.parse_args()
rows, expected = [], None
for method in ("multiqa", "rtqa"):
    for view in ("clean", "full"):
        scores = {}
        for k in (1, 10):
            path = args.results_root / method / view / f"scores_k{k}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            n = int(payload["questions"])
            expected = n if expected is None else expected
            if n != expected:
                raise SystemExit(f"Unpaired result size: {path}: {n} != {expected}")
            scores[k] = payload
        rows.append((method, view, scores))
lines = [
    "# MultiQA and RTQA — paired Clean vs Full", "",
    f"Every row uses the identical {expected}-question Test cohort and 1,000 candidates/question.", "",
    "| Method | Condition | N | Hits@1 | Hits@10 | Coverage@10 | BS@10 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for method, view, scores in rows:
    one, ten = scores[1], scores[10]
    lines.append(f"| {method} | {view} | {expected} | {one['hits']:.6f} | "
                 f"{ten['hits']:.6f} | {ten['coverage']:.6f} | {ten['bs']:.6f} |")
output = args.results_root / "summary.md"
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(output.read_text(encoding="utf-8"))
