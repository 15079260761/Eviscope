import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--results-root", type=Path, required=True)
args = p.parse_args()
rows = []
for configuration in ("soft_cronkgqa", "soft_tempoqr", "hard_tempoqr"):
    for view in ("clean", "full"):
        root = args.results_root / configuration / view
        one = json.loads((root / "scores_k1.json").read_text(encoding="utf-8"))
        ten = json.loads((root / "scores_k10.json").read_text(encoding="utf-8"))
        rows.append({
            "method": "TempoQR", "configuration": configuration, "view": view,
            "questions": one["questions"], "hits_at_1": one["hits"],
            "hits_at_10": ten["hits"], "bs_at_1": one["bs"],
            "bs_at_10": ten["bs"], "coverage_at_1": one["coverage"],
            "coverage_at_10": ten["coverage"],
        })
output = {"protocol": "paired_clean_full_v1", "rows": rows}
(args.results_root / "summary.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(output, indent=2, ensure_ascii=False))
