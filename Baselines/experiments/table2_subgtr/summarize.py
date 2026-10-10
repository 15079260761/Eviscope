import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--results-root", type=Path, required=True)
args = p.parse_args()

rows = []
for variant in ("wikidata_big", "wikidata_big_complex"):
    for view in ("clean", "full"):
        root = args.results_root / f"subgtr_{variant}" / view
        one = json.loads((root / "scores_k1.json").read_text(encoding="utf-8"))
        ten = json.loads((root / "scores_k10.json").read_text(encoding="utf-8"))
        rows.append({
            "method": "SubGTR", "variant": variant, "view": view,
            "questions": one.get("questions"),
            "hits_at_1": one.get("hits"), "hits_at_10": ten.get("hits"),
            "bs_at_1": one.get("bs"), "bs_at_10": ten.get("bs"),
            "coverage_at_1": one.get("coverage"), "coverage_at_10": ten.get("coverage"),
        })
output = {"protocol": "paired_clean_full_v1", "rows": rows}
(args.results_root / "summary.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(output, indent=2, ensure_ascii=False))
