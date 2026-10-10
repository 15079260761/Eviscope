"""Verify all six formal outputs before results are reported."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--results-root", type=Path, required=True)
args = p.parse_args()

report = {}
for configuration in ("soft_cronkgqa", "soft_tempoqr", "hard_tempoqr"):
    report[configuration] = {}
    for view in ("clean", "full"):
        root = args.results_root / configuration / view
        predictions = root / "predictions.jsonl"
        if not predictions.is_file():
            raise SystemExit(f"Missing {predictions}")
        with predictions.open(encoding="utf-8") as stream:
            count = sum(1 for line in stream if line.strip())
        if count != 1691:
            raise SystemExit(f"{configuration}/{view}: predictions={count}, expected 1691")
        entry = {"predictions": count}
        for k in (1, 10):
            score_path = root / f"scores_k{k}.json"
            score = json.loads(score_path.read_text(encoding="utf-8"))
            required = {"questions", "hits", "bs", "coverage", "k", "per_question"}
            if not required.issubset(score):
                raise SystemExit(f"{score_path}: missing {sorted(required - set(score))}")
            if score["questions"] != 1691 or len(score["per_question"]) != 1691 or score["k"] != k:
                raise SystemExit(f"{score_path}: invalid question/K accounting")
            entry[f"k{k}"] = {key: score[key] for key in ("hits", "bs", "coverage")}
        report[configuration][view] = entry
print(json.dumps({"status": "pass", "results": report}, indent=2, ensure_ascii=False))
