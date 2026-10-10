"""CUDA/model/data smoke test executed before any formal tmux run."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument("--subgtr-root", type=Path, required=True)
p.add_argument("--prepared", type=Path, required=True)
p.add_argument("--kg-root", type=Path, required=True)
p.add_argument("--tkbc-checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
sys.path.insert(0, str(args.subgtr_root))
os.chdir(args.subgtr_root)
os.environ["SUBGTR_KG_ROOT"] = str(args.kg_root)
os.environ.setdefault("SUBGTR_LM_MODEL", "distilbert-base-uncased")

import utils
from qa_datasets import QA_Dataset_SubGTR
from qa_subgtr import QA_SubGTR

legacy = Namespace(
    model="subgtr", time_sensitivity=True, supervision="hard",
    extra_entities=False, fuse="add", lm_frozen=1, frozen=1,
    aware_module=True, corrupt_hard=0.0, tkg_file="full.txt",
    dataset_name="wikidata_big",
)
tkbc = utils.loadTkbcModel(str(args.tkbc_checkpoint))
model = QA_SubGTR(tkbc, legacy).cuda().train()
smoke_root = args.output.parent / "runtime_smoke_inputs"
report = {"status": "pass", "cuda": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(0), "variants": {}}
for variant in ("wikidata_big", "wikidata_big_complex"):
    source = args.prepared / variant / "questions/train_10k.pickle"
    with source.open("rb") as stream:
        rows = pickle.load(stream)[:4]
    destination = smoke_root / variant
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "smoke.pickle").open("wb") as stream:
        pickle.dump(rows, stream, protocol=4)
    os.environ["SUBGTR_QUESTIONS_ROOT"] = str(destination)
    legacy.dataset_name = variant
    dataset = QA_Dataset_SubGTR("smoke", variant, legacy)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False,
                                 num_workers=0, collate_fn=dataset._collate_fn)))
    model.zero_grad(set_to_none=True)
    scores = model(batch)
    loss = model.loss(scores, batch[-1].cuda())
    loss.backward()
    if not torch.isfinite(loss):
        raise SystemExit(f"Non-finite smoke loss for {variant}: {loss.item()}")
    report["variants"][variant] = {
        "rows_checked": len(dataset), "score_shape": list(scores.shape),
        "loss": float(loss.item()),
    }
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
