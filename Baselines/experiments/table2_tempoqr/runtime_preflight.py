"""Exercise all three TempoQR branches with real CUDA forward/backward passes."""
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
p.add_argument("--tempoqr-root", type=Path, required=True)
p.add_argument("--prepared", type=Path, required=True)
p.add_argument("--kg-root", type=Path, required=True)
p.add_argument("--tkbc-checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
sys.path.insert(0, str(args.tempoqr_root))
os.chdir(args.tempoqr_root)
os.environ["TEMPOQR_KG_ROOT"] = str(args.kg_root)
os.environ["TEMPOQR_PROCESSED_NAME"] = "wikidata_big"
os.environ.setdefault("TEMPOQR_LM_MODEL", "distilbert-base-uncased")

from run_eviscope_tempoqr import build_model_and_dataset, legacy_args
from utils import loadTkbcModel

source = args.prepared / "wikidata_big/questions/train_10k.pickle"
with source.open("rb") as stream:
    rows = pickle.load(stream)[:4]
smoke_root = args.output.parent / "runtime_smoke_inputs"
smoke_root.mkdir(parents=True, exist_ok=True)
with (smoke_root / "smoke.pickle").open("wb") as stream:
    pickle.dump(rows, stream, protocol=4)
os.environ["TEMPOQR_QUESTIONS_ROOT"] = str(smoke_root)

tkbc = loadTkbcModel(str(args.tkbc_checkpoint))
report = {"status": "pass", "cuda": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(0), "configurations": {}}
for configuration in ("soft_cronkgqa", "soft_tempoqr", "hard_tempoqr"):
    legacy = legacy_args(Namespace(
        configuration=configuration, checkpoint=Path("unused"),
        tkbc_checkpoint=args.tkbc_checkpoint, lr=2e-4,
    ))
    model, dataset = build_model_and_dataset(tkbc, legacy, "smoke")
    model = model.cuda().train()
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False,
                                 num_workers=0, collate_fn=dataset._collate_fn)))
    model.zero_grad(set_to_none=True)
    scores = model(batch)
    loss = model.loss(scores, batch[-1].cuda())
    loss.backward()
    if not torch.isfinite(loss):
        raise SystemExit(f"Non-finite loss for {configuration}")
    report["configurations"][configuration] = {
        "rows_checked": len(dataset), "score_shape": list(scores.shape),
        "loss": float(loss.item()), "model": legacy.model,
        "supervision": legacy.supervision,
    }
    del model, dataset, batch, scores, loss
    torch.cuda.empty_cache()

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
