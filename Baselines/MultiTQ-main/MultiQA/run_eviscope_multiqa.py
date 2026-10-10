"""Train original MultiQA on 10k CronQuestions and rank EviScope evidence.

Clean and Full deliberately share one trained checkpoint.  Their only change
is the role-blind candidate fact pool presented at inference time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eviscope_adapter import EvidenceRanker, set_reproducible_seed
from qa_baselines import QA_MultiQA
from qa_datasets import QA_Dataset_MultiQA_cron
from utils import loadTkbcModel


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("train", "eval"), required=True)
    p.add_argument("--tkbc-checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-split", default="train_10k")
    p.add_argument("--dev-split", default="valid_2k")
    p.add_argument("--test-split", default="test_eviscope")
    p.add_argument("--eviscope-root", type=Path)
    p.add_argument("--view", choices=("clean", "full"))
    p.add_argument("--predictions-out", type=Path)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--lm-model", default="distilbert")
    p.add_argument("--frozen", type=int, default=1)
    p.add_argument("--lm-frozen", type=int, default=1)
    return p


def model_args(args):
    # Original constructors consume this legacy attribute surface.
    args.dataset_name = "CronQuestions"
    args.model = "multiqa"
    args.tkbc_model_file = str(args.tkbc_checkpoint)
    args.tkg_file = "full.txt"
    args.kg_dir = "kg"
    return args


@torch.no_grad()
def validation_hits1(model, dataset, args) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=dataset._collate_fn)
    correct = total = offset = 0
    for batch in tqdm(loader, desc="validation", unit="batch"):
        prediction = model(batch).argmax(dim=1)
        predictions = prediction.detach().cpu().tolist()
        gold_lists = dataset.prepared_data["answers_arr"][offset:offset + len(predictions)]
        correct += sum(int(value in gold) for value, gold in zip(predictions, gold_lists))
        total += len(predictions)
        offset += len(predictions)
    score = correct / max(total, 1)
    print(json.dumps({"validation_hits1": score, "questions": total}))
    return score


def train_model(model, train_set, dev_set, args) -> int:
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=train_set._collate_fn)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    best = -1.0
    optimizer_steps = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"train {epoch + 1}/{args.epochs}", unit="batch")
        for step, batch in enumerate(progress, 1):
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch)
            loss = model.loss(scores, batch[-1].cuda(non_blocking=True))
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            running += float(loss.item())
            progress.set_postfix(loss=f"{running / step:.5f}")
        score = validation_hits1(model, dev_set, args)
        if score > best:
            best = score
            torch.save(model.state_dict(), args.checkpoint)
            print(f"[checkpoint] saved best epoch={epoch + 1} hits1={best:.6f}: {args.checkpoint}")
    if not args.checkpoint.is_file():
        raise RuntimeError("Training completed without a checkpoint")
    return optimizer_steps


@torch.no_grad()
def evidence_inference(model, dataset, args) -> None:
    if not args.eviscope_root or not args.view or not args.predictions_out:
        raise SystemExit("eval requires --eviscope-root, --view and --predictions-out")
    ranker = EvidenceRanker(args.eviscope_root, args.view, dataset.all_dicts,
                            args.predictions_out, top_k=10)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=dataset._collate_fn)
    model.eval()
    offset = 0
    try:
        for batch in tqdm(loader, desc=f"MultiQA/{args.view}", unit="batch"):
            scores = model(batch)
            rows = dataset.data[offset:offset + scores.shape[0]]
            if len(rows) != scores.shape[0]:
                raise RuntimeError("Dataset/model batch alignment failure")
            for score, question in zip(scores, rows):
                ranker.write(score, question)
            offset += scores.shape[0]
        if offset != len(dataset):
            raise RuntimeError(f"Inference row count mismatch: {offset} != {len(dataset)}")
        print(json.dumps(ranker.close(), indent=2, ensure_ascii=False))
    except Exception:
        ranker.handle.close()
        raise


def main() -> None:
    args = model_args(parser().parse_args())
    set_reproducible_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("MultiQA requires a CUDA GPU")
    data_root = os.environ.get("MULTIQA_CRON_DATA_ROOT")
    if not data_root:
        raise SystemExit("MULTIQA_CRON_DATA_ROOT is required")
    tkbc = loadTkbcModel(str(args.tkbc_checkpoint))
    model = QA_MultiQA(tkbc, args).cuda()
    if args.mode == "train":
        train_set = QA_Dataset_MultiQA_cron(args.train_split, args=args)
        dev_set = QA_Dataset_MultiQA_cron(args.dev_split, args=args)
        optimizer_steps = train_model(model, train_set, dev_set, args)
        marker = Path(str(args.checkpoint) + ".training_complete.json")
        marker.write_text(json.dumps({
            "status": "complete", "train_samples": len(train_set),
            "dev_samples": len(dev_set), "epochs": args.epochs,
            "batch_size": args.batch_size,
            "batches_per_epoch": math.ceil(len(train_set) / args.batch_size),
            "optimizer_steps": optimizer_steps, "seed": args.seed,
            "checkpoint": str(args.checkpoint),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"[training-complete] {marker.read_text(encoding='utf-8')}")
        return
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing trained MultiQA checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    test_set = QA_Dataset_MultiQA_cron(args.test_split, args=args)
    evidence_inference(model, test_set, args)


if __name__ == "__main__":
    main()
