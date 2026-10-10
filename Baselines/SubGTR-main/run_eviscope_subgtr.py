"""Train published SubGTR configurations and rank paired EviScope evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from eviscope_adapter import EvidenceRanker, set_reproducible_seed
from qa_datasets import QA_Dataset_SubGTR
from qa_subgtr import QA_SubGTR


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("train", "eval"), required=True)
    p.add_argument("--variant", choices=("wikidata_big", "wikidata_big_complex"), required=True)
    p.add_argument("--tkbc-checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-split", default="train_10k")
    p.add_argument("--dev-split", default="valid_2k")
    p.add_argument("--test-split", default="test_eviscope")
    p.add_argument("--eviscope-root", type=Path)
    p.add_argument("--view", choices=("clean", "full"))
    p.add_argument("--predictions-out", type=Path)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=20260927)
    return p


def legacy_args(args):
    args.dataset_name = args.variant
    args.model = "subgtr"
    args.subgraph_reasoning = True
    args.time_sensitivity = True
    args.aware_module = True
    args.supervision = "hard"
    args.extra_entities = False
    args.fuse = "add"
    args.corrupt_hard = 0.0
    args.tkg_file = "full.txt"
    args.frozen = 1
    args.lm_frozen = 1
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def validation_hits1(model, dataset, args) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=0, collate_fn=dataset._collate_fn)
    correct = total = offset = 0
    for batch in tqdm(loader, desc="validation", unit="batch"):
        predicted = model(batch).argmax(dim=1).detach().cpu().tolist()
        gold = dataset.prepared_data["answers_arr"][offset:offset + len(predicted)]
        correct += sum(int(answer in targets) for answer, targets in zip(predicted, gold))
        total += len(predicted)
        offset += len(predicted)
    value = correct / max(total, 1)
    print(json.dumps({"validation_hits1": value, "questions": total}))
    return value


def train_model(model, train_set, dev_set, args) -> tuple[int, int]:
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=train_set._collate_fn)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    best, best_epoch, optimizer_steps = -1.0, 0, 0
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
            best, best_epoch = score, epoch + 1
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            torch.save(model.state_dict(), temporary)
            temporary.replace(args.checkpoint)
            print(f"[checkpoint] best epoch={best_epoch} hits1={best:.6f}: {args.checkpoint}")
    if not args.checkpoint.is_file():
        raise RuntimeError("Training completed without a checkpoint")
    return optimizer_steps, best_epoch


def _native_subgraph_scores(score: torch.Tensor, dataset, index: int) -> torch.Tensor:
    """Apply SubGTR's published top-100 symbolic reranking before fact lifting."""
    question = dataset.data[index]
    predicted = dataset.getAnswersFromScores(score, k=min(100, score.numel()))
    item = list(dataset[index])
    qtype = question.get("type")
    if qtype == "before_after":
        if "event_head" in question.get("annotation", {}):
            event = question["annotation"]["event_head"]
            if str(event).startswith("Q"):
                event = list(utils.event2time[event])[0][3]
            item[7] = item[8] = int(event)
        predicted = utils.rerank_ba(
            predicted, question["entities"], question["annotation"]["type"], item)
    elif qtype == "first_last" and question.get("answer_type") == "entity":
        predicted = utils.rerank_fl(
            predicted, question["entities"], question["annotation"]["adj"], item)
    elif qtype == "simple_time":
        predicted = utils.rerank_st(predicted, question["entities"], None, item)
    elif qtype == "time_join" and len(question.get("entities", [])) == 2:
        if "event_head" in question.get("annotation", {}):
            event = question["annotation"]["event_head"]
            if str(event).startswith("Q"):
                event = list(utils.event2time[event])[0][3]
            item[7] = item[8] = int(event)
        predicted = utils.rerank_tj(predicted, question["entities"], None, item)

    ids, seen = [], set()
    ent2id, ts2id = dataset.all_dicts["ent2id"], dataset.all_dicts["ts2id"]
    n_entities = len(ent2id)
    for value in predicted:
        if value in ent2id:
            answer_id = int(ent2id[value])
        else:
            try:
                answer_id = n_entities + int(ts2id[(int(str(value)[:4]), 0, 0)])
            except (KeyError, TypeError, ValueError):
                continue
        if answer_id not in seen:
            seen.add(answer_id)
            ids.append(answer_id)
    adjusted = score.clone()
    ceiling = float(score.max().item()) + len(ids) + 1.0
    for rank, answer_id in enumerate(ids):
        adjusted[answer_id] = ceiling - rank
    return adjusted


@torch.no_grad()
def evidence_inference(model, dataset, args) -> None:
    if not args.eviscope_root or not args.view or not args.predictions_out:
        raise SystemExit("eval requires --eviscope-root, --view and --predictions-out")
    ranker = EvidenceRanker(args.eviscope_root, args.view, dataset.all_dicts,
                            args.predictions_out, top_k=10)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=0, collate_fn=dataset._collate_fn)
    model.eval()
    offset = 0
    try:
        for batch in tqdm(loader, desc=f"SubGTR-{args.variant}/{args.view}", unit="batch"):
            scores = model(batch)
            rows = dataset.data[offset:offset + scores.shape[0]]
            if len(rows) != scores.shape[0]:
                raise RuntimeError("Dataset/model batch alignment failure")
            for local_index, (score, question) in enumerate(zip(scores, rows)):
                score = _native_subgraph_scores(score, dataset, offset + local_index)
                ranker.write(score, question)
            offset += scores.shape[0]
        if offset != len(dataset):
            raise RuntimeError(f"Inference row mismatch: {offset} != {len(dataset)}")
        print(json.dumps(ranker.close(), indent=2, ensure_ascii=False))
    except Exception:
        ranker.handle.close()
        raise


def main() -> None:
    args = legacy_args(parser().parse_args())
    set_reproducible_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("SubGTR requires a CUDA GPU")
    if not os.environ.get("SUBGTR_QUESTIONS_ROOT"):
        raise SystemExit("SUBGTR_QUESTIONS_ROOT is required")
    if not os.environ.get("SUBGTR_KG_ROOT"):
        raise SystemExit("SUBGTR_KG_ROOT is required")
    tkbc = utils.loadTkbcModel(str(args.tkbc_checkpoint))
    model = QA_SubGTR(tkbc, args).cuda()
    if args.mode == "train":
        train_set = QA_Dataset_SubGTR(args.train_split, args.variant, args)
        dev_set = QA_Dataset_SubGTR(args.dev_split, args.variant, args)
        if len(train_set) != 10_000:
            raise SystemExit(f"Refusing non-protocol train size: {len(train_set)}")
        if len(dev_set) != 2_000:
            raise SystemExit(f"Refusing non-protocol dev size: {len(dev_set)}")
        steps, best_epoch = train_model(model, train_set, dev_set, args)
        marker = Path(str(args.checkpoint) + ".training_complete.json")
        payload = {
            "status": "complete", "method": "SubGTR", "variant": args.variant,
            "train_samples": len(train_set), "dev_samples": len(dev_set),
            "epochs": args.epochs, "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "batches_per_epoch": math.ceil(len(train_set) / args.batch_size),
            "optimizer_steps": steps, "seed": args.seed,
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
            "subgraph_reasoning": True, "time_sensitivity": True, "aware_module": True,
        }
        temporary = marker.with_suffix(marker.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(marker)
        print(f"[training-complete]\n{marker.read_text(encoding='utf-8')}")
        return
    marker = Path(str(args.checkpoint) + ".training_complete.json")
    if not args.checkpoint.is_file() or not marker.is_file():
        raise SystemExit(f"Missing complete trained artifact: {args.checkpoint}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("variant") != args.variant or metadata.get("train_samples") != 10_000:
        raise SystemExit(f"Checkpoint marker does not match requested protocol: {metadata}")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    test_set = QA_Dataset_SubGTR(args.test_split, args.variant, args)
    if len(test_set) != 1_691:
        raise SystemExit(f"Refusing unexpected paired Test size: {len(test_set)}")
    evidence_inference(model, test_set, args)


if __name__ == "__main__":
    main()
