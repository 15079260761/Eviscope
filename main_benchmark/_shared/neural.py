from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable


def require_torch():
    try:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt inside the experiment virtual environment") from exc
    return torch, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def batched(values: list, size: int) -> Iterable[list]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch, *_ = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, tokenizer, output: Path, metadata: dict) -> None:
    import json
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
