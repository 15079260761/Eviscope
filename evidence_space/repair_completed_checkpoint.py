#!/usr/bin/env python3
"""Repair only the missing optimizer-step attestation of a visibly completed run.

This never changes model weights. It requires both epoch-completion log markers,
the expected 8,000-example metadata, matching variant flags, and model files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLAGS = {
    "anchor_only": {"use_substitutes": False, "use_distractors": False, "use_hdm": False},
    "sub": {"use_substitutes": True, "use_distractors": False, "use_hdm": False},
    "dis": {"use_substitutes": False, "use_distractors": True, "use_hdm": False},
    "sub_dis": {"use_substitutes": True, "use_distractors": True, "use_hdm": False},
    "full_eviret": {"use_substitutes": True, "use_distractors": True, "use_hdm": True},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp2_v2")
    parser.add_argument("--dataset", choices=("cron", "icews"), required=True)
    parser.add_argument("--variant", choices=tuple(FLAGS), required=True)
    args = parser.parse_args()

    checkpoint = (HERE / "training_side_train_settings" / "checkpoints" /
                  args.run_id / "eviret" / args.dataset / args.variant)
    metadata_path = checkpoint / "training_metadata.json"
    log_path = HERE / "logs" / args.run_id / f"training_eviret_{args.dataset}_{args.variant}.log"
    if not metadata_path.is_file() or not log_path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata or log: {metadata_path}, {log_path}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"missing model config: {checkpoint / 'config.json'}")
    if not ((checkpoint / "model.safetensors").is_file() or
            (checkpoint / "pytorch_model.bin").is_file()):
        raise FileNotFoundError(f"missing saved model weights in {checkpoint}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("optimizer_steps") not in (None, 1000):
        raise ValueError(f"refusing to replace optimizer_steps={metadata.get('optimizer_steps')!r}")
    if int(metadata.get("examples", -1)) != 8000 or int(metadata.get("epochs", -1)) != 2:
        raise ValueError("checkpoint does not attest 8000 examples and 2 epochs")
    for key, value in FLAGS[args.variant].items():
        if metadata.get(key) is not value:
            raise ValueError(f"checkpoint flag mismatch: {key}")

    log = log_path.read_text(encoding="utf-8", errors="replace")
    required = ("[train] epoch=1/2 batches=500", "[train] epoch=2/2 batches=500")
    if any(marker not in log for marker in required):
        raise ValueError("log does not contain both complete 500-batch epoch markers")

    before = sha256(metadata_path)
    metadata["optimizer_steps"] = 1000
    metadata["optimizer_steps_attestation"] = {
        "method": "recovered_from_complete_epoch_log",
        "log_path": str(log_path),
        "log_sha256": sha256(log_path),
        "required_markers": list(required),
        "metadata_sha256_before_repair": before,
        "model_weights_unchanged": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"[SUCCESS] repaired optimizer-step attestation only: {metadata_path}")


if __name__ == "__main__":
    main()
