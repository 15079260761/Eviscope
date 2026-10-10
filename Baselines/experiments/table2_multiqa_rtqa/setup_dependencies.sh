#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "[FATAL] virtual environment missing: $PYTHON_BIN"; exit 1; }
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
"$PYTHON_BIN" -m pip install -r "$EXPERIMENT_DIR/requirements_eviscope.txt"
"$PYTHON_BIN" - <<'PY'
import torch
from FlagEmbedding import BGEM3FlagModel
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
print("FlagEmbedding import: PASS")
PY
echo "[cache] downloading fixed model assets once before the four jobs start"
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
for model in ("distilbert-base-uncased", "BAAI/bge-m3"):
    # Some Hugging Face mirrors return HTTP 403 for macOS metadata shipped in
    # model-card image folders.  Those files are unrelated to model loading.
    path = snapshot_download(
        model,
        ignore_patterns=["*.DS_Store", "**/.DS_Store", "imgs/*", "imgs/**"],
    )
    print(model, "->", path)
PY
