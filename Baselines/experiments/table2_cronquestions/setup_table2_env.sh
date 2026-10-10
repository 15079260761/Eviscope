#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$BASELINES_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1 || {
  echo "[FATAL] Python not found: $PYTHON_BOOTSTRAP" >&2
  exit 1
}

"$PYTHON_BOOTSTRAP" - <<'PY'
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit(f"Python 3.10-3.12 is required, found {sys.version}")
print("Bootstrap Python:", sys.version)
PY

if [[ -e "$VENV_DIR" && ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
  echo "[FATAL] $VENV_DIR exists but is not a Python virtual environment" >&2
  exit 1
fi
if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
  echo "[setup] creating virtual environment: $VENV_DIR"
  "$PYTHON_BOOTSTRAP" -m venv --system-site-packages "$VENV_DIR"
else
  echo "[setup] reusing virtual environment: $VENV_DIR"
  if ! grep -Eiq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "$VENV_DIR/pyvenv.cfg"; then
    echo "[FATAL] existing environment does not inherit the image PyTorch." >&2
    echo "Remove this incomplete environment and run setup again: $VENV_DIR" >&2
    exit 1
  fi
fi

PYTHON_BIN="$VENV_DIR/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools

# The AutoDL image already supplies its matched GPU stack (PyTorch 2.12.1,
# CUDA 13.0).  Reuse it instead of downloading a conflicting CUDA wheel.
"$PYTHON_BIN" - <<'PY'
import torch
print("Inherited image torch:", torch.__version__)
print("Inherited CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("The image PyTorch cannot see CUDA inside this venv")
PY
"$PYTHON_BIN" -m pip install -r "$EXPERIMENT_DIR/requirements_eviscope.txt"

echo "[setup] caching the exact DistilBERT assets used by both methods"
echo "[setup] Hugging Face endpoint: $HF_ENDPOINT"
"$PYTHON_BIN" - <<'PY'
from transformers import DistilBertModel, DistilBertTokenizer
name = "distilbert-base-uncased"
DistilBertTokenizer.from_pretrained(name)
DistilBertModel.from_pretrained(name)
print("Hugging Face assets cached:", name)
PY

echo "[setup] validating imports and GPU"
"$PYTHON_BIN" - <<'PY'
import numpy, sklearn, torch, tqdm, transformers
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the new virtual environment")
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}:", torch.cuda.get_device_name(index))
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
print("scikit-learn:", sklearn.__version__)
PY

echo "======================================================"
echo "[SUCCESS] environment ready: $VENV_DIR"
echo "[SUCCESS] activate with: source $VENV_DIR/bin/activate"
echo "[SUCCESS] Python: $PYTHON_BIN"
echo "======================================================"
