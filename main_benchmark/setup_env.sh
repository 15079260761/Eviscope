#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple
"$VENV/bin/python" -m pip install -r "$PROJECT_ROOT/main_benchmark/requirements.txt" -i https://mirrors.aliyun.com/pypi/simple
"$VENV/bin/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
assert torch.cuda.is_available(), "CUDA is unavailable"
PY
