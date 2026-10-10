"""Server/runtime and TComplEx compatibility checks before expensive work."""
import argparse
import pickle
import sys
from pathlib import Path

import torch
import transformers  # noqa: F401


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--orig-root", type=Path, required=True)
parser.add_argument("--evi-root", type=Path, required=True)
args = parser.parse_args()

tools_root = args.evi_root / "tools"
required_tools = (
    tools_root / "src" / "common.py",
    tools_root / "src" / "bm25.py",
    tools_root / "src" / "encoder.py",
    tools_root / "evaluator" / "metrics.py",
)
missing = [str(path) for path in required_tools if not path.is_file()]
if missing:
    raise SystemExit("Incomplete evaluator package:\n" + "\n".join(missing))
sys.path.insert(0, str(tools_root))
from src.encoder import FrozenEncoder  # noqa: E402,F401
from evaluator.metrics import evaluate_all  # noqa: E402,F401

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch")
state = torch.load(args.checkpoint, map_location="cpu")
required = ("embeddings.0.weight", "embeddings.1.weight", "embeddings.2.weight")
if not all(key in state for key in required):
    raise SystemExit("Checkpoint is not a compatible TComplEx state_dict")
kg = args.orig_root / "wikidata_big" / "kg" / "tkbc_processed_data" / "wikidata_big"
for filename, key in zip(("ent_id", "rel_id", "ts_id"), required):
    with (kg / filename).open("rb") as stream:
        mapping = pickle.load(stream)
    needed = max(mapping.values(), default=-1) + 1
    available = int(state[key].shape[0])
    if needed > available:
        raise SystemExit(f"{filename} needs {needed} rows, checkpoint has {available}")
print("Preflight PASS")
print("CUDA:", torch.cuda.get_device_name(0))
print("TComplEx shapes:", *(tuple(state[key].shape) for key in required))
print("Evaluator imports: PASS")
