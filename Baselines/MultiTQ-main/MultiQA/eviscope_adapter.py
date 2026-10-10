"""Import the shared, audited EviScope/CronQuestions protocol."""
from pathlib import Path
import sys

_EXPERIMENT = Path(__file__).resolve().parents[2] / "experiments" / "table2_cronquestions"
if not (_EXPERIMENT / "eviscope_protocol.py").is_file():
    raise RuntimeError(f"Missing shared EviScope protocol: {_EXPERIMENT}")
sys.path.insert(0, str(_EXPERIMENT))

from eviscope_protocol import EvidenceRanker, prepare_questions, set_reproducible_seed  # noqa: E402,F401

__all__ = ["EvidenceRanker", "prepare_questions", "set_reproducible_seed"]
