"""CTRN entry point for the shared EviScope protocol."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "table2_cronquestions"))
from eviscope_protocol import *  # noqa: F401,F403
