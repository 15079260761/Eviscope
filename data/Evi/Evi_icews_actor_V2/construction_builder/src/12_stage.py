"""Stage 12 wrapper (spec section 25).

Usage:  python src/12_stage.py --dataset icews_actor|cronquestions [--split train|dev|test]
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_pipeline import main

if __name__ == "__main__":
    main(["--stage", "12"] + sys.argv[1:])
