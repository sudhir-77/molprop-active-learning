"""Reproduce every result from scratch: python run.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from molprop.pipeline import run_from_argv


def main() -> None:
    run_from_argv(None, ROOT)


if __name__ == "__main__":
    main()
