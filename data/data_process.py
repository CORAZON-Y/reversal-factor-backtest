"""Compatibility entry point for the factor analysis pipeline.

Preferred command:
    .venv/bin/python -m factor_analysis

This wrapper keeps the old command working:
    .venv/bin/python data/data_process.py
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factor_analysis.cli import main


if __name__ == "__main__":
    main()
