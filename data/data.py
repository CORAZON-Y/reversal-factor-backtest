"""Inspect a raw parquet table without running the analysis pipeline.

Examples:
    .venv/bin/python data/data.py
    .venv/bin/python data/data.py --file industry.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factor_analysis.constants import DATA_DIR, DAILY_DATA_FILE
from factor_analysis.data_loader import resolve_data_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one raw parquet data table.")
    parser.add_argument("--file", default=DAILY_DATA_FILE, help="File name under basic_data/.")
    parser.add_argument("--rows", type=int, default=5, help="Number of leading rows to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = resolve_data_file(DATA_DIR, args.file)
    frame = pd.read_parquet(path)
    print(f"file: {path}")
    print(f"shape: {frame.shape}")
    print(f"columns: {list(frame.columns)}")
    print(frame.head(max(args.rows, 0)))


if __name__ == "__main__":
    main()
