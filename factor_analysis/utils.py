"""Small shared utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.zfill(6)


def finite_mask(*arrays: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=arrays[0].index)
    for array in arrays:
        mask &= np.isfinite(array.to_numpy(dtype=float, na_value=np.nan))
    return mask


def setup_plot_cache() -> None:
    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "factor_analysis_cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
