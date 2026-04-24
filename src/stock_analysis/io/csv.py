from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_analysis.paths import ensure_parent


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    ensure_parent(path)
    df.to_csv(path, index=False)
    return path
