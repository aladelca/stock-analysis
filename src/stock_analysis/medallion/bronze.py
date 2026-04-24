from __future__ import annotations

import pandas as pd

from stock_analysis.domain.schemas import validate_columns
from stock_analysis.io.csv import write_csv
from stock_analysis.io.parquet import write_parquet
from stock_analysis.paths import ProjectPaths


def write_bronze_constituents(df: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    clean = validate_columns(df.copy(), "sp500_constituents")
    write_parquet(clean, paths.bronze_path("sp500_constituents"))
    write_csv(clean, paths.csv_mirror_path("bronze", "sp500_constituents"))
    return clean


def write_bronze_prices(df: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    clean = validate_columns(df.copy(), "daily_prices")
    clean = clean.sort_values(["ticker", "date"]).reset_index(drop=True)
    write_parquet(clean, paths.bronze_path("daily_prices"))
    write_csv(clean, paths.csv_mirror_path("bronze", "daily_prices"))
    return clean
