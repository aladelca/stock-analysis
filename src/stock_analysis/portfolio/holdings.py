from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_current_weights(path: Path | None) -> pd.Series:
    if path is None:
        return pd.Series(dtype=float, name="current_weight")
    if not path.exists():
        msg = f"Current holdings file does not exist: {path}"
        raise FileNotFoundError(msg)

    holdings = _read_holdings(path)
    if "ticker" not in holdings.columns:
        msg = "Current holdings must include a ticker column"
        raise ValueError(msg)
    holdings["ticker"] = holdings["ticker"].astype(str).str.strip()
    holdings = holdings.loc[holdings["ticker"].ne("")]
    if holdings["ticker"].duplicated().any():
        duplicates = sorted(holdings.loc[holdings["ticker"].duplicated(), "ticker"].unique())
        msg = f"Current holdings contains duplicate tickers: {duplicates}"
        raise ValueError(msg)

    weights = _weights_from_holdings(holdings)
    if (weights < 0).any():
        msg = "Current holdings weights cannot be negative in long-only mode"
        raise ValueError(msg)
    if float(weights.sum()) > 1.000001:
        msg = (
            "Current holdings weights must be expressed as decimal portfolio weights "
            "and sum to at most 1.0"
        )
        raise ValueError(msg)
    weights = weights[weights.gt(0)]
    weights.name = "current_weight"
    return weights.sort_index()


def align_current_weights(
    current_weights: pd.Series | dict[str, float] | None,
    tickers: list[str] | pd.Index,
) -> pd.Series:
    index = pd.Index([str(ticker) for ticker in tickers], name="ticker")
    if current_weights is None:
        return pd.Series(0.0, index=index, name="current_weight")
    if isinstance(current_weights, pd.Series):
        series = current_weights.copy()
    else:
        series = pd.Series(current_weights, dtype=float)
    series.index = series.index.astype(str)
    return series.reindex(index).fillna(0.0).astype(float).rename("current_weight")


def _read_holdings(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    msg = f"Unsupported current holdings format: {path.suffix}. Use CSV or Parquet."
    raise ValueError(msg)


def _weights_from_holdings(holdings: pd.DataFrame) -> pd.Series:
    if "current_weight" in holdings.columns:
        values = pd.to_numeric(holdings["current_weight"], errors="coerce")
    elif "weight" in holdings.columns:
        values = pd.to_numeric(holdings["weight"], errors="coerce")
    elif "market_value" in holdings.columns:
        market_value = pd.to_numeric(holdings["market_value"], errors="coerce")
        total = float(market_value.sum(skipna=True))
        if total <= 0:
            msg = "Current holdings market_value must sum to a positive value"
            raise ValueError(msg)
        values = market_value / total
    else:
        msg = "Current holdings must include current_weight, weight, or market_value"
        raise ValueError(msg)

    if values.isna().any():
        msg = "Current holdings contains non-numeric or missing weights"
        raise ValueError(msg)
    return pd.Series(values.to_numpy(dtype=float), index=holdings["ticker"].astype(str))
