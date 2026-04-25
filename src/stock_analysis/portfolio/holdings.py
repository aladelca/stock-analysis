from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PortfolioState:
    weights: pd.Series
    market_values: pd.Series
    cash_balance: float = 0.0
    portfolio_value: float | None = None

    @property
    def asset_value(self) -> float:
        if self.market_values.empty:
            if self.portfolio_value is None:
                return 0.0
            return max(float(self.portfolio_value) - self.cash_balance, 0.0)
        return float(self.market_values.sum())

    @property
    def resolved_portfolio_value(self) -> float | None:
        if self.portfolio_value is not None:
            return float(self.portfolio_value)
        if not self.market_values.empty or self.cash_balance > 0:
            return float(self.market_values.sum() + self.cash_balance)
        return None


def load_current_weights(path: Path | None) -> pd.Series:
    return load_portfolio_state(path).weights


def load_portfolio_state(
    path: Path | None,
    *,
    cash_balance: float = 0.0,
    portfolio_value: float | None = None,
) -> PortfolioState:
    if cash_balance < 0:
        msg = "Cash balance cannot be negative in long-only mode"
        raise ValueError(msg)
    if portfolio_value is not None and portfolio_value <= 0:
        msg = "Portfolio value must be positive when provided"
        raise ValueError(msg)
    if path is None:
        resolved_cash = cash_balance
        if portfolio_value is not None and cash_balance == 0:
            resolved_cash = portfolio_value
        return PortfolioState(
            weights=pd.Series(dtype=float, name="current_weight"),
            market_values=pd.Series(dtype=float, name="market_value"),
            cash_balance=float(resolved_cash),
            portfolio_value=portfolio_value,
        )
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

    market_values = _market_values_from_holdings(holdings)
    resolved_portfolio_value = _resolve_portfolio_value(
        market_values,
        cash_balance=cash_balance,
        portfolio_value=portfolio_value,
    )
    weights = _weights_from_holdings(
        holdings,
        market_values=market_values,
        portfolio_value=resolved_portfolio_value,
    )
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
    if not market_values.empty:
        market_values = market_values.reindex(weights.index).fillna(0.0)
        market_values.name = "market_value"
    return PortfolioState(
        weights=weights.sort_index(),
        market_values=market_values.sort_index(),
        cash_balance=float(cash_balance),
        portfolio_value=resolved_portfolio_value,
    )


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


def _weights_from_holdings(
    holdings: pd.DataFrame,
    *,
    market_values: pd.Series,
    portfolio_value: float | None,
) -> pd.Series:
    if "current_weight" in holdings.columns:
        values = pd.to_numeric(holdings["current_weight"], errors="coerce")
    elif "weight" in holdings.columns:
        values = pd.to_numeric(holdings["weight"], errors="coerce")
    elif not market_values.empty:
        if portfolio_value is None or portfolio_value <= 0:
            msg = "Portfolio value must be positive to normalize market_value holdings"
            raise ValueError(msg)
        values = market_values.reindex(holdings["ticker"].astype(str)) / portfolio_value
    else:
        msg = "Current holdings must include current_weight, weight, or market_value"
        raise ValueError(msg)

    if values.isna().any():
        msg = "Current holdings contains non-numeric or missing weights"
        raise ValueError(msg)
    weights = pd.Series(values.to_numpy(dtype=float), index=holdings["ticker"].astype(str))
    return weights


def _market_values_from_holdings(holdings: pd.DataFrame) -> pd.Series:
    if "market_value" not in holdings.columns:
        return pd.Series(dtype=float, name="market_value")
    values = pd.to_numeric(holdings["market_value"], errors="coerce")
    if values.isna().any():
        msg = "Current holdings contains non-numeric or missing market_value"
        raise ValueError(msg)
    if (values < 0).any():
        msg = "Current holdings market_value cannot be negative in long-only mode"
        raise ValueError(msg)
    if float(values.sum(skipna=True)) <= 0:
        msg = "Current holdings market_value must sum to a positive value"
        raise ValueError(msg)
    return pd.Series(values.to_numpy(dtype=float), index=holdings["ticker"].astype(str))


def _resolve_portfolio_value(
    market_values: pd.Series,
    *,
    cash_balance: float,
    portfolio_value: float | None,
) -> float | None:
    if portfolio_value is not None:
        if (
            not market_values.empty
            and float(market_values.sum()) + cash_balance > portfolio_value + 1e-6
        ):
            msg = "Portfolio value cannot be smaller than holdings market_value plus cash_balance"
            raise ValueError(msg)
        return float(portfolio_value)
    if market_values.empty and cash_balance <= 0:
        return None
    return float(market_values.sum() + cash_balance)
