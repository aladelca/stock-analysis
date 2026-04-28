from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd

OUTCOME_PENDING = "pending"
OUTCOME_REALIZED = "realized"
OUTCOME_UNAVAILABLE = "unavailable"

FORECAST_OUTCOME_COLUMNS: tuple[str, ...] = (
    "forecast_horizon_days",
    "forecast_start_date",
    "forecast_end_date",
    "realized_return",
    "realized_spy_return",
    "realized_active_return",
    "forecast_error",
    "forecast_hit",
    "outcome_status",
)


@dataclass(frozen=True)
class _Outcome:
    forecast_end_date: str | None
    realized_return: float | None
    realized_spy_return: float | None
    realized_active_return: float | None
    forecast_error: float | None
    forecast_hit: bool | None
    outcome_status: str


@dataclass(frozen=True)
class _ForwardReturn:
    end_date: str
    value: float


def attach_forecast_outcomes(
    recommendations: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    horizon_days: int,
    run_data_as_of_date: date | str | None = None,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    """Add pending or realized horizon outcome columns for recommendation rows."""

    result = recommendations.copy()
    if result.empty:
        return _ensure_outcome_columns(result)

    default_start = _date_text(run_data_as_of_date)
    if "forecast_horizon_days" not in result.columns:
        result["forecast_horizon_days"] = int(horizon_days)
    else:
        result["forecast_horizon_days"] = (
            pd.to_numeric(result["forecast_horizon_days"], errors="coerce")
            .fillna(int(horizon_days))
            .astype(int)
        )

    if "forecast_start_date" not in result.columns:
        source = result.get(
            "as_of_date", pd.Series([default_start] * len(result), index=result.index)
        )
        result["forecast_start_date"] = source.map(_date_text).fillna(default_start)
    else:
        result["forecast_start_date"] = result["forecast_start_date"].map(_date_text)
        if default_start is not None:
            result["forecast_start_date"] = result["forecast_start_date"].fillna(default_start)

    prepared_prices = _prepare_prices(daily_prices)
    price_paths: dict[str, pd.DataFrame] = {
        str(ticker): frame.reset_index(drop=True)
        for ticker, frame in prepared_prices.groupby("ticker", sort=False)
    }
    spy_path = price_paths.get(benchmark_ticker)

    rows = cast(list[dict[str, Any]], result.to_dict("records"))
    outcomes = [_outcome_for_row(row, price_paths, spy_path) for row in rows]
    for column in FORECAST_OUTCOME_COLUMNS:
        if column in {"forecast_horizon_days", "forecast_start_date"}:
            continue
        result[column] = [getattr(outcome, column) for outcome in outcomes]
    return _ensure_outcome_columns(result)


def _ensure_outcome_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in FORECAST_OUTCOME_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result


def _prepare_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    if daily_prices.empty:
        return pd.DataFrame(columns=["ticker", "date", "adj_close"])
    prices = daily_prices.copy()
    prices["ticker"] = prices["ticker"].astype(str)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    return (
        prices.dropna(subset=["ticker", "date", "adj_close"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def _outcome_for_row(
    row: dict[str, Any],
    price_paths: dict[str, pd.DataFrame],
    spy_path: pd.DataFrame | None,
) -> _Outcome:
    ticker = str(row.get("ticker") or "")
    start_date = _date_text(row.get("forecast_start_date"))
    horizon = _int_or_none(row.get("forecast_horizon_days"))
    if not ticker or start_date is None or horizon is None or horizon < 1:
        return _empty_outcome(OUTCOME_UNAVAILABLE)

    path = price_paths.get(ticker)
    if path is None or path.empty:
        return _empty_outcome(OUTCOME_UNAVAILABLE)

    realized = _forward_return(path, start_date, horizon)
    if realized is None:
        return _empty_outcome(_pending_or_unavailable(path, start_date))

    spy_realized = _forward_return(spy_path, start_date, horizon) if spy_path is not None else None
    active = realized.value - spy_realized.value if spy_realized is not None else None
    expected = _float_or_none(row.get("expected_return"))
    forecast_error = realized.value - expected if expected is not None else None
    return _Outcome(
        forecast_end_date=realized.end_date,
        realized_return=realized.value,
        realized_spy_return=spy_realized.value if spy_realized is not None else None,
        realized_active_return=active,
        forecast_error=forecast_error,
        forecast_hit=_forecast_hit(expected, realized.value),
        outcome_status=OUTCOME_REALIZED,
    )


def _forward_return(
    path: pd.DataFrame | None,
    start_date: str,
    horizon: int,
) -> _ForwardReturn | None:
    if path is None or path.empty:
        return None
    dates = pd.to_datetime(path["date"]).dt.date.astype(str)
    matches = dates[dates == start_date].index.tolist()
    if not matches:
        return None
    start_position = int(matches[0])
    end_position = start_position + horizon
    if end_position >= len(path):
        return None
    start_price = float(path.iloc[start_position]["adj_close"])
    end_price = float(path.iloc[end_position]["adj_close"])
    if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0:
        return None
    return _ForwardReturn(
        end_date=pd.Timestamp(path.iloc[end_position]["date"]).date().isoformat(),
        value=float(end_price / start_price - 1),
    )


def _pending_or_unavailable(path: pd.DataFrame, start_date: str) -> str:
    dates = pd.to_datetime(path["date"]).dt.date.astype(str)
    if start_date in set(dates.tolist()):
        return OUTCOME_PENDING
    return OUTCOME_UNAVAILABLE


def _empty_outcome(status: str) -> _Outcome:
    return _Outcome(
        forecast_end_date=None,
        realized_return=None,
        realized_spy_return=None,
        realized_active_return=None,
        forecast_error=None,
        forecast_hit=None,
        outcome_status=status,
    )


def _forecast_hit(expected: float | None, realized: float) -> bool | None:
    if expected is None or expected == 0:
        return None
    return bool(np.sign(expected) == np.sign(realized))


def _date_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    try:
        return pd.Timestamp(cast(Any, value)).date().isoformat()
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if _is_missing(value):
        return None
    result = float(cast(Any, value))
    return result if np.isfinite(result) else None


def _int_or_none(value: object) -> int | None:
    if _is_missing(value):
        return None
    return int(cast(Any, value))


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(cast(Any, value)))
    except (TypeError, ValueError):
        return False
