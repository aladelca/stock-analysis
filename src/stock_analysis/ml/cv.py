from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardCVConfig:
    train_window_years: int = 3
    val_window_months: int = 6
    step_months: int = 3
    embargo_days: int = 15
    expanding: bool = False
    purged: bool = True
    max_target_horizon_days: int = 5
    safety_margin_days: int = 5

    def __post_init__(self) -> None:
        if self.train_window_years < 1:
            msg = "train_window_years must be positive"
            raise ValueError(msg)
        if self.val_window_months < 1 or self.step_months < 1:
            msg = "validation and step windows must be positive"
            raise ValueError(msg)
        required_embargo = self.max_target_horizon_days + self.safety_margin_days
        if self.embargo_days < required_embargo:
            msg = (
                f"embargo_days={self.embargo_days} is too small; "
                f"must be >= target horizon + safety margin ({required_embargo})"
            )
            raise ValueError(msg)


def walk_forward_splits(
    dates: Iterable[object],
    config: WalkForwardCVConfig | None = None,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield time-ordered train/validation date slices with embargo and optional purging."""

    cfg = config or WalkForwardCVConfig()
    unique_dates = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(list(dates))).dropna().unique()
    ).sort_values()
    if unique_dates.empty:
        return

    first_date = unique_dates[0]
    last_date = unique_dates[-1]
    train_end_bound = first_date + pd.DateOffset(years=cfg.train_window_years)

    while train_end_bound < last_date:
        train_start_bound = (
            first_date
            if cfg.expanding
            else train_end_bound - pd.DateOffset(years=cfg.train_window_years)
        )
        train_dates = unique_dates[
            (unique_dates >= train_start_bound) & (unique_dates < train_end_bound)
        ]
        if train_dates.empty:
            train_end_bound += pd.DateOffset(months=cfg.step_months)
            continue

        min_val_start = train_dates.max() + pd.offsets.BDay(cfg.embargo_days)
        val_end_bound = min_val_start + pd.DateOffset(months=cfg.val_window_months)
        val_dates = unique_dates[(unique_dates >= min_val_start) & (unique_dates < val_end_bound)]
        if val_dates.empty:
            break

        if cfg.purged:
            train_dates = _purge_overlapping_target_dates(
                train_dates,
                val_dates.min(),
                cfg.max_target_horizon_days,
            )

        if not train_dates.empty:
            _assert_fold_is_valid(train_dates, val_dates, cfg)
            yield train_dates, val_dates

        train_end_bound += pd.DateOffset(months=cfg.step_months)


def _purge_overlapping_target_dates(
    train_dates: pd.DatetimeIndex,
    val_start: pd.Timestamp,
    target_horizon_days: int,
) -> pd.DatetimeIndex:
    target_ends = train_dates + pd.offsets.BDay(target_horizon_days)
    return train_dates[target_ends < val_start]


def _assert_fold_is_valid(
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    config: WalkForwardCVConfig,
) -> None:
    overlap = train_dates.intersection(val_dates)
    if len(overlap) > 0:
        msg = f"train/validation date overlap detected: {overlap[:3].tolist()}"
        raise ValueError(msg)
    min_val_start = train_dates.max() + pd.offsets.BDay(config.embargo_days)
    if val_dates.min() < min_val_start:
        msg = "validation fold starts before the configured embargo gap"
        raise ValueError(msg)
