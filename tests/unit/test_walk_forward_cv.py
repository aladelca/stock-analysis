from __future__ import annotations

import pandas as pd
import pytest

from stock_analysis.ml.cv import WalkForwardCVConfig, walk_forward_splits


def test_walk_forward_splits_respect_embargo_and_no_overlap() -> None:
    dates = pd.bdate_range("2020-01-01", "2025-12-31")
    config = WalkForwardCVConfig(
        train_window_years=3,
        val_window_months=6,
        step_months=6,
        embargo_days=15,
        max_target_horizon_days=5,
        safety_margin_days=5,
    )

    folds = list(walk_forward_splits(dates, config))

    assert folds
    for train_dates, val_dates in folds:
        assert len(train_dates.intersection(val_dates)) == 0
        assert val_dates.min() >= train_dates.max() + pd.offsets.BDay(15)
        assert (train_dates + pd.offsets.BDay(5) < val_dates.min()).all()


def test_walk_forward_config_rejects_short_embargo() -> None:
    with pytest.raises(ValueError, match="embargo_days"):
        WalkForwardCVConfig(embargo_days=5, max_target_horizon_days=5, safety_margin_days=5)
