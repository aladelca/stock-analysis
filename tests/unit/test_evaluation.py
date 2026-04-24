from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analysis.ml.evaluation import EvaluationConfig, evaluate


def test_evaluate_reports_reproducible_predictive_metrics() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=50)
    prediction = rng.normal(size=len(dates) * 5)
    target = 0.05 * prediction + np.sqrt(1 - 0.05**2) * rng.normal(size=len(prediction))
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates, 5),
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"] * len(dates),
            "prediction": prediction,
            "target": target,
        }
    )

    metrics = evaluate(
        frame,
        config=EvaluationConfig(bootstrap_samples=100, random_seed=7),
    )
    repeated = evaluate(
        frame,
        config=EvaluationConfig(bootstrap_samples=100, random_seed=7),
    )

    assert metrics["predictive"]["pearson_ic"] == pytest.approx(
        frame["prediction"].corr(frame["target"])
    )
    assert metrics["predictive"]["pearson_ic_ci_95"] == repeated["predictive"]["pearson_ic_ci_95"]
    assert "rank_ic" in metrics["predictive"]
