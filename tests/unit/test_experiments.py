from __future__ import annotations

import json

import pandas as pd

from stock_analysis.ml.experiments import run_experiment_from_config


def test_run_experiment_from_config_builds_heuristic_artifacts(tmp_path) -> None:
    run_root = tmp_path / "data" / "runs" / "phase1-run"
    (run_root / "silver").mkdir(parents=True)
    (run_root / "gold").mkdir(parents=True)
    dates = pd.bdate_range("2025-01-01", periods=45)
    tickers = ["AAA", "BBB", "CCC"]

    panel_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    for ticker_idx, ticker in enumerate(tickers):
        for current_date in dates:
            panel_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "momentum_5d": 0.1 - ticker_idx * 0.01,
                    "volatility_21d": 0.2,
                    "security": ticker,
                    "gics_sector": "Sector",
                }
            )
            label_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "fwd_return_5d": 0.01 + ticker_idx * 0.001,
                }
            )
            return_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "return_1d": 0.001 + ticker_idx * 0.0001,
                }
            )

    pd.DataFrame(panel_rows).to_parquet(
        run_root / "silver" / "asset_daily_features_panel.parquet", index=False
    )
    pd.DataFrame(label_rows).to_parquet(run_root / "gold" / "labels_panel.parquet", index=False)
    pd.DataFrame(return_rows).to_parquet(
        run_root / "silver" / "asset_daily_returns.parquet", index=False
    )

    config_path = tmp_path / "e0_heuristic.yaml"
    config_path.write_text(
        f"""
experiment_id: e0_heuristic
model: heuristic
input_run_root: {run_root}
tracking_root: {tmp_path / "experiments"}
horizon_days: 5
target_column: fwd_return_5d
momentum_column: momentum_5d
volatility_column: volatility_21d
volatility_penalty: 0.25
bootstrap_samples: 10
run_backtest: true
backtest:
  horizon_days: 5
  embargo_days: 10
  covariance_lookback_days: 20
  max_rebalances: 3
  feature_columns: [momentum_5d, volatility_21d]
optimizer:
  max_weight: 0.6
  risk_aversion: 0.1
  lambda_turnover: 0.001
""",
        encoding="utf-8",
    )

    run_dir = run_experiment_from_config(config_path)

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "predictions.parquet").exists()
    assert (run_dir / "backtest.parquet").exists()
    assert (run_dir / "metrics.json").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "predictive" in metrics
    assert "portfolio" in metrics
    index = pd.read_parquet(tmp_path / "experiments" / "experiments_index.parquet")
    assert index["experiment_id"].tolist() == ["e0_heuristic"]
