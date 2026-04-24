from __future__ import annotations

import pandas as pd

from stock_analysis.backtest.runner import BacktestConfig
from stock_analysis.config import OptimizerConfig
from stock_analysis.ml.phase2 import Phase2Config, run_phase2


def test_run_phase2_writes_reports_and_tracking_artifacts(tmp_path) -> None:
    run_root = tmp_path / "data" / "runs" / "phase1-run"
    (run_root / "silver").mkdir(parents=True)
    (run_root / "gold").mkdir(parents=True)
    dates = pd.bdate_range("2025-01-01", periods=50)
    tickers = ["AAA", "BBB", "CCC"]

    panel_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    for current_date in dates:
        benchmark_rows.append(
            {
                "date": current_date.date().isoformat(),
                "horizon_days": 5,
                "spy_return": 0.004,
            }
        )
        for ticker_idx, ticker in enumerate(tickers):
            panel_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "momentum_21d": 0.10 - ticker_idx * 0.01,
                    "momentum_252d": 0.12 - ticker_idx * 0.01,
                    "volatility_63d": 0.2,
                    "return_5d": 0.01,
                    "security": ticker,
                    "gics_sector": "Sector",
                }
            )
            label_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "fwd_return_5d": 0.01 + ticker_idx * 0.001,
                    "fwd_rank_5d": ticker_idx + 1,
                    "fwd_is_top_tercile_5d": int(ticker_idx == 2),
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
    pd.DataFrame(benchmark_rows).to_parquet(
        run_root / "silver" / "benchmark_returns.parquet", index=False
    )

    summary = run_phase2(
        Phase2Config(
            input_run_root=run_root,
            output_dir=tmp_path / "reports",
            tracking_root=tmp_path / "experiments",
            experiments=("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"),
            bootstrap_samples=10,
            feature_columns=("momentum_21d", "volatility_63d", "return_5d"),
            optimizer=OptimizerConfig(max_weight=0.6, risk_aversion=0.1),
            run_sweeps=False,
            lightgbm_nested_cv=False,
            backtest=BacktestConfig(
                horizon_days=5,
                embargo_days=10,
                covariance_lookback_days=20,
                max_rebalances=3,
            ),
        )
    )

    assert set(summary["experiment_id"]) == {
        "E0",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "E8",
    }
    assert (tmp_path / "reports" / "phase2-report.md").exists()
    detailed_report = (tmp_path / "reports" / "phase2-detailed-summary.md").read_text(
        encoding="utf-8"
    )
    assert "SPY-Relative IR Calculation" in detailed_report
    assert "Experimentation Outcome" in detailed_report
    assert (tmp_path / "reports" / "e0.md").exists()
    assert (tmp_path / "experiments" / "e0" / "backtest.parquet").exists()
    e0_dates = (
        pd.read_parquet(tmp_path / "experiments" / "e0" / "backtest.parquet")["rebalance_date"]
        .drop_duplicates()
        .tolist()
    )
    e1_dates = pd.read_parquet(tmp_path / "experiments" / "e1" / "backtest.parquet")[
        "rebalance_date"
    ].tolist()
    assert e1_dates == e0_dates
