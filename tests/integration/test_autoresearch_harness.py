from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from stock_analysis.ml.autoresearch_eval import (
    AutoresearchEvalConfig,
    append_result_tsv,
    evaluate_candidate,
)


def test_autoresearch_eval_runs_on_synthetic_phase1_artifacts(tmp_path: Path) -> None:
    run_root = _write_synthetic_artifacts(tmp_path)

    result = evaluate_candidate(
        AutoresearchEvalConfig(
            candidate_id="ridge_rank",
            input_run_root=run_root,
            max_assets=None,
            max_rebalances=5,
            optimizer_max_weight=0.6,
            risk_aversion=0.1,
            embargo_days=5,
            covariance_lookback_days=20,
        )
    )
    ledger = tmp_path / "results.tsv"
    append_result_tsv(ledger, result)

    comparison = result["metrics"]["comparison"]
    assert result["candidate"]["candidate_id"] == "ridge_rank"
    assert comparison["candidate_sharpe"] is not None
    assert comparison["spy_sharpe"] is not None
    assert comparison["active_return"] is not None
    assert comparison["tracking_error"] is not None
    assert comparison["information_ratio"] is not None
    assert comparison["ir_observations"] >= 2
    assert result["decision"]["status"] in {"rejected", "provisional", "go"}

    rows = list(csv.DictReader(ledger.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "ridge_rank"
    assert rows[0]["ir_observations"]


def _write_synthetic_artifacts(tmp_path: Path) -> Path:
    run_root = tmp_path / "data" / "runs" / "phase1-run"
    (run_root / "silver").mkdir(parents=True)
    (run_root / "gold").mkdir(parents=True)

    dates = pd.bdate_range("2025-01-01", periods=70)
    tickers = ["AAA", "BBB", "CCC"]
    panel_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []

    for day_idx, current_date in enumerate(dates):
        benchmark_rows.append(
            {
                "date": current_date.date().isoformat(),
                "horizon_days": 5,
                "spy_return": 0.002 + (day_idx % 4) * 0.0005,
            }
        )
        for ticker_idx, ticker in enumerate(tickers):
            strength = 0.03 - ticker_idx * 0.01 + day_idx * 0.0001
            panel_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "security": ticker,
                    "gics_sector": "Sector",
                    "momentum_21d": strength,
                    "volatility_21d": 0.18 + ticker_idx * 0.01,
                    "return_5d": strength / 2,
                    "dollar_volume_21d": 10_000_000 - ticker_idx * 100_000,
                }
            )
            label_rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "fwd_return_5d": 0.012 - ticker_idx * 0.002 + (day_idx % 3) * 0.0005,
                    "fwd_rank_5d": len(tickers) - ticker_idx,
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
        run_root / "silver" / "asset_daily_features_panel.parquet",
        index=False,
    )
    pd.DataFrame(label_rows).to_parquet(run_root / "gold" / "labels_panel.parquet", index=False)
    pd.DataFrame(return_rows).to_parquet(
        run_root / "silver" / "asset_daily_returns.parquet",
        index=False,
    )
    pd.DataFrame(benchmark_rows).to_parquet(
        run_root / "silver" / "benchmark_returns.parquet",
        index=False,
    )
    return run_root
