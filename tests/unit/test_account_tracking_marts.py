from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stock_analysis.portfolio.holdings import PortfolioState
from stock_analysis.portfolio.live_state import LivePortfolioState
from stock_analysis.storage.contracts import (
    AccountRecord,
    CashflowRecord,
    HoldingSnapshotRecord,
    PortfolioSnapshotRecord,
)
from stock_analysis.tableau.account_tracking_marts import build_account_tracking_marts
from stock_analysis.tableau.dashboard_mart import build_dashboard_mart


def test_account_tracking_marts_include_cashflows_performance_and_recommendations() -> None:
    live_state = _live_state()
    tables = build_account_tracking_marts(
        live_state=live_state,
        recommendations=_recommendations(),
        run_metadata=_run_metadata(),
        spy_daily=_spy_daily(),
        commission_rate=0.0,
    )

    assert set(tables) == {
        "cashflows",
        "portfolio_snapshots",
        "holding_snapshots",
        "recommendation_runs",
        "recommendation_lines",
        "performance_snapshots",
    }
    assert tables["cashflows"]["is_applied_to_recommendation"].tolist() == [False, True]
    assert tables["recommendation_runs"]["unapplied_cashflow_amount"].iat[0] == pytest.approx(100.0)
    assert tables["recommendation_lines"]["recommendation_key"].iat[0] == "run-1:AAPL"
    assert tables["recommendation_lines"]["forecast_score"].iat[0] == pytest.approx(0.2)
    assert tables["recommendation_lines"]["forecast_horizon_days"].iat[0] == 5
    assert tables["recommendation_lines"]["outcome_status"].iat[0] == "pending"

    performance = tables["performance_snapshots"].set_index("as_of_date")
    assert performance.loc["2026-01-02", "initial_value"] == pytest.approx(1000.0)
    assert performance.loc["2026-01-02", "invested_capital"] == pytest.approx(1000.0)
    assert performance.loc["2026-01-10", "total_deposits"] == pytest.approx(100.0)
    assert performance.loc["2026-01-10", "initial_value"] == pytest.approx(1000.0)
    assert performance.loc["2026-01-10", "invested_capital"] == pytest.approx(1100.0)
    assert performance.loc["2026-01-10", "return_on_invested_capital"] == pytest.approx(0.1)
    assert performance.loc["2026-01-10", "account_time_weighted_return"] == pytest.approx(0.11)
    assert performance.loc["2026-01-10", "spy_same_cashflow_value"] > 1000.0


def test_spy_same_cashflow_benchmark_maps_weekend_cashflows_to_next_trading_day() -> None:
    tables = build_account_tracking_marts(
        live_state=_live_state(first_cashflow_date=date(2026, 1, 3)),
        recommendations=_recommendations(),
        run_metadata=_run_metadata(),
        spy_daily=_spy_daily(),
        commission_rate=0.0,
    )

    performance = tables["performance_snapshots"].set_index("as_of_date")
    assert performance.loc["2026-01-10", "spy_same_cashflow_value"] == pytest.approx(1133.3311)


def test_dashboard_mart_exposes_latest_account_performance_fields() -> None:
    tables = build_account_tracking_marts(
        live_state=_live_state(),
        recommendations=_recommendations(),
        run_metadata=_run_metadata(),
        spy_daily=_spy_daily(),
        commission_rate=0.0,
    )

    mart = build_dashboard_mart(
        _recommendations(),
        _risk_metrics(),
        _sector_exposure(),
        _run_metadata(),
        performance_snapshots=tables["performance_snapshots"],
    )

    assert "account_total_value" in mart.columns
    assert mart["account_total_value"].iat[0] == pytest.approx(1210.0)
    assert mart["account_initial_value"].iat[0] == pytest.approx(1000.0)
    assert mart["account_invested_capital"].iat[0] == pytest.approx(1100.0)
    assert mart["run_live_cashflow_source"].iat[0] == "actual"


def test_dashboard_mart_ignores_solver_dust_for_selected_rows() -> None:
    recommendations = pd.concat(
        [
            _recommendations(),
            pd.DataFrame(
                [
                    {
                        **_recommendations().iloc[0].to_dict(),
                        "ticker": "DUST",
                        "target_weight": 1e-10,
                        "trade_weight": 1e-10,
                        "trade_abs_weight": 1e-10,
                        "rebalance_required": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    mart = build_dashboard_mart(
        recommendations,
        _risk_metrics(),
        _sector_exposure(),
        _run_metadata(),
    )
    by_ticker = mart.set_index("ticker")

    assert bool(by_ticker.loc["DUST", "is_solver_dust"]) is True
    assert bool(by_ticker.loc["DUST", "selected"]) is False
    assert by_ticker.loc["DUST", "display_target_weight"] == pytest.approx(0.0)


def _live_state(*, first_cashflow_date: date = date(2026, 1, 5)) -> LivePortfolioState:
    account = AccountRecord(
        id="account-1",
        slug="main",
        display_name="Main",
        benchmark_ticker="SPY",
    )
    snapshots = [
        PortfolioSnapshotRecord(
            id="snapshot-1",
            account_id="account-1",
            snapshot_date=date(2026, 1, 2),
            market_value=900.0,
            cash_balance=100.0,
            total_value=1000.0,
        ),
        PortfolioSnapshotRecord(
            id="snapshot-2",
            account_id="account-1",
            snapshot_date=date(2026, 1, 10),
            market_value=1100.0,
            cash_balance=110.0,
            total_value=1210.0,
        ),
    ]
    cashflows = [
        CashflowRecord(
            id="cashflow-1",
            account_id="account-1",
            cashflow_date=first_cashflow_date,
            amount=100.0,
            cashflow_type="deposit",
            included_in_snapshot_id="snapshot-2",
        ),
        CashflowRecord(
            id="cashflow-2",
            account_id="account-1",
            cashflow_date=date(2026, 1, 12),
            amount=100.0,
            cashflow_type="deposit",
        ),
    ]
    return LivePortfolioState(
        account=account,
        snapshot=snapshots[-1],
        state=PortfolioState(
            weights=pd.Series({"AAPL": 1100.0 / 1210.0}, name="current_weight"),
            market_values=pd.Series({"AAPL": 1100.0}, name="market_value"),
            cash_balance=110.0,
            portfolio_value=1210.0,
        ),
        contribution_amount=100.0,
        applied_cashflows=[cashflows[-1]],
        cashflows=cashflows,
        snapshots=snapshots,
        holdings=[
            HoldingSnapshotRecord(
                snapshot_id="snapshot-2",
                ticker="AAPL",
                market_value=1100.0,
            )
        ],
    )


def _recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "as_of_date": "2026-01-10",
                "ticker": "AAPL",
                "security": "Apple",
                "gics_sector": "Information Technology",
                "forecast_score": 0.2,
                "expected_return": 0.02,
                "calibrated_expected_return": pd.NA,
                "expected_return_is_calibrated": False,
                "forecast_horizon_days": 5,
                "forecast_start_date": "2026-01-10",
                "outcome_status": "pending",
                "volatility": 0.2,
                "current_weight": 0.9,
                "target_weight": 0.5,
                "trade_weight": -0.4,
                "trade_abs_weight": 0.4,
                "estimated_commission_weight": 0.0,
                "net_trade_weight_after_commission": -0.4,
                "cash_required_weight": 0.0,
                "cash_released_weight": 0.4,
                "portfolio_value_before_contribution": 1210.0,
                "contribution_amount": 100.0,
                "portfolio_value_after_contribution": 1310.0,
                "current_market_value": 1100.0,
                "target_market_value": 655.0,
                "trade_notional": -445.0,
                "commission_amount": 0.0,
                "deposit_used_amount": 0.0,
                "cash_after_trade_amount": 555.0,
                "no_trade_band_applied": False,
                "rebalance_required": True,
                "action": "SELL",
                "reason_code": "rebalance",
            }
        ]
    )


def _risk_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "expected_return", "value": 0.02},
            {"metric": "expected_volatility", "value": 0.2},
            {"metric": "num_holdings", "value": 1},
        ]
    )


def _sector_exposure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gics_sector": "Information Technology",
                "target_weight": 0.5,
            }
        ]
    )


def _run_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "requested_as_of_date": "2026-01-10",
                "data_as_of_date": "2026-01-10",
                "model_version": "e8-scale-0p5-contribution-aware-v1",
                "config_hash": "abcdef",
                "created_at_utc": "2026-01-10T00:00:00Z",
                "live_account_enabled": True,
                "live_account_slug": "main",
                "live_cashflow_source": "actual",
                "live_snapshot_id": "snapshot-2",
                "live_snapshot_date": "2026-01-10",
                "live_unapplied_cashflow_amount": 100.0,
                "live_unapplied_cashflow_count": 1,
            }
        ]
    )


def _spy_daily() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-01-02", "adj_close": 100.0, "return_1d": 0.0},
            {"date": "2026-01-05", "adj_close": 101.0, "return_1d": 0.01},
            {"date": "2026-01-06", "adj_close": 102.0, "return_1d": 0.01},
            {"date": "2026-01-10", "adj_close": 103.0, "return_1d": 0.01},
        ]
    )
