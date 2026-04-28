from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

CashflowType = Literal[
    "deposit",
    "withdrawal",
    "dividend",
    "interest",
    "fee",
    "tax",
    "transfer",
]


@dataclass(frozen=True)
class AccountRecord:
    slug: str
    display_name: str
    id: str | None = None
    owner_id: str | None = None
    base_currency: str = "USD"
    benchmark_ticker: str = "SPY"


@dataclass(frozen=True)
class CashflowRecord:
    account_id: str
    cashflow_date: date
    amount: float
    cashflow_type: CashflowType
    currency: str = "USD"
    settled_date: date | None = None
    source: str = "manual"
    external_ref: str | None = None
    notes: str | None = None
    included_in_snapshot_id: str | None = None
    id: str | None = None


@dataclass(frozen=True)
class PortfolioSnapshotRecord:
    account_id: str
    snapshot_date: date
    market_value: float
    cash_balance: float
    total_value: float
    currency: str = "USD"
    source: str = "manual"
    id: str | None = None


@dataclass(frozen=True)
class HoldingSnapshotRecord:
    snapshot_id: str
    ticker: str
    market_value: float
    quantity: float | None = None
    price: float | None = None
    currency: str = "USD"
    id: str | None = None


@dataclass(frozen=True)
class RecommendationRunRecord:
    account_id: str
    run_id: str
    as_of_date: date
    data_as_of_date: date
    model_version: str
    ml_score_scale: float
    config_hash: str
    expected_return_is_calibrated: bool = False
    optimizer_return_unit: str | None = None
    calibration_enabled: bool = False
    calibration_method: str | None = None
    calibration_target: str | None = None
    calibration_model_version: str | None = None
    calibration_status: str | None = None
    calibration_trained_through_date: date | None = None
    calibration_observations: int | None = None
    calibration_mae: float | None = None
    calibration_rmse: float | None = None
    calibration_rank_ic: float | None = None
    status: str = "completed"
    id: str | None = None


@dataclass(frozen=True)
class RecommendationLineRecord:
    recommendation_run_id: str
    ticker: str
    security: str | None = None
    gics_sector: str | None = None
    current_weight: float | None = None
    target_weight: float | None = None
    trade_weight: float | None = None
    trade_notional: float | None = None
    commission_amount: float | None = None
    cash_required_weight: float | None = None
    cash_released_weight: float | None = None
    deposit_used_amount: float | None = None
    cash_after_trade_amount: float | None = None
    action: str | None = None
    reason_code: str | None = None
    forecast_score: float | None = None
    expected_return: float | None = None
    calibrated_expected_return: float | None = None
    expected_return_is_calibrated: bool | None = None
    volatility: float | None = None
    forecast_horizon_days: int | None = None
    forecast_start_date: date | None = None
    forecast_end_date: date | None = None
    realized_return: float | None = None
    realized_spy_return: float | None = None
    realized_active_return: float | None = None
    forecast_error: float | None = None
    forecast_hit: bool | None = None
    outcome_status: str | None = None
    id: str | None = None


@dataclass(frozen=True)
class PerformanceSnapshotRecord:
    account_id: str
    as_of_date: date
    account_total_value: float
    total_deposits: float
    net_external_cashflow: float
    initial_value: float | None = None
    invested_capital: float | None = None
    return_on_invested_capital: float | None = None
    account_time_weighted_return: float | None = None
    account_money_weighted_return: float | None = None
    spy_same_cashflow_value: float | None = None
    spy_time_weighted_return: float | None = None
    spy_money_weighted_return: float | None = None
    active_value: float | None = None
    active_return: float | None = None
    id: str | None = None


class AccountTrackingRepository(Protocol):
    def get_account_by_slug(self, slug: str) -> AccountRecord | None: ...

    def upsert_account(self, account: AccountRecord) -> AccountRecord: ...

    def insert_cashflow(self, cashflow: CashflowRecord) -> CashflowRecord: ...

    def list_cashflows(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[CashflowRecord]: ...

    def insert_portfolio_snapshot(
        self,
        snapshot: PortfolioSnapshotRecord,
        holdings: list[HoldingSnapshotRecord] | None = None,
    ) -> PortfolioSnapshotRecord: ...

    def latest_portfolio_snapshot(
        self,
        account_id: str,
        *,
        as_of_date: date,
    ) -> PortfolioSnapshotRecord | None: ...

    def list_portfolio_snapshots(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[PortfolioSnapshotRecord]: ...

    def list_holding_snapshots(self, snapshot_id: str) -> list[HoldingSnapshotRecord]: ...

    def insert_recommendation_run(
        self, run: RecommendationRunRecord
    ) -> RecommendationRunRecord: ...

    def insert_recommendation_lines(
        self,
        lines: list[RecommendationLineRecord],
    ) -> list[RecommendationLineRecord]: ...

    def list_recommendation_runs(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[RecommendationRunRecord]: ...

    def list_recommendation_lines(
        self,
        recommendation_run_ids: list[str],
    ) -> list[RecommendationLineRecord]: ...

    def insert_performance_snapshot(
        self,
        snapshot: PerformanceSnapshotRecord,
    ) -> PerformanceSnapshotRecord: ...

    def list_performance_snapshots(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[PerformanceSnapshotRecord]: ...
