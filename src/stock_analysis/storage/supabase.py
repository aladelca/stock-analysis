from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import date
from typing import Any, cast

from stock_analysis.config import SupabaseConfig
from stock_analysis.storage.contracts import (
    AccountRecord,
    CashflowRecord,
    CashflowType,
    HoldingSnapshotRecord,
    PerformanceSnapshotRecord,
    PortfolioSnapshotRecord,
    RecommendationLineRecord,
    RecommendationRunRecord,
)

SUPABASE_PAGE_SIZE = 1000


class SupabaseConfigError(RuntimeError):
    """Raised when Supabase is enabled but credentials or dependencies are missing."""


class SupabaseRepositoryError(RuntimeError):
    """Raised when Supabase returns an unexpected repository response."""


def create_account_tracking_repository(
    config: SupabaseConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> SupabaseAccountTrackingRepository:
    if not config.enabled:
        msg = "Supabase is disabled in config."
        raise SupabaseConfigError(msg)
    client = create_supabase_client(config, environ=environ)
    return SupabaseAccountTrackingRepository(client, config)


def create_supabase_client(
    config: SupabaseConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> Any:
    env = os.environ if environ is None else environ
    url = env.get(config.url_env)
    key = env.get(config.key_env)
    if not url:
        msg = f"Missing Supabase URL environment variable: {config.url_env}"
        raise SupabaseConfigError(msg)
    if not key:
        msg = f"Missing Supabase key environment variable: {config.key_env}"
        raise SupabaseConfigError(msg)
    try:
        from supabase import create_client  # type: ignore[attr-defined]
    except ImportError as exc:
        msg = (
            "Supabase support requires the optional supabase extra. Run `uv sync --extra supabase`."
        )
        raise SupabaseConfigError(msg) from exc
    return create_client(url, key)


class SupabaseAccountTrackingRepository:
    def __init__(self, client: Any, config: SupabaseConfig) -> None:
        self._client = client
        self._config = config

    def get_account_by_slug(self, slug: str) -> AccountRecord | None:
        response = _execute(
            self._table(self._config.accounts_table).select("*").eq("slug", slug).limit(1)
        )
        if not response.data:
            return None
        return _account_from_row(response.data[0])

    def upsert_account(self, account: AccountRecord) -> AccountRecord:
        response = _execute(
            self._table(self._config.accounts_table).upsert(
                _record_payload(account),
                on_conflict="slug",
            )
        )
        return _account_from_row(_single_response_row(response, self._config.accounts_table))

    def insert_cashflow(self, cashflow: CashflowRecord) -> CashflowRecord:
        response = _execute(
            self._table(self._config.cashflows_table).insert(_record_payload(cashflow))
        )
        return _cashflow_from_row(_single_response_row(response, self._config.cashflows_table))

    def list_cashflows(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[CashflowRecord]:
        query = self._table(self._config.cashflows_table).select("*").eq("account_id", account_id)
        if start_date is not None:
            query = query.gte("cashflow_date", start_date.isoformat())
        if end_date is not None:
            query = query.lte("cashflow_date", end_date.isoformat())
        rows = _execute_all(query.order("cashflow_date").order("created_at"))
        return [_cashflow_from_row(row) for row in rows]

    def insert_portfolio_snapshot(
        self,
        snapshot: PortfolioSnapshotRecord,
        holdings: list[HoldingSnapshotRecord] | None = None,
    ) -> PortfolioSnapshotRecord:
        response = _execute(
            self._table(self._config.portfolio_snapshots_table).insert(_record_payload(snapshot))
        )
        inserted = _portfolio_snapshot_from_row(
            _single_response_row(response, self._config.portfolio_snapshots_table)
        )
        if holdings:
            if inserted.id is None:
                msg = "Inserted portfolio snapshot did not return an id."
                raise SupabaseRepositoryError(msg)
            rows = [
                _record_payload(replace(holding, snapshot_id=inserted.id)) for holding in holdings
            ]
            try:
                _execute(self._table(self._config.holding_snapshots_table).insert(rows))
            except SupabaseRepositoryError as exc:
                try:
                    _execute(
                        self._table(self._config.portfolio_snapshots_table)
                        .delete()
                        .eq("id", inserted.id)
                    )
                except SupabaseRepositoryError as cleanup_exc:
                    msg = (
                        "Failed to insert holding snapshots after inserting portfolio snapshot, "
                        "and cleanup of the partial snapshot also failed."
                    )
                    raise SupabaseRepositoryError(msg) from cleanup_exc
                msg = (
                    "Failed to insert holding snapshots after inserting portfolio snapshot; "
                    "deleted the partial portfolio snapshot."
                )
                raise SupabaseRepositoryError(msg) from exc
        return inserted

    def latest_portfolio_snapshot(
        self,
        account_id: str,
        *,
        as_of_date: date,
    ) -> PortfolioSnapshotRecord | None:
        response = _execute(
            self._table(self._config.portfolio_snapshots_table)
            .select("*")
            .eq("account_id", account_id)
            .lte("snapshot_date", as_of_date.isoformat())
            .order("snapshot_date", desc=True)
            .limit(1)
        )
        if not response.data:
            return None
        return _portfolio_snapshot_from_row(response.data[0])

    def list_portfolio_snapshots(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[PortfolioSnapshotRecord]:
        query = (
            self._table(self._config.portfolio_snapshots_table)
            .select("*")
            .eq("account_id", account_id)
        )
        if start_date is not None:
            query = query.gte("snapshot_date", start_date.isoformat())
        if end_date is not None:
            query = query.lte("snapshot_date", end_date.isoformat())
        rows = _execute_all(query.order("snapshot_date").order("created_at"))
        return [_portfolio_snapshot_from_row(row) for row in rows]

    def list_holding_snapshots(self, snapshot_id: str) -> list[HoldingSnapshotRecord]:
        rows = _execute_all(
            self._table(self._config.holding_snapshots_table)
            .select("*")
            .eq("snapshot_id", snapshot_id)
            .order("ticker")
        )
        return [_holding_snapshot_from_row(row) for row in rows]

    def insert_recommendation_run(self, run: RecommendationRunRecord) -> RecommendationRunRecord:
        response = _execute(
            self._table(self._config.recommendation_runs_table).upsert(
                _record_payload(run),
                on_conflict="run_id",
            )
        )
        return _recommendation_run_from_row(
            _single_response_row(response, self._config.recommendation_runs_table)
        )

    def insert_recommendation_lines(
        self,
        lines: list[RecommendationLineRecord],
    ) -> list[RecommendationLineRecord]:
        if not lines:
            return []
        response = _execute(
            self._table(self._config.recommendation_lines_table).upsert(
                [_record_payload(line) for line in lines],
                on_conflict="recommendation_run_id,ticker",
            )
        )
        return [_recommendation_line_from_row(row) for row in response.data or []]

    def list_recommendation_runs(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[RecommendationRunRecord]:
        query = (
            self._table(self._config.recommendation_runs_table)
            .select("*")
            .eq("account_id", account_id)
        )
        if start_date is not None:
            query = query.gte("data_as_of_date", start_date.isoformat())
        if end_date is not None:
            query = query.lte("data_as_of_date", end_date.isoformat())
        rows = _execute_all(query.order("data_as_of_date").order("created_at"))
        return [_recommendation_run_from_row(row) for row in rows]

    def list_recommendation_lines(
        self,
        recommendation_run_ids: list[str],
    ) -> list[RecommendationLineRecord]:
        if not recommendation_run_ids:
            return []
        rows = _execute_all(
            self._table(self._config.recommendation_lines_table)
            .select("*")
            .in_("recommendation_run_id", recommendation_run_ids)
            .order("ticker")
        )
        return [_recommendation_line_from_row(row) for row in rows]

    def insert_performance_snapshot(
        self,
        snapshot: PerformanceSnapshotRecord,
    ) -> PerformanceSnapshotRecord:
        response = _execute(
            self._table(self._config.performance_snapshots_table).upsert(
                _record_payload(snapshot),
                on_conflict="account_id,as_of_date",
            )
        )
        return _performance_snapshot_from_row(
            _single_response_row(response, self._config.performance_snapshots_table)
        )

    def list_performance_snapshots(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[PerformanceSnapshotRecord]:
        query = (
            self._table(self._config.performance_snapshots_table)
            .select("*")
            .eq("account_id", account_id)
        )
        if start_date is not None:
            query = query.gte("as_of_date", start_date.isoformat())
        if end_date is not None:
            query = query.lte("as_of_date", end_date.isoformat())
        rows = _execute_all(query.order("as_of_date").order("created_at"))
        return [_performance_snapshot_from_row(row) for row in rows]

    def _table(self, name: str) -> Any:
        if self._config.schema_name == "public":
            return self._client.table(name)
        return self._client.schema(self._config.schema_name).table(name)


def _execute(query: Any) -> Any:
    try:
        return query.execute()
    except Exception as exc:
        msg = f"Supabase query failed: {exc}"
        raise SupabaseRepositoryError(msg) from exc


def _execute_all(query: Any, *, page_size: int | None = None) -> list[Mapping[str, Any]]:
    page_size = SUPABASE_PAGE_SIZE if page_size is None else page_size
    rows: list[Mapping[str, Any]] = []
    start = 0
    while True:
        response = _execute(query.range(start, start + page_size - 1))
        batch = [cast(Mapping[str, Any], row) for row in response.data or []]
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        start += page_size


def _single_response_row(response: Any, table_name: str) -> Mapping[str, Any]:
    data = response.data or []
    if len(data) != 1:
        msg = f"Expected one Supabase row from {table_name}, got {len(data)}."
        raise SupabaseRepositoryError(msg)
    return cast(Mapping[str, Any], data[0])


def _record_payload(record: Any) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in asdict(record).items():
        if value is None:
            continue
        if isinstance(value, date):
            payload[key] = value.isoformat()
        else:
            payload[key] = value
    return payload


def _account_from_row(row: Mapping[str, Any]) -> AccountRecord:
    return AccountRecord(
        id=_optional_str(row.get("id")),
        owner_id=_optional_str(row.get("owner_id")),
        slug=str(row["slug"]),
        display_name=str(row["display_name"]),
        base_currency=str(row.get("base_currency") or "USD"),
        benchmark_ticker=str(row.get("benchmark_ticker") or "SPY"),
    )


def _cashflow_from_row(row: Mapping[str, Any]) -> CashflowRecord:
    return CashflowRecord(
        id=_optional_str(row.get("id")),
        account_id=str(row["account_id"]),
        cashflow_date=_date_from_value(row["cashflow_date"]),
        settled_date=_optional_date(row.get("settled_date")),
        amount=float(row["amount"]),
        currency=str(row.get("currency") or "USD"),
        cashflow_type=cast(CashflowType, str(row["cashflow_type"])),
        source=str(row.get("source") or "manual"),
        external_ref=_optional_str(row.get("external_ref")),
        notes=_optional_str(row.get("notes")),
        included_in_snapshot_id=_optional_str(row.get("included_in_snapshot_id")),
    )


def _portfolio_snapshot_from_row(row: Mapping[str, Any]) -> PortfolioSnapshotRecord:
    return PortfolioSnapshotRecord(
        id=_optional_str(row.get("id")),
        account_id=str(row["account_id"]),
        snapshot_date=_date_from_value(row["snapshot_date"]),
        market_value=float(row["market_value"]),
        cash_balance=float(row["cash_balance"]),
        total_value=float(row["total_value"]),
        currency=str(row.get("currency") or "USD"),
        source=str(row.get("source") or "manual"),
    )


def _holding_snapshot_from_row(row: Mapping[str, Any]) -> HoldingSnapshotRecord:
    return HoldingSnapshotRecord(
        id=_optional_str(row.get("id")),
        snapshot_id=str(row["snapshot_id"]),
        ticker=str(row["ticker"]),
        quantity=_optional_float(row.get("quantity")),
        market_value=float(row["market_value"]),
        price=_optional_float(row.get("price")),
        currency=str(row.get("currency") or "USD"),
    )


def _recommendation_run_from_row(row: Mapping[str, Any]) -> RecommendationRunRecord:
    return RecommendationRunRecord(
        id=_optional_str(row.get("id")),
        account_id=str(row["account_id"]),
        run_id=str(row["run_id"]),
        as_of_date=_date_from_value(row["as_of_date"]),
        data_as_of_date=_date_from_value(row["data_as_of_date"]),
        model_version=str(row["model_version"]),
        ml_score_scale=float(row["ml_score_scale"]),
        config_hash=str(row["config_hash"]),
        expected_return_is_calibrated=_optional_bool(row.get("expected_return_is_calibrated"))
        or False,
        status=str(row.get("status") or "completed"),
    )


def _recommendation_line_from_row(row: Mapping[str, Any]) -> RecommendationLineRecord:
    return RecommendationLineRecord(
        id=_optional_str(row.get("id")),
        recommendation_run_id=str(row["recommendation_run_id"]),
        ticker=str(row["ticker"]),
        security=_optional_str(row.get("security")),
        gics_sector=_optional_str(row.get("gics_sector")),
        current_weight=_optional_float(row.get("current_weight")),
        target_weight=_optional_float(row.get("target_weight")),
        trade_weight=_optional_float(row.get("trade_weight")),
        trade_notional=_optional_float(row.get("trade_notional")),
        commission_amount=_optional_float(row.get("commission_amount")),
        cash_required_weight=_optional_float(row.get("cash_required_weight")),
        cash_released_weight=_optional_float(row.get("cash_released_weight")),
        deposit_used_amount=_optional_float(row.get("deposit_used_amount")),
        cash_after_trade_amount=_optional_float(row.get("cash_after_trade_amount")),
        action=_optional_str(row.get("action")),
        reason_code=_optional_str(row.get("reason_code")),
        forecast_score=_optional_float(row.get("forecast_score")),
        expected_return=_optional_float(row.get("expected_return")),
        calibrated_expected_return=_optional_float(row.get("calibrated_expected_return")),
        expected_return_is_calibrated=_optional_bool(row.get("expected_return_is_calibrated")),
        volatility=_optional_float(row.get("volatility")),
        forecast_horizon_days=_optional_int(row.get("forecast_horizon_days")),
        forecast_start_date=_optional_date(row.get("forecast_start_date")),
        forecast_end_date=_optional_date(row.get("forecast_end_date")),
        realized_return=_optional_float(row.get("realized_return")),
        realized_spy_return=_optional_float(row.get("realized_spy_return")),
        realized_active_return=_optional_float(row.get("realized_active_return")),
        forecast_error=_optional_float(row.get("forecast_error")),
        forecast_hit=_optional_bool(row.get("forecast_hit")),
        outcome_status=_optional_str(row.get("outcome_status")),
    )


def _performance_snapshot_from_row(row: Mapping[str, Any]) -> PerformanceSnapshotRecord:
    return PerformanceSnapshotRecord(
        id=_optional_str(row.get("id")),
        account_id=str(row["account_id"]),
        as_of_date=_date_from_value(row["as_of_date"]),
        account_total_value=float(row["account_total_value"]),
        total_deposits=float(row["total_deposits"]),
        net_external_cashflow=float(row["net_external_cashflow"]),
        initial_value=_optional_float(row.get("initial_value")),
        invested_capital=_optional_float(row.get("invested_capital")),
        return_on_invested_capital=_optional_float(row.get("return_on_invested_capital")),
        account_time_weighted_return=_optional_float(row.get("account_time_weighted_return")),
        account_money_weighted_return=_optional_float(row.get("account_money_weighted_return")),
        spy_same_cashflow_value=_optional_float(row.get("spy_same_cashflow_value")),
        spy_time_weighted_return=_optional_float(row.get("spy_time_weighted_return")),
        spy_money_weighted_return=_optional_float(row.get("spy_money_weighted_return")),
        active_value=_optional_float(row.get("active_value")),
        active_return=_optional_float(row.get("active_return")),
    )


def _date_from_value(value: object) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _optional_date(value: object) -> date | None:
    if value in {None, ""}:
        return None
    return _date_from_value(value)


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    return float(cast(Any, value))


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(cast(Any, value))


def _optional_bool(value: object) -> bool | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "t", "1", "yes", "y"}
    return bool(value)


def _optional_str(value: object) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)
