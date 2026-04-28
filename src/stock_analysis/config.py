from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class RunConfig(BaseModel):
    as_of_date: date | None = None
    output_root: Path = Path("data")
    run_id: str | None = None


class UniverseConfig(BaseModel):
    provider: Literal["wikipedia_sp500"] = "wikipedia_sp500"
    source_url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class PriceConfig(BaseModel):
    provider: Literal["yfinance"] = "yfinance"
    lookback_years: int = Field(default=5, ge=1, le=30)
    batch_size: int = Field(default=100, ge=1, le=500)
    benchmark_tickers: list[str] = Field(default_factory=lambda: ["SPY"])
    include_benchmark_tickers_in_universe: bool = True


class FeatureConfig(BaseModel):
    min_history_days: int = Field(default=252, ge=2)
    momentum_windows: list[int] = Field(default_factory=lambda: [63, 126, 252])
    volatility_window: int = Field(default=63, ge=2)
    drawdown_window: int = Field(default=252, ge=2)
    moving_average_windows: list[int] = Field(default_factory=lambda: [50, 200])

    @field_validator("momentum_windows", "moving_average_windows")
    @classmethod
    def _positive_windows(cls, value: list[int]) -> list[int]:
        if not value or any(window < 2 for window in value):
            msg = "windows must contain positive values greater than one"
            raise ValueError(msg)
        return value


class PanelFeatureConfig(BaseModel):
    min_history_days: int = Field(default=252, ge=2)
    momentum_windows: list[int] = Field(default_factory=lambda: [21, 63, 126, 252])
    volatility_windows: list[int] = Field(default_factory=lambda: [21, 63, 126])
    drawdown_windows: list[int] = Field(default_factory=lambda: [63, 252])
    moving_average_windows: list[int] = Field(default_factory=lambda: [50, 200])
    return_windows: list[int] = Field(default_factory=lambda: [1, 5, 21])
    volume_zscore_window: int = Field(default=21, ge=2)
    compute_cross_sectional_ranks: bool = True

    @field_validator(
        "momentum_windows",
        "volatility_windows",
        "drawdown_windows",
        "moving_average_windows",
        "return_windows",
    )
    @classmethod
    def _positive_windows(cls, value: list[int]) -> list[int]:
        if not value or any(window < 1 for window in value):
            msg = "panel feature windows must be positive"
            raise ValueError(msg)
        return value


class ForecastConfig(BaseModel):
    engine: Literal["heuristic", "ml"] = "heuristic"
    momentum_window: int = Field(default=252, ge=2)
    volatility_penalty: float = Field(default=0.25, ge=0)
    covariance_lookback_days: int = Field(default=252, ge=2)
    label_horizons: list[int] = Field(default_factory=lambda: [5, 21, 63])
    ml_model_version: str = "lightgbm_return_zscore"
    ml_horizon_days: int = Field(default=5, ge=1)
    ml_max_assets: int | None = Field(default=100, ge=1)
    ml_feature_columns: list[str] = Field(default_factory=list)
    ml_score_scale: float = Field(default=1.0, gt=0)
    ml_lightgbm_nested_cv: bool = False
    ml_lightgbm_inner_folds: int = Field(default=2, ge=1)
    ml_random_seed: int = 42

    @field_validator("label_horizons")
    @classmethod
    def _positive_horizons(cls, value: list[int]) -> list[int]:
        if not value or any(horizon < 1 for horizon in value):
            msg = "label horizons must be positive"
            raise ValueError(msg)
        return value


class OptimizerConfig(BaseModel):
    max_weight: float = Field(default=0.05, gt=0, le=1)
    benchmark_candidate_max_weight: float | None = Field(default=0.8, gt=0, le=1)
    risk_aversion: float = Field(default=10.0, ge=0)
    min_trade_weight: float = Field(default=0.005, ge=0)
    min_rebalance_trade_weight: float = Field(default=0.005, ge=0)
    lambda_turnover: float = Field(default=5.0, ge=0)
    commission_rate: float = Field(default=0.02, ge=0, le=1)
    sector_max_weight: float | None = Field(default=None, gt=0, le=1)
    max_trade_abs_weight: float | None = Field(default=None, ge=0, le=2)
    preserve_outside_holdings: bool = False
    solver: str | None = None


class PortfolioStateConfig(BaseModel):
    current_holdings_path: Path | None = None
    portfolio_value: float | None = Field(default=None, gt=0)


class ContributionConfig(BaseModel):
    initial_portfolio_value: float = Field(default=1000.0, gt=0)
    monthly_deposit_amount: float = Field(default=0.0, ge=0)
    deposit_frequency_days: int = Field(default=30, ge=1)
    deposit_start_date: date | None = None
    rebalance_on_deposit_day: bool = True


class ExecutionConfig(BaseModel):
    cash_balance: float = Field(default=0.0, ge=0)
    no_trade_band: float = Field(default=0.0, ge=0, le=1)


class LiveAccountConfig(BaseModel):
    enabled: bool = False
    account_slug: str | None = None
    cashflow_source: Literal["scenario", "actual"] = "scenario"


class SupabaseConfig(BaseModel):
    enabled: bool = False
    url_env: str = "SUPABASE_URL"
    key_env: str = "SUPABASE_SERVICE_ROLE_KEY"
    schema_name: str = "public"
    accounts_table: str = "accounts"
    cashflows_table: str = "cashflows"
    portfolio_snapshots_table: str = "portfolio_snapshots"
    holding_snapshots_table: str = "holding_snapshots"
    recommendation_runs_table: str = "recommendation_runs"
    recommendation_lines_table: str = "recommendation_lines"
    performance_snapshots_table: str = "performance_snapshots"


class TableauConfig(BaseModel):
    export_csv: bool = True
    export_hyper: bool = False
    publish_enabled: bool = False
    prep_output_root: Path = Path("tableau_prep_outputs")
    server_url: str | None = None
    site_name: str | None = None
    project_name: str = "Default"
    datasource_name: str = "portfolio_dashboard_mart"
    workbook_name: str = "portfolio_recommendations"
    workbook_output_path: Path = Path("tableau/workbooks/portfolio_recommendations.twb")


class MLflowConfig(BaseModel):
    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str = "stock-analysis-portfolio"


class PortfolioConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    prices: PriceConfig = Field(default_factory=PriceConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    panel_features: PanelFeatureConfig = Field(default_factory=PanelFeatureConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    portfolio_state: PortfolioStateConfig = Field(default_factory=PortfolioStateConfig)
    contributions: ContributionConfig = Field(default_factory=ContributionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    live_account: LiveAccountConfig = Field(default_factory=LiveAccountConfig)
    supabase: SupabaseConfig = Field(default_factory=SupabaseConfig)
    tableau: TableauConfig = Field(default_factory=TableauConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


def load_config(path: Path) -> PortfolioConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return PortfolioConfig.model_validate(raw)
