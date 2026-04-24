from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrepInput:
    layer: str
    name: str
    required_columns: tuple[str, ...]


PREP_INPUTS: tuple[PrepInput, ...] = (
    PrepInput(
        layer="bronze",
        name="sp500_constituents",
        required_columns=("ticker", "security", "gics_sector", "gics_sub_industry"),
    ),
    PrepInput(
        layer="silver",
        name="asset_daily_features",
        required_columns=(
            "ticker",
            "latest_adj_close",
            "history_days",
            "eligible_for_optimization",
        ),
    ),
    PrepInput(
        layer="gold",
        name="portfolio_recommendations",
        required_columns=("ticker", "target_weight", "action", "reason_code"),
    ),
    PrepInput(
        layer="gold",
        name="portfolio_risk_metrics",
        required_columns=("metric", "value"),
    ),
    PrepInput(
        layer="gold",
        name="sector_exposure",
        required_columns=("gics_sector", "target_weight"),
    ),
)
