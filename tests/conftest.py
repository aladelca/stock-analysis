from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stock_analysis.config import (
    FeatureConfig,
    ForecastConfig,
    OptimizerConfig,
    PortfolioConfig,
    PriceConfig,
    RunConfig,
    TableauConfig,
)
from stock_analysis.ingestion.prices import PriceDownload


@pytest.fixture
def sample_html() -> str:
    return Path("tests/fixtures/sp500_sample.html").read_text(encoding="utf-8")


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    tickers = ["AAA", "BBB", "CCC", "BRK-B"]
    dates = pd.bdate_range("2026-01-01", periods=40)
    rows: list[dict[str, object]] = []
    for idx, ticker in enumerate(tickers):
        base = 100 + idx * 10
        for day, current_date in enumerate(dates):
            close = base * (1 + 0.002 * day + 0.0005 * idx * day)
            rows.append(
                {
                    "ticker": ticker.replace("-", "."),
                    "provider_ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "adj_close": close,
                    "volume": int(1_000_000 + idx * 1000 + day),
                    "as_of_date": "2026-04-24",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_config(tmp_path: Path) -> PortfolioConfig:
    return PortfolioConfig(
        run=RunConfig(
            as_of_date=date(2026, 4, 24), output_root=tmp_path / "data", run_id="test-run"
        ),
        prices=PriceConfig(
            lookback_years=1,
            batch_size=10,
            fail_on_missing_benchmark=False,
            fail_on_low_coverage=False,
        ),
        features=FeatureConfig(
            min_history_days=20,
            momentum_windows=[5, 10],
            volatility_window=5,
            drawdown_window=10,
            moving_average_windows=[5, 10],
        ),
        forecast=ForecastConfig(
            momentum_window=10, volatility_penalty=0.1, covariance_lookback_days=20
        ),
        optimizer=OptimizerConfig(max_weight=0.6, risk_aversion=1.0, min_trade_weight=0.001),
        tableau=TableauConfig(export_csv=True, export_hyper=False, publish_enabled=False),
    )


class StaticPriceProvider:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def get_daily_prices(
        self,
        tickers,
        start,
        end,
        as_of_date,
    ) -> PriceDownload:
        del start, end, as_of_date
        wanted = set(tickers)
        result = self.prices[self.prices["provider_ticker"].isin(wanted)].copy()
        return PriceDownload(
            prices=result,
            raw_payloads={"static_prices.csv": result.to_csv(index=False)},
        )


@pytest.fixture
def static_price_provider(sample_prices: pd.DataFrame) -> StaticPriceProvider:
    return StaticPriceProvider(sample_prices)
