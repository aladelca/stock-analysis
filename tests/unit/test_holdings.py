from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stock_analysis.portfolio.holdings import (
    align_current_weights,
    load_current_weights,
    load_portfolio_state,
)


def test_load_current_weights_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "holdings.csv"
    path.write_text("ticker,current_weight\nAAA,0.25\nBBB,0.15\n", encoding="utf-8")

    weights = load_current_weights(path)

    assert weights.to_dict() == {"AAA": 0.25, "BBB": 0.15}
    assert weights.name == "current_weight"


def test_load_current_weights_normalizes_market_value(tmp_path: Path) -> None:
    path = tmp_path / "holdings.csv"
    path.write_text("ticker,market_value\nAAA,300\nBBB,100\n", encoding="utf-8")

    weights = load_current_weights(path)

    assert weights.to_dict() == {"AAA": 0.75, "BBB": 0.25}


def test_load_portfolio_state_preserves_market_values_and_cash(tmp_path: Path) -> None:
    path = tmp_path / "holdings.csv"
    path.write_text("ticker,market_value\nAAA,300\nBBB,100\n", encoding="utf-8")

    state = load_portfolio_state(path, cash_balance=100)

    assert state.resolved_portfolio_value == pytest.approx(500)
    assert state.weights.to_dict() == {"AAA": 0.6, "BBB": 0.2}
    assert state.market_values.to_dict() == {"AAA": 300.0, "BBB": 100.0}
    assert state.cash_balance == 100


def test_load_portfolio_state_without_holdings_treats_value_as_cash() -> None:
    state = load_portfolio_state(None, portfolio_value=1000)

    assert state.weights.empty
    assert state.market_values.empty
    assert state.cash_balance == pytest.approx(1000)
    assert state.resolved_portfolio_value == pytest.approx(1000)


def test_load_current_weights_rejects_invalid_weights(tmp_path: Path) -> None:
    path = tmp_path / "holdings.csv"
    path.write_text("ticker,current_weight\nAAA,0.8\nBBB,0.4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sum to at most 1.0"):
        load_current_weights(path)


def test_load_current_weights_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "holdings.csv"
    path.write_text("ticker,current_weight\nAAA,0.2\nAAA,0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate tickers"):
        load_current_weights(path)


def test_align_current_weights_fills_missing_tickers() -> None:
    weights = align_current_weights(pd.Series({"AAA": 0.2}), ["AAA", "BBB"])

    assert weights.to_dict() == {"AAA": 0.2, "BBB": 0.0}
