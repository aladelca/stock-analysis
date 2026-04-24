from __future__ import annotations

from datetime import date

from stock_analysis.ingestion.universe import normalize_provider_ticker, parse_sp500_constituents


def test_parse_sp500_constituents(sample_html: str) -> None:
    result = parse_sp500_constituents(sample_html, date(2026, 4, 24))

    assert len(result) == 4
    assert "BRK.B" in result["ticker"].tolist()
    assert "BRK-B" in result["provider_ticker"].tolist()
    assert result["cik"].str.len().eq(10).all()


def test_normalize_provider_ticker() -> None:
    assert normalize_provider_ticker("BRK.B") == "BRK-B"
