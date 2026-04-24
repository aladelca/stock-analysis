from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class PriceDownload:
    prices: pd.DataFrame
    raw_payloads: dict[str, str]


class PriceProvider(Protocol):
    def get_daily_prices(
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
        as_of_date: date,
    ) -> PriceDownload:
        """Return normalized daily price bars plus provider-native raw payloads."""


class YFinancePriceProvider:
    def __init__(self, batch_size: int = 100) -> None:
        self.batch_size = batch_size

    def get_daily_prices(
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
        as_of_date: date,
    ) -> PriceDownload:
        import yfinance as yf

        frames: list[pd.DataFrame] = []
        raw_payloads: dict[str, str] = {}
        ticker_list = list(dict.fromkeys(tickers))
        for offset in range(0, len(ticker_list), self.batch_size):
            batch = ticker_list[offset : offset + self.batch_size]
            raw = yf.download(
                batch,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            batch_name = f"batch_{offset // self.batch_size:04d}.csv"
            raw_payloads[batch_name] = raw.to_csv()
            frames.extend(_normalize_yfinance_download(raw, batch, as_of_date))
        if not frames:
            msg = "No price data returned by yfinance"
            raise ValueError(msg)
        return PriceDownload(pd.concat(frames, ignore_index=True), raw_payloads)


def _normalize_yfinance_download(
    raw: pd.DataFrame,
    tickers: Sequence[str],
    as_of_date: date,
) -> list[pd.DataFrame]:
    if raw.empty:
        return []

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for ticker in tickers:
            if ticker not in available:
                continue
            ticker_frame = raw[ticker]
            if isinstance(ticker_frame, pd.DataFrame):
                frames.append(_normalize_single_ticker(ticker_frame, ticker, as_of_date))
    elif len(tickers) == 1:
        frames.append(_normalize_single_ticker(raw, tickers[0], as_of_date))
    return frames


def _normalize_single_ticker(
    frame: pd.DataFrame, provider_ticker: str, as_of_date: date
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    clean = frame.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "adj_close" not in clean.columns and "close" in clean.columns:
        clean["adj_close"] = clean["close"]
    clean["provider_ticker"] = provider_ticker
    clean["ticker"] = provider_ticker.replace("-", ".")
    clean["date"] = pd.to_datetime(clean["date"]).dt.date.astype(str)
    clean["as_of_date"] = as_of_date.isoformat()
    columns = [
        "ticker",
        "provider_ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "as_of_date",
    ]
    for column in columns:
        if column not in clean.columns:
            clean[column] = pd.NA
    return clean[columns].dropna(subset=["adj_close"])
