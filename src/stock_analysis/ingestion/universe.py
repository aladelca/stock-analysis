from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import requests

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_provider_ticker(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def fetch_sp500_html(url: str = WIKIPEDIA_SP500_URL) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": "stock-analysis/0.1 educational portfolio assistant"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def parse_sp500_constituents(html: str, as_of_date: date) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    source = next(
        (
            table
            for table in tables
            if {"Symbol", "Security", "GICS Sector"}.issubset(set(table.columns))
        ),
        None,
    )
    if source is None:
        msg = "Could not find S&P 500 constituents table in supplied HTML"
        raise ValueError(msg)

    df = source.rename(
        columns={
            "Symbol": "ticker",
            "Security": "security",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "Headquarters Location": "headquarters_location",
            "Date added": "date_added",
            "CIK": "cik",
            "Founded": "founded",
        }
    ).copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["provider_ticker"] = df["ticker"].map(normalize_provider_ticker)
    if "cik" in df.columns:
        df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    df["as_of_date"] = as_of_date.isoformat()

    ordered = [
        "ticker",
        "provider_ticker",
        "security",
        "gics_sector",
        "gics_sub_industry",
        "headquarters_location",
        "date_added",
        "cik",
        "founded",
        "as_of_date",
    ]
    for column in ordered:
        if column not in df.columns:
            df[column] = pd.NA
    return df[ordered].sort_values("ticker").reset_index(drop=True)
