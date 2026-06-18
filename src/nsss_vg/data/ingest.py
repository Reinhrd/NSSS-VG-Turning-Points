"""Data ingestion utilities for NSSS OHLCV data.

The default raw file is an Investing.com-style CSV with columns:
Date, Price, Open, High, Low, Vol., Change %
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd


def _parse_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if text == "":
        return np.nan
    return float(text)


def parse_volume(value):
    """Parse human-readable volume strings such as 18.23M, 850K, 1.2B."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    match = re.match(r"^([-+]?\d*\.?\d+)([KMB]?)$", text, re.IGNORECASE)
    if not match:
        return _parse_number(text)
    number = float(match.group(1))
    suffix = match.group(2).upper()
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix, 1)
    return number * multiplier


def load_ohlcv(csv_path: str | Path, date_format: str = "%m/%d/%Y") -> pd.DataFrame:
    """Load and standardize OHLCV columns."""
    path = Path(csv_path)
    df = pd.read_csv(path)

    rename_map = {
        "Price": "Close",
        "Vol.": "Volume",
        "Change %": "ChangePct",
    }
    df = df.rename(columns=rename_map)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], format=date_format)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].map(_parse_number)

    df["Volume"] = df["Volume"].map(parse_volume)

    if "ChangePct" in df.columns:
        df["ChangePct"] = (
            df["ChangePct"].astype(str).str.replace("%", "", regex=False).map(_parse_number) / 100
        )

    df = (
        df.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    df["Ticker"] = "NSSS.JK"

    return df
