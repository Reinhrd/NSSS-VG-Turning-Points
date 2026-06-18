"""Preprocessing and technical feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_slope(series: pd.Series, window: int) -> np.ndarray:
    """Rolling linear-regression slope using only trailing observations."""
    y = np.asarray(series, dtype=float)
    out = np.full(len(y), np.nan)
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    for i in range(window - 1, len(y)):
        yy = y[i - window + 1 : i + 1]
        if np.any(np.isnan(yy)):
            continue
        y_mean = yy.mean()
        out[i] = ((x - x_mean) * (yy - y_mean)).sum() / denom

    return out


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price-time, trend, volume, and candle features."""
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"]).diff()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["Slope5"] = rolling_slope(df["Close"], 5)
    df["Slope20"] = rolling_slope(df["Close"], 20)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()

    df["Volatility20"] = df["Return"].rolling(20).std()

    df["RollingLow60"] = df["Low"].rolling(60).min()
    df["RollingHigh60"] = df["High"].rolling(60).max()
    df["DistanceSupport60"] = (df["Close"] - df["RollingLow60"]) / df["RollingLow60"]
    df["DistanceResistance60"] = (df["RollingHigh60"] - df["Close"]) / df["Close"]
    df["Drawdown60"] = df["Close"] / df["RollingHigh60"] - 1

    df["VolumeMA20"] = df["Volume"].rolling(20).mean()
    df["VolumeSTD20"] = df["Volume"].rolling(20).std()
    df["VolumeRatio"] = df["Volume"] / df["VolumeMA20"]
    df["VolumeZ20"] = (df["Volume"] - df["VolumeMA20"]) / df["VolumeSTD20"]

    df["BodyPct"] = (df["Close"] - df["Open"]) / df["Open"]
    df["UpperShadowPct"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Open"]
    df["LowerShadowPct"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Open"]

    df["TrendState"] = np.select(
        [
            (df["Close"] > df["MA50"]) & (df["Slope20"] > 0),
            (df["Close"] < df["MA50"]) & (df["Slope20"] < 0),
        ],
        ["Uptrend", "Downtrend"],
        default="Sideways",
    )

    return df
