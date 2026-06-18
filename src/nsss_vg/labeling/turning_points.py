"""Objective turning-point labels.

Important:
- Labels are allowed to look forward because labels are ground truth.
- Features must not look forward. The pipeline keeps those two paths separate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def zigzag_pivots(close: pd.Series, threshold_pct: float = 0.10) -> np.ndarray:
    """Return pivot labels: +1 = swing high, -1 = swing low, 0 = none.

    This is a simple threshold-based zigzag. It is used as an objective
    labeling device, not as a tradable real-time feature.
    """
    prices = np.asarray(close, dtype=float)
    n = len(prices)
    pivots = np.zeros(n, dtype=int)

    if n == 0:
        return pivots

    last_pivot_idx = 0
    last_pivot_price = prices[0]
    trend = 0
    candidate_idx = 0
    candidate_price = prices[0]

    for i in range(1, n):
        price = prices[i]

        if trend == 0:
            change = (price - last_pivot_price) / last_pivot_price
            if change >= threshold_pct:
                trend = 1
                pivots[last_pivot_idx] = -1
                candidate_idx, candidate_price = i, price
            elif change <= -threshold_pct:
                trend = -1
                pivots[last_pivot_idx] = 1
                candidate_idx, candidate_price = i, price
            else:
                if price > candidate_price:
                    candidate_idx, candidate_price = i, price
                elif price < candidate_price:
                    candidate_idx, candidate_price = i, price

        elif trend == 1:
            if price > candidate_price:
                candidate_idx, candidate_price = i, price
            elif (price - candidate_price) / candidate_price <= -threshold_pct:
                pivots[candidate_idx] = 1
                last_pivot_idx, last_pivot_price = candidate_idx, candidate_price
                trend = -1
                candidate_idx, candidate_price = i, price

        elif trend == -1:
            if price < candidate_price:
                candidate_idx, candidate_price = i, price
            elif (price - candidate_price) / candidate_price >= threshold_pct:
                pivots[candidate_idx] = -1
                last_pivot_idx, last_pivot_price = candidate_idx, candidate_price
                trend = 1
                candidate_idx, candidate_price = i, price

    return pivots


def add_turning_point_labels(
    df: pd.DataFrame,
    threshold_pct: float = 0.10,
    lead_bars: int = 5,
    prominence_atr_mult: float = 1.0,
) -> pd.DataFrame:
    """Add pivot high/low labels and a forward target.

    The forward target answers:
    "Is there a pivot low within the next k bars?"
    """
    df = df.copy()

    zigzag = zigzag_pivots(df["Close"], threshold_pct=threshold_pct)

    median_atr = df["ATR14"].median()
    prominence = float(median_atr * prominence_atr_mult) if not np.isnan(median_atr) else df["Close"].std() * 0.10
    if prominence <= 0:
        prominence = df["Close"].std() * 0.10

    peaks, _ = find_peaks(df["Close"].values, prominence=prominence, distance=3)
    troughs, _ = find_peaks(-df["Close"].values, prominence=prominence, distance=3)

    df["PivotHigh_ZZ"] = zigzag == 1
    df["PivotLow_ZZ"] = zigzag == -1

    df["PivotHigh_Peak"] = False
    df["PivotLow_Peak"] = False

    df.loc[peaks, "PivotHigh_Peak"] = True
    df.loc[troughs, "PivotLow_Peak"] = True

    df["PivotHigh"] = df["PivotHigh_ZZ"] | df["PivotHigh_Peak"]
    df["PivotLow"] = df["PivotLow_ZZ"] | df["PivotLow_Peak"]

    target = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        target[i] = int(df["PivotLow"].iloc[i + 1 : i + 1 + lead_bars].any())

    df[f"TargetPivotLowNext{lead_bars}"] = target

    return df
