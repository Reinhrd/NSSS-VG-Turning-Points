"""Rule-based reversal scoring.

This is intentionally transparent. The goal is not to create a black-box
"prediction machine", but a reproducible scoring system for reversal zones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fundamental_score_from_config(config: dict) -> int:
    """Sum project-level fundamental context scores from config.yaml."""
    context = config.get("fundamental_context", {})
    return int(sum(context.values())) if context else 0


def add_reversal_scores(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create rule-based reversal scores and labels."""
    df = df.copy()

    scoring = config.get("scoring", {})
    pipeline = config.get("pipeline", {})
    mode = pipeline.get("graph_mode", "horizontal")
    degree_col = f"{mode}_last_degree_norm"
    betweenness_col = f"{mode}_last_betweenness"

    support_distance = scoring.get("support_distance_threshold", 0.08)
    volume_ratio_threshold = scoring.get("volume_ratio_threshold", 1.5)
    volume_zscore_threshold = scoring.get("volume_zscore_threshold", 1.25)
    lower_shadow_threshold = scoring.get("lower_shadow_threshold", 0.035)
    degree_quantile = scoring.get("vg_degree_quantile", 0.80)
    betweenness_quantile = scoring.get("vg_betweenness_quantile", 0.75)
    potential_threshold = scoring.get("potential_threshold", 4)
    strong_threshold = scoring.get("strong_threshold", 5)

    df["VG_DegreeThreshold"] = df[degree_col].rolling(120, min_periods=30).quantile(degree_quantile)
    df["VG_BetweennessThreshold"] = df[betweenness_col].rolling(120, min_periods=30).quantile(betweenness_quantile)

    df["SupportScore"] = (
        (df["DistanceSupport60"] <= support_distance)
        | (df["Close"] <= df["RollingLow60"] * (1 + support_distance))
    ).astype(int)

    df["SlopeScore"] = ((df["Slope5"] > 0) & (df["Slope5"] > df["Slope20"])).astype(int)
    df["SlopeRecoveryScore"] = ((df["Slope5"] > df["Slope20"]) & (df["Slope20"] < 0)).astype(int)

    previous_pivot_low = df["PivotLow"].shift(1).fillna(False).astype(bool)
    df["PivotCandidateScore"] = (
        (df["LowerShadowPct"] > lower_shadow_threshold) | previous_pivot_low
    ).astype(int)

    df["VolumeScore"] = (
        (df["VolumeRatio"] > volume_ratio_threshold)
        | (df["VolumeZ20"] > volume_zscore_threshold)
    ).astype(int)

    df["MomentumScore"] = (
        (df["Close"] > df["MA20"])
        & (df["Return"].rolling(3).sum() > 0)
    ).astype(int)

    df["VGHubScore"] = (
        (df[degree_col] >= df["VG_DegreeThreshold"]) & df[degree_col].notna()
    ).astype(int)

    df["VGBreakoutScore"] = (
        (df[betweenness_col] >= df["VG_BetweennessThreshold"]) & df[betweenness_col].notna()
    ).astype(int)

    score_cols = [
        "SupportScore",
        "SlopeScore",
        "SlopeRecoveryScore",
        "PivotCandidateScore",
        "VolumeScore",
        "MomentumScore",
        "VGHubScore",
        "VGBreakoutScore",
    ]

    df["TechnicalScore"] = df[score_cols].sum(axis=1)
    df["FundamentalScore"] = fundamental_score_from_config(config)
    df["ReversalScore"] = df["TechnicalScore"] + df["FundamentalScore"]

    df["Signal"] = np.select(
        [
            df["TechnicalScore"] >= strong_threshold,
            df["TechnicalScore"] >= potential_threshold,
            df["TechnicalScore"] >= max(2, potential_threshold - 1),
        ],
        ["Strong Reversal", "Potential Reversal", "Watchlist"],
        default="No Signal",
    )

    df["BuySignal"] = df["Signal"].isin(["Strong Reversal", "Potential Reversal"])

    return df
