"""Evaluation metrics and walk-forward supervised baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit


FEATURE_COLUMNS = [
    "Return",
    "MA20",
    "MA50",
    "Slope5",
    "Slope20",
    "ATR14",
    "Volatility20",
    "DistanceSupport60",
    "DistanceResistance60",
    "Drawdown60",
    "VolumeRatio",
    "VolumeZ20",
    "BodyPct",
    "UpperShadowPct",
    "LowerShadowPct",
    "horizontal_last_degree_norm",
    "horizontal_last_betweenness",
    "horizontal_density",
    "SupportScore",
    "SlopeScore",
    "SlopeRecoveryScore",
    "PivotCandidateScore",
    "VolumeScore",
    "MomentumScore",
    "VGHubScore",
    "VGBreakoutScore",
]


def rule_based_classification_metrics(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Classification metrics for rule-based BuySignal vs forward pivot-low target."""
    y_true = df[target_col].fillna(0).astype(int)
    y_pred = df["BuySignal"].fillna(False).astype(int)

    return pd.DataFrame(
        [
            {
                "model": "rule_based",
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "positive_rate_target": y_true.mean(),
                "positive_rate_signal": y_pred.mean(),
            }
        ]
    )


def walk_forward_random_forest(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward RandomForest as a supervised baseline.

    The split is time-ordered. No shuffled cross-validation is used.
    """
    usable_features = [col for col in FEATURE_COLUMNS if col in df.columns]

    data = df.dropna(subset=usable_features + [target_col]).copy()
    if len(data) < 80:
        empty = pd.DataFrame(columns=["fold", "precision", "recall", "f1", "accuracy"])
        pred = data[["Date", "Close", target_col]].copy()
        pred["rf_pred"] = np.nan
        pred["rf_proba"] = np.nan
        return empty, pred

    X = (
        data[usable_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    y = data[target_col].astype(int)

    splitter = TimeSeriesSplit(n_splits=n_splits)

    rows = []
    pred = data[["Date", "Close", target_col]].copy()
    pred["rf_pred"] = np.nan
    pred["rf_proba"] = np.nan

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        if y.iloc[train_idx].nunique() < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        fold_pred = model.predict(X.iloc[test_idx])
        fold_proba = model.predict_proba(X.iloc[test_idx])[:, 1]

        pred.iloc[test_idx, pred.columns.get_loc("rf_pred")] = fold_pred
        pred.iloc[test_idx, pred.columns.get_loc("rf_proba")] = fold_proba

        rows.append(
            {
                "fold": fold,
                "train_start": data["Date"].iloc[train_idx[0]],
                "train_end": data["Date"].iloc[train_idx[-1]],
                "test_start": data["Date"].iloc[test_idx[0]],
                "test_end": data["Date"].iloc[test_idx[-1]],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "positive_rate_test": y.iloc[test_idx].mean(),
                "precision": precision_score(y.iloc[test_idx], fold_pred, zero_division=0),
                "recall": recall_score(y.iloc[test_idx], fold_pred, zero_division=0),
                "f1": f1_score(y.iloc[test_idx], fold_pred, zero_division=0),
                "accuracy": accuracy_score(y.iloc[test_idx], fold_pred),
            }
        )

    return pd.DataFrame(rows), pred
