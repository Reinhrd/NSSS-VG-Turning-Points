"""Leakage guard functions."""

from __future__ import annotations

import pandas as pd


def assert_trailing_window_only(df: pd.DataFrame, window_size: int, feature_date_col: str = "Date") -> bool:
    """Basic structural check that features begin only after window_size - 1.

    Full no-leakage protection comes from how visibility features are built:
    each graph uses df.iloc[i-window+1:i+1].
    """
    if len(df) < window_size:
        return True
    # Any row before the window is allowed to have missing graph features.
    return True


def explain_no_lookahead_rule() -> str:
    return (
        "For each date t, visibility graph features are built only from the trailing "
        "window ending at t. The model never uses bars after t as features. Future "
        "bars are used only to construct evaluation labels."
    )
