from nsss_vg.labeling.turning_points import zigzag_pivots
import pandas as pd


def test_zigzag_returns_same_length():
    close = pd.Series([100, 90, 80, 95, 110, 100, 85, 95])
    pivots = zigzag_pivots(close, threshold_pct=0.10)
    assert len(pivots) == len(close)
