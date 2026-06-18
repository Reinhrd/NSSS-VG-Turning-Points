# NSSS Reversal Lab Report

## Dataset

- Ticker: NSSS.JK
- Rows: 333
- Period: 2025-01-02 to 2026-06-18
- Graph mode: horizontal
- Window size: 60 bars
- Lead target: pivot low within next 5 bars

## Signal Distribution

| Signal | Count |
|---|---:|
| No Signal | 164 |
| Watchlist | 79 |
| Potential Reversal | 57 |
| Strong Reversal | 33 |

## Rule-Based Classification Metrics

These metrics compare `BuySignal` to the objective label `TargetPivotLowNext5`.

| Metric | Value |
|---|---:|
| Precision | 0.1889 |
| Recall | 0.1932 |
| F1 | 0.1910 |
| Accuracy | 0.5676 |

## Backtest Summary

| Metric | Value |
|---|---:|
| Number of trades | 29 |
| Win rate | 51.72% |
| Average return per trade | 1.52% |
| Median return per trade | 0.68% |
| Total return | 37.60% |
| Max drawdown | -37.62% |
| Average holding bars | 3.76 |
| Profit factor | 1.42 |

## Interpretation

The current rule-based model is useful as a transparent baseline. It produces a tradable signal path and a reproducible backtest, but the classification quality is still modest. That is expected for a first version because the signal is built from a broad reversal-zone definition, while the label is a strict future pivot-low event.

The next improvement path is:

1. tune the pivot label threshold,
2. compare HVG vs Natural VG,
3. tune lead bars,
4. add CPO/IHSG relative-strength data,
5. add a better walk-forward threshold calibration,
6. compare against a simple baseline such as moving-average rebound or naive swing detector.

## No-Look-Ahead Guard

For each date t, visibility graph features are built only from the trailing window ending at t. The model never uses bars after t as features. Future bars are used only to construct evaluation labels.

## Generated Figures

- `reports/figures/01_price_signals.png`
- `reports/figures/02_volume_confirmation.png`
- `reports/figures/03_reversal_score_timeline.png`
- `reports/figures/04_visibility_graph_features.png`
- `reports/figures/05_visibility_graph_example.png`
- `reports/figures/06_backtest_equity_curve.png`
- `reports/figures/07_backtest_drawdown.png`
