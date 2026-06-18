"""Run the NSSS visibility-graph reversal project end-to-end.

Usage:
    python scripts/run_pipeline.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import pandas as pd
import yaml

# Allow running from repo root without installing the package first.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nsss_vg.data.ingest import load_ohlcv
from nsss_vg.data.preprocess import add_technical_features
from nsss_vg.labeling.turning_points import add_turning_point_labels
from nsss_vg.graph.visibility import add_visibility_features
from nsss_vg.models.rule_based import add_reversal_scores
from nsss_vg.evaluation.backtest import backtest_long_signals
from nsss_vg.evaluation.metrics import (
    rule_based_classification_metrics,
    walk_forward_random_forest,
)
from nsss_vg.evaluation.leakage import explain_no_lookahead_rule
from nsss_vg.viz.plots import (
    plot_price_signals,
    plot_volume_confirmation,
    plot_score_timeline,
    plot_graph_features,
    plot_visibility_graph_example,
    plot_equity_curve,
    plot_drawdown,
)


def run(config_path: str | Path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_csv = REPO_ROOT / config["project"]["raw_csv"]
    date_format = config["project"].get("date_format", "%m/%d/%Y")

    processed_dir = REPO_ROOT / "data" / "processed"
    figure_dir = REPO_ROOT / "reports" / "figures"
    table_dir = REPO_ROOT / "reports" / "tables"
    log_dir = REPO_ROOT / "reports" / "logs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pipeline = config["pipeline"]
    window_size = int(pipeline.get("window_size", 60))
    graph_mode = pipeline.get("graph_mode", "horizontal")
    lead_bars = int(pipeline.get("lead_bars", 5))

    print("[1/9] Load OHLCV data")
    df = load_ohlcv(raw_csv, date_format=date_format)
    df.to_csv(processed_dir / "01_clean_ohlcv.csv", index=False)

    print("[2/9] Add price-time and volume features")
    df = add_technical_features(df)
    df.to_csv(processed_dir / "02_technical_features.csv", index=False)

    print("[3/9] Add objective turning-point labels")
    df = add_turning_point_labels(
        df,
        threshold_pct=float(pipeline.get("zigzag_threshold_pct", 0.12)),
        lead_bars=lead_bars,
        prominence_atr_mult=float(pipeline.get("peak_prominence_atr_mult", 1.0)),
    )
    df.to_csv(processed_dir / "03_labels.csv", index=False)

    print("[4/9] Build trailing visibility graph features")
    df = add_visibility_features(
        df,
        window_size=window_size,
        price_col="Close",
        mode=graph_mode,
    )
    df.to_csv(processed_dir / "04_visibility_features.csv", index=False)

    print("[5/9] Add transparent reversal scoring")
    df = add_reversal_scores(df, config)
    signals_path = processed_dir / "05_reversal_signals.csv"
    df.to_csv(signals_path, index=False)

    print("[6/9] Backtest rule-based signal")
    trades, equity, backtest_summary = backtest_long_signals(df, config)
    trades.to_csv(table_dir / "backtest_trades.csv", index=False)
    equity.to_csv(table_dir / "equity_curve.csv", index=False)
    backtest_summary.to_csv(table_dir / "backtest_summary.csv", index=False)

    print("[7/9] Evaluate classification and walk-forward RF baseline")
    target_col = f"TargetPivotLowNext{lead_bars}"
    rule_metrics = rule_based_classification_metrics(df, target_col=target_col)
    rf_metrics, rf_predictions = walk_forward_random_forest(df, target_col=target_col, n_splits=5)

    rule_metrics.to_csv(table_dir / "rule_based_metrics.csv", index=False)
    rf_metrics.to_csv(table_dir / "rf_walk_forward_metrics.csv", index=False)
    rf_predictions.to_csv(table_dir / "rf_predictions.csv", index=False)

    print("[8/9] Generate visualizations")
    plot_price_signals(df, figure_dir / "01_price_signals.png")
    plot_volume_confirmation(df, figure_dir / "02_volume_confirmation.png")
    plot_score_timeline(df, figure_dir / "03_reversal_score_timeline.png")
    plot_graph_features(df, figure_dir / "04_visibility_graph_features.png", mode=graph_mode)
    plot_visibility_graph_example(
        df,
        figure_dir / "05_visibility_graph_example.png",
        window_size=window_size,
        mode=graph_mode,
    )
    plot_equity_curve(equity, figure_dir / "06_backtest_equity_curve.png")
    plot_drawdown(equity, figure_dir / "07_backtest_drawdown.png")

    print("[9/9] Write run summary")
    summary = {
        "project": config["project"]["name"],
        "ticker": config["project"]["ticker"],
        "rows": int(len(df)),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "window_size": window_size,
        "graph_mode": graph_mode,
        "lead_bars": lead_bars,
        "signals": df["Signal"].value_counts().to_dict(),
        "backtest_summary": backtest_summary.to_dict(orient="records")[0],
        "no_lookahead_rule": explain_no_lookahead_rule(),
        "output_signals": str(signals_path.relative_to(REPO_ROOT)),
    }

    with (log_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nDone.")
    print(json.dumps(summary, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    run(args.config)


if __name__ == "__main__":
    main()
