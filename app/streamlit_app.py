"""Minimal Streamlit dashboard for the generated outputs.

Run after the pipeline:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("Install streamlit first: pip install streamlit") from exc


ROOT = Path(__file__).resolve().parents[1]
signals_path = ROOT / "data" / "processed" / "05_reversal_signals.csv"
summary_path = ROOT / "reports" / "tables" / "backtest_summary.csv"

st.set_page_config(page_title="NSSS Reversal Lab", layout="wide")
st.title("NSSS Reversal Lab")
st.caption("Price-time geometry + visibility graph + volume confirmation")

if not signals_path.exists():
    st.warning("Run `python scripts/run_pipeline.py --config config/config.yaml` first.")
    st.stop()

df = pd.read_csv(signals_path, parse_dates=["Date"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(df))
col2.metric("Start", df["Date"].min().date())
col3.metric("End", df["Date"].max().date())
col4.metric("Latest Signal", df["Signal"].iloc[-1])

st.subheader("Price and Signal Data")
st.dataframe(
    df[["Date", "Close", "Volume", "TechnicalScore", "ReversalScore", "Signal"]].tail(50),
    use_container_width=True,
)

st.subheader("Figures")
fig_dir = ROOT / "reports" / "figures"
for image_name in [
    "01_price_signals.png",
    "02_volume_confirmation.png",
    "03_reversal_score_timeline.png",
    "04_visibility_graph_features.png",
    "05_visibility_graph_example.png",
    "06_backtest_equity_curve.png",
    "07_backtest_drawdown.png",
]:
    path = fig_dir / image_name
    if path.exists():
        st.image(str(path), caption=image_name, use_container_width=True)

if summary_path.exists():
    st.subheader("Backtest Summary")
    st.dataframe(pd.read_csv(summary_path), use_container_width=True)
