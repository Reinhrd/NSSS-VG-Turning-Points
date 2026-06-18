"""Project visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from nsss_vg.graph.visibility import build_visibility_graph


def _save(fig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_price_signals(df: pd.DataFrame, out_path: str | Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["Date"], df["Close"], label="Close", linewidth=1.5)
    ax.plot(df["Date"], df["MA20"], label="MA20", linewidth=1)
    ax.plot(df["Date"], df["MA50"], label="MA50", linewidth=1)

    strong = df["Signal"].eq("Strong Reversal")
    potential = df["Signal"].eq("Potential Reversal")
    piv_low = df["PivotLow"].fillna(False)

    ax.scatter(df.loc[potential, "Date"], df.loc[potential, "Close"], marker="^", label="Potential Reversal", s=45)
    ax.scatter(df.loc[strong, "Date"], df.loc[strong, "Close"], marker="*", label="Strong Reversal", s=80)
    ax.scatter(df.loc[piv_low, "Date"], df.loc[piv_low, "Close"], marker="o", label="Objective Pivot Low", s=30)

    ax.set_title("NSSS Price, Moving Averages, Pivot Lows, and Reversal Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)


def plot_volume_confirmation(df: pd.DataFrame, out_path: str | Path):
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(df["Date"], df["Volume"], label="Volume", alpha=0.6)
    ax1.set_ylabel("Volume")

    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["VolumeRatio"], label="Volume Ratio", linewidth=1.2)
    ax2.axhline(1.5, linestyle="--", linewidth=1, label="Volume Ratio 1.5x")
    ax2.set_ylabel("Volume / VolumeMA20")

    ax1.set_title("Volume Confirmation Layer")
    ax1.grid(True, alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    _save(fig, out_path)


def plot_score_timeline(df: pd.DataFrame, out_path: str | Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["Date"], df["TechnicalScore"], label="Technical Score", linewidth=1.5)
    ax.plot(df["Date"], df["ReversalScore"], label="Technical + Fundamental Context Score", linewidth=1)
    ax.axhline(4, linestyle="--", linewidth=1, label="Potential Threshold")
    ax.axhline(5, linestyle=":", linewidth=1, label="Strong Threshold")
    ax.set_title("Reversal Score Timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)


def plot_graph_features(df: pd.DataFrame, out_path: str | Path, mode: str = "horizontal"):
    fig, ax1 = plt.subplots(figsize=(14, 5))
    degree_col = f"{mode}_last_degree_norm"
    btw_col = f"{mode}_last_betweenness"

    ax1.plot(df["Date"], df["Close"], label="Close", linewidth=1.2)
    ax1.set_ylabel("Close")

    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df[degree_col], label="VG Last Node Degree", linewidth=1)
    ax2.plot(df["Date"], df[btw_col], label="VG Last Node Betweenness", linewidth=1)
    ax2.set_ylabel("Graph Feature Value")

    ax1.set_title("Price-Time Visibility Graph Features")
    ax1.grid(True, alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    _save(fig, out_path)


def plot_visibility_graph_example(
    df: pd.DataFrame,
    out_path: str | Path,
    window_size: int = 60,
    mode: str = "horizontal",
):
    sample = df.tail(window_size).reset_index(drop=True)
    values = sample["Close"].values
    graph = build_visibility_graph(values, mode=mode)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(values)), values, marker="o", markersize=3, linewidth=1.2)

    for i, j in graph.edges():
        ax.plot([i, j], [values[i], values[j]], alpha=0.18, linewidth=0.6)

    ax.set_title(f"{mode.title()} Visibility Graph Example: Latest {window_size} Bars")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Close")
    ax.grid(True, alpha=0.20)

    _save(fig, out_path)


def plot_equity_curve(equity: pd.DataFrame, out_path: str | Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity["Date"], equity["Equity"], label="Equity Curve", linewidth=1.5)
    ax.set_title("Rule-Based Signal Backtest Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity, initial = 1.0")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)


def plot_drawdown(equity: pd.DataFrame, out_path: str | Path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(equity["Date"], equity["Drawdown"] * 100, 0, alpha=0.35)
    ax.set_title("Backtest Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)
