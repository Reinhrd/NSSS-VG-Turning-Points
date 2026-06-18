"""Simple long-only event backtest for reversal signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_long_signals(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Enter on next bar's open after a BuySignal.

    This is deliberately simple and auditable:
    - Signal is observed at close of day t.
    - Entry uses open of day t+1.
    - Exit is first target, stop, or max holding period.
    """
    params = config.get("backtest", {})
    target_pct = params.get("target_profit_pct", 0.12)
    stop_pct = params.get("stop_loss_pct", 0.08)
    max_hold = params.get("max_hold_bars", 10)
    fee_bps_round_trip = params.get("fee_bps_round_trip", 30)

    trades = []
    i = 0
    n = len(df)

    while i < n - 1:
        if bool(df["BuySignal"].iloc[i]):
            entry_idx = i + 1
            entry_price = float(df["Open"].iloc[entry_idx])

            target = entry_price * (1 + target_pct)
            stop = entry_price * (1 - stop_pct)

            exit_idx = min(entry_idx + max_hold, n - 1)
            exit_price = float(df["Close"].iloc[exit_idx])
            exit_reason = "max_hold"

            for j in range(entry_idx, min(entry_idx + max_hold, n - 1) + 1):
                if df["Low"].iloc[j] <= stop:
                    exit_idx = j
                    exit_price = stop
                    exit_reason = "stop"
                    break

                if df["High"].iloc[j] >= target:
                    exit_idx = j
                    exit_price = target
                    exit_reason = "target"
                    break

            return_pct = (exit_price / entry_price - 1) - (fee_bps_round_trip / 10_000)

            trades.append(
                {
                    "signal_date": df["Date"].iloc[i],
                    "entry_date": df["Date"].iloc[entry_idx],
                    "exit_date": df["Date"].iloc[exit_idx],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": return_pct * 100,
                    "holding_bars": exit_idx - entry_idx,
                    "exit_reason": exit_reason,
                    "signal": df["Signal"].iloc[i],
                    "technical_score": df["TechnicalScore"].iloc[i],
                }
            )

            i = exit_idx + 1
        else:
            i += 1

    trades_df = pd.DataFrame(trades)

    equity = pd.DataFrame({"Date": df["Date"], "Equity": 1.0})

    if trades_df.empty:
        equity["Peak"] = equity["Equity"]
        equity["Drawdown"] = 0.0
        summary = pd.DataFrame([{"n_trades": 0}])
        return trades_df, equity, summary

    current_equity = 1.0

    for _, trade in trades_df.iterrows():
        current_equity *= 1 + trade["return_pct"] / 100
        equity.loc[equity["Date"] >= trade["exit_date"], "Equity"] = current_equity

    equity["Peak"] = equity["Equity"].cummax()
    equity["Drawdown"] = equity["Equity"] / equity["Peak"] - 1

    gross_profit = trades_df.loc[trades_df["return_pct"] > 0, "return_pct"].sum()
    gross_loss = abs(trades_df.loc[trades_df["return_pct"] < 0, "return_pct"].sum())

    summary = pd.DataFrame(
        [
            {
                "n_trades": len(trades_df),
                "win_rate_pct": (trades_df["return_pct"] > 0).mean() * 100,
                "avg_return_pct": trades_df["return_pct"].mean(),
                "median_return_pct": trades_df["return_pct"].median(),
                "total_return_pct": (equity["Equity"].iloc[-1] - 1) * 100,
                "max_drawdown_pct": equity["Drawdown"].min() * 100,
                "avg_holding_bars": trades_df["holding_bars"].mean(),
                "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
            }
        ]
    )

    return trades_df, equity, summary
