"""Optional data download script.

This repo already includes the uploaded raw CSV in data/raw/.
Use this script only when you want to replace it with a fresh export.

The yfinance route may not always have NSSS.JK coverage depending on vendor access.
"""

from __future__ import annotations

from pathlib import Path
import sys

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit("Install yfinance first: pip install yfinance") from exc


def main():
    out = Path("data/raw/NSSS_yfinance.csv")
    df = yf.download("NSSS.JK", start="2023-01-01", auto_adjust=False)
    df.reset_index().to_csv(out, index=False)
    print(f"Saved {out} with shape {df.shape}")


if __name__ == "__main__":
    main()
