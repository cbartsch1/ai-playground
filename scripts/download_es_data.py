#!/usr/bin/env python3
"""Download ES futures 5-minute data from Yahoo Finance.

Yahoo limits 5-minute data to ~60 days. For longer history,
use the thinkorSwim strategy report export method (see data/README.md).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")


def download_es(days: int = 60, output: str = "data/es_5m.csv") -> None:
    print(f"Downloading ES=F 5-minute data ({days} days)...")
    es = yf.download("ES=F", period=f"{days}d", interval="5m", progress=False)

    # Flatten multi-level columns from yfinance
    if isinstance(es.columns, pd.MultiIndex):
        es.columns = es.columns.get_level_values(0)

    # Convert UTC to ET
    es.index = es.index.tz_convert(ET)

    # Rename to match our expected format
    es = es.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    })

    # Keep only OHLCV
    es = es[["Open", "High", "Low", "Close", "Volume"]]

    # Add Date/Time column from index
    es.insert(0, "Date/Time", es.index.strftime("%Y-%m-%d %H:%M:%S"))

    # Reset index for clean CSV
    es = es.reset_index(drop=True)

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output)
    es.to_csv(output_path, index=False)
    print(f"Saved {len(es)} bars to {output_path}")
    print(f"Date range: {es['Date/Time'].iloc[0]} to {es['Date/Time'].iloc[-1]}")


if __name__ == "__main__":
    download_es()
