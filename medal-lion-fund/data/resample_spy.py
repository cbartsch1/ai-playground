#!/usr/bin/env python3
"""Resample SPY 1-minute RTH data to 1-hour OHLCV bars.

Source: spx-options/data/spy_1m_rth.parquet (489K bars, 5 years, Feb 2021 – Feb 2026)
Output: medallion-2.0/data/processed/spy_1h_5yr.parquet (~6,300 bars)

This gives us 5 years of HMM training data for walk-forward validation,
matching the SPX strategy data range (vs the 2yr yfinance limit).

Usage:
    cd ~/projects/ai-playground/medallion-2.0
    source .venv/bin/activate
    python data/resample_spy.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

MEDALLION_ROOT = Path(__file__).parent.parent
SPX_OPTIONS_ROOT = MEDALLION_ROOT.parent.parent / "spx-options"
OUTPUT_DIR = MEDALLION_ROOT / "data" / "processed"


def resample_1m_to_1h(input_path: str | Path, output_path: str | Path = None):
    """Resample 1-minute RTH data to 1-hour OHLCV bars.

    Resampling rules:
    - Open: first bar's open
    - High: max of all highs
    - Low: min of all lows
    - Close: last bar's close
    - Volume: sum of all volumes
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: Source data not found: {input_path}")
        sys.exit(1)

    print(f"  Loading {input_path.name}...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} 1-minute bars")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")

    # Standardize column names (spx-options uses lowercase)
    col_map = {}
    for col in df.columns:
        col_map[col] = col.capitalize() if col.islower() else col
    df = df.rename(columns=col_map)

    # Ensure timezone-aware index (America/New_York)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")

    # Resample to 1-hour OHLCV
    hourly = df.resample("1h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    # Filter to RTH hours only (9:30 - 15:59 ET → hours 9, 10, 11, 12, 13, 14, 15)
    # yfinance hourly data labels bars by the START of the hour:
    # 09:30 → 09:00 bar, 10:30 → 10:00 bar, etc.
    # We keep hours 9-15 to match yfinance convention
    hourly = hourly[hourly.index.hour.isin([9, 10, 11, 12, 13, 14, 15])]

    # Remove zero-volume bars (pre-market artifacts)
    hourly = hourly[hourly["Volume"] > 0]

    print(f"\n  Resampled to {len(hourly):,} 1-hour bars")
    print(f"  Range: {hourly.index[0]} to {hourly.index[-1]}")

    # Compute duration in trading days
    n_days = hourly.index.date
    unique_days = len(set(n_days))
    n_years = unique_days / 252
    print(f"  Trading days: {unique_days:,} (~{n_years:.1f} years)")

    # Quality checks
    print(f"\n  Quality checks:")
    ohlc_violations = ((hourly["High"] < hourly["Low"]) |
                       (hourly["Open"] > hourly["High"]) |
                       (hourly["Open"] < hourly["Low"]) |
                       (hourly["Close"] > hourly["High"]) |
                       (hourly["Close"] < hourly["Low"])).sum()
    print(f"    OHLC violations: {ohlc_violations}")
    zero_vol = (hourly["Volume"] == 0).sum()
    print(f"    Zero-volume bars: {zero_vol}")
    bars_per_day = len(hourly) / unique_days
    print(f"    Avg bars/day: {bars_per_day:.1f}")

    # Save
    if output_path is None:
        output_path = OUTPUT_DIR / "spy_1h_5yr.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_path)
    print(f"\n  Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    return hourly


def main():
    input_path = SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"

    print("=" * 70)
    print("  RESAMPLE SPY 1m → 1h (5-year dataset for HMM)")
    print("=" * 70)

    hourly = resample_1m_to_1h(input_path)

    # Show sample
    print(f"\n  First 5 bars:")
    print(hourly.head().to_string())
    print(f"\n  Last 5 bars:")
    print(hourly.tail().to_string())

    print(f"\n{'=' * 70}")
    print(f"  DONE — {len(hourly):,} hourly bars ready for HMM training")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
