#!/usr/bin/env python3
"""Download ES 5-min OHLCV data from Databento.

Outputs CSV in thinkorSwim format (Date/Time, Open, High, Low, Close, Volume)
compatible with the backtester's load_tos_csv().

Usage:
    # Download Year 2 (Feb 2024 - Feb 2025)
    python scripts/download_databento.py --start 2024-02-16 --end 2025-02-15 -o data/es_5m_databento_yr2.csv

    # Download any date range
    python scripts/download_databento.py --start 2023-01-01 --end 2024-01-01 -o data/es_5m_custom.csv
"""

import argparse
import os
import sys

import databento as db
import pandas as pd


API_KEY = os.environ.get("DATABENTO_API_KEY")
if not API_KEY:
    sys.exit("DATABENTO_API_KEY not set in environment (source ~/.databento.env)")


def main():
    parser = argparse.ArgumentParser(description="Download ES 5-min data from Databento")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument("--symbol", default="ES.FUT", help="Symbol (default: ES.FUT)")
    args = parser.parse_args()

    print(f"Connecting to Databento...")
    client = db.Historical(key=API_KEY)

    print(f"Requesting {args.symbol} 1-min OHLCV: {args.start} to {args.end}...")
    print(f"(Will resample to 5-min after download)")
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[args.symbol],
        stype_in="parent",
        schema="ohlcv-1m",
        start=args.start,
        end=args.end,
    )

    print("Converting to DataFrame...")
    df = data.to_df()
    print(f"Raw 1-min records: {len(df)}")

    # Prices are already in dollars (float64), no scaling needed

    # Filter to front-month only: for each timestamp, keep the row
    # with the highest volume (front-month contract has most volume)
    df = df.reset_index()
    df = df.sort_values(["ts_event", "volume"], ascending=[True, False])
    df = df.drop_duplicates(subset=["ts_event"], keep="first")
    df = df.set_index("ts_event").sort_index()
    print(f"After front-month filter: {len(df)}")

    # Convert index to ET
    df.index = df.index.tz_convert("America/New_York")

    # Resample 1-min to 5-min
    print("Resampling to 5-min bars...")
    df_5m = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    print(f"5-min records: {len(df_5m)}")

    df_5m = df_5m.reset_index()
    out = pd.DataFrame({
        "Date/Time": df_5m["ts_event"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Open": df_5m["open"].values,
        "High": df_5m["high"].values,
        "Low": df_5m["low"].values,
        "Close": df_5m["close"].values,
        "Volume": df_5m["volume"].astype(int).values,
    })

    # Drop any rows with zero prices (contract rollovers, gaps)
    out = out[(out["Open"] > 0) & (out["Close"] > 0)]

    print(f"Output records: {len(out)}")
    print(f"Date range: {out['Date/Time'].iloc[0]} to {out['Date/Time'].iloc[-1]}")

    out.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
