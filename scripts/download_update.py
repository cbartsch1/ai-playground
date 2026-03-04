#!/usr/bin/env python3
"""Download new ES 5-min OHLCV data from Databento and append to existing CSV.

Reads the last date from the existing 2yr CSV, downloads data from the next day
through the latest available date, and appends it.

Usage:
    python scripts/download_update.py
"""

import os
import sys
from datetime import datetime, timedelta

import databento as db
import pandas as pd


API_KEY = "db-QdAyrLRKET6vyi9mecsxU8T9MqQ6d"
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "es_5m_databento_2yr.csv"
)


def main():
    # 1. Read existing CSV to find last date
    print(f"Reading existing CSV: {CSV_PATH}")
    existing = pd.read_csv(CSV_PATH)
    last_datetime_str = existing["Date/Time"].iloc[-1]
    last_date = pd.Timestamp(last_datetime_str).date()
    print(f"Last date in existing data: {last_date}")
    print(f"Existing rows: {len(existing)}")

    # 2. Calculate start date (day after last date in CSV)
    start_date = last_date + timedelta(days=1)
    # Databento pay-per-use has a delay on the current day.
    # Use today as the exclusive end (gets data through yesterday).
    # If today's data is partially available, it will be included up to the cutoff.
    end_date = datetime(2026, 2, 24).date()  # exclusive end

    if start_date >= end_date:
        print(f"Data is already up to date (last date: {last_date}). Nothing to download.")
        return

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    print(f"\nDownloading ES.FUT 1-min OHLCV: {start_str} to {end_str} (exclusive)")

    # 3. Download from Databento
    client = db.Historical(key=API_KEY)

    print("Downloading data...")
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["ES.FUT"],
        stype_in="parent",
        schema="ohlcv-1m",
        start=start_str,
        end=end_str,
    )

    print("Converting to DataFrame...")
    df = data.to_df()
    print(f"Raw 1-min records: {len(df)}")

    if len(df) == 0:
        print("No data returned. The market may not have been open in this range.")
        return

    # 4. Front-month filter (keep highest volume per timestamp)
    df = df.reset_index()
    df = df.sort_values(["ts_event", "volume"], ascending=[True, False])
    df = df.drop_duplicates(subset=["ts_event"], keep="first")
    df = df.set_index("ts_event").sort_index()
    print(f"After front-month filter: {len(df)}")

    # 5. Convert to ET
    df.index = df.index.tz_convert("America/New_York")

    # 6. Resample 1-min to 5-min
    print("Resampling to 5-min bars...")
    df_5m = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    print(f"5-min records: {len(df_5m)}")

    # 7. Format to match existing CSV
    df_5m = df_5m.reset_index()
    out = pd.DataFrame({
        "Date/Time": df_5m["ts_event"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Open": df_5m["open"].values,
        "High": df_5m["high"].values,
        "Low": df_5m["low"].values,
        "Close": df_5m["close"].values,
        "Volume": df_5m["volume"].astype(int).values,
    })

    # Drop zero-price rows (contract rollovers, gaps)
    out = out[(out["Open"] > 0) & (out["Close"] > 0)]

    # Ensure no overlap with existing data
    out = out[out["Date/Time"] > last_datetime_str]
    print(f"New rows after dedup: {len(out)}")

    if len(out) == 0:
        print("No new data to append.")
        return

    print(f"New data range: {out['Date/Time'].iloc[0]} to {out['Date/Time'].iloc[-1]}")

    # 8. Append to existing CSV (no header)
    out.to_csv(CSV_PATH, mode="a", header=False, index=False)
    print(f"\nAppended {len(out)} new bars to {CSV_PATH}")

    # 9. Verify
    updated = pd.read_csv(CSV_PATH)
    print(f"Total rows after update: {len(updated)}")
    print(f"Full date range: {updated['Date/Time'].iloc[0]} to {updated['Date/Time'].iloc[-1]}")


if __name__ == "__main__":
    main()
