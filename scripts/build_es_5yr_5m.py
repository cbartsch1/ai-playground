"""Build a clean 5-year ES 5-min dataset from a single data source.

Window: 2017-01-01 → 2021-12-31 (5 calendar years)
Source: ES_1min_2010_2022_databento.csv (databento front-month, 1-min ET)

Regime coverage:
  2017 — low-vol grind
  2018 — Feb vol-pocalypse, Q4 bear
  2019 — full recovery
  2020 — COVID crash + rebound
  2021 — QE bull

Output: ~/projects/backtesting/es/data/es_5m_5yr.csv
Format matches the existing 2yr baseline: naive ET, Date/Time,Open,High,Low,Close,Volume.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HOME = Path.home()
SRC = HOME / "projects/backtesting/spx/data/ES_1min_2010_2022_databento.csv"
OUT = HOME / "projects/backtesting/es/data/es_5m_5yr.csv"

START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2022-01-01")


def load_1m() -> pd.DataFrame:
    print(f"[L] loading {SRC.name}")
    df = pd.read_csv(SRC)
    ts = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert("US/Eastern")
    df["Date/Time"] = ts.dt.tz_localize(None)
    df = df[(df["Date/Time"] >= START) & (df["Date/Time"] < END)].copy()
    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    df = df[["Date/Time", "Open", "High", "Low", "Close", "Volume"]]
    before = len(df)
    mask = (df["Open"] > 100) & (df["High"] > 100) & (df["Low"] > 100) & (df["Close"] > 100)
    df = df[mask].copy()
    dropped = before - len(df)
    print(f"[L] {len(df):,} 1m rows | {df['Date/Time'].min()} → {df['Date/Time'].max()}")
    print(f"[L] scrubbed {dropped:,} corrupt prints (OHLC <= 100, databento settlement artifacts)")
    return df


def resample_to_5m(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[R] resampling → 5m")
    df = df.sort_values("Date/Time").drop_duplicates("Date/Time").set_index("Date/Time")
    agg = (
        df.resample("5min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Open"])
    )
    agg = agg[agg["Volume"] > 0]
    print(f"[R] {len(agg):,} 5m bars | {agg.index.min()} → {agg.index.max()}")
    return agg.reset_index()


def sanity_check(df: pd.DataFrame) -> None:
    print("\n[S] sanity checks")
    by_year = df.groupby(df["Date/Time"].dt.year).agg(
        bars=("Close", "size"),
        min_close=("Close", "min"),
        max_close=("Close", "max"),
        total_volume=("Volume", "sum"),
    )
    print(by_year.to_string())

    diffs = df["Date/Time"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=10)]
    print(f"\n[S] gaps > 10min: {len(gaps):,} (weekend+maintenance windows expected ~260/yr)")
    if len(gaps) > 2000:
        print("[S] WARNING: excess gaps, inspect data continuity")

    bad_hl = df[df["High"] < df["Low"]]
    bad_oc = df[(df["Open"] < df["Low"]) | (df["Open"] > df["High"])]
    print(f"[S] High < Low rows: {len(bad_hl)}  | Open outside H/L: {len(bad_oc)}")


def main() -> int:
    df = load_1m()
    five = resample_to_5m(df)
    five.to_csv(OUT, index=False)
    print(f"\n[W] wrote {OUT} ({len(five):,} rows)")
    sanity_check(five)
    return 0


if __name__ == "__main__":
    sys.exit(main())
