"""Build a clean 5yr ES 5-minute parquet for drift/strategy backtesting.

Output: ~/projects/backtesting/es/data/es_5m_5yr_from_1m.parquet
Covers >= 2021-01-01 through the latest bar in the clean 2yr 5m file (~2026-02).

Sources (all read-only):
  1m databento  ES_1min_2010_2022_databento.csv  (clean at rest 2026-04-13; 2010-06 -> 2021-12-31)
  1m continuous ES_continuous_2022_2026.csv      (different provider; 2022-01-02 -> 2026-02-19)
  5m clean 2yr  es_5m_databento_2yr.csv          (clean at rest 2026-04-13; 2024-02-15 -> 2026-02-23)

Pipeline:
  - resample 1m -> 5m using OHLCV agg (open=first, high=max, low=min, close=last, volume=sum)
  - apply spread-tick scrub (drop any row with OHLC < $500 or OHLC consistency violation)
  - prefer the clean 2yr 5m file on overlap
  - fill the 2022-01 -> 2024-02 gap from the continuous (different-provider) 1m, after a
    cross-check against the clean 2yr 5m on their overlap range
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/chuck_mf_norris/projects")
DATABENTO_1M = ROOT / "backtesting/spx/data/ES_1min_2010_2022_databento.csv"
CONTINUOUS_1M = ROOT / "backtesting/spx/data/ES_continuous_2022_2026.csv"
CLEAN_5M_2YR = ROOT / "backtesting/es/data/es_5m_databento_2yr.csv"
OUTPUT = ROOT / "backtesting/es/data/es_5m_5yr_from_1m.parquet"

START = pd.Timestamp("2021-01-01", tz="US/Eastern")
OUTRIGHT_MIN_PRICE = 500.0
OHLC_COLS = ["open", "high", "low", "close"]


def scrub_ohlc(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n0 = len(df)
    outright = (df[OHLC_COLS] >= OUTRIGHT_MIN_PRICE).all(axis=1)
    consistent = (
        (df["low"] <= df["high"])
        & (df["open"].between(df["low"], df["high"]))
        & (df["close"].between(df["low"], df["high"]))
    )
    df = df[outright & consistent].copy()
    dropped = n0 - len(df)
    pct = (100 * dropped / n0) if n0 else 0.0
    print(f"  [{label}] scrubbed {dropped:,} rows of {n0:,} ({pct:.3f}%)")
    return df


def load_databento_1m(path: Path) -> pd.DataFrame:
    print(f"loading databento 1m: {path.name}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert("US/Eastern")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = scrub_ohlc(df, "databento_1m")
    return df


def load_continuous_1m(path: Path) -> pd.DataFrame:
    print(f"loading continuous 1m: {path.name}")
    df = pd.read_csv(path)
    naive = pd.to_datetime(df["Time - PST"], format="%y-%m-%d %H:%M")
    try:
        ts = naive.dt.tz_localize("US/Pacific", ambiguous="infer", nonexistent="shift_forward")
    except Exception as e:
        print(f"  [continuous_1m] DST infer failed ({e}); falling back to NaT + drop")
        ts = naive.dt.tz_localize("US/Pacific", ambiguous="NaT", nonexistent="NaT")
    df["timestamp"] = ts.dt.tz_convert("US/Eastern")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna(subset=["timestamp"])
    df = scrub_ohlc(df, "continuous_1m")
    return df


def load_clean_5m_2yr(path: Path) -> pd.DataFrame:
    print(f"loading clean 5m 2yr: {path.name}")
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Date/Time": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    naive = pd.to_datetime(df["timestamp"])
    try:
        ts = naive.dt.tz_localize("US/Eastern", ambiguous="infer", nonexistent="shift_forward")
    except Exception as e:
        print(f"  [clean_5m_2yr] DST infer failed ({e}); falling back to NaT + drop")
        ts = naive.dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
    df["timestamp"] = ts
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna(subset=["timestamp"])
    df = scrub_ohlc(df, "clean_5m_2yr")
    return df


def resample_1m_to_5m(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Aggregate 1m -> 5m bars labeled at bar-open."""
    df = df.set_index("timestamp").sort_index()
    out = df.resample("5min", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    out = out.dropna(subset=["open"]).reset_index()
    out = scrub_ohlc(out, f"{label}_post_resample")
    return out


def is_rth_mask(ts: pd.Series) -> pd.Series:
    """RTH = weekday 09:30 (incl) - 16:00 (excl) ET. Half-days flagged elsewhere; we keep them."""
    weekday = ts.dt.weekday < 5
    t = ts.dt.time
    return weekday & (t >= dt.time(9, 30)) & (t < dt.time(16, 0))


def cross_check_continuous_vs_clean(continuous_5m: pd.DataFrame, clean_5m: pd.DataFrame) -> dict:
    """Verify continuous (different provider) agrees with clean 2yr databento on overlap.

    Continuous covers 2022-01-02 -> 2026-02-19, clean 2yr covers 2024-02-15 -> 2026-02-23.
    Overlap: 2024-02-15 -> 2026-02-19. RTH bars only for the comparison.
    """
    ca = continuous_5m[is_rth_mask(continuous_5m["timestamp"])].set_index("timestamp")
    cb = clean_5m[is_rth_mask(clean_5m["timestamp"])].set_index("timestamp")
    common = ca.index.intersection(cb.index)
    if not len(common):
        return {"overlap_bars": 0}
    a = ca.loc[common, OHLC_COLS]
    b = cb.loc[common, OHLC_COLS]
    diff = (a - b).abs()
    pct = (diff / b).where(b > 0)
    stats = {
        "overlap_bars": int(len(common)),
        "first_overlap": str(common.min()),
        "last_overlap": str(common.max()),
        "median_abs_close_diff_pts": float(diff["close"].median()),
        "p99_abs_close_diff_pts": float(diff["close"].quantile(0.99)),
        "max_abs_close_diff_pts": float(diff["close"].max()),
        "median_abs_close_diff_bps": float(pct["close"].median() * 1e4),
        "p99_abs_close_diff_bps": float(pct["close"].quantile(0.99) * 1e4),
        "median_volume_ratio": float((ca.loc[common, "volume"] / cb.loc[common, "volume"]).replace([float("inf"), float("-inf")], pd.NA).median()),
    }
    return stats


def detect_gaps(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Find adjacent-row gaps > 24h in RTH bars (true session gaps; weekends excluded)."""
    rth = df[is_rth_mask(df["timestamp"])].sort_values("timestamp")
    if rth.empty:
        return []
    deltas = rth["timestamp"].diff()
    big = rth[deltas > pd.Timedelta("24h")]
    out = []
    for idx in big.index:
        prior_ts = rth["timestamp"].loc[:idx].iloc[-2]
        cur_ts = rth["timestamp"].loc[idx]
        delta_hours = (cur_ts - prior_ts).total_seconds() / 3600
        out.append((prior_ts, cur_ts, delta_hours))
    return out


def main():
    db_1m = load_databento_1m(DATABENTO_1M)
    db_1m = db_1m[db_1m["timestamp"] >= START]
    db_5m = resample_1m_to_5m(db_1m, "databento")
    print(f"  databento 5m: {len(db_5m):,} bars, {db_5m['timestamp'].min()} -> {db_5m['timestamp'].max()}")

    cont_1m = load_continuous_1m(CONTINUOUS_1M)
    cont_5m = resample_1m_to_5m(cont_1m, "continuous")
    print(f"  continuous 5m: {len(cont_5m):,} bars, {cont_5m['timestamp'].min()} -> {cont_5m['timestamp'].max()}")

    clean_5m = load_clean_5m_2yr(CLEAN_5M_2YR)
    print(f"  clean 5m 2yr: {len(clean_5m):,} bars, {clean_5m['timestamp'].min()} -> {clean_5m['timestamp'].max()}")

    print("\ncross-check: continuous (different provider) vs clean 2yr databento on overlap")
    cc = cross_check_continuous_vs_clean(cont_5m, clean_5m)
    for k, v in cc.items():
        print(f"  {k}: {v}")

    db_end = db_5m["timestamp"].max()
    clean_start = clean_5m["timestamp"].min()
    print(f"\ndatabento ends {db_end}; clean 2yr starts {clean_start}")
    print(f"continuous fills the gap from > {db_end} to < {clean_start}")

    cont_fill = cont_5m[(cont_5m["timestamp"] > db_end) & (cont_5m["timestamp"] < clean_start)]
    print(f"  continuous gap-fill bars: {len(cont_fill):,}")

    parts = [
        db_5m.assign(source="databento_1m_resampled"),
        cont_fill.assign(source="continuous_1m_resampled"),
        clean_5m.assign(source="clean_5m_2yr"),
    ]
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    merged["is_rth"] = is_rth_mask(merged["timestamp"])
    merged = merged.reset_index(drop=True)

    print(f"\nmerged 5m bars: {len(merged):,}")
    print(f"  range: {merged['timestamp'].min()} -> {merged['timestamp'].max()}")
    print(f"  by source:")
    print(merged["source"].value_counts().to_string())
    print(f"  RTH bars: {int(merged['is_rth'].sum()):,} / total {len(merged):,}")

    print("\ngap detection (RTH-RTH gaps > 24h):")
    gaps = detect_gaps(merged)
    if not gaps:
        print("  (none)")
    else:
        for prior, cur, hours in gaps:
            print(f"  {prior} -> {cur}  ({hours:.1f}h)")

    print("\nintegrity checks:")
    neg_low = (merged["low"] < OUTRIGHT_MIN_PRICE).sum()
    print(f"  rows with low < $500: {int(neg_low)}")
    rth_zero_vol = ((merged["is_rth"]) & (merged["volume"] == 0)).sum()
    print(f"  RTH bars with zero volume: {int(rth_zero_vol)}")
    bad_consistency = (
        (merged["high"] < merged["low"])
        | ~merged["open"].between(merged["low"], merged["high"])
        | ~merged["close"].between(merged["low"], merged["high"])
    ).sum()
    print(f"  rows with OHLC inconsistency: {int(bad_consistency)}")

    rth_per_day = merged[merged["is_rth"]].groupby(merged["timestamp"].dt.date).size()
    print(f"  RTH bars/day  mean={rth_per_day.mean():.1f}  median={rth_per_day.median():.0f}  p10={rth_per_day.quantile(0.10):.0f}  p90={rth_per_day.quantile(0.90):.0f}  trading days={len(rth_per_day)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT, index=False)
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\nwrote {OUTPUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
