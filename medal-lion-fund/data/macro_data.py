"""
Medallion 2.0 — Macro Economic Data

Downloads FRED economic series, caches as parquet, aligns to hourly index.
Computes yield curve (10Y-2Y) and credit stress (z-score of HY OAS).

Requires: FRED_API_KEY in .env or environment variable.
Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config.settings import CACHE_DIR, FRED_SERIES, FRED_CACHE_HOURS


def get_fred_api_key() -> str:
    """Load FRED API key from environment or .env file."""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("FRED_API_KEY=") and not line.startswith("#"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key and key != "your_key_here":
                    return key

    raise ValueError(
        "FRED_API_KEY not found. Set it in .env or as environment variable.\n"
        "Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html"
    )


def get_fred_client():
    """Create a FRED API client."""
    from fredapi import Fred
    return Fred(api_key=get_fred_api_key())


def download_fred_series(
    series_id: str,
    cache_hours: float = FRED_CACHE_HOURS,
    start_date: str = "2020-01-01",
) -> pd.Series:
    """
    Download a single FRED series with parquet caching.

    Args:
        series_id: FRED series ID (e.g., 'DGS10')
        cache_hours: Hours before cache expires
        start_date: Earliest date to fetch

    Returns:
        pd.Series with DatetimeIndex
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"fred_{series_id.lower()}.parquet"

    # Check cache freshness
    if cache_path.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
        if age_hours < cache_hours:
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0]

    # Download from FRED
    fred = get_fred_client()
    data = fred.get_series(series_id, observation_start=start_date)

    if data is None or data.empty:
        raise ValueError(f"No data returned for FRED series {series_id}")

    # Cache as parquet
    df = data.to_frame(name=series_id)
    df.to_parquet(cache_path)

    return data


def download_all_macro(cache_hours: float = FRED_CACHE_HOURS) -> pd.DataFrame:
    """
    Download all configured FRED series.

    Returns:
        DataFrame with columns for each series, DatetimeIndex (daily)
    """
    frames = {}
    for key, series_id in FRED_SERIES.items():
        try:
            frames[key] = download_fred_series(series_id, cache_hours=cache_hours)
        except Exception as e:
            print(f"  Warning: Failed to download {key} ({series_id}): {e}")

    if not frames:
        raise ValueError("Failed to download any FRED series")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df


def align_macro_to_hourly(macro_df: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Forward-fill daily/weekly/monthly macro data to hourly index.
    Uses shift(1) to prevent look-ahead bias (today's data only available after close).

    Args:
        macro_df: DataFrame with daily+ frequency data
        hourly_index: Target hourly DatetimeIndex

    Returns:
        DataFrame aligned to hourly_index with forward-filled values
    """
    # Ensure tz-naive for alignment
    macro_idx = macro_df.index
    if hasattr(macro_idx, 'tz') and macro_idx.tz is not None:
        macro_idx = macro_idx.tz_localize(None)

    target_idx = hourly_index
    if hasattr(target_idx, 'tz') and target_idx.tz is not None:
        target_idx = target_idx.tz_localize(None)

    macro_df = macro_df.copy()
    macro_df.index = macro_idx

    # Shift by 1 day to prevent look-ahead bias
    # Today's FRED data shouldn't be used until next trading day
    macro_df = macro_df.shift(1)

    # Reindex to hourly with forward-fill
    aligned = macro_df.reindex(target_idx, method="ffill")

    # Restore timezone if original had one
    if hasattr(hourly_index, 'tz') and hourly_index.tz is not None:
        aligned.index = aligned.index.tz_localize(hourly_index.tz)

    return aligned


def compute_yield_curve(macro_df: pd.DataFrame) -> pd.Series:
    """
    Compute yield curve spread: 10Y - 2Y.
    Negative = inverted (recession signal).
    """
    if "yield_10y" not in macro_df.columns or "yield_2y" not in macro_df.columns:
        return pd.Series(dtype=float, name="yield_curve")

    spread = macro_df["yield_10y"] - macro_df["yield_2y"]
    spread.name = "yield_curve"
    return spread


def compute_credit_stress(macro_df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    Compute z-score of HY OAS credit spread.
    High z-score = credit stress / risk-off environment.

    Args:
        macro_df: Must contain 'credit_spread' column
        lookback: Rolling window for z-score (default 252 = ~1yr daily)
    """
    if "credit_spread" not in macro_df.columns:
        return pd.Series(dtype=float, name="credit_stress_z")

    cs = macro_df["credit_spread"]
    mean = cs.rolling(lookback, min_periods=60).mean()
    std = cs.rolling(lookback, min_periods=60).std()
    z = (cs - mean) / std.replace(0, np.nan)
    z.name = "credit_stress_z"
    return z


if __name__ == "__main__":
    print("Downloading FRED macro data...")
    macro = download_all_macro()
    print(f"\nDownloaded {len(macro.columns)} series, {len(macro)} rows")
    print(f"Date range: {macro.index[0]} to {macro.index[-1]}")
    print(f"\nColumns: {list(macro.columns)}")
    print(f"\nLast values:")
    print(macro.dropna(how="all").tail(3))

    yc = compute_yield_curve(macro)
    print(f"\nYield curve (10Y-2Y): {yc.dropna().iloc[-1]:.2f}%")

    cs = compute_credit_stress(macro)
    if not cs.dropna().empty:
        print(f"Credit stress z-score: {cs.dropna().iloc[-1]:.2f}")
