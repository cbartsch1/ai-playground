"""
Medallion 2.0 — Market Internals

VIX term structure, dollar index (DXY), market breadth (RSP/SPY),
put/call ratio, and cross-asset regime detection (TLT, GLD, USO).

All data sourced from yfinance with parquet caching.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from config.settings import CACHE_DIR, VIX_CACHE_HOURS, INTERNALS_CACHE_HOURS


def _download_yfinance(
    ticker: str,
    period: str = "730d",
    interval: str = "1d",
    cache_hours: float = INTERNALS_CACHE_HOURS,
) -> pd.DataFrame:
    """Download data from yfinance with parquet caching."""
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("^", "_").replace(".", "_").replace("-", "_")
    cache_path = CACHE_DIR / f"yf_{safe_ticker}_{interval}_{period}.parquet"

    if cache_path.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
        if age_hours < cache_hours:
            return pd.read_parquet(cache_path)

    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_parquet(cache_path)
    return df


def download_vix_term_structure(
    period: str = "730d",
    cache_hours: float = VIX_CACHE_HOURS,
) -> pd.DataFrame:
    """
    Download VIX and VIX3M, compute term structure ratio.

    Returns DataFrame with:
        - vix: Current VIX
        - vix3m: 3-month VIX
        - vix_ratio: VIX / VIX3M (< 1.0 = contango/complacency, > 1.0 = backwardation/fear)
        - vix_contango: Boolean (True = contango)
    """
    vix = _download_yfinance("^VIX", period=period, interval="1d", cache_hours=cache_hours)
    vix3m = _download_yfinance("^VIX3M", period=period, interval="1d", cache_hours=cache_hours)

    result = pd.DataFrame(index=vix.index)
    result["vix"] = vix["Close"]
    result["vix3m"] = vix3m["Close"].reindex(vix.index, method="ffill")
    result["vix_ratio"] = result["vix"] / result["vix3m"].replace(0, np.nan)
    result["vix_contango"] = result["vix_ratio"] < 1.0

    return result.dropna()


def download_dxy(
    period: str = "730d",
    cache_hours: float = INTERNALS_CACHE_HOURS,
) -> pd.DataFrame:
    """
    Download US Dollar Index (DXY).

    Returns DataFrame with:
        - dxy: Dollar index close
        - dxy_sma50: 50-day SMA
        - dxy_weekly_chg: Weekly percentage change
    """
    df = _download_yfinance("DX-Y.NYB", period=period, interval="1d", cache_hours=cache_hours)

    result = pd.DataFrame(index=df.index)
    result["dxy"] = df["Close"]
    result["dxy_sma50"] = df["Close"].rolling(50).mean()
    result["dxy_weekly_chg"] = df["Close"].pct_change(5) * 100  # 5 trading days
    return result


def download_breadth_proxy(
    period: str = "730d",
    cache_hours: float = INTERNALS_CACHE_HOURS,
) -> pd.DataFrame:
    """
    Download RSP (equal-weight S&P) and SPY, compute breadth ratio.
    Rising RSP/SPY = broad participation (healthy). Falling = narrow leadership (risky).

    Returns DataFrame with:
        - rsp_spy_ratio: RSP/SPY ratio
        - breadth_sma20: 20-day SMA of ratio
        - breadth_slope: Slope of 20-day SMA (positive = broadening)
    """
    rsp = _download_yfinance("RSP", period=period, interval="1d", cache_hours=cache_hours)
    spy = _download_yfinance("SPY", period=period, interval="1d", cache_hours=cache_hours)

    # Align indices
    common = rsp.index.intersection(spy.index)

    result = pd.DataFrame(index=common)
    result["rsp_spy_ratio"] = rsp.loc[common, "Close"].values / spy.loc[common, "Close"].values
    result["breadth_sma20"] = result["rsp_spy_ratio"].rolling(20).mean()
    result["breadth_slope"] = result["breadth_sma20"].diff(5) / 5  # 5-day slope

    return result


def download_put_call_ratio(
    period: str = "730d",
    cache_hours: float = INTERNALS_CACHE_HOURS,
) -> pd.DataFrame:
    """
    Try to download put/call ratio. Falls back gracefully if unavailable.

    Returns DataFrame with:
        - put_call: Raw P/C ratio (if available)
        - put_call_sma10: 10-day SMA
    """
    try:
        # CBOE Total P/C ratio via yfinance (may not be available)
        df = _download_yfinance("^PCALL", period=period, interval="1d", cache_hours=cache_hours)
        result = pd.DataFrame(index=df.index)
        result["put_call"] = df["Close"]
        result["put_call_sma10"] = df["Close"].rolling(10).mean()
        return result
    except Exception:
        # Not available — return empty DataFrame
        return pd.DataFrame(columns=["put_call", "put_call_sma10"])


def download_cross_asset(
    tickers: list[str] = None,
    period: str = "3650d",
    cache_hours: float = INTERNALS_CACHE_HOURS,
) -> dict[str, pd.DataFrame]:
    """
    Download daily data for cross-asset regime analysis.

    Args:
        tickers: List of tickers (default: TLT, GLD, USO)
        period: Lookback period

    Returns:
        Dict of {ticker: OHLCV DataFrame}
    """
    if tickers is None:
        tickers = ["TLT", "GLD", "USO"]

    result = {}
    for ticker in tickers:
        try:
            df = _download_yfinance(ticker, period=period, interval="1d", cache_hours=cache_hours)
            result[ticker] = df
        except Exception as e:
            print(f"  Warning: Failed to download {ticker}: {e}")

    return result


def compute_cross_asset_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HMM features (returns, range, volume_vol) on daily bars.
    Same 3 features as SPY hourly but on daily data.
    """
    features = pd.DataFrame(index=ohlcv.index)
    features["returns"] = np.log(ohlcv["Close"] / ohlcv["Close"].shift(1))
    features["range"] = (ohlcv["High"] - ohlcv["Low"]) / ohlcv["Close"]
    log_vol = np.log(ohlcv["Volume"].replace(0, np.nan))
    features["volume_vol"] = log_vol.rolling(20).std()
    return features


def align_internals_to_hourly(
    internals_df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Forward-fill daily market internals data to hourly index.
    Same approach as macro data alignment.
    """
    # Ensure tz-naive
    int_idx = internals_df.index
    if hasattr(int_idx, 'tz') and int_idx.tz is not None:
        int_idx = int_idx.tz_localize(None)

    target_idx = hourly_index
    if hasattr(target_idx, 'tz') and target_idx.tz is not None:
        target_idx = target_idx.tz_localize(None)

    df = internals_df.copy()
    df.index = int_idx

    aligned = df.reindex(target_idx, method="ffill")

    if hasattr(hourly_index, 'tz') and hourly_index.tz is not None:
        aligned.index = aligned.index.tz_localize(hourly_index.tz)

    return aligned


if __name__ == "__main__":
    print("Downloading market internals...")

    print("\n--- VIX Term Structure ---")
    try:
        vix = download_vix_term_structure()
        last = vix.iloc[-1]
        print(f"VIX: {last['vix']:.2f}, VIX3M: {last['vix3m']:.2f}")
        print(f"Ratio: {last['vix_ratio']:.3f} ({'Contango' if last['vix_contango'] else 'Backwardation'})")
    except Exception as e:
        print(f"Failed: {e}")

    print("\n--- Dollar Index ---")
    try:
        dxy = download_dxy()
        last = dxy.dropna().iloc[-1]
        print(f"DXY: {last['dxy']:.2f}, SMA50: {last['dxy_sma50']:.2f}")
        print(f"Weekly change: {last['dxy_weekly_chg']:.2f}%")
    except Exception as e:
        print(f"Failed: {e}")

    print("\n--- Market Breadth ---")
    try:
        breadth = download_breadth_proxy()
        last = breadth.dropna().iloc[-1]
        print(f"RSP/SPY ratio: {last['rsp_spy_ratio']:.4f}")
        print(f"Slope (5d): {last['breadth_slope']:.6f}")
    except Exception as e:
        print(f"Failed: {e}")

    print("\n--- Cross-Asset ---")
    try:
        cross = download_cross_asset()
        for ticker, df in cross.items():
            print(f"{ticker}: {len(df)} bars, last close ${df['Close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"Failed: {e}")
