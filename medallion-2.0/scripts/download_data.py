"""
Medallion 2.0 — Data Download Script
Downloads market data from yfinance, FRED, and other free sources.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.settings import TICKERS, FRED_SERIES, RAW_DATA_DIR


def download_yahoo_data(tickers: list[str], period: str = "10y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Download OHLCV data from Yahoo Finance."""
    data = {}
    for ticker in tickers:
        print(f"  Downloading {ticker}...")
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if not df.empty:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"    {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
            else:
                print(f"    WARNING: No data for {ticker}")
        except Exception as e:
            print(f"    ERROR: {ticker} — {e}")
    return data


def download_fred_data(series: dict[str, str]) -> dict[str, pd.Series]:
    """Download economic data from FRED."""
    try:
        from fredapi import Fred
        fred = Fred(api_key="your_fred_api_key_here")  # TODO: Add FRED API key
        data = {}
        for name, series_id in series.items():
            print(f"  Downloading FRED {name} ({series_id})...")
            try:
                s = fred.get_series(series_id)
                data[name] = s
                print(f"    {len(s)} observations")
            except Exception as e:
                print(f"    ERROR: {name} — {e}")
        return data
    except Exception as e:
        print(f"  FRED download skipped (need API key): {e}")
        return {}


def build_feature_matrix(market_data: dict[str, pd.DataFrame], fred_data: dict[str, pd.Series]) -> pd.DataFrame:
    """Build the feature matrix from downloaded data."""
    spy = market_data.get("SPY")
    if spy is None:
        raise ValueError("SPY data required")

    features = pd.DataFrame(index=spy.index)

    # Log returns
    features["log_returns"] = np.log(spy["Close"] / spy["Close"].shift(1))

    # Realized volatility (multiple windows)
    for window in [5, 10, 21, 63]:
        features[f"realized_vol_{window}d"] = features["log_returns"].rolling(window).std() * np.sqrt(252)

    # Momentum
    for window in [5, 10, 21, 63, 126, 252]:
        features[f"momentum_{window}d"] = spy["Close"].pct_change(window)

    # Volume features
    features["volume_sma_ratio"] = spy["Volume"] / spy["Volume"].rolling(20).mean()

    # VIX
    vix = market_data.get("^VIX")
    if vix is not None:
        features["vix_level"] = vix["Close"].reindex(features.index, method="ffill")
        features["vix_change"] = features["vix_level"].pct_change()
        features["vix_term_structure"] = features["vix_level"] - features["realized_vol_21d"] * 100

    # Sector dispersion (cross-sectional vol of sector returns)
    sector_returns = pd.DataFrame()
    for ticker in TICKERS.get("sectors", []):
        if ticker in market_data:
            sector_returns[ticker] = np.log(market_data[ticker]["Close"] / market_data[ticker]["Close"].shift(1))
    if not sector_returns.empty:
        features["sector_dispersion"] = sector_returns.std(axis=1)

    # Bond/equity correlation (rolling 21-day)
    tlt = market_data.get("TLT")
    if tlt is not None:
        tlt_ret = np.log(tlt["Close"] / tlt["Close"].shift(1)).reindex(features.index)
        features["bond_equity_corr"] = features["log_returns"].rolling(21).corr(tlt_ret)

    # High-yield spread proxy (HYG vs LQD)
    hyg = market_data.get("HYG")
    lqd = market_data.get("LQD")
    if hyg is not None and lqd is not None:
        hyg_ret = np.log(hyg["Close"] / hyg["Close"].shift(1)).reindex(features.index)
        lqd_ret = np.log(lqd["Close"] / lqd["Close"].shift(1)).reindex(features.index)
        features["credit_spread_proxy"] = (lqd_ret - hyg_ret).rolling(21).mean() * 252

    # FRED data
    for name, series in fred_data.items():
        features[name] = series.reindex(features.index, method="ffill")

    # Yield curve slope (10Y - 2Y)
    if "yield_10y" in features.columns and "yield_2y" in features.columns:
        features["yield_curve_slope"] = features["yield_10y"] - features["yield_2y"]

    return features


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all tickers
    all_tickers = []
    for group in TICKERS.values():
        all_tickers.extend(group)
    all_tickers = list(set(all_tickers))

    print(f"=== Downloading {len(all_tickers)} tickers from Yahoo Finance ===")
    market_data = download_yahoo_data(all_tickers, period="10y")

    print(f"\n=== Downloading FRED economic data ===")
    fred_data = download_fred_data(FRED_SERIES)

    # Save raw data
    print(f"\n=== Saving raw data to {RAW_DATA_DIR} ===")
    for ticker, df in market_data.items():
        safe_name = ticker.replace("^", "").lower()
        path = RAW_DATA_DIR / f"{safe_name}_daily.parquet"
        df.to_parquet(path)
        print(f"  Saved {path.name}")

    for name, series in fred_data.items():
        path = RAW_DATA_DIR / f"fred_{name}.parquet"
        series.to_frame(name).to_parquet(path)
        print(f"  Saved {path.name}")

    # Build and save feature matrix
    print(f"\n=== Building feature matrix ===")
    features = build_feature_matrix(market_data, fred_data)
    features_path = Path(RAW_DATA_DIR).parent / "processed" / "features.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_path)
    print(f"  Saved {features_path.name}: {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"  Features: {list(features.columns)}")
    print(f"  Date range: {features.index[0].date()} to {features.index[-1].date()}")
    print(f"  NaN summary:\n{features.isna().sum()}")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
