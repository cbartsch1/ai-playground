"""
Medallion 2.0 — Data Loader

Downloads OHLCV data from yfinance, computes HMM features (returns, range,
volume_vol), 8 technical confirmations, and optionally 5 macro confirmations
(13 total) for the expanded voting system.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
import numpy as np
from config.settings import (
    DEFAULT_TICKER,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    CONFIRMATIONS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


def download_ohlcv(
    ticker: str = DEFAULT_TICKER,
    interval: str = DEFAULT_INTERVAL,
    period: str = DEFAULT_PERIOD,
) -> pd.DataFrame:
    """Download OHLCV data from yfinance."""
    print(f"Downloading {ticker} {interval} data ({period})...")
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure timezone-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")

    print(f"  {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 3 core HMM features:
    1. returns    — log returns
    2. range      — (high - low) / close (intrabar volatility proxy)
    3. volume_vol — rolling std of log volume (volume volatility)
    """
    features = pd.DataFrame(index=df.index)

    # Log returns
    features["returns"] = np.log(df["Close"] / df["Close"].shift(1))

    # Range: (High - Low) / Close — normalized intrabar volatility
    features["range"] = (df["High"] - df["Low"]) / df["Close"]

    # Volume volatility: rolling std of log volume (20-period)
    log_vol = np.log(df["Volume"].replace(0, np.nan))
    features["volume_vol"] = log_vol.rolling(20).std()

    return features


def compute_confirmations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 8 confirmation indicators and their pass/fail status.

    Returns DataFrame with indicator values and boolean pass columns.
    """
    conf = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 1. RSI (14-period) — pass if < 90 (not overbought)
    rsi_period = CONFIRMATIONS["rsi"]["period"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    conf["rsi"] = 100 - (100 / (1 + rs))
    conf["rsi_pass"] = conf["rsi"] < CONFIRMATIONS["rsi"]["threshold"]

    # 2. Momentum (20-period pct change) — pass if > 1%
    mom_period = CONFIRMATIONS["momentum"]["period"]
    conf["momentum"] = close.pct_change(mom_period)
    conf["momentum_pass"] = conf["momentum"] > CONFIRMATIONS["momentum"]["threshold"]

    # 3. Volatility (20-period rolling std of returns) — pass if < 6%
    vol_period = CONFIRMATIONS["volatility"]["period"]
    conf["volatility"] = np.log(close / close.shift(1)).rolling(vol_period).std()
    conf["volatility_pass"] = conf["volatility"] < CONFIRMATIONS["volatility"]["threshold"]

    # 4. Volume > 20-period SMA — pass if volume above average
    vol_sma_period = CONFIRMATIONS["volume"]["sma_period"]
    vol_sma = volume.rolling(vol_sma_period).mean()
    conf["volume_ratio"] = volume / vol_sma.replace(0, np.nan)
    conf["volume_pass"] = conf["volume_ratio"] > CONFIRMATIONS["volume"]["threshold"]

    # 5. ADX > 25 (14-period) — pass if trend is strong
    adx_period = CONFIRMATIONS["adx"]["period"]
    conf["adx"] = _compute_adx(high, low, close, adx_period)
    conf["adx_pass"] = conf["adx"] > CONFIRMATIONS["adx"]["threshold"]

    # 6. Price > EMA 50 — pass if above medium-term trend
    ema_50_period = CONFIRMATIONS["ema_50"]["period"]
    conf["ema_50"] = close.ewm(span=ema_50_period, adjust=False).mean()
    conf["ema_50_pass"] = close > conf["ema_50"]

    # 7. Price > EMA 200 — pass if above long-term trend
    ema_200_period = CONFIRMATIONS["ema_200"]["period"]
    conf["ema_200"] = close.ewm(span=ema_200_period, adjust=False).mean()
    conf["ema_200_pass"] = close > conf["ema_200"]

    # 8. MACD > Signal — pass if momentum is bullish
    macd_fast = CONFIRMATIONS["macd"]["fast"]
    macd_slow = CONFIRMATIONS["macd"]["slow"]
    macd_signal = CONFIRMATIONS["macd"]["signal"]
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    conf["macd_line"] = ema_fast - ema_slow
    conf["macd_signal"] = conf["macd_line"].ewm(span=macd_signal, adjust=False).mean()
    conf["macd_histogram"] = conf["macd_line"] - conf["macd_signal"]
    conf["macd_pass"] = conf["macd_line"] > conf["macd_signal"]

    # Total confirmations passing
    pass_cols = [c for c in conf.columns if c.endswith("_pass")]
    conf["confirmations_met"] = conf[pass_cols].sum(axis=1).astype(int)

    return conf


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX)."""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(period).mean()

    return adx


def compute_macro_confirmations(
    macro_aligned: pd.DataFrame,
    internals_aligned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute 5 macro confirmation indicators (indices 9-13).

    Args:
        macro_aligned: FRED macro data aligned to hourly index
        internals_aligned: Market internals aligned to hourly index

    Returns:
        DataFrame with 5 macro pass/fail columns + macro_confirmations_met count
    """
    from config.settings import MACRO_CONFIRMATIONS

    idx = macro_aligned.index
    conf = pd.DataFrame(index=idx)

    # 9. VIX Term Structure — pass if VIX/VIX3M < 1.0 (contango = complacency)
    if "vix_ratio" in internals_aligned.columns:
        conf["vix_term_structure"] = internals_aligned["vix_ratio"]
        threshold = MACRO_CONFIRMATIONS["vix_term_structure"]["threshold"]
        conf["vix_term_structure_pass"] = conf["vix_term_structure"] < threshold
    else:
        conf["vix_term_structure"] = np.nan
        conf["vix_term_structure_pass"] = False

    # 10. Credit Spread — pass if HY OAS < 5.0%
    if "credit_spread" in macro_aligned.columns:
        conf["credit_spread_val"] = macro_aligned["credit_spread"]
        threshold = MACRO_CONFIRMATIONS["credit_spread"]["threshold"]
        conf["credit_spread_pass"] = conf["credit_spread_val"] < threshold
    else:
        conf["credit_spread_val"] = np.nan
        conf["credit_spread_pass"] = False

    # 11. Yield Curve — pass if 10Y-2Y > 0 (not inverted)
    if "yield_10y" in macro_aligned.columns and "yield_2y" in macro_aligned.columns:
        conf["yield_curve"] = macro_aligned["yield_10y"] - macro_aligned["yield_2y"]
        threshold = MACRO_CONFIRMATIONS["yield_curve"]["threshold"]
        conf["yield_curve_pass"] = conf["yield_curve"] > threshold
    else:
        conf["yield_curve"] = np.nan
        conf["yield_curve_pass"] = False

    # 12. Market Breadth — pass if RSP/SPY slope > 0 (broadening participation)
    if "breadth_slope" in internals_aligned.columns:
        conf["breadth_slope"] = internals_aligned["breadth_slope"]
        threshold = MACRO_CONFIRMATIONS["market_breadth"]["threshold"]
        conf["breadth_pass"] = conf["breadth_slope"] > threshold
    else:
        conf["breadth_slope"] = np.nan
        conf["breadth_pass"] = False

    # 13. Dollar Strength — pass if DXY weekly change < 1% (dollar not surging)
    if "dxy_weekly_chg" in internals_aligned.columns:
        conf["dxy_weekly_chg"] = internals_aligned["dxy_weekly_chg"]
        threshold = MACRO_CONFIRMATIONS["dollar_strength"]["threshold"]
        conf["dollar_pass"] = conf["dxy_weekly_chg"].abs() < threshold
    else:
        conf["dxy_weekly_chg"] = np.nan
        conf["dollar_pass"] = False

    # Count macro confirmations
    macro_pass_cols = [c for c in conf.columns if c.endswith("_pass")]
    conf["macro_confirmations_met"] = conf[macro_pass_cols].sum(axis=1).astype(int)

    return conf


def load_data(
    ticker: str = DEFAULT_TICKER,
    interval: str = DEFAULT_INTERVAL,
    period: str = DEFAULT_PERIOD,
    cache: bool = True,
    include_macro: bool = False,
):
    """
    Load OHLCV data, compute HMM features and confirmations.

    Args:
        include_macro: If True, downloads FRED + market internals data and
                       computes 5 macro confirmations (13 total).

    Returns:
        Without macro (default): (ohlcv, hmm_features, confirmations) — 3-tuple
        With macro: (ohlcv, hmm_features, confirmations, macro_context) — 4-tuple
            where confirmations includes all 13 and macro_context is a dict
            with raw macro/internals DataFrames for dashboard use.
    """
    cache_path = PROCESSED_DATA_DIR / f"{ticker.lower()}_{interval}_{period}_cache.parquet"

    if cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        ohlcv = pd.read_parquet(cache_path)
    else:
        ohlcv = download_ohlcv(ticker, interval, period)
        # Cache the raw data
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ohlcv.to_parquet(cache_path)

    hmm_features = compute_hmm_features(ohlcv)
    confirmations = compute_confirmations(ohlcv)

    if not include_macro:
        return ohlcv, hmm_features, confirmations

    # === Download macro + internals data ===
    macro_context = {}
    macro_aligned = pd.DataFrame(index=ohlcv.index)
    internals_aligned = pd.DataFrame(index=ohlcv.index)

    try:
        from data.macro_data import download_all_macro, align_macro_to_hourly, compute_yield_curve, compute_credit_stress
        macro_raw = download_all_macro()
        macro_aligned = align_macro_to_hourly(macro_raw, ohlcv.index)
        macro_context["macro_raw"] = macro_raw
        macro_context["yield_curve"] = compute_yield_curve(macro_raw)
        macro_context["credit_stress"] = compute_credit_stress(macro_raw)
    except Exception as e:
        print(f"  Warning: Macro data unavailable: {e}")
        macro_context["macro_raw"] = pd.DataFrame()

    try:
        from data.market_internals import (
            download_vix_term_structure,
            download_dxy,
            download_breadth_proxy,
            download_put_call_ratio,
            align_internals_to_hourly,
        )
        # VIX term structure
        vix_data = download_vix_term_structure()
        macro_context["vix"] = vix_data

        # Dollar index
        dxy_data = download_dxy()
        macro_context["dxy"] = dxy_data

        # Market breadth
        breadth_data = download_breadth_proxy()
        macro_context["breadth"] = breadth_data

        # Put/call ratio
        pc_data = download_put_call_ratio()
        macro_context["put_call"] = pc_data

        # Combine internals for alignment
        all_internals = pd.DataFrame(index=vix_data.index)
        for df_name, df_data in [("vix", vix_data), ("dxy", dxy_data), ("breadth", breadth_data)]:
            for col in df_data.columns:
                all_internals[col] = df_data[col].reindex(all_internals.index, method="ffill")

        internals_aligned = align_internals_to_hourly(all_internals, ohlcv.index)
    except Exception as e:
        print(f"  Warning: Market internals unavailable: {e}")

    # Compute macro confirmations
    macro_confs = compute_macro_confirmations(macro_aligned, internals_aligned)

    # Merge technical + macro confirmations
    for col in macro_confs.columns:
        confirmations[col] = macro_confs[col].reindex(confirmations.index)

    # Recompute total (technical + macro)
    all_pass_cols = [c for c in confirmations.columns if c.endswith("_pass")]
    confirmations["total_confirmations_met"] = confirmations[all_pass_cols].sum(axis=1).astype(int)

    macro_context["macro_aligned"] = macro_aligned
    macro_context["internals_aligned"] = internals_aligned

    return ohlcv, hmm_features, confirmations, macro_context


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and process market data")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker symbol")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Bar interval")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="Lookback period")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download")
    parser.add_argument("--macro", action="store_true", help="Include macro data")
    args = parser.parse_args()

    result = load_data(
        ticker=args.ticker,
        interval=args.interval,
        period=args.period,
        cache=not args.no_cache,
        include_macro=args.macro,
    )

    if args.macro:
        ohlcv, hmm_features, confirmations, macro_context = result
    else:
        ohlcv, hmm_features, confirmations = result

    print(f"\nOHLCV: {ohlcv.shape[0]} bars")
    print(f"HMM Features: {list(hmm_features.columns)}")
    print(f"Confirmations: {confirmations['confirmations_met'].describe()}")
    print(f"\nSample confirmation breakdown (last bar):")
    last = confirmations.iloc[-1]
    tech_pass = [c for c in confirmations.columns if c.endswith("_pass") and c not in [
        "vix_term_structure_pass", "credit_spread_pass", "yield_curve_pass", "breadth_pass", "dollar_pass"
    ]]
    for col in tech_pass:
        name = col.replace("_pass", "").upper()
        status = "PASS" if last[col] else "FAIL"
        print(f"  {name}: {status}")
    print(f"  Technical: {int(last['confirmations_met'])}/8")

    if args.macro and "total_confirmations_met" in confirmations.columns:
        macro_pass = ["vix_term_structure_pass", "credit_spread_pass", "yield_curve_pass", "breadth_pass", "dollar_pass"]
        for col in macro_pass:
            if col in confirmations.columns:
                name = col.replace("_pass", "").upper()
                status = "PASS" if last[col] else "FAIL"
                print(f"  {name}: {status}")
        print(f"  Total: {int(last['total_confirmations_met'])}/13")
