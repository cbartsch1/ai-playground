#!/usr/bin/env python3
"""Trend Continuation ES — Indicator & Filter Research.

Baseline: 30m close < prior 30m LOW, short-only, 30bps stop, 15pt target, max hold 30min.
374 trades/2yr, PF 1.35, +$25K, p=0.0077.

This script computes 24 indicators from OHLCV data and tests each as a filter
on the baseline signal. The goal: find filters that improve PF while keeping
statistical significance and enough trades.

Usage:
    python3 scripts/test_trend_cont_es_indicators.py
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics
from backtester.position import Trade

# ── Constants ──
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "es_5m_databento_2yr.csv")
WF_SPLIT = "2025-02-16"
ES_POINT_VALUE = 50.0
COMMISSION_RT = 4.62
SLIPPAGE_PTS = 0.50
CONTRACTS = 1
INITIAL_CAPITAL = 100_000.0
FLATTEN_TIME = 1555


# ═══════════════════════════════════════════════════════════
#  30m Aggregation (from baseline)
# ═══════════════════════════════════════════════════════════

def aggregate_30m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5m RTH bars into clock-aligned 30m bars."""
    rth = df_5m[df_5m["is_rth"]].copy()
    if rth.empty:
        return pd.DataFrame()

    rth["m30_bucket"] = rth.index.floor("30min")
    grouped = rth.groupby([rth["session_date"], rth["m30_bucket"]])

    bars_30m = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    bars_30m = bars_30m.droplevel(0).sort_index()
    bars_30m["session_date"] = grouped["session_date"].first().droplevel(0)
    bars_30m["et_time"] = bars_30m.index.hour * 100 + bars_30m.index.minute
    return bars_30m


def compute_30m_lower_close(df_30m: pd.DataFrame) -> pd.DataFrame:
    """Flag 30m bars where close < prior 30m bar's LOW (within same session)."""
    df = df_30m.copy()
    c = df["close"].values
    l = df["low"].values
    s = df["session_date"].values
    n = len(df)

    lower_close = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if s[i] == s[i-1] and c[i] < l[i-1]:
            lower_close[i] = True

    df["lower_close"] = lower_close
    return df


# ═══════════════════════════════════════════════════════════
#  INDICATOR COMPUTATION (all from OHLCV)
# ═══════════════════════════════════════════════════════════

def compute_rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI on a price series."""
    n = len(series)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gain[1:period+1])
    avg_loss = np.mean(loss[1:period+1])

    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_macd(series: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and signal line."""
    ema_fast = pd.Series(series).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(series).ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    return macd_line, signal_line


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index."""
    n = len(high)
    adx = np.full(n, 0.0)
    if n < period * 2:
        return adx

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i-1]
        l_diff = low[i-1] - low[i]
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0

    # Wilder smoothing
    atr_smooth = np.zeros(n)
    plus_smooth = np.zeros(n)
    minus_smooth = np.zeros(n)

    atr_smooth[period] = np.sum(tr[1:period+1])
    plus_smooth[period] = np.sum(plus_dm[1:period+1])
    minus_smooth[period] = np.sum(minus_dm[1:period+1])

    for i in range(period+1, n):
        atr_smooth[i] = atr_smooth[i-1] - atr_smooth[i-1]/period + tr[i]
        plus_smooth[i] = plus_smooth[i-1] - plus_smooth[i-1]/period + plus_dm[i]
        minus_smooth[i] = minus_smooth[i-1] - minus_smooth[i-1]/period + minus_dm[i]

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_smooth[i] / atr_smooth[i]
        s = plus_di[i] + minus_di[i]
        if s > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / s

    # Smooth DX to get ADX
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """True Range and ATR (Wilder smoothing)."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def compute_bollinger(close: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Bollinger Bands — returns %B."""
    n = len(close)
    pct_b = np.full(n, 0.5)
    if n < period:
        return pct_b

    sma = pd.Series(close).rolling(period).mean().values
    std = pd.Series(close).rolling(period).std().values

    for i in range(period-1, n):
        if std[i] > 0:
            upper = sma[i] + std_mult * std[i]
            lower = sma[i] - std_mult * std[i]
            bw = upper - lower
            if bw > 0:
                pct_b[i] = (close[i] - lower) / bw
    return pct_b


def compute_roc(series: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of Change."""
    n = len(series)
    roc = np.zeros(n)
    for i in range(period, n):
        if series[i-period] != 0:
            roc[i] = (series[i] - series[i-period]) / series[i-period] * 100
    return roc


def compute_ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA."""
    return pd.Series(series).ewm(span=period, adjust=False).mean().values


def compute_tema(series: np.ndarray, period: int) -> np.ndarray:
    """Triple EMA."""
    ema1 = pd.Series(series).ewm(span=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, adjust=False).mean().values
    return 3 * ema1 - 3 * ema2 + ema3


def compute_vwap_session(df_5m: pd.DataFrame) -> np.ndarray:
    """Cumulative VWAP within each RTH session."""
    n = len(df_5m)
    vwap = np.full(n, np.nan)

    cum_vol = 0.0
    cum_pv = 0.0
    prev_sess = None

    sess = df_5m["session_date"].values
    hlc3 = ((df_5m["high"].values + df_5m["low"].values + df_5m["close"].values) / 3.0)
    vol = df_5m["volume"].values.astype(float)
    is_rth = df_5m["is_rth"].values

    for i in range(n):
        if not is_rth[i]:
            continue
        if sess[i] != prev_sess:
            cum_vol = 0.0
            cum_pv = 0.0
            prev_sess = sess[i]
        v = max(vol[i], 1)
        cum_vol += v
        cum_pv += hlc3[i] * v
        vwap[i] = cum_pv / cum_vol

    return vwap


def compute_session_levels(df_5m: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute session-level data: session high, session low, opening range, distance from high,
    consecutive lower closes on 30m, overnight high/low, prev day high/low/close."""
    n = len(df_5m)

    session_high = np.full(n, np.nan)
    session_low = np.full(n, np.nan)
    opening_range_low = np.full(n, np.nan)  # Low of first 30min (930-1000)
    opening_range_high = np.full(n, np.nan)
    dist_from_session_high = np.zeros(n)
    on_high = np.full(n, np.nan)
    on_low = np.full(n, np.nan)
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)

    sess = df_5m["session_date"].values
    is_rth = df_5m["is_rth"].values
    et_time = df_5m["et_time"].values
    hi = df_5m["high"].values
    lo = df_5m["low"].values
    cl = df_5m["close"].values

    # Track state
    cur_sess = None
    cur_hi = np.nan
    cur_lo = np.nan
    cur_or_hi = np.nan
    cur_or_lo = np.nan
    cur_on_hi = np.nan
    cur_on_lo = np.nan
    p_day_hi = np.nan
    p_day_lo = np.nan
    p_day_cl = np.nan
    last_rth_close = np.nan

    for i in range(n):
        t = et_time[i]

        # Track overnight levels (globex: 18:00 - 9:29)
        is_globex = (t >= 1800) or (t < 930)

        if is_rth[i]:
            if sess[i] != cur_sess:
                # New session — freeze overnight, store prev day
                if not np.isnan(cur_hi):
                    p_day_hi = cur_hi
                    p_day_lo = cur_lo
                if not np.isnan(last_rth_close):
                    p_day_cl = last_rth_close

                cur_sess = sess[i]
                cur_hi = hi[i]
                cur_lo = lo[i]
                cur_or_hi = np.nan
                cur_or_lo = np.nan
                # Freeze overnight
                # on_hi/on_lo carry forward from globex accumulation
            else:
                cur_hi = max(cur_hi, hi[i]) if not np.isnan(cur_hi) else hi[i]
                cur_lo = min(cur_lo, lo[i]) if not np.isnan(cur_lo) else lo[i]

            # Opening range (first 30min: 930-959)
            if 930 <= t < 1000:
                if np.isnan(cur_or_hi):
                    cur_or_hi = hi[i]
                    cur_or_lo = lo[i]
                else:
                    cur_or_hi = max(cur_or_hi, hi[i])
                    cur_or_lo = min(cur_or_lo, lo[i])

            last_rth_close = cl[i]

        elif is_globex:
            # Check for new globex session start (transition from RTH to globex)
            if i > 0 and is_rth[i-1] and not is_rth[i]:
                cur_on_hi = hi[i]
                cur_on_lo = lo[i]
            elif np.isnan(cur_on_hi):
                cur_on_hi = hi[i]
                cur_on_lo = lo[i]
            else:
                cur_on_hi = max(cur_on_hi, hi[i])
                cur_on_lo = min(cur_on_lo, lo[i])

        session_high[i] = cur_hi
        session_low[i] = cur_lo
        opening_range_high[i] = cur_or_hi
        opening_range_low[i] = cur_or_lo
        on_high[i] = cur_on_hi
        on_low[i] = cur_on_lo
        prev_day_high[i] = p_day_hi
        prev_day_low[i] = p_day_lo
        prev_day_close[i] = p_day_cl

        if not np.isnan(cur_hi):
            dist_from_session_high[i] = cur_hi - cl[i]

    return {
        "session_high": session_high,
        "session_low": session_low,
        "opening_range_high": opening_range_high,
        "opening_range_low": opening_range_low,
        "dist_from_session_high": dist_from_session_high,
        "on_high": on_high,
        "on_low": on_low,
        "prev_day_high": prev_day_high,
        "prev_day_low": prev_day_low,
        "prev_day_close": prev_day_close,
    }


# ═══════════════════════════════════════════════════════════
#  PRECOMPUTE ALL 30m INDICATORS
# ═══════════════════════════════════════════════════════════

def compute_30m_indicators(df_30m: pd.DataFrame) -> pd.DataFrame:
    """Compute all 30m-timeframe indicators."""
    df = df_30m.copy()
    cl = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    vol = df["volume"].values.astype(float)
    sess = df["session_date"].values
    n = len(df)

    # 1. RSI(14) on 30m close
    df["rsi_14"] = compute_rsi(cl, 14)

    # 2. RSI divergence — price making new 5-bar low but RSI isn't
    rsi = df["rsi_14"].values
    df["rsi_divergence"] = False
    for i in range(5, n):
        if sess[i] == sess[i-5]:
            price_new_low = cl[i] < np.min(cl[i-5:i])
            rsi_not_new_low = rsi[i] > np.min(rsi[i-5:i])
            df.iloc[i, df.columns.get_loc("rsi_divergence")] = price_new_low and rsi_not_new_low

    # 3. MACD below signal line
    macd_line, signal_line = compute_macd(cl)
    df["macd_below_signal"] = macd_line < signal_line

    # 4. ADX
    df["adx"] = compute_adx(hi, lo, cl, 14)

    # 5. ROC (10-bar on 30m)
    df["roc_10"] = compute_roc(cl, 10)
    # Also compute prior bar's ROC for "accelerating" check
    roc = df["roc_10"].values
    df["roc_accelerating"] = False
    for i in range(1, n):
        if sess[i] == sess[i-1]:
            df.iloc[i, df.columns.get_loc("roc_accelerating")] = (roc[i] < roc[i-1]) and (roc[i] < 0)

    # 6. ATR expanding — current ATR(14) > ATR(20) SMA
    atr_14 = compute_atr(hi, lo, cl, 14)
    df["atr_14"] = atr_14
    atr_sma_20 = pd.Series(atr_14).rolling(20, min_periods=1).mean().values
    df["atr_expanding"] = atr_14 > atr_sma_20

    # 7. Bollinger %B
    df["boll_pct_b"] = compute_bollinger(cl, 20, 2.0)

    # 8. ATR percentile (rolling 100-bar window on 30m)
    atr_series = pd.Series(atr_14)
    df["atr_pct_rank"] = atr_series.rolling(100, min_periods=20).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
    ).values

    # 9-11. Volume indicators on 30m
    vol_float = vol.astype(float)
    vol_ma_5 = pd.Series(vol_float).rolling(5, min_periods=1).mean().values
    vol_ma_20 = pd.Series(vol_float).rolling(20, min_periods=1).mean().values
    df["vol_surge"] = vol_float > (vol_ma_20 * 1.5)
    df["vol_trend_rising"] = vol_ma_5 > vol_ma_20

    # 10. Cumulative delta proxy: (close - low) / (high - low)
    bar_range = hi - lo
    bar_range[bar_range == 0] = 0.01  # avoid div by zero
    df["cum_delta_proxy"] = (cl - lo) / bar_range

    # 16. Consecutive 30m lower closes (within session)
    df["consec_lower_close"] = 0
    for i in range(1, n):
        if sess[i] == sess[i-1] and cl[i] < cl[i-1]:
            df.iloc[i, df.columns.get_loc("consec_lower_close")] = (
                df.iloc[i-1, df.columns.get_loc("consec_lower_close")] + 1
            )

    return df


# ═══════════════════════════════════════════════════════════
#  STRATEGY ENGINE (enhanced with indicator filters)
# ═══════════════════════════════════════════════════════════

@dataclass
class IndicatorConfig:
    """Configuration for the trend continuation with indicator filters."""
    # Baseline params (FIXED — these define the passing strategy)
    # 374 trades/2yr, PF 1.35, +$25K, p=0.0077
    entry_start: int = 935
    entry_end: int = 1300
    max_hold_bars: int = 6      # 30 minutes
    stop_bps: float = 30.0
    max_trades_day: int = 1     # 1 trade per day is the validated baseline
    min_gap_bars: int = 6
    flatten_time: int = FLATTEN_TIME
    exit_mode: str = "fixed_target"
    target_pts: float = 15.0

    # Filter functions — list of (name, callable) pairs
    # Each callable takes (merged_df, i, indicators_30m) and returns bool (True = pass)
    filters: list = field(default_factory=list)
    filter_name: str = "baseline"

    # Smart exit mode (overrides fixed target when set)
    smart_exit: str = ""  # "", "atr_scaled", "prev_day_low", "on_low", "rsi_exit", "vol_climax"
    atr_exit_mult: float = 1.5


def run_with_indicators(df_5m: pd.DataFrame, df_30m: pd.DataFrame,
                        df_30m_ind: pd.DataFrame, session_levels: dict,
                        cfg: IndicatorConfig) -> List[Trade]:
    """Run Trend Continuation with indicator filters."""
    rth_5m = df_5m[df_5m["is_rth"]].copy()
    if rth_5m.empty or df_30m.empty:
        return []

    # Compute 30m lower_close signals
    df_30m_sig = compute_30m_lower_close(df_30m)

    # Build signal lookup (shifted forward to avoid lookahead)
    signal_df = df_30m_sig[["lower_close", "session_date"]].copy()
    signal_df.columns = ["m30_lower_close", "m30_session"]
    signal_df.index = signal_df.index + pd.Timedelta(minutes=30)

    # Also build 30m indicator lookup (same shift)
    ind_cols = [c for c in df_30m_ind.columns if c not in ["open", "high", "low", "close", "volume",
                                                            "session_date", "et_time", "lower_close"]]
    ind_df = df_30m_ind[ind_cols].copy()
    ind_df.index = df_30m_ind.index + pd.Timedelta(minutes=30)

    # Merge 5m with 30m signals
    merged = pd.merge_asof(
        rth_5m.reset_index(),
        signal_df.reset_index(),
        left_on="datetime",
        right_on=signal_df.index.name or "datetime",
        direction="backward",
    ).set_index("datetime")
    merged["m30_lower_close"] = merged["m30_lower_close"].fillna(False)

    # Merge 30m indicators
    merged = pd.merge_asof(
        merged.reset_index(),
        ind_df.reset_index(),
        left_on="datetime",
        right_on=ind_df.index.name or "datetime",
        direction="backward",
    ).set_index("datetime")

    # Add session levels to merged
    # Need to align by original 5m index
    rth_mask = df_5m["is_rth"].values
    rth_indices = np.where(rth_mask)[0]
    for key, arr in session_levels.items():
        rth_vals = arr[rth_indices]
        merged[key] = rth_vals[:len(merged)]

    # Compute 5m indicators for multi-timeframe filters
    cl_5m = merged["close"].values
    op_5m = merged["open"].values
    hi_5m = merged["high"].values
    lo_5m = merged["low"].values
    vol_5m = merged["volume"].values.astype(float)

    # 5m TEMA
    tema_9_5m = compute_tema(cl_5m, 9)
    tema_21_5m = compute_tema(cl_5m, 21)
    merged["tema9_5m"] = tema_9_5m
    merged["tema21_5m"] = tema_21_5m

    # 5m EMA
    ema9_5m = compute_ema(cl_5m, 9)
    ema21_5m = compute_ema(cl_5m, 21)
    merged["ema9_5m"] = ema9_5m
    merged["ema21_5m"] = ema21_5m

    # 5m RSI for exit
    merged["rsi_5m"] = compute_rsi(cl_5m, 14)

    # 5m volume for exit
    vol_ma_20_5m = pd.Series(vol_5m).rolling(20, min_periods=1).mean().values
    merged["vol_ratio_5m"] = np.where(vol_ma_20_5m > 0, vol_5m / vol_ma_20_5m, 1.0)

    # VWAP
    vwap_full = compute_vwap_session(df_5m)
    merged["vwap"] = vwap_full[rth_indices][:len(merged)]

    # 5m ATR for ATR-scaled exits
    atr_5m = compute_atr(hi_5m, lo_5m, cl_5m, 14)
    merged["atr_5m"] = atr_5m

    # Extract arrays for speed
    n = len(merged)
    et = merged["et_time"].values
    sess = merged["session_date"].values
    cl = merged["close"].values
    op = merged["open"].values
    hi = merged["high"].values
    lo = merged["low"].values
    sig = merged["m30_lower_close"].values
    times = merged.index

    # Bar-by-bar simulation
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    entry_idx = 0
    stop_price = 0.0
    target_price = 0.0

    trade_count = {}
    last_entry_bar = {}

    def make_trade(exit_time, exit_price, exit_reason):
        pnl_pts = entry_price - exit_price  # short
        pnl_dollar = (pnl_pts * ES_POINT_VALUE - COMMISSION_RT) * CONTRACTS - SLIPPAGE_PTS * ES_POINT_VALUE * CONTRACTS
        return Trade(
            setup="TC", direction=-1,
            entry_time=entry_time, entry_price=entry_price,
            exit_time=exit_time, exit_price=exit_price,
            exit_reason=exit_reason, pnl_pts=pnl_pts,
            pnl_dollar=pnl_dollar, stop=stop_price, target=target_price,
        )

    for i in range(n):
        s = sess[i]
        t = et[i]

        # ── Check exits first ──
        if in_position:
            entry_sess = sess[entry_idx]
            bars_held = i - entry_idx

            # Session changed
            if s != entry_sess:
                trades.append(make_trade(times[i-1] if i > 0 else times[i], cl[i-1] if i > 0 else cl[i], "session_end"))
                in_position = False

            # Stop hit
            elif hi[i] >= stop_price:
                trades.append(make_trade(times[i], stop_price, "stop"))
                in_position = False
                continue

            # Flatten time
            elif t >= cfg.flatten_time:
                trades.append(make_trade(times[i], cl[i], "flatten"))
                in_position = False
                continue

            # Fixed target hit
            elif target_price > 0 and lo[i] <= target_price:
                trades.append(make_trade(times[i], target_price, "target"))
                in_position = False
                continue

            # Smart exits
            elif cfg.smart_exit == "rsi_exit":
                rsi_5m_val = merged["rsi_5m"].iloc[i]
                if rsi_5m_val > 70 and bars_held >= 2:
                    trades.append(make_trade(times[i], cl[i], "rsi_exit"))
                    in_position = False
                    continue
            elif cfg.smart_exit == "vol_climax":
                vr = merged["vol_ratio_5m"].iloc[i]
                if vr > 3.0 and cl[i] < op[i] and bars_held >= 2:
                    trades.append(make_trade(times[i], cl[i], "vol_climax"))
                    in_position = False
                    continue

            # Max hold
            if in_position and bars_held >= cfg.max_hold_bars:
                trades.append(make_trade(times[i], cl[i], "max_hold"))
                in_position = False
                continue

            if in_position:
                continue

        # ── Check entry ──
        if in_position:
            continue

        if t < cfg.entry_start or t >= cfg.entry_end:
            continue
        if trade_count.get(s, 0) >= cfg.max_trades_day:
            continue
        if s in last_entry_bar and (i - last_entry_bar[s]) < cfg.min_gap_bars:
            continue
        if not sig[i]:
            continue

        # Apply indicator filters
        filter_pass = True
        for fname, ffunc in cfg.filters:
            if not ffunc(merged, i):
                filter_pass = False
                break

        if not filter_pass:
            continue

        # ENTER SHORT
        entry_price = cl[i]
        entry_time = times[i]
        entry_idx = i
        stop_price = entry_price * (1.0 + cfg.stop_bps / 10000.0)
        in_position = True

        # Target computation
        if cfg.smart_exit == "atr_scaled":
            atr_val = merged["atr_14"].iloc[i] if "atr_14" in merged.columns else 10.0
            target_price = entry_price - cfg.atr_exit_mult * atr_val
        elif cfg.smart_exit == "prev_day_low":
            pdl = merged["prev_day_low"].iloc[i]
            if not np.isnan(pdl) and pdl < entry_price:
                target_price = pdl
            else:
                target_price = entry_price - cfg.target_pts
        elif cfg.smart_exit == "on_low":
            onl = merged["on_low"].iloc[i]
            if not np.isnan(onl) and onl < entry_price:
                target_price = onl
            else:
                target_price = entry_price - cfg.target_pts
        else:
            target_price = entry_price - cfg.target_pts

        trade_count[s] = trade_count.get(s, 0) + 1
        last_entry_bar[s] = i

    # Close any open position
    if in_position:
        trades.append(make_trade(times[-1], cl[-1], "data_end"))

    return trades


# ═══════════════════════════════════════════════════════════
#  FILTER DEFINITIONS
# ═══════════════════════════════════════════════════════════

def make_filters() -> List[Tuple[str, str, list]]:
    """Build all filter variants to test.
    Returns list of (name, description, filter_list) tuples."""

    filters = []

    # ── MOMENTUM CONFIRMATION ──

    # 1. RSI < 40 on 30m
    filters.append(("RSI<40", "30m RSI < 40",
        [("rsi<40", lambda df, i: df["rsi_14"].iloc[i] < 40)]))

    filters.append(("RSI<45", "30m RSI < 45",
        [("rsi<45", lambda df, i: df["rsi_14"].iloc[i] < 45)]))

    filters.append(("RSI<50", "30m RSI < 50",
        [("rsi<50", lambda df, i: df["rsi_14"].iloc[i] < 50)]))

    # 2. RSI divergence — SKIP signal (momentum exhausting)
    filters.append(("NO_RSI_DIV", "Skip RSI divergence (momentum exhausting)",
        [("no_rsi_div", lambda df, i: not df["rsi_divergence"].iloc[i])]))

    # 3. MACD below signal line on 30m
    filters.append(("MACD<SIG", "MACD below signal line",
        [("macd_below", lambda df, i: df["macd_below_signal"].iloc[i])]))

    # 4. ADX > 25 on 30m
    filters.append(("ADX>25", "ADX > 25 (strong trend)",
        [("adx>25", lambda df, i: df["adx"].iloc[i] > 25)]))

    filters.append(("ADX>20", "ADX > 20",
        [("adx>20", lambda df, i: df["adx"].iloc[i] > 20)]))

    filters.append(("ADX>30", "ADX > 30",
        [("adx>30", lambda df, i: df["adx"].iloc[i] > 30)]))

    # 5. ROC negative and accelerating
    filters.append(("ROC_NEG", "ROC(10) < 0",
        [("roc_neg", lambda df, i: df["roc_10"].iloc[i] < 0)]))

    filters.append(("ROC_ACCEL", "ROC negative and accelerating",
        [("roc_accel", lambda df, i: df["roc_accelerating"].iloc[i])]))

    # ── VOLATILITY CONTEXT ──

    # 6. ATR expanding
    filters.append(("ATR_EXPAND", "ATR(14) > SMA(ATR,20)",
        [("atr_exp", lambda df, i: df["atr_expanding"].iloc[i])]))

    # 7. Bollinger %B < 0.2
    filters.append(("BOLL_B<0.2", "Bollinger %B < 0.2",
        [("boll_b", lambda df, i: df["boll_pct_b"].iloc[i] < 0.2)]))

    filters.append(("BOLL_B<0.3", "Bollinger %B < 0.3",
        [("boll_b3", lambda df, i: df["boll_pct_b"].iloc[i] < 0.3)]))

    # 8. ATR percentile > 70th
    filters.append(("ATR_P70", "ATR percentile > 70th",
        [("atr_p70", lambda df, i: (
            not np.isnan(df["atr_pct_rank"].iloc[i]) and df["atr_pct_rank"].iloc[i] > 0.70
        ))]))

    filters.append(("ATR_P50", "ATR percentile > 50th",
        [("atr_p50", lambda df, i: (
            not np.isnan(df["atr_pct_rank"].iloc[i]) and df["atr_pct_rank"].iloc[i] > 0.50
        ))]))

    # ── VOLUME CONFIRMATION ──

    # 9. Volume surge on break bar
    filters.append(("VOL_SURGE", "Volume > 1.5x MA(20)",
        [("vol_surge", lambda df, i: df["vol_surge"].iloc[i])]))

    # 10. Cumulative delta proxy < 0.3
    filters.append(("CDP<0.3", "Cum Delta Proxy < 0.3 (sellers dominating)",
        [("cdp<0.3", lambda df, i: df["cum_delta_proxy"].iloc[i] < 0.3)]))

    filters.append(("CDP<0.4", "Cum Delta Proxy < 0.4",
        [("cdp<0.4", lambda df, i: df["cum_delta_proxy"].iloc[i] < 0.4)]))

    # 11. Volume trend rising
    filters.append(("VOL_TREND", "5-bar vol MA > 20-bar vol MA",
        [("vol_trend", lambda df, i: df["vol_trend_rising"].iloc[i])]))

    # ── PRICE STRUCTURE ──

    # 12. Below session VWAP
    filters.append(("BELOW_VWAP", "Price below session VWAP",
        [("below_vwap", lambda df, i: (
            not np.isnan(df["vwap"].iloc[i]) and df["close"].iloc[i] < df["vwap"].iloc[i]
        ))]))

    # 13. Below opening range low
    filters.append(("BELOW_ORL", "Below opening range low",
        [("below_orl", lambda df, i: (
            not np.isnan(df["opening_range_low"].iloc[i]) and
            df["close"].iloc[i] < df["opening_range_low"].iloc[i]
        ))]))

    # 14. Below overnight low
    filters.append(("BELOW_ONL", "Below overnight low",
        [("below_onl", lambda df, i: (
            not np.isnan(df["on_low"].iloc[i]) and
            df["close"].iloc[i] < df["on_low"].iloc[i]
        ))]))

    # 15. Distance from session high > 15pts
    filters.append(("DIST_HI>15", "Distance from session high > 15pts",
        [("dist_hi_15", lambda df, i: df["dist_from_session_high"].iloc[i] > 15)]))

    filters.append(("DIST_HI>10", "Distance from session high > 10pts",
        [("dist_hi_10", lambda df, i: df["dist_from_session_high"].iloc[i] > 10)]))

    filters.append(("DIST_HI>20", "Distance from session high > 20pts",
        [("dist_hi_20", lambda df, i: df["dist_from_session_high"].iloc[i] > 20)]))

    # 16. Consecutive 30m lower closes >= 2
    filters.append(("CONSEC_LC>=2", "2+ consecutive 30m lower closes",
        [("consec_lc", lambda df, i: df["consec_lower_close"].iloc[i] >= 2)]))

    filters.append(("CONSEC_LC>=1", "1+ consecutive 30m lower closes",
        [("consec_lc1", lambda df, i: df["consec_lower_close"].iloc[i] >= 1)]))

    # ── MULTI-TIMEFRAME ──

    # 17. 5m TEMA bearish at entry
    filters.append(("TEMA5m_BEAR", "5m TEMA(9) < TEMA(21)",
        [("tema5m_bear", lambda df, i: df["tema9_5m"].iloc[i] < df["tema21_5m"].iloc[i])]))

    # 18. 5m EMA 9 < EMA 21 at entry
    filters.append(("EMA5m_BEAR", "5m EMA(9) < EMA(21)",
        [("ema5m_bear", lambda df, i: df["ema9_5m"].iloc[i] < df["ema21_5m"].iloc[i])]))

    # 19. Both 5m AND 30m bearish
    filters.append(("MULTI_TF", "5m TEMA bearish AND MACD below signal",
        [("tema5m_bear", lambda df, i: df["tema9_5m"].iloc[i] < df["tema21_5m"].iloc[i]),
         ("macd_below", lambda df, i: df["macd_below_signal"].iloc[i])]))

    return filters


# ═══════════════════════════════════════════════════════════
#  WALK-FORWARD + STATS
# ═══════════════════════════════════════════════════════════

def ttest_pval(pnls):
    if len(pnls) > 1 and np.std(pnls) > 0:
        t, p = stats.ttest_1samp(pnls, 0)
        return p / 2 if t > 0 else 1.0
    return 1.0


def walk_forward(df_5m, df_30m, df_30m_ind, session_levels, cfg, split_date=WF_SPLIT):
    """Walk-forward split and test."""
    split_ts = pd.Timestamp(split_date, tz=df_5m.index.tz)

    df_5m_is = df_5m[df_5m.index < split_ts]
    df_5m_oos = df_5m[df_5m.index >= split_ts]
    df_30m_is = df_30m[df_30m.index < split_ts]
    df_30m_oos = df_30m[df_30m.index >= split_ts]
    df_30m_ind_is = df_30m_ind[df_30m_ind.index < split_ts]
    df_30m_ind_oos = df_30m_ind[df_30m_ind.index >= split_ts]

    # Session levels need to be split based on 5m alignment
    n_is = len(df_5m_is)
    n_oos = len(df_5m_oos)
    sl_is = {k: v[:n_is] for k, v in session_levels.items()}
    sl_oos = {k: v[n_is:n_is+n_oos] for k, v in session_levels.items()}

    trades_is = run_with_indicators(df_5m_is, df_30m_is, df_30m_ind_is, sl_is, cfg)
    trades_oos = run_with_indicators(df_5m_oos, df_30m_oos, df_30m_ind_oos, sl_oos, cfg)

    m_is = compute_metrics(trades_is, INITIAL_CAPITAL) if trades_is else None
    m_oos = compute_metrics(trades_oos, INITIAL_CAPITAL) if trades_oos else None

    pnls_is = [t.pnl_dollar for t in trades_is]
    pnls_oos = [t.pnl_dollar for t in trades_oos]

    pf_ratio = 0.0
    if m_is and m_oos and m_is.profit_factor > 0:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor

    return {
        "is_trades": len(trades_is),
        "is_pf": m_is.profit_factor if m_is else 0,
        "is_pnl": m_is.net_pnl if m_is else 0,
        "is_wr": m_is.win_rate if m_is else 0,
        "is_p": ttest_pval(pnls_is),
        "oos_trades": len(trades_oos),
        "oos_pf": m_oos.profit_factor if m_oos else 0,
        "oos_pnl": m_oos.net_pnl if m_oos else 0,
        "oos_wr": m_oos.win_rate if m_oos else 0,
        "oos_p": ttest_pval(pnls_oos),
        "pf_ratio": pf_ratio,
    }


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 80)
    print("  TREND CONTINUATION ES — INDICATOR & FILTER RESEARCH")
    print("  Baseline: 30m close < prior 30m LOW, short, 30bps stop, 15pt target")
    print("=" * 80)

    # ── Load data ──
    print(f"\n  Loading data from {DATA_PATH}...")
    df_5m = load_tos_csv(DATA_PATH)
    print(f"  Loaded {len(df_5m):,} bars, {df_5m.index[0]} to {df_5m.index[-1]}")

    # ── Aggregate 30m ──
    print("  Aggregating 30m bars...")
    df_30m = aggregate_30m(df_5m)
    print(f"  Generated {len(df_30m):,} 30m bars")

    # ── Compute 30m indicators ──
    print("  Computing 30m indicators (RSI, MACD, ADX, ATR, Bollinger, ROC, volume)...")
    df_30m_ind = compute_30m_indicators(compute_30m_lower_close(df_30m))
    print(f"  30m indicators computed: {[c for c in df_30m_ind.columns if c not in df_30m.columns]}")

    # ── Compute session levels ──
    print("  Computing session levels (VWAP, OR, ON, prev day, session hi/lo)...")
    session_levels = compute_session_levels(df_5m)
    print(f"  Session levels: {list(session_levels.keys())}")

    # ═══════════════════════════════════════════════════════
    #  PHASE 1: BASELINE
    # ═══════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  PHASE 1: BASELINE (no filters)")
    print(f"{'='*80}")

    baseline_cfg = IndicatorConfig()
    baseline_trades = run_with_indicators(df_5m, df_30m, df_30m_ind, session_levels, baseline_cfg)
    baseline_m = compute_metrics(baseline_trades, INITIAL_CAPITAL)
    baseline_pnls = [t.pnl_dollar for t in baseline_trades]
    baseline_p = ttest_pval(baseline_pnls)

    print(f"\n  BASELINE:")
    print(f"  Trades: {baseline_m.total_trades}  |  WR: {baseline_m.win_rate:.1f}%  |  "
          f"PF: {baseline_m.profit_factor:.2f}  |  P&L: ${baseline_m.net_pnl:,.0f}  |  "
          f"p: {baseline_p:.4f}  |  DD: ${baseline_m.max_drawdown:,.0f}  |  "
          f"Sharpe: {baseline_m.sharpe:.2f}")

    # ═══════════════════════════════════════════════════════
    #  PHASE 2: INDIVIDUAL FILTER TESTING
    # ═══════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  PHASE 2: INDIVIDUAL FILTER TESTING ({len(make_filters())} filters)")
    print(f"{'='*80}")

    all_filters = make_filters()
    results = []

    for fname, fdesc, flist in all_filters:
        cfg = IndicatorConfig(filters=flist, filter_name=fname)
        trades = run_with_indicators(df_5m, df_30m, df_30m_ind, session_levels, cfg)

        if len(trades) < 5:
            results.append({
                "name": fname, "desc": fdesc, "trades": len(trades),
                "wr": 0, "pf": 0, "pnl": 0, "avg": 0, "dd": 0, "sharpe": 0, "p": 1.0,
            })
            continue

        m = compute_metrics(trades, INITIAL_CAPITAL)
        pnls = [t.pnl_dollar for t in trades]
        p = ttest_pval(pnls)

        results.append({
            "name": fname, "desc": fdesc, "trades": m.total_trades,
            "wr": m.win_rate, "pf": m.profit_factor, "pnl": m.net_pnl,
            "avg": m.avg_trade, "dd": m.max_drawdown, "sharpe": m.sharpe, "p": p,
        })

    # Sort by PF descending
    results.sort(key=lambda x: x["pf"], reverse=True)

    print(f"\n  {'Name':<16} {'Desc':<42} {'Trades':>6} {'WR%':>6} {'PF':>6} "
          f"{'Net P&L':>10} {'Avg':>8} {'DD':>8} {'Sharpe':>7} {'p':>7}")
    print(f"  {'─'*130}")

    # Print baseline first
    print(f"  {'BASELINE':<16} {'(no filter)':<42} {baseline_m.total_trades:>6} "
          f"{baseline_m.win_rate:>5.1f}% {baseline_m.profit_factor:>6.2f} "
          f"${baseline_m.net_pnl:>9,.0f} ${baseline_m.avg_trade:>7,.0f} "
          f"${baseline_m.max_drawdown:>7,.0f} {baseline_m.sharpe:>7.2f} {baseline_p:>6.4f}")
    print(f"  {'─'*130}")

    for r in results:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        beat = ">>>" if r["pf"] > baseline_m.profit_factor and r["trades"] >= 100 else ""
        print(f"  {r['name']:<16} {r['desc']:<42} {r['trades']:>6} "
              f"{r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"${r['pnl']:>9,.0f} ${r['avg']:>7,.0f} "
              f"${r['dd']:>7,.0f} {r['sharpe']:>7.2f} {r['p']:>6.4f}{sig} {beat}")

    # ═══════════════════════════════════════════════════════
    #  PHASE 3: SMART EXITS
    # ═══════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  PHASE 3: SMART EXIT MODES")
    print(f"{'='*80}")

    exit_configs = [
        ("ATR_1.5x", "atr_scaled", 1.5),
        ("ATR_2.0x", "atr_scaled", 2.0),
        ("ATR_1.0x", "atr_scaled", 1.0),
        ("PDL_TARGET", "prev_day_low", 0),
        ("ONL_TARGET", "on_low", 0),
        ("RSI_EXIT", "rsi_exit", 0),
        ("VOL_CLIMAX", "vol_climax", 0),
    ]

    exit_results = []
    for ename, emode, emult in exit_configs:
        cfg = IndicatorConfig(smart_exit=emode, atr_exit_mult=emult, filter_name=ename)
        trades = run_with_indicators(df_5m, df_30m, df_30m_ind, session_levels, cfg)

        if len(trades) < 5:
            continue

        m = compute_metrics(trades, INITIAL_CAPITAL)
        pnls = [t.pnl_dollar for t in trades]
        p = ttest_pval(pnls)

        # Exit breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        exit_results.append({
            "name": ename, "mode": emode, "trades": m.total_trades,
            "wr": m.win_rate, "pf": m.profit_factor, "pnl": m.net_pnl,
            "avg": m.avg_trade, "dd": m.max_drawdown, "sharpe": m.sharpe,
            "p": p, "exits": exit_reasons,
        })

    print(f"\n  {'Name':<14} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net P&L':>10} "
          f"{'Avg':>8} {'DD':>8} {'Sharpe':>7} {'p':>7}  Exit Breakdown")
    print(f"  {'─'*120}")
    print(f"  {'BASELINE':<14} {baseline_m.total_trades:>6} "
          f"{baseline_m.win_rate:>5.1f}% {baseline_m.profit_factor:>6.2f} "
          f"${baseline_m.net_pnl:>9,.0f} ${baseline_m.avg_trade:>7,.0f} "
          f"${baseline_m.max_drawdown:>7,.0f} {baseline_m.sharpe:>7.2f} {baseline_p:>6.4f}")
    print(f"  {'─'*120}")

    for r in exit_results:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        ex_str = ", ".join(f"{k}:{v}" for k,v in sorted(r["exits"].items()))
        print(f"  {r['name']:<14} {r['trades']:>6} "
              f"{r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"${r['pnl']:>9,.0f} ${r['avg']:>7,.0f} "
              f"${r['dd']:>7,.0f} {r['sharpe']:>7.2f} {r['p']:>6.4f}{sig}  {ex_str}")

    # ═══════════════════════════════════════════════════════
    #  PHASE 4: TOP FILTER COMBINATIONS
    # ═══════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  PHASE 4: TOP FILTER COMBINATIONS")
    print(f"{'='*80}")

    # Identify top individual filters (beat baseline PF, >= 100 trades, p < 0.10)
    top_individual = [r for r in results
                      if r["pf"] > baseline_m.profit_factor
                      and r["trades"] >= 80
                      and r["p"] < 0.10]

    if len(top_individual) < 2:
        # Relax constraints
        top_individual = [r for r in results
                          if r["pf"] > baseline_m.profit_factor
                          and r["trades"] >= 50]

    top_names = [r["name"] for r in top_individual[:8]]  # Cap at 8 to limit combos
    print(f"\n  Top individual filters for combination: {top_names}")

    # Get filter definitions
    all_filter_map = {f[0]: f for f in make_filters()}

    # Test all pairs
    combo_results = []
    tested = set()

    for i, n1 in enumerate(top_names):
        for j, n2 in enumerate(top_names):
            if i >= j:
                continue
            key = tuple(sorted([n1, n2]))
            if key in tested:
                continue
            tested.add(key)

            f1 = all_filter_map[n1]
            f2 = all_filter_map[n2]
            combined_filters = f1[2] + f2[2]
            combo_name = f"{n1}+{n2}"

            cfg = IndicatorConfig(filters=combined_filters, filter_name=combo_name)
            trades = run_with_indicators(df_5m, df_30m, df_30m_ind, session_levels, cfg)

            if len(trades) < 20:
                continue

            m = compute_metrics(trades, INITIAL_CAPITAL)
            pnls = [t.pnl_dollar for t in trades]
            p = ttest_pval(pnls)

            combo_results.append({
                "name": combo_name, "trades": m.total_trades,
                "wr": m.win_rate, "pf": m.profit_factor, "pnl": m.net_pnl,
                "avg": m.avg_trade, "dd": m.max_drawdown, "sharpe": m.sharpe, "p": p,
            })

    # Test all triples from top 5
    for i, n1 in enumerate(top_names[:5]):
        for j, n2 in enumerate(top_names[:5]):
            for k, n3 in enumerate(top_names[:5]):
                if i >= j or j >= k:
                    continue
                key = tuple(sorted([n1, n2, n3]))
                if key in tested:
                    continue
                tested.add(key)

                f1 = all_filter_map[n1]
                f2 = all_filter_map[n2]
                f3 = all_filter_map[n3]
                combined_filters = f1[2] + f2[2] + f3[2]
                combo_name = f"{n1}+{n2}+{n3}"

                cfg = IndicatorConfig(filters=combined_filters, filter_name=combo_name)
                trades = run_with_indicators(df_5m, df_30m, df_30m_ind, session_levels, cfg)

                if len(trades) < 20:
                    continue

                m = compute_metrics(trades, INITIAL_CAPITAL)
                pnls = [t.pnl_dollar for t in trades]
                p = ttest_pval(pnls)

                combo_results.append({
                    "name": combo_name, "trades": m.total_trades,
                    "wr": m.win_rate, "pf": m.profit_factor, "pnl": m.net_pnl,
                    "avg": m.avg_trade, "dd": m.max_drawdown, "sharpe": m.sharpe, "p": p,
                })

    combo_results.sort(key=lambda x: x["pf"], reverse=True)

    print(f"\n  {'Combination':<40} {'Trades':>6} {'WR%':>6} {'PF':>6} "
          f"{'Net P&L':>10} {'Avg':>8} {'DD':>8} {'Sharpe':>7} {'p':>7}")
    print(f"  {'─'*110}")
    print(f"  {'BASELINE':<40} {baseline_m.total_trades:>6} "
          f"{baseline_m.win_rate:>5.1f}% {baseline_m.profit_factor:>6.2f} "
          f"${baseline_m.net_pnl:>9,.0f} ${baseline_m.avg_trade:>7,.0f} "
          f"${baseline_m.max_drawdown:>7,.0f} {baseline_m.sharpe:>7.2f} {baseline_p:>6.4f}")
    print(f"  {'─'*110}")

    for r in combo_results[:20]:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"  {r['name']:<40} {r['trades']:>6} "
              f"{r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"${r['pnl']:>9,.0f} ${r['avg']:>7,.0f} "
              f"${r['dd']:>7,.0f} {r['sharpe']:>7.2f} {r['p']:>6.4f}{sig}")

    # ═══════════════════════════════════════════════════════
    #  PHASE 5: WALK-FORWARD ON TOP COMBOS
    # ═══════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  PHASE 5: WALK-FORWARD VALIDATION ON TOP COMBOS")
    print(f"{'='*80}")

    # Select top combos for WF (PF > baseline, trades >= 80, p < 0.10)
    wf_candidates = [r for r in combo_results
                     if r["pf"] > baseline_m.profit_factor
                     and r["trades"] >= 80
                     and r["p"] < 0.10]

    if not wf_candidates:
        wf_candidates = [r for r in combo_results
                         if r["pf"] > baseline_m.profit_factor
                         and r["trades"] >= 50][:10]

    # Also add top individual filters
    wf_individual = [r for r in results
                     if r["pf"] > baseline_m.profit_factor
                     and r["trades"] >= 100
                     and r["p"] < 0.05]

    # Combine and deduplicate
    wf_all = []
    seen_names = set()
    for r in wf_individual[:5] + wf_candidates[:10]:
        if r["name"] not in seen_names:
            wf_all.append(r)
            seen_names.add(r["name"])

    # Add baseline WF
    wf_baseline = walk_forward(df_5m, df_30m, df_30m_ind, session_levels,
                               IndicatorConfig())

    print(f"\n  {'BASELINE WF':<40} IS: {wf_baseline['is_trades']}t PF {wf_baseline['is_pf']:.2f} "
          f"${wf_baseline['is_pnl']:,.0f}  |  OOS: {wf_baseline['oos_trades']}t PF {wf_baseline['oos_pf']:.2f} "
          f"${wf_baseline['oos_pnl']:,.0f}  |  PF Ratio: {wf_baseline['pf_ratio']:.2f}")

    print(f"\n  {'Name':<40} {'IS Tr':>5} {'IS PF':>6} {'IS P&L':>9} "
          f"{'OOS Tr':>6} {'OOS PF':>7} {'OOS P&L':>9} {'WF Rat':>7} {'OOS p':>7} {'Result':>8}")
    print(f"  {'─'*120}")

    wf_results = []

    for r in wf_all:
        fname = r["name"]
        # Reconstruct filters
        if "+" in fname:
            parts = fname.split("+")
            combined_filters = []
            for part in parts:
                if part in all_filter_map:
                    combined_filters.extend(all_filter_map[part][2])
            cfg = IndicatorConfig(filters=combined_filters, filter_name=fname)
        elif fname in all_filter_map:
            cfg = IndicatorConfig(filters=all_filter_map[fname][2], filter_name=fname)
        else:
            continue

        wf = walk_forward(df_5m, df_30m, df_30m_ind, session_levels, cfg)

        result = "PASS" if wf["pf_ratio"] > 0.7 and wf["oos_pf"] > 1.0 else "FAIL"
        sig = "*" if wf["oos_p"] < 0.05 else ""

        wf_results.append({
            "name": fname, **wf, "result": result,
            "full_trades": r["trades"], "full_pf": r["pf"], "full_pnl": r["pnl"], "full_p": r["p"],
        })

        print(f"  {fname:<40} {wf['is_trades']:>5} {wf['is_pf']:>6.2f} ${wf['is_pnl']:>8,.0f} "
              f"{wf['oos_trades']:>6} {wf['oos_pf']:>7.2f} ${wf['oos_pnl']:>8,.0f} "
              f"{wf['pf_ratio']:>7.2f} {wf['oos_p']:>6.4f}{sig} {result:>8}")

    # ═══════════════════════════════════════════════════════
    #  PHASE 6: BEST COMBO + BEST EXIT COMBINED
    # ═══════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  PHASE 6: BEST FILTER + BEST EXIT COMBINATION")
    print(f"{'='*80}")

    # Pick best passing WF result
    passing_wf = [r for r in wf_results if r["result"] == "PASS"]
    if not passing_wf:
        passing_wf = sorted(wf_results, key=lambda x: x.get("oos_pf", 0), reverse=True)[:3]
        print(f"\n  (No WF passes — testing top 3 by OOS PF)")

    # Also pick best exit mode
    best_exit = max(exit_results, key=lambda x: x["pf"]) if exit_results else None

    for pwf in passing_wf[:3]:
        fname = pwf["name"]
        print(f"\n  Testing: {fname}")

        # Reconstruct filters
        if "+" in fname:
            parts = fname.split("+")
            combined_filters = []
            for part in parts:
                if part in all_filter_map:
                    combined_filters.extend(all_filter_map[part][2])
        elif fname in all_filter_map:
            combined_filters = all_filter_map[fname][2]
        else:
            continue

        # Test filter alone with detailed metrics
        cfg = IndicatorConfig(filters=combined_filters, filter_name=fname)
        trades = run_with_indicators(df_5m, df_30m, df_30m_ind, session_levels, cfg)
        m = compute_metrics(trades, INITIAL_CAPITAL)
        pnls = [t.pnl_dollar for t in trades]
        p = ttest_pval(pnls)

        print(f"  Filter only:  {m.total_trades} trades, PF {m.profit_factor:.2f}, "
              f"${m.net_pnl:,.0f}, WR {m.win_rate:.1f}%, p={p:.4f}")

        # Exit breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        print(f"  Exits: {exit_reasons}")

        # Monthly breakdown
        monthly = {}
        for t in trades:
            if hasattr(t.entry_time, 'strftime'):
                month = t.entry_time.strftime("%Y-%m")
            else:
                month = str(t.entry_time)[:7]
            monthly.setdefault(month, []).append(t.pnl_dollar)

        print(f"\n  {'Month':<10} {'Trades':>7} {'Net P&L':>10} {'Avg':>8} {'WR%':>6}")
        win_months = 0
        for month in sorted(monthly.keys()):
            mp = monthly[month]
            net = sum(mp)
            avg = np.mean(mp)
            wr = sum(1 for x in mp if x > 0) / len(mp) * 100
            if net > 0:
                win_months += 1
            print(f"  {month:<10} {len(mp):>7} ${net:>9,.0f} ${avg:>7,.0f} {wr:>5.1f}%")
        print(f"  Winning months: {win_months}/{len(monthly)} ({win_months/len(monthly)*100:.0f}%)")

        # Test with best exit if available
        if best_exit and best_exit["pf"] > baseline_m.profit_factor:
            cfg2 = IndicatorConfig(
                filters=combined_filters, filter_name=f"{fname}+{best_exit['name']}",
                smart_exit=best_exit["mode"],
                atr_exit_mult=best_exit.get("mult", 1.5),
            )
            trades2 = run_with_indicators(df_5m, df_30m, df_30m_ind, session_levels, cfg2)
            if len(trades2) > 10:
                m2 = compute_metrics(trades2, INITIAL_CAPITAL)
                pnls2 = [t.pnl_dollar for t in trades2]
                p2 = ttest_pval(pnls2)
                print(f"\n  Filter + {best_exit['name']}: {m2.total_trades} trades, PF {m2.profit_factor:.2f}, "
                      f"${m2.net_pnl:,.0f}, WR {m2.win_rate:.1f}%, p={p2:.4f}")

    # ═══════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  BASELINE:  {baseline_m.total_trades} trades, PF {baseline_m.profit_factor:.2f}, "
          f"${baseline_m.net_pnl:,.0f}, p={baseline_p:.4f}")

    # Best individual filters that beat baseline
    print(f"\n  TOP INDIVIDUAL FILTERS (beat PF {baseline_m.profit_factor:.2f}, >= 100 trades):")
    top_ind = [r for r in results if r["pf"] > baseline_m.profit_factor and r["trades"] >= 100]
    top_ind.sort(key=lambda x: x["pf"], reverse=True)
    for r in top_ind[:10]:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"    {r['name']:<20} {r['trades']:>4}t  PF {r['pf']:.2f}  ${r['pnl']:>9,.0f}  p={r['p']:.4f}{sig}")

    # Best combos that pass WF
    wf_passes = [r for r in wf_results if r["result"] == "PASS"]
    if wf_passes:
        print(f"\n  WALK-FORWARD PASSES:")
        for r in wf_passes:
            print(f"    {r['name']:<40} Full: {r['full_trades']}t PF {r['full_pf']:.2f} "
                  f"${r['full_pnl']:,.0f}  |  OOS: {r['oos_trades']}t PF {r['oos_pf']:.2f} "
                  f"WF Ratio {r['pf_ratio']:.2f}")
    else:
        print(f"\n  No filter combinations passed walk-forward validation.")
        print(f"  Top 3 by OOS PF:")
        top_oos = sorted(wf_results, key=lambda x: x.get("oos_pf", 0), reverse=True)[:3]
        for r in top_oos:
            print(f"    {r['name']:<40} Full: {r['full_trades']}t PF {r['full_pf']:.2f}  "
                  f"|  OOS: {r['oos_trades']}t PF {r['oos_pf']:.2f}  WF {r['pf_ratio']:.2f}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
