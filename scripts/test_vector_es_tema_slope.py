#!/usr/bin/env python3
"""Vector ES — TEMA SLOPE STUDY: Getting ahead of the EMA 9/21 cross.

HYPOTHESIS: The EMA 9/21 bearish cross fires AFTER the move. By the time it
triggers, oscillators have already collapsed. TEMA slope (which leads the cross)
combined with the ON gap-down >= 15pt inventory filter can catch the same moves
5-10 bars earlier.

BASELINE TO BEAT:
  EMA 9/21 bearish cross + ON gap-down >= 15pts + TEMA bearish + 20bps stop
  + time stop 15:55 → 252 trades/2yr, PF 2.674, +$104K, p=0.00004

TIERS:
  1. TEMA slope-based entries (getting ahead of the cross)
  2. Faster cross variants (TEMA 5/13, 7/17, EMA 5/13, HMA, DEMA)
  3. Hybrid signals (fastest + confirmed)

Every variant tested WITH and WITHOUT the ON inventory filter.
Walk-forward split at 2025-02-16. T-test significance.
Entry timing comparison vs EMA 9/21 cross.
"""

import os
import sys
import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics
from backtester.position import Trade
from backtester.indicators import compute_indicators, ema, tema, sma


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "es_5m_databento_2yr.csv",
)
WF_SPLIT = "2025-02-16"
INITIAL_CAPITAL = 100_000.0


@dataclass
class SlopeConfig:
    """Config for TEMA slope variant testing."""
    label: str = ""

    # Signal type
    signal_type: str = "ema_cross"  # See SIGNAL_TYPES dict

    # Signal params (for slope-based)
    slope_threshold: float = 0.0     # tema_slope < threshold to trigger
    require_acceleration: bool = False  # tema_slope < prev tema_slope
    require_both_slopes_neg: bool = False  # both TEMA 9 and 21 slopes negative
    require_tema_below_21: bool = False    # TEMA 9 < TEMA 21 (state)

    # Faster cross params
    fast_period: int = 9
    slow_period: int = 21

    # Hybrid filters (applied ON TOP of signal)
    require_price_below_ema21: bool = False
    require_volume_surge: bool = False    # vol_ratio > 1.5
    require_bar_below_prior: bool = False  # close < prior bar low
    require_below_vwap: bool = False
    require_price_below_tema9: bool = False

    # Core filters
    require_tema_bearish: bool = True
    min_30m_range: float = 8.0

    # ON inventory filter: gap-down >= N pts
    on_gap_down_pts: float = 15.0   # 0 = no ON filter
    use_on_filter: bool = True

    # Exit
    exit_type: str = "time_stop"  # time_stop, tema_slope_exit, on_low_target, prev_day_low_target
    stop_bps: float = 20.0
    entry_start: int = 935
    entry_end: int = 1500
    time_stop_time: int = 1555
    max_trades_per_day: int = 2

    # Instrument
    point_value: float = 50.0
    commission_rt: float = 5.0
    slippage_ticks: int = 1
    tick_size: float = 0.25


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING & ENRICHMENT
# ══════════════════════════════════════════════════════════════════════

def load_data():
    """Load ES 5m data with all indicators + extra columns for this study."""
    print(f"Loading {DATA_PATH}...")
    df = load_tos_csv(DATA_PATH, instrument="ES")
    compute_indicators(df)

    close = df["close"]

    # ── Additional TEMA periods for faster crosses ──
    df["tema_5"] = tema(close, 5)
    df["tema_7"] = tema(close, 7)
    df["tema_13"] = tema(close, 13)
    df["tema_17"] = tema(close, 17)

    # ── DEMA (Double EMA): 2*EMA - EMA(EMA) ──
    for p in [9, 21]:
        e1 = ema(close, p)
        e2 = ema(e1, p)
        df[f"dema_{p}"] = 2 * e1 - e2

    # ── EMA 5/13 for faster EMA cross ──
    df["ema_5"] = ema(close, 5)
    df["ema_13"] = ema(close, 13)

    # ── HMA (Hull Moving Average) ──
    # HMA(n) = WMA(2*WMA(close, n/2) - WMA(close, n), sqrt(n))
    # We'll compute for n=9 and n=21
    for n in [9, 21]:
        half_n = max(int(n / 2), 1)
        sqrt_n = max(int(math.sqrt(n)), 1)
        wma_half = close.rolling(window=half_n, min_periods=half_n).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
        wma_full = close.rolling(window=n, min_periods=n).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
        raw_hull = 2 * wma_half - wma_full
        df[f"hma_{n}"] = raw_hull.rolling(window=sqrt_n, min_periods=sqrt_n).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)

    # ── TEMA slope enrichment ──
    # tema_slope is already computed (tema_fast - tema_fast.shift(3))
    # Add slope of TEMA 21 (slow slope)
    df["tema_slow_slope"] = df["tema_slow"] - df["tema_slow"].shift(3)

    # ── Session VWAP ──
    df["cum_vol"] = 0.0
    df["session_vwap"] = np.nan
    sess = df["session_date"].values
    vol = df["volume"].values
    hlc3 = df["hlc3"].values
    is_rth = df["is_rth"].values
    cum_vol = 0.0
    cum_num = 0.0
    prev_s = None
    vwap_arr = np.full(len(df), np.nan)
    for i in range(len(df)):
        if sess[i] != prev_s:
            cum_vol = 0.0
            cum_num = 0.0
            prev_s = sess[i]
        if is_rth[i]:
            cum_vol += vol[i]
            cum_num += hlc3[i] * vol[i]
            if cum_vol > 0:
                vwap_arr[i] = cum_num / cum_vol
    df["session_vwap"] = vwap_arr
    df["session_vwap"] = df["session_vwap"].ffill()

    # ── ON gap data (per session) ──
    dates = sorted(df[df["is_rth"]]["session_date"].unique())
    on_gap_data = {}  # session_date -> {"gap_pts": float, "on_low": float}
    prev_day_low_map = {}
    prev_day_close_map = {}

    for i, d in enumerate(dates):
        if i == 0:
            continue
        prev_d = dates[i-1]
        prev_rth = df[(df["session_date"] == prev_d) & df["is_rth"]]
        if prev_rth.empty:
            continue
        prev_close = prev_rth.iloc[-1]["close"]
        prev_day_close_map[d] = prev_close
        prev_day_low_map[d] = prev_rth["low"].min()

        # Overnight (globex) bars for this session
        globex = df[(df["session_date"] == d) & ~df["is_rth"]]
        if globex.empty:
            on_gap_data[d] = {"gap_pts": 0.0, "on_low": prev_close, "on_high": prev_close}
            continue

        on_low = globex["low"].min()
        on_high = globex["high"].max()
        gap_pts = prev_close - on_low  # positive = gap down
        on_gap_data[d] = {"gap_pts": gap_pts, "on_low": on_low, "on_high": on_high}

    df["on_gap_pts"] = df["session_date"].map(
        lambda d: on_gap_data.get(d, {}).get("gap_pts", 0.0))
    df["on_low"] = df["session_date"].map(
        lambda d: on_gap_data.get(d, {}).get("on_low", np.nan))
    df["on_high"] = df["session_date"].map(
        lambda d: on_gap_data.get(d, {}).get("on_high", np.nan))
    df["prev_day_close"] = df["session_date"].map(prev_day_close_map)
    df["prev_day_low"] = df["session_date"].map(prev_day_low_map)

    print(f"  {len(df):,} bars | {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    return df


# ══════════════════════════════════════════════════════════════════════
#  30-MINUTE RANGE
# ══════════════════════════════════════════════════════════════════════

def build_30m_bars(df: pd.DataFrame) -> pd.DataFrame:
    rth = df[df["is_rth"]].copy()
    if rth.empty:
        return pd.DataFrame()
    et = rth["et_time"].values
    minutes_from_open = ((et // 100 - 9) * 60 + (et % 100)) - 30
    rth["period_30m"] = minutes_from_open // 30
    bars_30m = rth.groupby(["session_date", "period_30m"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        et_time=("et_time", "last"),
    ).reset_index()
    bars_30m["range"] = bars_30m["high"] - bars_30m["low"]
    return bars_30m


def build_range_lookup(bars_30m: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in bars_30m.iterrows():
        sd = row["session_date"]
        rng = row["range"]
        et = row["et_time"]
        if sd not in lookup:
            lookup[sd] = []
        prev_max = lookup[sd][-1][1] if lookup[sd] else 0.0
        lookup[sd].append((et, max(prev_max, rng)))
    return lookup


def get_max_30m_range(range_lookup: dict, session_date, et_time: int) -> float:
    entries = range_lookup.get(session_date)
    if not entries:
        return 0.0
    max_range = 0.0
    for bar_et, cum_max in entries:
        if bar_et <= et_time:
            max_range = cum_max
        else:
            break
    return max_range


# ══════════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION — ALL VARIANTS
# ══════════════════════════════════════════════════════════════════════

def detect_all_signals(df: pd.DataFrame) -> dict:
    """Pre-compute all signal arrays we need."""
    n = len(df)
    sess = df["session_date"].values
    cl = df["close"].values
    lo = df["low"].values
    signals = {}

    # ── EMA 9/21 cross (BASELINE) ──
    ema_9 = df["ema_9"].values
    ema_21 = df["ema_21"].values
    ema_bear_9_21 = np.zeros(n, dtype=bool)
    ema_bull_9_21 = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if ema_9[i-1] >= ema_21[i-1] and ema_9[i] < ema_21[i]:
            ema_bear_9_21[i] = True
        if ema_9[i-1] <= ema_21[i-1] and ema_9[i] > ema_21[i]:
            ema_bull_9_21[i] = True
    signals["ema_cross_9_21_bear"] = ema_bear_9_21
    signals["ema_cross_9_21_bull"] = ema_bull_9_21

    # ── TEMA slope signals ──
    tema_slope = df["tema_slope"].values
    tema_slow_slope = df["tema_slow_slope"].values

    # 1. TEMA slope turns negative
    slope_neg = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if tema_slope[i] < 0 and tema_slope[i-1] >= 0:
            slope_neg[i] = True
    signals["tema_slope_turns_neg"] = slope_neg

    # 2. TEMA slope acceleration (current slope < previous slope AND slope < 0)
    slope_accel = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if tema_slope[i] < tema_slope[i-1] and tema_slope[i] < 0:
            slope_accel[i] = True
    signals["tema_slope_accel"] = slope_accel

    # 3. TEMA slope < threshold (various thresholds tested via config)
    # We'll check in the backtest loop

    # 4. Both TEMA 9 and 21 slopes negative (first bar where both are negative)
    both_neg = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        cur_both = (tema_slope[i] < 0) and (tema_slow_slope[i] < 0)
        prev_both = (tema_slope[i-1] < 0) and (tema_slow_slope[i-1] < 0)
        if cur_both and not prev_both:
            both_neg[i] = True
    signals["both_slopes_neg"] = both_neg

    # 5. TEMA 9 below TEMA 21 (state) + slope steepening
    tema_fast_v = df["tema_fast"].values
    tema_slow_v = df["tema_slow"].values
    below_and_steepening = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if tema_fast_v[i] < tema_slow_v[i]:
            if tema_slope[i] < tema_slope[i-1]:
                below_and_steepening[i] = True
    signals["below_21_steepening"] = below_and_steepening

    # ── Faster cross variants ──

    # 6. TEMA 5/13 cross
    tema_5 = df["tema_5"].values
    tema_13 = df["tema_13"].values
    tema_5_13_bear = np.zeros(n, dtype=bool)
    tema_5_13_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if not np.isnan(tema_5[i]) and not np.isnan(tema_13[i]):
            if tema_5[i-1] >= tema_13[i-1] and tema_5[i] < tema_13[i]:
                tema_5_13_bear[i] = True
            if tema_5[i-1] <= tema_13[i-1] and tema_5[i] > tema_13[i]:
                tema_5_13_bull[i] = True
    signals["tema_5_13_bear"] = tema_5_13_bear
    signals["tema_5_13_bull"] = tema_5_13_bull

    # 7. TEMA 7/17 cross
    tema_7 = df["tema_7"].values
    tema_17 = df["tema_17"].values
    tema_7_17_bear = np.zeros(n, dtype=bool)
    tema_7_17_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if not np.isnan(tema_7[i]) and not np.isnan(tema_17[i]):
            if tema_7[i-1] >= tema_17[i-1] and tema_7[i] < tema_17[i]:
                tema_7_17_bear[i] = True
            if tema_7[i-1] <= tema_17[i-1] and tema_7[i] > tema_17[i]:
                tema_7_17_bull[i] = True
    signals["tema_7_17_bear"] = tema_7_17_bear
    signals["tema_7_17_bull"] = tema_7_17_bull

    # 8. EMA 5/13 cross
    ema_5 = df["ema_5"].values
    ema_13 = df["ema_13"].values
    ema_5_13_bear = np.zeros(n, dtype=bool)
    ema_5_13_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if ema_5[i-1] >= ema_13[i-1] and ema_5[i] < ema_13[i]:
            ema_5_13_bear[i] = True
        if ema_5[i-1] <= ema_13[i-1] and ema_5[i] > ema_13[i]:
            ema_5_13_bull[i] = True
    signals["ema_5_13_bear"] = ema_5_13_bear
    signals["ema_5_13_bull"] = ema_5_13_bull

    # 9. HMA 9/21 cross
    hma_9 = df["hma_9"].values
    hma_21 = df["hma_21"].values
    hma_bear = np.zeros(n, dtype=bool)
    hma_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if np.isnan(hma_9[i]) or np.isnan(hma_21[i]) or np.isnan(hma_9[i-1]) or np.isnan(hma_21[i-1]):
            continue
        if hma_9[i-1] >= hma_21[i-1] and hma_9[i] < hma_21[i]:
            hma_bear[i] = True
        if hma_9[i-1] <= hma_21[i-1] and hma_9[i] > hma_21[i]:
            hma_bull[i] = True
    signals["hma_9_21_bear"] = hma_bear
    signals["hma_9_21_bull"] = hma_bull

    # 10. DEMA 9/21 cross
    dema_9 = df["dema_9"].values
    dema_21 = df["dema_21"].values
    dema_bear = np.zeros(n, dtype=bool)
    dema_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if np.isnan(dema_9[i]) or np.isnan(dema_21[i]) or np.isnan(dema_9[i-1]) or np.isnan(dema_21[i-1]):
            continue
        if dema_9[i-1] >= dema_21[i-1] and dema_9[i] < dema_21[i]:
            dema_bear[i] = True
        if dema_9[i-1] <= dema_21[i-1] and dema_9[i] > dema_21[i]:
            dema_bull[i] = True
    signals["dema_9_21_bear"] = dema_bear
    signals["dema_9_21_bull"] = dema_bull

    # ── Hybrid signals ──

    # 11. TEMA slope negative + price below EMA 21
    slope_neg_price_below = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if tema_slope[i] < 0 and tema_slope[i-1] >= 0 and cl[i] < ema_21[i]:
            slope_neg_price_below[i] = True
    signals["slope_neg_price_below_ema21"] = slope_neg_price_below

    # 12. TEMA acceleration + volume surge (vol_ratio > 1.5)
    vol_ratio = df["vol_ratio"].values
    accel_vol = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if tema_slope[i] < tema_slope[i-1] and tema_slope[i] < 0 and vol_ratio[i] > 1.5:
            accel_vol[i] = True
    signals["accel_vol_surge"] = accel_vol

    # 13. TEMA slope negative + bar closes below prior bar low
    slope_neg_below_prior = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if tema_slope[i] < 0 and tema_slope[i-1] >= 0 and cl[i] < lo[i-1]:
            slope_neg_below_prior[i] = True
    signals["slope_neg_below_prior_low"] = slope_neg_below_prior

    # 14. TEMA slope steepening + below VWAP
    vwap_arr = df["session_vwap"].values
    slope_steep_below_vwap = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if tema_slope[i] < tema_slope[i-1] and tema_slope[i] < 0:
            if not np.isnan(vwap_arr[i]) and cl[i] < vwap_arr[i]:
                slope_steep_below_vwap[i] = True
    signals["slope_steep_below_vwap"] = slope_steep_below_vwap

    # 15. Price crosses below TEMA 9
    price_below_tema9 = np.zeros(n, dtype=bool)
    price_above_tema9 = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if cl[i-1] >= tema_fast_v[i-1] and cl[i] < tema_fast_v[i]:
            price_below_tema9[i] = True
        if cl[i-1] <= tema_fast_v[i-1] and cl[i] > tema_fast_v[i]:
            price_above_tema9[i] = True
    signals["price_below_tema9"] = price_below_tema9
    signals["price_above_tema9"] = price_above_tema9

    # ── TEMA 9/21 cross (from indicators) ──
    signals["tema_cross_9_21_bear"] = df["tema_cross_down"].values.copy()
    signals["tema_cross_9_21_bull"] = df["tema_cross_up"].values.copy()

    return signals


# ══════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════

# Map signal_type -> (bear_signal_key, bull_signal_key_or_None)
SIGNAL_MAP = {
    "ema_cross":                ("ema_cross_9_21_bear",       "ema_cross_9_21_bull"),
    "tema_slope_turns_neg":     ("tema_slope_turns_neg",      None),
    "tema_slope_accel":         ("tema_slope_accel",          None),
    "tema_slope_threshold":     (None, None),  # handled inline
    "both_slopes_neg":          ("both_slopes_neg",           None),
    "below_21_steepening":      ("below_21_steepening",       None),
    "tema_5_13_cross":          ("tema_5_13_bear",            "tema_5_13_bull"),
    "tema_7_17_cross":          ("tema_7_17_bear",            "tema_7_17_bull"),
    "tema_9_21_cross":          ("tema_cross_9_21_bear",      "tema_cross_9_21_bull"),
    "ema_5_13_cross":           ("ema_5_13_bear",             "ema_5_13_bull"),
    "hma_9_21_cross":           ("hma_9_21_bear",             "hma_9_21_bull"),
    "dema_9_21_cross":          ("dema_9_21_bear",            "dema_9_21_bull"),
    "slope_neg_price_below":    ("slope_neg_price_below_ema21", None),
    "accel_vol_surge":          ("accel_vol_surge",           None),
    "slope_neg_below_prior":    ("slope_neg_below_prior_low", None),
    "slope_steep_below_vwap":   ("slope_steep_below_vwap",    None),
    "price_below_tema9":        ("price_below_tema9",         "price_above_tema9"),
}


def run_backtest(df: pd.DataFrame, cfg: SlopeConfig,
                 range_lookup: dict, signals: dict) -> List[Trade]:
    """Run backtest for a single configuration."""

    # Get signal arrays
    bear_key, bull_key = SIGNAL_MAP.get(cfg.signal_type, (None, None))

    # For slope threshold, we build inline
    if cfg.signal_type == "tema_slope_threshold":
        tema_slope = df["tema_slope"].values
        sess_arr = df["session_date"].values
        n = len(df)
        sig_bear = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if sess_arr[i] != sess_arr[i-1]:
                continue
            if tema_slope[i] < cfg.slope_threshold and tema_slope[i-1] >= cfg.slope_threshold:
                sig_bear[i] = True
        sig_bull = None
    else:
        sig_bear = signals.get(bear_key, np.zeros(len(df), dtype=bool))
        sig_bull = signals.get(bull_key) if bull_key else None

    # Data arrays
    et = df["et_time"].values
    sess = df["session_date"].values
    cl = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    is_rth = df["is_rth"].values
    timestamps = df.index
    n = len(df)

    # Filter arrays
    tema_bearish = df["tema_bearish"].values
    tema_slope_arr = df["tema_slope"].values
    on_gap_pts = df["on_gap_pts"].values
    on_low_arr = df["on_low"].values if "on_low" in df.columns else np.full(n, np.nan)
    prev_day_low_arr = df["prev_day_low"].values if "prev_day_low" in df.columns else np.full(n, np.nan)

    slippage = cfg.slippage_ticks * cfg.tick_size
    warmup = 60

    trades = []
    position = None
    trades_today = 0
    current_session = None

    for i in range(warmup, n):
        s = sess[i]

        # Session reset
        if s != current_session:
            if position is not None:
                exit_price = cl[i-1] + slippage
                pnl_pts = position["entry_price"] - exit_price
                pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
                trades.append(Trade(
                    setup="SLOPE", direction=-1,
                    entry_time=timestamps[position["entry_idx"]],
                    entry_price=position["entry_price"],
                    exit_time=timestamps[i-1], exit_price=exit_price,
                    exit_reason="session_end",
                    pnl_pts=pnl_pts, pnl_dollar=pnl_dollar,
                    stop=position["stop"], target=position.get("target", 0),
                ))
                position = None
            current_session = s
            trades_today = 0

        # ── EXIT CHECKS ──
        if position is not None:
            exit_reason = None
            exit_price = None

            # Stop loss
            if hi[i] >= position["stop"]:
                exit_reason = "stop"
                exit_price = position["stop"] + slippage

            # Target hit
            if exit_reason is None and position["target"] > 0 and lo[i] <= position["target"]:
                exit_reason = "target"
                exit_price = position["target"] - slippage

            # Time stop
            if exit_reason is None and et[i] >= cfg.time_stop_time and is_rth[i]:
                exit_reason = "time_stop"
                exit_price = cl[i]

            # Opposite cross exit (for cross-based signals)
            if exit_reason is None and sig_bull is not None and sig_bull[i]:
                exit_reason = "opposite_cross"
                exit_price = cl[i]

            # TEMA slope exit: slope turns positive
            if exit_reason is None and cfg.exit_type == "tema_slope_exit":
                if tema_slope_arr[i] > 0 and tema_slope_arr[i-1] <= 0:
                    exit_reason = "slope_exit"
                    exit_price = cl[i]

            if exit_reason is not None:
                pnl_pts = position["entry_price"] - exit_price
                pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
                trades.append(Trade(
                    setup="SLOPE", direction=-1,
                    entry_time=timestamps[position["entry_idx"]],
                    entry_price=position["entry_price"],
                    exit_time=timestamps[i], exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pts=pnl_pts, pnl_dollar=pnl_dollar,
                    stop=position["stop"], target=position.get("target", 0),
                ))
                position = None

        # ── ENTRY CHECKS ──
        if position is None and is_rth[i]:
            if et[i] < cfg.entry_start or et[i] >= cfg.entry_end:
                continue
            if trades_today >= cfg.max_trades_per_day:
                continue

            # Signal
            if not sig_bear[i]:
                continue

            # ── FILTERS ──

            # TEMA bearish (tema_fast < tema_slow) — skip for slope-based signals
            # that already embed TEMA state
            if cfg.require_tema_bearish and not tema_bearish[i]:
                continue

            # 30m range
            max_range = get_max_30m_range(range_lookup, s, et[i])
            if max_range < cfg.min_30m_range:
                continue

            # ON inventory filter: gap-down >= N points
            if cfg.use_on_filter and cfg.on_gap_down_pts > 0:
                if on_gap_pts[i] < cfg.on_gap_down_pts:
                    continue

            # ── ENTER SHORT ──
            entry_price = cl[i] - slippage
            stop_pts = entry_price * cfg.stop_bps / 10000.0
            stop_price = entry_price + stop_pts

            # Target based on exit type
            target_price = 0.0
            if cfg.exit_type == "on_low_target":
                on_l = on_low_arr[i]
                if not np.isnan(on_l) and on_l < entry_price:
                    target_price = on_l
            elif cfg.exit_type == "prev_day_low_target":
                pdl = prev_day_low_arr[i]
                if not np.isnan(pdl) and pdl < entry_price:
                    target_price = pdl

            position = {
                "entry_idx": i,
                "entry_price": entry_price,
                "stop": stop_price,
                "target": target_price,
                "session": s,
            }
            trades_today += 1

    # Close remaining
    if position is not None:
        exit_price = cl[-1] + slippage
        pnl_pts = position["entry_price"] - exit_price
        pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
        trades.append(Trade(
            setup="SLOPE", direction=-1,
            entry_time=timestamps[position["entry_idx"]],
            entry_price=position["entry_price"],
            exit_time=timestamps[-1], exit_price=exit_price,
            exit_reason="data_end",
            pnl_pts=pnl_pts, pnl_dollar=pnl_dollar,
            stop=position["stop"], target=position.get("target", 0),
        ))

    return trades


# ══════════════════════════════════════════════════════════════════════
#  STATISTICS
# ══════════════════════════════════════════════════════════════════════

def t_test_pval(trades: List[Trade]) -> float:
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t.pnl_dollar for t in trades])
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    return t_pval / 2 if t_stat > 0 else 1 - t_pval / 2


def walk_forward_metrics(df, cfg, split=WF_SPLIT):
    """Walk-forward: rebuild everything for each sub-period."""
    df_is = df[df.index < split].copy()
    df_oos = df[df.index >= split].copy()

    bars_30m_is = build_30m_bars(df_is)
    rl_is = build_range_lookup(bars_30m_is)
    bars_30m_oos = build_30m_bars(df_oos)
    rl_oos = build_range_lookup(bars_30m_oos)

    sig_is = detect_all_signals(df_is)
    sig_oos = detect_all_signals(df_oos)

    trades_is = run_backtest(df_is, cfg, rl_is, sig_is)
    trades_oos = run_backtest(df_oos, cfg, rl_oos, sig_oos)

    m_is = compute_metrics(trades_is, INITIAL_CAPITAL) if trades_is else None
    m_oos = compute_metrics(trades_oos, INITIAL_CAPITAL) if trades_oos else None

    return m_is, m_oos, trades_is, trades_oos


# ══════════════════════════════════════════════════════════════════════
#  ENTRY TIMING COMPARISON
# ══════════════════════════════════════════════════════════════════════

def compare_entry_timing(df, baseline_trades, variant_trades):
    """Compare entry times: how many bars earlier does variant trigger vs baseline?

    For each day where BOTH strategies trade, find the entry bar index difference.
    Returns: (avg_bars_earlier, median_bars_earlier, n_compared, pct_earlier)
    """
    # Group by session date
    baseline_by_date = {}
    for t in baseline_trades:
        d = t.entry_time.date() if hasattr(t.entry_time, 'date') else None
        if d:
            baseline_by_date.setdefault(d, []).append(t)

    variant_by_date = {}
    for t in variant_trades:
        d = t.entry_time.date() if hasattr(t.entry_time, 'date') else None
        if d:
            variant_by_date.setdefault(d, []).append(t)

    # For overlapping dates, compare first entry time
    bars_earlier = []
    common_dates = set(baseline_by_date.keys()) & set(variant_by_date.keys())

    for d in common_dates:
        b_first = min(baseline_by_date[d], key=lambda t: t.entry_time)
        v_first = min(variant_by_date[d], key=lambda t: t.entry_time)

        # Convert to bar count (5-min bars)
        delta = b_first.entry_time - v_first.entry_time
        delta_bars = delta.total_seconds() / 300.0  # positive = variant is earlier
        bars_earlier.append(delta_bars)

    if not bars_earlier:
        return 0, 0, 0, 0

    arr = np.array(bars_earlier)
    n_earlier = np.sum(arr > 0)
    return (np.mean(arr), np.median(arr), len(arr),
            n_earlier / len(arr) * 100)


# ══════════════════════════════════════════════════════════════════════
#  RESULT FORMATTING
# ══════════════════════════════════════════════════════════════════════

def format_result(label, m, pval=None, wf_ratio=None):
    if m is None or m.total_trades == 0:
        return f"  {label:<42} {'--':>6} trades"
    p_str = f"p={pval:.4f}" if pval is not None else ""
    wf_str = f"WF={wf_ratio:.2f}" if wf_ratio is not None else ""
    flag = ""
    if pval is not None and pval < 0.05 and m.profit_factor >= 2.674:
        flag = " <<< BEATS BASELINE"
    elif pval is not None and pval < 0.05 and m.profit_factor > 2.0:
        flag = " << STRONG"
    elif pval is not None and pval < 0.05:
        flag = " *"
    return (f"  {label:<42} {m.total_trades:>5} trades | WR {m.win_rate:>5.1f}% | "
            f"PF {m.profit_factor:>6.3f} | ${m.net_pnl:>+10,.0f} | "
            f"DD ${m.max_drawdown:>8,.0f} | Sharpe {m.sharpe:>5.2f} | "
            f"{p_str:>12} {wf_str:>8}{flag}")


def run_variant(df, cfg, range_lookup, signals, do_wf=True):
    """Run a single variant: full + walk-forward + t-test."""
    trades = run_backtest(df, cfg, range_lookup, signals)
    m = compute_metrics(trades, INITIAL_CAPITAL) if trades else None
    pval = t_test_pval(trades) if trades else 1.0
    wf_ratio = None

    if do_wf and trades and len(trades) >= 15:
        m_is, m_oos, _, _ = walk_forward_metrics(df, cfg)
        if m_is and m_oos and m_is.profit_factor > 0:
            wf_ratio = m_oos.profit_factor / m_is.profit_factor

    return m, trades, pval, wf_ratio


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    df = load_data()

    # Pre-build shared lookups
    bars_30m = build_30m_bars(df)
    range_lookup = build_range_lookup(bars_30m)
    signals = detect_all_signals(df)

    results_all = []  # (label, m, trades, pval, wf_ratio, cfg)

    print(f"\n{'='*140}")
    print("  VECTOR ES — TEMA SLOPE STUDY: Getting Ahead of the EMA 9/21 Cross")
    print(f"  BASELINE: EMA 9/21 cross + ON gap-down >= 15pts + TEMA bearish + 20bps stop + time 15:55")
    print(f"  TARGET: Beat PF 2.674 / 252 trades / +$104K")
    print(f"{'='*140}")

    # ══════════════════════════════════════════════════════════════
    # BASELINE: EMA 9/21 cross + ON gap-down >= 15pts
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*136}")
    print("  BASELINE")
    print(f"  {'─'*136}")

    baseline_cfg = SlopeConfig(
        label="BASELINE: EMA 9/21 cross + ON gap >= 15",
        signal_type="ema_cross",
        use_on_filter=True, on_gap_down_pts=15.0,
        require_tema_bearish=True, min_30m_range=8.0,
    )
    m_base, tr_base, pval_base, wf_base = run_variant(df, baseline_cfg, range_lookup, signals)
    print(format_result(baseline_cfg.label, m_base, pval_base, wf_base))
    results_all.append((baseline_cfg.label, m_base, tr_base, pval_base, wf_base, baseline_cfg))

    # Also run baseline WITHOUT ON filter for comparison
    no_on_cfg = SlopeConfig(
        label="EMA 9/21 cross (NO ON filter)",
        signal_type="ema_cross",
        use_on_filter=False,
        require_tema_bearish=True, min_30m_range=8.0,
    )
    m_no_on, tr_no_on, pval_no_on, wf_no_on = run_variant(df, no_on_cfg, range_lookup, signals)
    print(format_result(no_on_cfg.label, m_no_on, pval_no_on, wf_no_on))
    results_all.append((no_on_cfg.label, m_no_on, tr_no_on, pval_no_on, wf_no_on, no_on_cfg))

    # ══════════════════════════════════════════════════════════════
    # TIER 1: TEMA SLOPE-BASED ENTRIES
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*136}")
    print("  TIER 1: TEMA SLOPE-BASED ENTRIES (getting ahead of the cross)")
    print(f"  {'─'*136}")

    tier1_variants = []

    # 1. TEMA slope turns negative (first bar where slope < 0)
    for on in [True, False]:
        lbl = f"T1.1 TEMA slope turns neg" + (" + ON" if on else " (no ON)")
        cfg = SlopeConfig(label=lbl, signal_type="tema_slope_turns_neg",
                          use_on_filter=on, require_tema_bearish=False)
        tier1_variants.append((lbl, cfg))

    # 2. TEMA slope acceleration (slope steepening downward)
    for on in [True, False]:
        lbl = f"T1.2 TEMA slope accel" + (" + ON" if on else " (no ON)")
        cfg = SlopeConfig(label=lbl, signal_type="tema_slope_accel",
                          use_on_filter=on, require_tema_bearish=False)
        tier1_variants.append((lbl, cfg))

    # 3. TEMA slope < threshold (various thresholds)
    for thresh in [-0.5, -1.0, -1.5, -2.0, -3.0]:
        for on in [True, False]:
            lbl = f"T1.3 slope < {thresh}" + (" + ON" if on else " (no ON)")
            cfg = SlopeConfig(label=lbl, signal_type="tema_slope_threshold",
                              slope_threshold=thresh,
                              use_on_filter=on, require_tema_bearish=False)
            tier1_variants.append((lbl, cfg))

    # 4. Both TEMA 9 and 21 slopes negative
    for on in [True, False]:
        lbl = f"T1.4 Both slopes neg" + (" + ON" if on else " (no ON)")
        cfg = SlopeConfig(label=lbl, signal_type="both_slopes_neg",
                          use_on_filter=on, require_tema_bearish=False)
        tier1_variants.append((lbl, cfg))

    # 5. TEMA 9 below 21 + slope steepening
    for on in [True, False]:
        lbl = f"T1.5 Below 21 + steepening" + (" + ON" if on else " (no ON)")
        cfg = SlopeConfig(label=lbl, signal_type="below_21_steepening",
                          use_on_filter=on, require_tema_bearish=False)
        tier1_variants.append((lbl, cfg))

    for lbl, cfg in tier1_variants:
        m, tr, pval, wf = run_variant(df, cfg, range_lookup, signals)
        print(format_result(lbl, m, pval, wf))
        results_all.append((lbl, m, tr, pval, wf, cfg))

    # ══════════════════════════════════════════════════════════════
    # TIER 2: FASTER CROSS VARIANTS
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*136}")
    print("  TIER 2: FASTER CROSS VARIANTS")
    print(f"  {'─'*136}")

    tier2_variants = []

    cross_types = [
        ("T2.6 TEMA 5/13 cross",  "tema_5_13_cross"),
        ("T2.7 TEMA 7/17 cross",  "tema_7_17_cross"),
        ("T2.7b TEMA 9/21 cross", "tema_9_21_cross"),
        ("T2.8 EMA 5/13 cross",   "ema_5_13_cross"),
        ("T2.9 HMA 9/21 cross",   "hma_9_21_cross"),
        ("T2.10 DEMA 9/21 cross", "dema_9_21_cross"),
    ]

    for base_lbl, sig_type in cross_types:
        for on in [True, False]:
            lbl = f"{base_lbl}" + (" + ON" if on else " (no ON)")
            cfg = SlopeConfig(label=lbl, signal_type=sig_type,
                              use_on_filter=on, require_tema_bearish=True)
            tier2_variants.append((lbl, cfg))

    for lbl, cfg in tier2_variants:
        m, tr, pval, wf = run_variant(df, cfg, range_lookup, signals)
        print(format_result(lbl, m, pval, wf))
        results_all.append((lbl, m, tr, pval, wf, cfg))

    # ══════════════════════════════════════════════════════════════
    # TIER 3: HYBRID SIGNALS
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*136}")
    print("  TIER 3: HYBRID SIGNALS (fastest trigger + confirmation)")
    print(f"  {'─'*136}")

    tier3_variants = []

    # 11. TEMA slope neg + price below EMA 21
    for on in [True, False]:
        lbl = f"T3.11 Slope neg + price<EMA21" + (" + ON" if on else " (no ON)")
        cfg = SlopeConfig(label=lbl, signal_type="slope_neg_price_below",
                          use_on_filter=on, require_tema_bearish=False)
        tier3_variants.append((lbl, cfg))

    # 12. Acceleration + volume surge
    for on in [True, False]:
        lbl = f"T3.12 Accel + vol surge" + (" + ON" if on else " (no ON)")
        cfg = SlopeConfig(label=lbl, signal_type="accel_vol_surge",
                          use_on_filter=on, require_tema_bearish=False)
        tier3_variants.append((lbl, cfg))

    # 13. Slope neg + below prior bar low
    for on in [True, False]:
        lbl = f"T3.13 Slope neg + bar<prior low" + (" + ON" if on else " (no ON)")
        cfg = SlopeConfig(label=lbl, signal_type="slope_neg_below_prior",
                          use_on_filter=on, require_tema_bearish=False)
        tier3_variants.append((lbl, cfg))

    # 14. Slope steepening + below VWAP
    for on in [True, False]:
        lbl = f"T3.14 Slope steep + <VWAP" + (" + ON" if on else " (no ON)")
        cfg = SlopeConfig(label=lbl, signal_type="slope_steep_below_vwap",
                          use_on_filter=on, require_tema_bearish=False)
        tier3_variants.append((lbl, cfg))

    # 15. Price crosses below TEMA 9
    for on in [True, False]:
        lbl = f"T3.15 Price < TEMA 9" + (" + ON" if on else " (no ON)")
        cfg = SlopeConfig(label=lbl, signal_type="price_below_tema9",
                          use_on_filter=on, require_tema_bearish=False)
        tier3_variants.append((lbl, cfg))

    for lbl, cfg in tier3_variants:
        m, tr, pval, wf = run_variant(df, cfg, range_lookup, signals)
        print(format_result(lbl, m, pval, wf))
        results_all.append((lbl, m, tr, pval, wf, cfg))

    # ══════════════════════════════════════════════════════════════
    # TIER 4: EXIT VARIANTS ON BEST SIGNALS
    # Test alternative exits on the most promising entry signals
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*136}")
    print("  TIER 4: EXIT VARIANTS (best signals from above + alternative exits)")
    print(f"  {'─'*136}")

    # We'll test exit variants on baseline + a few promising slope signals
    exit_signal_types = [
        ("EMA 9/21 cross", "ema_cross"),
        ("TEMA slope turns neg", "tema_slope_turns_neg"),
        ("Both slopes neg", "both_slopes_neg"),
        ("Price < TEMA 9", "price_below_tema9"),
        ("TEMA 9/21 cross", "tema_9_21_cross"),
    ]

    exit_types = [
        ("tema_slope_exit", "TEMA slope exit"),
        ("on_low_target", "ON low target"),
        ("prev_day_low_target", "Prev day low target"),
    ]

    for sig_lbl, sig_type in exit_signal_types:
        for exit_type, exit_lbl in exit_types:
            lbl = f"T4 {sig_lbl} + {exit_lbl} + ON"
            tema_bear = sig_type not in ["tema_slope_turns_neg", "both_slopes_neg"]
            cfg = SlopeConfig(label=lbl, signal_type=sig_type,
                              exit_type=exit_type, use_on_filter=True,
                              require_tema_bearish=tema_bear)
            m, tr, pval, wf = run_variant(df, cfg, range_lookup, signals, do_wf=True)
            print(format_result(lbl, m, pval, wf))
            results_all.append((lbl, m, tr, pval, wf, cfg))

    # ══════════════════════════════════════════════════════════════
    # TIER 5: PARAMETER VARIANTS ON BEST SLOPE SIGNALS
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*136}")
    print("  TIER 5: TEMA FILTER / RANGE ABLATION ON BEST SIGNALS + ON FILTER")
    print(f"  {'─'*136}")

    # Test with TEMA bearish filter ON vs OFF, range variants
    ablation_signals = [
        ("EMA 9/21 cross", "ema_cross", True),
        ("TEMA slope turns neg", "tema_slope_turns_neg", False),
        ("Both slopes neg", "both_slopes_neg", False),
        ("Price < TEMA 9", "price_below_tema9", False),
    ]

    for sig_lbl, sig_type, default_tema in ablation_signals:
        # TEMA bearish toggle
        for tema_on in [True, False]:
            lbl = f"T5 {sig_lbl} + ON + TEMA={'ON' if tema_on else 'OFF'}"
            cfg = SlopeConfig(label=lbl, signal_type=sig_type,
                              use_on_filter=True, require_tema_bearish=tema_on)
            m, tr, pval, wf = run_variant(df, cfg, range_lookup, signals, do_wf=False)
            print(format_result(lbl, m, pval, wf))
            results_all.append((lbl, m, tr, pval, wf, cfg))

        # Range variants
        for rng in [0, 5, 10, 12, 15]:
            lbl = f"T5 {sig_lbl} + ON + range>={rng}"
            cfg = SlopeConfig(label=lbl, signal_type=sig_type,
                              use_on_filter=True, require_tema_bearish=default_tema,
                              min_30m_range=rng)
            m, tr, pval, wf = run_variant(df, cfg, range_lookup, signals, do_wf=False)
            print(format_result(lbl, m, pval, wf))
            results_all.append((lbl, m, tr, pval, wf, cfg))

    # ══════════════════════════════════════════════════════════════
    # TIER 6: ON GAP-DOWN THRESHOLD SWEEP
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*136}")
    print("  TIER 6: ON GAP-DOWN THRESHOLD SWEEP (how much gap-down is optimal?)")
    print(f"  {'─'*136}")

    for sig_lbl, sig_type in [("EMA 9/21 cross", "ema_cross"),
                               ("TEMA slope turns neg", "tema_slope_turns_neg"),
                               ("Price < TEMA 9", "price_below_tema9")]:
        tema_bear = sig_type not in ["tema_slope_turns_neg"]
        for gap_pts in [0, 5, 10, 15, 20, 25, 30]:
            lbl = f"T6 {sig_lbl} + gap>={gap_pts}pt"
            use_on = gap_pts > 0
            cfg = SlopeConfig(label=lbl, signal_type=sig_type,
                              use_on_filter=use_on, on_gap_down_pts=gap_pts,
                              require_tema_bearish=tema_bear)
            m, tr, pval, wf = run_variant(df, cfg, range_lookup, signals, do_wf=False)
            print(format_result(lbl, m, pval, wf))
            results_all.append((lbl, m, tr, pval, wf, cfg))

    # ══════════════════════════════════════════════════════════════
    # RANKING
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("  TOP 20 BY PROFIT FACTOR (min 20 trades, p < 0.10)")
    print(f"{'='*140}")

    valid = [(lbl, m, tr, pval, wf, cfg)
             for lbl, m, tr, pval, wf, cfg in results_all
             if m is not None and m.total_trades >= 20 and pval is not None and pval < 0.10]
    valid.sort(key=lambda x: x[1].profit_factor, reverse=True)

    for i, (lbl, m, tr, pval, wf, cfg) in enumerate(valid[:20]):
        beats = "BEATS" if m.profit_factor >= 2.674 else ""
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        wf_str = f"WF={wf:.2f}" if wf is not None else "WF=n/a"
        wf_pass = "PASS" if wf is not None and wf >= 0.70 else "FAIL" if wf is not None else ""
        print(f"  #{i+1:>2} {lbl:<44} PF {m.profit_factor:>6.3f} | "
              f"{m.total_trades:>4} trades | ${m.net_pnl:>+10,.0f} | "
              f"p={pval:.4f}{sig:>3} | {wf_str} {wf_pass:>4} {beats}")

    # ══════════════════════════════════════════════════════════════
    # FULL VALIDATION ON TOP CANDIDATES (p < 0.05 + WF)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("  FULL VALIDATION — CANDIDATES WITH p < 0.05 + PF > 2.0")
    print(f"{'='*140}")

    # Re-run WF on top candidates that don't have it yet
    candidates = [(lbl, m, tr, pval, wf, cfg)
                  for lbl, m, tr, pval, wf, cfg in results_all
                  if m is not None and m.profit_factor > 2.0 and m.total_trades >= 15
                  and pval is not None and pval < 0.05]
    candidates.sort(key=lambda x: x[1].profit_factor, reverse=True)

    for lbl, m, tr, pval, wf, cfg in candidates[:10]:
        # Re-run with walk-forward if missing
        if wf is None:
            m_is, m_oos, _, _ = walk_forward_metrics(df, cfg)
            if m_is and m_oos and m_is.profit_factor > 0:
                wf = m_oos.profit_factor / m_is.profit_factor

        print(f"\n  --- {lbl} ---")
        print(f"    Trades: {m.total_trades} | WR: {m.win_rate:.1f}% | PF: {m.profit_factor:.3f}")
        print(f"    P&L: ${m.net_pnl:,.0f} | DD: ${m.max_drawdown:,.0f} | Sharpe: {m.sharpe:.2f}")
        print(f"    p-value: {pval:.6f}")
        if wf is not None:
            wf_pass = "PASS" if wf >= 0.70 else "FAIL"
            print(f"    WF ratio: {wf:.3f} ({wf_pass})")
        else:
            print(f"    WF ratio: n/a (too few trades)")

        # Exit reason breakdown
        reasons = {}
        for t in tr:
            reasons.setdefault(t.exit_reason, {"count": 0, "pnl": 0.0, "wins": 0})
            reasons[t.exit_reason]["count"] += 1
            reasons[t.exit_reason]["pnl"] += t.pnl_dollar
            if t.pnl_dollar > 0:
                reasons[t.exit_reason]["wins"] += 1
        print(f"    Exit reasons:")
        for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["pnl"]):
            wr = data["wins"] / data["count"] * 100 if data["count"] else 0
            print(f"      {reason:<18} {data['count']:>4} | WR {wr:>5.1f}% | ${data['pnl']:>+9,.0f}")

    # ══════════════════════════════════════════════════════════════
    # ENTRY TIMING COMPARISON
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("  ENTRY TIMING COMPARISON vs EMA 9/21 CROSS BASELINE")
    print(f"  (positive = variant enters EARLIER than baseline)")
    print(f"{'='*140}")

    # Compare every valid variant against baseline
    timing_results = []
    for lbl, m, tr, pval, wf, cfg in results_all:
        if m is None or m.total_trades < 10 or pval is None or pval >= 0.10:
            continue
        if lbl == baseline_cfg.label:
            continue

        avg_earlier, med_earlier, n_comp, pct_earlier = compare_entry_timing(
            df, tr_base, tr)

        if n_comp >= 5:
            timing_results.append((lbl, m, pval, avg_earlier, med_earlier, n_comp, pct_earlier))

    timing_results.sort(key=lambda x: x[3], reverse=True)  # sort by avg bars earlier

    print(f"\n  {'Variant':<44} {'Avg Bars':>10} {'Med Bars':>10} {'N Days':>8} {'% Earlier':>10} {'PF':>8} {'p':>10}")
    print(f"  {'-'*104}")
    for lbl, m, pval, avg_e, med_e, n_comp, pct_e in timing_results[:25]:
        print(f"  {lbl:<44} {avg_e:>+10.1f} {med_e:>+10.1f} {n_comp:>8} {pct_e:>9.1f}% "
              f"{m.profit_factor:>8.3f} p={pval:.4f}")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("  FINAL SUMMARY")
    print(f"{'='*140}")

    # Best by PF with p < 0.05 and WF >= 0.70
    gold = [(lbl, m, tr, pval, wf, cfg)
            for lbl, m, tr, pval, wf, cfg in results_all
            if m is not None and pval is not None and pval < 0.05
            and wf is not None and wf >= 0.70 and m.total_trades >= 15]
    gold.sort(key=lambda x: x[1].profit_factor, reverse=True)

    if gold:
        print(f"\n  GOLD TIER (p < 0.05, WF >= 0.70):")
        for lbl, m, tr, pval, wf, cfg in gold[:10]:
            beats = ">>> BEATS BASELINE" if m.profit_factor >= 2.674 else ""
            print(f"    {lbl:<44} PF {m.profit_factor:.3f} | {m.total_trades} trades | "
                  f"${m.net_pnl:>+10,.0f} | p={pval:.4f} | WF={wf:.2f} {beats}")
    else:
        print(f"\n  NO variants passed the Gold Tier gate (p < 0.05 + WF >= 0.70)")

    # Best by PF with p < 0.05 only
    silver = [(lbl, m, tr, pval, wf, cfg)
              for lbl, m, tr, pval, wf, cfg in results_all
              if m is not None and pval is not None and pval < 0.05
              and m.total_trades >= 15]
    silver.sort(key=lambda x: x[1].profit_factor, reverse=True)

    if silver:
        print(f"\n  SILVER TIER (p < 0.05, any WF):")
        for lbl, m, tr, pval, wf, cfg in silver[:10]:
            wf_str = f"WF={wf:.2f}" if wf is not None else "WF=n/a"
            beats = ">>> BEATS BASELINE" if m.profit_factor >= 2.674 else ""
            print(f"    {lbl:<44} PF {m.profit_factor:.3f} | {m.total_trades} trades | "
                  f"${m.net_pnl:>+10,.0f} | p={pval:.4f} | {wf_str} {beats}")

    # Best by trade count with decent PF (more trades at similar PF = better)
    volume = [(lbl, m, tr, pval, wf, cfg)
              for lbl, m, tr, pval, wf, cfg in results_all
              if m is not None and pval is not None and pval < 0.05
              and m.profit_factor >= 1.5 and m.total_trades >= 30]
    volume.sort(key=lambda x: x[1].total_trades, reverse=True)

    if volume:
        print(f"\n  VOLUME TIER (p < 0.05, PF >= 1.5, 30+ trades — more trades at good PF):")
        for lbl, m, tr, pval, wf, cfg in volume[:10]:
            wf_str = f"WF={wf:.2f}" if wf is not None else "WF=n/a"
            print(f"    {lbl:<44} {m.total_trades} trades | PF {m.profit_factor:.3f} | "
                  f"${m.net_pnl:>+10,.0f} | p={pval:.4f} | {wf_str}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s | {len(results_all)} variants tested")
    print(f"  Data: {df.index[0].date()} to {df.index[-1].date()} | WF split: {WF_SPLIT}")


if __name__ == "__main__":
    main()
