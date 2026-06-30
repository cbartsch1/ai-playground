#!/usr/bin/env python3
"""Trend Continuation ES v2 — Structural Improvement Search.

Baseline: 30m close < prior 30m LOW, short-only, 30bps stop, 15pt fixed target,
          max hold 6 bars (30min on 5m), 1 trade/day, 935-1300.
          374 trades, PF 1.35, +$25K, Sharpe 2.00, p=0.0077, WF ratio 1.24.

This script tests STRUCTURAL variants — not parameter tweaks:
  1. Signal enhancements (double lower close, TEMA confirm, volume surge, range expansion, VWAP, conviction pts)
  2. Exit improvements (ATR-scaled, support targets, trailing ATR, time-decayed, opposite signal, stagger, expanding)
  3. Filters (gap-up skip, downtrend context, skip Friday, high-vol regime, time bucketing, max trades)
  4. Best-of combinations

Usage:
    python3 scripts/test_trend_cont_es_v2.py
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

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


# ── 30m Aggregation ──

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

    # Compute 30m bar ranges for range-expansion filter
    bars_30m["bar_range"] = bars_30m["high"] - bars_30m["low"]
    bars_30m["avg_range"] = bars_30m["bar_range"].rolling(20, min_periods=5).mean()

    return bars_30m


def compute_30m_signals(df_30m: pd.DataFrame) -> pd.DataFrame:
    """Compute all 30m-level signals needed by all variants."""
    df = df_30m.copy()
    c = df["close"].values
    lo = df["low"].values
    hi = df["high"].values
    s = df["session_date"].values
    vol = df["volume"].values
    n = len(df)

    # Basic: close < prior low
    lower_close = np.zeros(n, dtype=bool)
    # Double: 2 consecutive bars each close < their prior bar's low
    double_lower = np.zeros(n, dtype=bool)
    # Conviction: close below prior low by N pts
    conviction_pts = np.zeros(n)

    for i in range(1, n):
        if s[i] == s[i-1] and c[i] < lo[i-1]:
            lower_close[i] = True
            conviction_pts[i] = lo[i-1] - c[i]

    for i in range(2, n):
        if s[i] == s[i-1] == s[i-2] and lower_close[i] and lower_close[i-1]:
            double_lower[i] = True

    df["lower_close"] = lower_close
    df["double_lower"] = double_lower
    df["conviction_pts"] = conviction_pts

    # Range expansion: bar range > avg range
    df["range_expanded"] = df["bar_range"] > df["avg_range"]

    # Volume surge: volume > 1.2x rolling average
    df["vol_avg"] = df["volume"].rolling(20, min_periods=5).mean()
    df["vol_surge"] = df["volume"] > (df["vol_avg"] * 1.2)

    # Opposite signal: close > prior high (for exit)
    higher_close = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if s[i] == s[i-1] and c[i] > hi[i-1]:
            higher_close[i] = True
    df["higher_close"] = higher_close

    return df


def compute_atr(series_h, series_l, series_c, period=14):
    """Compute ATR from arrays."""
    n = len(series_h)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(series_h[i] - series_l[i],
                     abs(series_h[i] - series_c[i-1]),
                     abs(series_l[i] - series_c[i-1]))
    tr[0] = series_h[0] - series_l[0]

    atr = np.zeros(n)
    if period <= n:
        atr[:period] = np.mean(tr[:period])
    else:
        atr[:] = tr[0]
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr


def compute_session_vwap(df_5m: pd.DataFrame) -> np.ndarray:
    """Compute running VWAP per RTH session."""
    n = len(df_5m)
    vwap = np.full(n, np.nan)
    cum_pv = 0.0
    cum_v = 0.0
    prev_sess = None

    sess = df_5m["session_date"].values
    hlc3 = ((df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0).values
    vol = df_5m["volume"].values
    is_rth = df_5m["is_rth"].values

    for i in range(n):
        if not is_rth[i]:
            continue
        if sess[i] != prev_sess:
            cum_pv = 0.0
            cum_v = 0.0
            prev_sess = sess[i]
        v = max(vol[i], 1)
        cum_pv += hlc3[i] * v
        cum_v += v
        vwap[i] = cum_pv / cum_v

    # Forward fill for non-RTH bars
    last = np.nan
    for i in range(n):
        if np.isnan(vwap[i]):
            vwap[i] = last
        else:
            last = vwap[i]
    return vwap


def compute_prev_day_levels(df_5m: pd.DataFrame):
    """Compute previous day high/low/close for each bar."""
    n = len(df_5m)
    prev_high = np.full(n, np.nan)
    prev_low = np.full(n, np.nan)
    prev_close_arr = np.full(n, np.nan)
    on_low = np.full(n, np.nan)

    sess = df_5m["session_date"].values
    hi = df_5m["high"].values
    lo = df_5m["low"].values
    cl = df_5m["close"].values
    is_rth = df_5m["is_rth"].values
    new_rth = df_5m["new_rth"].values

    # Track daily stats
    day_high = np.nan
    day_low = np.nan
    day_close = np.nan
    p_high = np.nan
    p_low = np.nan
    p_close = np.nan
    current_on_low = np.nan
    frozen_on_low = np.nan
    prev_sess = None

    for i in range(n):
        if new_rth[i]:
            # Store yesterday
            if not np.isnan(day_high):
                p_high = day_high
                p_low = day_low
                p_close = day_close
            # Freeze ON low
            frozen_on_low = current_on_low
            # Reset for today
            day_high = hi[i]
            day_low = lo[i]
            day_close = cl[i]
            current_on_low = np.nan
        elif is_rth[i]:
            if np.isnan(day_high):
                day_high = hi[i]
                day_low = lo[i]
            else:
                day_high = max(day_high, hi[i])
                day_low = min(day_low, lo[i])
            day_close = cl[i]
        else:
            # Globex / overnight
            if np.isnan(current_on_low):
                current_on_low = lo[i]
            else:
                current_on_low = min(current_on_low, lo[i])

        prev_high[i] = p_high
        prev_low[i] = p_low
        prev_close_arr[i] = p_close
        on_low[i] = frozen_on_low

    return prev_high, prev_low, prev_close_arr, on_low


# ── Strategy Config ──

@dataclass
class V2Config:
    """Configuration for a single variant test."""
    name: str = "baseline"

    # Entry window
    entry_start: int = 935
    entry_end: int = 1300
    max_hold_bars: int = 6
    stop_bps: float = 30.0
    max_trades_day: int = 1
    min_gap_bars: int = 6
    flatten_time: int = FLATTEN_TIME

    # Signal variants
    signal_mode: str = "basic"   # basic, double_lower, volume_surge, range_expand, conviction
    conviction_min_pts: float = 2.0  # for conviction mode

    # Signal filters
    use_tema_confirm: bool = False    # TEMA 9 < TEMA 21
    use_below_vwap: bool = False      # price below session VWAP
    use_trend_context: bool = False   # close < TEMA 55

    # Exit mode
    exit_mode: str = "fixed_target"
    target_pts: float = 15.0

    # ATR-scaled exit
    atr_target_mult: float = 1.5     # for atr_target mode

    # Trailing stop
    trail_trigger_pts: float = 8.0
    trail_atr_mult: float = 1.0

    # Time-decayed target
    am_target_pts: float = 20.0
    pm_target_pts: float = 10.0

    # Stagger exit
    stagger_t1_pts: float = 10.0
    stagger_t2_pts: float = 25.0

    # Expanding target
    expand_fast_pts: float = 12.0   # if exit < 3 bars
    expand_slow_pts: float = 20.0   # if holding longer

    # Filters
    skip_gap_up: bool = False
    gap_up_threshold: float = 10.0
    skip_friday: bool = False
    require_high_vol: bool = False   # vol_ratio > 1.0
    time_bucket: str = "full"        # full, morning (935-1100), midday (1100-1300)


# ── Strategy Engine ──

def run_variant(df_5m: pd.DataFrame, df_30m_sig: pd.DataFrame,
                cfg: V2Config,
                vwap: np.ndarray,
                tema_fast: np.ndarray, tema_slow: np.ndarray, tema_trend: np.ndarray,
                atr_5m: np.ndarray,
                prev_day_high: np.ndarray, prev_day_low: np.ndarray,
                prev_day_close: np.ndarray, on_low_arr: np.ndarray,
                vol_ratio_5m: np.ndarray) -> List[Trade]:
    """Run a single variant backtest. Returns list of trades."""

    rth_5m = df_5m[df_5m["is_rth"]].copy()
    if rth_5m.empty or df_30m_sig.empty:
        return []

    # Build 30m signal lookup — shift forward to avoid lookahead
    cols_needed = ["lower_close", "double_lower", "vol_surge", "range_expanded",
                   "conviction_pts", "higher_close", "session_date"]
    signal_df = df_30m_sig[cols_needed].copy()
    signal_df.index = signal_df.index + pd.Timedelta(minutes=30)

    merged = pd.merge_asof(
        rth_5m.reset_index(),
        signal_df.reset_index(),
        left_on="datetime",
        right_on=signal_df.index.name or "datetime",
        direction="backward",
        suffixes=("", "_m30"),
    ).set_index("datetime")

    for col in ["lower_close", "double_lower", "vol_surge", "range_expanded", "higher_close"]:
        merged[col] = merged[col].fillna(False)
    merged["conviction_pts"] = merged["conviction_pts"].fillna(0.0)

    # Map precomputed arrays to merged index positions
    # We need to map from df_5m index to merged index
    rth_mask = df_5m["is_rth"].values
    rth_indices = np.where(rth_mask)[0]

    n = len(merged)
    et = merged["et_time"].values
    sess = merged["session_date"].values
    cl = merged["close"].values
    op = merged["open"].values
    hi = merged["high"].values
    lo = merged["low"].values
    times = merged.index

    sig_lower = merged["lower_close"].values
    sig_double = merged["double_lower"].values
    sig_vol_surge = merged["vol_surge"].values
    sig_range_exp = merged["range_expanded"].values
    sig_conviction = merged["conviction_pts"].values
    sig_higher_close = merged["higher_close"].values

    # Map indicator arrays to RTH-only bars
    vwap_rth = vwap[rth_indices[:n]] if len(rth_indices) >= n else np.full(n, np.nan)
    tema_f_rth = tema_fast[rth_indices[:n]] if len(rth_indices) >= n else np.full(n, np.nan)
    tema_s_rth = tema_slow[rth_indices[:n]] if len(rth_indices) >= n else np.full(n, np.nan)
    tema_t_rth = tema_trend[rth_indices[:n]] if len(rth_indices) >= n else np.full(n, np.nan)
    atr_rth = atr_5m[rth_indices[:n]] if len(rth_indices) >= n else np.full(n, 15.0)
    pdh_rth = prev_day_high[rth_indices[:n]] if len(rth_indices) >= n else np.full(n, np.nan)
    pdl_rth = prev_day_low[rth_indices[:n]] if len(rth_indices) >= n else np.full(n, np.nan)
    pdc_rth = prev_day_close[rth_indices[:n]] if len(rth_indices) >= n else np.full(n, np.nan)
    onl_rth = on_low_arr[rth_indices[:n]] if len(rth_indices) >= n else np.full(n, np.nan)
    vr_rth = vol_ratio_5m[rth_indices[:n]] if len(rth_indices) >= n else np.ones(n)
    weekday_rth = merged.index.weekday.values  # 0=Mon, 4=Fri

    # Determine entry window based on time_bucket
    entry_start = cfg.entry_start
    entry_end = cfg.entry_end
    if cfg.time_bucket == "morning":
        entry_start = 935
        entry_end = 1100
    elif cfg.time_bucket == "midday":
        entry_start = 1100
        entry_end = 1300

    # Bar-by-bar simulation
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    entry_idx = 0
    stop_price = 0.0
    target_price = 0.0
    best_price = 0.0
    trail_active = False
    # For stagger: track partial fills
    stagger_t1_hit = False
    # For opposite signal exit
    last_higher_close = False

    trade_count = {}
    last_entry_bar = {}

    def make_trade(exit_time, exit_price, exit_reason, mult=1.0):
        pnl_pts = entry_price - exit_price  # short
        pnl_dollar = (pnl_pts * ES_POINT_VALUE * mult - COMMISSION_RT * mult) * CONTRACTS \
                     - SLIPPAGE_PTS * ES_POINT_VALUE * CONTRACTS * mult
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
            bars_held = i - entry_idx

            # Session changed
            if s != sess[entry_idx]:
                trades.append(make_trade(times[i-1] if i > 0 else times[i],
                                         cl[i-1] if i > 0 else cl[i], "session_end"))
                in_position = False

            # Stop hit (high >= stop for short)
            elif hi[i] >= stop_price:
                trades.append(make_trade(times[i], stop_price, "stop"))
                in_position = False
                continue

            # Flatten time
            elif t >= cfg.flatten_time:
                trades.append(make_trade(times[i], cl[i], "flatten"))
                in_position = False
                continue

            # ── Exit mode logic ──
            elif cfg.exit_mode == "fixed_target":
                if target_price > 0 and lo[i] <= target_price:
                    trades.append(make_trade(times[i], target_price, "target"))
                    in_position = False
                    continue

            elif cfg.exit_mode == "atr_target":
                if target_price > 0 and lo[i] <= target_price:
                    trades.append(make_trade(times[i], target_price, "atr_target"))
                    in_position = False
                    continue

            elif cfg.exit_mode == "support_target":
                if target_price > 0 and lo[i] <= target_price:
                    trades.append(make_trade(times[i], target_price, "support_target"))
                    in_position = False
                    continue

            elif cfg.exit_mode == "trail_atr":
                # Fixed target as ceiling
                if target_price > 0 and lo[i] <= target_price:
                    trades.append(make_trade(times[i], target_price, "target"))
                    in_position = False
                    continue
                # Trailing logic
                best_price = min(best_price, lo[i])
                profit = entry_price - best_price
                if profit >= cfg.trail_trigger_pts:
                    trail_active = True
                if trail_active:
                    trail_dist = atr_rth[i] * cfg.trail_atr_mult if not np.isnan(atr_rth[i]) else 10.0
                    trail_level = best_price + trail_dist
                    if hi[i] >= trail_level:
                        trades.append(make_trade(times[i], trail_level, "trail"))
                        in_position = False
                        continue

            elif cfg.exit_mode == "time_decayed":
                if target_price > 0 and lo[i] <= target_price:
                    trades.append(make_trade(times[i], target_price, "time_target"))
                    in_position = False
                    continue

            elif cfg.exit_mode == "opposite_signal":
                # Exit on 30m close above prior 30m high (trend reversal)
                if sig_higher_close[i]:
                    trades.append(make_trade(times[i], cl[i], "opp_signal"))
                    in_position = False
                    continue
                # Still need a profit target as ceiling
                if target_price > 0 and lo[i] <= target_price:
                    trades.append(make_trade(times[i], target_price, "target"))
                    in_position = False
                    continue

            elif cfg.exit_mode == "expanding":
                if bars_held < 3:
                    tgt = entry_price - cfg.expand_fast_pts
                else:
                    tgt = entry_price - cfg.expand_slow_pts
                if lo[i] <= tgt:
                    trades.append(make_trade(times[i], tgt, "expanding_target"))
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

        if t < entry_start or t >= entry_end:
            continue
        if trade_count.get(s, 0) >= cfg.max_trades_day:
            continue
        if s in last_entry_bar and (i - last_entry_bar[s]) < cfg.min_gap_bars:
            continue

        # ── Signal check ──
        signal_ok = False
        if cfg.signal_mode == "basic":
            signal_ok = bool(sig_lower[i])
        elif cfg.signal_mode == "double_lower":
            signal_ok = bool(sig_double[i])
        elif cfg.signal_mode == "volume_surge":
            signal_ok = bool(sig_lower[i]) and bool(sig_vol_surge[i])
        elif cfg.signal_mode == "range_expand":
            signal_ok = bool(sig_lower[i]) and bool(sig_range_exp[i])
        elif cfg.signal_mode == "conviction":
            signal_ok = bool(sig_lower[i]) and sig_conviction[i] >= cfg.conviction_min_pts

        if not signal_ok:
            continue

        # ── Signal confirmation filters ──
        if cfg.use_tema_confirm:
            if np.isnan(tema_f_rth[i]) or np.isnan(tema_s_rth[i]):
                continue
            if tema_f_rth[i] >= tema_s_rth[i]:
                continue

        if cfg.use_below_vwap:
            if np.isnan(vwap_rth[i]):
                continue
            if cl[i] >= vwap_rth[i]:
                continue

        if cfg.use_trend_context:
            if np.isnan(tema_t_rth[i]):
                continue
            if cl[i] >= tema_t_rth[i]:
                continue

        # ── Filters ──
        if cfg.skip_friday and weekday_rth[i] == 4:
            continue

        if cfg.skip_gap_up:
            if not np.isnan(pdc_rth[i]):
                gap = cl[i] - pdc_rth[i]  # approximate — use open of day ideally
                # Actually check if today's first bar opened above prev close
                # Approximate: if current price is > prev_close + threshold early in session
                if t <= 1000 and not np.isnan(pdc_rth[i]) and op[i] > pdc_rth[i] + cfg.gap_up_threshold:
                    continue

        if cfg.require_high_vol:
            if vr_rth[i] < 1.0:
                continue

        # ENTER SHORT
        entry_price = cl[i]
        entry_time = times[i]
        entry_idx = i
        stop_price = entry_price * (1.0 + cfg.stop_bps / 10000.0)
        best_price = entry_price
        trail_active = False
        stagger_t1_hit = False
        in_position = True

        # Compute target based on exit mode
        if cfg.exit_mode == "fixed_target":
            target_price = entry_price - cfg.target_pts
        elif cfg.exit_mode == "atr_target":
            atr_val = atr_rth[i] if not np.isnan(atr_rth[i]) else 15.0
            target_price = entry_price - (atr_val * cfg.atr_target_mult)
        elif cfg.exit_mode == "support_target":
            # Target nearest support: prev_day_low or on_low
            candidates = []
            if not np.isnan(pdl_rth[i]) and pdl_rth[i] < entry_price - 3:
                candidates.append(pdl_rth[i])
            if not np.isnan(onl_rth[i]) and onl_rth[i] < entry_price - 3:
                candidates.append(onl_rth[i])
            if candidates:
                # Pick closest support that gives at least 5pt
                valid = [c for c in candidates if entry_price - c >= 5]
                if valid:
                    target_price = max(valid)  # closest (highest support below)
                else:
                    target_price = entry_price - 15.0  # fallback
            else:
                target_price = entry_price - 15.0  # fallback
        elif cfg.exit_mode == "trail_atr":
            target_price = entry_price - 25.0  # ceiling target
        elif cfg.exit_mode == "time_decayed":
            if t < 1100:
                target_price = entry_price - cfg.am_target_pts
            else:
                target_price = entry_price - cfg.pm_target_pts
        elif cfg.exit_mode == "opposite_signal":
            target_price = entry_price - 20.0  # ceiling
        elif cfg.exit_mode == "expanding":
            target_price = 0  # handled dynamically in exit logic
        else:
            target_price = entry_price - cfg.target_pts

        trade_count[s] = trade_count.get(s, 0) + 1
        last_entry_bar[s] = i

    # Close any open position
    if in_position:
        trades.append(make_trade(times[-1], cl[-1], "data_end"))

    return trades


# ── Walk-Forward ──

def walk_forward(df_5m, df_30m, cfg, precomputed, split_date=WF_SPLIT):
    """Run walk-forward validation. Returns dict of IS/OOS metrics."""
    split_ts = pd.Timestamp(split_date, tz=df_5m.index.tz)

    is_mask = df_5m.index < split_ts
    oos_mask = df_5m.index >= split_ts

    df_5m_is = df_5m[is_mask]
    df_5m_oos = df_5m[oos_mask]

    # Split 30m
    df_30m_is = df_30m[df_30m.index < split_ts]
    df_30m_oos = df_30m[df_30m.index >= split_ts]

    # Re-compute 30m signals for each split
    sig_is = compute_30m_signals(df_30m_is)
    sig_oos = compute_30m_signals(df_30m_oos)

    # Slice precomputed arrays
    is_mask_arr = np.array(is_mask)
    oos_mask_arr = np.array(oos_mask)
    is_idx = np.where(is_mask_arr)[0]
    oos_idx = np.where(oos_mask_arr)[0]

    def slice_pre(arr, idx):
        return arr[idx] if len(idx) <= len(arr) else arr[:len(idx)]

    trades_is = run_variant(df_5m_is, sig_is, cfg,
                            slice_pre(precomputed["vwap"], is_idx),
                            slice_pre(precomputed["tema_fast"], is_idx),
                            slice_pre(precomputed["tema_slow"], is_idx),
                            slice_pre(precomputed["tema_trend"], is_idx),
                            slice_pre(precomputed["atr"], is_idx),
                            slice_pre(precomputed["pdh"], is_idx),
                            slice_pre(precomputed["pdl"], is_idx),
                            slice_pre(precomputed["pdc"], is_idx),
                            slice_pre(precomputed["onl"], is_idx),
                            slice_pre(precomputed["vol_ratio"], is_idx))

    trades_oos = run_variant(df_5m_oos, sig_oos, cfg,
                             slice_pre(precomputed["vwap"], oos_idx),
                             slice_pre(precomputed["tema_fast"], oos_idx),
                             slice_pre(precomputed["tema_slow"], oos_idx),
                             slice_pre(precomputed["tema_trend"], oos_idx),
                             slice_pre(precomputed["atr"], oos_idx),
                             slice_pre(precomputed["pdh"], oos_idx),
                             slice_pre(precomputed["pdl"], oos_idx),
                             slice_pre(precomputed["pdc"], oos_idx),
                             slice_pre(precomputed["onl"], oos_idx),
                             slice_pre(precomputed["vol_ratio"], oos_idx))

    m_is = compute_metrics(trades_is, INITIAL_CAPITAL) if trades_is else None
    m_oos = compute_metrics(trades_oos, INITIAL_CAPITAL) if trades_oos else None

    def ttest_pval(trades_list):
        pnls = [t.pnl_dollar for t in trades_list]
        if len(pnls) > 1 and np.std(pnls) > 0:
            t_stat, p = stats.ttest_1samp(pnls, 0)
            return p / 2 if t_stat > 0 else 1.0
        return 1.0

    pf_ratio = 0.0
    if m_is and m_oos and m_is.profit_factor > 0:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor

    return {
        "is_trades": len(trades_is),
        "is_pf": m_is.profit_factor if m_is else 0,
        "is_pnl": m_is.net_pnl if m_is else 0,
        "is_sharpe": m_is.sharpe if m_is else 0,
        "is_p": ttest_pval(trades_is),
        "oos_trades": len(trades_oos),
        "oos_pf": m_oos.profit_factor if m_oos else 0,
        "oos_pnl": m_oos.net_pnl if m_oos else 0,
        "oos_wr": m_oos.win_rate if m_oos else 0,
        "oos_sharpe": m_oos.sharpe if m_oos else 0,
        "oos_p": ttest_pval(trades_oos),
        "pf_ratio": pf_ratio,
    }


# ── Reporting Helpers ──

def ttest_pval(trades):
    pnls = [t.pnl_dollar for t in trades]
    if len(pnls) > 1 and np.std(pnls) > 0:
        t_stat, p = stats.ttest_1samp(pnls, 0)
        return p / 2 if t_stat > 0 else 1.0
    return 1.0


def report_line(name, trades, baseline_pf=1.35):
    """Return a formatted result line."""
    if not trades or len(trades) < 10:
        return f"  {name:<45} {'SKIP (<10 trades)':>30}"
    m = compute_metrics(trades, INITIAL_CAPITAL)
    p = ttest_pval(trades)
    pf_delta = m.profit_factor - baseline_pf
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    beat = ">>>" if m.profit_factor > baseline_pf and p < 0.05 else "   "

    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    top_reasons = sorted(reasons.items(), key=lambda x: -x[1])[:3]
    reason_str = " ".join(f"{k}:{v}" for k, v in top_reasons)

    return (f"  {beat} {name:<42} {m.total_trades:>4} trades  "
            f"PF {m.profit_factor:>5.2f} ({pf_delta:>+5.2f})  "
            f"${m.net_pnl:>8,.0f}  WR {m.win_rate:>5.1f}%  "
            f"Sharpe {m.sharpe:>5.2f}  p={p:.4f}{sig:<3}  "
            f"DD ${m.max_drawdown:>6,.0f}  {reason_str}")


def report_detail(name, trades):
    """Print detailed metrics for a variant."""
    if not trades:
        print(f"\n  {name}: 0 trades")
        return
    m = compute_metrics(trades, INITIAL_CAPITAL)
    p = ttest_pval(trades)

    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    reason_str = ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))

    print(f"\n  {name}")
    print(f"  {'─'*60}")
    print(f"  Trades:        {m.total_trades}")
    print(f"  Win Rate:      {m.win_rate:.1f}%")
    print(f"  Profit Factor: {m.profit_factor:.2f}")
    print(f"  Net P&L:       ${m.net_pnl:,.0f}")
    print(f"  Avg Trade:     ${m.avg_trade:,.0f}")
    print(f"  Avg Win:       ${m.avg_win:,.0f}")
    print(f"  Avg Loss:      ${m.avg_loss:,.0f}")
    print(f"  Max Drawdown:  ${m.max_drawdown:,.0f} ({m.max_drawdown_pct:.1f}%)")
    print(f"  Sharpe:        {m.sharpe:.2f}")
    print(f"  p-value:       {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")
    print(f"  Win Streak:    {m.longest_win_streak}  |  Loss Streak: {m.longest_lose_streak}")
    print(f"  Exits:         {reason_str}")


# ── Build All Variants ──

def build_all_variants():
    """Return list of (name, V2Config) for every structural variant to test."""
    variants = []

    # ═══ BASELINE ═══
    variants.append(("BASELINE: 15pt fixed target", V2Config(
        name="baseline", target_pts=15.0)))

    # ═══ SIGNAL VARIANTS ═══
    variants.append(("SIG: double lower close", V2Config(
        name="double_lower", signal_mode="double_lower")))

    variants.append(("SIG: lower close + TEMA bearish", V2Config(
        name="tema_confirm", use_tema_confirm=True)))

    variants.append(("SIG: lower close + volume surge", V2Config(
        name="vol_surge", signal_mode="volume_surge")))

    variants.append(("SIG: lower close + range expansion", V2Config(
        name="range_expand", signal_mode="range_expand")))

    variants.append(("SIG: lower close + below VWAP", V2Config(
        name="below_vwap", use_below_vwap=True)))

    variants.append(("SIG: lower close + trend context (TEMA55)", V2Config(
        name="trend_ctx", use_trend_context=True)))

    for pts in [1.0, 2.0, 3.0, 5.0]:
        variants.append((f"SIG: conviction >= {pts}pt below", V2Config(
            name=f"conviction_{pts}", signal_mode="conviction", conviction_min_pts=pts)))

    # Combined signal filters
    variants.append(("SIG: TEMA bearish + below VWAP", V2Config(
        name="tema_vwap", use_tema_confirm=True, use_below_vwap=True)))

    variants.append(("SIG: TEMA bearish + trend ctx", V2Config(
        name="tema_trend", use_tema_confirm=True, use_trend_context=True)))

    variants.append(("SIG: vol surge + TEMA bearish", V2Config(
        name="vol_tema", signal_mode="volume_surge", use_tema_confirm=True)))

    # ═══ EXIT VARIANTS ═══
    for mult in [1.0, 1.5, 2.0, 2.5]:
        variants.append((f"EXIT: ATR x{mult} target", V2Config(
            name=f"atr_{mult}x", exit_mode="atr_target", atr_target_mult=mult)))

    variants.append(("EXIT: support level target", V2Config(
        name="support_tgt", exit_mode="support_target")))

    for trig in [5.0, 8.0, 10.0]:
        for mult in [0.8, 1.0, 1.5]:
            variants.append((f"EXIT: trail ATR (trig={trig}, mult={mult})", V2Config(
                name=f"trail_{trig}_{mult}", exit_mode="trail_atr",
                trail_trigger_pts=trig, trail_atr_mult=mult)))

    variants.append(("EXIT: time-decayed (AM=20, PM=10)", V2Config(
        name="time_decay_20_10", exit_mode="time_decayed", am_target_pts=20.0, pm_target_pts=10.0)))
    variants.append(("EXIT: time-decayed (AM=25, PM=12)", V2Config(
        name="time_decay_25_12", exit_mode="time_decayed", am_target_pts=25.0, pm_target_pts=12.0)))
    variants.append(("EXIT: time-decayed (AM=18, PM=8)", V2Config(
        name="time_decay_18_8", exit_mode="time_decayed", am_target_pts=18.0, pm_target_pts=8.0)))

    variants.append(("EXIT: opposite signal (30m > prior high)", V2Config(
        name="opp_signal", exit_mode="opposite_signal")))

    variants.append(("EXIT: expanding (12pt fast / 20pt slow)", V2Config(
        name="expand_12_20", exit_mode="expanding", expand_fast_pts=12.0, expand_slow_pts=20.0)))
    variants.append(("EXIT: expanding (10pt fast / 25pt slow)", V2Config(
        name="expand_10_25", exit_mode="expanding", expand_fast_pts=10.0, expand_slow_pts=25.0)))
    variants.append(("EXIT: expanding (15pt fast / 30pt slow)", V2Config(
        name="expand_15_30", exit_mode="expanding", expand_fast_pts=15.0, expand_slow_pts=30.0)))

    # Fixed target alternatives around 15pt
    for pts in [10.0, 12.0, 18.0, 20.0, 25.0]:
        variants.append((f"EXIT: fixed {pts}pt target", V2Config(
            name=f"fixed_{pts}", target_pts=pts)))

    # ═══ FILTER VARIANTS ═══
    variants.append(("FILTER: skip gap-up days (>10pt)", V2Config(
        name="no_gapup", skip_gap_up=True, gap_up_threshold=10.0)))
    variants.append(("FILTER: skip gap-up days (>5pt)", V2Config(
        name="no_gapup5", skip_gap_up=True, gap_up_threshold=5.0)))

    variants.append(("FILTER: skip Friday", V2Config(
        name="no_fri", skip_friday=True)))

    variants.append(("FILTER: require high vol (vol_ratio>1.0)", V2Config(
        name="high_vol", require_high_vol=True)))

    variants.append(("FILTER: morning only (935-1100)", V2Config(
        name="morning", time_bucket="morning")))

    variants.append(("FILTER: midday only (1100-1300)", V2Config(
        name="midday", time_bucket="midday")))

    # Max trades/day
    for mt in [2, 3]:
        variants.append((f"FILTER: max {mt} trades/day", V2Config(
            name=f"max_{mt}", max_trades_day=mt)))

    # ═══ HOLD TIME VARIANTS ═══
    for hold in [3, 9, 12, 18]:
        variants.append((f"HOLD: max {hold} bars ({hold*5}min)", V2Config(
            name=f"hold_{hold}", max_hold_bars=hold)))

    # Longer hold + larger target
    variants.append(("HOLD+TGT: 12 bars + 20pt target", V2Config(
        name="hold12_tgt20", max_hold_bars=12, target_pts=20.0)))
    variants.append(("HOLD+TGT: 18 bars + 25pt target", V2Config(
        name="hold18_tgt25", max_hold_bars=18, target_pts=25.0)))

    return variants


def build_combination_variants(best_signal, best_exit, best_filter):
    """Build combinations of the best individual improvements."""
    combos = []

    # We'll pass the actual config dicts and merge them
    # For now, build specific combos based on what wins

    return combos


# ── Main ──

def main():
    t_start = time.time()
    print("=" * 90)
    print("  TREND CONTINUATION ES v2 — STRUCTURAL IMPROVEMENT SEARCH")
    print("  Baseline: 30m close < prior 30m LOW | SHORT | 30bps stop | 15pt target | 6 bar hold")
    print("  Baseline: 374 trades, PF 1.35, +$25K, Sharpe 2.00, p=0.0077, WF ratio 1.24")
    print("=" * 90)

    # ── Load and Prepare Data ──
    print(f"\n  Loading data from {DATA_PATH}...")
    df_5m = load_tos_csv(DATA_PATH)
    print(f"  Loaded {len(df_5m):,} 5m bars, {df_5m.index[0].date()} to {df_5m.index[-1].date()}")

    print("  Aggregating 30m bars...")
    df_30m = aggregate_30m(df_5m)
    print(f"  Generated {len(df_30m):,} 30m bars")

    print("  Computing 30m signals...")
    df_30m_sig = compute_30m_signals(df_30m)

    print("  Precomputing indicators on 5m bars...")
    # TEMA
    cl_series = pd.Series(df_5m["close"].values, index=df_5m.index)
    ema1_f = cl_series.ewm(span=9, adjust=False).mean().values
    ema2_f = pd.Series(ema1_f).ewm(span=9, adjust=False).mean().values
    ema3_f = pd.Series(ema2_f).ewm(span=9, adjust=False).mean().values
    tema_fast = 3 * ema1_f - 3 * ema2_f + ema3_f

    ema1_s = cl_series.ewm(span=21, adjust=False).mean().values
    ema2_s = pd.Series(ema1_s).ewm(span=21, adjust=False).mean().values
    ema3_s = pd.Series(ema2_s).ewm(span=21, adjust=False).mean().values
    tema_slow = 3 * ema1_s - 3 * ema2_s + ema3_s

    ema1_t = cl_series.ewm(span=55, adjust=False).mean().values
    ema2_t = pd.Series(ema1_t).ewm(span=55, adjust=False).mean().values
    ema3_t = pd.Series(ema2_t).ewm(span=55, adjust=False).mean().values
    tema_trend = 3 * ema1_t - 3 * ema2_t + ema3_t

    # ATR
    atr_5m = compute_atr(df_5m["high"].values, df_5m["low"].values, df_5m["close"].values, 14)

    # Vol ratio
    atr_avg = pd.Series(atr_5m).rolling(50, min_periods=10).mean().values
    vol_ratio_5m = np.where(atr_avg > 0, atr_5m / atr_avg, 1.0)

    # VWAP
    print("  Computing session VWAP...")
    vwap = compute_session_vwap(df_5m)

    # Previous day levels
    print("  Computing previous day levels...")
    pdh, pdl, pdc, onl = compute_prev_day_levels(df_5m)

    precomputed = {
        "tema_fast": tema_fast,
        "tema_slow": tema_slow,
        "tema_trend": tema_trend,
        "atr": atr_5m,
        "vol_ratio": vol_ratio_5m,
        "vwap": vwap,
        "pdh": pdh,
        "pdl": pdl,
        "pdc": pdc,
        "onl": onl,
    }

    elapsed = time.time() - t_start
    print(f"  Data prep complete in {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: RUN ALL VARIANTS ON FULL DATASET
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  PHASE 1: FULL-DATASET VARIANT SWEEP")
    print(f"{'='*90}")

    variants = build_all_variants()
    print(f"\n  Testing {len(variants)} structural variants...\n")

    results = []
    baseline_pf = 1.35

    for name, cfg in variants:
        trades = run_variant(df_5m, df_30m_sig, cfg,
                             vwap, tema_fast, tema_slow, tema_trend,
                             atr_5m, pdh, pdl, pdc, onl, vol_ratio_5m)
        m = compute_metrics(trades, INITIAL_CAPITAL) if trades else None
        p = ttest_pval(trades) if trades else 1.0

        results.append({
            "name": name,
            "cfg": cfg,
            "trades": trades,
            "n_trades": len(trades),
            "pf": m.profit_factor if m else 0,
            "pnl": m.net_pnl if m else 0,
            "wr": m.win_rate if m else 0,
            "sharpe": m.sharpe if m else 0,
            "dd": m.max_drawdown if m else 0,
            "p": p,
        })
        print(report_line(name, trades, baseline_pf))

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: RANK AND IDENTIFY CANDIDATES
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*90}")
    print(f"  PHASE 2: RANKING — CANDIDATES THAT BEAT BASELINE")
    print(f"{'='*90}")

    # Filter: >= 100 trades, PF > baseline, p < 0.10
    candidates = [r for r in results
                  if r["n_trades"] >= 100 and r["pf"] > baseline_pf and r["p"] < 0.10]

    candidates.sort(key=lambda x: x["pf"], reverse=True)

    if not candidates:
        print("\n  NO variants beat baseline with >= 100 trades and p < 0.10")
        # Show best anyway
        viable = [r for r in results if r["n_trades"] >= 50 and r["pf"] > 1.0]
        viable.sort(key=lambda x: x["pf"], reverse=True)
        candidates = viable[:10]
        if candidates:
            print(f"  Showing top {len(candidates)} with >= 50 trades and PF > 1.0:")

    print(f"\n  {'Rank':<5} {'Name':<45} {'Trades':>6} {'PF':>6} {'P&L':>10} "
          f"{'WR%':>6} {'Sharpe':>7} {'p':>8} {'DD':>8}")
    print(f"  {'─'*105}")

    for rank, r in enumerate(candidates[:15], 1):
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"  {rank:<5} {r['name']:<45} {r['n_trades']:>6} {r['pf']:>6.2f} "
              f"${r['pnl']:>9,.0f} {r['wr']:>5.1f}% {r['sharpe']:>7.2f} "
              f"{r['p']:>7.4f}{sig:<3} ${r['dd']:>7,.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: COMBINATION TESTING
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*90}")
    print(f"  PHASE 3: COMBINATION TESTING — BEST SIGNAL + EXIT + FILTER")
    print(f"{'='*90}")

    # Identify best in each category from phase 1
    sig_results = [r for r in results if r["name"].startswith("SIG:") and r["n_trades"] >= 50]
    exit_results = [r for r in results if r["name"].startswith("EXIT:") and r["n_trades"] >= 50]
    filter_results = [r for r in results if r["name"].startswith("FILTER:") and r["n_trades"] >= 50]

    best_sigs = sorted(sig_results, key=lambda x: x["pf"], reverse=True)[:3]
    best_exits = sorted(exit_results, key=lambda x: x["pf"], reverse=True)[:4]
    best_filters = sorted(filter_results, key=lambda x: x["pf"], reverse=True)[:3]

    print(f"\n  Best signal enhancements: {[r['name'].replace('SIG: ','') for r in best_sigs]}")
    print(f"  Best exits: {[r['name'].replace('EXIT: ','') for r in best_exits]}")
    print(f"  Best filters: {[r['name'].replace('FILTER: ','') for r in best_filters]}")

    combo_results = []

    # Build combinations
    combo_configs = []

    # Signal + exit combos
    for sr in best_sigs:
        for er in best_exits:
            s_cfg = sr["cfg"]
            e_cfg = er["cfg"]
            combo_name = f"COMBO: {sr['name'].replace('SIG: ','')} + {er['name'].replace('EXIT: ','')}"
            cfg = V2Config(
                name=combo_name,
                signal_mode=s_cfg.signal_mode,
                conviction_min_pts=s_cfg.conviction_min_pts,
                use_tema_confirm=s_cfg.use_tema_confirm,
                use_below_vwap=s_cfg.use_below_vwap,
                use_trend_context=s_cfg.use_trend_context,
                exit_mode=e_cfg.exit_mode,
                target_pts=e_cfg.target_pts,
                atr_target_mult=e_cfg.atr_target_mult,
                trail_trigger_pts=e_cfg.trail_trigger_pts,
                trail_atr_mult=e_cfg.trail_atr_mult,
                am_target_pts=e_cfg.am_target_pts,
                pm_target_pts=e_cfg.pm_target_pts,
                expand_fast_pts=e_cfg.expand_fast_pts,
                expand_slow_pts=e_cfg.expand_slow_pts,
            )
            combo_configs.append((combo_name, cfg))

    # Signal + filter combos
    for sr in best_sigs:
        for fr in best_filters:
            s_cfg = sr["cfg"]
            f_cfg = fr["cfg"]
            combo_name = f"COMBO: {sr['name'].replace('SIG: ','')} + {fr['name'].replace('FILTER: ','')}"
            cfg = V2Config(
                name=combo_name,
                signal_mode=s_cfg.signal_mode,
                conviction_min_pts=s_cfg.conviction_min_pts,
                use_tema_confirm=s_cfg.use_tema_confirm,
                use_below_vwap=s_cfg.use_below_vwap,
                use_trend_context=s_cfg.use_trend_context,
                skip_gap_up=f_cfg.skip_gap_up,
                gap_up_threshold=f_cfg.gap_up_threshold,
                skip_friday=f_cfg.skip_friday,
                require_high_vol=f_cfg.require_high_vol,
                time_bucket=f_cfg.time_bucket,
                max_trades_day=f_cfg.max_trades_day,
            )
            combo_configs.append((combo_name, cfg))

    # Exit + filter combos
    for er in best_exits:
        for fr in best_filters:
            e_cfg = er["cfg"]
            f_cfg = fr["cfg"]
            combo_name = f"COMBO: {er['name'].replace('EXIT: ','')} + {fr['name'].replace('FILTER: ','')}"
            cfg = V2Config(
                name=combo_name,
                exit_mode=e_cfg.exit_mode,
                target_pts=e_cfg.target_pts,
                atr_target_mult=e_cfg.atr_target_mult,
                trail_trigger_pts=e_cfg.trail_trigger_pts,
                trail_atr_mult=e_cfg.trail_atr_mult,
                am_target_pts=e_cfg.am_target_pts,
                pm_target_pts=e_cfg.pm_target_pts,
                expand_fast_pts=e_cfg.expand_fast_pts,
                expand_slow_pts=e_cfg.expand_slow_pts,
                skip_gap_up=f_cfg.skip_gap_up,
                gap_up_threshold=f_cfg.gap_up_threshold,
                skip_friday=f_cfg.skip_friday,
                require_high_vol=f_cfg.require_high_vol,
                time_bucket=f_cfg.time_bucket,
                max_trades_day=f_cfg.max_trades_day,
            )
            combo_configs.append((combo_name, cfg))

    # Triple combos: best signal + best exit + best filter
    for sr in best_sigs[:2]:
        for er in best_exits[:3]:
            for fr in best_filters[:2]:
                s_cfg = sr["cfg"]
                e_cfg = er["cfg"]
                f_cfg = fr["cfg"]
                combo_name = (f"TRIPLE: {sr['name'].replace('SIG: ','')[:15]} + "
                              f"{er['name'].replace('EXIT: ','')[:15]} + "
                              f"{fr['name'].replace('FILTER: ','')[:15]}")
                cfg = V2Config(
                    name=combo_name,
                    signal_mode=s_cfg.signal_mode,
                    conviction_min_pts=s_cfg.conviction_min_pts,
                    use_tema_confirm=s_cfg.use_tema_confirm,
                    use_below_vwap=s_cfg.use_below_vwap,
                    use_trend_context=s_cfg.use_trend_context,
                    exit_mode=e_cfg.exit_mode,
                    target_pts=e_cfg.target_pts,
                    atr_target_mult=e_cfg.atr_target_mult,
                    trail_trigger_pts=e_cfg.trail_trigger_pts,
                    trail_atr_mult=e_cfg.trail_atr_mult,
                    am_target_pts=e_cfg.am_target_pts,
                    pm_target_pts=e_cfg.pm_target_pts,
                    expand_fast_pts=e_cfg.expand_fast_pts,
                    expand_slow_pts=e_cfg.expand_slow_pts,
                    skip_gap_up=f_cfg.skip_gap_up,
                    gap_up_threshold=f_cfg.gap_up_threshold,
                    skip_friday=f_cfg.skip_friday,
                    require_high_vol=f_cfg.require_high_vol,
                    time_bucket=f_cfg.time_bucket,
                    max_trades_day=f_cfg.max_trades_day,
                )
                combo_configs.append((combo_name, cfg))

    print(f"\n  Testing {len(combo_configs)} combinations...\n")

    for name, cfg in combo_configs:
        trades = run_variant(df_5m, df_30m_sig, cfg,
                             vwap, tema_fast, tema_slow, tema_trend,
                             atr_5m, pdh, pdl, pdc, onl, vol_ratio_5m)
        m = compute_metrics(trades, INITIAL_CAPITAL) if trades else None
        p = ttest_pval(trades) if trades else 1.0

        combo_results.append({
            "name": name,
            "cfg": cfg,
            "trades": trades,
            "n_trades": len(trades),
            "pf": m.profit_factor if m else 0,
            "pnl": m.net_pnl if m else 0,
            "wr": m.win_rate if m else 0,
            "sharpe": m.sharpe if m else 0,
            "dd": m.max_drawdown if m else 0,
            "p": p,
        })
        print(report_line(name, trades, baseline_pf))

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: WALK-FORWARD ON TOP CANDIDATES
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*90}")
    print(f"  PHASE 4: WALK-FORWARD VALIDATION (split: {WF_SPLIT})")
    print(f"{'='*90}")

    # Merge all results, pick top candidates
    all_results = results + combo_results
    wf_candidates = [r for r in all_results
                     if r["n_trades"] >= 80 and r["pf"] > 1.0 and r["p"] < 0.10]
    wf_candidates.sort(key=lambda x: x["pf"], reverse=True)
    wf_candidates = wf_candidates[:15]  # Top 15

    # Always include baseline
    baseline_result = results[0]
    if baseline_result not in wf_candidates:
        wf_candidates.insert(0, baseline_result)

    print(f"\n  Walk-forward testing {len(wf_candidates)} candidates...\n")

    print(f"  {'Name':<48} {'Full':>5} {'IS':>5} {'OOS':>5} {'IS PF':>7} {'OOS PF':>8} "
          f"{'Ratio':>7} {'OOS p':>8} {'OOS $':>10} {'Result':>8}")
    print(f"  {'─'*120}")

    wf_results = []
    for r in wf_candidates:
        wf = walk_forward(df_5m, df_30m, r["cfg"], precomputed)

        result = "PASS" if (wf["pf_ratio"] > 0.7 and wf["oos_pf"] > 1.0) else "FAIL"
        sig = "*" if wf["oos_p"] < 0.05 else ""

        wf_results.append({
            "name": r["name"],
            "cfg": r["cfg"],
            "trades": r["trades"],
            "full_pf": r["pf"],
            "full_pnl": r["pnl"],
            "full_p": r["p"],
            "n_trades": r["n_trades"],
            **wf,
            "wf_result": result,
        })

        print(f"  {r['name']:<48} {r['n_trades']:>5} {wf['is_trades']:>5} {wf['oos_trades']:>5} "
              f"{wf['is_pf']:>7.2f} {wf['oos_pf']:>8.2f} {wf['pf_ratio']:>7.2f} "
              f"{wf['oos_p']:>7.4f}{sig:<1} ${wf['oos_pnl']:>9,.0f} {result:>8}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: DETAILED ANALYSIS OF BEST CANDIDATE
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*90}")
    print(f"  PHASE 5: BEST CANDIDATE — DETAILED ANALYSIS")
    print(f"{'='*90}")

    # Best = highest OOS PF among PASS results, with > baseline full PF
    passed = [r for r in wf_results if r["wf_result"] == "PASS" and r["full_pf"] > baseline_pf]
    if not passed:
        passed = [r for r in wf_results if r["wf_result"] == "PASS"]
    if not passed:
        passed = sorted(wf_results, key=lambda x: x.get("oos_pf", 0), reverse=True)

    best = passed[0] if passed else wf_results[0]

    report_detail(f"BEST: {best['name']}", best["trades"])

    # Monthly breakdown
    if best["trades"]:
        print(f"\n  MONTHLY BREAKDOWN:")
        monthly = {}
        for t in best["trades"]:
            if hasattr(t.entry_time, 'strftime'):
                month = t.entry_time.strftime("%Y-%m")
            else:
                month = str(t.entry_time)[:7]
            monthly.setdefault(month, []).append(t.pnl_dollar)

        print(f"  {'Month':<10} {'Trades':>7} {'Net P&L':>10} {'Avg':>8} {'WR%':>6}")
        win_months = 0
        for month in sorted(monthly.keys()):
            pnls = monthly[month]
            net = sum(pnls)
            avg = np.mean(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            if net > 0:
                win_months += 1
            print(f"  {month:<10} {len(pnls):>7} ${net:>9,.0f} ${avg:>7,.0f} {wr:>5.1f}%")
        print(f"  Winning months: {win_months}/{len(monthly)} ({win_months/len(monthly)*100:.0f}%)")

    # Baseline detailed for comparison
    report_detail("BASELINE COMPARISON", baseline_result["trades"])

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 6: FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*90}")
    print(f"  FINAL VERDICT")
    print(f"{'='*90}")

    # Count how many variants beat baseline
    beat_baseline = [r for r in all_results
                     if r["n_trades"] >= 100 and r["pf"] > baseline_pf and r["p"] < 0.05]
    wf_pass_beat = [r for r in wf_results
                    if r["wf_result"] == "PASS" and r["full_pf"] > baseline_pf]

    print(f"\n  Variants tested:              {len(variants) + len(combo_configs)}")
    print(f"  Beat baseline (full, p<0.05): {len(beat_baseline)}")
    print(f"  WF PASS + beat baseline:      {len(wf_pass_beat)}")

    if wf_pass_beat:
        print(f"\n  VALIDATED IMPROVEMENTS:")
        print(f"  {'─'*100}")
        for r in sorted(wf_pass_beat, key=lambda x: x.get("oos_pf", 0), reverse=True):
            print(f"  {r['name']}")
            print(f"    Full: {r['n_trades']} trades, PF {r['full_pf']:.2f}, ${r['full_pnl']:,.0f}, p={r['full_p']:.4f}")
            print(f"    WF:   IS PF {r['is_pf']:.2f}, OOS PF {r['oos_pf']:.2f}, ratio {r['pf_ratio']:.2f}, "
                  f"OOS ${r['oos_pnl']:,.0f}")
            print()

        best_final = sorted(wf_pass_beat, key=lambda x: x.get("oos_pf", 0), reverse=True)[0]
        print(f"\n  WINNER: {best_final['name']}")
        print(f"  Full:   {best_final['n_trades']} trades, PF {best_final['full_pf']:.2f}, "
              f"${best_final['full_pnl']:,.0f}, p={best_final['full_p']:.4f}")
        print(f"  WF:     IS PF {best_final['is_pf']:.2f}, OOS PF {best_final['oos_pf']:.2f}, "
              f"ratio {best_final['pf_ratio']:.2f}")
        print(f"\n  vs BASELINE: PF {baseline_pf:.2f} -> {best_final['full_pf']:.2f} "
              f"(+{(best_final['full_pf']-baseline_pf)/baseline_pf*100:.1f}%)")
    else:
        print(f"\n  RESULT: No structural variant PASSED walk-forward AND beat baseline.")
        print(f"  The baseline (15pt fixed target, 30bps stop, 6 bar hold) is already strong.")
        print(f"  Recommendation: Keep baseline as-is. Look for edge elsewhere (new setups, not tweaks).")

        # Still show the closest contenders
        near_misses = sorted(wf_results, key=lambda x: x.get("oos_pf", 0), reverse=True)[:3]
        if near_misses:
            print(f"\n  Closest contenders:")
            for r in near_misses:
                print(f"    {r['name']}: Full PF {r['full_pf']:.2f}, OOS PF {r.get('oos_pf',0):.2f}, "
                      f"WF {r['wf_result']}")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
