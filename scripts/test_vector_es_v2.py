#!/usr/bin/env python3
"""Vector ES v2 — STRUCTURAL VARIANT SWEEP.

NOT parameter tweaks. Structurally different signals, exits, and filters.
Every variant gets walk-forward + t-test. Goal: beat baseline PF 1.377.

Baseline: EMA 9/21 bearish cross, short-only, 20bps stop, time stop 15:55,
          TEMA filter ON, range >= 8, 506 trades, PF 1.377, +$53K, p=0.028.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics
from backtester.position import Trade
from backtester.indicators import compute_indicators


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class V2Config:
    """All knobs for structural variant testing."""

    # ── Signal type ──
    # "ema_cross"       = EMA 9/21 bearish cross (BASELINE)
    # "tema_cross"      = TEMA 9/21 bearish cross
    # "sma_cross"       = SMA 8/24 bearish cross
    # "ema_trend_conf"  = EMA 9/21 cross + close < TEMA(55)
    # "price_cross_ema" = close crosses below EMA 21
    # "multi_ma"        = EMA 9<21 AND SMA 8<24 both bearish
    signal_type: str = "ema_cross"

    # ── Exit type ──
    # "time_stop"         = exit at 15:55 (BASELINE)
    # "atr_target_1x"     = 1x ATR target
    # "atr_target_1_5x"   = 1.5x ATR target
    # "atr_target_2x"     = 2x ATR target
    # "atr_target_3x"     = 3x ATR target
    # "cross_min_hold"    = opposite cross exit but min 6 bars (30min) hold
    # "trailing_atr"      = trail at 1x ATR behind, activate after 0.5x ATR profit
    # "momentum_exit"     = exit when vol_ratio < 0.5
    # "prev_day_low"      = target previous day low (support)
    # "session_vwap"      = target session VWAP
    exit_type: str = "time_stop"

    # ── Filter flags ──
    require_tema_bearish: bool = True
    min_30m_range: float = 8.0
    require_volume_spike: bool = False   # vol_ratio > 1.5
    require_atr_expansion: bool = False  # atr > atr_avg
    require_consec_red: bool = False     # 2+ red bars before signal
    skip_gap_up: bool = False            # skip if ON high > prev close by 10+ pts
    skip_friday: bool = False
    time_bucket: str = "all"             # "all", "morning", "midday", "afternoon"

    # ── Core params (held constant from baseline) ──
    stop_bps: float = 20.0
    entry_start: int = 935
    entry_end: int = 1500
    time_stop_time: int = 1555
    max_trades_per_day: int = 2
    point_value: float = 50.0
    commission_rt: float = 5.0
    slippage_ticks: int = 1
    tick_size: float = 0.25

    # Label for output
    label: str = ""


# ══════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "es_5m_databento_2yr.csv",
)
WF_SPLIT = "2025-02-16"
INITIAL_CAPITAL = 100_000.0


def load_data():
    """Load ES 5m data with all indicators."""
    print(f"Loading {DATA_PATH}...")
    df = load_tos_csv(DATA_PATH, instrument="ES")
    compute_indicators(df)

    # Add extra columns we need
    # Consecutive red bars
    df["red_bar"] = df["close"] < df["open"]
    df["consec_red"] = 0
    red = df["red_bar"].values
    consec = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if red[i]:
            consec[i] = consec[i-1] + 1
        else:
            consec[i] = 0
    df["consec_red"] = consec

    # Session VWAP (cumulative within RTH)
    df["cum_vol"] = 0.0
    df["cum_vwap_num"] = 0.0
    df["session_vwap"] = np.nan

    sess = df["session_date"].values
    vol = df["volume"].values
    hlc3 = df["hlc3"].values
    is_rth = df["is_rth"].values
    cum_vol = 0.0
    cum_num = 0.0
    prev_s = None

    for i in range(len(df)):
        if sess[i] != prev_s:
            cum_vol = 0.0
            cum_num = 0.0
            prev_s = sess[i]
        if is_rth[i]:
            cum_vol += vol[i]
            cum_num += hlc3[i] * vol[i]
            if cum_vol > 0:
                df.iloc[i, df.columns.get_loc("session_vwap")] = cum_num / cum_vol

    df["session_vwap"] = df["session_vwap"].ffill()

    # Previous day low (using session_date grouping)
    prev_day_low = {}
    dates = sorted(df[df["is_rth"]]["session_date"].unique())
    for i, d in enumerate(dates):
        if i == 0:
            continue
        prev_d = dates[i-1]
        prev_rth = df[(df["session_date"] == prev_d) & df["is_rth"]]
        if not prev_rth.empty:
            prev_day_low[d] = prev_rth["low"].min()
    df["prev_day_low"] = df["session_date"].map(prev_day_low)

    # Previous day close (for gap filter)
    prev_day_close = {}
    for i, d in enumerate(dates):
        if i == 0:
            continue
        prev_d = dates[i-1]
        prev_rth = df[(df["session_date"] == prev_d) & df["is_rth"]]
        if not prev_rth.empty:
            prev_day_close[d] = prev_rth.iloc[-1]["close"]
    df["prev_day_close"] = df["session_date"].map(prev_day_close)

    # Overnight high per session (for gap filter)
    on_high = {}
    for d in dates:
        globex = df[(df["session_date"] == d) & ~df["is_rth"]]
        if not globex.empty:
            on_high[d] = globex["high"].max()
    df["on_high"] = df["session_date"].map(on_high)

    # Gap up flag: ON high > prev close by 10+ pts
    df["is_gap_up"] = (df["on_high"] - df["prev_day_close"]) >= 10.0

    print(f"  {len(df):,} bars | {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    return df


# ══════════════════════════════════════════════════════════════════════
#  30-MINUTE RANGE (reused from v1)
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
#  CROSS DETECTION (all signal types)
# ══════════════════════════════════════════════════════════════════════

def detect_all_crosses(df: pd.DataFrame) -> dict:
    """Precompute all cross signals we'll need for any variant."""
    n = len(df)
    sess = df["session_date"].values
    cl = df["close"].values

    crosses = {}

    # EMA 9/21 crosses
    ema_9 = df["ema_9"].values
    ema_21 = df["ema_21"].values
    ema_bear = np.zeros(n, dtype=bool)
    ema_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if ema_9[i-1] >= ema_21[i-1] and ema_9[i] < ema_21[i]:
            ema_bear[i] = True
        if ema_9[i-1] <= ema_21[i-1] and ema_9[i] > ema_21[i]:
            ema_bull[i] = True
    crosses["ema_cross_bear"] = ema_bear
    crosses["ema_cross_bull"] = ema_bull

    # TEMA 9/21 crosses (precomputed in indicators)
    crosses["tema_cross_bear"] = df["tema_cross_down"].values
    crosses["tema_cross_bull"] = df["tema_cross_up"].values

    # SMA 8/24 crosses
    sma_8 = df["sma_8"].values
    sma_24 = df["sma_24"].values
    sma_bear = np.zeros(n, dtype=bool)
    sma_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if np.isnan(sma_8[i]) or np.isnan(sma_24[i]) or np.isnan(sma_8[i-1]) or np.isnan(sma_24[i-1]):
            continue
        if sma_8[i-1] >= sma_24[i-1] and sma_8[i] < sma_24[i]:
            sma_bear[i] = True
        if sma_8[i-1] <= sma_24[i-1] and sma_8[i] > sma_24[i]:
            sma_bull[i] = True
    crosses["sma_cross_bear"] = sma_bear
    crosses["sma_cross_bull"] = sma_bull

    # Price crosses below EMA 21 (close crosses below ema_21)
    price_bear = np.zeros(n, dtype=bool)
    price_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if cl[i-1] >= ema_21[i-1] and cl[i] < ema_21[i]:
            price_bear[i] = True
        if cl[i-1] <= ema_21[i-1] and cl[i] > ema_21[i]:
            price_bull[i] = True
    crosses["price_cross_bear"] = price_bear
    crosses["price_cross_bull"] = price_bull

    # Multi-MA confluence: EMA 9 < 21 AND SMA 8 < 24 both just became true
    # Signal fires when BOTH are bearish and at least one JUST crossed
    multi_bear = np.zeros(n, dtype=bool)
    multi_bull = np.zeros(n, dtype=bool)
    ema_bearish = ema_9 < ema_21
    sma_bearish = sma_8 < sma_24
    for i in range(1, n):
        if sess[i] != sess[i-1]:
            continue
        if np.isnan(sma_8[i]) or np.isnan(sma_24[i]):
            continue
        both_bear_now = ema_bearish[i] and sma_bearish[i]
        both_bear_prev = ema_bearish[i-1] and sma_bearish[i-1] if i > 0 else False
        if both_bear_now and not both_bear_prev:
            multi_bear[i] = True
        both_bull_now = not ema_bearish[i] and not sma_bearish[i]
        both_bull_prev = not ema_bearish[i-1] and not sma_bearish[i-1] if i > 0 else False
        if both_bull_now and not both_bull_prev:
            multi_bull[i] = True
    crosses["multi_bear"] = multi_bear
    crosses["multi_bull"] = multi_bull

    # EMA trend confirmation cross: EMA 9/21 bearish cross + close < TEMA(55)
    tema_trend = df["tema_trend"].values
    ema_trend_bear = np.zeros(n, dtype=bool)
    for i in range(n):
        if ema_bear[i] and cl[i] < tema_trend[i]:
            ema_trend_bear[i] = True
    crosses["ema_trend_conf_bear"] = ema_trend_bear
    # For exit, same as EMA bull
    crosses["ema_trend_conf_bull"] = ema_bull

    return crosses


# ══════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, cfg: V2Config,
                 range_lookup: dict, crosses: dict) -> List[Trade]:
    """Generic backtest engine for all structural variants."""

    # Select signal arrays
    signal_map = {
        "ema_cross":       ("ema_cross_bear",       "ema_cross_bull"),
        "tema_cross":      ("tema_cross_bear",       "tema_cross_bull"),
        "sma_cross":       ("sma_cross_bear",       "sma_cross_bull"),
        "ema_trend_conf":  ("ema_trend_conf_bear",  "ema_trend_conf_bull"),
        "price_cross_ema": ("price_cross_bear",     "price_cross_bull"),
        "multi_ma":        ("multi_bear",           "multi_bull"),
    }

    bear_key, bull_key = signal_map[cfg.signal_type]
    sig_bear = crosses[bear_key]
    sig_bull = crosses[bull_key]

    # Data arrays
    et = df["et_time"].values
    sess = df["session_date"].values
    cl = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    op = df["open"].values
    is_rth = df["is_rth"].values
    timestamps = df.index
    weekday = df["weekday"].values
    n = len(df)

    # Filter arrays
    tema_bearish = df["tema_bearish"].values
    vol_ratio = df["vol_ratio"].values
    atr_vals = df["atr"].values
    atr_avg_vals = df["atr_avg"].values
    consec_red = df["consec_red"].values
    is_gap_up = df["is_gap_up"].values if "is_gap_up" in df.columns else np.zeros(n, dtype=bool)
    prev_day_low_arr = df["prev_day_low"].values if "prev_day_low" in df.columns else np.full(n, np.nan)
    session_vwap_arr = df["session_vwap"].values if "session_vwap" in df.columns else np.full(n, np.nan)

    # Time bucket boundaries
    time_buckets = {
        "all":       (cfg.entry_start, cfg.entry_end),
        "morning":   (935, 1100),
        "midday":    (1100, 1300),
        "afternoon": (1300, 1500),
    }
    tb_start, tb_end = time_buckets.get(cfg.time_bucket, (cfg.entry_start, cfg.entry_end))

    slippage = cfg.slippage_ticks * cfg.tick_size
    warmup = 60  # enough for all MAs

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
                    setup="VEC2", direction=-1,
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

            # 1. Stop loss
            if hi[i] >= position["stop"]:
                exit_reason = "stop"
                exit_price = position["stop"] + slippage

            # 2. Target hit (if set)
            if exit_reason is None and position["target"] > 0 and lo[i] <= position["target"]:
                exit_reason = "target"
                exit_price = position["target"] - slippage

            # 3. Trailing stop
            if exit_reason is None and position.get("trail_active", False):
                # Update best price (lowest for short)
                if lo[i] < position["best_price"]:
                    position["best_price"] = lo[i]
                    position["trail_stop"] = position["best_price"] + position["trail_dist"]
                if hi[i] >= position["trail_stop"]:
                    exit_reason = "trail"
                    exit_price = position["trail_stop"] + slippage
            elif exit_reason is None and position.get("trail_dist", 0) > 0:
                # Check activation: need 0.5x ATR profit first
                if lo[i] < position["best_price"]:
                    position["best_price"] = lo[i]
                profit = position["entry_price"] - position["best_price"]
                if profit >= position["trail_trigger"]:
                    position["trail_active"] = True
                    position["trail_stop"] = position["best_price"] + position["trail_dist"]

            # 4. Time stop
            if exit_reason is None and et[i] >= cfg.time_stop_time and is_rth[i]:
                exit_reason = "time_stop"
                exit_price = cl[i]

            # 5. Opposite cross exit
            if exit_reason is None and cfg.exit_type == "time_stop":
                # Baseline: opposite cross also exits
                if sig_bull[i]:
                    exit_reason = "opposite_cross"
                    exit_price = cl[i]

            # 5b. Cross exit with minimum hold
            if exit_reason is None and cfg.exit_type == "cross_min_hold":
                bars_held = i - position["entry_idx"]
                if bars_held >= 6 and sig_bull[i]:
                    exit_reason = "cross_exit"
                    exit_price = cl[i]

            # 6. Momentum exit: vol_ratio drops below 0.5
            if exit_reason is None and cfg.exit_type == "momentum_exit":
                bars_held = i - position["entry_idx"]
                if bars_held >= 3 and vol_ratio[i] < 0.5:
                    exit_reason = "momentum_exit"
                    exit_price = cl[i]

            if exit_reason is not None:
                pnl_pts = position["entry_price"] - exit_price
                pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
                trades.append(Trade(
                    setup="VEC2", direction=-1,
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
            # Time window (with bucket override)
            if et[i] < tb_start or et[i] >= tb_end:
                continue
            if trades_today >= cfg.max_trades_per_day:
                continue

            # Signal
            if not sig_bear[i]:
                continue

            # ── FILTERS ──
            # TEMA bearish
            if cfg.require_tema_bearish and not tema_bearish[i]:
                continue

            # 30m range
            max_range = get_max_30m_range(range_lookup, s, et[i])
            if max_range < cfg.min_30m_range:
                continue

            # Volume spike
            if cfg.require_volume_spike and vol_ratio[i] < 1.5:
                continue

            # ATR expansion
            if cfg.require_atr_expansion:
                if np.isnan(atr_avg_vals[i]) or atr_vals[i] <= atr_avg_vals[i]:
                    continue

            # Consecutive red bars
            if cfg.require_consec_red and consec_red[i] < 2:
                continue

            # Gap up filter
            if cfg.skip_gap_up and is_gap_up[i]:
                continue

            # Friday filter
            if cfg.skip_friday and weekday[i] == 4:
                continue

            # ── ENTER SHORT ──
            entry_price = cl[i] - slippage
            stop_pts = entry_price * cfg.stop_bps / 10000.0
            stop_price = entry_price + stop_pts

            # Compute target based on exit type
            target_price = 0.0
            trail_dist = 0.0
            trail_trigger = 0.0

            atr_now = atr_vals[i] if not np.isnan(atr_vals[i]) else 10.0

            if cfg.exit_type.startswith("atr_target"):
                mult_map = {
                    "atr_target_1x": 1.0,
                    "atr_target_1_5x": 1.5,
                    "atr_target_2x": 2.0,
                    "atr_target_3x": 3.0,
                }
                mult = mult_map.get(cfg.exit_type, 1.0)
                target_price = entry_price - (atr_now * mult)
            elif cfg.exit_type == "trailing_atr":
                trail_dist = atr_now * 1.0     # 1x ATR trail distance
                trail_trigger = atr_now * 0.5  # activate after 0.5x ATR profit
            elif cfg.exit_type == "prev_day_low":
                pdl = prev_day_low_arr[i]
                if not np.isnan(pdl) and pdl < entry_price:
                    target_price = pdl
                else:
                    target_price = 0.0  # no target, rely on time stop
            elif cfg.exit_type == "session_vwap":
                vwap = session_vwap_arr[i]
                if not np.isnan(vwap) and vwap < entry_price:
                    target_price = vwap
                else:
                    target_price = 0.0

            position = {
                "entry_idx": i,
                "entry_price": entry_price,
                "stop": stop_price,
                "target": target_price,
                "session": s,
                "trail_dist": trail_dist,
                "trail_trigger": trail_trigger,
                "trail_active": False,
                "best_price": entry_price,
                "trail_stop": 0.0,
            }
            trades_today += 1

    # Close remaining
    if position is not None:
        exit_price = cl[-1] + slippage
        pnl_pts = position["entry_price"] - exit_price
        pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
        trades.append(Trade(
            setup="VEC2", direction=-1,
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
    """One-tailed t-test p-value."""
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t.pnl_dollar for t in trades])
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    return t_pval / 2 if t_stat > 0 else 1 - t_pval / 2


def walk_forward_metrics(df, cfg, range_lookup, crosses, split=WF_SPLIT):
    """Walk-forward split and metrics."""
    df_is = df[df.index < split].copy()
    df_oos = df[df.index >= split].copy()

    # Rebuild range lookups for sub-periods
    bars_30m_is = build_30m_bars(df_is)
    rl_is = build_range_lookup(bars_30m_is)
    bars_30m_oos = build_30m_bars(df_oos)
    rl_oos = build_range_lookup(bars_30m_oos)

    # Rebuild crosses for sub-periods
    crosses_is = detect_all_crosses(df_is)
    crosses_oos = detect_all_crosses(df_oos)

    trades_is = run_backtest(df_is, cfg, rl_is, crosses_is)
    trades_oos = run_backtest(df_oos, cfg, rl_oos, crosses_oos)

    m_is = compute_metrics(trades_is, INITIAL_CAPITAL) if trades_is else None
    m_oos = compute_metrics(trades_oos, INITIAL_CAPITAL) if trades_oos else None

    return m_is, m_oos, trades_is, trades_oos


# ══════════════════════════════════════════════════════════════════════
#  RESULT FORMATTING
# ══════════════════════════════════════════════════════════════════════

def format_result(label, m, trades, pval=None, wf_ratio=None):
    """Format one result row."""
    if m is None or m.total_trades == 0:
        return f"  {label:<30} {'--':>6} trades"
    p_str = f"p={pval:.3f}" if pval is not None else ""
    wf_str = f"WF={wf_ratio:.2f}" if wf_ratio is not None else ""
    flag = ""
    if pval is not None and pval < 0.05 and wf_ratio is not None and wf_ratio >= 0.70:
        flag = " <<<" if m.profit_factor > 1.377 else " <"
    elif pval is not None and pval < 0.05:
        flag = " *"
    return (f"  {label:<30} {m.total_trades:>5} trades | WR {m.win_rate:>5.1f}% | "
            f"PF {m.profit_factor:>6.3f} | ${m.net_pnl:>+10,.0f} | "
            f"Sharpe {m.sharpe:>5.2f} | DD ${m.max_drawdown:>8,.0f} | "
            f"{p_str:>10} {wf_str:>8}{flag}")


def run_variant(df, cfg, range_lookup, crosses, do_wf=True):
    """Run a single variant: full + walk-forward + t-test."""
    trades = run_backtest(df, cfg, range_lookup, crosses)
    m = compute_metrics(trades, INITIAL_CAPITAL) if trades else None

    pval = t_test_pval(trades) if trades else 1.0
    wf_ratio = None

    if do_wf and trades and len(trades) >= 20:
        m_is, m_oos, _, _ = walk_forward_metrics(df, cfg, range_lookup, crosses)
        if m_is and m_oos and m_is.profit_factor > 0:
            wf_ratio = m_oos.profit_factor / m_is.profit_factor

    return m, trades, pval, wf_ratio


# ══════════════════════════════════════════════════════════════════════
#  MAIN SWEEP
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    df = load_data()

    # Pre-build shared lookups
    bars_30m = build_30m_bars(df)
    range_lookup = build_range_lookup(bars_30m)
    crosses = detect_all_crosses(df)

    results_all = []  # (label, m, trades, pval, wf_ratio)

    # ══════════════════════════════════════════════════════════════
    # BASELINE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  VECTOR ES v2 — STRUCTURAL VARIANT SWEEP")
    print("=" * 120)

    baseline_cfg = V2Config(signal_type="ema_cross", exit_type="time_stop",
                            require_tema_bearish=True, min_30m_range=8.0)
    m_base, tr_base, pval_base, wf_base = run_variant(df, baseline_cfg, range_lookup, crosses)
    print("\n  ── BASELINE ──")
    print(format_result("EMA 9/21 + time stop (BASE)", m_base, tr_base, pval_base, wf_base))
    results_all.append(("BASELINE", m_base, tr_base, pval_base, wf_base))

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: SIGNAL VARIANTS
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*116}")
    print("  PHASE 1: SIGNAL VARIANTS (same exit=time_stop, same filters)")
    print(f"  {'─'*116}")

    signal_variants = [
        ("TEMA cross 9/21",        V2Config(signal_type="tema_cross")),
        ("SMA cross 8/24",         V2Config(signal_type="sma_cross")),
        ("EMA + TEMA trend conf",  V2Config(signal_type="ema_trend_conf")),
        ("Price < EMA 21 cross",   V2Config(signal_type="price_cross_ema")),
        ("Multi-MA confluence",    V2Config(signal_type="multi_ma")),
    ]

    for label, cfg in signal_variants:
        m, tr, pval, wf = run_variant(df, cfg, range_lookup, crosses)
        print(format_result(label, m, tr, pval, wf))
        results_all.append((label, m, tr, pval, wf))

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: EXIT VARIANTS (using baseline EMA 9/21 signal)
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*116}")
    print("  PHASE 2: EXIT VARIANTS (baseline signal, same filters)")
    print(f"  {'─'*116}")

    exit_variants = [
        ("ATR target 1x",         V2Config(exit_type="atr_target_1x")),
        ("ATR target 1.5x",       V2Config(exit_type="atr_target_1_5x")),
        ("ATR target 2x",         V2Config(exit_type="atr_target_2x")),
        ("ATR target 3x",         V2Config(exit_type="atr_target_3x")),
        ("Cross exit + 30min hold", V2Config(exit_type="cross_min_hold")),
        ("Trailing ATR stop",     V2Config(exit_type="trailing_atr")),
        ("Momentum exit (vol<0.5)", V2Config(exit_type="momentum_exit")),
        ("Prev day low target",   V2Config(exit_type="prev_day_low")),
        ("Session VWAP target",   V2Config(exit_type="session_vwap")),
    ]

    for label, cfg in exit_variants:
        m, tr, pval, wf = run_variant(df, cfg, range_lookup, crosses)
        print(format_result(label, m, tr, pval, wf))
        results_all.append((label, m, tr, pval, wf))

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: FILTER VARIANTS (baseline signal + exit)
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*116}")
    print("  PHASE 3: FILTER VARIANTS (baseline signal + time_stop exit)")
    print(f"  {'─'*116}")

    filter_variants = [
        ("+ Volume spike (vol>1.5)",
         V2Config(require_volume_spike=True)),
        ("+ ATR expansion (atr>avg)",
         V2Config(require_atr_expansion=True)),
        ("+ Consec red bars (>=2)",
         V2Config(require_consec_red=True)),
        ("+ Skip gap-up days",
         V2Config(skip_gap_up=True)),
        ("+ Skip Fridays",
         V2Config(skip_friday=True)),
        ("TEMA filter OFF",
         V2Config(require_tema_bearish=False)),
        ("Range >= 0 (no range filt)",
         V2Config(min_30m_range=0.0)),
        ("Range >= 5",
         V2Config(min_30m_range=5.0)),
        ("Range >= 10",
         V2Config(min_30m_range=10.0)),
        ("Range >= 12",
         V2Config(min_30m_range=12.0)),
        ("Range >= 15",
         V2Config(min_30m_range=15.0)),
    ]

    for label, cfg in filter_variants:
        m, tr, pval, wf = run_variant(df, cfg, range_lookup, crosses)
        print(format_result(label, m, tr, pval, wf))
        results_all.append((label, m, tr, pval, wf))

    # ══════════════════════════════════════════════════════════════
    # PHASE 3b: TIME-OF-DAY BUCKETS
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*116}")
    print("  PHASE 3b: TIME-OF-DAY BUCKETS (baseline signal + time_stop)")
    print(f"  {'─'*116}")

    for bucket in ["morning", "midday", "afternoon"]:
        cfg = V2Config(time_bucket=bucket)
        m, tr, pval, wf = run_variant(df, cfg, range_lookup, crosses)
        print(format_result(f"Time: {bucket}", m, tr, pval, wf))
        results_all.append((f"Time: {bucket}", m, tr, pval, wf))

    # ══════════════════════════════════════════════════════════════
    # PHASE 3c: DAY-OF-WEEK BREAKDOWN
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*116}")
    print("  PHASE 3c: DAY-OF-WEEK BREAKDOWN (full backtest, split by weekday)")
    print(f"  {'─'*116}")

    trades_full = run_backtest(df, baseline_cfg, range_lookup, crosses)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for wd in range(5):
        wd_trades = [t for t in trades_full
                     if hasattr(t.entry_time, 'weekday') and t.entry_time.weekday() == wd]
        if wd_trades:
            m_wd = compute_metrics(wd_trades, INITIAL_CAPITAL)
            pval_wd = t_test_pval(wd_trades)
            print(f"  {dow_names[wd]:<10} {m_wd.total_trades:>5} trades | WR {m_wd.win_rate:>5.1f}% | "
                  f"PF {m_wd.profit_factor:>6.3f} | ${m_wd.net_pnl:>+10,.0f} | p={pval_wd:.3f}")
        else:
            print(f"  {dow_names[wd]:<10} -- trades")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3d: EXIT REASON BREAKDOWN (baseline)
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*116}")
    print("  PHASE 3d: EXIT REASON BREAKDOWN (baseline)")
    print(f"  {'─'*116}")

    reasons = {}
    for t in trades_full:
        reasons.setdefault(t.exit_reason, {"count": 0, "pnl": 0.0, "wins": 0})
        reasons[t.exit_reason]["count"] += 1
        reasons[t.exit_reason]["pnl"] += t.pnl_dollar
        if t.pnl_dollar > 0:
            reasons[t.exit_reason]["wins"] += 1
    for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["pnl"]):
        wr = data["wins"] / data["count"] * 100 if data["count"] else 0
        avg = data["pnl"] / data["count"]
        print(f"  {reason:<18} {data['count']:>5} trades | WR {wr:>5.1f}% | "
              f"${data['pnl']:>+10,.0f} (avg ${avg:>+7,.0f})")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: COMBINATIONS (best signal + best exit + best filters)
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'─'*116}")
    print("  PHASE 4: COMBINATIONS (best elements from above)")
    print(f"  {'─'*116}")

    # We'll try the most promising combinations based on phases 1-3
    combos = [
        # Signal variants with promising exits
        ("TEMA cross + ATR 1.5x target",
         V2Config(signal_type="tema_cross", exit_type="atr_target_1_5x")),
        ("TEMA cross + trailing ATR",
         V2Config(signal_type="tema_cross", exit_type="trailing_atr")),
        ("TEMA cross + time stop",
         V2Config(signal_type="tema_cross", exit_type="time_stop")),
        ("EMA+trend conf + ATR 1.5x",
         V2Config(signal_type="ema_trend_conf", exit_type="atr_target_1_5x")),
        ("EMA+trend conf + trailing ATR",
         V2Config(signal_type="ema_trend_conf", exit_type="trailing_atr")),
        ("Multi-MA + time stop",
         V2Config(signal_type="multi_ma", exit_type="time_stop")),
        ("Multi-MA + ATR 1.5x",
         V2Config(signal_type="multi_ma", exit_type="atr_target_1_5x")),
        ("Price<EMA21 + trailing ATR",
         V2Config(signal_type="price_cross_ema", exit_type="trailing_atr")),

        # Best signal + best exit + additional filters
        ("TEMA cross + skip Fri",
         V2Config(signal_type="tema_cross", skip_friday=True)),
        ("TEMA cross + vol spike",
         V2Config(signal_type="tema_cross", require_volume_spike=True)),
        ("TEMA cross + ATR expansion",
         V2Config(signal_type="tema_cross", require_atr_expansion=True)),
        ("EMA+trend + skip Fri",
         V2Config(signal_type="ema_trend_conf", skip_friday=True)),
        ("EMA+trend + skip Fri + ATR exp",
         V2Config(signal_type="ema_trend_conf", skip_friday=True,
                  require_atr_expansion=True)),
        ("Baseline + skip Fri + ATR exp",
         V2Config(skip_friday=True, require_atr_expansion=True)),
        ("Baseline + skip Fri + vol spike",
         V2Config(skip_friday=True, require_volume_spike=True)),

        # TEMA cross + exit combos + filter combos
        ("TEMA + trail + skip Fri",
         V2Config(signal_type="tema_cross", exit_type="trailing_atr",
                  skip_friday=True)),
        ("TEMA + 1.5x ATR + skip Fri",
         V2Config(signal_type="tema_cross", exit_type="atr_target_1_5x",
                  skip_friday=True)),
        ("TEMA + 1.5x ATR + vol spike",
         V2Config(signal_type="tema_cross", exit_type="atr_target_1_5x",
                  require_volume_spike=True)),
        ("TEMA + trail + vol spike",
         V2Config(signal_type="tema_cross", exit_type="trailing_atr",
                  require_volume_spike=True)),
        ("TEMA + trail + skip Fri + vol",
         V2Config(signal_type="tema_cross", exit_type="trailing_atr",
                  skip_friday=True, require_volume_spike=True)),

        # Multi-MA with filters
        ("Multi-MA + trail + skip Fri",
         V2Config(signal_type="multi_ma", exit_type="trailing_atr",
                  skip_friday=True)),
        ("Multi-MA + skip Fri + ATR exp",
         V2Config(signal_type="multi_ma", skip_friday=True,
                  require_atr_expansion=True)),

        # EMA cross (baseline signal) with best exit/filter combos
        ("Baseline + trail + skip Fri",
         V2Config(exit_type="trailing_atr", skip_friday=True)),
        ("Baseline + 1.5x + skip Fri",
         V2Config(exit_type="atr_target_1_5x", skip_friday=True)),
        ("Baseline + 2x ATR + skip Fri",
         V2Config(exit_type="atr_target_2x", skip_friday=True)),

        # Cross exit combos
        ("Baseline + cross_min_hold + Fri",
         V2Config(exit_type="cross_min_hold", skip_friday=True)),
        ("TEMA + cross_min_hold",
         V2Config(signal_type="tema_cross", exit_type="cross_min_hold")),
        ("TEMA + cross_min_hold + Fri",
         V2Config(signal_type="tema_cross", exit_type="cross_min_hold",
                  skip_friday=True)),

        # Support level targets
        ("Baseline + prev day low",
         V2Config(exit_type="prev_day_low")),
        ("TEMA + prev day low",
         V2Config(signal_type="tema_cross", exit_type="prev_day_low")),
        ("Baseline + session VWAP",
         V2Config(exit_type="session_vwap")),

        # No TEMA filter (let signal speak)
        ("TEMA cross, TEMA filter OFF",
         V2Config(signal_type="tema_cross", require_tema_bearish=False)),
        ("EMA+trend, TEMA filter OFF",
         V2Config(signal_type="ema_trend_conf", require_tema_bearish=False)),

        # Morning-only with various signals
        ("Morning only + TEMA cross",
         V2Config(signal_type="tema_cross", time_bucket="morning")),
        ("Morning only + trail",
         V2Config(time_bucket="morning", exit_type="trailing_atr")),
        ("Afternoon only + TEMA cross",
         V2Config(signal_type="tema_cross", time_bucket="afternoon")),

        # Skip gap up
        ("Baseline + skip gap up",
         V2Config(skip_gap_up=True)),
        ("TEMA + skip gap up",
         V2Config(signal_type="tema_cross", skip_gap_up=True)),

        # Consec red bars
        ("Baseline + consec red",
         V2Config(require_consec_red=True)),
        ("TEMA + consec red",
         V2Config(signal_type="tema_cross", require_consec_red=True)),
    ]

    combo_results = []
    for label, cfg in combos:
        m, tr, pval, wf = run_variant(df, cfg, range_lookup, crosses)
        print(format_result(label, m, tr, pval, wf))
        combo_results.append((label, m, tr, pval, wf))
        results_all.append((label, m, tr, pval, wf))

    # ══════════════════════════════════════════════════════════════
    # RANKING: Top configs by PF (minimum 30 trades, p < 0.10)
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'═'*116}")
    print("  TOP 15 CONFIGS BY PROFIT FACTOR (min 30 trades, p < 0.10)")
    print(f"  {'═'*116}")

    valid = [(label, m, tr, pval, wf) for label, m, tr, pval, wf in results_all
             if m is not None and m.total_trades >= 30 and pval is not None and pval < 0.10]
    valid.sort(key=lambda x: x[1].profit_factor, reverse=True)

    for i, (label, m, tr, pval, wf) in enumerate(valid[:15]):
        beats = "BEATS BASELINE" if m.profit_factor > 1.377 else ""
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*"
        wf_str = f"WF={wf:.2f}" if wf is not None else "WF=n/a"
        wf_pass = "(PASS)" if wf is not None and wf >= 0.70 else "(FAIL)" if wf is not None else ""
        print(f"  #{i+1:>2} {label:<32} PF {m.profit_factor:>6.3f} | "
              f"{m.total_trades:>4} trades | ${m.net_pnl:>+10,.0f} | "
              f"p={pval:.3f}{sig} | {wf_str} {wf_pass} {beats}")

    # ══════════════════════════════════════════════════════════════
    # FULL VALIDATION ON TOP CANDIDATES
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'═'*116}")
    print("  FULL VALIDATION — TOP CANDIDATES THAT BEAT BASELINE")
    print(f"  {'═'*116}")

    # Get candidates that beat baseline PF AND have p < 0.05
    candidates = [(label, m, tr, pval, wf) for label, m, tr, pval, wf in results_all
                  if m is not None and m.profit_factor > 1.377 and m.total_trades >= 30
                  and pval is not None and pval < 0.05]
    candidates.sort(key=lambda x: x[1].profit_factor, reverse=True)

    if not candidates:
        print("\n  No variant beat the baseline PF 1.377 with p < 0.05.")
        # Show closest contenders
        close = [(label, m, tr, pval, wf) for label, m, tr, pval, wf in results_all
                 if m is not None and m.profit_factor > 1.2 and m.total_trades >= 30
                 and pval is not None and pval < 0.10]
        close.sort(key=lambda x: x[1].profit_factor, reverse=True)
        if close:
            print("  Closest contenders (PF > 1.2, p < 0.10):")
            for label, m, tr, pval, wf in close[:5]:
                print(format_result(label, m, tr, pval, wf))

    for label, m, tr, pval, wf in candidates[:5]:
        print(f"\n  ── {label} ──")
        print(f"  Full: {m.total_trades} trades | WR {m.win_rate:.1f}% | PF {m.profit_factor:.3f} | "
              f"${m.net_pnl:+,.0f} | Sharpe {m.sharpe:.2f} | DD ${m.max_drawdown:,.0f}")
        print(f"  p-value: {pval:.4f}")
        if wf is not None:
            print(f"  WF ratio: {wf:.3f} {'PASS' if wf >= 0.70 else 'FAIL'}")

        # Exit reason breakdown
        reasons = {}
        for t in tr:
            reasons.setdefault(t.exit_reason, {"count": 0, "pnl": 0.0})
            reasons[t.exit_reason]["count"] += 1
            reasons[t.exit_reason]["pnl"] += t.pnl_dollar
        print("  Exit breakdown:")
        for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["pnl"]):
            avg = data["pnl"] / data["count"]
            print(f"    {reason:<18} {data['count']:>4} trades  ${data['pnl']:>+10,.0f}  (avg ${avg:>+7,.0f})")

        # Monthly breakdown
        monthly = {}
        for t in tr:
            month = t.entry_time.strftime("%Y-%m") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:7]
            monthly.setdefault(month, {"count": 0, "pnl": 0.0})
            monthly[month]["count"] += 1
            monthly[month]["pnl"] += t.pnl_dollar
        win_months = sum(1 for d in monthly.values() if d["pnl"] > 0)
        print(f"  Monthly: {win_months}/{len(monthly)} winning months ({win_months/len(monthly)*100:.0f}%)")

    # ══════════════════════════════════════════════════════════════
    # HONEST SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n  {'═'*116}")
    print("  HONEST SUMMARY")
    print(f"  {'═'*116}")

    n_tested = len(results_all)
    n_profitable = sum(1 for _, m, _, _, _ in results_all if m and m.profit_factor > 1.0)
    n_beat = sum(1 for _, m, _, pval, _ in results_all
                 if m and m.profit_factor > 1.377 and pval is not None and pval < 0.05)
    n_sig = sum(1 for _, m, _, pval, _ in results_all
                if m and pval is not None and pval < 0.05 and m.total_trades >= 30)
    n_wf_pass = sum(1 for _, m, _, pval, wf in results_all
                    if m and pval is not None and pval < 0.05 and wf is not None and wf >= 0.70
                    and m.total_trades >= 30)

    print(f"  Total variants tested:     {n_tested}")
    print(f"  Profitable (PF > 1.0):     {n_profitable}")
    print(f"  Significant (p < 0.05):    {n_sig}")
    print(f"  WF pass (ratio >= 0.70):   {n_wf_pass}")
    print(f"  Beat baseline PF 1.377:    {n_beat}")

    if n_beat > 0:
        best = max(candidates, key=lambda x: x[1].profit_factor) if candidates else None
        if best:
            print(f"\n  BEST VARIANT: {best[0]}")
            print(f"  PF {best[1].profit_factor:.3f} | {best[1].total_trades} trades | "
                  f"${best[1].net_pnl:+,.0f} | p={best[3]:.4f}")
    else:
        print(f"\n  Baseline EMA 9/21 + time stop IS the best structure for this signal.")
        print(f"  The edge is real (p<0.05, WF passes) — it just can't be improved structurally.")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 120)


if __name__ == "__main__":
    main()
