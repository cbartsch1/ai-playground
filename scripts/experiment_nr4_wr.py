#!/usr/bin/env python3
"""Experiment: NR4/WR Prior Day Range Classification.

Hypothesis: After wide-range days, rejection setups thrive (levels proven strong).
After narrow-range days (NR4), breakouts explode (coiled energy releases).

Classifies each trading day by PRIOR day's full RTH range relative to 20-day EMA.
Splits v8.1 and v13 trades by prior-day classification.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pandas as pd
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
from backtester.session import SessionState, update_session
from backtester.indicators import compute_indicators
from backtester.setups import level_rejection

DATA = "data/es_5m_databento_2yr.csv"
SPLIT_DATE = pd.Timestamp("2025-02-14", tz="America/New_York")


def v81_config():
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_ib_break = True
    cfg.use_trend_filter = True
    cfg.use_ib_reject = True
    cfg.rej_wide_only = True
    cfg.rej_target = "ib_low"
    cfg.rej_stop_buffer = 8.0
    cfg.max_rej_trades = 8
    return cfg


def v13_config():
    cfg = v81_config()
    cfg.use_level_reject = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = False
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 7.0
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_rr = 0.5
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_target_skip = 2
    return cfg


def build_day_type_map(df, cfg):
    """Run through data bar-by-bar to build a map of session_date -> prior day type.

    Uses the new prev_day_is_wide / prev_day_is_narrow from SessionState.
    """
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    prev_bar = None
    day_types = {}

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)
        level_rejection.update_level_state(bar, state, cfg)
        level_rejection.update_support_level_state(bar, state, cfg)

        # At start of each RTH session, record the prior day type
        if bar["new_rth"]:
            session_date = bar.get("session_date", idx.date() if hasattr(idx, 'date') else None)
            if session_date:
                if state.prev_day_is_wide:
                    day_types[session_date] = "wide"
                elif state.prev_day_is_narrow:
                    day_types[session_date] = "narrow"
                else:
                    day_types[session_date] = "normal"

        prev_bar = bar

    return day_types


def print_metrics(label, m):
    print(f"  {label}: {m.total_trades} trades, PF {m.profit_factor:.2f}, "
          f"WR {m.win_rate:.1f}%, ${m.net_pnl:+,.0f}, Sharpe {m.sharpe:.2f}")


def run_experiment(df, cfg, label):
    """Run backtest and split results by prior day type."""
    print(f"\n{'='*70}")
    print(f"  {label} — NR4/WR Day Classification")
    print(f"{'='*70}")

    # Build day type map
    print("  Building prior-day classification map...")
    day_types = build_day_type_map(df.copy(), cfg)
    counts = {}
    for dt in day_types.values():
        counts[dt] = counts.get(dt, 0) + 1
    print(f"  Day types: {counts}")

    # Run full backtest
    all_trades = run_backtest(df.copy(), cfg)
    print(f"  Total trades: {len(all_trades)}")

    # Split trades by prior day type
    buckets = {"wide": [], "normal": [], "narrow": [], "unknown": []}
    for t in all_trades:
        trade_date = t.entry_time.date() if hasattr(t.entry_time, 'date') else None
        if trade_date and trade_date in day_types:
            buckets[day_types[trade_date]].append(t)
        else:
            buckets["unknown"].append(t)

    print(f"\n  --- Full 2-Year Split by Prior Day Type ---")
    for dtype in ["wide", "normal", "narrow"]:
        trades = buckets[dtype]
        if trades:
            m = compute_metrics(trades)
            print_metrics(f"After {dtype.upper():>7} day", m)
        else:
            print(f"  After {dtype.upper():>7} day: no trades")

    if buckets["unknown"]:
        print(f"  Unknown: {len(buckets['unknown'])} trades (first few days, no prior data)")

    # IS/OOS split for the best-performing day type
    print(f"\n  --- IS/OOS Validation ---")
    df_is = df[df.index < SPLIT_DATE].copy()
    df_oos = df[df.index >= SPLIT_DATE].copy()

    trades_is = run_backtest(df_is, cfg)
    trades_oos = run_backtest(df_oos, cfg)

    for dtype in ["wide", "normal", "narrow"]:
        is_bucket = [t for t in trades_is
                     if hasattr(t.entry_time, 'date') and t.entry_time.date() in day_types
                     and day_types[t.entry_time.date()] == dtype]
        oos_bucket = [t for t in trades_oos
                      if hasattr(t.entry_time, 'date') and t.entry_time.date() in day_types
                      and day_types[t.entry_time.date()] == dtype]

        if is_bucket and oos_bucket:
            m_is = compute_metrics(is_bucket)
            m_oos = compute_metrics(oos_bucket)
            print(f"\n  After {dtype.upper()} day:")
            print_metrics("  IS ", m_is)
            print_metrics("  OOS", m_oos)
            if m_is.profit_factor > 0:
                print(f"    PF Ratio: {m_oos.profit_factor / m_is.profit_factor:.2f}")
        else:
            print(f"\n  After {dtype.upper()} day: insufficient IS or OOS trades")

    # Specific hypothesis test: "After wide day, only rejections"
    rej_after_wide = [t for t in all_trades
                      if ("REJ" in t.setup or "LVL" in t.setup)
                      and hasattr(t.entry_time, 'date')
                      and t.entry_time.date() in day_types
                      and day_types[t.entry_time.date()] == "wide"]
    brk_after_narrow = [t for t in all_trades
                        if "IB" in t.setup and "REJ" not in t.setup and "LVL" not in t.setup
                        and hasattr(t.entry_time, 'date')
                        and t.entry_time.date() in day_types
                        and day_types[t.entry_time.date()] == "narrow"]

    print(f"\n  --- Hypothesis Tests ---")
    if rej_after_wide:
        m = compute_metrics(rej_after_wide)
        print_metrics("Rejections after WIDE day", m)
    else:
        print("  No rejection trades after wide days")

    if brk_after_narrow:
        m = compute_metrics(brk_after_narrow)
        print_metrics("IB Breakouts after NARROW day", m)
    else:
        print("  No IB breakout trades after narrow days")


def main():
    print("Loading data...")
    df = load_tos_csv(DATA, instrument="ES")
    print(f"Loaded {len(df)} bars")

    run_experiment(df.copy(), v81_config(), "v8.1 (IB Breakout + Rejection)")
    run_experiment(df.copy(), v13_config(), "v13 (ONH Level Rejection)")

    print(f"\n{'='*70}")
    print("  NR4/WR Experiment Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
