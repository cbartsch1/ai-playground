#!/usr/bin/env python3
"""Experiment: VWAP Filter — shorts only above VWAP.

Hypothesis: Shorting below VWAP fights institutional flow. Filtering to shorts-only-above-VWAP
should improve win rate and PF by aligning with dealer selling pressure.

Tests both v8.1 (IB Breakout + Rejection) and v13 (Level Rejection).
Runs IS/OOS split for walk-forward validation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics

DATA = "data/es_5m_databento_2yr.csv"
SPLIT_DATE = pd.Timestamp("2025-02-14", tz="America/New_York")


def v81_config():
    """v8.1 baseline config."""
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
    """v13 ONH Level Rejection config."""
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
    cfg.lvl_target_skip = 2  # 3rd support (C3)
    return cfg


def print_metrics(label, m):
    print(f"  {label}: {m.total_trades} trades, PF {m.profit_factor:.2f}, "
          f"WR {m.win_rate:.1f}%, ${m.net_pnl:+,.0f}, Sharpe {m.sharpe:.2f}")


def run_experiment(df, cfg, label):
    """Run baseline vs VWAP-filtered, IS/OOS split."""
    df_is = df[df.index < SPLIT_DATE].copy()
    df_oos = df[df.index >= SPLIT_DATE].copy()

    # Baseline
    trades_is = run_backtest(df_is, cfg)
    trades_oos = run_backtest(df_oos, cfg)
    m_is = compute_metrics(trades_is)
    m_oos = compute_metrics(trades_oos)

    # VWAP-filtered
    cfg_vwap = StrategyConfig(**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()})
    cfg_vwap.use_vwap_filter = True
    trades_is_v = run_backtest(df_is, cfg_vwap)
    trades_oos_v = run_backtest(df_oos, cfg_vwap)
    m_is_v = compute_metrics(trades_is_v)
    m_oos_v = compute_metrics(trades_oos_v)

    print(f"\n{'='*70}")
    print(f"  {label} — VWAP Filter Test")
    print(f"{'='*70}")
    print(f"\n  BASELINE (no VWAP filter):")
    print_metrics("IS ", m_is)
    print_metrics("OOS", m_oos)
    if m_is.profit_factor > 0:
        print(f"  PF Ratio: {m_oos.profit_factor / m_is.profit_factor:.2f}")

    print(f"\n  WITH VWAP FILTER (shorts only above VWAP):")
    print_metrics("IS ", m_is_v)
    print_metrics("OOS", m_oos_v)
    if m_is_v.profit_factor > 0:
        print(f"  PF Ratio: {m_oos_v.profit_factor / m_is_v.profit_factor:.2f}")

    # Delta
    print(f"\n  DELTA (VWAP - Baseline):")
    print(f"  IS  trades: {m_is_v.total_trades - m_is.total_trades:+d}, "
          f"PF: {m_is_v.profit_factor - m_is.profit_factor:+.2f}, "
          f"P&L: ${m_is_v.net_pnl - m_is.net_pnl:+,.0f}")
    print(f"  OOS trades: {m_oos_v.total_trades - m_oos.total_trades:+d}, "
          f"PF: {m_oos_v.profit_factor - m_oos.profit_factor:+.2f}, "
          f"P&L: ${m_oos_v.net_pnl - m_oos.net_pnl:+,.0f}")

    # Verdict
    pf_improved_is = m_is_v.profit_factor > m_is.profit_factor
    pf_improved_oos = m_oos_v.profit_factor > m_oos.profit_factor
    if pf_improved_is and pf_improved_oos:
        print(f"\n  VERDICT: CONFIRMED — VWAP filter improves both IS and OOS")
    elif pf_improved_oos:
        print(f"\n  VERDICT: PROMISING — OOS improves but IS degrades (possible overfitting removal)")
    elif pf_improved_is:
        print(f"\n  VERDICT: FAILED — IS improves but OOS degrades (curve fit)")
    else:
        print(f"\n  VERDICT: FAILED — both degrade")


def main():
    print("Loading data...")
    df = load_tos_csv(DATA, instrument="ES")
    print(f"Loaded {len(df)} bars\n")

    # Test v8.1
    run_experiment(df.copy(), v81_config(), "v8.1 (IB Breakout + Rejection)")

    # Test v13 (LVL only)
    run_experiment(df.copy(), v13_config(), "v13 (ONH Level Rejection)")

    print(f"\n{'='*70}")
    print("  VWAP Filter Experiment Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
