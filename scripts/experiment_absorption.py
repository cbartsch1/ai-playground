#!/usr/bin/env python3
"""Experiment: Absorption Proxy Filter for ONH Level Rejection.

Hypothesis: ONH rejections preceded by 3+ bars at the level with decreasing
range and elevated volume (absorption pattern) are higher quality signals.
Institutional defenders absorbing sell orders = level more likely to hold.

Tests v13 (ONH Level Rejection, 3x C3) with absorption filter ON vs OFF.
Sweeps min_bars (2-5) and vol_mult (0.5-2.0) to find optimal settings.
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


def v13_config():
    """v13 ONH Level Rejection config (baseline)."""
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


def print_metrics(label, m, trades):
    pnl = sum(t.pnl_dollar for t in trades)
    print(f"  {label}: {m.total_trades:>4} trades, PF {m.profit_factor:>5.2f}, "
          f"WR {m.win_rate:>5.1f}%, ${pnl:>+10,.0f}")


def run_comparison(df, label, cfg_baseline, cfg_absorption):
    """Run baseline vs absorption-filtered and show comparison."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Full 2yr
    trades_base = run_backtest(df.copy(), cfg_baseline)
    trades_abs = run_backtest(df.copy(), cfg_absorption)

    m_base = compute_metrics(trades_base)
    m_abs = compute_metrics(trades_abs)

    print(f"\n  --- Full 2-Year ---")
    print_metrics("Baseline     ", m_base, trades_base)
    print_metrics("+ Absorption ", m_abs, trades_abs)

    # Split by setup
    for setup_name in ["IB", "REJ", "LVL_ONH"]:
        t_base = [t for t in trades_base if t.setup == setup_name]
        t_abs = [t for t in trades_abs if t.setup == setup_name]
        if t_base:
            m_b = compute_metrics(t_base) if len(t_base) >= 3 else None
            m_a = compute_metrics(t_abs) if len(t_abs) >= 3 else None
            pnl_b = sum(t.pnl_dollar for t in t_base)
            pnl_a = sum(t.pnl_dollar for t in t_abs)
            if m_b:
                print(f"\n  {setup_name} setup:")
                print(f"    Baseline:    {len(t_base):>4} trades, PF {m_b.profit_factor:.2f}, ${pnl_b:>+10,.0f}")
            if m_a and len(t_abs) >= 3:
                print(f"    + Absorption: {len(t_abs):>4} trades, PF {m_a.profit_factor:.2f}, ${pnl_a:>+10,.0f}")
            elif t_abs:
                print(f"    + Absorption: {len(t_abs):>4} trades (too few for metrics)")

    # IS/OOS split
    print(f"\n  --- IS/OOS Split ---")
    df_is = df[df.index < SPLIT_DATE].copy()
    df_oos = df[df.index >= SPLIT_DATE].copy()

    for period_label, period_df in [("IS ", df_is), ("OOS", df_oos)]:
        t_b = run_backtest(period_df.copy(), cfg_baseline)
        t_a = run_backtest(period_df.copy(), cfg_absorption)
        if t_b:
            m_b = compute_metrics(t_b)
            pnl_b = sum(t.pnl_dollar for t in t_b)
            print(f"    {period_label} Baseline:    {m_b.total_trades:>4} trades, PF {m_b.profit_factor:.2f}, ${pnl_b:>+10,.0f}")
        if t_a:
            m_a = compute_metrics(t_a)
            pnl_a = sum(t.pnl_dollar for t in t_a)
            print(f"    {period_label} + Absorption: {m_a.total_trades:>4} trades, PF {m_a.profit_factor:.2f}, ${pnl_a:>+10,.0f}")
        else:
            print(f"    {period_label} + Absorption:    0 trades")

    return trades_base, trades_abs


def main():
    print("Loading ES data...")
    df = load_tos_csv(DATA, instrument="ES")
    print(f"Loaded {len(df)} bars")

    # ── Baseline ──
    cfg_base = v13_config()

    # ── Absorption ON (default params: 3 bars, 1.0x volume) ──
    cfg_abs = v13_config()
    cfg_abs.lvl_use_absorption = True
    cfg_abs.lvl_absorption_min_bars = 3
    cfg_abs.lvl_absorption_vol_mult = 1.0

    run_comparison(df, "v13 Baseline vs Absorption (3 bars, 1.0x vol)", cfg_base, cfg_abs)

    # ── Parameter Sweep: min_bars ──
    print(f"\n{'='*70}")
    print(f"  Parameter Sweep: min_bars (vol_mult=1.0)")
    print(f"{'='*70}")
    print(f"  {'min_bars':>8} {'Trades':>7} {'LVL Trades':>11} {'PF':>7} {'WR':>7} {'P&L':>12}")
    print(f"  {'-'*55}")

    for min_bars in [2, 3, 4, 5]:
        cfg = v13_config()
        cfg.lvl_use_absorption = True
        cfg.lvl_absorption_min_bars = min_bars
        cfg.lvl_absorption_vol_mult = 1.0
        trades = run_backtest(df.copy(), cfg)
        m = compute_metrics(trades)
        pnl = sum(t.pnl_dollar for t in trades)
        lvl_trades = [t for t in trades if t.setup.startswith("LVL")]
        print(f"  {min_bars:>8} {m.total_trades:>7} {len(lvl_trades):>11} {m.profit_factor:>7.2f} "
              f"{m.win_rate:>6.1f}% ${pnl:>+10,.0f}")

    # Baseline for comparison
    trades_base = run_backtest(df.copy(), cfg_base)
    m_base = compute_metrics(trades_base)
    pnl_base = sum(t.pnl_dollar for t in trades_base)
    lvl_base = [t for t in trades_base if t.setup.startswith("LVL")]
    print(f"  {'OFF':>8} {m_base.total_trades:>7} {len(lvl_base):>11} {m_base.profit_factor:>7.2f} "
          f"{m_base.win_rate:>6.1f}% ${pnl_base:>+10,.0f}")

    # ── Parameter Sweep: vol_mult ──
    print(f"\n{'='*70}")
    print(f"  Parameter Sweep: vol_mult (min_bars=3)")
    print(f"{'='*70}")
    print(f"  {'vol_mult':>8} {'Trades':>7} {'LVL Trades':>11} {'PF':>7} {'WR':>7} {'P&L':>12}")
    print(f"  {'-'*55}")

    for vol_mult in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        cfg = v13_config()
        cfg.lvl_use_absorption = True
        cfg.lvl_absorption_min_bars = 3
        cfg.lvl_absorption_vol_mult = vol_mult
        trades = run_backtest(df.copy(), cfg)
        m = compute_metrics(trades)
        pnl = sum(t.pnl_dollar for t in trades)
        lvl_trades = [t for t in trades if t.setup.startswith("LVL")]
        print(f"  {vol_mult:>8.1f} {m.total_trades:>7} {len(lvl_trades):>11} {m.profit_factor:>7.2f} "
              f"{m.win_rate:>6.1f}% ${pnl:>+10,.0f}")

    print(f"  {'OFF':>8} {m_base.total_trades:>7} {len(lvl_base):>11} {m_base.profit_factor:>7.2f} "
          f"{m_base.win_rate:>6.1f}% ${pnl_base:>+10,.0f}")

    # ── Best config IS/OOS validation ──
    # (We'll print all combos and the user can assess which is best)
    print(f"\n{'='*70}")
    print(f"  IS/OOS Validation: All Combos")
    print(f"{'='*70}")
    print(f"  {'Config':>20} {'IS PF':>7} {'IS P&L':>10} {'OOS PF':>7} {'OOS P&L':>10} {'PF Ratio':>9}")
    print(f"  {'-'*65}")

    df_is = df[df.index < SPLIT_DATE].copy()
    df_oos = df[df.index >= SPLIT_DATE].copy()

    combos = [
        ("OFF", False, 3, 1.0),
        ("3 bars, 0.0x", True, 3, 0.0),
        ("3 bars, 0.5x", True, 3, 0.5),
        ("3 bars, 1.0x", True, 3, 1.0),
        ("3 bars, 1.5x", True, 3, 1.5),
        ("2 bars, 1.0x", True, 2, 1.0),
        ("4 bars, 1.0x", True, 4, 1.0),
    ]

    for label, use_abs, min_bars, vol_mult in combos:
        cfg = v13_config()
        cfg.lvl_use_absorption = use_abs
        cfg.lvl_absorption_min_bars = min_bars
        cfg.lvl_absorption_vol_mult = vol_mult

        t_is = run_backtest(df_is.copy(), cfg)
        t_oos = run_backtest(df_oos.copy(), cfg)

        if t_is and t_oos:
            m_is = compute_metrics(t_is)
            m_oos = compute_metrics(t_oos)
            pnl_is = sum(t.pnl_dollar for t in t_is)
            pnl_oos = sum(t.pnl_dollar for t in t_oos)
            pf_ratio = m_oos.profit_factor / m_is.profit_factor if m_is.profit_factor > 0 else 0
            print(f"  {label:>20} {m_is.profit_factor:>7.2f} ${pnl_is:>+9,.0f} "
                  f"{m_oos.profit_factor:>7.2f} ${pnl_oos:>+9,.0f} {pf_ratio:>9.2f}")
        else:
            print(f"  {label:>20} — insufficient data")

    print(f"\n{'='*70}")
    print("  Absorption Proxy Experiment Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
