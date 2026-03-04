#!/usr/bin/env python3
"""Market Structure strategy — Dalton's setups + SMA 8/24 timing.

Tests the MS setup at all structural levels with real VP-derived levels.

Variants:
  1. MS Default (all levels, SMA 8/24, real VP)
  2. MS + TEMA (same but with TEMA 9/21 instead of SMA)
  3. MS Prev VA Only (only prev day VAH/VAL — pure VA Fade test)
  4. MS + Entry Lag (1-bar MA confirmation delay)
  5. v13 Baseline (reference)
  6. MS + v13 Combined

Usage:
    python scripts/compare_ms_strategy.py data/es_5m_databento_2yr.csv
"""

import argparse
import sys
import os

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.stagger_engine import run_backtest_stagger
from backtester.metrics import compute_metrics, per_setup_breakdown


def make_ms_default():
    """MS with all levels, SMA 8/24, real VP, 68% VA."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 5.0
    cfg.ms_min_target_pts = 4.0
    cfg.ms_min_rr = 0.5
    cfg.ms_max_risk = 15.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = True
    cfg.ms_use_dev_va = True
    cfg.ms_use_poc = True
    return cfg


def make_ms_tema():
    """MS with TEMA 9/21 instead of SMA 8/24."""
    cfg = make_ms_default()
    cfg.ms_ma_type = "tema"
    return cfg


def make_ms_prev_va_only():
    """MS at prev day VA only — pure VA Fade test with real VP."""
    cfg = make_ms_default()
    cfg.ms_use_on_levels = False
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False
    return cfg


def make_ms_with_lag():
    """MS with 1-bar entry lag (algo shakeout protection)."""
    cfg = make_ms_default()
    cfg.ms_ma_confirm_bars = 1
    return cfg


def make_v13_config():
    """v13 Level Rejection baseline."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = True
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = True
    cfg.lvl_ma_filter = "tema"
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 8.0
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_min_rr = 0.5
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3
    cfg.lvl_ibh_wide_only = True
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_ms = False
    return cfg


def make_ms_plus_v13():
    """v13 + MS combined."""
    cfg = make_v13_config()
    cfg.direction_filter = "both"
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 5.0
    cfg.ms_min_target_pts = 4.0
    cfg.ms_min_rr = 0.5
    cfg.ms_max_risk = 15.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = True
    cfg.ms_use_dev_va = True
    cfg.ms_use_poc = True
    return cfg


def print_metrics(label, trades, show_detail=False):
    if not trades:
        print(f"  {label:<40s}  NO TRADES")
        return None

    m = compute_metrics(trades)
    print(f"  {label:<40s}  {m.total_trades:>5d} trades  "
          f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+10,.0f}  DD ${m.max_drawdown:>8,.0f}  "
          f"Sharpe {m.sharpe:>5.2f}  "
          f"Trd/Day {m.trades_per_day:>4.1f}")

    if show_detail and trades:
        breakdown = per_setup_breakdown(trades)
        for setup, sm in sorted(breakdown.items()):
            print(f"    {setup:<20s}  {sm.total_trades:>5d}  "
                  f"WR {sm.win_rate:>5.1f}%  PF {sm.profit_factor:>6.3f}  "
                  f"P&L ${sm.net_pnl:>+10,.0f}")

        # Direction breakdown
        longs = [t for t in trades if t.direction == 1]
        shorts = [t for t in trades if t.direction == -1]
        if longs:
            lm = compute_metrics(longs)
            print(f"    {'LONGS':<20s}  {lm.total_trades:>5d}  "
                  f"WR {lm.win_rate:>5.1f}%  PF {lm.profit_factor:>6.3f}  "
                  f"P&L ${lm.net_pnl:>+10,.0f}")
        if shorts:
            sm_s = compute_metrics(shorts)
            print(f"    {'SHORTS':<20s}  {sm_s.total_trades:>5d}  "
                  f"WR {sm_s.win_rate:>5.1f}%  PF {sm_s.profit_factor:>6.3f}  "
                  f"P&L ${sm_s.net_pnl:>+10,.0f}")

        # Exit reasons
        reasons = {}
        for t in trades:
            r = t.exit_reason
            if r not in reasons:
                reasons[r] = {"count": 0, "pnl": 0}
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t.pnl_dollar
        print(f"    Exits:")
        for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            print(f"      {r:<12s}  {d['count']:>4d}  ${d['pnl']:>+10,.0f}")

    return m


def run_comparison(df, label="FULL 2-YEAR"):
    print(f"\n{'='*110}")
    print(f"  {label}")
    print(f"{'='*110}")

    variants = []

    # 1. v13 Baseline
    cfg1 = make_v13_config()
    trades1 = run_backtest_stagger(df.copy(), cfg1, n_contracts=3, uniform_skip=2)
    m1 = print_metrics("v13 Baseline (3x stagger)", trades1)
    variants.append(("v13 Baseline", m1, trades1))

    # 2. MS Default (all levels, SMA 8/24)
    cfg2 = make_ms_default()
    trades2 = run_backtest(df.copy(), cfg2)
    m2 = print_metrics("MS Default (SMA 8/24, all levels)", trades2, show_detail=True)
    variants.append(("MS Default", m2, trades2))

    # 3. MS + TEMA
    cfg3 = make_ms_tema()
    trades3 = run_backtest(df.copy(), cfg3)
    m3 = print_metrics("MS + TEMA 9/21", trades3, show_detail=True)
    variants.append(("MS TEMA", m3, trades3))

    # 4. MS Prev VA Only (pure VA Fade)
    cfg4 = make_ms_prev_va_only()
    trades4 = run_backtest(df.copy(), cfg4)
    m4 = print_metrics("MS Prev VA Only (VA Fade)", trades4, show_detail=True)
    variants.append(("MS VA Only", m4, trades4))

    # 5. MS + Entry Lag
    cfg5 = make_ms_with_lag()
    trades5 = run_backtest(df.copy(), cfg5)
    m5 = print_metrics("MS + Entry Lag (1 bar)", trades5, show_detail=True)
    variants.append(("MS + Lag", m5, trades5))

    # 6. MS + v13 Combined
    cfg6 = make_ms_plus_v13()
    trades6 = run_backtest(df.copy(), cfg6)
    m6 = print_metrics("v13 + MS Combined", trades6, show_detail=True)
    variants.append(("v13 + MS", m6, trades6))

    # Summary
    print(f"\n  --- SUMMARY ---")
    print(f"  {'Variant':<30s} {'Trades':>7s} {'WR':>6s} {'PF':>7s} {'P&L':>12s} "
          f"{'DD':>10s} {'Sharpe':>7s} {'Trd/Day':>8s}")
    print(f"  {'-'*95}")
    for name, m, _ in variants:
        if m:
            print(f"  {name:<30s} {m.total_trades:>7d} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
                  f"{'${:,.0f}'.format(m.net_pnl):>12s} {'${:,.0f}'.format(m.max_drawdown):>10s} "
                  f"{m.sharpe:>7.2f} {m.trades_per_day:>8.2f}")
        else:
            print(f"  {name:<30s}  NO TRADES")

    # Statistical tests
    print(f"\n  --- STATISTICAL TESTS ---")
    for name, m, trades in variants:
        if trades and len(trades) >= 10:
            pnls = [t.pnl_dollar for t in trades]
            t_stat, p_val = stats.ttest_1samp(pnls, 0)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"  {name:<30s}  n={len(trades):>5d}  t={t_stat:>6.3f}  p={p_val:.4f} {sig}")
        elif trades:
            print(f"  {name:<30s}  too few trades ({len(trades)})")

    return variants


def main():
    parser = argparse.ArgumentParser(description="Market Structure Strategy Comparison")
    parser.add_argument("csv_file", help="Path to CSV data file")
    parser.add_argument("--split-date", type=str, default="2025-02-14")
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"RTH bars: {df['is_rth'].sum()}, Sessions: {df['new_rth'].sum()}")

    # Full comparison
    full_variants = run_comparison(df, "FULL 2-YEAR")

    # Walk-forward
    split_idx = df.index.get_indexer([args.split_date], method="nearest")[0]
    if split_idx <= 0 or split_idx >= len(df):
        split_idx = len(df) // 2

    df_y1 = df.iloc[:split_idx].copy()
    df_y2 = df.iloc[split_idx:].copy()

    print(f"\n  Year 1: {df_y1.index[0].date()} to {df_y1.index[-1].date()}")
    print(f"  Year 2: {df_y2.index[0].date()} to {df_y2.index[-1].date()}")

    y1_variants = run_comparison(df_y1, "YEAR 1 — IN-SAMPLE")
    y2_variants = run_comparison(df_y2, "YEAR 2 — OUT-OF-SAMPLE")

    # Walk-forward ratios
    print(f"\n{'='*110}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"{'='*110}")
    for (n1, m1, t1), (n2, m2, t2) in zip(y1_variants, y2_variants):
        if m1 and m2 and m1.profit_factor > 0:
            ratio = m2.profit_factor / m1.profit_factor
            verdict = "PASS" if ratio > 0.7 else "MARGINAL" if ratio > 0.5 else "FAIL"
            print(f"  {n1:<30s}  IS PF={m1.profit_factor:.3f} ({m1.total_trades}t)  "
                  f"OOS PF={m2.profit_factor:.3f} ({m2.total_trades}t)  "
                  f"ratio={ratio:.2f}  {verdict}")
            if t2 and len(t2) >= 5:
                _, p_val = stats.ttest_1samp([t.pnl_dollar for t in t2], 0)
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                print(f"  {'':30s}  OOS p={p_val:.4f} {sig}")


if __name__ == "__main__":
    main()
