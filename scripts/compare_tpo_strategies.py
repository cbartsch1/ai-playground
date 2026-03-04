#!/usr/bin/env python3
"""Compare TPO-based strategies: v13 baseline vs VAR vs PTF vs combinations.

Runs 5 variants through the engine on 2-year data:
  1. v13 Baseline — existing level rejection via stagger engine (3 contracts, verify unchanged)
  2. VAR Only — Value Area Rotation in isolation (single contract)
  3. PTF Only — Post-Trend Day Fade in isolation (single contract)
  4. VAR + PTF — both new strategies combined (single contract)
  5. v13 + VAR + PTF — all strategies together (single contract engine)

Usage:
    python scripts/compare_tpo_strategies.py data/es_5m_databento_2yr.csv
    python scripts/compare_tpo_strategies.py data/es_5m_databento_2yr.csv --year-split
"""

import argparse
import sys
import os

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.stagger_engine import run_backtest_stagger
from backtester.metrics import compute_metrics, per_setup_breakdown


def _base_config():
    """Shared base: v8 filters (short-only, pct stop, Friday, blackout)."""
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
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    cfg.use_var = False
    cfg.use_ptf = False
    return cfg


def make_v13_config():
    """v13 Level Rejection baseline."""
    cfg = _base_config()
    cfg.use_level_reject = True
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
    return cfg


def make_var_only():
    """VAR only — no baseline setups, just Value Area Rotation."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = True
    cfg.use_ptf = False
    cfg.var_zone_pts = 3.0
    cfg.var_target_pts = 0.0
    cfg.var_stop_buffer = 4.0
    cfg.var_min_ib_periods = 4
    cfg.var_require_rotation = True
    cfg.var_max_otf = 2
    cfg.max_var_trades = 8
    cfg.var_min_rr = 0.8
    return cfg


def make_ptf_only():
    """PTF only — no baseline setups, just Post-Trend Fade."""
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
    cfg.use_ptf = True
    cfg.ptf_target = "prev_poc"
    cfg.ptf_stop_buffer = 5.0
    cfg.ptf_min_otf = 4
    cfg.ptf_entry_zone = "single_prints"
    cfg.ptf_require_reversal = True
    cfg.max_ptf_trades = 2
    cfg.ptf_min_target_pts = 8.0
    return cfg


def make_var_ptf():
    """VAR + PTF combined — both new strategies, no v13."""
    cfg = make_var_only()
    cfg.use_ptf = True
    cfg.ptf_target = "prev_poc"
    cfg.ptf_stop_buffer = 5.0
    cfg.ptf_min_otf = 4
    cfg.ptf_entry_zone = "single_prints"
    cfg.ptf_require_reversal = True
    cfg.max_ptf_trades = 2
    cfg.ptf_min_target_pts = 8.0
    return cfg


def make_all_combined():
    """v13 + VAR + PTF — everything together (single contract engine)."""
    cfg = make_v13_config()
    cfg.direction_filter = "both"
    cfg.use_var = True
    cfg.var_zone_pts = 3.0
    cfg.var_target_pts = 0.0
    cfg.var_stop_buffer = 4.0
    cfg.var_min_ib_periods = 4
    cfg.var_require_rotation = True
    cfg.var_max_otf = 2
    cfg.max_var_trades = 8
    cfg.var_min_rr = 0.8
    cfg.use_ptf = True
    cfg.ptf_target = "prev_poc"
    cfg.ptf_stop_buffer = 5.0
    cfg.ptf_min_otf = 4
    cfg.ptf_entry_zone = "single_prints"
    cfg.ptf_require_reversal = True
    cfg.max_ptf_trades = 2
    cfg.ptf_min_target_pts = 8.0
    return cfg


def print_metrics(label, trades, show_detail=False):
    """Print compact metrics line with optional per-setup breakdown."""
    if not trades:
        print(f"  {label:<35s}  NO TRADES")
        return None

    m = compute_metrics(trades)
    print(f"  {label:<35s}  {m.total_trades:>5d} trades  "
          f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+10,.0f}  DD ${m.max_drawdown:>8,.0f}  "
          f"Sharpe {m.sharpe:>5.2f}")

    if show_detail and trades:
        breakdown = per_setup_breakdown(trades)
        for setup, sm in breakdown.items():
            print(f"    {setup:<20s}  {sm.total_trades:>5d}  "
                  f"WR {sm.win_rate:>5.1f}%  PF {sm.profit_factor:>6.3f}  "
                  f"P&L ${sm.net_pnl:>+10,.0f}")

        # Exit reason breakdown
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
    """Run all 5 variants and compare. Returns list of (name, metrics, trades)."""
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")

    variants = []

    # 1. v13 Baseline (stagger engine, 3 contracts, uniform_skip=2)
    cfg1 = make_v13_config()
    trades1 = run_backtest_stagger(df.copy(), cfg1, n_contracts=3, uniform_skip=2)
    m1 = print_metrics("v13 Baseline (3x stagger)", trades1)
    variants.append(("v13 Baseline", m1, trades1))

    # 2. VAR Only
    cfg2 = make_var_only()
    trades2 = run_backtest(df.copy(), cfg2)
    m2 = print_metrics("VAR Only", trades2, show_detail=True)
    variants.append(("VAR Only", m2, trades2))

    # 3. PTF Only
    cfg3 = make_ptf_only()
    trades3 = run_backtest(df.copy(), cfg3)
    m3 = print_metrics("PTF Only", trades3, show_detail=True)
    variants.append(("PTF Only", m3, trades3))

    # 4. VAR + PTF
    cfg4 = make_var_ptf()
    trades4 = run_backtest(df.copy(), cfg4)
    m4 = print_metrics("VAR + PTF", trades4, show_detail=True)
    variants.append(("VAR + PTF", m4, trades4))

    # 5. v13 + VAR + PTF (single contract engine — all together)
    cfg5 = make_all_combined()
    trades5 = run_backtest(df.copy(), cfg5)
    m5 = print_metrics("v13 + VAR + PTF", trades5, show_detail=True)
    variants.append(("v13 + VAR + PTF", m5, trades5))

    # Summary table
    print(f"\n  --- SUMMARY ---")
    print(f"  {'Variant':<25s} {'Trades':>7s} {'WR':>6s} {'PF':>7s} {'P&L':>12s} "
          f"{'DD':>10s} {'Sharpe':>7s} {'Avg':>8s} {'Trd/Day':>8s}")
    print(f"  {'-'*95}")
    for name, m, _ in variants:
        if m:
            print(f"  {name:<25s} {m.total_trades:>7d} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
                  f"{'${:,.0f}'.format(m.net_pnl):>12s} {'${:,.0f}'.format(m.max_drawdown):>10s} "
                  f"{m.sharpe:>7.2f} {'${:,.0f}'.format(m.avg_trade):>8s} {m.trades_per_day:>8.2f}")
        else:
            print(f"  {name:<25s}  NO TRADES")

    # Statistical significance (t-test each vs zero mean)
    print(f"\n  --- STATISTICAL TESTS (one-sample t-test, H0: mean trade = 0) ---")
    for name, m, trades in variants:
        if trades and len(trades) >= 5:
            pnls = [t.pnl_dollar for t in trades]
            t_stat, p_val = stats.ttest_1samp(pnls, 0)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"  {name:<25s}  t={t_stat:>6.3f}  p={p_val:.4f} {sig}")
        elif trades:
            print(f"  {name:<25s}  too few trades ({len(trades)})")
        else:
            print(f"  {name:<25s}  no trades")

    return variants


def main():
    parser = argparse.ArgumentParser(description="Compare TPO strategies")
    parser.add_argument("csv_file", help="Path to 2yr CSV data file")
    parser.add_argument("--year-split", action="store_true",
                        help="Also run per-year walk-forward comparison")
    parser.add_argument("--split-date", type=str, default="2025-02-14",
                        help="Date to split years (default: 2025-02-14)")
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    rth_bars = df["is_rth"].sum()
    sessions = df["new_rth"].sum()
    print(f"RTH bars: {rth_bars}, Trading sessions: {sessions}")

    # Full 2-year comparison
    full_variants = run_comparison(df, "FULL 2-YEAR")

    if args.year_split:
        split_idx = df.index.get_indexer([args.split_date], method="nearest")[0]
        if split_idx <= 0 or split_idx >= len(df):
            split_idx = len(df) // 2

        df_y1 = df.iloc[:split_idx].copy()
        df_y2 = df.iloc[split_idx:].copy()

        print(f"\n  Year 1: {df_y1.index[0].date()} to {df_y1.index[-1].date()}")
        print(f"  Year 2: {df_y2.index[0].date()} to {df_y2.index[-1].date()}")

        y1_variants = run_comparison(df_y1, "YEAR 1 (IN-SAMPLE)")
        y2_variants = run_comparison(df_y2, "YEAR 2 (OUT-OF-SAMPLE)")

        # Walk-forward PF ratio
        print(f"\n  --- WALK-FORWARD PF RATIO (OOS PF / IS PF, > 0.7 = robust) ---")
        for (name1, m1, _), (name2, m2, _) in zip(y1_variants, y2_variants):
            if m1 and m2 and m1.profit_factor > 0:
                ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
                robust = "PASS" if ratio > 0.7 else "FAIL"
                print(f"  {name1:<25s}  IS PF={m1.profit_factor:.3f}  "
                      f"OOS PF={m2.profit_factor:.3f}  ratio={ratio:.2f}  {robust}")
            elif m1:
                print(f"  {name1:<25s}  IS PF={m1.profit_factor:.3f}  OOS: no trades")
            else:
                print(f"  {name1:<25s}  no data")


if __name__ == "__main__":
    main()
