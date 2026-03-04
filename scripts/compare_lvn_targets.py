#!/usr/bin/env python3
"""Compare v13 baseline targets vs LVN-aware dynamic targets.

Runs 3 variants through the stagger engine on 2-year data:
  1. v13 Baseline: 3x C3 (all 3 contracts target 3rd support, uniform_skip=2)
  2. LVN Dynamic:  3 contracts, LVN-scored target selection (best air pocket targets)
  3. LVN Split:    1 contract at nearest strong support + 2 at best LVN target

If LVN targeting improves fill rate or P&L, we'll walk-forward validate.

Usage:
    python scripts/compare_lvn_targets.py data/es_5m_databento_2yr.csv
    python scripts/compare_lvn_targets.py data/es_5m_databento_2yr.csv --year-split
"""

import argparse
import sys
import os

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.stagger_engine import run_backtest_stagger
from backtester.metrics import compute_metrics


def make_v81_baseline():
    """v8.1 baseline: IB Breakout + IB Rejection (wide days)."""
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
    return cfg


def make_v13_config():
    """v13 Level Rejection: ONH, TEMA, 8pt stop, R:R >= 0.5, min 5pt target."""
    cfg = make_v81_baseline()
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


def make_hvn_filter_config():
    """v13 + HVN filter: only take trades where target sits on composite HVN."""
    cfg = make_v13_config()
    cfg.lvl_use_hvn_targets = True
    cfg.lvl_hvn_lookback_days = 5
    cfg.lvl_hvn_tolerance = 3.0
    cfg.lvl_hvn_min_distance = 5.0
    cfg.lvl_hvn_max_distance = 60.0
    cfg.lvl_hvn_prefer_confluence = True   # FILTER: skip trades where target is not on HVN
    cfg.lvl_lvn_path_bonus = False         # No adjust, just filter
    return cfg


def make_hvn_search_config():
    """v13 + HVN SEARCH: find deepest HVN-confirmed support, fallback to 3rd."""
    cfg = make_v13_config()
    cfg.lvl_use_hvn_targets = True
    cfg.lvl_hvn_lookback_days = 5
    cfg.lvl_hvn_tolerance = 8.0
    cfg.lvl_hvn_min_distance = 5.0
    cfg.lvl_hvn_max_distance = 60.0
    cfg.lvl_hvn_prefer_confluence = True   # Search for HVN-confirmed
    cfg.lvl_lvn_path_bonus = True          # SEARCH mode (both True)
    return cfg


def make_hvn_adjust_config():
    """v13 + HVN adjust: fine-tune target to nearest HVN center (no filter)."""
    cfg = make_v13_config()
    cfg.lvl_use_hvn_targets = True
    cfg.lvl_hvn_lookback_days = 5
    cfg.lvl_hvn_tolerance = 8.0
    cfg.lvl_hvn_min_distance = 5.0
    cfg.lvl_hvn_max_distance = 60.0
    cfg.lvl_hvn_prefer_confluence = False  # No filter, take all trades
    cfg.lvl_lvn_path_bonus = True          # ADJUST: shift target to HVN center
    return cfg


def print_metrics(label, trades, show_detail=False):
    """Print compact metrics line."""
    if not trades:
        print(f"  {label:<40s}  NO TRADES")
        return None

    m = compute_metrics(trades)
    print(f"  {label:<40s}  {m.total_trades:>4d} trades  "
          f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+10,.0f}  DD ${m.max_drawdown:>8,.0f}  "
          f"Sharpe {m.sharpe:>5.2f}")

    if show_detail and trades:
        # LVL-only trades
        lvl_trades = [t for t in trades if t.setup.startswith("LVL_")]
        baseline_trades = [t for t in trades if not t.setup.startswith("LVL_")]

        if lvl_trades:
            ml = compute_metrics(lvl_trades)
            print(f"    LVL only:  {ml.total_trades:>4d} fills  "
                  f"WR {ml.win_rate:>5.1f}%  PF {ml.profit_factor:>6.3f}  "
                  f"P&L ${ml.net_pnl:>+10,.0f}")

        if baseline_trades:
            mb = compute_metrics(baseline_trades)
            print(f"    Baseline:  {mb.total_trades:>4d} trades  "
                  f"WR {mb.win_rate:>5.1f}%  PF {mb.profit_factor:>6.3f}  "
                  f"P&L ${mb.net_pnl:>+10,.0f}")

        # Exit reason breakdown for LVL
        if lvl_trades:
            reasons = {}
            for t in lvl_trades:
                r = t.exit_reason
                if r not in reasons:
                    reasons[r] = {"count": 0, "pnl": 0}
                reasons[r]["count"] += 1
                reasons[r]["pnl"] += t.pnl_dollar
            print(f"    LVL exits:")
            for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
                print(f"      {r:<12s}  {d['count']:>4d}  ${d['pnl']:>+10,.0f}")

        # Per-contract breakdown for LVL
        contracts = {}
        for t in lvl_trades:
            c = getattr(t, 'contract', 1)
            if c not in contracts:
                contracts[c] = []
            contracts[c].append(t)
        if len(contracts) > 1:
            print(f"    Per contract:")
            for c in sorted(contracts.keys()):
                ct = contracts[c]
                mc = compute_metrics(ct)
                print(f"      C{c}: {mc.total_trades:>4d} fills  "
                      f"WR {mc.win_rate:>5.1f}%  PF {mc.profit_factor:>6.3f}  "
                      f"P&L ${mc.net_pnl:>+10,.0f}")

    return m


def run_comparison(df, label="FULL 2-YEAR"):
    """Run 5-min vs 30-min VP sweep and compare."""
    print(f"\n{'='*80}")
    print(f"  {label} — 5-min vs 30-min VP COMPARISON")
    print(f"{'='*80}")

    # Baseline: v13 (3x C3, uniform_skip=2) — uses 30-min VP by default now
    cfg1 = make_v13_config()
    trades1 = run_backtest_stagger(df.copy(), cfg1, n_contracts=3, uniform_skip=2)
    m1 = print_metrics("v13 Baseline (no HVN filter)", trades1, show_detail=False)

    # Mode comparison with 30-min VP
    variants = []

    # FILTER: reject trades where target not on HVN
    cfg2 = make_hvn_filter_config()
    cfg2.lvl_hvn_tolerance = 8.0
    trades2 = run_backtest_stagger(df.copy(), cfg2, n_contracts=3, uniform_skip=2)
    m2 = print_metrics("30m FILTER tol=8pt", trades2, show_detail=True)
    variants.append(("FILTER 8pt", m2, trades2))

    # SEARCH: find deepest HVN-confirmed support, fallback to 3rd
    cfg3 = make_hvn_search_config()
    trades3 = run_backtest_stagger(df.copy(), cfg3, n_contracts=3, uniform_skip=2)
    m3 = print_metrics("30m SEARCH tol=8pt", trades3, show_detail=True)
    variants.append(("SEARCH 8pt", m3, trades3))

    # SEARCH with wider tolerance
    cfg4 = make_hvn_search_config()
    cfg4.lvl_hvn_tolerance = 12.0
    trades4 = run_backtest_stagger(df.copy(), cfg4, n_contracts=3, uniform_skip=2)
    m4 = print_metrics("30m SEARCH tol=12pt", trades4, show_detail=True)
    variants.append(("SEARCH 12pt", m4, trades4))

    # ADJUST: keep all trades, shift to HVN center
    cfg5 = make_hvn_adjust_config()
    trades5 = run_backtest_stagger(df.copy(), cfg5, n_contracts=3, uniform_skip=2)
    m5 = print_metrics("30m ADJUST tol=8pt", trades5, show_detail=True)
    variants.append(("ADJUST 8pt", m5, trades5))

    # Summary table
    if m1:
        print(f"\n  --- 30-MIN VP MODE COMPARISON ---")
        print(f"  {'Variant':<25s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'P&L':>12s} {'DD':>10s} {'Sharpe':>7s} {'AvgTrd':>8s}")
        print(f"  {'-'*85}")
        print(f"  {'v13 Baseline':<25s} {m1.total_trades:>7d} {m1.win_rate:>5.1f}% {m1.profit_factor:>6.3f} "
              f"{'${:,.0f}'.format(m1.net_pnl):>12s} {'${:,.0f}'.format(m1.max_drawdown):>10s} "
              f"{m1.sharpe:>7.2f} {'${:,.0f}'.format(m1.avg_trade):>8s}")
        for name, m, _ in variants:
            if m:
                print(f"  {name:<25s} {m.total_trades:>7d} {m.win_rate:>5.1f}% {m.profit_factor:>6.3f} "
                      f"{'${:,.0f}'.format(m.net_pnl):>12s} {'${:,.0f}'.format(m.max_drawdown):>10s} "
                      f"{m.sharpe:>7.2f} {'${:,.0f}'.format(m.avg_trade):>8s}")

    # Stat tests
    lvl1 = [t.pnl_dollar for t in trades1 if t.setup.startswith("LVL_")]
    print(f"\n  Statistical tests (LVL trades vs baseline):")
    for name, m, trades in variants:
        lvl2 = [t.pnl_dollar for t in trades if t.setup.startswith("LVL_")]
        if lvl1 and lvl2:
            t_stat, p_val = stats.ttest_ind(lvl1, lvl2, equal_var=False)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"  {name}: t={t_stat:>6.3f}, p={p_val:.4f} {sig}")

    return trades1, variants


def main():
    parser = argparse.ArgumentParser(description="Compare LVN targets vs v13 baseline")
    parser.add_argument("csv_file", help="Path to 2yr CSV data file")
    parser.add_argument("--year-split", action="store_true",
                        help="Also run per-year comparison")
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
    run_comparison(df, "FULL 2-YEAR")

    if args.year_split:
        # Split into two halves
        split_idx = df.index.get_indexer([args.split_date], method="nearest")[0]
        if split_idx <= 0 or split_idx >= len(df):
            # Fallback: split in half
            split_idx = len(df) // 2

        df_y1 = df.iloc[:split_idx].copy()
        df_y2 = df.iloc[split_idx:].copy()

        print(f"\n  Year 1: {df_y1.index[0].date()} to {df_y1.index[-1].date()}")
        print(f"  Year 2: {df_y2.index[0].date()} to {df_y2.index[-1].date()}")

        run_comparison(df_y1, "YEAR 1 (IN-SAMPLE)")
        run_comparison(df_y2, "YEAR 2 (OUT-OF-SAMPLE)")


if __name__ == "__main__":
    main()
