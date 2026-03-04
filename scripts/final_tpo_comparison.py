#!/usr/bin/env python3
"""Final TPO Strategy Comparison — optimized VAR + PTF + walk-forward validation.

Runs optimized configurations based on exhaustive parameter sweeps:
  - VAR: Long-only, stop=10, rr=0.8, POC target, periods>=8, max_trades=2
  - PTF: OTF>=5, no reversal, prev_poc, stop=7, min_tgt=5, max_trades=2

Variants tested:
  1. v13 Baseline (stagger engine, 3 contracts) — reference
  2. VAR Optimized (long-only, wider stops, fewer trades)
  3. PTF Optimized (otf>=5, no reversal, lower min target)
  4. VAR + PTF Combined (both optimized, both directions)
  5. v13 + VAR + PTF (all strategies together)

Usage:
    python scripts/final_tpo_comparison.py data/es_5m_databento_2yr.csv
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


# ════════════════════════════════════════════════════════════════
#  Configuration Factories
# ════════════════════════════════════════════════════════════════

def make_v13_config():
    """v13 Level Rejection baseline (unchanged from proven config)."""
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
    return cfg


def make_var_optimized():
    """VAR Optimized — long-only, wider stops, dynamic POC, fewer trades.

    Key findings from optimization:
      - Longs PF 1.127 (+$12.7K), Shorts PF 0.918 (-$10.4K) → long-only
      - stop=10 with rr=0.8 → best balance of P&L and significance
      - Dynamic POC target beats all fixed targets
      - periods>=8 → more reliable developing VA
      - max_trades=2 → cleanest risk profile
    """
    cfg = StrategyConfig()
    cfg.direction_filter = "long"     # Longs carry the edge
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = True
    cfg.use_ptf = False
    cfg.var_zone_pts = 3.0            # Proximity to dev_VAL to trigger
    cfg.var_target_pts = 0.0          # Dynamic POC target (moves as profile develops)
    cfg.var_stop_buffer = 10.0        # Wider stop captures mean reversion
    cfg.var_min_ib_periods = 8        # Wait 4 hours for reliable VA
    cfg.var_require_rotation = True   # Must be rotation day (no OTF)
    cfg.var_max_otf = 2               # Max OTF streak to still qualify
    cfg.max_var_trades = 2            # 2 trades/day max (cleaner)
    cfg.var_min_rr = 0.8              # Min reward:risk
    return cfg


def make_ptf_optimized():
    """PTF Optimized — strong trend filter, no reversal, lower min target.

    Key findings from optimization:
      - OTF>=5 is the critical filter (separates strong from weak trends)
      - Reversal requirement is destructive (kills good entries)
      - prev_poc target >> single_print_mid
      - stop=7 is sweet spot (not too tight, not too wide)
      - min_tgt=5 allows more entries without hurting quality
      - Longs PF 2.3-2.8 but both-direction PF 1.75
    """
    cfg = StrategyConfig()
    cfg.direction_filter = "both"     # Both directions viable
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = True
    cfg.ptf_target = "prev_poc"       # Previous POC (value magnet)
    cfg.ptf_stop_buffer = 7.0         # Sweet spot stop width
    cfg.ptf_min_otf = 5               # Strong trend days only
    cfg.ptf_entry_zone = "single_prints"
    cfg.ptf_require_reversal = False  # Reversal filter is destructive
    cfg.max_ptf_trades = 2
    cfg.ptf_min_target_pts = 5.0      # Lower threshold = more entries
    return cfg


def make_var_ptf_combined():
    """VAR + PTF combined — both optimized strategies together."""
    cfg = make_var_optimized()
    # Add PTF on top of VAR
    cfg.direction_filter = "both"     # PTF needs both directions
    cfg.use_ptf = True
    cfg.ptf_target = "prev_poc"
    cfg.ptf_stop_buffer = 7.0
    cfg.ptf_min_otf = 5
    cfg.ptf_entry_zone = "single_prints"
    cfg.ptf_require_reversal = False
    cfg.max_ptf_trades = 2
    cfg.ptf_min_target_pts = 5.0
    return cfg


def make_all_combined():
    """v13 + VAR + PTF — all strategies together (single contract engine)."""
    cfg = make_v13_config()
    cfg.direction_filter = "both"     # VAR/PTF need both directions
    # Add VAR optimized
    cfg.use_var = True
    cfg.var_zone_pts = 3.0
    cfg.var_target_pts = 0.0
    cfg.var_stop_buffer = 10.0
    cfg.var_min_ib_periods = 8
    cfg.var_require_rotation = True
    cfg.var_max_otf = 2
    cfg.max_var_trades = 2
    cfg.var_min_rr = 0.8
    # Add PTF optimized
    cfg.use_ptf = True
    cfg.ptf_target = "prev_poc"
    cfg.ptf_stop_buffer = 7.0
    cfg.ptf_min_otf = 5
    cfg.ptf_entry_zone = "single_prints"
    cfg.ptf_require_reversal = False
    cfg.max_ptf_trades = 2
    cfg.ptf_min_target_pts = 5.0
    return cfg


# ════════════════════════════════════════════════════════════════
#  Output Helpers
# ════════════════════════════════════════════════════════════════

def print_metrics(label, trades, show_detail=False):
    """Print compact metrics line with optional per-setup breakdown."""
    if not trades:
        print(f"  {label:<40s}  NO TRADES")
        return None

    m = compute_metrics(trades)
    print(f"  {label:<40s}  {m.total_trades:>5d} trades  "
          f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+10,.0f}  DD ${m.max_drawdown:>8,.0f}  "
          f"Sharpe {m.sharpe:>5.2f}")

    if show_detail and trades:
        breakdown = per_setup_breakdown(trades)
        for setup, sm in breakdown.items():
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
    """Run all variants and compare. Returns list of (name, metrics, trades)."""
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")

    variants = []

    # 1. v13 Baseline (stagger engine, 3 contracts)
    cfg1 = make_v13_config()
    trades1 = run_backtest_stagger(df.copy(), cfg1, n_contracts=3, uniform_skip=2)
    m1 = print_metrics("v13 Baseline (3x stagger)", trades1)
    variants.append(("v13 Baseline", m1, trades1))

    # 2. VAR Optimized (long-only, wider stops)
    cfg2 = make_var_optimized()
    trades2 = run_backtest(df.copy(), cfg2)
    m2 = print_metrics("VAR Optimized (long-only)", trades2, show_detail=True)
    variants.append(("VAR Optimized", m2, trades2))

    # 3. PTF Optimized (otf>=5, no reversal)
    cfg3 = make_ptf_optimized()
    trades3 = run_backtest(df.copy(), cfg3)
    m3 = print_metrics("PTF Optimized (otf>=5, norev)", trades3, show_detail=True)
    variants.append(("PTF Optimized", m3, trades3))

    # 4. VAR + PTF Combined
    cfg4 = make_var_ptf_combined()
    trades4 = run_backtest(df.copy(), cfg4)
    m4 = print_metrics("VAR + PTF Combined", trades4, show_detail=True)
    variants.append(("VAR + PTF", m4, trades4))

    # 5. v13 + VAR + PTF (all together, single contract)
    cfg5 = make_all_combined()
    trades5 = run_backtest(df.copy(), cfg5)
    m5 = print_metrics("v13 + VAR + PTF (all)", trades5, show_detail=True)
    variants.append(("v13 + VAR + PTF", m5, trades5))

    # ── Summary Table ──
    print(f"\n  --- SUMMARY ---")
    print(f"  {'Variant':<30s} {'Trades':>7s} {'WR':>6s} {'PF':>7s} {'P&L':>12s} "
          f"{'DD':>10s} {'Sharpe':>7s} {'Avg':>8s} {'Trd/Day':>8s}")
    print(f"  {'-'*100}")
    for name, m, _ in variants:
        if m:
            print(f"  {name:<30s} {m.total_trades:>7d} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
                  f"{'${:,.0f}'.format(m.net_pnl):>12s} {'${:,.0f}'.format(m.max_drawdown):>10s} "
                  f"{m.sharpe:>7.2f} {'${:,.0f}'.format(m.avg_trade):>8s} {m.trades_per_day:>8.2f}")
        else:
            print(f"  {name:<30s}  NO TRADES")

    # ── Statistical Significance ──
    print(f"\n  --- STATISTICAL TESTS (one-sample t-test, H0: mean trade = 0) ---")
    for name, m, trades in variants:
        if trades and len(trades) >= 5:
            pnls = [t.pnl_dollar for t in trades]
            t_stat, p_val = stats.ttest_1samp(pnls, 0)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"  {name:<30s}  n={len(trades):>4d}  t={t_stat:>6.3f}  p={p_val:.4f} {sig}")
        elif trades:
            print(f"  {name:<30s}  too few trades ({len(trades)})")
        else:
            print(f"  {name:<30s}  no trades")

    return variants


def main():
    parser = argparse.ArgumentParser(description="Final TPO Strategy Comparison")
    parser.add_argument("csv_file", help="Path to 2yr CSV data file")
    parser.add_argument("--split-date", type=str, default="2025-02-14",
                        help="Date to split years (default: 2025-02-14)")
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    rth_bars = df["is_rth"].sum()
    sessions = df["new_rth"].sum()
    print(f"RTH bars: {rth_bars}, Trading sessions: {sessions}")

    # ═══════════════════════════════════════════════════════
    #  Full 2-year comparison
    # ═══════════════════════════════════════════════════════
    full_variants = run_comparison(df, "FULL 2-YEAR (ALL DATA)")

    # ═══════════════════════════════════════════════════════
    #  Walk-forward: Year 1 (IS) vs Year 2 (OOS)
    # ═══════════════════════════════════════════════════════
    split_idx = df.index.get_indexer([args.split_date], method="nearest")[0]
    if split_idx <= 0 or split_idx >= len(df):
        split_idx = len(df) // 2

    df_y1 = df.iloc[:split_idx].copy()
    df_y2 = df.iloc[split_idx:].copy()

    print(f"\n  Year 1: {df_y1.index[0].date()} to {df_y1.index[-1].date()}")
    print(f"  Year 2: {df_y2.index[0].date()} to {df_y2.index[-1].date()}")

    y1_variants = run_comparison(df_y1, "YEAR 1 — IN-SAMPLE")
    y2_variants = run_comparison(df_y2, "YEAR 2 — OUT-OF-SAMPLE")

    # ── Walk-forward PF Ratio ──
    print(f"\n{'='*100}")
    print(f"  WALK-FORWARD VALIDATION (OOS PF / IS PF, > 0.7 = robust)")
    print(f"{'='*100}")
    for (name1, m1, t1), (name2, m2, t2) in zip(y1_variants, y2_variants):
        if m1 and m2 and m1.profit_factor > 0:
            ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
            robust = "PASS" if ratio > 0.7 else "MARGINAL" if ratio > 0.5 else "FAIL"
            print(f"  {name1:<30s}  IS PF={m1.profit_factor:.3f} ({m1.total_trades}t)  "
                  f"OOS PF={m2.profit_factor:.3f} ({m2.total_trades}t)  "
                  f"ratio={ratio:.2f}  {robust}")

            # OOS statistical test
            if t2 and len(t2) >= 5:
                pnls = [t.pnl_dollar for t in t2]
                _, p_val = stats.ttest_1samp(pnls, 0)
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                print(f"  {'':30s}  OOS p={p_val:.4f} {sig}")
        elif m1:
            print(f"  {name1:<30s}  IS PF={m1.profit_factor:.3f}  OOS: no trades")
        else:
            print(f"  {name1:<30s}  no data")

    # ═══════════════════════════════════════════════════════
    #  Monthly Consistency Check (best strategies)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  MONTHLY P&L CONSISTENCY")
    print(f"{'='*100}")
    for name, m, trades in full_variants:
        if not trades or len(trades) < 10:
            continue
        # Group trades by month
        monthly = {}
        for t in trades:
            month_key = t.entry_time.strftime("%Y-%m")
            if month_key not in monthly:
                monthly[month_key] = 0
            monthly[month_key] += t.pnl_dollar

        months = sorted(monthly.keys())
        winning_months = sum(1 for m_pnl in monthly.values() if m_pnl > 0)
        total_months = len(months)

        print(f"\n  {name} ({winning_months}/{total_months} winning months)")
        for mo in months:
            bar_len = int(abs(monthly[mo]) / 500)  # Scale bar
            bar_char = "+" if monthly[mo] > 0 else "-"
            bar = bar_char * min(bar_len, 40)
            print(f"    {mo}  ${monthly[mo]:>+8,.0f}  {bar}")


if __name__ == "__main__":
    main()
