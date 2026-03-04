#!/usr/bin/env python3
"""Walk-forward validation for IB Rejection (wide days) + IB Breakout combined.

Splits 2-year Databento data into two halves:
  - In-sample (Year 2):   Feb 2024 – Feb 2025
  - Out-of-sample (Year 1): Feb 2025 – Feb 2026

Runs v8 baseline (IB Breakout only) and v8 + IB Rejection (wide-day only)
on each half. If IB Rejection's edge persists out-of-sample, it's structural.

Also runs statistical significance tests (t-test, permutation, bootstrap).

Usage:
    python scripts/walk_forward_ib_rejection.py data/es_5m_databento_2yr.csv
    python scripts/walk_forward_ib_rejection.py data/es_5m_databento_2yr.csv --split 2025-02-14
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
from backtester.metrics import compute_metrics, per_setup_breakdown


def make_v8_baseline():
    """v8 baseline — IB Breakout only, no rejection."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_ib_reject = False
    return cfg


def make_v8_plus_rej():
    """v8 + IB Rejection on wide days — the combined setup."""
    cfg = make_v8_baseline()
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    return cfg


def run_on_slice(df_slice, cfg, label):
    """Run backtest on a DataFrame slice, return (Metrics, trades list)."""
    trades = run_backtest(df_slice.copy(), cfg)
    if not trades:
        print(f"  {label}: 0 trades")
        return None, []
    m = compute_metrics(trades, cfg.initial_capital)
    print(f"  {label}: {m.total_trades} trades | "
          f"WR {m.win_rate:.1f}% | PF {m.profit_factor:.3f} | "
          f"P&L ${m.net_pnl:,.0f} | DD ${m.max_drawdown:,.0f} | "
          f"Sharpe {m.sharpe:.2f}")
    return m, trades


def setup_breakdown(trades, label):
    """Print per-setup breakdown (IB vs REJ)."""
    by_setup = {}
    for t in trades:
        if t.setup not in by_setup:
            by_setup[t.setup] = []
        by_setup[t.setup].append(t)

    for setup, tlist in sorted(by_setup.items()):
        pnls = [t.pnl_dollar for t in tlist]
        wins = sum(1 for p in pnls if p > 0)
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = wins / len(pnls) * 100
        print(f"    {setup}: {len(pnls)} trades | WR {wr:.1f}% | PF {pf:.3f} | P&L ${sum(pnls):+,.0f}")


def stat_tests(trades):
    """Run statistical significance tests on trade P&Ls."""
    if not trades:
        return

    pnls = np.array([t.pnl_dollar for t in trades])
    n = len(pnls)
    mean_pnl = np.mean(pnls)

    # t-test: is mean P&L significantly > 0?
    t_stat, t_pval = stats.ttest_1samp(pnls, 0)
    t_pval_one = t_pval / 2 if t_stat > 0 else 1 - t_pval / 2

    # Permutation test (10,000 iterations)
    np.random.seed(42)
    n_perm = 10000
    perm_means = np.array([
        np.mean(pnls * np.random.choice([-1, 1], size=n))
        for _ in range(n_perm)
    ])
    perm_pval = np.mean(perm_means >= mean_pnl)

    # Bootstrap P(profit)
    n_boot = 10000
    boot_means = np.array([
        np.mean(np.random.choice(pnls, size=n, replace=True))
        for _ in range(n_boot)
    ])
    p_profit = np.mean(boot_means > 0) * 100

    # Monthly breakdown
    by_month = {}
    for t in trades:
        key = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, 'strftime') else "unknown"
        by_month.setdefault(key, 0.0)
        by_month[key] += t.pnl_dollar
    winning_months = sum(1 for v in by_month.values() if v > 0)
    total_months = len(by_month)

    print(f"\n  Statistical Tests ({n} trades):")
    print(f"    Mean P&L per trade: ${mean_pnl:+,.0f}")
    print(f"    t-test p-value (one-sided): {t_pval_one:.4f} {'*** SIGNIFICANT' if t_pval_one < 0.05 else '(not significant)'}")
    print(f"    Permutation p-value: {perm_pval:.4f} {'*** SIGNIFICANT' if perm_pval < 0.05 else '(not significant)'}")
    print(f"    Bootstrap P(profit): {p_profit:.1f}%")
    print(f"    Monthly winners: {winning_months}/{total_months} ({winning_months/total_months*100:.0f}%)" if total_months > 0 else "")


def main():
    parser = argparse.ArgumentParser(description="IB Rejection Walk-Forward Validation")
    parser.add_argument("csv_file", help="Path to 2-year Databento CSV")
    parser.add_argument("--split", default="2025-02-14",
                        help="Split date YYYY-MM-DD (default: 2025-02-14, midpoint of 2yr)")
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file)
    print(f"Loaded {len(df)} bars: {df.index[0]} to {df.index[-1]}\n")

    split_date = args.split
    df_is = df[df.index < split_date]   # In-sample (Year 2 = older data)
    df_oos = df[df.index >= split_date]  # Out-of-sample (Year 1 = recent data)

    print(f"In-sample:      {df_is.index[0].date()} to {df_is.index[-1].date()} "
          f"({len(df_is)} bars)")
    print(f"Out-of-sample:  {df_oos.index[0].date()} to {df_oos.index[-1].date()} "
          f"({len(df_oos)} bars)")

    # ═══════════════════════════════════════════════════════════════
    # IN-SAMPLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  IN-SAMPLE (parameter fitting period)")
    print(f"{'='*70}")

    m_is_base, t_is_base = run_on_slice(df_is, make_v8_baseline(), "v8 Baseline")
    m_is_rej, t_is_rej = run_on_slice(df_is, make_v8_plus_rej(), "v8 + REJ Wide")

    if t_is_rej:
        print(f"\n  Setup breakdown (in-sample):")
        setup_breakdown(t_is_rej, "in-sample")

    # ═══════════════════════════════════════════════════════════════
    # OUT-OF-SAMPLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  OUT-OF-SAMPLE (unseen data)")
    print(f"{'='*70}")

    m_oos_base, t_oos_base = run_on_slice(df_oos, make_v8_baseline(), "v8 Baseline")
    m_oos_rej, t_oos_rej = run_on_slice(df_oos, make_v8_plus_rej(), "v8 + REJ Wide")

    if t_oos_rej:
        print(f"\n  Setup breakdown (out-of-sample):")
        setup_breakdown(t_oos_rej, "out-of-sample")

    # ═══════════════════════════════════════════════════════════════
    # FULL 2-YEAR (sanity check)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  FULL 2-YEAR (sanity check)")
    print(f"{'='*70}")

    m_full_base, t_full_base = run_on_slice(df, make_v8_baseline(), "v8 Baseline")
    m_full_rej, t_full_rej = run_on_slice(df, make_v8_plus_rej(), "v8 + REJ Wide")

    if t_full_rej:
        print(f"\n  Setup breakdown (full 2-year):")
        setup_breakdown(t_full_rej, "full 2-year")

    # ═══════════════════════════════════════════════════════════════
    # STATISTICAL SIGNIFICANCE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE")
    print(f"{'='*70}")

    print(f"\n  --- Out-of-sample (combined IB + REJ) ---")
    stat_tests(t_oos_rej)

    # OOS REJ-only stats
    rej_oos = [t for t in t_oos_rej if t.setup == "REJ"]
    if rej_oos:
        print(f"\n  --- Out-of-sample (REJ trades only) ---")
        stat_tests(rej_oos)

    print(f"\n  --- Full 2-year (combined) ---")
    stat_tests(t_full_rej)

    # ═══════════════════════════════════════════════════════════════
    # WALK-FORWARD SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'='*70}")

    if m_is_rej and m_oos_rej and m_is_base and m_oos_base:
        # PF ratio (OOS / IS)
        pf_ratio = (m_oos_rej.profit_factor / m_is_rej.profit_factor
                    if m_is_rej.profit_factor > 0 else 0)

        # P&L improvement over baseline
        is_delta = m_is_rej.net_pnl - m_is_base.net_pnl
        oos_delta = m_oos_rej.net_pnl - m_oos_base.net_pnl

        print(f"\n  Combined (IB Breakout + IB Rejection Wide):")
        print(f"    In-sample:      {m_is_rej.total_trades} trades | PF {m_is_rej.profit_factor:.3f} | P&L ${m_is_rej.net_pnl:+,.0f}")
        print(f"    Out-of-sample:  {m_oos_rej.total_trades} trades | PF {m_oos_rej.profit_factor:.3f} | P&L ${m_oos_rej.net_pnl:+,.0f}")

        print(f"\n  v8 Baseline (IB Breakout only):")
        print(f"    In-sample:      {m_is_base.total_trades} trades | PF {m_is_base.profit_factor:.3f} | P&L ${m_is_base.net_pnl:+,.0f}")
        print(f"    Out-of-sample:  {m_oos_base.total_trades} trades | PF {m_oos_base.profit_factor:.3f} | P&L ${m_oos_base.net_pnl:+,.0f}")

        print(f"\n  IB Rejection improvement over v8 baseline:")
        print(f"    In-sample:      ${is_delta:+,.0f}")
        print(f"    Out-of-sample:  ${oos_delta:+,.0f}")

        print(f"\n  PF ratio (OOS / IS): {pf_ratio:.2f}", end="")
        if pf_ratio >= 0.7:
            print(f"  *** ROBUST (>= 0.7)")
        elif pf_ratio >= 0.5:
            print(f"  ** ACCEPTABLE (0.5-0.7)")
        else:
            print(f"  * LIKELY OVERFIT (< 0.5)")

        # Final verdict
        oos_pf_good = m_oos_rej.profit_factor > 1.2
        oos_adds_value = oos_delta > 0
        pf_stable = pf_ratio >= 0.5

        print(f"\n  ─── VERDICT ───")
        if oos_pf_good and oos_adds_value and pf_stable:
            print(f"  *** PASS: IB Rejection edge persists out-of-sample")
            print(f"      OOS PF {m_oos_rej.profit_factor:.3f} > 1.2")
            print(f"      OOS improvement over baseline: ${oos_delta:+,.0f}")
            print(f"      PF ratio {pf_ratio:.2f} indicates {'robust' if pf_ratio >= 0.7 else 'acceptable'} stability")
        elif oos_adds_value and m_oos_rej.profit_factor > 1.0:
            print(f"  ** MARGINAL: Edge exists but weaker out-of-sample")
            print(f"      OOS PF {m_oos_rej.profit_factor:.3f}")
            print(f"      PF ratio {pf_ratio:.2f}")
        else:
            print(f"  * FAIL: IB Rejection edge does NOT persist out-of-sample")
            print(f"      OOS PF {m_oos_rej.profit_factor:.3f}")
            print(f"      OOS delta vs baseline: ${oos_delta:+,.0f}")


if __name__ == "__main__":
    main()
