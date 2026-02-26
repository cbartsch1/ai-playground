#!/usr/bin/env python3
"""Walk-forward validation for 3-contract staggered Level Rejection (ONH) + v8.1 baseline.

Splits 2-year Databento data into two halves:
  - In-sample (Year 1):   Feb 2024 – Feb 2025
  - Out-of-sample (Year 2): Feb 2025 – Feb 2026

Runs v8.1 baseline (IB Breakout + IB Rejection) alone, then adds 3-contract
staggered Level Rejection (ONH, TEMA filter, R:R 0.5, min target 5pt).

Uses the stagger engine (single backtest) so that:
  - Baseline trades are counted ONCE (not triple-counted)
  - LVL uses ONE trade-per-day counter across all 3 contracts
  - All positions share one SessionState

If the LVL edge persists out-of-sample, it's structural.

Usage:
    python scripts/walk_forward_level_rejection.py data/es_5m_databento_2yr.csv
    python scripts/walk_forward_level_rejection.py data/es_5m_databento_2yr.csv --split 2025-02-14
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
from backtester.metrics import compute_metrics

N_CONTRACTS = 3


def make_v81_baseline():
    """v8.1 baseline — IB Breakout + IB Rejection (wide days), no level rejection."""
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
    # IB Rejection (wide days)
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    return cfg


def make_lvl_stagger_config():
    """v8.1 + Level Rejection ONH (for stagger engine — skip handled internally)."""
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
    return cfg


def run_on_slice(df_slice, cfg, label):
    """Run baseline backtest on a DataFrame slice, return (Metrics, trades list)."""
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


def run_stagger_slice(df_slice, cfg, label, n_contracts=N_CONTRACTS):
    """Run stagger backtest on a DataFrame slice.

    Returns dict with all trades, per-contract breakdown, and combined metrics.
    """
    trades = run_backtest_stagger(df_slice.copy(), cfg, n_contracts=n_contracts)

    if not trades:
        print(f"  {label}: 0 trades")
        return {"trades": [], "baseline_trades": [], "lvl_trades": [],
                "by_contract": {}, "pnl": 0, "pf": 0, "dd": 0}

    # Split by type
    baseline_trades = [t for t in trades if not t.setup.startswith("LVL")]
    lvl_trades = [t for t in trades if t.setup.startswith("LVL")]

    # Per-contract LVL breakdown
    by_contract = {}
    for c in range(1, n_contracts + 1):
        c_trades = [t for t in lvl_trades if t.contract == c]
        by_contract[c] = c_trades

    # Combined metrics
    pnls = [t.pnl_dollar for t in trades]
    total_pnl = sum(pnls)
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100

    # Equity curve for combined DD
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    equity = 100_000.0
    peak = equity
    max_dd = 0
    for t in sorted_trades:
        equity += t.pnl_dollar
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Print summary
    print(f"\n  {label} COMBINED: {len(trades)} trades | "
          f"WR {wr:.1f}% | PF {pf:.3f} | P&L ${total_pnl:+,.0f} | DD ${max_dd:,.0f}")

    # Baseline breakdown
    if baseline_trades:
        b_pnls = [t.pnl_dollar for t in baseline_trades]
        b_win = sum(p for p in b_pnls if p > 0)
        b_loss = abs(sum(p for p in b_pnls if p <= 0))
        b_pf = b_win / b_loss if b_loss > 0 else float("inf")
        print(f"    Baseline (IB+REJ): {len(baseline_trades)} trades | "
              f"PF {b_pf:.3f} | P&L ${sum(b_pnls):+,.0f}")

    # LVL breakdown
    if lvl_trades:
        l_pnls = [t.pnl_dollar for t in lvl_trades]
        l_win = sum(p for p in l_pnls if p > 0)
        l_loss = abs(sum(p for p in l_pnls if p <= 0))
        l_pf = l_win / l_loss if l_loss > 0 else float("inf")
        print(f"    LVL total: {len(lvl_trades)} fills | "
              f"PF {l_pf:.3f} | P&L ${sum(l_pnls):+,.0f}")

    # Per-contract breakdown
    for c in range(1, n_contracts + 1):
        c_trades = by_contract[c]
        if c_trades:
            c_pnls = [t.pnl_dollar for t in c_trades]
            c_win = sum(p for p in c_pnls if p > 0)
            c_loss = abs(sum(p for p in c_pnls if p <= 0))
            c_pf = c_win / c_loss if c_loss > 0 else float("inf")
            c_wr = sum(1 for p in c_pnls if p > 0) / len(c_pnls) * 100
            targets = {"nearest", "2nd", "3rd"}
            tgt_label = ["nearest", "2nd support", "3rd support"][c - 1]
            print(f"      C{c} ({tgt_label}): {len(c_trades)} trades | "
                  f"WR {c_wr:.1f}% | PF {c_pf:.3f} | P&L ${sum(c_pnls):+,.0f}")

    return {
        "trades": trades,
        "baseline_trades": baseline_trades,
        "lvl_trades": lvl_trades,
        "by_contract": by_contract,
        "pnl": total_pnl,
        "pf": pf,
        "dd": max_dd,
    }


def setup_breakdown(trades, label):
    """Print per-setup breakdown."""
    by_setup = {}
    for t in trades:
        by_setup.setdefault(t.setup, []).append(t)

    for setup, tlist in sorted(by_setup.items()):
        pnls = [t.pnl_dollar for t in tlist]
        wins = sum(1 for p in pnls if p > 0)
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = wins / len(pnls) * 100
        print(f"    {setup}: {len(pnls)} trades | WR {wr:.1f}% | PF {pf:.3f} | P&L ${sum(pnls):+,.0f}")


def stat_tests(trades, label=""):
    """Run statistical significance tests on trade P&Ls."""
    if not trades:
        return {}

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

    print(f"\n  Statistical Tests — {label} ({n} trades):")
    print(f"    Mean P&L per trade: ${mean_pnl:+,.0f}")
    print(f"    t-test p-value (one-sided): {t_pval_one:.4f} {'*** SIGNIFICANT' if t_pval_one < 0.05 else '(not significant)'}")
    print(f"    Permutation p-value: {perm_pval:.4f} {'*** SIGNIFICANT' if perm_pval < 0.05 else '(not significant)'}")
    print(f"    Bootstrap P(profit): {p_profit:.1f}%")
    if total_months > 0:
        print(f"    Monthly winners: {winning_months}/{total_months} ({winning_months/total_months*100:.0f}%)")

    return {
        "t_pval": t_pval_one,
        "perm_pval": perm_pval,
        "p_profit": p_profit,
        "winning_months_pct": winning_months / total_months * 100 if total_months > 0 else 0,
    }


def _quick_pf(trades):
    """Compute profit factor from a trade list."""
    if not trades:
        return 0.0
    pnls = [t.pnl_dollar for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    return gross_win / gross_loss if gross_loss > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser(description="Level Rejection Walk-Forward Validation (Stagger Engine)")
    parser.add_argument("csv_file", help="Path to 2-year Databento CSV")
    parser.add_argument("--split", default="2025-02-14",
                        help="Split date YYYY-MM-DD (default: 2025-02-14)")
    parser.add_argument("--contracts", type=int, default=3,
                        help="Number of stagger contracts (default: 3)")
    args = parser.parse_args()

    n_contracts = args.contracts

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file)
    print(f"Loaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")
    print(f"Stagger contracts: {n_contracts}\n")

    split_date = args.split
    df_is = df[df.index < split_date]
    df_oos = df[df.index >= split_date]

    print(f"In-sample:      {df_is.index[0].date()} to {df_is.index[-1].date()} "
          f"({len(df_is)} bars)")
    print(f"Out-of-sample:  {df_oos.index[0].date()} to {df_oos.index[-1].date()} "
          f"({len(df_oos)} bars)")

    cfg_base = make_v81_baseline()
    cfg_stagger = make_lvl_stagger_config()

    # ═══════════════════════════════════════════════════════════════
    # IN-SAMPLE — v8.1 Baseline (single engine, no LVL)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  IN-SAMPLE — v8.1 Baseline (IB Breakout + IB Rejection)")
    print(f"{'='*70}")

    m_is_base, t_is_base = run_on_slice(df_is, cfg_base, "IS Baseline")
    if t_is_base:
        setup_breakdown(t_is_base, "IS Baseline")

    # ═══════════════════════════════════════════════════════════════
    # IN-SAMPLE — v8.1 + Level Rejection (stagger engine)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  IN-SAMPLE — v8.1 + Level Rejection ONH ({n_contracts}-contract stagger)")
    print(f"{'='*70}")

    is_result = run_stagger_slice(df_is, cfg_stagger, "IS", n_contracts=n_contracts)

    # ═══════════════════════════════════════════════════════════════
    # OUT-OF-SAMPLE — v8.1 Baseline
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  OUT-OF-SAMPLE — v8.1 Baseline (IB Breakout + IB Rejection)")
    print(f"{'='*70}")

    m_oos_base, t_oos_base = run_on_slice(df_oos, cfg_base, "OOS Baseline")
    if t_oos_base:
        setup_breakdown(t_oos_base, "OOS Baseline")

    # ═══════════════════════════════════════════════════════════════
    # OUT-OF-SAMPLE — v8.1 + Level Rejection (stagger engine)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  OUT-OF-SAMPLE — v8.1 + Level Rejection ONH ({n_contracts}-contract stagger)")
    print(f"{'='*70}")

    oos_result = run_stagger_slice(df_oos, cfg_stagger, "OOS", n_contracts=n_contracts)

    # ═══════════════════════════════════════════════════════════════
    # FULL 2-YEAR (sanity check)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  FULL 2-YEAR (sanity check)")
    print(f"{'='*70}")

    m_full_base, t_full_base = run_on_slice(df, cfg_base, "Full Baseline")
    full_result = run_stagger_slice(df, cfg_stagger, "Full", n_contracts=n_contracts)

    # ═══════════════════════════════════════════════════════════════
    # STATISTICAL SIGNIFICANCE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE")
    print(f"{'='*70}")

    # OOS all trades (baseline + LVL combined)
    print(f"\n  --- OOS: All trades (IB + REJ + LVL, stagger) ---")
    oos_stats_all = stat_tests(oos_result["trades"], "OOS All Trades")

    # OOS LVL-only
    if oos_result["lvl_trades"]:
        print(f"\n  --- OOS: LVL trades only ({n_contracts} contracts) ---")
        oos_stats_lvl = stat_tests(oos_result["lvl_trades"], "OOS LVL Only")

    # Full 2-year all trades
    print(f"\n  --- Full 2-year: All trades ---")
    stat_tests(full_result["trades"], "Full 2yr All")

    if full_result["lvl_trades"]:
        print(f"\n  --- Full 2-year: LVL trades only ---")
        stat_tests(full_result["lvl_trades"], "Full 2yr LVL")

    # ═══════════════════════════════════════════════════════════════
    # PER-CONTRACT IS vs OOS STABILITY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PER-CONTRACT IS vs OOS STABILITY (LVL trades only)")
    print(f"{'='*70}")

    for c in range(1, n_contracts + 1):
        tgt_label = ["nearest", "2nd support", "3rd support"][c - 1]
        is_c = is_result["by_contract"].get(c, [])
        oos_c = oos_result["by_contract"].get(c, [])

        is_pf = _quick_pf(is_c)
        oos_pf = _quick_pf(oos_c)
        is_pnl = sum(t.pnl_dollar for t in is_c) if is_c else 0
        oos_pnl = sum(t.pnl_dollar for t in oos_c) if oos_c else 0

        if is_pf > 0 and oos_pf > 0:
            pf_ratio = oos_pf / is_pf
            print(f"\n  C{c} ({tgt_label}):")
            print(f"    IS:  {len(is_c)} trades | PF {is_pf:.3f} | P&L ${is_pnl:+,.0f}")
            print(f"    OOS: {len(oos_c)} trades | PF {oos_pf:.3f} | P&L ${oos_pnl:+,.0f}")
            print(f"    PF ratio (OOS/IS): {pf_ratio:.2f}", end="")
            if pf_ratio >= 0.7:
                print(f"  *** ROBUST")
            elif pf_ratio >= 0.5:
                print(f"  ** ACCEPTABLE")
            else:
                print(f"  * WEAK")
        elif is_c:
            print(f"\n  C{c} ({tgt_label}): IS only — {len(is_c)} trades | PF {is_pf:.3f}")
        elif oos_c:
            print(f"\n  C{c} ({tgt_label}): OOS only — {len(oos_c)} trades | PF {oos_pf:.3f}")
        else:
            print(f"\n  C{c} ({tgt_label}): no trades")

    # ═══════════════════════════════════════════════════════════════
    # WALK-FORWARD SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'='*70}")

    is_pf = is_result["pf"]
    oos_pf = oos_result["pf"]
    is_pnl = is_result["pnl"]
    oos_pnl = oos_result["pnl"]

    pf_ratio = oos_pf / is_pf if is_pf > 0 else 0

    # LVL-only improvement over baseline
    is_base_pnl = m_is_base.net_pnl if m_is_base else 0
    oos_base_pnl = m_oos_base.net_pnl if m_oos_base else 0
    is_delta = is_pnl - is_base_pnl
    oos_delta = oos_pnl - oos_base_pnl

    # LVL-only P&L
    is_lvl_pnl = sum(t.pnl_dollar for t in is_result["lvl_trades"])
    oos_lvl_pnl = sum(t.pnl_dollar for t in oos_result["lvl_trades"])

    print(f"\n  {n_contracts}-Contract Stagger (IB Breakout + IB Rejection + LVL ONH):")
    print(f"    In-sample:      PF {is_pf:.3f} | P&L ${is_pnl:+,.0f} | DD ${is_result['dd']:,.0f}")
    print(f"    Out-of-sample:  PF {oos_pf:.3f} | P&L ${oos_pnl:+,.0f} | DD ${oos_result['dd']:,.0f}")

    print(f"\n  v8.1 Baseline (IB Breakout + IB Rejection):")
    if m_is_base:
        print(f"    In-sample:      PF {m_is_base.profit_factor:.3f} | P&L ${m_is_base.net_pnl:+,.0f}")
    if m_oos_base:
        print(f"    Out-of-sample:  PF {m_oos_base.profit_factor:.3f} | P&L ${m_oos_base.net_pnl:+,.0f}")

    print(f"\n  LVL improvement over v8.1 baseline:")
    print(f"    In-sample:      ${is_delta:+,.0f} (LVL-only: ${is_lvl_pnl:+,.0f})")
    print(f"    Out-of-sample:  ${oos_delta:+,.0f} (LVL-only: ${oos_lvl_pnl:+,.0f})")

    print(f"\n  PF ratio (OOS / IS): {pf_ratio:.2f}", end="")
    if pf_ratio >= 0.7:
        print(f"  *** ROBUST (>= 0.7)")
    elif pf_ratio >= 0.5:
        print(f"  ** ACCEPTABLE (0.5-0.7)")
    else:
        print(f"  * LIKELY OVERFIT (< 0.5)")

    # Annual return estimate
    full_pnl = full_result["pnl"]
    annual_pnl = full_pnl / 2  # 2-year data
    sp500_annual = 100_000 * 0.10  # 10% annual
    multiplier = annual_pnl / sp500_annual if sp500_annual > 0 else 0

    print(f"\n  Annual projection (full 2yr / 2):")
    print(f"    Annual P&L: ${annual_pnl:+,.0f}")
    print(f"    vs S&P 500 (10%): {multiplier:.1f}x")

    # ─── VERDICT ───
    print(f"\n  {'─'*50}")
    print(f"  VERDICT")
    print(f"  {'─'*50}")

    oos_pf_good = oos_pf > 1.15
    oos_adds_value = oos_delta > 0
    pf_stable = pf_ratio >= 0.5
    oos_profitable = oos_pnl > 0

    if oos_pf_good and oos_adds_value and pf_stable:
        print(f"  *** PASS: Level Rejection ONH edge persists out-of-sample")
        print(f"      OOS PF {oos_pf:.3f} > 1.15")
        print(f"      OOS improvement over baseline: ${oos_delta:+,.0f}")
        print(f"      PF ratio {pf_ratio:.2f} → {'robust' if pf_ratio >= 0.7 else 'acceptable'} stability")
        print(f"      Annual projection: ${annual_pnl:+,.0f} ({multiplier:.1f}x S&P)")
    elif oos_profitable and oos_adds_value:
        print(f"  ** MARGINAL: Edge exists but weaker out-of-sample")
        print(f"      OOS PF {oos_pf:.3f}")
        print(f"      PF ratio {pf_ratio:.2f}")
        print(f"      Deploy with smaller size, monitor closely")
    else:
        print(f"  * FAIL: Level Rejection edge does NOT persist out-of-sample")
        print(f"      OOS PF {oos_pf:.3f}")
        print(f"      OOS delta vs baseline: ${oos_delta:+,.0f}")
        print(f"      Do NOT deploy — likely overfit")

    print()


if __name__ == "__main__":
    main()
