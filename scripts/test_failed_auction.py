#!/usr/bin/env python3
"""Test Failed Auction (FA) setup — Dalton's IB breakout failure reversal.

Tests FA in isolation and combined with Market Structure:
  1. FA Default (both directions, no MA filter)
  2. FA Short-only
  3. FA + SMA confirmation
  4. FA + MS Winners (combined)
  5. FA parameter sweep: max_break_bars, stop_buffer, min_rr

Includes walk-forward validation (Year 1 in-sample, Year 2 out-of-sample).

Usage:
    python scripts/test_failed_auction.py data/es_5m_databento_2yr.csv
"""

import argparse
import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


# ═══════════════════════════════════════════════════════════════
# Config Builders
# ═══════════════════════════════════════════════════════════════

def _fa_base() -> StrategyConfig:
    """Base config with ONLY FA enabled (all other setups OFF)."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"
    # Disable all other setups
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_ms = False
    # Enable FA
    cfg.use_fa = True
    cfg.fa_max_break_bars = 6
    cfg.fa_stop_buffer = 3.0
    cfg.fa_min_rr = 0.5
    cfg.fa_max_risk = 20.0
    cfg.fa_require_ma = False
    cfg.fa_ma_type = "sma"
    cfg.max_fa_trades = 2
    # Standard filters
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    return cfg


def make_fa_default():
    """FA Default — both directions, no MA filter."""
    return _fa_base()


def make_fa_short_only():
    """FA Short-only — only fade upside breakout failures."""
    cfg = _fa_base()
    cfg.direction_filter = "short"
    return cfg


def make_fa_long_only():
    """FA Long-only — only fade downside breakout failures."""
    cfg = _fa_base()
    cfg.direction_filter = "long"
    return cfg


def make_fa_sma_confirm():
    """FA + SMA 8/24 confirmation."""
    cfg = _fa_base()
    cfg.fa_require_ma = True
    cfg.fa_ma_type = "sma"
    return cfg


def make_fa_tema_confirm():
    """FA + TEMA 9/21 confirmation."""
    cfg = _fa_base()
    cfg.fa_require_ma = True
    cfg.fa_ma_type = "tema"
    return cfg


def make_ms_default():
    """MS with all levels, SMA 8/24 — reference baseline."""
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
    cfg.use_fa = False
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
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    return cfg


def make_fa_plus_ms():
    """FA + MS Combined — both setups enabled."""
    cfg = make_ms_default()
    cfg.use_fa = True
    cfg.fa_max_break_bars = 6
    cfg.fa_stop_buffer = 3.0
    cfg.fa_min_rr = 0.5
    cfg.fa_max_risk = 20.0
    cfg.fa_require_ma = False
    cfg.max_fa_trades = 2
    return cfg


def make_fa_sweep(max_break_bars, stop_buffer, min_rr):
    """FA with specific parameter values for sweep."""
    cfg = _fa_base()
    cfg.fa_max_break_bars = max_break_bars
    cfg.fa_stop_buffer = stop_buffer
    cfg.fa_min_rr = min_rr
    return cfg


# ═══════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════

def print_metrics(label, trades, show_detail=False):
    """Print metrics for a set of trades."""
    if not trades:
        print(f"  {label:<45s}  NO TRADES")
        return None

    m = compute_metrics(trades)
    print(f"  {label:<45s}  {m.total_trades:>5d} trades  "
          f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+10,.0f}  DD ${m.max_drawdown:>8,.0f}  "
          f"Sharpe {m.sharpe:>5.2f}  "
          f"Avg ${m.avg_trade:>+7,.0f}")

    if show_detail and trades:
        breakdown = per_setup_breakdown(trades)
        for setup, sm in sorted(breakdown.items()):
            print(f"    {setup:<22s}  {sm.total_trades:>5d}  "
                  f"WR {sm.win_rate:>5.1f}%  PF {sm.profit_factor:>6.3f}  "
                  f"P&L ${sm.net_pnl:>+10,.0f}  Avg ${sm.avg_trade:>+7,.0f}")

        # Direction breakdown
        longs = [t for t in trades if t.direction == 1]
        shorts = [t for t in trades if t.direction == -1]
        if longs:
            lm = compute_metrics(longs)
            print(f"    {'LONGS':<22s}  {lm.total_trades:>5d}  "
                  f"WR {lm.win_rate:>5.1f}%  PF {lm.profit_factor:>6.3f}  "
                  f"P&L ${lm.net_pnl:>+10,.0f}")
        if shorts:
            sm_s = compute_metrics(shorts)
            print(f"    {'SHORTS':<22s}  {sm_s.total_trades:>5d}  "
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


def print_summary_table(variants):
    """Print a clean summary table."""
    print(f"\n  {'Variant':<40s} {'Trades':>7s} {'WR':>6s} {'PF':>7s} {'P&L':>12s} "
          f"{'DD':>10s} {'Sharpe':>7s} {'Avg':>8s}")
    print(f"  {'-'*100}")
    for name, m, _ in variants:
        if m:
            print(f"  {name:<40s} {m.total_trades:>7d} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
                  f"{'${:,.0f}'.format(m.net_pnl):>12s} {'${:,.0f}'.format(m.max_drawdown):>10s} "
                  f"{m.sharpe:>7.2f} {'${:,.0f}'.format(m.avg_trade):>8s}")
        else:
            print(f"  {name:<40s}  NO TRADES")


def print_stat_tests(variants):
    """Print statistical significance tests."""
    print(f"\n  --- STATISTICAL TESTS ---")
    for name, m, trades in variants:
        if trades and len(trades) >= 10:
            pnls = [t.pnl_dollar for t in trades]
            t_stat, p_val = stats.ttest_1samp(pnls, 0)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"  {name:<40s}  n={len(trades):>5d}  t={t_stat:>6.3f}  p={p_val:.4f} {sig}")
        elif trades:
            print(f"  {name:<40s}  too few trades ({len(trades)})")
        else:
            print(f"  {name:<40s}  no trades")


# ═══════════════════════════════════════════════════════════════
# Main Test Routines
# ═══════════════════════════════════════════════════════════════

def run_core_variants(df, label="FULL 2-YEAR"):
    """Run core FA variants and MS comparison."""
    print(f"\n{'='*120}")
    print(f"  {label} — CORE VARIANTS")
    print(f"{'='*120}")

    variants = []

    # 1. FA Default
    cfg = make_fa_default()
    trades = run_backtest(df.copy(), cfg)
    m = print_metrics("FA Default (both, no MA)", trades, show_detail=True)
    variants.append(("FA Default", m, trades))

    # 2. FA Short-only
    cfg = make_fa_short_only()
    trades = run_backtest(df.copy(), cfg)
    m = print_metrics("FA Short-only", trades, show_detail=True)
    variants.append(("FA Short-only", m, trades))

    # 3. FA Long-only
    cfg = make_fa_long_only()
    trades = run_backtest(df.copy(), cfg)
    m = print_metrics("FA Long-only", trades, show_detail=True)
    variants.append(("FA Long-only", m, trades))

    # 4. FA + SMA confirmation
    cfg = make_fa_sma_confirm()
    trades = run_backtest(df.copy(), cfg)
    m = print_metrics("FA + SMA 8/24 confirm", trades, show_detail=True)
    variants.append(("FA + SMA", m, trades))

    # 5. FA + TEMA confirmation
    cfg = make_fa_tema_confirm()
    trades = run_backtest(df.copy(), cfg)
    m = print_metrics("FA + TEMA 9/21 confirm", trades, show_detail=True)
    variants.append(("FA + TEMA", m, trades))

    # 6. MS Default (reference)
    cfg = make_ms_default()
    trades = run_backtest(df.copy(), cfg)
    m = print_metrics("MS Default (reference)", trades, show_detail=False)
    variants.append(("MS Default (ref)", m, trades))

    # 7. FA + MS Combined
    cfg = make_fa_plus_ms()
    trades = run_backtest(df.copy(), cfg)
    m = print_metrics("FA + MS Combined", trades, show_detail=True)
    variants.append(("FA + MS Combined", m, trades))

    print_summary_table(variants)
    print_stat_tests(variants)

    return variants


def run_parameter_sweep(df, label="FULL 2-YEAR"):
    """Run parameter sweep over max_break_bars, stop_buffer, min_rr."""
    print(f"\n{'='*120}")
    print(f"  {label} — PARAMETER SWEEP")
    print(f"{'='*120}")

    max_break_bars_vals = [4, 6, 8, 10]
    stop_buffer_vals = [2.0, 3.0, 5.0]
    min_rr_vals = [0.3, 0.5, 0.8]

    results = []

    print(f"\n  {'break_bars':>10s} {'stop_buf':>9s} {'min_rr':>7s} | "
          f"{'Trades':>7s} {'WR':>6s} {'PF':>7s} {'P&L':>12s} {'DD':>10s} {'Sharpe':>7s} {'Avg':>8s}")
    print(f"  {'-'*100}")

    for bb, sb, rr in itertools.product(max_break_bars_vals, stop_buffer_vals, min_rr_vals):
        cfg = make_fa_sweep(bb, sb, rr)
        trades = run_backtest(df.copy(), cfg)
        if trades:
            m = compute_metrics(trades)
            results.append((bb, sb, rr, m, trades))
            print(f"  {bb:>10d} {sb:>9.1f} {rr:>7.1f} | "
                  f"{m.total_trades:>7d} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
                  f"{'${:,.0f}'.format(m.net_pnl):>12s} {'${:,.0f}'.format(m.max_drawdown):>10s} "
                  f"{m.sharpe:>7.2f} {'${:,.0f}'.format(m.avg_trade):>8s}")
        else:
            results.append((bb, sb, rr, None, []))
            print(f"  {bb:>10d} {sb:>9.1f} {rr:>7.1f} |  NO TRADES")

    # Find best by PF (min 15 trades)
    valid = [(bb, sb, rr, m, t) for bb, sb, rr, m, t in results if m and m.total_trades >= 15]
    if valid:
        best_pf = max(valid, key=lambda x: x[3].profit_factor)
        best_pnl = max(valid, key=lambda x: x[3].net_pnl)
        best_sharpe = max(valid, key=lambda x: x[3].sharpe)
        print(f"\n  BEST by PF:     break_bars={best_pf[0]}, stop_buf={best_pf[1]}, min_rr={best_pf[2]}  "
              f"-> PF {best_pf[3].profit_factor:.3f}, P&L ${best_pf[3].net_pnl:,.0f}, "
              f"{best_pf[3].total_trades} trades")
        print(f"  BEST by P&L:    break_bars={best_pnl[0]}, stop_buf={best_pnl[1]}, min_rr={best_pnl[2]}  "
              f"-> PF {best_pnl[3].profit_factor:.3f}, P&L ${best_pnl[3].net_pnl:,.0f}, "
              f"{best_pnl[3].total_trades} trades")
        print(f"  BEST by Sharpe: break_bars={best_sharpe[0]}, stop_buf={best_sharpe[1]}, min_rr={best_sharpe[2]}  "
              f"-> Sharpe {best_sharpe[3].sharpe:.2f}, P&L ${best_sharpe[3].net_pnl:,.0f}, "
              f"{best_sharpe[3].total_trades} trades")

    return results


def run_walk_forward(df, split_date="2025-02-14"):
    """Walk-forward validation: Year 1 (IS) vs Year 2 (OOS)."""
    print(f"\n{'='*120}")
    print(f"  WALK-FORWARD VALIDATION (split: {split_date})")
    print(f"{'='*120}")

    split_idx = df.index.get_indexer([split_date], method="nearest")[0]
    if split_idx <= 0 or split_idx >= len(df):
        split_idx = len(df) // 2

    df_y1 = df.iloc[:split_idx].copy()
    df_y2 = df.iloc[split_idx:].copy()

    print(f"  Year 1 (IS):  {df_y1.index[0].date()} to {df_y1.index[-1].date()} ({len(df_y1)} bars)")
    print(f"  Year 2 (OOS): {df_y2.index[0].date()} to {df_y2.index[-1].date()} ({len(df_y2)} bars)")

    configs = [
        ("FA Default", make_fa_default()),
        ("FA Short-only", make_fa_short_only()),
        ("FA + SMA", make_fa_sma_confirm()),
        ("FA + MS Combined", make_fa_plus_ms()),
    ]

    wf_results = []

    for name, cfg in configs:
        print(f"\n  --- {name} ---")
        t_y1 = run_backtest(df_y1.copy(), cfg)
        t_y2 = run_backtest(df_y2.copy(), cfg)

        m_y1 = None
        m_y2 = None

        if t_y1:
            m_y1 = compute_metrics(t_y1)
            print(f"    IS:   {m_y1.total_trades:>5d} trades  WR {m_y1.win_rate:>5.1f}%  "
                  f"PF {m_y1.profit_factor:>6.3f}  P&L ${m_y1.net_pnl:>+10,.0f}  "
                  f"DD ${m_y1.max_drawdown:>8,.0f}  Sharpe {m_y1.sharpe:>5.2f}")
        else:
            print(f"    IS:   NO TRADES")

        if t_y2:
            m_y2 = compute_metrics(t_y2)
            print(f"    OOS:  {m_y2.total_trades:>5d} trades  WR {m_y2.win_rate:>5.1f}%  "
                  f"PF {m_y2.profit_factor:>6.3f}  P&L ${m_y2.net_pnl:>+10,.0f}  "
                  f"DD ${m_y2.max_drawdown:>8,.0f}  Sharpe {m_y2.sharpe:>5.2f}")
        else:
            print(f"    OOS:  NO TRADES")

        if m_y1 and m_y2 and m_y1.profit_factor > 0:
            ratio = m_y2.profit_factor / m_y1.profit_factor
            verdict = "PASS" if ratio > 0.7 else "MARGINAL" if ratio > 0.5 else "FAIL"
            print(f"    PF ratio (OOS/IS): {ratio:.2f}  -> {verdict}")

            if t_y2 and len(t_y2) >= 5:
                _, p_val = stats.ttest_1samp([t.pnl_dollar for t in t_y2], 0)
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                print(f"    OOS p-value: {p_val:.4f} {sig}")
        else:
            print(f"    PF ratio: N/A")

        wf_results.append((name, m_y1, m_y2, t_y1, t_y2))

    # Summary table
    print(f"\n  {'Variant':<30s} | {'IS Trades':>9s} {'IS PF':>7s} {'IS P&L':>12s} | "
          f"{'OOS Trades':>10s} {'OOS PF':>7s} {'OOS P&L':>12s} | {'Ratio':>6s} {'Verdict':>8s}")
    print(f"  {'-'*115}")
    for name, m1, m2, _, _ in wf_results:
        if m1 and m2 and m1.profit_factor > 0:
            ratio = m2.profit_factor / m1.profit_factor
            verdict = "PASS" if ratio > 0.7 else "MARGINAL" if ratio > 0.5 else "FAIL"
            print(f"  {name:<30s} | {m1.total_trades:>9d} {m1.profit_factor:>7.3f} "
                  f"{'${:,.0f}'.format(m1.net_pnl):>12s} | {m2.total_trades:>10d} "
                  f"{m2.profit_factor:>7.3f} {'${:,.0f}'.format(m2.net_pnl):>12s} | "
                  f"{ratio:>6.2f} {verdict:>8s}")
        elif m1:
            print(f"  {name:<30s} | {m1.total_trades:>9d} {m1.profit_factor:>7.3f} "
                  f"{'${:,.0f}'.format(m1.net_pnl):>12s} | {'NO TRADES':>10s} "
                  f"{'':>7s} {'':>12s} | {'N/A':>6s} {'':>8s}")
        else:
            print(f"  {name:<30s} | {'NO TRADES':>9s} {'':>7s} {'':>12s} | "
                  f"{'':>10s} {'':>7s} {'':>12s} | {'N/A':>6s} {'':>8s}")

    return wf_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Failed Auction Setup Test")
    parser.add_argument("csv_file", help="Path to CSV data file")
    parser.add_argument("--split-date", type=str, default="2025-02-14",
                        help="Walk-forward split date (default: 2025-02-14)")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip parameter sweep (faster)")
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"RTH bars: {df['is_rth'].sum()}, Sessions: {df['new_rth'].sum()}")

    # 1. Core variants on full dataset
    core_variants = run_core_variants(df, "FULL 2-YEAR")

    # 2. Parameter sweep
    if not args.skip_sweep:
        sweep_results = run_parameter_sweep(df, "FULL 2-YEAR")

    # 3. Walk-forward validation
    wf_results = run_walk_forward(df, split_date=args.split_date)

    # Final verdict
    print(f"\n{'='*120}")
    print(f"  FINAL VERDICT")
    print(f"{'='*120}")
    print(f"  Review the results above. Key things to look for:")
    print(f"  - Does FA generate a meaningful number of trades (>30 over 2 years)?")
    print(f"  - Is the profit factor > 1.0 (edge exists)?")
    print(f"  - Does the walk-forward PF ratio exceed 0.7 (not overfit)?")
    print(f"  - Is the p-value < 0.10 (statistically significant)?")
    print(f"  - Does combining FA + MS improve on either alone?")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
