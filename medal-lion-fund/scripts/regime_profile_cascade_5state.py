#!/usr/bin/env python3
"""Cascade (Bear Breakdown) — Per-Regime P&L Breakdown using Walk-Forward Labels.

Runs Cascade UNFILTERED (no regime gating), then tags every trade with
its entry-time regime from the 5-state walk-forward labels. Groups by
regime and computes: trade count, total P&L, profit factor, win rate,
avg trade P&L, t-test p-value, and a BLOCK/ALLOW verdict.

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    .venv/bin/python scripts/regime_profile_cascade_5state.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# Project roots
MEDALLION_ROOT = Path(__file__).parent.parent
AI_PLAYGROUND_ROOT = MEDALLION_ROOT.parent
SPX_OPTIONS_ROOT = AI_PLAYGROUND_ROOT.parent / "spx-options"

WF_PARQUET = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"

REGIME_ORDER = [
    "Crash (Panic)",
    "Bear Trend",
    "Accumulation (Chop)",
    "Recovery",
    "Bull Run (Trend)",
]


def _swap_path(project_root):
    """Set sys.path so only MEDALLION + the given project root are available."""
    cleaned = [p for p in sys.path if p not in (
        str(AI_PLAYGROUND_ROOT), str(SPX_OPTIONS_ROOT)
    )]
    sys.path[:] = cleaned

    if str(MEDALLION_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDALLION_ROOT))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    mods_to_remove = [k for k in sys.modules if k.startswith("backtester")]
    for k in mods_to_remove:
        del sys.modules[k]


def run_cascade_unfiltered():
    """Run Cascade (Bear Breakdown) with NO regime filter — all trades."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.bear_breakdown import BearBreakdown
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = BearBreakdown()
    trades = strategy.run_backtest(df, vix_data=vix, regime_filter=None)
    return trades


def load_wf_regime_filter():
    """Load WalkForwardRegimeFilter from parquet."""
    _swap_path(MEDALLION_ROOT)
    from models.wf_regime_api import WalkForwardRegimeFilter
    return WalkForwardRegimeFilter(str(WF_PARQUET))


def tag_trades_with_regime(trades, rf):
    """For each trade, look up the regime at entry time. Returns list of dicts."""
    tagged = []
    for t in trades:
        entry_time = t.entry_time
        pnl = t.net_pnl
        regime_info = rf.get_regime_at(entry_time)
        label = regime_info.get("label", "Unknown")
        tagged.append({
            "entry_time": entry_time,
            "pnl": pnl,
            "regime": label if label else "Unknown",
        })
    return tagged


def compute_regime_stats(tagged_trades):
    """Group trades by regime. Compute stats for each group."""
    groups = defaultdict(list)
    for t in tagged_trades:
        groups[t["regime"]].append(t["pnl"])

    results = []
    for regime in REGIME_ORDER:
        pnls = groups.get(regime, [])
        if not pnls:
            results.append({
                "regime": regime,
                "trades": 0,
                "pnl": 0,
                "pf": 0,
                "wr": 0,
                "avg_trade": 0,
                "p_value": 1.0,
                "verdict": "NO TRADES",
            })
            continue

        n = len(pnls)
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        avg_trade = total_pnl / n

        # One-sample t-test: H0 = mean PnL is 0
        if n >= 2:
            t_stat, p_value = stats.ttest_1samp(pnls, 0)
            p_value = float(p_value)
        else:
            p_value = 1.0

        # Verdict: BLOCK if P&L negative AND p < 0.05
        if total_pnl < 0 and p_value < 0.05:
            verdict = "BLOCK"
        else:
            verdict = "ALLOW"

        results.append({
            "regime": regime,
            "trades": n,
            "pnl": total_pnl,
            "pf": pf,
            "wr": wr,
            "avg_trade": avg_trade,
            "p_value": p_value,
            "verdict": verdict,
        })

    # Handle any unknown/unmapped regimes
    for regime, pnls in groups.items():
        if regime not in REGIME_ORDER:
            n = len(pnls)
            total_pnl = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / n * 100 if n > 0 else 0
            gross_win = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p <= 0))
            pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
            avg_trade = total_pnl / n if n > 0 else 0
            if n >= 2:
                _, p_value = stats.ttest_1samp(pnls, 0)
                p_value = float(p_value)
            else:
                p_value = 1.0
            verdict = "BLOCK" if total_pnl < 0 and p_value < 0.05 else "ALLOW"
            results.append({
                "regime": regime,
                "trades": n,
                "pnl": total_pnl,
                "pf": pf,
                "wr": wr,
                "avg_trade": avg_trade,
                "p_value": p_value,
                "verdict": verdict,
            })

    return results


def print_report(regime_stats, total_trades, total_pnl):
    """Print the formatted report."""
    print()
    print("CASCADE (Bear Breakdown) — Per-Regime P&L Breakdown (Walk-Forward Labels)")
    print("=" * 95)

    header = (f"  {'Regime':<22s} {'Trades':>7s} {'P&L':>12s} {'PF':>7s} "
              f"{'WR':>7s} {'Avg Trade':>11s} {'p-value':>9s}   {'Verdict'}")
    print(header)
    print("  " + "-" * 91)

    block_list = []
    allow_list = []

    for r in regime_stats:
        if r["trades"] == 0:
            print(f"  {r['regime']:<22s} {'--':>7s} {'--':>12s} {'--':>7s} "
                  f"{'--':>7s} {'--':>11s} {'--':>9s}   {r['verdict']}")
            continue

        pf_str = f"{r['pf']:.2f}" if r['pf'] < 100 else "inf"
        p_str = f"{r['p_value']:.4f}" if r['p_value'] < 1.0 else "1.0000"

        print(f"  {r['regime']:<22s} {r['trades']:>7d} ${r['pnl']:>10,.0f} "
              f"{pf_str:>7s} {r['wr']:>6.1f}% ${r['avg_trade']:>9,.0f} "
              f"{p_str:>9s}   {r['verdict']}")

        if r["verdict"] == "BLOCK":
            block_list.append(r["regime"])
        elif r["verdict"] == "ALLOW":
            allow_list.append(r["regime"])

    print("  " + "-" * 91)
    print(f"  {'TOTAL':<22s} {total_trades:>7d} ${total_pnl:>10,.0f}")

    print()
    print(f"RECOMMENDATION:  Block = {{{', '.join(block_list) if block_list else 'none'}}}")
    print(f"                 Allow = {{{', '.join(allow_list) if allow_list else 'none'}}}")
    print()


def main():
    print("=" * 70)
    print("  CASCADE (Bear Breakdown) — REGIME PROFILING")
    print("  Walk-Forward 5-State Labels (bias-free)")
    print("  Running UNFILTERED — tagging trades post-hoc")
    print("=" * 70)

    # Load walk-forward regime filter
    print("\n  Loading walk-forward regime labels...")
    rf = load_wf_regime_filter()

    # Run Cascade unfiltered
    print("\n  Running Cascade (Bear Breakdown) unfiltered...")
    trades = run_cascade_unfiltered()
    print(f"  Total trades: {len(trades)}")

    # Tag each trade with its entry-time regime
    print("  Tagging trades with entry-time regime...")
    tagged = tag_trades_with_regime(trades, rf)

    total_pnl = sum(t["pnl"] for t in tagged)
    print(f"  Total P&L: ${total_pnl:+,.0f}")

    # Compute per-regime stats
    regime_stats = compute_regime_stats(tagged)

    # Print report
    print_report(regime_stats, len(tagged), total_pnl)


if __name__ == "__main__":
    main()
