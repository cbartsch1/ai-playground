#!/usr/bin/env python3
"""Regime Profiling — Fortress (Bull Credit Spread) — 5-State Walk-Forward.

Runs Fortress UNFILTERED, then post-hoc tags every trade with the
walk-forward regime at entry time (bias-free, 5-state HMM).

Per-regime breakdown with t-test p-values and BLOCK/ALLOW verdicts.

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    .venv/bin/python scripts/regime_profile_fortress_5state.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats

# Project roots
MEDALLION_ROOT = Path(__file__).parent.parent
AI_PLAYGROUND_ROOT = MEDALLION_ROOT.parent
SPX_OPTIONS_ROOT = AI_PLAYGROUND_ROOT.parent / "spx-options"


def _swap_path(project_root):
    """Set sys.path so only MEDALLION + the given project root are available."""
    cleaned = [p for p in sys.path if p not in (
        str(AI_PLAYGROUND_ROOT), str(SPX_OPTIONS_ROOT),
    )]
    sys.path[:] = cleaned

    if str(MEDALLION_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDALLION_ROOT))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    mods_to_remove = [k for k in sys.modules if k.startswith("backtester")]
    for k in mods_to_remove:
        del sys.modules[k]


# ═══════════════════════════════════════════════════════════════
#  LOAD WALK-FORWARD REGIME FILTER
# ═══════════════════════════════════════════════════════════════

def load_wf_regime_filter():
    """Load walk-forward regime labels (bias-free, 5-state)."""
    _swap_path(MEDALLION_ROOT)
    from models.wf_regime_api import WalkForwardRegimeFilter

    wf_path = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"
    if not wf_path.exists():
        print("ERROR: Walk-forward regimes not found.")
        print(f"  Expected: {wf_path}")
        sys.exit(1)

    print("=" * 80)
    print("  REGIME PROFILING — FORTRESS (Bull Credit Spread)")
    print("  Walk-Forward 5-State Labels (Bias-Free)")
    print("=" * 80)

    rf = WalkForwardRegimeFilter(str(wf_path))

    # Show regime distribution
    dist = rf.predictions["regime_label"].value_counts(normalize=True).sort_index()
    print(f"\n  Regime Distribution ({len(rf.predictions)} hourly bars):")
    for label, pct in dist.items():
        print(f"    {label:<25s} {pct:>6.1%}")

    return rf


# ═══════════════════════════════════════════════════════════════
#  RUN FORTRESS UNFILTERED
# ═══════════════════════════════════════════════════════════════

def run_fortress_unfiltered():
    """Run Fortress (BullCreditSpread) with NO regime filter."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.bull_credit_spread import BullCreditSpread
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions
    from backtester.metrics import compute_metrics

    print(f"\n{'=' * 80}")
    print(f"  RUNNING FORTRESS BACKTEST (UNFILTERED — no regime gate)")
    print(f"{'=' * 80}")

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    print(f"  Data: {len(df):,} bars, {df.index[0].date()} to {df.index[-1].date()}")

    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = BullCreditSpread()
    trades = strategy.run_backtest(df, vix_data=vix, regime_filter=None)

    print(f"  Total trades: {len(trades)}")
    if trades:
        pnls = [t.net_pnl for t in trades]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"  WR {wr:.1f}% | PF {pf:.2f} | P&L ${sum(pnls):+,.0f}")

    return trades


# ═══════════════════════════════════════════════════════════════
#  TAG TRADES WITH REGIME
# ═══════════════════════════════════════════════════════════════

def tag_trades_with_regime(rf, trades):
    """Post-hoc tag each trade with walk-forward regime at entry time."""
    tagged = []
    no_data_count = 0
    for t in trades:
        regime = rf.get_regime_at(t.entry_time)
        label = regime.get("label")
        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = "No Data (before model)"
            no_data_count += 1
        tagged.append((t, label))

    if no_data_count > 0:
        print(f"\n  Note: {no_data_count} trades before WF model coverage → tagged 'No Data'")

    return tagged


# ═══════════════════════════════════════════════════════════════
#  PER-REGIME BREAKDOWN WITH T-TEST
# ═══════════════════════════════════════════════════════════════

def compute_regime_breakdown(tagged_trades):
    """Compute per-regime stats with t-test p-value."""
    by_regime = defaultdict(list)

    for t, label in tagged_trades:
        by_regime[label].append(t.net_pnl)

    # Canonical order for 5-state
    regime_order = [
        "Crash (Panic)", "Bear Trend", "Accumulation (Chop)",
        "Recovery", "Bull Run (Trend)", "No Data (before model)",
    ]

    breakdown = {}
    for label in regime_order:
        if label not in by_regime:
            continue
        pnls = by_regime[label]
        n = len(pnls)
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100 if n > 0 else 0
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        avg = total_pnl / n if n > 0 else 0

        # One-sample t-test: is mean P&L significantly different from zero?
        if n >= 2:
            t_stat, p_val = stats.ttest_1samp(pnls, 0)
        else:
            t_stat, p_val = 0.0, 1.0

        breakdown[label] = {
            "trades": n,
            "pnl": total_pnl,
            "win_rate": wr,
            "avg_trade": avg,
            "profit_factor": pf,
            "gross_win": gross_win,
            "gross_loss": gross_loss,
            "t_stat": t_stat,
            "p_value": p_val,
            "pnls": pnls,
        }

    return breakdown


# ═══════════════════════════════════════════════════════════════
#  PRINT RESULTS
# ═══════════════════════════════════════════════════════════════

def print_regime_table(breakdown):
    """Print per-regime breakdown with verdict."""
    print(f"\n{'=' * 105}")
    print(f"  FORTRESS — Per-Regime P&L Breakdown (Walk-Forward 5-State)")
    print(f"{'=' * 105}")

    header = (f"  {'Regime':<25s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
              f"{'P&L':>12s} {'Avg Trade':>10s} {'t-stat':>8s} {'p-value':>9s}  {'Verdict':<8s}")
    print(header)
    print(f"  {'-' * 101}")

    for regime, d in breakdown.items():
        pf_str = f"{d['profit_factor']:.2f}" if d['profit_factor'] < 100 else "inf"
        p_str = f"{d['p_value']:.4f}" if d['p_value'] < 1.0 else "1.0000"

        # Verdict: BLOCK if P&L negative AND p < 0.05
        if d["pnl"] < 0 and d["p_value"] < 0.05:
            verdict = "BLOCK"
        else:
            verdict = "ALLOW"

        # Skip "No Data" from verdicts
        if regime == "No Data (before model)":
            verdict = "N/A"

        print(f"  {regime:<25s} {d['trades']:>7d} {d['win_rate']:>6.1f}% "
              f"{pf_str:>7s} ${d['pnl']:>10,.0f} ${d['avg_trade']:>8,.0f} "
              f"{d['t_stat']:>8.2f} {p_str:>9s}  {verdict:<8s}")

    # Totals
    total_trades = sum(d["trades"] for d in breakdown.values())
    total_pnl = sum(d["pnl"] for d in breakdown.values())
    total_wins = sum(d["trades"] * d["win_rate"] / 100 for d in breakdown.values())
    total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    total_gross_win = sum(d["gross_win"] for d in breakdown.values())
    total_gross_loss = sum(d["gross_loss"] for d in breakdown.values())
    total_pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else float("inf")
    total_avg = total_pnl / total_trades if total_trades > 0 else 0

    all_pnls = []
    for d in breakdown.values():
        all_pnls.extend(d["pnls"])
    if len(all_pnls) >= 2:
        t_all, p_all = stats.ttest_1samp(all_pnls, 0)
    else:
        t_all, p_all = 0.0, 1.0

    pf_str = f"{total_pf:.2f}" if total_pf < 100 else "inf"
    p_str = f"{p_all:.4f}" if p_all < 1.0 else "1.0000"

    print(f"  {'-' * 101}")
    print(f"  {'TOTAL':<25s} {total_trades:>7d} {total_wr:>6.1f}% "
          f"{pf_str:>7s} ${total_pnl:>10,.0f} ${total_avg:>8,.0f} "
          f"{t_all:>8.2f} {p_str:>9s}")


def print_summary(breakdown):
    """Print summary with blocking recommendations."""
    print(f"\n{'=' * 105}")
    print(f"  VERDICT SUMMARY")
    print(f"{'=' * 105}")

    blocked = []
    allowed = []

    for regime, d in breakdown.items():
        if regime == "No Data (before model)":
            continue
        if d["pnl"] < 0 and d["p_value"] < 0.05:
            blocked.append((regime, d))
        else:
            allowed.append((regime, d))

    if blocked:
        print(f"\n  BLOCK ({len(blocked)} regimes):")
        for regime, d in blocked:
            print(f"    {regime:<25s}  {d['trades']} trades, ${d['pnl']:+,.0f}, "
                  f"p={d['p_value']:.4f}")
        total_blocked_pnl = sum(d["pnl"] for _, d in blocked)
        total_blocked_trades = sum(d["trades"] for _, d in blocked)
        print(f"    → Removing {total_blocked_trades} trades bleeding ${total_blocked_pnl:+,.0f}")
    else:
        print(f"\n  No regimes meet BLOCK criteria (negative P&L AND p < 0.05)")

    print(f"\n  ALLOW ({len(allowed)} regimes):")
    for regime, d in allowed:
        p_note = f"p={d['p_value']:.4f}" if d['trades'] >= 2 else "too few trades"
        pnl_note = "profitable" if d["pnl"] >= 0 else "negative but NOT significant"
        print(f"    {regime:<25s}  {d['trades']} trades, ${d['pnl']:+,.0f}, "
              f"{p_note} — {pnl_note}")

    print(f"\n{'=' * 105}\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Step 1: Load walk-forward regime labels
    rf = load_wf_regime_filter()

    # Step 2: Run Fortress unfiltered
    trades = run_fortress_unfiltered()

    if not trades:
        print("\nERROR: No trades found.")
        sys.exit(1)

    # Step 3: Tag each trade with entry-time regime
    tagged = tag_trades_with_regime(rf, trades)

    # Step 4: Compute per-regime breakdown
    breakdown = compute_regime_breakdown(tagged)

    # Step 5: Print results
    print_regime_table(breakdown)
    print_summary(breakdown)


if __name__ == "__main__":
    main()
