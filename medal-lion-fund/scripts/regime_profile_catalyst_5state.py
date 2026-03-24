#!/usr/bin/env python3
"""Regime Profiling — Catalyst (Home Run) with 5-State Walk-Forward Labels.

Runs Catalyst (Home Run / 30m Structure Break → 0DTE Puts) UNFILTERED,
tags every trade with its entry-time regime from the walk-forward labels,
and computes per-regime P&L breakdown with statistical significance.

Walk-forward labels are bias-free: each bar's regime was computed by a
model that only saw prior data (no look-ahead).

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    .venv/bin/python scripts/regime_profile_catalyst_5state.py
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


# ═══════════════════════════════════════════════════════════════
#  LOAD WALK-FORWARD REGIME LABELS
# ═══════════════════════════════════════════════════════════════

def load_wf_regime_filter():
    """Load pre-computed walk-forward regime labels (bias-free)."""
    _swap_path(MEDALLION_ROOT)
    from models.wf_regime_api import WalkForwardRegimeFilter

    wf_path = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"
    if not wf_path.exists():
        print(f"ERROR: Walk-forward regimes not found at {wf_path}")
        sys.exit(1)

    print("=" * 70)
    print("  LOADING WALK-FORWARD REGIME LABELS (5-state, bias-free)")
    print("=" * 70)

    rf = WalkForwardRegimeFilter(str(wf_path))

    dist = rf.predictions["regime_label"].value_counts(normalize=True).sort_index()
    print(f"\n  Regime Distribution ({len(rf.predictions):,} hourly bars):")
    for label, pct in dist.items():
        print(f"    {label:<25s} {pct:>6.1%}")

    return rf


# ═══════════════════════════════════════════════════════════════
#  RUN CATALYST (HOME RUN) — UNFILTERED
# ═══════════════════════════════════════════════════════════════

def run_catalyst_unfiltered():
    """Run Catalyst (Home Run) with NO regime filter — all trades taken."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.structure_break import HomeRun
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    print(f"\n{'=' * 70}")
    print(f"  RUNNING CATALYST (Home Run) — UNFILTERED")
    print(f"{'=' * 70}")

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)

    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    # Load all-sessions data for AT scoring
    df_all_path = SPX_OPTIONS_ROOT / "data" / "spy_1m_all.parquet"
    df_all = None
    if df_all_path.exists():
        df_all = load_spy_data(str(df_all_path))
        df_all = tag_sessions(df_all)

    strategy = HomeRun()
    # NO regime_filter passed — takes every trade
    trades = strategy.run_backtest(df, vix_data=vix, df_all_sessions=df_all)

    if trades:
        pnls = [t.net_pnl for t in trades]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"  Trades: {len(trades)}")
        print(f"  P&L: ${sum(pnls):+,.0f}  |  WR: {wr:.1f}%  |  PF: {pf:.2f}")
        print(f"  Date range: {trades[0].entry_time.date()} to {trades[-1].entry_time.date()}")

    return trades


# ═══════════════════════════════════════════════════════════════
#  TAG TRADES WITH REGIME
# ═══════════════════════════════════════════════════════════════

def tag_trades_with_regime(rf, trades):
    """Post-hoc tag each trade with WF regime at entry time."""
    tagged = []
    for t in trades:
        regime = rf.get_regime_at(t.entry_time)
        label = regime.get("label")
        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = "No Data (before model)"
        tagged.append((t, label))
    return tagged


# ═══════════════════════════════════════════════════════════════
#  COMPUTE PER-REGIME BREAKDOWN WITH T-TEST
# ═══════════════════════════════════════════════════════════════

def compute_regime_breakdown(tagged_trades):
    """Compute per-regime stats including t-test p-value."""
    by_regime = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0, "pnls": []})

    for t, label in tagged_trades:
        d = by_regime[label]
        d["trades"] += 1
        d["pnl"] += t.net_pnl
        d["pnls"].append(t.net_pnl)
        if t.net_pnl > 0:
            d["wins"] += 1

    breakdown = {}
    for label, d in by_regime.items():
        pnls = np.array(d["pnls"])
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
        avg = d["pnl"] / d["trades"] if d["trades"] > 0 else 0

        # T-test: is average P&L significantly different from zero?
        if len(pnls) >= 2:
            t_stat, p_value = stats.ttest_1samp(pnls, 0)
        else:
            t_stat, p_value = 0.0, 1.0

        breakdown[label] = {
            "trades": d["trades"],
            "pnl": d["pnl"],
            "win_rate": wr,
            "avg_trade": avg,
            "profit_factor": pf,
            "gross_win": gross_win,
            "gross_loss": gross_loss,
            "t_stat": t_stat,
            "p_value": p_value,
        }

    return breakdown


# ═══════════════════════════════════════════════════════════════
#  PRINT RESULTS
# ═══════════════════════════════════════════════════════════════

# Canonical ordering for 5-state model
REGIME_ORDER = [
    "Crash (Panic)",
    "Bear Trend",
    "Accumulation (Chop)",
    "Recovery",
    "Bull Run (Trend)",
    "No Data (before model)",
]


def print_results(breakdown):
    """Print the per-regime P&L breakdown table."""
    print(f"\nCATALYST (Home Run) — Per-Regime P&L Breakdown (Walk-Forward Labels)")
    print("=" * 95)

    header = (f"{'Regime':<23s} {'Trades':>6s}  {'P&L':>10s}  {'PF':>6s}  "
              f"{'WR':>6s}  {'Avg Trade':>10s}  {'p-value':>8s}  {'Verdict':<8s}")
    print(header)
    print("-" * 95)

    # Print in canonical order, skip missing
    for regime in REGIME_ORDER:
        if regime not in breakdown:
            continue
        d = breakdown[regime]
        pf_str = f"{d['profit_factor']:.2f}" if d['profit_factor'] < 100 else "inf"
        p_str = f"{d['p_value']:.3f}" if d['p_value'] < 1.0 else "1.000"

        # Verdict: BLOCK if P&L < 0 AND p < 0.05
        if d["pnl"] < 0 and d["p_value"] < 0.05:
            verdict = "BLOCK"
        else:
            verdict = "ALLOW"

        print(f"{regime:<23s} {d['trades']:>6d}  ${d['pnl']:>9,.0f}  {pf_str:>6s}  "
              f"{d['win_rate']:>5.1f}%  ${d['avg_trade']:>9,.0f}  {p_str:>8s}  {verdict:<8s}")

    # Total row
    total_trades = sum(d["trades"] for d in breakdown.values())
    total_pnl = sum(d["pnl"] for d in breakdown.values())
    total_wins = sum(d["trades"] * d["win_rate"] / 100 for d in breakdown.values())
    total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    total_gross_win = sum(d["gross_win"] for d in breakdown.values())
    total_gross_loss = sum(d["gross_loss"] for d in breakdown.values())
    total_pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else float("inf")
    total_avg = total_pnl / total_trades if total_trades > 0 else 0
    pf_str = f"{total_pf:.2f}" if total_pf < 100 else "inf"

    print("-" * 95)
    print(f"{'TOTAL (unfiltered)':<23s} {total_trades:>6d}  ${total_pnl:>9,.0f}  {pf_str:>6s}  "
          f"{total_wr:>5.1f}%  ${total_avg:>9,.0f}")

    # Recommendation
    blocked = []
    allowed = []
    for regime in REGIME_ORDER:
        if regime not in breakdown or regime == "No Data (before model)":
            continue
        d = breakdown[regime]
        if d["pnl"] < 0 and d["p_value"] < 0.05:
            blocked.append(regime)
        else:
            allowed.append(regime)

    print(f"\n{'=' * 95}")
    print(f"RECOMMENDATION:")
    if blocked:
        print(f"  Block  = {{{', '.join(blocked)}}}")
        blocked_pnl = sum(breakdown[r]["pnl"] for r in blocked)
        blocked_trades = sum(breakdown[r]["trades"] for r in blocked)
        print(f"           ({blocked_trades} trades, ${blocked_pnl:+,.0f} — statistically significant losers)")
    else:
        print(f"  Block  = {{}} (none)")
    print(f"  Allow  = {{{', '.join(allowed)}}}")
    print(f"{'=' * 95}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Step 1: Load walk-forward regime labels
    rf = load_wf_regime_filter()

    # Step 2: Run Catalyst unfiltered
    trades = run_catalyst_unfiltered()
    if not trades:
        print("\nERROR: No trades returned. Check data paths.")
        sys.exit(1)

    # Step 3: Tag every trade with its entry-time regime
    tagged = tag_trades_with_regime(rf, trades)

    # Quick check — how many trades have regime data?
    no_data = sum(1 for _, label in tagged if label == "No Data (before model)")
    print(f"\n  Trades with regime data: {len(tagged) - no_data} / {len(tagged)}")
    if no_data > 0:
        print(f"  Trades before model range: {no_data} (excluded from verdict)")

    # Step 4: Compute per-regime breakdown
    breakdown = compute_regime_breakdown(tagged)

    # Step 5: Print results
    print_results(breakdown)


if __name__ == "__main__":
    main()
