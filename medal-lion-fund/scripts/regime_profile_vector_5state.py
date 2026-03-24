#!/usr/bin/env python3
"""Regime Profiling — Vector (EMA Cross Directional Best) with 5-State Walk-Forward Labels.

Runs Vector UNFILTERED, tags every trade with entry-time regime from the
NEW walk-forward 5-state regime labels, then computes per-regime P&L breakdown.

Walk-forward regimes (bias-free):
  - Crash (Panic)
  - Bear Trend
  - Accumulation (Chop)
  - Recovery
  - Bull Run (Trend)

Verdict: BLOCK if P&L negative AND p < 0.05. ALLOW otherwise.

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    .venv/bin/python scripts/regime_profile_vector_5state.py
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
#  LOAD WALK-FORWARD REGIME FILTER
# ═══════════════════════════════════════════════════════════════

def load_wf_regime_filter():
    """Load walk-forward 5-state regime labels."""
    _swap_path(MEDALLION_ROOT)
    from models.wf_regime_api import WalkForwardRegimeFilter

    wf_path = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"
    if not wf_path.exists():
        print(f"ERROR: Walk-forward regimes not found at {wf_path}")
        sys.exit(1)

    print("=" * 70)
    print("  LOADING WALK-FORWARD 5-STATE REGIME LABELS")
    print("=" * 70)

    rf = WalkForwardRegimeFilter(str(wf_path))
    labels = rf.predictions["regime_label"].value_counts()
    print(f"\n  Regime distribution (hourly bars):")
    for label, count in labels.items():
        print(f"    {label:<25s} {count:>5d} bars")

    return rf


# ═══════════════════════════════════════════════════════════════
#  RUN VECTOR UNFILTERED
# ═══════════════════════════════════════════════════════════════

def run_vector_unfiltered():
    """Run Vector (EMACrossDirectionalBest) with NO regime filter."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.ema_cross_directional import EMACrossDirectionalBest
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    print("\n" + "=" * 70)
    print("  RUNNING VECTOR (EMA Cross Dir Best) — UNFILTERED")
    print("=" * 70)

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    df_all_path = SPX_OPTIONS_ROOT / "data" / "spy_1m_all.parquet"
    df_all = None
    if df_all_path.exists():
        df_all = load_spy_data(str(df_all_path))
        df_all = tag_sessions(df_all)

    strategy = EMACrossDirectionalBest()
    trades = strategy.run_backtest(df, vix_data=vix, df_all_sessions=df_all)

    print(f"  Total trades: {len(trades)}")
    total_pnl = sum(t.net_pnl for t in trades)
    print(f"  Total P&L:    ${total_pnl:+,.0f}")

    return trades


# ═══════════════════════════════════════════════════════════════
#  TAG TRADES WITH REGIMES & COMPUTE PER-REGIME STATS
# ═══════════════════════════════════════════════════════════════

def tag_and_analyze(trades, rf):
    """Tag each trade with entry-time regime and compute per-regime breakdown."""

    # 5-state labels in display order
    REGIME_ORDER = [
        "Crash (Panic)",
        "Bear Trend",
        "Accumulation (Chop)",
        "Recovery",
        "Bull Run (Trend)",
    ]

    # Tag every trade
    tagged = []
    for t in trades:
        regime = rf.get_regime_at(t.entry_time)
        label = regime.get("label", "Unknown")
        tagged.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "net_pnl": t.net_pnl,
            "regime": label if label else "Unknown",
        })

    df = pd.DataFrame(tagged)

    # Per-regime stats
    regime_stats = []
    for regime in REGIME_ORDER:
        subset = df[df["regime"] == regime]
        n = len(subset)
        if n == 0:
            regime_stats.append({
                "regime": regime,
                "trades": 0,
                "pnl": 0.0,
                "pf": 0.0,
                "wr": 0.0,
                "avg_trade": 0.0,
                "t_stat": 0.0,
                "p_value": 1.0,
                "verdict": "NO TRADES",
            })
            continue

        pnls = subset["net_pnl"].values
        total_pnl = pnls.sum()
        wins = (pnls > 0).sum()
        wr = wins / n * 100
        gross_win = pnls[pnls > 0].sum()
        gross_loss = abs(pnls[pnls <= 0].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        avg_trade = total_pnl / n

        # One-sample t-test: H0 = mean P&L = 0
        if n >= 2 and pnls.std() > 0:
            t_stat, p_value = stats.ttest_1samp(pnls, 0)
        else:
            t_stat, p_value = 0.0, 1.0

        # Verdict logic
        if total_pnl < 0 and p_value < 0.05:
            verdict = "BLOCK"
        elif total_pnl < 0 and p_value >= 0.05:
            verdict = "ALLOW (not sig)"
        elif total_pnl >= 0:
            verdict = "ALLOW"
        else:
            verdict = "ALLOW"

        regime_stats.append({
            "regime": regime,
            "trades": n,
            "pnl": total_pnl,
            "pf": pf,
            "wr": wr,
            "avg_trade": avg_trade,
            "t_stat": t_stat,
            "p_value": p_value,
            "verdict": verdict,
        })

    # Also handle any trades with "Unknown" regime
    unknown = df[~df["regime"].isin(REGIME_ORDER)]
    if len(unknown) > 0:
        pnls = unknown["net_pnl"].values
        n = len(unknown)
        total_pnl = pnls.sum()
        wins = (pnls > 0).sum()
        wr = wins / n * 100
        gross_win = pnls[pnls > 0].sum()
        gross_loss = abs(pnls[pnls <= 0].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        avg_trade = total_pnl / n
        if n >= 2 and pnls.std() > 0:
            t_stat, p_value = stats.ttest_1samp(pnls, 0)
        else:
            t_stat, p_value = 0.0, 1.0
        regime_stats.append({
            "regime": "Unknown / No Label",
            "trades": n,
            "pnl": total_pnl,
            "pf": pf,
            "wr": wr,
            "avg_trade": avg_trade,
            "t_stat": t_stat,
            "p_value": p_value,
            "verdict": "CHECK",
        })

    return regime_stats, df


# ═══════════════════════════════════════════════════════════════
#  PRINT RESULTS
# ═══════════════════════════════════════════════════════════════

def print_results(regime_stats, df_tagged):
    """Print the per-regime P&L breakdown table."""

    print("\n")
    print("=" * 110)
    print("  VECTOR (EMA Cross Dir Best) — PER-REGIME P&L BREAKDOWN (5-State Walk-Forward)")
    print("=" * 110)

    print(f"\n  {'Regime':<25s} {'Trades':>7s} {'P&L':>12s} {'PF':>7s} {'WR':>7s} "
          f"{'Avg Trade':>11s} {'t-stat':>8s} {'p-value':>9s}  {'Verdict'}")
    print(f"  {'-' * 106}")

    for r in regime_stats:
        pf_str = f"{r['pf']:.2f}" if r['pf'] < 100 else "inf"
        p_str = f"{r['p_value']:.4f}" if r['trades'] > 0 else "  n/a"
        t_str = f"{r['t_stat']:+.2f}" if r['trades'] > 0 else "  n/a"

        # Highlight blocked regimes
        verdict = r['verdict']
        marker = " ***" if verdict == "BLOCK" else ""

        print(f"  {r['regime']:<25s} {r['trades']:>7d} ${r['pnl']:>10,.0f} {pf_str:>7s} "
              f"{r['wr']:>6.1f}% ${r['avg_trade']:>9,.0f} {t_str:>8s} {p_str:>9s}  "
              f"{verdict}{marker}")

    print(f"  {'-' * 106}")

    # Portfolio total
    all_pnls = df_tagged["net_pnl"].values
    n = len(all_pnls)
    total_pnl = all_pnls.sum()
    wins = (all_pnls > 0).sum()
    wr = wins / n * 100
    gross_win = all_pnls[all_pnls > 0].sum()
    gross_loss = abs(all_pnls[all_pnls <= 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_trade = total_pnl / n
    t_stat, p_value = stats.ttest_1samp(all_pnls, 0) if n >= 2 else (0, 1)
    pf_str = f"{pf:.2f}" if pf < 100 else "inf"

    print(f"  {'ALL TRADES':<25s} {n:>7d} ${total_pnl:>10,.0f} {pf_str:>7s} "
          f"{wr:>6.1f}% ${avg_trade:>9,.0f} {t_stat:>+8.2f} {p_value:>9.4f}")

    # Summary
    blocked = [r for r in regime_stats if r["verdict"] == "BLOCK"]
    print(f"\n{'=' * 110}")
    print(f"  VERDICT SUMMARY")
    print(f"{'=' * 110}")

    if blocked:
        print(f"\n  REGIMES TO BLOCK ({len(blocked)}):")
        for r in blocked:
            print(f"    - {r['regime']}: ${r['pnl']:+,.0f} P&L, p={r['p_value']:.4f}")
        potential_save = sum(r["pnl"] for r in blocked)
        print(f"\n  Potential P&L improvement from blocking: ${-potential_save:+,.0f}")
        remaining = total_pnl - potential_save
        remaining_trades = n - sum(r["trades"] for r in blocked)
        print(f"  Filtered P&L: ${remaining:+,.0f} ({remaining_trades} trades)")
    else:
        print(f"\n  NO REGIMES TO BLOCK.")
        print(f"  All regimes either profitable or losses not statistically significant.")
        print(f"  Vector runs unfiltered across all 5 walk-forward regime states.")

    print(f"\n{'=' * 110}\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # 1. Load walk-forward regime labels
    rf = load_wf_regime_filter()

    # 2. Run Vector unfiltered
    trades = run_vector_unfiltered()

    if not trades:
        print("\n  ERROR: No trades returned from Vector backtest.")
        sys.exit(1)

    # 3. Tag trades with entry-time regime and compute per-regime stats
    regime_stats, df_tagged = tag_and_analyze(trades, rf)

    # 4. Print results
    print_results(regime_stats, df_tagged)


if __name__ == "__main__":
    main()
