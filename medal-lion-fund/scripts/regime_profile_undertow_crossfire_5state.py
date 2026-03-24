#!/usr/bin/env python3
"""Regime Profile: Undertow & Crossfire (5-state walk-forward)

Runs both strategies UNFILTERED, tags every trade with its entry-time regime
from the walk-forward labels, and computes per-regime P&L breakdown.

Crossfire is split by side:
  - Call credit (direction="short") = bearish side
  - Put credit (direction="long")  = bullish side

Verdict per regime:
  BLOCK  if P&L negative AND p < 0.05 (statistically significant loser)
  ALLOW  otherwise

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    .venv/bin/python scripts/regime_profile_undertow_crossfire_5state.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Project paths ──
MEDALLION_ROOT = Path(__file__).parent.parent
AI_PLAYGROUND_ROOT = MEDALLION_ROOT.parent
SPX_OPTIONS_ROOT = AI_PLAYGROUND_ROOT.parent / "spx-options"

WF_PATH = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"

REGIME_NAMES = [
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
    # Clear backtester modules to avoid cross-project conflicts
    mods_to_remove = [k for k in sys.modules if k.startswith("backtester")]
    for k in mods_to_remove:
        del sys.modules[k]


# ═══════════════════════════════════════════════════════════════
#  LOAD WALK-FORWARD REGIME FILTER
# ═══════════════════════════════════════════════════════════════

def load_wf_regime_filter():
    """Load walk-forward regime labels as WalkForwardRegimeFilter."""
    _swap_path(MEDALLION_ROOT)
    from models.wf_regime_api import WalkForwardRegimeFilter
    rf = WalkForwardRegimeFilter(str(WF_PATH))
    return rf


# ═══════════════════════════════════════════════════════════════
#  STRATEGY RUNNERS (UNFILTERED)
# ═══════════════════════════════════════════════════════════════

def run_undertow_unfiltered():
    """Run Undertow (AMT-TEMA / ES IB Breakout Short) with NO regime filter."""
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.config import StrategyConfig
    from backtester.data_loader import load_tos_csv
    from backtester.engine import run_backtest

    data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento_2yr.csv"
    df = load_tos_csv(str(data_path), instrument="ES")

    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.use_va_fade = False
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.use_ib_reject = True
    cfg.rej_wide_only = True
    cfg.rej_target = "ib_low"
    cfg.rej_stop_buffer = 8.0
    cfg.max_rej_trades = 8

    # NO regime filter, NO blocked regimes
    trades = run_backtest(df, cfg, regime_filter=None, regime_blocked=None)
    return trades


def run_crossfire_unfiltered():
    """Run Crossfire (EMA Cross Credit Spreads Both Sides) with NO regime filter."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.ema_cross import EMACross
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = EMACross()
    trades = strategy.run_backtest(df, vix_data=vix)
    return trades


# ═══════════════════════════════════════════════════════════════
#  TAG TRADES WITH REGIME
# ═══════════════════════════════════════════════════════════════

def tag_trades_with_regime(trades, rf, pnl_attr="pnl_dollar"):
    """Tag each trade with its entry-time regime label.

    Returns list of dicts: {entry_time, pnl, regime_label}
    """
    tagged = []
    for t in trades:
        entry_time = getattr(t, "entry_time", None)
        if entry_time is None:
            continue

        # Get P&L
        pnl = getattr(t, pnl_attr, None)
        if pnl is None or pnl == 0:
            pnl = getattr(t, "net_pnl", 0)
        if pnl is None or pnl == 0:
            pnl = getattr(t, "pnl_dollar", 0)

        # Get regime at entry time
        regime = rf.get_regime_at(entry_time)
        label = regime.get("label", "Unknown")

        tagged.append({
            "entry_time": entry_time,
            "pnl": pnl,
            "regime_label": label if label else "Unknown",
        })
    return tagged


def tag_crossfire_trades(trades, rf):
    """Tag Crossfire trades with regime AND split by side.

    direction="short" → call credit spread (bearish side)
    direction="long"  → put credit spread (bullish side)

    Returns (call_credit_tagged, put_credit_tagged)
    """
    call_credit = []
    put_credit = []

    for t in trades:
        entry_time = getattr(t, "entry_time", None)
        direction = getattr(t, "direction", None)
        pnl = getattr(t, "net_pnl", 0)

        if entry_time is None or direction is None:
            continue

        regime = rf.get_regime_at(entry_time)
        label = regime.get("label", "Unknown")
        label = label if label else "Unknown"

        record = {
            "entry_time": entry_time,
            "pnl": pnl,
            "regime_label": label,
        }

        if direction == "short":
            call_credit.append(record)
        elif direction == "long":
            put_credit.append(record)

    return call_credit, put_credit


# ═══════════════════════════════════════════════════════════════
#  PER-REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════

def compute_regime_breakdown(tagged_trades):
    """Compute per-regime P&L breakdown with statistical test.

    Returns dict: regime_label -> {trades, pnl, avg, win_rate, pf, p_value, verdict}
    """
    by_regime = defaultdict(list)
    for t in tagged_trades:
        by_regime[t["regime_label"]].append(t["pnl"])

    results = {}
    for regime in REGIME_NAMES:
        pnls = by_regime.get(regime, [])
        n = len(pnls)
        if n == 0:
            results[regime] = {
                "trades": 0, "pnl": 0.0, "avg": 0.0,
                "win_rate": 0.0, "pf": 0.0, "p_value": 1.0,
                "verdict": "ALLOW (no trades)",
            }
            continue

        total = sum(pnls)
        avg = total / n
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        # One-sample t-test: H0: mean P&L = 0
        if n >= 2:
            t_stat, p_val = scipy_stats.ttest_1samp(pnls, 0)
            # We care about NEGATIVE mean specifically, so one-sided
            # If mean < 0, p_one_sided = p_val / 2; if mean >= 0, not a loser
            if avg < 0:
                p_one_sided = p_val / 2
            else:
                p_one_sided = 1.0  # Not a loser
        else:
            p_one_sided = 1.0

        # Verdict: BLOCK if negative P&L AND p < 0.05
        if total < 0 and p_one_sided < 0.05:
            verdict = "BLOCK"
        elif total < 0:
            verdict = f"ALLOW (negative but p={p_one_sided:.3f})"
        else:
            verdict = "ALLOW"

        results[regime] = {
            "trades": n,
            "pnl": total,
            "avg": avg,
            "win_rate": wr,
            "pf": pf,
            "p_value": p_one_sided,
            "verdict": verdict,
        }

    # Also handle unknown
    unknown_pnls = by_regime.get("Unknown", [])
    if unknown_pnls:
        n = len(unknown_pnls)
        total = sum(unknown_pnls)
        results["(No regime data)"] = {
            "trades": n, "pnl": total, "avg": total / n,
            "win_rate": sum(1 for p in unknown_pnls if p > 0) / n * 100,
            "pf": 0.0, "p_value": 1.0, "verdict": "N/A",
        }

    return results


# ═══════════════════════════════════════════════════════════════
#  PRINT TABLE
# ═══════════════════════════════════════════════════════════════

def print_regime_table(title, breakdown, tagged_trades):
    """Print a formatted regime P&L table."""
    total_pnl = sum(t["pnl"] for t in tagged_trades)
    total_trades = len(tagged_trades)

    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"  Total: {total_trades} trades, ${total_pnl:+,.0f} P&L")
    print(f"{'=' * 100}")

    header = (f"  {'Regime':<25s} {'Trades':>7s} {'P&L':>12s} {'Avg':>10s} "
              f"{'WR':>7s} {'PF':>7s} {'p-val':>8s}  {'Verdict':<30s}")
    print(header)
    print(f"  {'-' * 96}")

    for regime in REGIME_NAMES:
        r = breakdown.get(regime, {})
        n = r.get("trades", 0)
        if n == 0:
            print(f"  {regime:<25s} {'0':>7s} {'$0':>12s} {'$0':>10s} "
                  f"{'---':>7s} {'---':>7s} {'---':>8s}  {'ALLOW (no trades)':<30s}")
            continue

        pf_str = f"{r['pf']:.2f}" if r['pf'] < 100 else "inf"
        p_str = f"{r['p_value']:.4f}" if r['p_value'] < 1.0 else "---"

        # Color indicator
        if "BLOCK" in r["verdict"] and not r["verdict"].startswith("ALLOW"):
            indicator = "*** BLOCK ***"
        elif r["pnl"] < 0:
            indicator = r["verdict"]
        else:
            indicator = "ALLOW"

        print(f"  {regime:<25s} {n:>7d} ${r['pnl']:>10,.0f} ${r['avg']:>8,.0f} "
              f"{r['win_rate']:>6.1f}% {pf_str:>7s} {p_str:>8s}  {indicator:<30s}")

    # Unknown/no-data
    if "(No regime data)" in breakdown:
        r = breakdown["(No regime data)"]
        print(f"  {'(No regime data)':<25s} {r['trades']:>7d} ${r['pnl']:>10,.0f} "
              f"${r['avg']:>8,.0f} {r['win_rate']:>6.1f}% {'---':>7s} {'---':>8s}  {'N/A':<30s}")

    print(f"  {'-' * 96}")


def print_summary_verdicts(undertow_bk, call_bk, put_bk):
    """Print final summary of all blocking decisions."""
    print(f"\n{'=' * 100}")
    print(f"  FINAL VERDICTS")
    print(f"{'=' * 100}")

    sections = [
        ("Undertow (ES Short)", undertow_bk),
        ("Crossfire Call Credit (Bearish)", call_bk),
        ("Crossfire Put Credit (Bullish)", put_bk),
    ]

    for label, bk in sections:
        blocked = [r for r in REGIME_NAMES if bk.get(r, {}).get("verdict", "").startswith("BLOCK")]
        allowed = [r for r in REGIME_NAMES if not bk.get(r, {}).get("verdict", "").startswith("BLOCK")]

        print(f"\n  {label}:")
        if blocked:
            print(f"    BLOCK:  {', '.join(blocked)}")
        else:
            print(f"    BLOCK:  (none)")
        print(f"    ALLOW:  {', '.join(allowed)}")

    print(f"\n{'=' * 100}\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print("  REGIME PROFILE: Undertow & Crossfire (5-state walk-forward)")
    print("  Both strategies run UNFILTERED, then tagged by entry-time regime")
    print("  Verdict: BLOCK if P&L < 0 AND p < 0.05 (one-sided t-test)")
    print("=" * 100)

    # ── 1. Load walk-forward regime labels ──
    print("\n  Loading walk-forward regime labels...")
    rf = load_wf_regime_filter()

    # ── 2. Run Undertow UNFILTERED ──
    print("\n  Running Undertow (ES short, unfiltered)...")
    undertow_trades = run_undertow_unfiltered()
    print(f"  Got {len(undertow_trades)} trades")

    undertow_tagged = tag_trades_with_regime(undertow_trades, rf, pnl_attr="pnl_dollar")
    undertow_breakdown = compute_regime_breakdown(undertow_tagged)

    print_regime_table("UNDERTOW (AMT-TEMA / ES IB Breakout Short) — UNFILTERED",
                       undertow_breakdown, undertow_tagged)

    # ── 3. Run Crossfire UNFILTERED ──
    print("\n  Running Crossfire (EMA Cross Credit Spreads, unfiltered)...")
    crossfire_trades = run_crossfire_unfiltered()
    print(f"  Got {len(crossfire_trades)} trades total")

    call_tagged, put_tagged = tag_crossfire_trades(crossfire_trades, rf)
    print(f"  Call credit (bearish): {len(call_tagged)} trades")
    print(f"  Put credit (bullish):  {len(put_tagged)} trades")

    call_breakdown = compute_regime_breakdown(call_tagged)
    put_breakdown = compute_regime_breakdown(put_tagged)

    print_regime_table("CROSSFIRE — CALL CREDIT SIDE (bearish, sell call spreads) — UNFILTERED",
                       call_breakdown, call_tagged)

    print_regime_table("CROSSFIRE — PUT CREDIT SIDE (bullish, sell put spreads) — UNFILTERED",
                       put_breakdown, put_tagged)

    # ── 4. Final verdicts ──
    print_summary_verdicts(undertow_breakdown, call_breakdown, put_breakdown)


if __name__ == "__main__":
    main()
