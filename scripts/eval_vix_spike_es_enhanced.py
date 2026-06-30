#!/usr/bin/env python3
"""VIX Spike ES Enhanced — Validation script for optimized config.

Runs the exact winning config from the Apr 8 threshold sweep:
  - 5% threshold (was 7%)
  - 50bps stop (was 30bps)
  - Fixed target 20pt (was hold-all-day)
  - ES move filter -0.1% (new)
  - Max hold 18 bars / 90min

Validates: full period metrics, walk-forward, significance, exit breakdown.
"""

import os
import sys

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.test_vix_spike_es import (
    VixSpikeConfig, load_data, run_vix_spike, compute_simple_metrics,
    run_significance, walk_forward, print_metrics, INITIAL_CAPITAL,
)

WF_SPLIT = "2025-02-16"


def main():
    df, vix_lookup = load_data()

    # ── OLD CONFIG (production baseline) ──
    old_cfg = VixSpikeConfig(
        spike_threshold=0.07,
        max_hold_bars=200,      # hold all day
        stop_bps=30.0,
        exit_mode="green_bar",  # original exit
        es_move_filter=0.0,     # no filter
        entry_start=935,
        entry_end=1500,
    )

    # ── NEW CONFIG (enhanced — sweep winner) ──
    new_cfg = VixSpikeConfig(
        spike_threshold=0.05,
        max_hold_bars=18,       # 90 min
        stop_bps=50.0,
        exit_mode="fixed_target",
        target_pts=20.0,
        es_move_filter=-0.001,  # ES must be down 0.1% from open
        entry_start=935,
        entry_end=1500,
    )

    print("=" * 70)
    print("  VIX SPIKE ES — ENHANCED CONFIG VALIDATION")
    print("  Apr 8, 2026")
    print("=" * 70)

    # ── BASELINE ──
    print(f"\n{'─'*70}")
    print("  BASELINE (production: 7%, green_bar, 30bps, no filter)")
    print(f"{'─'*70}")

    old_trades = run_vix_spike(df, vix_lookup, old_cfg)
    old_m = compute_simple_metrics(old_trades)
    print_metrics(old_m, "  Full")

    old_is, old_oos, old_m_is, old_m_oos = walk_forward(df, vix_lookup, old_cfg)
    print_metrics(old_m_is, "  IS ")
    print_metrics(old_m_oos, "  OOS")

    old_t, old_perm, old_boot = run_significance(old_trades)
    print(f"    p-value: {old_t:.6f} | bootstrap: {old_boot:.2%}")

    # ── ENHANCED ──
    print(f"\n{'─'*70}")
    print("  ENHANCED (5%, fixed_target 20pt, 50bps, ES -0.1% filter)")
    print(f"{'─'*70}")

    new_trades = run_vix_spike(df, vix_lookup, new_cfg)
    new_m = compute_simple_metrics(new_trades)
    print_metrics(new_m, "  Full")

    new_is, new_oos, new_m_is, new_m_oos = walk_forward(df, vix_lookup, new_cfg)
    print_metrics(new_m_is, "  IS ")
    print_metrics(new_m_oos, "  OOS")

    new_t, new_perm, new_boot = run_significance(new_trades)
    oos_t, _, oos_boot = run_significance(new_oos, seed=123)
    print(f"    Full p-value: {new_t:.6f} | bootstrap: {new_boot:.2%}")
    print(f"    OOS  p-value: {oos_t:.6f} | bootstrap: {oos_boot:.2%}")

    if new_m_is["pf"] > 0 and new_m_is["pf"] != float("inf"):
        wf_ratio = new_m_oos["pf"] / new_m_is["pf"]
    else:
        wf_ratio = 0
    print(f"    WF PF ratio: {wf_ratio:.3f}")

    # ── EXIT BREAKDOWN ──
    print(f"\n{'─'*70}")
    print("  EXIT REASON BREAKDOWN (enhanced)")
    print(f"{'─'*70}")

    reasons = {}
    for t in new_trades:
        r = t.exit_reason
        if r not in reasons:
            reasons[r] = {"count": 0, "pnl": 0, "wins": 0}
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t.pnl_dollar
        if t.pnl_dollar > 0:
            reasons[r]["wins"] += 1

    for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        wr = data["wins"] / data["count"] * 100 if data["count"] > 0 else 0
        print(f"    {reason:<16} {data['count']:>4} trades  "
              f"WR {wr:>5.1f}%  ${data['pnl']:>+10,.0f}")

    # ── COMPARISON ──
    print(f"\n{'='*70}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'OLD (7%/green)':>18} {'NEW (5%/target)':>18} {'Delta':>12}")
    print(f"  {'─'*68}")

    def fmt_delta(old_v, new_v, is_dollar=False):
        d = new_v - old_v
        if is_dollar:
            return f"${d:+,.0f}"
        return f"{d:+.3f}" if abs(d) < 100 else f"{d:+,.0f}"

    rows = [
        ("Trades", old_m["total"], new_m["total"], False),
        ("Win Rate %", old_m["win_rate"], new_m["win_rate"], False),
        ("Profit Factor", old_m["pf"], new_m["pf"], False),
        ("Net P&L", old_m["net_pnl"], new_m["net_pnl"], True),
        ("Max DD", old_m["max_dd"], new_m["max_dd"], True),
        ("Sharpe", old_m["sharpe"], new_m["sharpe"], False),
        ("OOS PF", old_m_oos.get("pf", 0), new_m_oos.get("pf", 0), False),
        ("OOS Trades", old_m_oos.get("total", 0), new_m_oos.get("total", 0), False),
    ]

    for label, old_v, new_v, is_dollar in rows:
        if is_dollar:
            old_s = f"${old_v:>+,.0f}"
            new_s = f"${new_v:>+,.0f}"
        elif isinstance(old_v, int):
            old_s = f"{old_v:>d}"
            new_s = f"{new_v:>d}"
        else:
            old_s = f"{old_v:>.3f}" if old_v < 100 else f"{old_v:>,.0f}"
            new_s = f"{new_v:>.3f}" if new_v < 100 else f"{new_v:>,.0f}"
        print(f"  {label:<20} {old_s:>18} {new_s:>18} {fmt_delta(old_v, new_v, is_dollar):>12}")

    # ── VERDICT ──
    print(f"\n{'='*70}")
    passed = (
        new_t < 0.05
        and new_m["pf"] > 1.0
        and new_m_oos["pf"] > 1.0
        and oos_t < 0.05
        and new_m_oos["total"] >= 10
    )

    if passed:
        print("  VERDICT: **DEPLOY** — all criteria met")
        print(f"    Full: {new_m['total']}t, PF {new_m['pf']:.3f}, "
              f"${new_m['net_pnl']:+,.0f}, p={new_t:.6f}")
        print(f"    OOS:  {new_m_oos['total']}t, PF {new_m_oos['pf']:.3f}, "
              f"p={oos_t:.6f}, WF ratio={wf_ratio:.3f}")
    else:
        print("  VERDICT: **FAIL**")
        if new_t >= 0.05:
            print(f"    Full p-value {new_t:.4f} >= 0.05")
        if new_m["pf"] <= 1.0:
            print(f"    Full PF {new_m['pf']:.3f} <= 1.0")
        if new_m_oos["pf"] <= 1.0:
            print(f"    OOS PF {new_m_oos['pf']:.3f} <= 1.0")
        if oos_t >= 0.05:
            print(f"    OOS p-value {oos_t:.4f} >= 0.05")

    print(f"{'='*70}")

    # ── TRADE LOG (last 15) ──
    print(f"\n  TRADE LOG (last 15 — enhanced config):")
    for t in new_trades[-15:]:
        pnl_sign = "+" if t.pnl_dollar > 0 else ""
        print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} → "
              f"{t.exit_time.strftime('%H:%M')} "
              f"| {t.exit_reason:<10} | {pnl_sign}${t.pnl_dollar:,.0f} "
              f"| VIX {t.vix_open:.1f}→{t.vix_high:.1f} ({t.vix_spike_pct:.1%})")


if __name__ == "__main__":
    main()
