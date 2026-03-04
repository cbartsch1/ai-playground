#!/usr/bin/env python3
"""Sweep stop buffer sizes on 3x C3 to find optimal tightness."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.stagger_engine import run_backtest_stagger


def make_lvl(stop_buffer, pct_stop_bps=30.0):
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = pct_stop_bps
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
    cfg.rej_stop_buffer = stop_buffer  # <-- THIS IS WHAT WE'RE SWEEPING
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = True
    cfg.lvl_ma_filter = "tema"
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = stop_buffer  # <-- AND THIS
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_min_rr = 0.5
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3
    return cfg


def analyze(trades, label):
    lvl = [t for t in trades if t.setup.startswith("LVL")]
    base = [t for t in trades if not t.setup.startswith("LVL")]

    if not lvl:
        return None

    pnls = [t.pnl_dollar for t in lvl]
    total = sum(pnls)
    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gw / gl if gl > 0 else float("inf")
    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100

    # Exit breakdown
    targets = [t for t in lvl if t.exit_reason == "target"]
    stops = [t for t in lvl if t.exit_reason == "stop"]
    flattens = [t for t in lvl if t.exit_reason == "flatten"]

    stop_loss = sum(t.pnl_dollar for t in stops)
    target_gain = sum(t.pnl_dollar for t in targets)
    flatten_pnl = sum(t.pnl_dollar for t in flattens)

    avg_stop_dist = np.mean([t.stop - t.entry_price for t in lvl]) if lvl else 0
    avg_stop_loss = np.mean([t.pnl_dollar for t in stops]) if stops else 0

    base_pnl = sum(t.pnl_dollar for t in base)

    return {
        "label": label,
        "lvl_trades": len(lvl),
        "entries": len(lvl) // 3,
        "pf": pf,
        "wr": wr,
        "total_pnl": total,
        "target_gain": target_gain,
        "stop_loss": stop_loss,
        "flatten_pnl": flatten_pnl,
        "target_count": len(targets),
        "stop_count": len(stops),
        "flatten_count": len(flattens),
        "avg_stop_dist": avg_stop_dist,
        "avg_stop_loss": avg_stop_loss,
        "base_pnl": base_pnl,
        "combined_pnl": total + base_pnl,
    }


def main():
    df = load_tos_csv("data/es_5m_databento_2yr.csv")
    split = "2025-02-14"
    df_oos = df[df.index >= split]
    df_full = df

    # Sweep stop buffer from 2 to 12 points
    buffers = [2, 3, 4, 5, 6, 7, 8, 10, 12]

    print("=" * 90)
    print("  STOP BUFFER SWEEP — 3x C3 (3 contracts all targeting 3rd support)")
    print("=" * 90)

    # OOS results
    print("\n  OUT-OF-SAMPLE:")
    print(f"  {'Buffer':>6} {'Fills':>6} {'PF':>6} {'WR':>6} {'LVL P&L':>10} "
          f"{'Target$':>10} {'Stop$':>10} {'Flat$':>10} "
          f"{'Tgt#':>5} {'Stp#':>5} {'Flt#':>5} {'AvgStopDist':>11} {'AvgStopLoss':>12} {'Combined':>10}")
    print(f"  {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 10} "
          f"{'-' * 10} {'-' * 10} {'-' * 10} "
          f"{'-' * 5} {'-' * 5} {'-' * 5} {'-' * 11} {'-' * 12} {'-' * 10}")

    oos_results = []
    for buf in buffers:
        cfg = make_lvl(buf)
        trades = run_backtest_stagger(df_oos.copy(), cfg, n_contracts=3, uniform_skip=2)
        r = analyze(trades, f"{buf}pt")
        if r:
            oos_results.append(r)
            print(f"  {buf:>5}pt {r['lvl_trades']:>6} {r['pf']:>6.2f} {r['wr']:>5.1f}% "
                  f"${r['total_pnl']:>+9,.0f} ${r['target_gain']:>+9,.0f} "
                  f"${r['stop_loss']:>+9,.0f} ${r['flatten_pnl']:>+9,.0f} "
                  f"{r['target_count']:>5} {r['stop_count']:>5} {r['flatten_count']:>5} "
                  f"{r['avg_stop_dist']:>10.1f}pt ${r['avg_stop_loss']:>+10,.0f} "
                  f"${r['combined_pnl']:>+9,.0f}")

    # Full 2yr results
    print("\n  FULL 2-YEAR:")
    print(f"  {'Buffer':>6} {'Fills':>6} {'PF':>6} {'WR':>6} {'LVL P&L':>10} "
          f"{'Target$':>10} {'Stop$':>10} {'Flat$':>10} "
          f"{'Tgt#':>5} {'Stp#':>5} {'Flt#':>5} {'AvgStopDist':>11} {'AvgStopLoss':>12} {'Combined':>10}")
    print(f"  {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 10} "
          f"{'-' * 10} {'-' * 10} {'-' * 10} "
          f"{'-' * 5} {'-' * 5} {'-' * 5} {'-' * 11} {'-' * 12} {'-' * 10}")

    full_results = []
    for buf in buffers:
        cfg = make_lvl(buf)
        trades = run_backtest_stagger(df_full.copy(), cfg, n_contracts=3, uniform_skip=2)
        r = analyze(trades, f"{buf}pt")
        if r:
            full_results.append(r)
            print(f"  {buf:>5}pt {r['lvl_trades']:>6} {r['pf']:>6.2f} {r['wr']:>5.1f}% "
                  f"${r['total_pnl']:>+9,.0f} ${r['target_gain']:>+9,.0f} "
                  f"${r['stop_loss']:>+9,.0f} ${r['flatten_pnl']:>+9,.0f} "
                  f"{r['target_count']:>5} {r['stop_count']:>5} {r['flatten_count']:>5} "
                  f"{r['avg_stop_dist']:>10.1f}pt ${r['avg_stop_loss']:>+10,.0f} "
                  f"${r['combined_pnl']:>+9,.0f}")

    # Key insight: what happens to stop losses as buffer tightens?
    print("\n" + "=" * 90)
    print("  KEY INSIGHT: WHERE DOES THE MONEY GO?")
    print("=" * 90)

    if len(oos_results) >= 2:
        current = next((r for r in oos_results if "8pt" in r["label"]), None)
        best = max(oos_results, key=lambda r: r["total_pnl"])

        if current and best:
            print(f"\n  Current (8pt buffer):")
            print(f"    LVL P&L: ${current['total_pnl']:+,.0f}")
            print(f"    Target wins: ${current['target_gain']:+,.0f} ({current['target_count']} hits)")
            print(f"    Stop losses: ${current['stop_loss']:+,.0f} ({current['stop_count']} stops)")
            print(f"    Stop loss per fill: ${current['avg_stop_loss']:+,.0f}")

            print(f"\n  Best ({best['label']} buffer):")
            print(f"    LVL P&L: ${best['total_pnl']:+,.0f}")
            print(f"    Target wins: ${best['target_gain']:+,.0f} ({best['target_count']} hits)")
            print(f"    Stop losses: ${best['stop_loss']:+,.0f} ({best['stop_count']} stops)")
            print(f"    Stop loss per fill: ${best['avg_stop_loss']:+,.0f}")

            saved = best["total_pnl"] - current["total_pnl"]
            print(f"\n  Difference: ${saved:+,.0f} "
                  f"({'tighter is better' if saved > 0 else 'current stop is better'})")

    print()


if __name__ == "__main__":
    main()
