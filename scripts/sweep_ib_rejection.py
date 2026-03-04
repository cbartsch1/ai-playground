#!/usr/bin/env python3
"""Sweep IB Rejection trigger × target × TEMA filter combos on 2yr ES data.

Runs on top of v8 baseline (short-only, 30bps pct stop, skip Friday, noon blackout).
Tests IB Rejection as an ADDITIONAL setup alongside IB Breakout.

Usage:
    python scripts/sweep_ib_rejection.py data/es_5m_databento_2yr.csv
"""

import argparse
import copy
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


def make_v8_base():
    """v8 baseline config."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    return cfg


TRIGGERS = ["any", "bearish_close", "wick", "failed_break"]
TARGETS = ["vwap", "ib_mid", "ib_low", "prev_poc", "fixed"]
TEMA_OPTS = [False, True]  # False first (aggressive, user preference)


def run_sweep(df):
    """Run Stage 1 sweep: trigger × target × TEMA."""
    results = []

    # v8 baseline first
    cfg_base = make_v8_base()
    t0 = time.time()
    trades_base = run_backtest(df.copy(), cfg_base)
    base_time = time.time() - t0
    m_base = compute_metrics(trades_base)
    print(f"  v8 baseline: {m_base.total_trades} trades, PF {m_base.profit_factor:.3f}, "
          f"P&L ${m_base.net_pnl:+,.0f} ({base_time:.1f}s)")

    total = len(TRIGGERS) * len(TARGETS) * len(TEMA_OPTS)
    done = 0

    for trigger in TRIGGERS:
        for target in TARGETS:
            for tema_req in TEMA_OPTS:
                done += 1
                cfg = make_v8_base()
                cfg.use_ib_reject = True
                cfg.rej_trigger = trigger
                cfg.rej_target = target
                cfg.rej_target_pts = 25.0
                cfg.rej_zone_pts = 5.0
                cfg.rej_stop_buffer = 3.0
                cfg.rej_require_tema = tema_req
                cfg.max_rej_trades = 5

                t0 = time.time()
                trades = run_backtest(df.copy(), cfg)
                elapsed = time.time() - t0
                m = compute_metrics(trades)

                # Separate REJ trades from IB trades
                rej_trades = [t for t in trades if t.setup == "REJ"]
                ib_trades = [t for t in trades if t.setup == "IB"]
                m_rej = compute_metrics(rej_trades) if rej_trades else compute_metrics([])

                results.append({
                    "trigger": trigger,
                    "target": target,
                    "tema": "ON" if tema_req else "OFF",
                    "trades": m.total_trades,
                    "rej_trades": m_rej.total_trades,
                    "ib_trades": len(ib_trades),
                    "wr": m.win_rate,
                    "pf": m.profit_factor,
                    "pnl": m.net_pnl,
                    "rej_pnl": m_rej.net_pnl,
                    "rej_wr": m_rej.win_rate,
                    "rej_pf": m_rej.profit_factor,
                    "avg_trade": m.avg_trade,
                    "max_dd": m.max_drawdown,
                    "sharpe": m.sharpe,
                })

                sys.stdout.write(f"\r  [{done}/{total}] {trigger:14s} {target:8s} TEMA={tema_req!s:5s} "
                                 f"→ {m.total_trades:>4d} trades  PF {m.profit_factor:>6.3f}  "
                                 f"P&L ${m.net_pnl:>+10,.0f}  ({elapsed:.1f}s)")
                sys.stdout.flush()

    print()
    return results, m_base


def print_results(results, m_base):
    """Print results sorted by combined P&L."""
    print(f"\n{'='*120}")
    print(f"  IB REJECTION SWEEP — Stage 1 (v8 baseline + IB Rejection)")
    print(f"  Fixed: zone=5pt, stop_buffer=3pt, max_trades=5/day, target_pts=25 (fixed only)")
    print(f"  v8 baseline: {m_base.total_trades} trades, PF {m_base.profit_factor:.3f}, P&L ${m_base.net_pnl:+,.0f}")
    print(f"{'='*120}")

    # Sort by combined P&L descending
    results.sort(key=lambda r: r["pnl"], reverse=True)

    header = (f"{'Trigger':<14} {'Target':<8} {'TEMA':<4} "
              f"{'Total':>5} {'REJ':>4} {'IB':>3} "
              f"{'WR%':>6} {'PF':>6} {'P&L':>11} "
              f"{'REJ P&L':>10} {'REJ WR':>6} {'REJ PF':>6} "
              f"{'Avg$':>7} {'MaxDD':>9} {'Sharpe':>6}")
    print(header)
    print("-" * 120)

    for r in results:
        delta = r["pnl"] - m_base.net_pnl
        marker = " **" if r["pnl"] > m_base.net_pnl and r["rej_pf"] > 1.0 else ""
        print(f"{r['trigger']:<14} {r['target']:<8} {r['tema']:<4} "
              f"{r['trades']:>5} {r['rej_trades']:>4} {r['ib_trades']:>3} "
              f"{r['wr']:>5.1f}% {r['pf']:>6.3f} ${r['pnl']:>+10,.0f} "
              f"${r['rej_pnl']:>+9,.0f} {r['rej_wr']:>5.1f}% {r['rej_pf']:>6.3f} "
              f"${r['avg_trade']:>6.0f} ${r['max_dd']:>8,.0f} {r['sharpe']:>6.2f}{marker}")

    # Summary
    profitable = [r for r in results if r["rej_pf"] > 1.0]
    print(f"\n  REJ setups with PF > 1.0: {len(profitable)}/{len(results)}")

    if profitable:
        best = max(profitable, key=lambda r: r["rej_pnl"])
        print(f"  Best REJ standalone: {best['trigger']}/{best['target']}/TEMA={best['tema']} "
              f"→ {best['rej_trades']} trades, PF {best['rej_pf']:.3f}, P&L ${best['rej_pnl']:+,.0f}")

    # Top 5 by REJ P&L
    print(f"\n  Top 5 by REJ P&L:")
    by_rej = sorted(results, key=lambda r: r["rej_pnl"], reverse=True)[:5]
    for r in by_rej:
        print(f"    {r['trigger']:<14} {r['target']:<8} TEMA={r['tema']:<3} "
              f"→ {r['rej_trades']:>4} trades, {r['rej_wr']:>5.1f}% WR, PF {r['rej_pf']:>6.3f}, "
              f"P&L ${r['rej_pnl']:>+9,.0f}")


def run_stage2(df, top_combos):
    """Stage 2: Fine-tune top combos with zone, stop_buffer, max_trades variations."""
    print(f"\n{'='*120}")
    print(f"  STAGE 2 — Fine-tuning top {len(top_combos)} combos")
    print(f"{'='*120}")

    zones = [3, 5, 8, 12]
    stop_buffers = [2, 3, 5, 8]
    max_trades_opts = [3, 5, 8]

    results = []
    total = len(top_combos) * len(zones) * len(stop_buffers) * len(max_trades_opts)
    done = 0

    for combo in top_combos:
        for zone in zones:
            for stop_buf in stop_buffers:
                for max_t in max_trades_opts:
                    done += 1
                    cfg = make_v8_base()
                    cfg.use_ib_reject = True
                    cfg.rej_trigger = combo["trigger"]
                    cfg.rej_target = combo["target"]
                    cfg.rej_target_pts = 25.0
                    cfg.rej_zone_pts = zone
                    cfg.rej_stop_buffer = stop_buf
                    cfg.rej_require_tema = combo["tema"] == "ON"
                    cfg.max_rej_trades = max_t

                    trades = run_backtest(df.copy(), cfg)
                    m = compute_metrics(trades)
                    rej_trades = [t for t in trades if t.setup == "REJ"]
                    m_rej = compute_metrics(rej_trades) if rej_trades else compute_metrics([])

                    results.append({
                        "trigger": combo["trigger"],
                        "target": combo["target"],
                        "tema": combo["tema"],
                        "zone": zone,
                        "stop_buf": stop_buf,
                        "max_t": max_t,
                        "trades": m.total_trades,
                        "rej_trades": m_rej.total_trades,
                        "wr": m.win_rate,
                        "pf": m.profit_factor,
                        "pnl": m.net_pnl,
                        "rej_pnl": m_rej.net_pnl,
                        "rej_wr": m_rej.win_rate,
                        "rej_pf": m_rej.profit_factor,
                        "max_dd": m.max_drawdown,
                        "sharpe": m.sharpe,
                    })

                    sys.stdout.write(f"\r  [{done}/{total}]")
                    sys.stdout.flush()

    print()

    # Sort by combined P&L
    results.sort(key=lambda r: r["pnl"], reverse=True)

    header = (f"{'Trigger':<14} {'Target':<8} {'TEMA':<4} {'Zone':>4} {'SBuf':>4} {'MaxT':>4} "
              f"{'Total':>5} {'REJ':>4} "
              f"{'PF':>6} {'P&L':>11} {'REJ PF':>6} {'REJ P&L':>10} "
              f"{'MaxDD':>9} {'Sharpe':>6}")
    print(header)
    print("-" * 120)

    for r in results[:30]:  # Top 30
        print(f"{r['trigger']:<14} {r['target']:<8} {r['tema']:<4} {r['zone']:>4} {r['stop_buf']:>4} {r['max_t']:>4} "
              f"{r['trades']:>5} {r['rej_trades']:>4} "
              f"{r['pf']:>6.3f} ${r['pnl']:>+10,.0f} {r['rej_pf']:>6.3f} ${r['rej_pnl']:>+9,.0f} "
              f"${r['max_dd']:>8,.0f} {r['sharpe']:>6.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="IB Rejection Sweep")
    parser.add_argument("csv_file", help="Path to CSV data file")
    parser.add_argument("--stage2", action="store_true", help="Also run Stage 2 fine-tuning")
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"RTH bars: {df['is_rth'].sum()}, Sessions: {df['new_rth'].sum()}")

    print(f"\n--- Stage 1: Trigger × Target × TEMA ({len(TRIGGERS)}×{len(TARGETS)}×{len(TEMA_OPTS)} = "
          f"{len(TRIGGERS)*len(TARGETS)*len(TEMA_OPTS)} combos) ---")
    results, m_base = run_sweep(df)
    print_results(results, m_base)

    if args.stage2:
        # Pick top 3 by REJ P&L (must be profitable)
        profitable = [r for r in results if r["rej_pf"] > 1.0]
        if profitable:
            profitable.sort(key=lambda r: r["rej_pnl"], reverse=True)
            top3 = profitable[:3]
            labels = [r["trigger"] + "/" + r["target"] + "/TEMA=" + r["tema"] for r in top3]
            print(f"\n  Stage 2 candidates: {labels}")
            stage2_results = run_stage2(df, top3)
        else:
            print("\n  No profitable REJ combos found — skipping Stage 2")


if __name__ == "__main__":
    main()
