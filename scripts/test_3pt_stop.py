#!/usr/bin/env python3
"""Test 3pt stop on 3x C3 — with and without cooldown.

Compares 3pt stop (user's friend's style) against 7pt and 8pt baselines.
Also tests 3pt with no cooldown (live re-entry style).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.stagger_engine import run_backtest_stagger


def make_cfg(stop_buffer, cooldown_bars=2, pct_stop_bps=30.0):
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = pct_stop_bps
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.cooldown_bars = cooldown_bars
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = True
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = stop_buffer
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = True
    cfg.lvl_ma_filter = "tema"
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = stop_buffer
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

    targets = [t for t in lvl if t.exit_reason == "target"]
    stops = [t for t in lvl if t.exit_reason == "stop"]
    flattens = [t for t in lvl if t.exit_reason == "flatten"]

    avg_stop_loss = np.mean([t.pnl_dollar for t in stops]) if stops else 0
    base_pnl = sum(t.pnl_dollar for t in base)

    return {
        "label": label,
        "lvl_fills": len(lvl),
        "pf": pf,
        "wr": wr,
        "lvl_pnl": total,
        "target_gain": sum(t.pnl_dollar for t in targets),
        "stop_loss": sum(t.pnl_dollar for t in stops),
        "flatten_pnl": sum(t.pnl_dollar for t in flattens),
        "tgt_count": len(targets),
        "stp_count": len(stops),
        "flt_count": len(flattens),
        "avg_stop_loss": avg_stop_loss,
        "base_pnl": base_pnl,
        "combined": total + base_pnl,
    }


def print_row(r):
    print(f"  {r['label']:>16}  {r['lvl_fills']:>5}  {r['pf']:>5.2f}  {r['wr']:>5.1f}%  "
          f"${r['lvl_pnl']:>+9,.0f}  ${r['target_gain']:>+9,.0f}  "
          f"${r['stop_loss']:>+9,.0f}  ${r['flatten_pnl']:>+9,.0f}  "
          f"{r['tgt_count']:>4}  {r['stp_count']:>4}  {r['flt_count']:>4}  "
          f"${r['avg_stop_loss']:>+8,.0f}  ${r['combined']:>+9,.0f}")


def print_header():
    print(f"  {'Config':>16}  {'Fills':>5}  {'PF':>5}  {'WR':>6}  "
          f"{'LVL P&L':>10}  {'Target$':>10}  "
          f"{'Stop$':>10}  {'Flat$':>10}  "
          f"{'Tgt#':>4}  {'Stp#':>4}  {'Flt#':>4}  "
          f"{'AvgStp$':>9}  {'Combined':>10}")
    print(f"  {'-'*16}  {'-'*5}  {'-'*5}  {'-'*6}  "
          f"{'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  "
          f"{'-'*4}  {'-'*4}  {'-'*4}  "
          f"{'-'*9}  {'-'*10}")


def main():
    df = load_tos_csv("data/es_5m_databento_2yr.csv")
    split = "2025-02-14"
    df_is = df[df.index < split]
    df_oos = df[df.index >= split]

    configs = [
        ("3pt / cd=2", 3, 2),
        ("3pt / cd=0", 3, 0),
        ("5pt / cd=2", 5, 2),
        ("7pt / cd=2", 7, 2),
        ("8pt / cd=2", 8, 2),
    ]

    print("=" * 130)
    print("  3-POINT STOP TEST — 3x C3 (all targeting 3rd support)")
    print("  cd=2 = standard 2-bar cooldown | cd=0 = no cooldown (live re-entry)")
    print("=" * 130)

    # --- OOS ---
    print("\n  OUT-OF-SAMPLE (2025-02-14 onward):")
    print_header()
    oos_results = {}
    for label, buf, cd in configs:
        cfg = make_cfg(buf, cooldown_bars=cd)
        trades = run_backtest_stagger(df_oos.copy(), cfg, n_contracts=3, uniform_skip=2)
        r = analyze(trades, label)
        if r:
            oos_results[label] = r
            print_row(r)

    # --- IS ---
    print("\n  IN-SAMPLE (before 2025-02-14):")
    print_header()
    is_results = {}
    for label, buf, cd in configs:
        cfg = make_cfg(buf, cooldown_bars=cd)
        trades = run_backtest_stagger(df_is.copy(), cfg, n_contracts=3, uniform_skip=2)
        r = analyze(trades, label)
        if r:
            is_results[label] = r
            print_row(r)

    # --- Full 2yr ---
    print("\n  FULL 2-YEAR:")
    print_header()
    full_results = {}
    for label, buf, cd in configs:
        cfg = make_cfg(buf, cooldown_bars=cd)
        trades = run_backtest_stagger(df.copy(), cfg, n_contracts=3, uniform_skip=2)
        r = analyze(trades, label)
        if r:
            full_results[label] = r
            print_row(r)

    # --- Walk-Forward PF Ratios ---
    print("\n" + "=" * 130)
    print("  WALK-FORWARD PF RATIOS (OOS/IS) — threshold: >= 0.7 ROBUST, >= 0.5 ACCEPTABLE")
    print("=" * 130)
    for label, _, _ in configs:
        oos_r = oos_results.get(label)
        is_r = is_results.get(label)
        if oos_r and is_r and is_r["pf"] > 0:
            ratio = oos_r["pf"] / is_r["pf"]
            grade = "ROBUST" if ratio >= 0.7 else "ACCEPTABLE" if ratio >= 0.5 else "WEAK"
            print(f"  {label:>16}:  IS PF {is_r['pf']:.2f}  ->  OOS PF {oos_r['pf']:.2f}  "
                  f"->  Ratio {ratio:.2f}  {grade}")

    # --- Key comparison ---
    print("\n" + "=" * 130)
    print("  HEAD-TO-HEAD: 3pt vs 8pt (standard cooldown)")
    print("=" * 130)

    three = oos_results.get("3pt / cd=2")
    eight = oos_results.get("8pt / cd=2")
    if three and eight:
        print(f"\n  3pt stop (OOS):")
        print(f"    Fills: {three['lvl_fills']}  |  Targets: {three['tgt_count']}  |  Stops: {three['stp_count']}  |  Flattens: {three['flt_count']}")
        print(f"    LVL P&L: ${three['lvl_pnl']:+,.0f}  |  Avg stop loss: ${three['avg_stop_loss']:+,.0f}/fill")
        print(f"\n  8pt stop (OOS):")
        print(f"    Fills: {eight['lvl_fills']}  |  Targets: {eight['tgt_count']}  |  Stops: {eight['stp_count']}  |  Flattens: {eight['flt_count']}")
        print(f"    LVL P&L: ${eight['lvl_pnl']:+,.0f}  |  Avg stop loss: ${eight['avg_stop_loss']:+,.0f}/fill")
        diff = three['lvl_pnl'] - eight['lvl_pnl']
        print(f"\n  Difference: ${diff:+,.0f} ({'3pt wins' if diff > 0 else '8pt wins'})")
        print(f"  3pt saves on stops: ${three['stop_loss'] - eight['stop_loss']:+,.0f}")
        print(f"  3pt loses on targets: ${three['target_gain'] - eight['target_gain']:+,.0f}")

    # --- No cooldown comparison ---
    three_nc = oos_results.get("3pt / cd=0")
    if three and three_nc:
        print(f"\n  NO COOLDOWN EFFECT (3pt stop, OOS):")
        print(f"    With cooldown:    {three['lvl_fills']} fills, ${three['lvl_pnl']:+,.0f}")
        print(f"    Without cooldown: {three_nc['lvl_fills']} fills, ${three_nc['lvl_pnl']:+,.0f}")
        extra_fills = three_nc['lvl_fills'] - three['lvl_fills']
        extra_pnl = three_nc['lvl_pnl'] - three['lvl_pnl']
        print(f"    Extra fills: {extra_fills}  |  Extra P&L: ${extra_pnl:+,.0f}")
        if extra_fills > 0:
            print(f"    Per extra fill: ${extra_pnl / extra_fills:+,.0f}")

    print()


if __name__ == "__main__":
    main()
