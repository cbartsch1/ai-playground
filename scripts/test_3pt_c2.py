#!/usr/bin/env python3
"""Test 3pt stop targeting 2nd support (C2) vs 3rd support (C3).

Compare: does a tighter stop pair better with a closer target?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.stagger_engine import run_backtest_stagger


def make_cfg(stop_buffer):
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
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

    avg_tgt_pts = np.mean([t.entry_price - t.target for t in lvl]) if lvl else 0
    avg_stp_pts = np.mean([t.stop - t.entry_price for t in lvl]) if lvl else 0
    avg_rr = avg_tgt_pts / avg_stp_pts if avg_stp_pts > 0 else 0
    avg_stop_loss = np.mean([t.pnl_dollar for t in stops]) if stops else 0
    avg_tgt_win = np.mean([t.pnl_dollar for t in targets]) if targets else 0
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
        "avg_tgt_pts": avg_tgt_pts,
        "avg_stp_pts": avg_stp_pts,
        "avg_rr": avg_rr,
        "avg_stop_loss": avg_stop_loss,
        "avg_tgt_win": avg_tgt_win,
        "base_pnl": base_pnl,
        "combined": total + base_pnl,
    }


def print_row(r):
    print(f"  {r['label']:>18}  {r['lvl_fills']:>5}  {r['pf']:>5.2f}  {r['wr']:>5.1f}%  "
          f"${r['lvl_pnl']:>+9,.0f}  "
          f"{r['tgt_count']:>4}  {r['stp_count']:>4}  {r['flt_count']:>4}  "
          f"{r['avg_tgt_pts']:>6.1f}  {r['avg_stp_pts']:>6.1f}  {r['avg_rr']:>5.2f}  "
          f"${r['avg_tgt_win']:>+7,.0f}  ${r['avg_stop_loss']:>+7,.0f}  "
          f"${r['combined']:>+9,.0f}")


def print_header():
    print(f"  {'Config':>18}  {'Fills':>5}  {'PF':>5}  {'WR':>6}  "
          f"{'LVL P&L':>10}  "
          f"{'Tgt#':>4}  {'Stp#':>4}  {'Flt#':>4}  "
          f"{'TgtPt':>6}  {'StpPt':>6}  {'R:R':>5}  "
          f"{'AvgWin$':>8}  {'AvgStp$':>8}  "
          f"{'Combined':>10}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*5}  {'-'*6}  "
          f"{'-'*10}  "
          f"{'-'*4}  {'-'*4}  {'-'*4}  "
          f"{'-'*6}  {'-'*6}  {'-'*5}  "
          f"{'-'*8}  {'-'*8}  "
          f"{'-'*10}")


def main():
    df = load_tos_csv("data/es_5m_databento_2yr.csv")
    split = "2025-02-14"
    df_is = df[df.index < split]
    df_oos = df[df.index >= split]

    # (label, stop_buffer, uniform_skip)
    configs = [
        ("3pt / 2nd support", 3, 1),
        ("3pt / 3rd support", 3, 2),
        ("5pt / 2nd support", 5, 1),
        ("5pt / 3rd support", 5, 2),
        ("7pt / 2nd support", 7, 1),
        ("7pt / 3rd support", 7, 2),
    ]

    print("=" * 140)
    print("  STOP vs TARGET PAIRING — 3x contracts, all targeting same support level")
    print("  Does a tighter stop pair better with a closer target?")
    print("=" * 140)

    for period_label, df_slice in [("OUT-OF-SAMPLE", df_oos),
                                    ("IN-SAMPLE", df_is),
                                    ("FULL 2-YEAR", df)]:
        print(f"\n  {period_label}:")
        print_header()
        results = {}
        for label, buf, skip in configs:
            cfg = make_cfg(buf)
            trades = run_backtest_stagger(df_slice.copy(), cfg, n_contracts=3, uniform_skip=skip)
            r = analyze(trades, label)
            if r:
                results[label] = r
                print_row(r)

        if period_label == "OUT-OF-SAMPLE":
            oos_results = results

    # Walk-forward
    print("\n" + "=" * 140)
    print("  WALK-FORWARD PF RATIOS")
    print("=" * 140)

    for label, buf, skip in configs:
        cfg = make_cfg(buf)
        is_trades = run_backtest_stagger(df_is.copy(), cfg, n_contracts=3, uniform_skip=skip)
        oos_trades = run_backtest_stagger(df_oos.copy(), cfg, n_contracts=3, uniform_skip=skip)
        is_r = analyze(is_trades, label)
        oos_r = analyze(oos_trades, label)
        if is_r and oos_r and is_r["pf"] > 0:
            ratio = oos_r["pf"] / is_r["pf"]
            grade = "ROBUST" if ratio >= 0.7 else "ACCEPTABLE" if ratio >= 0.5 else "WEAK"
            print(f"  {label:>18}:  IS PF {is_r['pf']:.2f}  ->  OOS PF {oos_r['pf']:.2f}  "
                  f"->  Ratio {ratio:.2f}  {grade}")

    # Head to head
    print("\n" + "=" * 140)
    print("  THE QUESTION: 3pt/C2 vs 3pt/C3 vs 7pt/C3")
    print("=" * 140)

    c2_3 = oos_results.get("3pt / 2nd support")
    c3_3 = oos_results.get("3pt / 3rd support")
    c3_7 = oos_results.get("7pt / 3rd support")

    if c2_3 and c3_3 and c3_7:
        print(f"\n  3pt / 2nd support:  ${c2_3['lvl_pnl']:>+9,.0f} LVL  |  "
              f"WR {c2_3['wr']:.0f}%  |  R:R {c2_3['avg_rr']:.2f}  |  "
              f"Combined ${c2_3['combined']:>+9,.0f}")
        print(f"  3pt / 3rd support:  ${c3_3['lvl_pnl']:>+9,.0f} LVL  |  "
              f"WR {c3_3['wr']:.0f}%  |  R:R {c3_3['avg_rr']:.2f}  |  "
              f"Combined ${c3_3['combined']:>+9,.0f}")
        print(f"  7pt / 3rd support:  ${c3_7['lvl_pnl']:>+9,.0f} LVL  |  "
              f"WR {c3_7['wr']:.0f}%  |  R:R {c3_7['avg_rr']:.2f}  |  "
              f"Combined ${c3_7['combined']:>+9,.0f}")

        best_label = max([(c2_3, "3pt/C2"), (c3_3, "3pt/C3"), (c3_7, "7pt/C3")],
                         key=lambda x: x[0]["lvl_pnl"])
        print(f"\n  Winner: {best_label[1]} with ${best_label[0]['lvl_pnl']:+,.0f} OOS LVL P&L")

    print()


if __name__ == "__main__":
    main()
