#!/usr/bin/env python3
"""Analyze C3 stop location, exit breakdown, and flatten details."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.stagger_engine import run_backtest_stagger


def make_lvl():
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
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = True
    cfg.lvl_ma_filter = "tema"
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 8.0
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_min_rr = 0.5
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3
    return cfg


def main():
    df = load_tos_csv("data/es_5m_databento_2yr.csv")
    split = "2025-02-14"
    df_oos = df[df.index >= split]

    cfg = make_lvl()

    oos_trades = run_backtest_stagger(df_oos.copy(), cfg, n_contracts=3, uniform_skip=2)
    full_trades = run_backtest_stagger(df.copy(), cfg, n_contracts=3, uniform_skip=2)

    oos_lvl = [t for t in oos_trades if t.setup.startswith("LVL")]
    full_lvl = [t for t in full_trades if t.setup.startswith("LVL")]

    print("=" * 70)
    print("  STOP CONFIGURATION")
    print("=" * 70)
    print(f"  Stop = ONH level + {cfg.lvl_stop_buffer} pts (buffer)")
    print(f"  Capped by: {cfg.pct_stop_bps} bps above entry (whichever is tighter)")
    print(f"  At ES ~6000: pct stop = 6000 * 30/10000 = {6000 * 30 / 10000:.1f} pts above entry")
    print(f"  So stop is MIN(ONH + 8, entry + ~18) -- typically the 8pt buffer wins")
    print()

    # Stop distance analysis
    for label, trades in [("OOS", oos_lvl), ("Full 2yr", full_lvl)]:
        if not trades:
            continue
        stops = [t.stop - t.entry_price for t in trades]
        targets = [t.entry_price - t.target for t in trades]
        rrs = [tgt / stp if stp > 0 else 0 for stp, tgt in zip(stops, targets)]

        print(f"  {label} LVL Stop/Target Analysis ({len(trades)} fills):")
        print(f"    Avg stop distance:   {np.mean(stops):.1f} pts (${np.mean(stops) * 50:.0f})")
        print(f"    Avg target distance: {np.mean(targets):.1f} pts (${np.mean(targets) * 50:.0f})")
        print(f"    Avg R:R ratio:       {np.mean(rrs):.2f}")
        print(f"    Median R:R:          {np.median(rrs):.2f}")
        print()

    print("=" * 70)
    print("  EXIT BREAKDOWN (C3 trades)")
    print("=" * 70)

    for label, trades in [("OOS", oos_lvl), ("Full 2yr", full_lvl)]:
        if not trades:
            continue

        by_reason = {}
        for t in trades:
            by_reason.setdefault(t.exit_reason, []).append(t)

        total = len(trades)
        entries = total // 3
        print(f"\n  {label} ({total} fills, 3 contracts per entry = {entries} entries):")

        for reason in ["target", "stop", "flatten"]:
            rtrades = by_reason.get(reason, [])
            count = len(rtrades)
            pct = count / total * 100
            pnl = sum(t.pnl_dollar for t in rtrades)
            avg = np.mean([t.pnl_dollar for t in rtrades]) if rtrades else 0
            print(f"    {reason:>8}: {count:>4} ({pct:5.1f}%) | P&L ${pnl:+,.0f} | avg ${avg:+,.0f}/fill")

        target_hits = len(by_reason.get("target", []))
        stops = len(by_reason.get("stop", []))
        flattens = len(by_reason.get("flatten", []))

        print(f"\n    Target hit rate: {target_hits}/{total} = {target_hits / total * 100:.1f}%")
        print(f"    Stop out rate:   {stops}/{total} = {stops / total * 100:.1f}%")
        print(f"    Flatten rate:    {flattens}/{total} = {flattens / total * 100:.1f}%")

        # Flatten analysis
        flatten_trades = by_reason.get("flatten", [])
        if flatten_trades:
            flatten_winners = [t for t in flatten_trades if t.pnl_dollar > 0]
            flatten_losers = [t for t in flatten_trades if t.pnl_dollar <= 0]
            print(f"\n    Flatten detail:")
            print(f"      Winning flattens: {len(flatten_winners)} (moved our way, didnt reach C3 target)")
            print(f"      Losing flattens:  {len(flatten_losers)} (moved against, didnt hit stop)")
            if flatten_winners:
                fw_pnls = [t.pnl_dollar for t in flatten_winners]
                print(f"      Winning flatten total: ${sum(fw_pnls):+,.0f} | avg ${np.mean(fw_pnls):+,.0f}")
                print(f"\n      Sample winning flattens (money left on table):")
                for t in flatten_winners[:8]:
                    dist_to_target = t.entry_price - t.target
                    dist_achieved = t.entry_price - t.exit_price
                    pct_of_target = dist_achieved / dist_to_target * 100 if dist_to_target > 0 else 0
                    remaining = dist_to_target - dist_achieved
                    print(f"        Entry {t.entry_price:.2f} -> Exit {t.exit_price:.2f} | "
                          f"Got {pct_of_target:.0f}% of target | "
                          f"{remaining:.1f} pts left to go | +${t.pnl_dollar:,.0f}")

    print()


if __name__ == "__main__":
    main()
