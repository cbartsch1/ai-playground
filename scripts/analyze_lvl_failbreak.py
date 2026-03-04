#!/usr/bin/env python3
"""Targeted analysis: failed_break + TEMA combined sweep for Level Rejection.

Tests the best filter combo from Stage 2, with year split and param sweeps.
"""

import sys, os
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


def make_cfg(**overrides):
    """Level Rejection: failed_break + TEMA required on v8 base."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_level_reject = True
    cfg.lvl_trigger = "failed_break"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 8.0
    cfg.lvl_require_tema = True
    cfg.max_lvl_trades = 4
    cfg.lvl_ibh_wide_only = True
    cfg.lvl_max_tests = 3
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def pm(label, trades):
    m = compute_metrics(trades)
    print(f"  {label:<40s}  {m.total_trades:>4d} trades  "
          f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+10,.0f}  DD ${m.max_drawdown:>8,.0f}  "
          f"Sharpe {m.sharpe:>5.2f}")
    return m


def hdr(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/es_5m_databento_2yr.csv"
    print(f"Loading {csv_file}...")
    df = load_tos_csv(csv_file, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    midpoint = len(df) // 2
    df_y2 = df.iloc[:midpoint].copy()
    df_y1 = df.iloc[midpoint:].copy()
    print(f"Year 2: {df_y2.index[0].date()} → {df_y2.index[-1].date()}")
    print(f"Year 1: {df_y1.index[0].date()} → {df_y1.index[-1].date()}")

    # ── Baseline: failed_break + TEMA (default params) ──
    hdr("BASELINE: failed_break + TEMA required")
    cfg_base = make_cfg()

    print("\n  --- v8 reference ---")
    from scripts.run_backtest import apply_v8_flags
    cfg_v8 = StrategyConfig()
    apply_v8_flags(cfg_v8)
    pm("v8 (IB Breakout only)", run_backtest(df.copy(), cfg_v8))

    print("\n  --- failed_break + TEMA (full 2yr) ---")
    trades_all = run_backtest(df.copy(), cfg_base)
    m_all = pm("Combined", trades_all)

    print("\n  --- Year split ---")
    trades_y2 = run_backtest(df_y2, cfg_base)
    m_y2 = pm("Year 2 (older, ~ES 5000)", trades_y2)
    trades_y1 = run_backtest(df_y1, cfg_base)
    m_y1 = pm("Year 1 (recent, ~ES 6800)", trades_y1)

    if m_y1.net_pnl > 0 and m_y2.net_pnl > 0:
        print(f"\n  BOTH YEARS PROFITABLE")
    else:
        losers = []
        if m_y2.net_pnl <= 0: losers.append("Year 2")
        if m_y1.net_pnl <= 0: losers.append("Year 1")
        print(f"\n  WARNING: {', '.join(losers)} losing")

    # Per-setup breakdown
    if trades_all:
        print(f"\n  Per-Setup Breakdown (full 2yr):")
        for setup, sm in sorted(per_setup_breakdown(trades_all).items()):
            avg = sm.net_pnl / sm.total_trades if sm.total_trades > 0 else 0
            print(f"    {setup:<10s}  {sm.total_trades:>4d} trades  "
                  f"WR {sm.win_rate:>5.1f}%  PF {sm.profit_factor:>6.3f}  "
                  f"P&L ${sm.net_pnl:>+10,.0f}  Avg ${avg:>+7,.0f}")

    # Exit reasons
    if trades_all:
        print(f"\n  Exit Reasons:")
        reasons = defaultdict(lambda: {"n": 0, "pnl": 0})
        for t in trades_all:
            reasons[t.exit_reason]["n"] += 1
            reasons[t.exit_reason]["pnl"] += t.pnl_dollar
        for r, d in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
            print(f"    {r:<12s}  {d['n']:>4d} trades  ${d['pnl']:>+10,.0f}")

    # ── Parameter sweeps (all with failed_break + TEMA) ──
    hdr("PARAMETER SWEEPS (failed_break + TEMA base)")

    print("\n  --- Zone Points ---")
    for zp in [2.0, 3.0, 5.0, 8.0]:
        pm(f"zone={zp}", run_backtest(df.copy(), make_cfg(lvl_zone_pts=zp)))

    print("\n  --- Stop Buffer ---")
    for sb in [3.0, 5.0, 8.0, 12.0, 15.0]:
        pm(f"stop_buf={sb}", run_backtest(df.copy(), make_cfg(lvl_stop_buffer=sb)))

    print("\n  --- Max Tests ---")
    for mt in [2, 3, 4, 99]:
        label = f"max_tests={mt}" if mt < 99 else "max_tests=unlimited"
        pm(label, run_backtest(df.copy(), make_cfg(lvl_max_tests=mt)))

    print("\n  --- Max Trades/Day ---")
    for mt in [2, 3, 4, 6]:
        pm(f"max_trades={mt}", run_backtest(df.copy(), make_cfg(max_lvl_trades=mt)))

    print("\n  --- IBH Wide Filter ---")
    pm("IBH wide-only", run_backtest(df.copy(), make_cfg(lvl_ibh_wide_only=True)))
    pm("IBH all days", run_backtest(df.copy(), make_cfg(lvl_ibh_wide_only=False)))

    print("\n  --- TEMA on vs off (with failed_break) ---")
    pm("failed_break + TEMA", run_backtest(df.copy(), make_cfg(lvl_require_tema=True)))
    pm("failed_break only", run_backtest(df.copy(), make_cfg(lvl_require_tema=False)))

    # ── Best combo: sweep top combos with year split ──
    hdr("TOP COMBOS — YEAR SPLIT")

    combos = [
        ("Base (zone=5, buf=8, tests=3)", {}),
        ("zone=3, buf=5, tests=3", dict(lvl_zone_pts=3.0, lvl_stop_buffer=5.0)),
        ("zone=3, buf=8, tests=2", dict(lvl_zone_pts=3.0, lvl_max_tests=2)),
        ("zone=3, buf=5, tests=2", dict(lvl_zone_pts=3.0, lvl_stop_buffer=5.0, lvl_max_tests=2)),
        ("zone=5, buf=5, tests=2", dict(lvl_stop_buffer=5.0, lvl_max_tests=2)),
        ("zone=3, buf=3, tests=3", dict(lvl_zone_pts=3.0, lvl_stop_buffer=3.0)),
        ("zone=5, buf=12, tests=4", dict(lvl_stop_buffer=12.0, lvl_max_tests=4)),
    ]

    print(f"\n  {'Config':<35s}  {'2yr':>6s}  {'Y2 PF':>6s}  {'Y1 PF':>6s}  {'2yr P&L':>10s}  {'Y2 P&L':>10s}  {'Y1 P&L':>10s}  {'2yr Sharpe':>10s}")
    print(f"  {'-'*110}")

    for label, overrides in combos:
        cfg = make_cfg(**overrides)
        t_all = run_backtest(df.copy(), cfg)
        t_y2 = run_backtest(df_y2.copy(), cfg)
        t_y1 = run_backtest(df_y1.copy(), cfg)
        ma = compute_metrics(t_all)
        m2 = compute_metrics(t_y2)
        m1 = compute_metrics(t_y1)
        both = "Y" if m1.net_pnl > 0 and m2.net_pnl > 0 else "N"
        print(f"  {label:<35s}  {ma.total_trades:>4d}t  {m2.profit_factor:>6.3f}  {m1.profit_factor:>6.3f}  "
              f"${ma.net_pnl:>+9,.0f}  ${m2.net_pnl:>+9,.0f}  ${m1.net_pnl:>+9,.0f}  "
              f"{ma.sharpe:>6.2f}  {'BOTH' if both=='Y' else 'FAIL'}")


if __name__ == "__main__":
    main()
