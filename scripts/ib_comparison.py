#!/usr/bin/env python3
"""IB Comparison: 60-min vs 30-min ORB across all ES strategies.

Runs AMT-TEMA v8, Nexus (MS+OS), and Sentinel (LVL v13) with both
60-min IB (12 bars, original) and 30-min IB (6 bars, corrected).

Usage:
    cd ~/projects/ai-playground
    .venv/bin/python scripts/ib_comparison.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv, tag_sessions
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


def make_v8_config():
    """AMT-TEMA v8: IB Breakout, short-only, 30bps pct stop."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    return cfg


def make_nexus_config():
    """Nexus (MS+OS): Market Structure + Overnight Sweep."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_fa = False
    # MS Config B
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 4.0
    cfg.ms_min_target_pts = 8.0
    cfg.ms_min_rr = 0.3
    cfg.ms_max_risk = 25.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False
    cfg.ms_level_directions = {
        "MS_ONH": "both",
        "MS_ONL": "both",
        "MS_pVAH": "short",
    }
    # OS Best
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = 5.0
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 3.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_require_ma = False
    cfg.max_os_trades = 1
    cfg.os_min_gap = 3.0
    cfg.os_max_gap = 20.0
    cfg.os_entry_window = 1
    return cfg


def make_sentinel_config():
    """Sentinel (LVL v13): ONH Level Rejection, 3x C3."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_fa = False
    cfg.use_ms = False
    cfg.use_os = False
    # LVL v13
    cfg.use_level_reject = True
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 7.0
    cfg.lvl_require_tema = True
    cfg.lvl_ma_filter = "tema"
    cfg.max_lvl_trades = 4
    cfg.lvl_ibh_wide_only = True
    cfg.lvl_max_tests = 3
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_rr = 0.5
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_target_skip = 2          # 3rd support (C3)
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.pct_stop_mode = False        # LVL uses fixed buffer, not pct
    return cfg


def retag_ib(df, ib_end, trade_start):
    """Re-tag session columns with new IB timing."""
    tag_sessions(df, ib_end=ib_end, trade_start=trade_start)


def run_strategy(df, cfg, label):
    """Run backtest and return metrics dict."""
    trades = run_backtest(df.copy(), cfg)
    if not trades:
        return {"label": label, "trades": 0, "pf": 0, "pnl": 0, "wr": 0,
                "sharpe": 0, "dd": 0, "p_val": 1.0, "raw_trades": []}

    m = compute_metrics(trades, cfg.initial_capital)
    pnls = [t.pnl_dollar for t in trades]
    _, p_val = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)

    # Walk-forward split at midpoint
    split_date = df.index[len(df) // 2]
    t1 = [t for t in trades if t.entry_time < split_date]
    t2 = [t for t in trades if t.entry_time >= split_date]
    wf_ratio = 0
    y1_pf = 0
    y2_pf = 0
    y1_pnl = 0
    y2_pnl = 0
    if t1 and t2:
        m1 = compute_metrics(t1, cfg.initial_capital)
        m2 = compute_metrics(t2, cfg.initial_capital)
        y1_pf = m1.profit_factor
        y2_pf = m2.profit_factor
        y1_pnl = m1.net_pnl
        y2_pnl = m2.net_pnl
        wf_ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0

    # Per-setup breakdown
    setups = {}
    for t in trades:
        if t.setup not in setups:
            setups[t.setup] = {"count": 0, "pnl": 0, "wins": 0}
        setups[t.setup]["count"] += 1
        setups[t.setup]["pnl"] += t.pnl_dollar
        if t.pnl_dollar > 0:
            setups[t.setup]["wins"] += 1

    return {
        "label": label,
        "trades": m.total_trades,
        "pf": m.profit_factor,
        "pnl": m.net_pnl,
        "wr": m.win_rate,
        "sharpe": m.sharpe,
        "dd": m.max_drawdown,
        "p_val": p_val,
        "y1_pf": y1_pf, "y2_pf": y2_pf,
        "y1_pnl": y1_pnl, "y2_pnl": y2_pnl,
        "wf_ratio": wf_ratio,
        "setups": setups,
        "raw_trades": trades,
    }


def print_result(r):
    """Print one strategy result."""
    wf_verdict = "PASS" if r["wf_ratio"] > 0.7 and r.get("y2_pf", 0) > 1.0 else "MARGINAL" if r["wf_ratio"] > 0.5 else "FAIL"
    sig = "***" if r["p_val"] < 0.01 else "**" if r["p_val"] < 0.05 else "*" if r["p_val"] < 0.10 else "NS"

    print(f"  Trades:     {r['trades']:>6d}")
    print(f"  Net P&L:    ${r['pnl']:>+12,.0f}")
    print(f"  Win Rate:   {r['wr']:>8.1f}%")
    print(f"  Profit Fac: {r['pf']:>8.3f}")
    print(f"  Sharpe:     {r['sharpe']:>8.2f}")
    print(f"  Max DD:     ${r['dd']:>8,.0f}")
    print(f"  p-value:    {r['p_val']:>8.4f}  {sig}")
    print(f"  Y1: PF={r.get('y1_pf',0):.3f} ${r.get('y1_pnl',0):>+9,.0f}  |  Y2: PF={r.get('y2_pf',0):.3f} ${r.get('y2_pnl',0):>+9,.0f}")
    print(f"  WF Ratio:   {r['wf_ratio']:.2f}  → {wf_verdict}")

    if r.get("setups"):
        print(f"  Per-Setup:")
        for s, d in sorted(r["setups"].items(), key=lambda x: -x[1]["pnl"]):
            wr = d["wins"] / d["count"] * 100 if d["count"] else 0
            print(f"    {s:<15s}  {d['count']:>4d}t  {wr:>5.1f}% WR  ${d['pnl']:>+9,.0f}")


def print_comparison(label, r60, r30):
    """Print side-by-side comparison."""
    print(f"\n{'─' * 70}")
    print(f"  {label}: 60-min IB vs 30-min IB")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<20s}  {'60-min IB':>15s}  {'30-min IB':>15s}  {'Delta':>12s}")
    print(f"  {'─' * 65}")

    rows = [
        ("Trades", r60["trades"], r30["trades"], "d"),
        ("Net P&L", r60["pnl"], r30["pnl"], "$"),
        ("Win Rate %", r60["wr"], r30["wr"], "%"),
        ("Profit Factor", r60["pf"], r30["pf"], "f"),
        ("Sharpe", r60["sharpe"], r30["sharpe"], "f"),
        ("Max DD", r60["dd"], r30["dd"], "$"),
        ("p-value", r60["p_val"], r30["p_val"], "p"),
        ("WF Ratio", r60["wf_ratio"], r30["wf_ratio"], "f"),
    ]

    for name, v60, v30, fmt in rows:
        delta = v30 - v60
        if fmt == "$":
            print(f"  {name:<20s}  ${v60:>+13,.0f}  ${v30:>+13,.0f}  ${delta:>+10,.0f}")
        elif fmt == "d":
            print(f"  {name:<20s}  {v60:>15d}  {v30:>15d}  {delta:>+12d}")
        elif fmt == "%":
            print(f"  {name:<20s}  {v60:>14.1f}%  {v30:>14.1f}%  {delta:>+11.1f}%")
        elif fmt == "p":
            s60 = "***" if v60 < 0.01 else "**" if v60 < 0.05 else "NS"
            s30 = "***" if v30 < 0.01 else "**" if v30 < 0.05 else "NS"
            print(f"  {name:<20s}  {v60:>12.4f} {s60:>2s}  {v30:>12.4f} {s30:>2s}")
        else:
            print(f"  {name:<20s}  {v60:>15.3f}  {v30:>15.3f}  {delta:>+12.3f}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

print("Loading data...")
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
print(f"Loaded {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
print(f"RTH bars: {df['is_rth'].sum()}, Sessions: {df['new_rth'].sum()}")

strategies = [
    ("AMT-TEMA v8 (IB Breakout)", make_v8_config),
    ("Nexus (MS+OS)", make_nexus_config),
    ("Sentinel (LVL v13)", make_sentinel_config),
]

results_60 = {}
results_30 = {}

# ── Run with 60-min IB (original: 9:30-10:30, 12 bars) ──
print(f"\n{'=' * 70}")
print(f"  PHASE 1: 60-MINUTE IB (original, ib_end=1030)")
print(f"{'=' * 70}")
retag_ib(df, ib_end=1030, trade_start=1035)

for name, make_cfg in strategies:
    print(f"\n  Running {name}...")
    cfg = make_cfg()
    cfg.ib_end_time = 1030
    cfg.trade_start = 1035
    r = run_strategy(df, cfg, name)
    results_60[name] = r
    print_result(r)

# ── Run with 30-min IB (corrected: 9:30-10:00, 6 bars) ──
print(f"\n{'=' * 70}")
print(f"  PHASE 2: 30-MINUTE IB (corrected, ib_end=1000)")
print(f"{'=' * 70}")
retag_ib(df, ib_end=1000, trade_start=1005)

for name, make_cfg in strategies:
    print(f"\n  Running {name}...")
    cfg = make_cfg()
    cfg.ib_end_time = 1000
    cfg.trade_start = 1005
    r = run_strategy(df, cfg, name)
    results_30[name] = r
    print_result(r)

# ── Comparisons ──
print(f"\n\n{'=' * 70}")
print(f"  COMPARISON: 60-MIN IB vs 30-MIN IB")
print(f"{'=' * 70}")

for name, _ in strategies:
    print_comparison(name, results_60[name], results_30[name])

# ── Summary verdict ──
print(f"\n\n{'=' * 70}")
print(f"  VERDICT")
print(f"{'=' * 70}")

for name, _ in strategies:
    r60 = results_60[name]
    r30 = results_30[name]
    better = "30-min" if r30["pnl"] > r60["pnl"] else "60-min"
    sig60 = r60["p_val"] < 0.05
    sig30 = r30["p_val"] < 0.05
    wf60 = r60["wf_ratio"] > 0.7 and r60.get("y2_pf", 0) > 1.0
    wf30 = r30["wf_ratio"] > 0.7 and r30.get("y2_pf", 0) > 1.0

    print(f"\n  {name}:")
    print(f"    60-min: {'SIGNIFICANT' if sig60 else 'NOT SIG'}, {'WF PASS' if wf60 else 'WF FAIL'}, PF={r60['pf']:.3f}, ${r60['pnl']:+,.0f}")
    print(f"    30-min: {'SIGNIFICANT' if sig30 else 'NOT SIG'}, {'WF PASS' if wf30 else 'WF FAIL'}, PF={r30['pf']:.3f}, ${r30['pnl']:+,.0f}")
    print(f"    Better IB: {better} (by ${abs(r30['pnl'] - r60['pnl']):,.0f})")

print(f"\n{'=' * 70}")
print(f"  DONE")
print(f"{'=' * 70}")
