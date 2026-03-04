#!/usr/bin/env python3
"""Combined OS + MS Strategy Test.

OS = Overnight Sweep (gap-up fade, short-only)
MS = Market Structure Config B (ON bilateral + pVAH short, SMA 8/24)

Tests:
  1. MS Config B alone (baseline) — verify ~325t, PF 1.399
  2. OS Wide alone (WF-passing) — gap2-40, buf5, win12
  3. OS Best alone (highest PF, fails WF) — gap3-20, buf5, win1
  4. MS + OS Wide — combined
  5. MS + OS Best — combined
  6. Walk-forward on all configs
  7. Per-setup breakdown on combined configs

Usage:
    python3 scripts/os_ms_combined.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
import numpy as np

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


def _base_off():
    """All setups OFF."""
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
    cfg.use_ms = False
    cfg.use_fa = False
    cfg.use_os = False
    return cfg


def make_ms_b():
    """MS Config B — ON bilateral + pVAH Short."""
    cfg = _base_off()
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
    return cfg


def make_os_wide():
    """OS Wide — walk-forward passing config (gap2-40, buf5, win12)."""
    cfg = _base_off()
    cfg.direction_filter = "short"
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = 5.0
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 4.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_require_ma = False
    cfg.max_os_trades = 2
    cfg.os_min_gap = 2.0
    cfg.os_max_gap = 40.0
    cfg.os_entry_window = 12
    return cfg


def make_os_best():
    """OS Best — highest PF from deep sweep (gap3-20, buf5, win1)."""
    cfg = _base_off()
    cfg.direction_filter = "short"
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


def make_combined_wide():
    """MS Config B + OS Wide."""
    cfg = make_ms_b()
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = 5.0
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 4.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_require_ma = False
    cfg.max_os_trades = 2
    cfg.os_min_gap = 2.0
    cfg.os_max_gap = 40.0
    cfg.os_entry_window = 12
    # Combined uses direction_filter="both" so MS longs work too
    cfg.direction_filter = "both"
    return cfg


def make_combined_best():
    """MS Config B + OS Best."""
    cfg = make_ms_b()
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
    cfg.direction_filter = "both"
    return cfg


def make_combined_tight():
    """MS Config B + OS Tight (WF-passing, gap5-25, buf2, win3)."""
    cfg = make_ms_b()
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = 2.0
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 4.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_require_ma = False
    cfg.max_os_trades = 2
    cfg.os_min_gap = 5.0
    cfg.os_max_gap = 25.0
    cfg.os_entry_window = 3
    cfg.direction_filter = "both"
    return cfg


def pr(label, trades, show_detail=False):
    """Print results for a config."""
    if not trades:
        print(f"  {label:<50s}  NO TRADES")
        return None
    m = compute_metrics(trades)
    pnls = [t.pnl_dollar for t in trades]
    _, p = stats.ttest_1samp(pnls, 0) if len(trades) >= 5 else (0, 1.0)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    l_pnl = sum(t.pnl_dollar for t in longs)
    s_pnl = sum(t.pnl_dollar for t in shorts)

    print(f"  {label:<50s}  {m.total_trades:>4d}t  WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+9,.0f}  DD ${m.max_drawdown:>7,.0f}  Sh {m.sharpe:>5.2f}  "
          f"p={p:.3f}{sig}  T/D {m.trades_per_day:.2f}  "
          f"L${l_pnl:>+7,.0f}({len(longs)}) S${s_pnl:>+7,.0f}({len(shorts)})")

    if show_detail:
        # Per-setup breakdown
        setups = {}
        for t in trades:
            s = t.setup
            setups.setdefault(s, [])
            setups[s].append(t)
        print(f"    {'Setup':<20s}  {'#':>4s}  {'WR':>5s}  {'PF':>6s}  {'P&L':>9s}  {'Avg':>7s}")
        for s, st in sorted(setups.items(), key=lambda x: -sum(t.pnl_dollar for t in x[1])):
            sm = compute_metrics(st)
            avg = sum(t.pnl_dollar for t in st) / len(st)
            print(f"    {s:<20s}  {len(st):>4d}  {sm.win_rate:>5.1f}  {sm.profit_factor:>6.3f}  "
                  f"${sum(t.pnl_dollar for t in st):>+9,.0f}  ${avg:>+7,.0f}")

        # Exit reasons
        reasons = {}
        for t in trades:
            r = t.exit_reason
            reasons.setdefault(r, {"count": 0, "pnl": 0})
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t.pnl_dollar
        print(f"    {'Exit':<12s}  {'#':>4s}  {'P&L':>9s}  {'WR':>5s}")
        for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            wr = len([t for t in trades if t.exit_reason == r and t.pnl_dollar > 0]) / d["count"] * 100
            print(f"    {r:<12s}  {d['count']:>4d}  ${d['pnl']:>+9,.0f}  {wr:>5.1f}%")

    return m


def walk_forward(label, cfg, df, split_idx):
    """Run walk-forward validation."""
    df_y1 = df.iloc[:split_idx].copy()
    df_y2 = df.iloc[split_idx:].copy()

    t1 = run_backtest(df_y1, cfg)
    t2 = run_backtest(df_y2, cfg)
    t_all = run_backtest(df.copy(), cfg)

    if not t1 or not t2 or not t_all:
        no = "Y1" if not t1 else "Y2" if not t2 else "FULL"
        print(f"    {label:<45s}  NO TRADES ({no})")
        return

    m1 = compute_metrics(t1)
    m2 = compute_metrics(t2)
    m_all = compute_metrics(t_all)
    _, p_all = stats.ttest_1samp([t.pnl_dollar for t in t_all], 0) if len(t_all) >= 5 else (0, 1.0)
    _, p2 = stats.ttest_1samp([t.pnl_dollar for t in t2], 0) if len(t2) >= 5 else (0, 1.0)
    ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
    verdict = "PASS" if ratio > 0.7 and m2.profit_factor > 1.0 else "MARGINAL" if ratio > 0.5 else "FAIL"

    sig_all = "***" if p_all < 0.01 else "**" if p_all < 0.05 else "*" if p_all < 0.10 else ""
    sig_2 = "***" if p2 < 0.01 else "**" if p2 < 0.05 else "*" if p2 < 0.10 else ""

    print(f"    {label}")
    print(f"      FULL: {m_all.total_trades:>4d}t  PF={m_all.profit_factor:.3f}  ${m_all.net_pnl:>+9,.0f}  "
          f"DD=${m_all.max_drawdown:>7,.0f}  Sh={m_all.sharpe:.2f}  p={p_all:.4f}{sig_all}")
    print(f"      Y1:   {m1.total_trades:>4d}t  PF={m1.profit_factor:.3f}  ${m1.net_pnl:>+9,.0f}")
    print(f"      Y2:   {m2.total_trades:>4d}t  PF={m2.profit_factor:.3f}  ${m2.net_pnl:>+9,.0f}  p={p2:.4f}{sig_2}")
    print(f"      WF ratio={ratio:.2f}  {verdict}\n")


# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
print(f"Loaded {len(df)} bars\n")

split_idx = df.index.get_indexer(["2025-02-14"], method="nearest")[0]
print(f"Walk-forward split: Y1 ends {df.index[split_idx-1].date()}, Y2 starts {df.index[split_idx].date()}\n")

# ═══════════════════════════════════════════════════════════════
# Stage 1: Individual Strategies (verify baselines)
# ═══════════════════════════════════════════════════════════════
print("=" * 160)
print("  STAGE 1: INDIVIDUAL STRATEGIES")
print("=" * 160)

configs = {
    "MS Config B (ON + pVAH short)": make_ms_b(),
    "OS Wide (gap2-40, buf5, win12)": make_os_wide(),
    "OS Best (gap3-20, buf5, win1)": make_os_best(),
}

for label, cfg in configs.items():
    trades = run_backtest(df.copy(), cfg)
    pr(label, trades, show_detail=True)
    print()

# ═══════════════════════════════════════════════════════════════
# Stage 2: Combined Strategies
# ═══════════════════════════════════════════════════════════════
print("=" * 160)
print("  STAGE 2: COMBINED MS + OS")
print("=" * 160)

combined = {
    "MS+OS Wide (gap2-40, buf5, win12)": make_combined_wide(),
    "MS+OS Best (gap3-20, buf5, win1)": make_combined_best(),
    "MS+OS Tight (gap5-25, buf2, win3)": make_combined_tight(),
}

for label, cfg in combined.items():
    trades = run_backtest(df.copy(), cfg)
    pr(label, trades, show_detail=True)
    print()

# ═══════════════════════════════════════════════════════════════
# Stage 3: Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════
print("=" * 160)
print("  STAGE 3: WALK-FORWARD VALIDATION")
print("=" * 160)

all_configs = {**configs, **combined}
for label, cfg in all_configs.items():
    walk_forward(label, cfg, df, split_idx)

# ═══════════════════════════════════════════════════════════════
# Stage 4: Interaction Analysis (does OS add or cannibalize?)
# ═══════════════════════════════════════════════════════════════
print("=" * 160)
print("  STAGE 4: INTERACTION ANALYSIS")
print("=" * 160)

# Run MS alone and combined to check overlap
ms_trades = run_backtest(df.copy(), make_ms_b())
combo_wide_trades = run_backtest(df.copy(), make_combined_wide())
combo_best_trades = run_backtest(df.copy(), make_combined_best())

ms_m = compute_metrics(ms_trades)
cw_m = compute_metrics(combo_wide_trades)
cb_m = compute_metrics(combo_best_trades)

print(f"\n  MS alone:       {ms_m.total_trades:>4d} trades  ${ms_m.net_pnl:>+9,.0f}  PF {ms_m.profit_factor:.3f}")
print(f"  MS+OS Wide:     {cw_m.total_trades:>4d} trades  ${cw_m.net_pnl:>+9,.0f}  PF {cw_m.profit_factor:.3f}")
print(f"  MS+OS Best:     {cb_m.total_trades:>4d} trades  ${cb_m.net_pnl:>+9,.0f}  PF {cb_m.profit_factor:.3f}")

# Check OS trades in combined
os_in_wide = [t for t in combo_wide_trades if t.setup.startswith("OS_")]
os_in_best = [t for t in combo_best_trades if t.setup.startswith("OS_")]
ms_in_wide = [t for t in combo_wide_trades if t.setup.startswith("MS_")]
ms_in_best = [t for t in combo_best_trades if t.setup.startswith("MS_")]

print(f"\n  MS+OS Wide breakdown:")
print(f"    MS trades: {len(ms_in_wide):>4d}  ${sum(t.pnl_dollar for t in ms_in_wide):>+9,.0f}")
print(f"    OS trades: {len(os_in_wide):>4d}  ${sum(t.pnl_dollar for t in os_in_wide):>+9,.0f}")
print(f"    Total:     {len(combo_wide_trades):>4d}  ${sum(t.pnl_dollar for t in combo_wide_trades):>+9,.0f}")

print(f"\n  MS+OS Best breakdown:")
print(f"    MS trades: {len(ms_in_best):>4d}  ${sum(t.pnl_dollar for t in ms_in_best):>+9,.0f}")
print(f"    OS trades: {len(os_in_best):>4d}  ${sum(t.pnl_dollar for t in os_in_best):>+9,.0f}")
print(f"    Total:     {len(combo_best_trades):>4d}  ${sum(t.pnl_dollar for t in combo_best_trades):>+9,.0f}")

# Check if OS is cannibalizing MS (fewer MS trades when OS is on)
ms_alone_count = ms_m.total_trades
ms_in_combo_wide = len(ms_in_wide)
ms_in_combo_best = len(ms_in_best)
print(f"\n  Cannibalization check:")
print(f"    MS alone:     {ms_alone_count} trades")
print(f"    MS in +Wide:  {ms_in_combo_wide} trades ({ms_in_combo_wide - ms_alone_count:+d} delta)")
print(f"    MS in +Best:  {ms_in_combo_best} trades ({ms_in_combo_best - ms_alone_count:+d} delta)")

# Additive P&L check
os_wide_alone = run_backtest(df.copy(), make_os_wide())
os_best_alone = run_backtest(df.copy(), make_os_best())
os_wide_pnl = sum(t.pnl_dollar for t in os_wide_alone)
os_best_pnl = sum(t.pnl_dollar for t in os_best_alone)
ms_pnl = ms_m.net_pnl

print(f"\n  Additive vs actual P&L:")
print(f"    MS + OS Wide (sum):    ${ms_pnl + os_wide_pnl:>+9,.0f}  (actual: ${cw_m.net_pnl:>+9,.0f}, diff: ${cw_m.net_pnl - (ms_pnl + os_wide_pnl):>+,.0f})")
print(f"    MS + OS Best (sum):    ${ms_pnl + os_best_pnl:>+9,.0f}  (actual: ${cb_m.net_pnl:>+9,.0f}, diff: ${cb_m.net_pnl - (ms_pnl + os_best_pnl):>+,.0f})")

# Trades per day
trading_days = len(set(df[df["is_rth"]].index.date))
print(f"\n  Trades per day ({trading_days} trading days):")
print(f"    MS alone:     {ms_m.total_trades / trading_days:.2f}")
print(f"    MS+OS Wide:   {cw_m.total_trades / trading_days:.2f}")
print(f"    MS+OS Best:   {cb_m.total_trades / trading_days:.2f}")

print(f"\n{'=' * 160}")
print("  DONE")
print(f"{'=' * 160}")
