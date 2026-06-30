#!/usr/bin/env python3
"""Post-hoc analysis: Does VIX filter improve Nexus Long trades?

Grok's thesis: VIX > 18 → longs work better because:
  - Elevated fear = panic sellers covering
  - Gap-down bounces snap harder
  - Support levels hold better (capitulation → mean reversion)

Method: Run long-only backtest, then slice trades by VIX level.
If edge concentrates in high-VIX periods, wire it into the engine.

Usage:
    cd ~/projects/backtesting/es
    python3 scripts/nexus_long_vix_filter.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


def make_long_config():
    """Long-only MS+OS — Config A from nexus_long_test.py."""
    cfg = StrategyConfig()
    cfg.direction_filter = "long"
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
        "MS_pVAH": "short",  # won't fire with direction_filter="long"
    }

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


def analyze_vix_slice(trades, label, show_setups=True):
    """Analyze a slice of trades."""
    if not trades:
        print(f"  {label:<30s}  {'---':>4s}  {'---':>6s}  {'---':>7s}  {'---':>10s}  {'---':>8s}")
        return

    pnls = [t.pnl_dollar for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p <= 0))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = wins / len(trades) * 100
    net = sum(pnls)
    avg = net / len(trades)

    _, p_val = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

    print(f"  {label:<30s}  {len(trades):>4d}  {wr:>5.1f}%  {pf:>7.3f}  ${net:>+9,.0f}  ${avg:>+7,.0f}  p={p_val:.4f} {sig}")

    if show_setups and len(trades) >= 10:
        by_setup = defaultdict(list)
        for t in trades:
            by_setup[t.setup].append(t)
        for setup in sorted(by_setup.keys()):
            st = by_setup[setup]
            sp = [t.pnl_dollar for t in st]
            sw = sum(1 for p in sp if p > 0)
            sgw = sum(p for p in sp if p > 0)
            sgl = abs(sum(p for p in sp if p <= 0))
            spf = sgw / sgl if sgl > 0 else float("inf")
            swr = sw / len(st) * 100
            print(f"    {setup:<28s}  {len(st):>4d}  {swr:>5.1f}%  {spf:>7.3f}  ${sum(sp):>+9,.0f}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

print("Loading data...")
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
vix = pd.read_parquet("../spx/data/vix_daily.parquet")
print(f"ES: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")
print(f"VIX: {len(vix)} days ({vix.index[0].date()} to {vix.index[-1].date()})")

# Run long-only backtest
cfg = make_long_config()
trades = run_backtest(df.copy(), cfg)
print(f"\nLong-only trades: {len(trades)}")

# Attach VIX data to each trade
for t in trades:
    trade_date = t.entry_time.date()
    # Find VIX for this date (use .date() on both)
    vix_row = vix[vix.index.date == trade_date]
    if len(vix_row) > 0:
        t.vix_open = vix_row.iloc[0]["open"]
        t.vix_high = vix_row.iloc[0]["high"]
        t.vix_close = vix_row.iloc[0]["close"]
    else:
        t.vix_open = None
        t.vix_high = None
        t.vix_close = None

# Filter out trades without VIX data
trades_with_vix = [t for t in trades if t.vix_open is not None]
trades_no_vix = [t for t in trades if t.vix_open is None]
print(f"Trades with VIX data: {len(trades_with_vix)}, without: {len(trades_no_vix)}")

# ═══════════════════════════════════════════════════════════════
#  VIX THRESHOLD SWEEP (using VIX open — known at entry time)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print(f"  VIX THRESHOLD SWEEP — Long Trades Only")
print(f"  Filter: VIX open > threshold (known at RTH open, no lookahead)")
print(f"{'=' * 90}")
print(f"  {'Filter':<30s}  {'#':>4s}  {'WR':>6s}  {'PF':>7s}  {'P&L':>10s}  {'Avg':>8s}  {'Sig':>12s}")
print(f"  {'─' * 85}")

analyze_vix_slice(trades_with_vix, "ALL (no VIX filter)")

thresholds = [14, 16, 18, 20, 22, 25]
for thresh in thresholds:
    above = [t for t in trades_with_vix if t.vix_open > thresh]
    analyze_vix_slice(above, f"VIX open > {thresh}")

print(f"\n  {'─' * 85}")
print(f"  INVERSE (low VIX = calm markets)")
print(f"  {'─' * 85}")
for thresh in [14, 16, 18, 20]:
    below = [t for t in trades_with_vix if t.vix_open <= thresh]
    analyze_vix_slice(below, f"VIX open <= {thresh}", show_setups=False)

# ═══════════════════════════════════════════════════════════════
#  VIX HIGH FILTER (intraday VIX spike — uses daily high)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print(f"  VIX HIGH FILTER — Did VIX spike intraday?")
print(f"  Note: VIX high includes full day, slight lookahead for OS (first bar)")
print(f"{'=' * 90}")
print(f"  {'Filter':<30s}  {'#':>4s}  {'WR':>6s}  {'PF':>7s}  {'P&L':>10s}  {'Avg':>8s}  {'Sig':>12s}")
print(f"  {'─' * 85}")

for thresh in [18, 20, 22, 25]:
    above = [t for t in trades_with_vix if t.vix_high > thresh]
    analyze_vix_slice(above, f"VIX high > {thresh}")

# ═══════════════════════════════════════════════════════════════
#  VIX SPIKE FILTER (intraday spike = high/open > threshold)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print(f"  VIX INTRADAY SPIKE — VIX high / VIX open (fear acceleration)")
print(f"{'=' * 90}")
print(f"  {'Filter':<30s}  {'#':>4s}  {'WR':>6s}  {'PF':>7s}  {'P&L':>10s}  {'Avg':>8s}  {'Sig':>12s}")
print(f"  {'─' * 85}")

for spike_pct in [1.03, 1.05, 1.07, 1.10]:
    spiked = [t for t in trades_with_vix if t.vix_high / t.vix_open >= spike_pct]
    analyze_vix_slice(spiked, f"VIX spike >= {(spike_pct-1)*100:.0f}%")

# ═══════════════════════════════════════════════════════════════
#  YEAR 2 ONLY (where the edge was emerging)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 90}")
print(f"  YEAR 2 ONLY (2025-02-14 onward) + VIX FILTER")
print(f"{'=' * 90}")

y2_trades = [t for t in trades_with_vix if t.entry_time >= pd.Timestamp("2025-02-14", tz="US/Eastern")]
print(f"  Y2 trades: {len(y2_trades)}")
print(f"  {'Filter':<30s}  {'#':>4s}  {'WR':>6s}  {'PF':>7s}  {'P&L':>10s}  {'Avg':>8s}  {'Sig':>12s}")
print(f"  {'─' * 85}")

analyze_vix_slice(y2_trades, "Y2 ALL (no VIX filter)")
for thresh in [14, 16, 18, 20, 22]:
    above = [t for t in y2_trades if t.vix_open > thresh]
    analyze_vix_slice(above, f"Y2 VIX open > {thresh}")

print()
