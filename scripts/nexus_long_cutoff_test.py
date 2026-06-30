#!/usr/bin/env python3
"""Quick test: entry time cutoff for Nexus Long.

15:00 hour is the only losing entry window (-$2,180).
Test cutoffs to find the sweet spot, then bake it into AR config.

Usage:
    cd ~/projects/backtesting/es
    python3 scripts/nexus_long_cutoff_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


def make_config(trade_end=1600):
    cfg = StrategyConfig()
    cfg.direction_filter = "long"
    cfg.trade_end = trade_end
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
    cfg.ms_use_dev_va = True
    cfg.ms_use_poc = True
    cfg.ms_level_directions = {
        "MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short",
        "MS_dVAL": "long", "MS_dPOC": "long",
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


def apply_vix_filter(trades, vix_df, ms_thresh=17, os_thresh=17):
    result = []
    for t in trades:
        vix_row = vix_df[vix_df.index.date == t.entry_time.date()]
        if len(vix_row) == 0:
            continue
        t.vix_open = vix_row.iloc[0]["open"]
        if t.setup.startswith("MS_") and t.vix_open > ms_thresh:
            result.append(t)
        elif t.setup.startswith("OS_") and t.vix_open <= os_thresh:
            result.append(t)
    return result


print("Loading data...")
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
vix = pd.read_parquet("../spx/data/vix_daily.parquet")
print(f"ES: {len(df):,} bars\n")

print(f"  {'Cutoff':<12s}  {'#':>4s}  {'WR':>6s}  {'PF':>7s}  {'P&L':>10s}  {'Avg':>8s}  {'DD':>8s}  {'Sharpe':>7s}  {'p-val':>10s}")
print(f"  {'─' * 95}")

for cutoff in [1600, 1500, 1455, 1430, 1400, 1345, 1300]:
    cfg = make_config(trade_end=cutoff)
    trades = run_backtest(df.copy(), cfg)
    trades = apply_vix_filter(trades, vix)

    if not trades:
        print(f"  {cutoff:<12d}    0t")
        continue

    m = compute_metrics(trades, cfg.initial_capital)
    pnls = [t.pnl_dollar for t in trades]
    _, p = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    # WF
    split = pd.Timestamp("2025-02-14", tz="US/Eastern")
    t1 = [t for t in trades if t.entry_time < split]
    t2 = [t for t in trades if t.entry_time >= split]
    wf = ""
    if t1 and t2 and len(t1) >= 3 and len(t2) >= 3:
        p1 = [t.pnl_dollar for t in t1]
        p2 = [t.pnl_dollar for t in t2]
        gw1 = sum(p for p in p1 if p > 0); gl1 = abs(sum(p for p in p1 if p <= 0))
        gw2 = sum(p for p in p2 if p > 0); gl2 = abs(sum(p for p in p2 if p <= 0))
        pf1 = gw1/gl1 if gl1 > 0 else float("inf")
        pf2 = gw2/gl2 if gl2 > 0 else float("inf")
        ratio = pf2/pf1 if pf1 > 0 else 0
        _, p2v = stats.ttest_1samp(p2, 0) if len(p2) >= 5 else (0, 1.0)
        v = "PASS" if ratio > 0.7 and pf2 > 1.0 else "FAIL"
        wf = f"  WF {ratio:.2f} {v} (Y2: {len(t2)}t PF {pf2:.3f} p={p2v:.4f})"

    print(f"  {cutoff:<12d}  {len(trades):>4d}  {m.win_rate:>5.1f}%  {m.profit_factor:>7.3f}  ${m.net_pnl:>+9,.0f}  ${m.avg_trade:>+7,.0f}  ${m.max_drawdown:>7,.0f}  {m.sharpe:>6.2f}  p={p:.4f} {sig}{wf}")

print()
