#!/usr/bin/env python3
"""Nexus Long — AutoResearch v3 wrapper.

Reads tunable params from nexus_long_ar_config.py, runs backtest with
VIX inverted filter and entry cutoff, outputs in AR-compatible format.

Walk-forward split: 2025-02-14 (1yr IS / 1yr OOS).

Baseline: 255t, PF 1.589, +$27,962, p=0.0043, WF 1.03 PASS
  Config: dVAL + dPOC + ONL/ONH + OS gap-down, VIX inversion @17, cutoff 15:00
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
from backtester.strategies.nexus_long_ar_config import cfg


def t_test_pnls(pnls):
    if len(pnls) < 5:
        return 1.0
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    return t_pval / 2 if t_stat > 0 else 1 - t_pval / 2


def profit_factor(pnls):
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return wins / losses if losses > 0 else float("inf")


def make_config():
    """Build StrategyConfig from AR params."""
    sc = StrategyConfig()
    sc.direction_filter = "long"

    # All OFF except MS + OS
    sc.use_ib_break = False
    sc.use_va_fade = False
    sc.use_eighty = False
    sc.use_tema_cross = False
    sc.use_level_reject = False
    sc.use_level_reject_long = False
    sc.use_ib_reject = False
    sc.use_var = False
    sc.use_ptf = False
    sc.use_fa = False

    # MS from AR config
    sc.use_ms = True
    sc.ms_zone_pts = cfg.ms_zone_pts
    sc.ms_stop_buffer = cfg.ms_stop_buffer
    sc.ms_min_target_pts = cfg.ms_min_target_pts
    sc.ms_min_rr = cfg.ms_min_rr
    sc.ms_max_risk = 25.0
    sc.ms_ma_type = "sma"
    sc.ms_ma_confirm_bars = 0
    sc.max_ms_trades = 8
    sc.ms_use_vp_levels = True
    sc.ms_use_prev_va = True
    sc.ms_use_on_levels = True
    sc.ms_use_ib_levels = False
    sc.ms_use_dev_va = True
    sc.ms_use_poc = True
    sc.ms_level_directions = {
        "MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short",
        "MS_dVAL": "long", "MS_dPOC": "long",
    }

    # OS from AR config
    sc.use_os = True
    sc.os_stop_mode = "on_extreme"
    sc.os_stop_buffer = cfg.os_stop_buffer
    sc.os_max_risk = 25.0
    sc.os_target_mode = "cascade"
    sc.os_min_target_pts = 3.0
    sc.os_min_rr = 0.5
    sc.os_require_on_sweep = True
    sc.os_require_ma = False
    sc.max_os_trades = 1
    sc.os_min_gap = 3.0
    sc.os_max_gap = 20.0
    sc.os_entry_window = 1

    return sc


def apply_vix_filter(trades, vix_df):
    """Inverted VIX: MS when VIX > thresh, OS when VIX <= thresh."""
    result = []
    cutoff_h = cfg.entry_cutoff // 100
    cutoff_m = cfg.entry_cutoff % 100

    for t in trades:
        # Entry time cutoff
        if t.entry_time.hour > cutoff_h or (t.entry_time.hour == cutoff_h and t.entry_time.minute >= cutoff_m):
            continue

        vix_row = vix_df[vix_df.index.date == t.entry_time.date()]
        if len(vix_row) == 0:
            continue
        vix_open = vix_row.iloc[0]["open"]

        if t.setup.startswith("MS_") and vix_open > cfg.vix_ms_thresh:
            result.append(t)
        elif t.setup.startswith("OS_") and vix_open <= cfg.vix_os_thresh:
            result.append(t)
    return result


def run_on_period(df_period, vix_df, sc):
    """Run backtest on a data period and apply filters."""
    raw_trades = run_backtest(df_period.copy(), sc)
    return apply_vix_filter(raw_trades, vix_df)


def main():
    # Load data
    df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
    vix = pd.read_parquet("../spx/data/vix_daily.parquet")

    sc = make_config()

    # Full period backtest
    all_raw = run_backtest(df.copy(), sc)
    trades = apply_vix_filter(all_raw, vix)

    if not trades:
        print("Total Trades: 0")
        print("Profit Factor: 0.000")
        return

    pnls = [t.pnl_dollar for t in trades]
    m = compute_metrics(trades, sc.initial_capital)
    p_val = t_test_pnls(pnls)

    print("=" * 60)
    print("  Nexus Long (dVAL+dPOC+VIX) — AR Backtest")
    print("=" * 60)
    print(f"Total Trades: {m.total_trades}")
    print(f"Win Rate: {m.win_rate:.1f}%")
    print(f"Profit Factor: {m.profit_factor:.3f}")
    print(f"Total P&L: ${m.net_pnl:+,.2f}")
    print(f"Sharpe Ratio: {m.sharpe:.2f}")
    print()

    # T-Test section
    print("--- T-Test ---")
    print(f"  p-value: {p_val:.6f}")
    print()

    # Walk-forward: 1yr IS / 1yr OOS
    WF_SPLIT = pd.Timestamp("2025-02-14", tz="US/Eastern")

    is_trades = [t for t in trades if t.entry_time < WF_SPLIT]
    oos_trades = [t for t in trades if t.entry_time >= WF_SPLIT]

    is_pnls = [t.pnl_dollar for t in is_trades]
    oos_pnls = [t.pnl_dollar for t in oos_trades]

    if is_pnls and oos_pnls:
        is_pf = profit_factor(is_pnls)
        oos_pf = profit_factor(oos_pnls)
        oos_p = t_test_pnls(oos_pnls)

        if is_pf > 0 and is_pf != float("inf"):
            wf_ratio = oos_pf / is_pf
        else:
            wf_ratio = 0.0

        print("--- Walk-Forward (1yr/1yr split) ---")
        print(f"  OOS PF:        {oos_pf:.3f}")
        print(f"  WF PF ratio:   {wf_ratio:.3f}")
        print(f"  OOS Trades:    {len(oos_pnls)}")
        print(f"  OOS p-value:   {oos_p:.6f}")
    else:
        print("--- Walk-Forward ---")
        print("  Insufficient data for IS/OOS split")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
