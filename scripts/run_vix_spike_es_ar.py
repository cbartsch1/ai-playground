#!/usr/bin/env python3
"""VIX Spike ES — AutoResearch v3 wrapper.

Reads tunable params from vix_spike_es_ar_config.py, runs backtest
with hold-all-day exit, outputs in AR-compatible format.

Walk-forward split: 2025-02-16 (1yr IS / 1yr OOS, ES convention).
Production config: NO ES move filter (filter-free gives 119t/PF 5.00 vs
filtered 107t/PF 4.26 — filter delays entries on winning days).
"""

import os
import sys

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.test_vix_spike_es_v2 import (
    Config, load_data, run_backtest, compute_metrics,
)
from backtester.strategies.vix_spike_es_ar_config import cfg


def t_test_pnls(pnls):
    if len(pnls) < 5:
        return 1.0
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    return t_pval / 2 if t_stat > 0 else 1 - t_pval / 2


def profit_factor(pnls):
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return wins / losses if losses > 0 else float("inf")


def main():
    import argparse
    from scripts.test_vix_spike_es_v2 import LOOKAHEAD_MSG
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-lookahead-daily-mode", action="store_true",
                    help="Explicit opt-in to the NON-CAUSAL daily-VIX-HIGH "
                         "spike detection (diagnostics only).")
    args = ap.parse_args()
    if not args.allow_lookahead_daily_mode:
        raise SystemExit("VOIDED (2026-06-11 audit): "
                         + LOOKAHEAD_MSG.format(mode="daily_spike"))

    df, vix_lookup, session_opens, prev_session_data = load_data()

    # Build config from AR params — hold_all_day exit, NO ES move filter
    backtest_cfg = Config(
        signal_mode="daily_spike",
        allow_lookahead_daily_mode=True,
        spike_threshold=cfg.spike_threshold,
        entry_mode="red_bar",
        exit_mode="hold_all_day",
        stop_bps=cfg.stop_bps,
        max_hold_bars=cfg.max_hold_bars,
        es_move_filter=0.0,  # NO filter — proven better for hold-all-day
        entry_start=cfg.entry_start,
        entry_end=cfg.entry_end,
    )

    # Full period backtest
    trades = run_backtest(df, vix_lookup, session_opens, prev_session_data, backtest_cfg)

    if not trades:
        print("Total Trades: 0")
        print("Profit Factor: 0.000")
        return

    pnls = [t.pnl_dollar for t in trades]
    m = compute_metrics(trades)

    p_val = t_test_pnls(pnls)

    print("=" * 60)
    print("  VIX Spike ES (Hold All Day) — AR Backtest")
    print("=" * 60)
    print(f"Total Trades: {m['total']}")
    print(f"Win Rate: {m['win_rate']:.1f}%")
    print(f"Profit Factor: {m['pf']:.3f}")
    print(f"Total P&L: ${m['net_pnl']:+,.2f}")
    print(f"Sharpe Ratio: {m['sharpe']:.2f}")
    print()

    # T-Test section
    print("--- T-Test ---")
    print(f"  p-value: {p_val:.6f}")
    print()

    # Walk-forward: 1yr IS / 1yr OOS (split at 2025-02-16)
    WF_SPLIT = "2025-02-16"
    df_is = df[df.index < WF_SPLIT]
    df_oos = df[df.index >= WF_SPLIT]

    is_trades = run_backtest(df_is, vix_lookup, session_opens, prev_session_data, backtest_cfg)
    oos_trades = run_backtest(df_oos, vix_lookup, session_opens, prev_session_data, backtest_cfg)

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
