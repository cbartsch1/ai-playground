#!/usr/bin/env python3
"""Trend Continuation ES — 5yr Revalidation (post spread-tick cleanup).

Runs the PRODUCTION config from
  trade-violently-dashboard/execution/strategies/trend_cont_es.py
against clean 5yr ES 5m data (es_5m_5yr.csv, 2017-01-02..2021-12-31)
and compares to the prior 2yr audit numbers.

Caveats (documented, NOT silently papered over):
- Production strategy adds a 1.2x volume surge filter and a Friday filter.
  The test engine (run_trend_cont in test_trend_cont_es.py) does NOT implement
  either. This revalidation therefore tests a SUPERSET of prod entries
  (without vol-surge requirement, no weekday filter). Results here are a
  pessimistic-for-HOLD / optimistic-for-BREAK floor: if edge survives the
  larger trade pool it survives the filtered one; if it breaks here it is
  definitely broken.
- Production signal uses SPX 30m bars; backtest uses ES 30m bars. Prior
  "180 trades/2yr, PF 1.58" validated number came from THIS test script
  (ES-native). Apples-to-apples comparison.
- SPY target 2.5 pts maps to ~25 ES pts (10:1 SPX->SPY scale -> ~25 ES pts
  given recent price ratios). Using target_pts=25.0 in fixed_target mode.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from anywhere
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics
from scripts.test_trend_cont_es import (
    TrendContConfig,
    aggregate_30m,
    run_trend_cont,
    walk_forward_validate,
    print_metrics,
    INITIAL_CAPITAL,
)
from scipy import stats

# ── Paths ──
DATA_5YR = PROJECT_DIR / "data" / "es_5m_5yr.csv"
DATA_2YR = PROJECT_DIR / "data" / "es_5m_databento_2yr.csv"

# ── Production config (from execution/strategies/trend_cont_es.py) ──
# TARGET_PTS        = 2.5     (SPY pts) ~= 25 ES pts at 10:1 SPX/SPY ratio
# STOP_BPS          = 30      (pct stop)
# MAX_TRADES_DAY    = 1
# MAX_HOLD_BARS     = 6       (6 x 5m = 30 min)
# ENTRY_START       = 09:35   (935)
# ENTRY_END         = 13:00   (1300)
# VOL_SURGE_MULT    = 1.2     (NOT implemented in test engine — see caveats)
# Friday filter     = ON      (NOT implemented in test engine — see caveats)
PROD_CFG = TrendContConfig(
    entry_start=935,
    entry_end=1300,
    max_hold_bars=6,
    stop_bps=30.0,
    max_trades_day=1,
    min_gap_bars=6,
    exit_mode="fixed_target",
    target_pts=25.0,       # SPY 2.5 pts -> ~25 ES pts
    use_tema=False,
)


def ttest_pval(trades):
    pnls = [t.pnl_dollar for t in trades]
    if len(pnls) > 1 and np.std(pnls) > 0:
        t_stat, p = stats.ttest_1samp(pnls, 0)
        return p / 2 if t_stat > 0 else 1.0
    return 1.0


def yearly_breakdown(trades, label):
    if not trades:
        print(f"\n  Yearly P&L ({label}): no trades")
        return
    df = pd.DataFrame([
        {"year": t.entry_time.year, "pnl": t.pnl_dollar, "win": t.pnl_dollar > 0}
        for t in trades
    ])
    print(f"\n  Yearly breakdown ({label})")
    print(f"  {'─'*55}")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>7} {'Net P&L':>12} {'Avg':>10}")
    print(f"  {'─'*55}")
    for year, sub in df.groupby("year"):
        net = sub["pnl"].sum()
        wr = 100 * sub["win"].mean()
        avg = sub["pnl"].mean()
        print(f"  {int(year):<6} {len(sub):>7} {wr:>6.1f}% {net:>11,.0f} {avg:>9,.0f}")


def run_period(df_5m, label):
    df_30m = aggregate_30m(df_5m)
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  Bars 5m: {len(df_5m):,}  |  Bars 30m: {len(df_30m):,}")
    if len(df_5m):
        print(f"  Range: {df_5m.index.min()}  ->  {df_5m.index.max()}")
    print(f"{'='*72}")

    trades = run_trend_cont(df_5m, df_30m, PROD_CFG)
    print_metrics(f"FULL PERIOD — {label}", trades)
    yearly_breakdown(trades, label)
    return trades, df_30m


def run_wf(df_5m, df_30m, split_date, label):
    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD — {label}  (split: {split_date})")
    print(f"{'='*72}")
    wf = walk_forward_validate(df_5m, df_30m, PROD_CFG, split_date=split_date)
    print(f"\n  IN-SAMPLE  trades={wf['is_trades']:>4}  WR={wf['is_wr']:>5.1f}%  "
          f"PF={wf['is_pf']:>5.2f}  P&L=${wf['is_pnl']:>9,.0f}  "
          f"DD=${wf['is_dd']:>8,.0f}  Sharpe={wf['is_sharpe']:>5.2f}  p={wf['is_p']:.4f}")
    print(f"  OUT-SAMPLE trades={wf['oos_trades']:>4}  WR={wf['oos_wr']:>5.1f}%  "
          f"PF={wf['oos_pf']:>5.2f}  P&L=${wf['oos_pnl']:>9,.0f}  "
          f"DD=${wf['oos_dd']:>8,.0f}  Sharpe={wf['oos_sharpe']:>5.2f}  p={wf['oos_p']:.4f}")
    print(f"  PF ratio (OOS/IS) = {wf['pf_ratio']:.2f}  (>0.7 robust, >1.0 improving)")
    return wf


def main():
    print("=" * 72)
    print("  TREND CONT ES — 5YR REVALIDATION (clean ES data, post spread-tick purge)")
    print("=" * 72)
    print(f"\nProduction config (from execution/strategies/trend_cont_es.py):")
    print(f"  entry window:    {PROD_CFG.entry_start:04d} - {PROD_CFG.entry_end:04d} ET")
    print(f"  max hold bars:   {PROD_CFG.max_hold_bars} (= {PROD_CFG.max_hold_bars*5} min)")
    print(f"  stop:            {PROD_CFG.stop_bps} bps (pct-based)")
    print(f"  target:          {PROD_CFG.target_pts} ES pts (= SPY 2.5 pts @ 10:1)")
    print(f"  max trades/day:  {PROD_CFG.max_trades_day}")
    print(f"  exit mode:       {PROD_CFG.exit_mode}")
    print(f"  TEMA filter:     {PROD_CFG.use_tema}")
    print(f"  NOTE: prod's volume-surge (1.2x) and Friday filter NOT modeled here")
    print(f"        -> this test is a superset of prod entries (MORE trades than prod)")

    # ── 5yr CLEAN revalidation ──
    print(f"\n\n[1/2] Loading 5yr clean data: {DATA_5YR}")
    df5 = load_tos_csv(str(DATA_5YR), instrument="ES")
    trades_5yr, df5_30m = run_period(df5, "5YR CLEAN (2017-2021)")

    # WF split 2020-01-01: 3yr IS / 2yr OOS
    wf_5yr = run_wf(df5, df5_30m, "2020-01-01", "5YR split 2020-01-01")

    # ── 2yr CLEAN comparison vs prior audit ──
    print(f"\n\n[2/2] Loading 2yr clean data: {DATA_2YR}")
    df2 = load_tos_csv(str(DATA_2YR), instrument="ES")
    trades_2yr, df2_30m = run_period(df2, "2YR CLEAN (Databento)")
    wf_2yr = run_wf(df2, df2_30m, "2025-02-16", "2YR split 2025-02-16")

    # ── Summary ──
    print("\n\n" + "=" * 72)
    print("  SUMMARY — Revalidation Verdict Inputs")
    print("=" * 72)
    m5 = compute_metrics(trades_5yr, INITIAL_CAPITAL) if trades_5yr else None
    m2 = compute_metrics(trades_2yr, INITIAL_CAPITAL) if trades_2yr else None

    print(f"\n  Prior 2yr audit (per STRAT_STATE): 180 trades, PF 1.58, WF 1.52")

    if m5:
        print(f"\n  5yr CLEAN:      {m5.total_trades:>4}t  PF {m5.profit_factor:.2f}  "
              f"WR {m5.win_rate:.1f}%  P&L ${m5.net_pnl:,.0f}  Sharpe {m5.sharpe:.2f}  "
              f"DD ${m5.max_drawdown:,.0f}  p={ttest_pval(trades_5yr):.4f}")
        print(f"    5yr WF OOS:   {wf_5yr['oos_trades']:>4}t  PF {wf_5yr['oos_pf']:.2f}  "
              f"P&L ${wf_5yr['oos_pnl']:,.0f}  p={wf_5yr['oos_p']:.4f}  "
              f"ratio {wf_5yr['pf_ratio']:.2f}")

    if m2:
        print(f"\n  2yr CLEAN:      {m2.total_trades:>4}t  PF {m2.profit_factor:.2f}  "
              f"WR {m2.win_rate:.1f}%  P&L ${m2.net_pnl:,.0f}  Sharpe {m2.sharpe:.2f}  "
              f"DD ${m2.max_drawdown:,.0f}  p={ttest_pval(trades_2yr):.4f}")
        print(f"    2yr WF OOS:   {wf_2yr['oos_trades']:>4}t  PF {wf_2yr['oos_pf']:.2f}  "
              f"P&L ${wf_2yr['oos_pnl']:,.0f}  p={wf_2yr['oos_p']:.4f}  "
              f"ratio {wf_2yr['pf_ratio']:.2f}")

    # ── Verdict ──
    print("\n" + "=" * 72)
    if m5 and m5.profit_factor > 1.3 and wf_5yr['oos_pf'] > 1.0 and wf_5yr['oos_p'] < 0.05:
        verdict = "HOLD — edge confirmed on 5yr clean data"
    elif m5 and m5.profit_factor >= 1.0 and wf_5yr['oos_pf'] > 0.9:
        verdict = "DEGRADE — weaker edge but survives"
    else:
        verdict = "BREAK — edge does not hold on clean 5yr window"
    print(f"  VERDICT: {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
