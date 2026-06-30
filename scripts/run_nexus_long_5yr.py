#!/usr/bin/env python3
"""Nexus Long — 5yr Revalidation on CLEAN data.

Uses production AR-optimized config from nexus_long_ar_config.py.
Data: es_5m_5yr.csv (2017-01-02 -> 2021-12-31, databento single-provider).
VIX: VIX_daily_1990_2026.csv (full history, close-value only).

Baseline on 2yr clean (just reconfirmed): 182t, PF 1.643, +$23,240, OOS PF 1.931, WF 1.56.

WF split: 2020-01-01 (3yr IS 2017-2019 / 2yr OOS 2020-2021).
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
    sc = StrategyConfig()
    sc.direction_filter = "long"

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


def load_vix_long_history():
    """Load VIX daily from 1990-2026 CSV, return as dict {date: close_value}."""
    path = "../spx/data/VIX_daily_1990_2026.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return dict(zip(df["Date"], df["VIX"]))


def apply_vix_filter(trades, vix_by_date):
    """Inverted VIX: MS when VIX > thresh, OS when VIX <= thresh.
    Uses daily VIX close as proxy for 'open' (CSV has close-only)."""
    result = []
    cutoff_h = cfg.entry_cutoff // 100
    cutoff_m = cfg.entry_cutoff % 100

    for t in trades:
        if t.entry_time.hour > cutoff_h or (t.entry_time.hour == cutoff_h and t.entry_time.minute >= cutoff_m):
            continue

        trade_date = t.entry_time.date()
        vix_level = vix_by_date.get(trade_date)
        if vix_level is None:
            continue

        if t.setup.startswith("MS_") and vix_level > cfg.vix_ms_thresh:
            result.append(t)
        elif t.setup.startswith("OS_") and vix_level <= cfg.vix_os_thresh:
            result.append(t)
    return result


def report_block(label, trades, initial_capital):
    if not trades:
        print(f"  {label}: 0 trades")
        return None
    pnls = [t.pnl_dollar for t in trades]
    m = compute_metrics(trades, initial_capital)
    p_val = t_test_pnls(pnls)
    print(f"  {label}: {m.total_trades}t  PF={m.profit_factor:.3f}  "
          f"${m.net_pnl:+,.0f}  WR={m.win_rate:.1f}%  Sh={m.sharpe:.2f}  p={p_val:.4f}")
    return m


def main():
    print("=" * 70)
    print("  NEXUS LONG — 5yr CLEAN DATA REVALIDATION")
    print("=" * 70)

    df = load_tos_csv("data/es_5m_5yr.csv", instrument="ES")
    print(f"  Bars: {len(df):,} | {df.index[0].date()} -> {df.index[-1].date()}")

    vix_by_date = load_vix_long_history()
    print(f"  VIX days loaded: {len(vix_by_date):,} (1990-2026 CSV)")

    sc = make_config()
    print(f"  Config: MS zone={cfg.ms_zone_pts}pt stop_buf={cfg.ms_stop_buffer} target={cfg.ms_min_target_pts}pt RR>={cfg.ms_min_rr}")
    print(f"          OS stop_buf={cfg.os_stop_buffer}")
    print(f"          VIX: MS>{cfg.vix_ms_thresh} OS<={cfg.vix_os_thresh} | cutoff={cfg.entry_cutoff}ET")

    # Full-period backtest
    print("\n  Running backtest...")
    all_raw = run_backtest(df.copy(), sc)
    trades = apply_vix_filter(all_raw, vix_by_date)

    if not trades:
        print("\n  FAIL: zero trades after VIX filter.")
        return

    print(f"\n  Raw signals: {len(all_raw)} | After VIX filter: {len(trades)}")

    # ── Full 5yr ──
    print(f"\n  {'-' * 66}")
    print(f"  FULL 5YR PERIOD")
    print(f"  {'-' * 66}")
    m_all = report_block("FULL", trades, sc.initial_capital)

    # ── Walk-forward (3yr IS / 2yr OOS) ──
    WF_SPLIT = pd.Timestamp("2020-01-01", tz="US/Eastern")
    is_trades = [t for t in trades if t.entry_time < WF_SPLIT]
    oos_trades = [t for t in trades if t.entry_time >= WF_SPLIT]

    print(f"\n  {'-' * 66}")
    print(f"  WALK-FORWARD (split 2020-01-01, 3yr IS / 2yr OOS)")
    print(f"  {'-' * 66}")
    m_is = report_block("IS  ", is_trades, sc.initial_capital)
    m_oos = report_block("OOS ", oos_trades, sc.initial_capital)

    if m_is and m_oos and m_is.profit_factor > 0:
        wf_ratio = m_oos.profit_factor / m_is.profit_factor
        print(f"  WF PF ratio: {wf_ratio:.3f}")

    # ── Yearly breakdown ──
    print(f"\n  {'-' * 66}")
    print(f"  YEARLY P&L")
    print(f"  {'-' * 66}")
    for year in [2017, 2018, 2019, 2020, 2021]:
        year_trades = [t for t in trades if t.entry_time.year == year]
        if year_trades:
            pnl = sum(t.pnl_dollar for t in year_trades)
            wins = sum(1 for t in year_trades if t.pnl_dollar > 0)
            wr = wins / len(year_trades) * 100
            print(f"  {year}: {len(year_trades):>4}t  ${pnl:>+10,.0f}  WR {wr:.1f}%")
        else:
            print(f"  {year}: 0 trades")

    # ── Verdict ──
    print(f"\n  {'-' * 66}")
    print(f"  VERDICT")
    print(f"  {'-' * 66}")
    if m_all and m_oos:
        tests = [
            ("5yr PF > 1.3", m_all.profit_factor > 1.3),
            ("5yr p < 0.05", t_test_pnls([t.pnl_dollar for t in trades]) < 0.05),
            ("OOS PF > 1.0", m_oos.profit_factor > 1.0),
            ("OOS p < 0.10", t_test_pnls([t.pnl_dollar for t in oos_trades]) < 0.10),
        ]
        passes = sum(1 for _, p in tests if p)
        for name, p in tests:
            print(f"  [{'PASS' if p else 'FAIL'}] {name}")
        if passes == 4:
            print(f"\n  >>> HOLD — edge confirmed on 5yr clean data <<<")
        elif passes >= 2:
            print(f"\n  >>> DEGRADE — partial edge on 5yr <<<")
        else:
            print(f"\n  >>> BREAK — edge does not hold on 5yr <<<")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
