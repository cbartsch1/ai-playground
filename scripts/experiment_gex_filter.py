#!/usr/bin/env python3
"""Experiment: GEX Regime Filter — Join SqueezeMetrics GEX to ES Backtester.

Hypothesis: IB Rejection PF jumps on high GEX, IB Breakout PF jumps on low GEX.
Positive GEX = dealers buy dips/sell rallies = mean-reversion = rejection alpha.
Negative/low GEX = dealers amplify moves = momentum = breakout alpha.

Data: lab/data/squeezemetrics_dix_gex.csv (3,728 days, 2011-2026, FREE)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics

DATA = "data/es_5m_databento_2yr.csv"
GEX_DATA = os.path.expanduser("~/projects/lab/data/squeezemetrics_dix_gex.csv")
SPLIT_DATE = pd.Timestamp("2025-02-14", tz="America/New_York")


def v81_config():
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_ib_break = True
    cfg.use_trend_filter = True
    cfg.use_ib_reject = True
    cfg.rej_wide_only = True
    cfg.rej_target = "ib_low"
    cfg.rej_stop_buffer = 8.0
    cfg.max_rej_trades = 8
    return cfg


def v13_config():
    cfg = v81_config()
    cfg.use_level_reject = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = False
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 7.0
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_rr = 0.5
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_target_skip = 2
    return cfg


def load_gex():
    """Load SqueezeMetrics GEX data and compute quintiles."""
    gex = pd.read_csv(GEX_DATA, parse_dates=["date"])
    gex = gex.rename(columns={"date": "date", "gex": "gex", "dix": "dix", "price": "price"})
    gex["date_key"] = gex["date"].dt.date
    return gex


def assign_quintile(gex_df, col="gex"):
    """Assign GEX quintile labels (1=lowest, 5=highest)."""
    gex_df["gex_quintile"] = pd.qcut(gex_df[col], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    return gex_df


def print_metrics(label, m, trades):
    pnl = sum(t.pnl_dollar for t in trades)
    print(f"  {label}: {m.total_trades:>4} trades, PF {m.profit_factor:>5.2f}, "
          f"WR {m.win_rate:>5.1f}%, ${pnl:>+10,.0f}")


def run_gex_analysis(df, cfg, label, gex_df):
    """Run backtest and split results by GEX quintile."""
    print(f"\n{'='*70}")
    print(f"  {label} — GEX Quintile Analysis")
    print(f"{'='*70}")

    # Run full backtest
    all_trades = run_backtest(df.copy(), cfg)
    print(f"  Total trades: {len(all_trades)}")

    # Build GEX lookup by date
    gex_lookup = {}
    for _, row in gex_df.iterrows():
        gex_lookup[row["date_key"]] = {
            "gex": row["gex"],
            "quintile": row["gex_quintile"],
            "dix": row["dix"],
        }

    # Split trades by GEX quintile
    quintile_buckets = {}
    setup_buckets = {}  # Also split by setup within quintile
    unmatched = 0

    for t in all_trades:
        trade_date = t.entry_time.date() if hasattr(t.entry_time, 'date') else None
        if trade_date and trade_date in gex_lookup:
            q = gex_lookup[trade_date]["quintile"]
            if q not in quintile_buckets:
                quintile_buckets[q] = []
            quintile_buckets[q].append(t)

            # Track by setup+quintile
            key = (q, t.setup)
            if key not in setup_buckets:
                setup_buckets[key] = []
            setup_buckets[key].append(t)
        else:
            unmatched += 1

    if unmatched:
        print(f"  Trades without GEX data: {unmatched}")

    # Report by quintile
    print(f"\n  --- Full 2-Year by GEX Quintile ---")
    print(f"  {'Quintile':<16} {'Trades':>7} {'PF':>7} {'WR':>7} {'P&L':>12}")
    print(f"  {'-'*52}")

    for q in ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]:
        trades = quintile_buckets.get(q, [])
        if len(trades) >= 5:
            m = compute_metrics(trades)
            pnl = sum(t.pnl_dollar for t in trades)
            print(f"  {q:<16} {m.total_trades:>7} {m.profit_factor:>7.2f} {m.win_rate:>6.1f}% ${pnl:>+10,.0f}")
        else:
            print(f"  {q:<16} {len(trades):>7} trades (too few)")

    # Report by setup within quintile
    setups = sorted(set(t.setup for t in all_trades))
    if len(setups) > 1:
        print(f"\n  --- By Setup × GEX Quintile ---")
        for setup in setups:
            print(f"\n  {setup}:")
            print(f"  {'Quintile':<16} {'Trades':>7} {'PF':>7} {'WR':>7} {'P&L':>12}")
            print(f"  {'-'*52}")
            for q in ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]:
                key = (q, setup)
                trades = setup_buckets.get(key, [])
                if len(trades) >= 3:
                    m = compute_metrics(trades)
                    pnl = sum(t.pnl_dollar for t in trades)
                    print(f"  {q:<16} {m.total_trades:>7} {m.profit_factor:>7.2f} {m.win_rate:>6.1f}% ${pnl:>+10,.0f}")
                else:
                    print(f"  {q:<16} {len(trades):>7} trades (too few)")

    # GEX value stats for matched trades
    print(f"\n  --- GEX Distribution for Our Trading Period ---")
    our_dates = set()
    for t in all_trades:
        if hasattr(t.entry_time, 'date'):
            our_dates.add(t.entry_time.date())

    our_gex = [gex_lookup[d]["gex"] for d in our_dates if d in gex_lookup]
    if our_gex:
        our_gex = np.array(our_gex)
        print(f"  Trading days with GEX data: {len(our_gex)}")
        print(f"  GEX range: ${our_gex.min()/1e9:.2f}B to ${our_gex.max()/1e9:.2f}B")
        print(f"  GEX mean:  ${our_gex.mean()/1e9:.2f}B")
        print(f"  GEX median: ${np.median(our_gex)/1e9:.2f}B")
        pct_positive = (our_gex > 0).sum() / len(our_gex) * 100
        print(f"  Positive GEX: {pct_positive:.1f}%")

    # IS/OOS split by quintile
    print(f"\n  --- IS/OOS by GEX Quintile ---")
    trades_is = run_backtest(df[df.index < SPLIT_DATE].copy(), cfg)
    trades_oos = run_backtest(df[df.index >= SPLIT_DATE].copy(), cfg)

    for period_label, trades in [("IS ", trades_is), ("OOS", trades_oos)]:
        print(f"\n  {period_label}:")
        q_buckets = {}
        for t in trades:
            trade_date = t.entry_time.date() if hasattr(t.entry_time, 'date') else None
            if trade_date and trade_date in gex_lookup:
                q = gex_lookup[trade_date]["quintile"]
                if q not in q_buckets:
                    q_buckets[q] = []
                q_buckets[q].append(t)

        for q in ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]:
            trades_q = q_buckets.get(q, [])
            if len(trades_q) >= 3:
                m = compute_metrics(trades_q)
                pnl = sum(t.pnl_dollar for t in trades_q)
                print(f"    {q:<16} {m.total_trades:>4} trades, PF {m.profit_factor:>5.2f}, ${pnl:>+9,.0f}")
            else:
                print(f"    {q:<16} {len(trades_q):>4} trades (too few)")


def main():
    print("Loading ES data...")
    df = load_tos_csv(DATA, instrument="ES")
    print(f"Loaded {len(df)} bars")

    print("Loading GEX data...")
    gex_df = load_gex()
    print(f"Loaded {len(gex_df)} GEX records")

    # Filter GEX to our trading period
    es_dates = set(df.index.date)
    gex_match = gex_df[gex_df["date_key"].isin(es_dates)]
    print(f"GEX records matching ES dates: {len(gex_match)}")

    # Assign quintiles on FULL history (not just our period)
    gex_df = assign_quintile(gex_df)

    # Show quintile boundaries
    print(f"\n  GEX Quintile Boundaries (full history):")
    for q in ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]:
        subset = gex_df[gex_df["gex_quintile"] == q]["gex"]
        print(f"    {q}: ${subset.min()/1e9:.2f}B to ${subset.max()/1e9:.2f}B ({len(subset)} days)")

    run_gex_analysis(df.copy(), v81_config(), "v8.1 (IB Breakout + Rejection)", gex_df)
    run_gex_analysis(df.copy(), v13_config(), "v13 (ONH Level Rejection)", gex_df)

    print(f"\n{'='*70}")
    print("  GEX Filter Experiment Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
