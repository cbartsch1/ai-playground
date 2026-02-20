#!/usr/bin/env python3
"""Compare Python backtester trades vs TradingView trade list.

Usage:
    python scripts/validate_vs_tv.py data/es_5m.csv output/trade_log.csv

The TradingView trade list can be exported from Strategy Tester > List of Trades tab.
This script compares:
  1. Total trade count
  2. Per-trade direction, entry time, setup
  3. Aggregate P&L
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from backtester.config import ES_DEFAULTS, SPX_DEFAULTS
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Validate Python vs TradingView trades")
    parser.add_argument("csv_file", help="Path to thinkorSwim CSV data")
    parser.add_argument("tv_trades", nargs="?", default=None,
                        help="Path to TradingView trade list CSV (optional)")
    parser.add_argument("--instrument", "-i", default="ES", choices=["ES", "SPX"])
    args = parser.parse_args()

    cfg = SPX_DEFAULTS if args.instrument == "SPX" else ES_DEFAULTS

    print(f"Loading data from {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument=args.instrument)
    print(f"Loaded {len(df)} bars\n")

    print("Running Python backtest...")
    trades = run_backtest(df, cfg)
    m = compute_metrics(trades, cfg.initial_capital)

    print(f"\n{'='*50}")
    print(f"  Python Backtest Results")
    print(f"{'='*50}")
    print(f"  Trades:  {m.total_trades}")
    print(f"  WR:      {m.win_rate:.1f}%")
    print(f"  PF:      {m.profit_factor:.3f}")
    print(f"  P&L:     ${m.net_pnl:+,.2f}")
    print(f"  Max DD:  ${m.max_drawdown:,.2f}")

    # Print trade list for manual comparison
    print(f"\n{'='*50}")
    print(f"  Trade List (for TradingView comparison)")
    print(f"{'='*50}")
    print(f"  {'#':>3}  {'Setup':6}  {'Dir':5}  {'Entry Time':20}  {'Entry':>9}  {'Exit':>9}  {'P&L':>8}  {'Reason'}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*5}  {'-'*20}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*7}")

    for i, t in enumerate(trades, 1):
        direction = "LONG" if t.direction == 1 else "SHORT"
        entry_str = str(t.entry_time)[:19] if hasattr(t.entry_time, 'strftime') else str(t.entry_time)
        print(f"  {i:3d}  {t.setup:6s}  {direction:5s}  {entry_str:20s}  "
              f"{t.entry_price:9.2f}  {t.exit_price:9.2f}  {t.pnl_dollar:+8.2f}  {t.exit_reason}")

    if args.tv_trades:
        print(f"\n{'='*50}")
        print(f"  TradingView Comparison")
        print(f"{'='*50}")
        tv = pd.read_csv(args.tv_trades)
        print(f"  TV Trades:  {len(tv)}")
        print(f"  Py Trades:  {m.total_trades}")
        diff = abs(len(tv) - m.total_trades)
        print(f"  Difference: {diff} trades")

        if "Profit" in tv.columns:
            tv_pnl = tv["Profit"].sum()
            print(f"  TV P&L:     ${tv_pnl:+,.2f}")
            print(f"  Py P&L:     ${m.net_pnl:+,.2f}")
            print(f"  P&L Diff:   ${abs(tv_pnl - m.net_pnl):,.2f}")

    print()


if __name__ == "__main__":
    main()
