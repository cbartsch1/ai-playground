#!/usr/bin/env python3
"""CLI entry point for AMT-TEMA backtester.

Usage:
    # ES backtest (default)
    python scripts/run_backtest.py data/es_5m.csv

    # SPX backtest
    python scripts/run_backtest.py data/spx_5m.csv --instrument SPX

    # Custom output directory
    python scripts/run_backtest.py data/es_5m.csv -o output/es_test1

    # With 80% Rule enabled
    python scripts/run_backtest.py data/es_5m.csv --eighty
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig, ES_DEFAULTS, SPX_DEFAULTS
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="AMT-TEMA Backtester")
    parser.add_argument("csv_file", help="Path to thinkorSwim CSV export")
    parser.add_argument("--instrument", "-i", default="ES", choices=["ES", "SPX"],
                        help="Instrument preset (default: ES)")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--eighty", action="store_true",
                        help="Enable 80%% Rule (default OFF)")
    parser.add_argument("--no-vol-filter", action="store_true",
                        help="Disable volatility filter")
    parser.add_argument("--optimistic", action="store_true",
                        help="Use optimistic fills (target wins on ambiguous bars)")
    parser.add_argument("--no-va", action="store_true",
                        help="Disable VA Fade setup")
    parser.add_argument("--blackout", type=str, default=None,
                        help="Blackout window HHMM-HHMM (e.g., 1200-1300)")
    parser.add_argument("--skip-friday", action="store_true",
                        help="Skip all entries on Fridays")
    parser.add_argument("--ib-max-stop", type=float, default=None,
                        help="Override IB max stop pts (default 20)")
    parser.add_argument("--trade-end", type=int, default=None,
                        help="Override trade end time HHMM (default 1500)")
    parser.add_argument("--short-only", action="store_true",
                        help="Only take short entries")
    parser.add_argument("--long-only", action="store_true",
                        help="Only take long entries")
    parser.add_argument("--pct-stop", type=float, default=None,
                        help="Use percentage-based stops (basis points, e.g., 30 = 0.3%%)")
    args = parser.parse_args()

    # Select config preset
    if args.instrument == "SPX":
        cfg = SPX_DEFAULTS
    else:
        cfg = ES_DEFAULTS

    # Apply CLI overrides
    if args.eighty:
        cfg.use_eighty = True
    if args.no_vol_filter:
        cfg.use_vol_filter = False
    if args.optimistic:
        cfg.pessimistic_fills = False
    if args.no_va:
        cfg.use_va_fade = False
    if args.blackout:
        parts = args.blackout.split("-")
        cfg.blackout_start = int(parts[0])
        cfg.blackout_end = int(parts[1])
    if args.skip_friday:
        cfg.skip_friday = True
    if args.ib_max_stop is not None:
        cfg.ib_max_stop_pts = args.ib_max_stop
    if args.trade_end is not None:
        cfg.trade_end = args.trade_end
    if args.short_only:
        cfg.direction_filter = "short"
    if args.long_only:
        cfg.direction_filter = "long"
    if args.pct_stop is not None:
        cfg.pct_stop_mode = True
        cfg.pct_stop_bps = args.pct_stop

    # Load data
    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument=args.instrument)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    rth_bars = df["is_rth"].sum()
    sessions = df["new_rth"].sum()
    print(f"RTH bars: {rth_bars}, Trading sessions: {sessions}")

    # Run backtest
    print(f"\nRunning AMT-TEMA v6.1 backtest ({args.instrument})...")
    trades = run_backtest(df, cfg)
    print(f"Completed: {len(trades)} trades\n")

    # Generate report
    generate_report(trades, cfg, output_dir=args.output)


if __name__ == "__main__":
    main()
