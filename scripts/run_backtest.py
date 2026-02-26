#!/usr/bin/env python3
"""CLI entry point for AMT-TEMA backtester.

Usage:
    # v8 baseline (short-only, pct stop, Friday/noon filters)
    python scripts/run_backtest.py data/es_5m_databento_2yr.csv --short-only --pct-stop 30 --skip-friday --blackout 1200-1300 --no-va -o output/v8

    # v9 (all v9 features enabled)
    python scripts/run_backtest.py data/es_5m_databento_2yr.csv --v9 -o output/v9

    # v8 vs v9 comparison
    python scripts/run_backtest.py data/es_5m_databento_2yr.csv --compare -o output/compare
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


def apply_v8_flags(cfg):
    """Apply v8 baseline config: short-only, pct stop, Friday/noon filters, no VA."""
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False


def apply_v9_flags(cfg):
    """Apply all v9 features on top of v8 baseline."""
    apply_v8_flags(cfg)
    cfg.tp_atr_mult = 3.0
    cfg.max_ib_trades = 3
    cfg.use_trail_stop = True
    cfg.trail_trigger_bps = 15.0
    cfg.trail_dist_bps = 20.0
    cfg.use_tema_exit = True
    cfg.use_tema_cross = True
    cfg.tx_day_type_filter = "narrow"
    cfg.max_tx_trades = 2
    cfg.tx_stop_bps = 30.0
    cfg.tx_tp_atr_mult = 2.0


def run_single(cfg, df, label, output_dir):
    """Run backtest and print summary."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    trades = run_backtest(df, cfg)
    print(f"Completed: {len(trades)} trades")

    if trades:
        generate_report(trades, cfg, output_dir=output_dir)

    return trades


def print_comparison(v8_trades, v9_trades):
    """Print side-by-side comparison of v8 vs v9."""
    from backtester.metrics import compute_metrics

    m8 = compute_metrics(v8_trades)
    m9 = compute_metrics(v9_trades)

    print(f"\n{'='*70}")
    print(f"  AMT-TEMA v8 vs v9 COMPARISON (2-year)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'v8':>15} {'v9':>15} {'Delta':>12}")
    print(f"{'-'*70}")

    rows = [
        ("Trades", m8.total_trades, m9.total_trades),
        ("Net P&L", m8.net_pnl, m9.net_pnl),
        ("Win Rate %", m8.win_rate, m9.win_rate),
        ("Profit Factor", m8.profit_factor, m9.profit_factor),
        ("Max Drawdown", m8.max_drawdown, m9.max_drawdown),
        ("Sharpe", m8.sharpe, m9.sharpe),
        ("Avg Trade $", m8.avg_trade, m9.avg_trade),
        ("Avg Win $", m8.avg_win, m9.avg_win),
        ("Avg Loss $", m8.avg_loss, m9.avg_loss),
    ]

    for label, v8_val, v9_val in rows:
        delta = v9_val - v8_val
        if label in ("Net P&L", "Avg Trade $", "Avg Win $", "Avg Loss $", "Max Drawdown"):
            print(f"{label:<25} {'${:,.0f}'.format(v8_val):>15} {'${:,.0f}'.format(v9_val):>15} {'${:+,.0f}'.format(delta):>12}")
        elif label == "Trades":
            print(f"{label:<25} {v8_val:>15.0f} {v9_val:>15.0f} {delta:>+12.0f}")
        elif label in ("Win Rate %",):
            print(f"{label:<25} {v8_val:>14.1f}% {v9_val:>14.1f}% {delta:>+11.1f}%")
        else:
            print(f"{label:<25} {v8_val:>15.2f} {v9_val:>15.2f} {delta:>+12.2f}")

    # Exit reason breakdown for v9
    print(f"\n  v9 Exit Reasons:")
    reasons = {}
    for t in v9_trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pnl = sum(t.pnl_dollar for t in v9_trades if t.exit_reason == reason)
        print(f"    {reason:<12} {count:>4} trades  ${pnl:>+10,.0f}")

    # Setup breakdown for v9
    print(f"\n  v9 Setup Breakdown:")
    setups = {}
    for t in v9_trades:
        if t.setup not in setups:
            setups[t.setup] = {"count": 0, "pnl": 0, "wins": 0}
        setups[t.setup]["count"] += 1
        setups[t.setup]["pnl"] += t.pnl_dollar
        if t.pnl_dollar > 0:
            setups[t.setup]["wins"] += 1
    for setup, s in sorted(setups.items()):
        wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
        print(f"    {setup:<6} {s['count']:>4} trades  {wr:>5.1f}% WR  ${s['pnl']:>+10,.0f}")


def main():
    parser = argparse.ArgumentParser(description="AMT-TEMA Backtester")
    parser.add_argument("csv_file", help="Path to CSV data file")
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
    parser.add_argument("--v9", action="store_true",
                        help="Enable all v9 features (scaled TP, trail, TEMA exit, TX cross)")
    parser.add_argument("--compare", action="store_true",
                        help="Run v8 vs v9 comparison")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument=args.instrument)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    rth_bars = df["is_rth"].sum()
    sessions = df["new_rth"].sum()
    print(f"RTH bars: {rth_bars}, Trading sessions: {sessions}")

    if args.compare:
        # Run both v8 and v9
        cfg8 = StrategyConfig() if args.instrument == "ES" else SPX_DEFAULTS
        apply_v8_flags(cfg8)
        v8_trades = run_single(cfg8, df.copy(), "AMT-TEMA v8 (baseline)", os.path.join(args.output, "v8"))

        cfg9 = StrategyConfig() if args.instrument == "ES" else SPX_DEFAULTS
        apply_v9_flags(cfg9)
        v9_trades = run_single(cfg9, df.copy(), "AMT-TEMA v9 (new)", os.path.join(args.output, "v9"))

        print_comparison(v8_trades, v9_trades)
        return

    # Single run
    if args.instrument == "SPX":
        cfg = SPX_DEFAULTS
    else:
        cfg = StrategyConfig()

    if args.v9:
        apply_v9_flags(cfg)
    else:
        # Apply individual CLI overrides
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

    # Run backtest
    version = "v9" if args.v9 else "AMT-TEMA"
    print(f"\nRunning {version} backtest ({args.instrument})...")
    trades = run_backtest(df, cfg)
    print(f"Completed: {len(trades)} trades\n")

    # Generate report
    generate_report(trades, cfg, output_dir=args.output)


if __name__ == "__main__":
    main()
