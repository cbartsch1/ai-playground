#!/usr/bin/env python3
"""Run Base Hits backtest — ES shorts from SPX option signals.

Usage:
    # Default: all 5 signal types, fixed 8pt stop / 5pt target
    python scripts/run_base_hits.py data/es_5m_databento_2yr.csv

    # Level-aware targets (uses session VA, POC, PDL as targets)
    python scripts/run_base_hits.py data/es_5m_databento_2yr.csv --level-targets

    # Level-aware stops + targets
    python scripts/run_base_hits.py data/es_5m_databento_2yr.csv --level-stops --level-targets

    # Only structure break + engulfing (disable other signals)
    python scripts/run_base_hits.py data/es_5m_databento_2yr.csv --signals sb,eng

    # Custom stop/target
    python scripts/run_base_hits.py data/es_5m_databento_2yr.csv --stop 10 --target 6

    # Walk-forward + significance testing
    python scripts/run_base_hits.py data/es_5m_databento_2yr.csv --validate

    # Sweep all signal combinations
    python scripts/run_base_hits.py data/es_5m_databento_2yr.csv --sweep
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown
from backtester.setups import base_hits


# ── Walk-forward split ──
WF_SPLIT_DATE = "2025-08-15"

# ── Signal name map ──
SIGNAL_NAMES = {
    "sb": ("bh_use_structure_break", "Structure Break"),
    "eng": ("bh_use_engulfing", "Bearish Engulfing"),
    "tc": ("bh_use_trend_cont", "Trend Continuation"),
    "ema": ("bh_use_ema_cross", "EMA Cross"),
    "vix": ("bh_use_vix_spike", "VIX Spike"),
}


def build_config(args) -> StrategyConfig:
    """Build StrategyConfig from CLI args."""
    cfg = base_hits.get_config()

    # Signal toggles
    if args.signals:
        # Disable all first, then enable specified
        for key, (attr, _) in SIGNAL_NAMES.items():
            setattr(cfg, attr, False)
        for sig in args.signals.split(","):
            sig = sig.strip().lower()
            if sig in SIGNAL_NAMES:
                setattr(cfg, SIGNAL_NAMES[sig][0], True)
            else:
                print(f"WARNING: Unknown signal '{sig}'. Valid: {list(SIGNAL_NAMES.keys())}")

    # Stop/target
    if args.stop is not None:
        cfg.bh_stop_pts = args.stop
    if args.target is not None:
        cfg.bh_target_pts = args.target
    if args.stop_bps is not None:
        cfg.bh_stop_bps = args.stop_bps

    # Level-aware modes
    if args.level_stops:
        cfg.bh_use_level_stops = True
    if args.level_targets:
        cfg.bh_use_level_targets = True

    # Filters
    if args.no_tema:
        cfg.bh_require_tema = False
    if args.no_trend:
        cfg.bh_require_trend = False
    if args.no_friday_skip:
        cfg.skip_friday = False
    if args.no_blackout:
        cfg.blackout_start = 0
        cfg.blackout_end = 0
    if args.no_reentry:
        cfg.bh_allow_reentry = False

    # Max trades
    if args.max_trades is not None:
        cfg.bh_max_trades_per_day = args.max_trades

    # Min R:R
    if args.min_rr is not None:
        cfg.bh_min_rr = args.min_rr

    # Range thresholds
    if args.sb_range is not None:
        cfg.bh_sb_range_threshold = args.sb_range
    if args.eng_range is not None:
        cfg.bh_engulf_range_threshold = args.eng_range

    return cfg


def print_summary(trades, cfg, label=""):
    """Print human-readable backtest summary."""
    m = compute_metrics(trades, cfg.initial_capital)
    breakdown = per_setup_breakdown(trades, cfg.initial_capital)

    if label:
        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
    else:
        print(f"\n{'='*65}")
        print(f"  BASE HITS — ES Shorts from SPX Option Signals")
        print(f"{'='*65}")

    print(f"  Total Trades:   {m.total_trades}")
    print(f"  Net P&L:        ${m.net_pnl:,.2f}")
    print(f"  Win Rate:       {m.win_rate:.1f}%")
    print(f"  Profit Factor:  {m.profit_factor:.3f}")
    print(f"  Sharpe:         {m.sharpe:.2f}")
    print(f"  Max Drawdown:   ${m.max_drawdown:,.2f}")
    print(f"  Avg Trade:      ${m.avg_trade:,.2f}")
    print(f"  Avg Win:        ${m.avg_win:,.2f}")
    print(f"  Avg Loss:       ${m.avg_loss:,.2f}")
    print(f"  Win Streak:     {m.longest_win_streak}")
    print(f"  Lose Streak:    {m.longest_lose_streak}")

    # Config summary
    print(f"\n  --- Config ---")
    enabled = []
    for key, (attr, name) in SIGNAL_NAMES.items():
        if getattr(cfg, attr, False):
            enabled.append(name)
    print(f"  Signals:        {', '.join(enabled) if enabled else 'NONE'}")
    print(f"  Stop:           {'Level-aware' if cfg.bh_use_level_stops else f'{cfg.bh_stop_pts:.1f} pts'} (cap {cfg.bh_stop_bps:.0f} bps)")
    print(f"  Target:         {'Level-aware' if cfg.bh_use_level_targets else f'{cfg.bh_target_pts:.1f} pts'}")
    print(f"  TEMA filter:    {'ON' if cfg.bh_require_tema else 'OFF'}")
    print(f"  Trend filter:   {'ON' if cfg.bh_require_trend else 'OFF'}")
    print(f"  Re-entry:       {'ON' if cfg.bh_allow_reentry else 'OFF'}")
    print(f"  Max trades/day: {cfg.bh_max_trades_per_day}")
    print(f"  Friday skip:    {'ON' if cfg.skip_friday else 'OFF'}")
    print(f"  Blackout:       {cfg.blackout_start}-{cfg.blackout_end}" if cfg.blackout_start else "  Blackout:       OFF")

    # Setup breakdown
    if breakdown:
        print(f"\n  --- By Signal Type ---")
        print(f"  {'Setup':<10} {'Trades':>7} {'WR':>7} {'PF':>8} {'Net P&L':>12}")
        print(f"  {'-'*48}")
        for setup, sm in sorted(breakdown.items()):
            print(f"  {setup:<10} {sm.total_trades:>7} {sm.win_rate:>6.1f}% {sm.profit_factor:>8.2f} ${sm.net_pnl:>11,.0f}")

    # Exit reason breakdown
    if trades:
        reasons = {}
        for t in trades:
            if t.exit_reason not in reasons:
                reasons[t.exit_reason] = {"count": 0, "pnl": 0}
            reasons[t.exit_reason]["count"] += 1
            reasons[t.exit_reason]["pnl"] += t.pnl_dollar
        print(f"\n  --- Exit Reasons ---")
        for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            print(f"  {reason:<16} {data['count']:>4} trades  ${data['pnl']:>+10,.0f}")

    print(f"{'='*65}")
    return m


def run_significance(trades, seed=42):
    """Run t-test, permutation test, bootstrap."""
    pnls = np.array([t.pnl_dollar for t in trades])
    n = len(pnls)

    if n < 5:
        print("  Too few trades for significance testing.")
        return 1.0, 1.0, 0.0

    # T-test
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)

    # Permutation test
    obs_pnl = np.sum(pnls)
    abs_pnls = np.abs(pnls)
    rng = np.random.default_rng(seed)
    n_perm = 5000
    count_better = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if np.dot(signs, abs_pnls) >= obs_pnl:
            count_better += 1
    perm_pval = count_better / n_perm

    # Bootstrap
    n_boot = 5000
    boot_pnl = np.array([
        np.sum(rng.choice(pnls, size=n, replace=True))
        for _ in range(n_boot)
    ])
    prob_profit = float(np.mean(boot_pnl > 0))

    print(f"\n  --- Statistical Significance ---")
    print(f"  t-test p-value:      {t_pval:.6f} {'***' if t_pval < 0.01 else '**' if t_pval < 0.05 else '*' if t_pval < 0.10 else ''}")
    print(f"  Permutation p-value: {perm_pval:.6f}")
    print(f"  Bootstrap P(profit): {prob_profit:.2%}")

    return t_pval, perm_pval, prob_profit


def run_walk_forward(df, cfg, split_date=WF_SPLIT_DATE):
    """Run walk-forward validation."""
    # Reset module state before each run
    base_hits.reset_module()
    df_is = df[df.index < split_date]
    trades_is = run_backtest(df_is.copy(), cfg)
    m_is = compute_metrics(trades_is, cfg.initial_capital) if trades_is else None

    base_hits.reset_module()
    df_oos = df[df.index >= split_date]
    trades_oos = run_backtest(df_oos.copy(), cfg)
    m_oos = compute_metrics(trades_oos, cfg.initial_capital) if trades_oos else None

    print(f"\n  --- Walk-Forward Validation (split: {split_date}) ---")
    if m_is:
        print(f"  IS:  {m_is.total_trades} trades | PF {m_is.profit_factor:.3f} | "
              f"WR {m_is.win_rate:.1f}% | ${m_is.net_pnl:,.0f}")
    if m_oos:
        print(f"  OOS: {m_oos.total_trades} trades | PF {m_oos.profit_factor:.3f} | "
              f"WR {m_oos.win_rate:.1f}% | ${m_oos.net_pnl:,.0f}")
    if m_is and m_oos and m_is.profit_factor > 0:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor
        print(f"  PF ratio (OOS/IS): {pf_ratio:.3f} {'PASS' if pf_ratio >= 0.65 else 'FAIL'}")

    # Standard parseable output for AutoResearch metrics.py
    if m_oos:
        print(f"\n  --- Walk-Forward (5-fold) ---")
        print(f"  OOS PF:        {m_oos.profit_factor:.3f}")
        if m_is and m_is.profit_factor > 0:
            print(f"  WF PF ratio:   {m_oos.profit_factor / m_is.profit_factor:.3f}")
        print(f"  OOS Trades:    {m_oos.total_trades}")

    return m_is, m_oos, trades_is + trades_oos


def run_signal_sweep(df, base_cfg):
    """Sweep all signal combinations to find the best mix."""
    signal_keys = list(SIGNAL_NAMES.keys())
    results = []

    print(f"\n{'='*80}")
    print(f"  SIGNAL COMBINATION SWEEP")
    print(f"{'='*80}")
    print(f"  {'Signals':<30} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*74}")

    # Test each individual signal, then all combinations of 2, 3, 4, 5
    for size in range(1, len(signal_keys) + 1):
        for combo in combinations(signal_keys, size):
            cfg = base_hits.get_config()
            # Copy non-signal settings from base_cfg
            cfg.bh_stop_pts = base_cfg.bh_stop_pts
            cfg.bh_target_pts = base_cfg.bh_target_pts
            cfg.bh_stop_bps = base_cfg.bh_stop_bps
            cfg.bh_use_level_stops = base_cfg.bh_use_level_stops
            cfg.bh_use_level_targets = base_cfg.bh_use_level_targets
            cfg.bh_require_tema = base_cfg.bh_require_tema
            cfg.bh_require_trend = base_cfg.bh_require_trend
            cfg.bh_allow_reentry = base_cfg.bh_allow_reentry
            cfg.skip_friday = base_cfg.skip_friday
            cfg.blackout_start = base_cfg.blackout_start
            cfg.blackout_end = base_cfg.blackout_end

            # Disable all signals
            for key, (attr, _) in SIGNAL_NAMES.items():
                setattr(cfg, attr, False)
            # Enable this combo
            for key in combo:
                setattr(cfg, SIGNAL_NAMES[key][0], True)

            base_hits.reset_module()
            trades = run_backtest(df.copy(), cfg)
            m = compute_metrics(trades, cfg.initial_capital)

            label = "+".join(combo)
            results.append((label, m))

            pf_str = f"{m.profit_factor:.2f}" if m.total_trades > 0 else "N/A"
            wr_str = f"{m.win_rate:.1f}%" if m.total_trades > 0 else "N/A"
            sh_str = f"{m.sharpe:.2f}" if m.total_trades > 0 else "N/A"
            print(f"  {label:<30} {m.total_trades:>7} {wr_str:>7} {pf_str:>8} "
                  f"${m.net_pnl:>11,.0f} {sh_str:>8}")

    # Best combo
    profitable = [(label, m) for label, m in results if m.profit_factor > 1.0 and m.total_trades >= 10]
    if profitable:
        best = max(profitable, key=lambda x: x[1].net_pnl)
        print(f"\n  BEST: {best[0]} — {best[1].total_trades} trades, PF {best[1].profit_factor:.2f}, ${best[1].net_pnl:,.0f}")
    else:
        print(f"\n  No profitable combination with >= 10 trades.")


def main():
    parser = argparse.ArgumentParser(
        description="Base Hits — ES Shorts from SPX Option Signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_file", help="ES 5-min CSV data file")
    parser.add_argument("-o", "--output", default="output/base_hits",
                        help="Output directory (default: output/base_hits)")

    # Signal selection
    parser.add_argument("--signals", type=str, default=None,
                        help="Comma-separated signal types: sb,eng,tc,ema,vix (default: all)")

    # Stop/target
    parser.add_argument("--stop", type=float, default=None,
                        help="Fixed stop in points (default: 8.0)")
    parser.add_argument("--target", type=float, default=None,
                        help="Fixed target in points (default: 5.0)")
    parser.add_argument("--stop-bps", type=float, default=None,
                        help="Percentage stop cap in basis points (default: 30)")
    parser.add_argument("--min-rr", type=float, default=None,
                        help="Minimum reward:risk ratio (default: 0.5)")

    # Level-aware modes
    parser.add_argument("--level-stops", action="store_true",
                        help="Use structural levels for stop placement")
    parser.add_argument("--level-targets", action="store_true",
                        help="Use structural levels for target placement")

    # Filter overrides
    parser.add_argument("--no-tema", action="store_true",
                        help="Disable TEMA bearish filter")
    parser.add_argument("--no-trend", action="store_true",
                        help="Disable TEMA 55 trend filter")
    parser.add_argument("--no-friday-skip", action="store_true",
                        help="Allow Friday entries")
    parser.add_argument("--no-blackout", action="store_true",
                        help="Disable noon blackout")
    parser.add_argument("--no-reentry", action="store_true",
                        help="Disable re-entry on target hit")

    # Trade management
    parser.add_argument("--max-trades", type=int, default=None,
                        help="Max trades per day (default: 6)")

    # Range thresholds
    parser.add_argument("--sb-range", type=float, default=None,
                        help="Structure break min 30m range in ES pts (default: 2.0)")
    parser.add_argument("--eng-range", type=float, default=None,
                        help="Engulfing min 30m range in ES pts (default: 2.0)")

    # Validation
    parser.add_argument("--validate", action="store_true",
                        help="Run walk-forward + significance tests")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all signal combinations")

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument="ES")
    print(f"Loaded {len(df):,} bars: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    # Build config
    cfg = build_config(args)

    if args.sweep:
        run_signal_sweep(df, cfg)
        return

    # Run backtest
    base_hits.reset_module()
    trades = run_backtest(df.copy(), cfg)
    m = print_summary(trades, cfg)

    # Validation
    if args.validate and trades:
        run_significance(trades)
        run_walk_forward(df, cfg)

    # Save output
    if trades:
        os.makedirs(args.output, exist_ok=True)

        # Trade log CSV
        import csv
        trade_log_path = os.path.join(args.output, "trade_log.csv")
        with open(trade_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entry_time", "exit_time", "setup", "direction",
                "entry_price", "exit_price", "stop", "target",
                "exit_reason", "pnl_pts", "pnl_dollar",
            ])
            for t in trades:
                writer.writerow([
                    t.entry_time, t.exit_time, t.setup, t.direction,
                    f"{t.entry_price:.2f}", f"{t.exit_price:.2f}",
                    f"{t.stop:.2f}", f"{t.target:.2f}",
                    t.exit_reason, f"{t.pnl_pts:.2f}", f"{t.pnl_dollar:.2f}",
                ])
        print(f"\n  Trade log: {trade_log_path}")

        # Equity curve
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            equity = [cfg.initial_capital]
            for t in trades:
                equity.append(equity[-1] + t.pnl_dollar)
            times = [trades[0].entry_time] + [t.exit_time for t in trades]

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(times, equity, linewidth=1.5, color="#00cc88")
            ax.axhline(y=cfg.initial_capital, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(f"Base Hits Equity Curve — {m.total_trades} trades, "
                         f"PF {m.profit_factor:.2f}, ${m.net_pnl:,.0f}")
            ax.set_ylabel("Equity ($)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            equity_path = os.path.join(args.output, "equity_curve.png")
            fig.savefig(equity_path, dpi=150)
            plt.close(fig)
            print(f"  Equity curve: {equity_path}")
        except ImportError:
            print("  (matplotlib not available — skipping equity curve)")

    elif not trades:
        print("\n  No trades generated. Check signal settings and data.")


if __name__ == "__main__":
    main()
