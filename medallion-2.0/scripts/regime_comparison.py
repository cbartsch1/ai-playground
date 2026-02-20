#!/usr/bin/env python3
"""Regime Filter Comparison — Run AMT-TEMA v8 and Home Run with/without regime gating.

Loads saved Medallion 2.0 HMM model, creates RegimeFilter, and runs both strategies
with and without regime filtering to measure impact.

Usage:
    cd ~/projects/ai-playground/medallion-2.0
    source .venv/bin/activate
    python scripts/regime_comparison.py

    # AMT-TEMA only
    python scripts/regime_comparison.py --strategy amt-tema

    # Home Run only
    python scripts/regime_comparison.py --strategy home-run

    # Custom blocked regimes for AMT-TEMA
    python scripts/regime_comparison.py --blocked "Strong Bull (Trend),Bull Run (Trend),Recovery"
"""

import argparse
import importlib
import sys
import os
from pathlib import Path

import json
from datetime import datetime

import pandas as pd
import numpy as np

# Project roots
MEDALLION_ROOT = Path(__file__).parent.parent
AI_PLAYGROUND_ROOT = MEDALLION_ROOT.parent
SPX_OPTIONS_ROOT = AI_PLAYGROUND_ROOT.parent / "spx-options"


def _swap_path(project_root):
    """Set sys.path so only MEDALLION + the given project root are available.
    Clears cached 'backtester' modules to avoid cross-project import conflicts."""
    # Remove any previous project roots (except medallion)
    cleaned = [p for p in sys.path if p not in (
        str(AI_PLAYGROUND_ROOT), str(SPX_OPTIONS_ROOT)
    )]
    sys.path[:] = cleaned

    # Ensure medallion is on path (for models/ and config/)
    if str(MEDALLION_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDALLION_ROOT))

    # Add target project
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Clear cached backtester modules
    mods_to_remove = [k for k in sys.modules if k.startswith("backtester")]
    for k in mods_to_remove:
        del sys.modules[k]


def load_regime_filter():
    """Load saved HMM model, download fresh SPY hourly data, create RegimeFilter."""
    # Ensure medallion is on path
    if str(MEDALLION_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDALLION_ROOT))

    from models.hmm_regime import RegimeDetector
    from models.regime_api import RegimeFilter
    from data.data_loader import download_ohlcv, compute_hmm_features

    print("=" * 70)
    print("  LOADING MEDALLION 2.0 REGIME MODEL")
    print("=" * 70)

    # Load saved model
    detector = RegimeDetector.load_latest(n_regimes=7)
    if detector is None:
        print("ERROR: No saved 7-state model found. Run dashboard first to fit.")
        sys.exit(1)

    print(f"  Model: 7-state Gaussian HMM")
    print(f"  Features: {detector.feature_names}")
    print(f"  Labels: {list(detector.regime_labels.values())}")

    # Download SPY hourly data and compute HMM features
    spy_hourly = download_ohlcv("SPY", "1h", "730d")
    hmm_features = compute_hmm_features(spy_hourly)

    # Merge features with OHLCV for prediction
    full_df = spy_hourly.join(hmm_features)

    # Create filter
    rf = RegimeFilter(detector, full_df)

    # Show current state
    current = detector.get_current_regime(full_df)
    print(f"\n  Current regime: {current['label']} ({current['confidence']:.1%} confidence)")
    print(f"  Signal: {current['signal']}")
    print(f"  Streak: {current['streak']} bars")

    # Show regime distribution
    predictions = rf.predictions
    dist = predictions["regime_label"].value_counts(normalize=True).sort_index()
    print(f"\n  Regime Distribution ({len(predictions)} hourly bars):")
    for label, pct in dist.items():
        print(f"    {label:<25s} {pct:>6.1%}")

    # Date range
    valid = predictions.dropna(subset=["regime_label"])
    print(f"\n  Date range: {valid.index[0]} to {valid.index[-1]}")

    return rf, predictions


def run_amt_tema(regime_filter=None, blocked_regimes=None):
    """Run AMT-TEMA v8 backtest on 2yr ES data."""
    _swap_path(AI_PLAYGROUND_ROOT)

    from backtester.config import StrategyConfig
    from backtester.data_loader import load_tos_csv
    from backtester.engine import run_backtest
    from backtester.metrics import compute_metrics

    data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento_2yr.csv"
    if not data_path.exists():
        print(f"  WARNING: 2yr data not found, trying 1yr...")
        data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento.csv"

    df = load_tos_csv(str(data_path), instrument="ES")

    # v8 config: short-only, no VA, skip Friday, noon blackout, 30bps pct stop
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.use_va_fade = False
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0

    trades = run_backtest(df, cfg, regime_filter=regime_filter,
                          regime_blocked=blocked_regimes)
    metrics = compute_metrics(trades, cfg.initial_capital)

    return trades, metrics


def run_home_run(regime_filter=None):
    """Run Home Run backtest on 5yr SPY data."""
    _swap_path(SPX_OPTIONS_ROOT)

    spx_backtester = SPX_OPTIONS_ROOT / "backtester"
    if not spx_backtester.exists():
        print(f"  ERROR: SPX Options project not found at {SPX_OPTIONS_ROOT}")
        return [], None

    from backtester.strategies.structure_break import HomeRun
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions
    from backtester.metrics import compute_metrics

    data_path = SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"

    df = load_spy_data(str(data_path))
    df = tag_sessions(df)

    vix = None
    if vix_path.exists():
        vix = load_vix_data(str(vix_path))

    strategy = HomeRun()
    trades = strategy.run_backtest(df, vix_data=vix, regime_filter=regime_filter)

    report = None
    if trades:
        report = compute_metrics(trades, initial_capital=25_000)

    return trades, report


def print_comparison(label, baseline_trades, baseline_metrics,
                     filtered_trades, filtered_metrics):
    """Print before/after comparison table."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    header = (f"  {'Variant':<25s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
              f"{'Net P&L':>12s} {'Max DD':>10s} {'Sharpe':>7s} {'Avg Trade':>10s}")
    print(header)
    print(f"  {'-' * 67}")

    def fmt_row(name, m):
        if m is None or m.total_trades == 0:
            return f"  {name:<25s} {'N/A':>7s}"
        # Handle both AMT-TEMA Metrics and SPX PerformanceReport field names
        pnl = getattr(m, 'net_pnl', None) or getattr(m, 'total_pnl', 0)
        wr = getattr(m, 'win_rate', 0)
        if wr <= 1.0 and m.total_trades > 1:
            wr *= 100  # SPX uses 0-1, AMT-TEMA uses 0-100
        pf = getattr(m, 'profit_factor', 0)
        dd = getattr(m, 'max_drawdown', 0)
        sharpe = getattr(m, 'sharpe', None) or getattr(m, 'sharpe_ratio', 0)
        avg = getattr(m, 'avg_trade', 0)
        return (f"  {name:<25s} {m.total_trades:>7d} {wr:>6.1f}% "
                f"{pf:>6.2f} ${pnl:>10,.0f} "
                f"${dd:>8,.0f} {sharpe:>6.2f} "
                f"${avg:>8,.0f}")

    print(fmt_row("Baseline (no filter)", baseline_metrics))
    print(fmt_row("+ Regime Filter", filtered_metrics))

    # Delta row
    if (baseline_metrics and filtered_metrics
            and baseline_metrics.total_trades > 0
            and filtered_metrics.total_trades > 0):
        dt = filtered_metrics.total_trades - baseline_metrics.total_trades
        b_pnl = getattr(baseline_metrics, 'net_pnl', None) or getattr(baseline_metrics, 'total_pnl', 0)
        f_pnl = getattr(filtered_metrics, 'net_pnl', None) or getattr(filtered_metrics, 'total_pnl', 0)
        dpnl = f_pnl - b_pnl
        dpf = filtered_metrics.profit_factor - baseline_metrics.profit_factor
        b_wr = getattr(baseline_metrics, 'win_rate', 0)
        f_wr = getattr(filtered_metrics, 'win_rate', 0)
        if b_wr <= 1.0 and baseline_metrics.total_trades > 1:
            b_wr *= 100
            f_wr *= 100
        dwr = f_wr - b_wr
        print(f"  {'-' * 67}")
        print(f"  {'Delta':<25s} {dt:>+7d} {dwr:>+6.1f}% "
              f"{dpf:>+6.2f} ${dpnl:>+10,.0f}")
        pct_removed = (1 - filtered_metrics.total_trades / baseline_metrics.total_trades) * 100
        print(f"\n  Trades removed by regime filter: {pct_removed:.1f}%")

        removed = baseline_metrics.total_trades - filtered_metrics.total_trades
        if removed > 0:
            print(f"  Removed {removed} trades that occurred during unfavorable regimes")


def show_regime_overlap(regime_filter, trades, strategy_name):
    """Show which regimes the strategy's trades fall into."""
    if not trades:
        return

    print(f"\n  {strategy_name} — Trades by Regime:")

    regime_trades = {}
    for t in trades:
        # Get entry time — handle both Trade types
        ts = getattr(t, 'entry_time', None) or getattr(t, 'time_enter', None)
        if ts is None:
            continue

        regime = regime_filter.get_regime_at(ts)
        label = regime.get("label")
        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = "No Data (before model)"
        if label not in regime_trades:
            regime_trades[label] = {"count": 0, "pnl": 0.0, "wins": 0}
        regime_trades[label]["count"] += 1

        pnl = getattr(t, 'net_pnl', None) or getattr(t, 'pnl_dollar', 0)
        regime_trades[label]["pnl"] += pnl
        if pnl > 0:
            regime_trades[label]["wins"] += 1

    print(f"  {'Regime':<25s} {'Trades':>7s} {'WR':>7s} {'P&L':>12s} {'Avg':>10s}")
    print(f"  {'-' * 63}")
    for label in sorted(regime_trades.keys()):
        d = regime_trades[label]
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        avg = d["pnl"] / d["count"] if d["count"] > 0 else 0
        print(f"  {label:<25s} {d['count']:>7d} {wr:>6.1f}% "
              f"${d['pnl']:>10,.0f} ${avg:>8,.0f}")


def run_bear_breakdown(regime_filter=None):
    """Run Bear Breakdown backtest on 5yr SPY data."""
    _swap_path(SPX_OPTIONS_ROOT)

    spx_backtester = SPX_OPTIONS_ROOT / "backtester"
    if not spx_backtester.exists():
        print(f"  ERROR: SPX Options project not found at {SPX_OPTIONS_ROOT}")
        return [], None

    from backtester.strategies.bear_breakdown import BearBreakdown
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions
    from backtester.metrics import compute_metrics

    data_path = SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"

    df = load_spy_data(str(data_path))
    df = tag_sessions(df)

    vix = None
    if vix_path.exists():
        vix = load_vix_data(str(vix_path))

    strategy = BearBreakdown()
    trades = strategy.run_backtest(df, vix_data=vix, regime_filter=regime_filter)

    report = None
    if trades:
        report = compute_metrics(trades, initial_capital=25_000)

    return trades, report


def run_bull_credit_spread(regime_filter=None):
    """Run Bull Credit Spread backtest on 5yr SPY data."""
    _swap_path(SPX_OPTIONS_ROOT)

    spx_backtester = SPX_OPTIONS_ROOT / "backtester"
    if not spx_backtester.exists():
        print(f"  ERROR: SPX Options project not found at {SPX_OPTIONS_ROOT}")
        return [], None

    from backtester.strategies.bull_credit_spread import BullCreditSpread
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions
    from backtester.metrics import compute_metrics

    data_path = SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"

    df = load_spy_data(str(data_path))
    df = tag_sessions(df)

    vix = None
    if vix_path.exists():
        vix = load_vix_data(str(vix_path))

    strategy = BullCreditSpread()
    trades = strategy.run_backtest(df, vix_data=vix, regime_filter=regime_filter)

    report = None
    if trades:
        report = compute_metrics(trades, initial_capital=25_000)

    return trades, report


def _build_regime_breakdown(regime_filter, trades):
    """Build regime breakdown dict for a set of trades."""
    breakdown = {}
    for t in trades:
        ts = getattr(t, 'entry_time', None) or getattr(t, 'time_enter', None)
        if ts is None:
            continue
        regime = regime_filter.get_regime_at(ts)
        label = regime.get("label")
        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = "No Data (before model)"
        if label not in breakdown:
            breakdown[label] = {"count": 0, "pnl": 0.0, "wins": 0}
        breakdown[label]["count"] += 1
        pnl = getattr(t, 'net_pnl', None) or getattr(t, 'pnl_dollar', 0)
        breakdown[label]["pnl"] += pnl
        if pnl > 0:
            breakdown[label]["wins"] += 1
    # Compute WR and avg
    for d in breakdown.values():
        d["win_rate"] = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        d["avg_trade"] = d["pnl"] / d["count"] if d["count"] > 0 else 0
    return breakdown


def _metrics_to_dict(m):
    """Convert either Metrics or PerformanceReport to a standard dict."""
    return {
        "trades": m.total_trades,
        "win_rate": getattr(m, 'win_rate', 0) * (100 if getattr(m, 'win_rate', 0) <= 1 and m.total_trades > 1 else 1),
        "profit_factor": getattr(m, 'profit_factor', 0),
        "net_pnl": getattr(m, 'net_pnl', None) or getattr(m, 'total_pnl', 0),
        "max_drawdown": getattr(m, 'max_drawdown', 0),
        "sharpe": getattr(m, 'sharpe', None) or getattr(m, 'sharpe_ratio', 0),
        "avg_trade": getattr(m, 'avg_trade', 0),
    }


def save_results(results_dict):
    """Save comparison results to JSON for dashboard consumption."""
    import json
    out_path = MEDALLION_ROOT / "data" / "processed" / "regime_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(x) for x in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(results_dict), f, indent=2)

    print(f"\n  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Regime Filter Comparison")
    parser.add_argument("--strategy", default="all",
                        choices=["amt-tema", "home-run", "bear-breakdown", "bull-credit", "both", "all"],
                        help="Which strategy to test (default: all)")
    parser.add_argument("--blocked", default=None,
                        help="Comma-separated regime labels to block for AMT-TEMA")
    args = parser.parse_args()

    blocked_regimes = None
    if args.blocked:
        blocked_regimes = set(args.blocked.split(","))

    # Load regime model
    rf, predictions = load_regime_filter()
    all_results = {"generated_at": datetime.now().isoformat()}

    # ── AMT-TEMA v8 ──
    if args.strategy in ("amt-tema", "both", "all"):
        blocked = blocked_regimes or {"Strong Bull (Trend)", "Bull Run (Trend)"}
        print(f"\n{'=' * 70}")
        print(f"  AMT-TEMA v8 (ES 5m, Short-Only)")
        print(f"  Blocked regimes: {blocked}")
        print(f"{'=' * 70}")

        print("\n  Running baseline (no regime filter)...")
        base_trades, base_metrics = run_amt_tema()
        print(f"  -> {base_metrics.total_trades} trades, "
              f"PF {base_metrics.profit_factor:.3f}, "
              f"P&L ${base_metrics.net_pnl:,.0f}")

        print("\n  Running with regime filter...")
        filt_trades, filt_metrics = run_amt_tema(
            regime_filter=rf, blocked_regimes=blocked_regimes
        )
        print(f"  -> {filt_metrics.total_trades} trades, "
              f"PF {filt_metrics.profit_factor:.3f}, "
              f"P&L ${filt_metrics.net_pnl:,.0f}")

        print_comparison("AMT-TEMA v8 -- Regime Filter Impact",
                         base_trades, base_metrics,
                         filt_trades, filt_metrics)

        show_regime_overlap(rf, base_trades, "AMT-TEMA v8 (all trades)")

        all_results["amt_tema"] = {
            "blocked_regimes": list(blocked),
            "baseline": _metrics_to_dict(base_metrics),
            "filtered": _metrics_to_dict(filt_metrics),
            "regime_breakdown": _build_regime_breakdown(rf, base_trades),
        }

    # ── Home Run ──
    if args.strategy in ("home-run", "both", "all"):
        print(f"\n{'=' * 70}")
        print(f"  HOME RUN (SPY 0DTE Puts, 30m Range >= 20)")
        print(f"  Filter: only trade during bearish regimes (Crash, Bear, Distribution)")
        print(f"{'=' * 70}")

        print("\n  Running baseline (no regime filter)...")
        base_trades, base_report = run_home_run()
        if base_report:
            print(f"  -> {base_report.total_trades} trades, "
                  f"PF {base_report.profit_factor:.2f}, "
                  f"P&L ${base_report.total_pnl:,.0f}")

        print("\n  Running with regime filter...")
        filt_trades, filt_report = run_home_run(regime_filter=rf)
        if filt_report:
            print(f"  -> {filt_report.total_trades} trades, "
                  f"PF {filt_report.profit_factor:.2f}, "
                  f"P&L ${filt_report.total_pnl:,.0f}")

        if base_report and filt_report:
            print_comparison("Home Run -- Regime Filter Impact",
                             base_trades, base_report,
                             filt_trades, filt_report)

            show_regime_overlap(rf, base_trades, "Home Run (all trades)")

            all_results["home_run"] = {
                "allowed_regimes": ["Crash (Panic)", "Bear Trend", "Distribution"],
                "baseline": _metrics_to_dict(base_report),
                "filtered": _metrics_to_dict(filt_report),
                "regime_breakdown": _build_regime_breakdown(rf, base_trades),
            }

    # ── Bear Breakdown ──
    if args.strategy in ("bear-breakdown", "all"):
        print(f"\n{'=' * 70}")
        print(f"  BEAR BREAKDOWN (SPY 0DTE Puts, 30m Bearish Engulfing, Range >= 20)")
        print(f"  Filter: only trade during bearish regimes (Crash, Bear, Distribution)")
        print(f"{'=' * 70}")

        print("\n  Running baseline (no regime filter)...")
        base_trades, base_report = run_bear_breakdown()
        if base_report:
            print(f"  -> {base_report.total_trades} trades, "
                  f"PF {base_report.profit_factor:.2f}, "
                  f"P&L ${base_report.total_pnl:,.0f}")

        print("\n  Running with regime filter...")
        filt_trades, filt_report = run_bear_breakdown(regime_filter=rf)
        if filt_report:
            print(f"  -> {filt_report.total_trades} trades, "
                  f"PF {filt_report.profit_factor:.2f}, "
                  f"P&L ${filt_report.total_pnl:,.0f}")

        if base_report and filt_report:
            print_comparison("Bear Breakdown -- Regime Filter Impact",
                             base_trades, base_report,
                             filt_trades, filt_report)

            show_regime_overlap(rf, base_trades, "Bear Breakdown (all trades)")

            all_results["bear_breakdown"] = {
                "allowed_regimes": ["Crash (Panic)", "Bear Trend", "Distribution"],
                "baseline": _metrics_to_dict(base_report),
                "filtered": _metrics_to_dict(filt_report),
                "regime_breakdown": _build_regime_breakdown(rf, base_trades),
            }

    # ── Bull Credit Spread ──
    if args.strategy in ("bull-credit", "all"):
        print(f"\n{'=' * 70}")
        print(f"  BULL CREDIT SPREAD (SPY Put Credit Spread, 30m Momentum)")
        print(f"  Filter: only trade during bullish regimes (Recovery, Bull Run, Strong Bull)")
        print(f"{'=' * 70}")

        print("\n  Running baseline (no regime filter)...")
        base_trades, base_report = run_bull_credit_spread()
        if base_report:
            print(f"  -> {base_report.total_trades} trades, "
                  f"PF {base_report.profit_factor:.2f}, "
                  f"P&L ${base_report.total_pnl:,.0f}")

        print("\n  Running with regime filter...")
        filt_trades, filt_report = run_bull_credit_spread(regime_filter=rf)
        if filt_report:
            print(f"  -> {filt_report.total_trades} trades, "
                  f"PF {filt_report.profit_factor:.2f}, "
                  f"P&L ${filt_report.total_pnl:,.0f}")

        if base_report and filt_report:
            print_comparison("Bull Credit Spread -- Regime Filter Impact",
                             base_trades, base_report,
                             filt_trades, filt_report)

            show_regime_overlap(rf, base_trades, "Bull Credit Spread (all trades)")

            all_results["bull_credit_spread"] = {
                "allowed_regimes": ["Recovery", "Bull Run (Trend)", "Strong Bull (Trend)"],
                "baseline": _metrics_to_dict(base_report),
                "filtered": _metrics_to_dict(filt_report),
                "regime_breakdown": _build_regime_breakdown(rf, base_trades),
            }

    save_results(all_results)


if __name__ == "__main__":
    main()
