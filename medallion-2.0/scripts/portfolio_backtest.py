#!/usr/bin/env python3
"""Medallion Portfolio Backtest — All 5 strategies, regime-gated.

Runs every registered strategy with and without HMM regime filtering,
then combines into a portfolio-level equity curve and summary.

Strategies:
  1. AMT-TEMA v8      (ES 5m, short)  — blocked: Strong Bull, Bull Run
  2. LVL Rejection v13 (ES 5m, short) — unfiltered (non-stationary regime-PnL)
  3. Home Run          (SPX 0DTE, short) — allowed: Crash, Bear, Distribution
  4. Bear Breakdown    (SPX 0DTE, short) — allowed: Crash, Bear, Distribution
  5. Bull Credit Spread (SPX credit, long) — blocked: Crash, Distribution

Usage:
    cd ~/projects/ai-playground/medallion-2.0
    source .venv/bin/activate
    python scripts/portfolio_backtest.py
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Project roots
MEDALLION_ROOT = Path(__file__).parent.parent
AI_PLAYGROUND_ROOT = MEDALLION_ROOT.parent
SPX_OPTIONS_ROOT = AI_PLAYGROUND_ROOT.parent / "spx-options"


def _swap_path(project_root):
    """Set sys.path so only MEDALLION + the given project root are available."""
    cleaned = [p for p in sys.path if p not in (
        str(AI_PLAYGROUND_ROOT), str(SPX_OPTIONS_ROOT)
    )]
    sys.path[:] = cleaned

    if str(MEDALLION_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDALLION_ROOT))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    mods_to_remove = [k for k in sys.modules if k.startswith("backtester")]
    for k in mods_to_remove:
        del sys.modules[k]


# ═══════════════════════════════════════════════════════════════
#  REGIME MODEL
# ═══════════════════════════════════════════════════════════════

def load_regime_filter():
    """Load saved HMM model, download SPY hourly data, create RegimeFilter."""
    _swap_path(MEDALLION_ROOT)

    from models.hmm_regime import RegimeDetector
    from models.regime_api import RegimeFilter
    from data.data_loader import download_ohlcv, compute_hmm_features

    print("=" * 70)
    print("  MEDALLION PORTFOLIO BACKTEST")
    print("  Loading 7-state HMM regime model...")
    print("=" * 70)

    detector = RegimeDetector.load_latest(n_regimes=7)
    if detector is None:
        print("ERROR: No saved 7-state model found.")
        sys.exit(1)

    spy_hourly = download_ohlcv("SPY", "1h", "730d")
    hmm_features = compute_hmm_features(spy_hourly)
    full_df = spy_hourly.join(hmm_features)
    rf = RegimeFilter(detector, full_df)

    current = detector.get_current_regime(full_df)
    print(f"  Current regime: {current['label']} ({current['confidence']:.1%})")

    valid = rf.predictions.dropna(subset=["regime_label"])
    print(f"  Model range: {valid.index[0].date()} to {valid.index[-1].date()}")

    return rf


# ═══════════════════════════════════════════════════════════════
#  STRATEGY RUNNERS
# ═══════════════════════════════════════════════════════════════

def run_amt_tema(regime_filter=None, blocked_regimes=None):
    """AMT-TEMA v8 (ES 5m, short-only)."""
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.config import StrategyConfig
    from backtester.data_loader import load_tos_csv
    from backtester.engine import run_backtest
    from backtester.metrics import compute_metrics

    data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento_2yr.csv"
    df = load_tos_csv(str(data_path), instrument="ES")

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


def run_lvl_v13():
    """LVL Rejection v13 (ES 5m, short-only, unfiltered)."""
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.config import StrategyConfig
    from backtester.data_loader import load_tos_csv
    from backtester.stagger_engine import run_backtest_stagger

    data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento_2yr.csv"
    df = load_tos_csv(str(data_path), instrument="ES")

    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject_long = False
    # IB Rejection (wide days)
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    # Level Rejection — ONH only
    cfg.use_level_reject = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = True
    cfg.lvl_ma_filter = "tema"
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 7.0
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_min_rr = 0.5
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3

    all_trades = run_backtest_stagger(df, cfg, n_contracts=3, uniform_skip=2)
    # Extract LVL-only trades (baseline IB/REJ already counted by AMT-TEMA v8)
    lvl_trades = [t for t in all_trades if t.setup.startswith("LVL")]
    return lvl_trades


def run_home_run(regime_filter=None):
    """Home Run (SPX 0DTE puts, short-only)."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.structure_break import HomeRun
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions
    from backtester.metrics import compute_metrics

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = HomeRun()
    trades = strategy.run_backtest(df, vix_data=vix, regime_filter=regime_filter)
    report = compute_metrics(trades, initial_capital=25_000) if trades else None
    return trades, report


def run_bear_breakdown(regime_filter=None):
    """Bear Breakdown (SPX 0DTE puts, short-only)."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.bear_breakdown import BearBreakdown
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions
    from backtester.metrics import compute_metrics

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = BearBreakdown()
    trades = strategy.run_backtest(df, vix_data=vix, regime_filter=regime_filter)
    report = compute_metrics(trades, initial_capital=25_000) if trades else None
    return trades, report


def run_bull_credit(regime_filter=None):
    """Bull Credit Spread (SPX put credit spread, bullish)."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.bull_credit_spread import BullCreditSpread
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions
    from backtester.metrics import compute_metrics

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = BullCreditSpread()
    trades = strategy.run_backtest(df, vix_data=vix, regime_filter=regime_filter)
    report = compute_metrics(trades, initial_capital=25_000) if trades else None
    return trades, report


# ═══════════════════════════════════════════════════════════════
#  PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════════

def normalize_trade(t, strategy_name):
    """Extract (exit_time, pnl) from any trade type.

    ES trades (backtester.position.Trade): .entry_time, .exit_time, .pnl_dollar
    SPX trades (backtester.metrics.TradeRecord): .entry_time, .exit_time, .net_pnl
    """
    exit_time = getattr(t, 'exit_time', None) or getattr(t, 'time_exit', None)
    entry_time = getattr(t, 'entry_time', None) or getattr(t, 'time_enter', None)
    # SPX uses net_pnl, ES uses pnl_dollar
    pnl = getattr(t, 'pnl_dollar', None)
    if pnl is None or pnl == 0:
        pnl = getattr(t, 'net_pnl', 0)
    return {
        "strategy": strategy_name,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl": pnl,
    }


def quick_stats(trades_normalized, label):
    """Compute stats from list of {strategy, exit_time, pnl} dicts."""
    if not trades_normalized:
        return None

    pnls = [t["pnl"] for t in trades_normalized]
    n = len(pnls)
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_trade = total_pnl / n

    # Equity curve & drawdown
    sorted_trades = sorted(trades_normalized, key=lambda t: t["exit_time"])
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    monthly_pnl = defaultdict(float)

    for t in sorted_trades:
        equity += t["pnl"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
        month_key = t["exit_time"].strftime("%Y-%m") if hasattr(t["exit_time"], "strftime") else "unknown"
        monthly_pnl[month_key] += t["pnl"]

    winning_months = sum(1 for v in monthly_pnl.values() if v > 0)
    total_months = len(monthly_pnl)

    # Sharpe from daily P&L
    daily_pnl = defaultdict(float)
    for t in sorted_trades:
        day = t["exit_time"].strftime("%Y-%m-%d") if hasattr(t["exit_time"], "strftime") else "unknown"
        daily_pnl[day] += t["pnl"]
    daily_arr = np.array(list(daily_pnl.values()))
    sharpe = 0.0
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = (daily_arr.mean() / daily_arr.std()) * np.sqrt(252)

    return {
        "label": label,
        "trades": n,
        "pnl": total_pnl,
        "win_rate": wr,
        "profit_factor": pf,
        "avg_trade": avg_trade,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "winning_months": winning_months,
        "total_months": total_months,
        "monthly_pnl": dict(monthly_pnl),
    }


def print_strategy_table(results):
    """Print the per-strategy comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  STRATEGY-LEVEL RESULTS (regime-gated)")
    print(f"{'=' * 90}")

    header = (f"  {'Strategy':<25s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
              f"{'Net P&L':>12s} {'Max DD':>10s} {'Sharpe':>7s} {'Avg':>10s}")
    print(header)
    print(f"  {'-' * 87}")

    for r in results:
        if r is None:
            continue
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        print(f"  {r['label']:<25s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
              f"{pf_str:>7s} ${r['pnl']:>10,.0f} "
              f"${r['max_dd']:>8,.0f} {r['sharpe']:>6.2f} "
              f"${r['avg_trade']:>8,.0f}")


def print_portfolio_summary(combined_stats, per_strategy):
    """Print portfolio-level summary."""
    print(f"\n{'=' * 90}")
    print(f"  COMBINED PORTFOLIO")
    print(f"{'=' * 90}")

    s = combined_stats
    pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] < 100 else "inf"
    print(f"  Total trades:     {s['trades']:,d}")
    print(f"  Net P&L:          ${s['pnl']:+,.0f}")
    print(f"  Profit factor:    {pf_str}")
    print(f"  Win rate:         {s['win_rate']:.1f}%")
    print(f"  Avg trade:        ${s['avg_trade']:+,.0f}")
    print(f"  Max drawdown:     ${s['max_dd']:,.0f}")
    print(f"  Sharpe ratio:     {s['sharpe']:.2f}")
    print(f"  Winning months:   {s['winning_months']}/{s['total_months']} "
          f"({s['winning_months']/s['total_months']*100:.0f}%)" if s['total_months'] > 0 else "")

    # Data range note
    es_strats = [r for r in per_strategy if r and "(ES)" in r["label"]]
    spx_strats = [r for r in per_strategy if r and "(SPX)" in r["label"]]
    if es_strats and spx_strats:
        print(f"\n  Note: ES strategies cover 2yr, SPX strategies cover 5yr.")
        print(f"  Portfolio P&L is the sum across all strategies and all available data.")

    # Monthly P&L timeline (combined)
    print(f"\n  {'Month':<10s} {'Trades':>7s} {'P&L':>12s} {'Cum P&L':>12s}")
    print(f"  {'-' * 45}")

    monthly = s["monthly_pnl"]
    cum = 0.0
    for month in sorted(monthly.keys()):
        # Count trades in this month
        cum += monthly[month]
        bar_len = min(int(abs(monthly[month]) / 2000), 25)
        bar = "+" * bar_len if monthly[month] > 0 else "-" * bar_len
        print(f"  {month:<10s} {'':>7s} ${monthly[month]:>10,.0f} ${cum:>10,.0f}  {bar}")


def save_results(per_strategy, combined, regime_label):
    """Save portfolio results to JSON."""
    out_path = MEDALLION_ROOT / "data" / "processed" / "portfolio_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    data = {
        "generated_at": datetime.now().isoformat(),
        "current_regime": regime_label,
        "strategies": _clean([r for r in per_strategy if r]),
        "combined": _clean(combined),
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n  Results saved to {out_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    rf = load_regime_filter()

    all_trades = []         # combined portfolio trades
    per_strategy = []       # per-strategy stats

    # ── 1. AMT-TEMA v8 (ES, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  1/5  AMT-TEMA v8 (ES 5m, short)")
    print(f"       Blocked: Strong Bull (Trend), Bull Run (Trend)")
    print(f"{'=' * 70}")

    blocked = {"Strong Bull (Trend)", "Bull Run (Trend)"}
    trades, metrics = run_amt_tema(regime_filter=rf, blocked_regimes=blocked)
    normalized = [normalize_trade(t, "AMT-TEMA v8") for t in trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "AMT-TEMA v8 (ES)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 2. LVL Rejection v13 (ES, unfiltered) ──
    print(f"\n{'=' * 70}")
    print(f"  2/5  LVL Rejection v13 (ES 5m, short)")
    print(f"       Unfiltered (regime-PnL non-stationary, WF 0.92 unfiltered)")
    print(f"{'=' * 70}")

    lvl_trades = run_lvl_v13()
    normalized = [normalize_trade(t, "LVL v13") for t in lvl_trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "LVL v13 (ES)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} fills | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 3. Home Run (SPX, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  3/5  Home Run (SPX 0DTE puts, short)")
    print(f"       Allowed: Crash, Bear Trend, Distribution")
    print(f"{'=' * 70}")

    trades, report = run_home_run(regime_filter=rf)
    normalized = [normalize_trade(t, "Home Run") for t in trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Home Run (SPX)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 4. Bear Breakdown (SPX, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  4/5  Bear Breakdown (SPX 0DTE puts, short)")
    print(f"       Allowed: Crash, Bear Trend, Distribution")
    print(f"{'=' * 70}")

    trades, report = run_bear_breakdown(regime_filter=rf)
    normalized = [normalize_trade(t, "Bear Breakdown") for t in trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Bear Breakdown (SPX)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 5. Bull Credit Spread (SPX, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  5/5  Bull Credit Spread (SPX put credit, long)")
    print(f"       Blocked: Crash (Panic), Distribution")
    print(f"{'=' * 70}")

    trades, report = run_bull_credit(regime_filter=rf)
    normalized = [normalize_trade(t, "Bull Credit") for t in trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Bull Credit (SPX)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── Portfolio Summary ──
    combined = quick_stats(all_trades, "PORTFOLIO")

    print_strategy_table(per_strategy)
    print_portfolio_summary(combined, per_strategy)

    # Get current regime label for JSON
    _swap_path(MEDALLION_ROOT)
    from models.hmm_regime import RegimeDetector
    from data.data_loader import download_ohlcv, compute_hmm_features
    detector = RegimeDetector.load_latest(n_regimes=7)
    spy_hourly = download_ohlcv("SPY", "1h", "730d")
    hmm_features = compute_hmm_features(spy_hourly)
    full_df = spy_hourly.join(hmm_features)
    current = detector.get_current_regime(full_df)

    save_results(per_strategy, combined, current["label"])

    print(f"\n{'=' * 90}")
    print(f"  MEDALLION PORTFOLIO BACKTEST COMPLETE")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
