#!/usr/bin/env python3
"""Medal-Lion Fund Portfolio Backtest — All 8 strategies, regime-gated.

Runs every registered strategy with and without HMM regime filtering,
then combines into a portfolio-level equity curve and summary.

Strategies:
  1. Undertow          (ES 5m, short)  — blocked: Strong Bull, Bull Run
  2. Sentinel          (ES 5m, short)  — unfiltered (non-stationary regime-PnL)
  3. Nexus             (ES 5m, both)   — regime-gated per profiling results
  4. Catalyst          (SPX 0DTE, short) — allowed: Crash, Bear, Distribution
  5. Cascade           (SPX 0DTE, short) — allowed: Crash, Bear, Distribution
  6. Fortress          (SPX credit, long) — blocked: Crash, Distribution
  7. Crossfire         (SPX credit, both) — regime-gated per-side
  8. Vector            (SPX 0DTE, short)  — regime-gated per profiling results

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    source .venv/bin/activate
    python scripts/portfolio_backtest.py
    python scripts/portfolio_backtest.py --biased    # Full-sample regimes (for comparison)
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
    print("  MEDAL-LION FUND PORTFOLIO BACKTEST")
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
    """Undertow (ES 5m, short-only + IB Rejection).

    v8.1 adds IB Rejection setup (wide days only, target ib_low).
    Same regime gating as v8 (block Strong Bull + Bull Run).
    """
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
    # v8.1: IB Rejection (wide days)
    cfg.use_ib_reject = True
    cfg.rej_wide_only = True
    cfg.rej_target = "ib_low"
    cfg.rej_stop_buffer = 8.0
    cfg.max_rej_trades = 8

    trades = run_backtest(df, cfg, regime_filter=regime_filter,
                          regime_blocked=blocked_regimes)
    metrics = compute_metrics(trades, cfg.initial_capital)
    return trades, metrics


def run_lvl_v13():
    """Sentinel (ES 5m, short-only, unfiltered)."""
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
    """Catalyst (SPX 0DTE puts, short-only)."""
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
    """Cascade (SPX 0DTE puts, short-only)."""
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
    """Fortress (SPX put credit spread, bullish)."""
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


def run_ms_os(regime_filter=None):
    """Nexus (ES 5m, both directions). Post-hoc regime filtering."""
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.config import StrategyConfig
    from backtester.data_loader import load_tos_csv
    from backtester.engine import run_backtest
    from backtester.metrics import compute_metrics

    data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento_2yr.csv"
    df = load_tos_csv(str(data_path), instrument="ES")

    cfg = StrategyConfig()
    cfg.direction_filter = "both"

    # All setups OFF except MS and OS
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_fa = False

    # MS Config B
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 4.0
    cfg.ms_min_target_pts = 8.0
    cfg.ms_min_rr = 0.3
    cfg.ms_max_risk = 25.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False
    cfg.ms_level_directions = {
        "MS_ONH": "both",
        "MS_ONL": "both",
        "MS_pVAH": "short",
    }

    # OS Best
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = 5.0
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 3.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_require_ma = False
    cfg.max_os_trades = 1
    cfg.os_min_gap = 3.0
    cfg.os_max_gap = 20.0
    cfg.os_entry_window = 1

    trades = run_backtest(df, cfg)
    metrics = compute_metrics(trades, cfg.initial_capital) if trades else None
    return trades, metrics, regime_filter


def run_ema_cross(regime_filter=None):
    """Crossfire (SPX credit spreads both sides). Post-hoc regime filtering."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.ema_cross import EMACross
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = EMACross()
    trades = strategy.run_backtest(df, vix_data=vix)
    return trades, regime_filter


def run_ema_cross_dir(regime_filter=None):
    """Vector (SPX 0DTE puts, short-only). Post-hoc regime filtering."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.ema_cross_directional import EMACrossDirectionalBest
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    # Load all-sessions data for overnight inventory filter
    df_all_path = SPX_OPTIONS_ROOT / "data" / "spy_1m_all.parquet"
    df_all = None
    if df_all_path.exists():
        df_all = load_spy_data(str(df_all_path))
        df_all = tag_sessions(df_all)

    strategy = EMACrossDirectionalBest()
    trades = strategy.run_backtest(df, vix_data=vix, df_all_sessions=df_all)
    return trades, regime_filter


def _load_regime_profile(filename):
    """Load regime profile JSON and return blocked regimes."""
    profile_path = MEDALLION_ROOT / "data" / "processed" / filename
    if profile_path.exists():
        with open(profile_path) as f:
            data = json.load(f)
        return data
    return None


def _post_hoc_filter(trades, regime_filter, blocked_regimes, pnl_attr="pnl_dollar"):
    """Post-hoc filter trades — remove those in blocked regimes."""
    if not blocked_regimes or regime_filter is None:
        return trades
    kept = []
    for t in trades:
        regime = regime_filter.get_regime_at(t.entry_time)
        label = regime.get("label", "")
        if label not in blocked_regimes:
            kept.append(t)
    return kept


def _post_hoc_filter_per_side(trades, regime_filter, blocked_call, blocked_put):
    """Post-hoc filter EMA Cross trades per-side."""
    if regime_filter is None:
        return trades
    kept = []
    for t in trades:
        regime = regime_filter.get_regime_at(t.entry_time)
        label = regime.get("label", "")
        if t.direction == "short" and label in blocked_call:
            continue
        if t.direction == "long" and label in blocked_put:
            continue
        kept.append(t)
    return kept


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


def save_results(per_strategy, combined, regime_label, regime_mode="biased"):
    """Save portfolio results to JSON.

    Saves to both the generic file (for portfolio_allocation.py) and a
    mode-specific file (for side-by-side comparison).
    """
    out_dir = MEDALLION_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "regime_mode": regime_mode,
        "strategies": _clean([r for r in per_strategy if r]),
        "combined": _clean(combined),
    }

    # Save generic file (always overwritten — used by portfolio_allocation.py)
    generic_path = out_dir / "portfolio_backtest.json"
    with open(generic_path, "w") as f:
        json.dump(data, f, indent=2)

    # Save mode-specific file (for side-by-side comparison)
    mode_suffix = "wf" if regime_mode == "walk-forward" else "biased"
    mode_path = out_dir / f"portfolio_backtest_{mode_suffix}.json"
    with open(mode_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n  Results saved to {generic_path}")
    print(f"  Mode-specific: {mode_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Medal-Lion Fund Portfolio Backtest")
    parser.add_argument("--biased", action="store_true",
                        help="Use full-sample (biased) regime labels instead of walk-forward")
    args = parser.parse_args()

    # Load regime filter — biased (full-sample) or walk-forward
    if args.biased:
        rf = load_regime_filter()
        regime_mode = "biased"
    else:
        # Try walk-forward first, fall back to biased
        wf_path = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"
        if wf_path.exists():
            _swap_path(MEDALLION_ROOT)
            from models.wf_regime_api import WalkForwardRegimeFilter
            rf = WalkForwardRegimeFilter(str(wf_path))
            regime_mode = "walk-forward"
            print("=" * 70)
            print("  MEDAL-LION FUND PORTFOLIO BACKTEST (walk-forward regimes)")
            print(f"  Regime source: {wf_path.name}")
            print("=" * 70)
        else:
            rf = load_regime_filter()
            regime_mode = "biased"
            print("  NOTE: Walk-forward regimes not found, using full-sample (biased)")

    # Load regime profile results for post-hoc filtering
    ms_os_profile = _load_regime_profile("regime_profile_ms_os.json")
    ema_cross_profile = _load_regime_profile("regime_profile_ema_cross.json")

    ms_os_blocked = set(ms_os_profile["blocked_regimes"]) if ms_os_profile else set()
    ema_blocked_call = set(ema_cross_profile.get("blocked_regimes_call_credit", [])) if ema_cross_profile else set()
    ema_blocked_put = set(ema_cross_profile.get("blocked_regimes_put_credit", [])) if ema_cross_profile else set()

    all_trades = []
    per_strategy = []

    # ── 1. Undertow (ES, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  1/8  Undertow (ES 5m, short + IB Rejection)")
    print(f"       Blocked: Strong Bull (Trend), Bull Run (Trend)")
    print(f"{'=' * 70}")

    blocked = {"Strong Bull (Trend)", "Bull Run (Trend)"}
    trades, metrics = run_amt_tema(regime_filter=rf, blocked_regimes=blocked)
    normalized = [normalize_trade(t, "Undertow") for t in trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Undertow (ES)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 2. Sentinel (ES, unfiltered) ──
    print(f"\n{'=' * 70}")
    print(f"  2/8  Sentinel (ES 5m, short)")
    print(f"       Unfiltered (regime-PnL non-stationary)")
    print(f"{'=' * 70}")

    lvl_trades = run_lvl_v13()
    normalized = [normalize_trade(t, "Sentinel") for t in lvl_trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Sentinel (ES)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} fills | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 3. Nexus (ES, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  3/8  Nexus (ES 5m, both)")
    if ms_os_blocked:
        print(f"       Blocked: {ms_os_blocked}")
    else:
        print(f"       No regime profile found — running unfiltered")
    print(f"{'=' * 70}")

    ms_trades, ms_metrics, _ = run_ms_os(regime_filter=rf)
    ms_trades = _post_hoc_filter(ms_trades, rf, ms_os_blocked)
    normalized = [normalize_trade(t, "Nexus") for t in ms_trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Nexus (ES)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 4. Catalyst (SPX, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  4/8  Catalyst (SPX 0DTE puts, short)")
    print(f"       Allowed: Crash, Bear Trend, Distribution")
    print(f"{'=' * 70}")

    trades, report = run_home_run(regime_filter=rf)
    normalized = [normalize_trade(t, "Catalyst") for t in trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Catalyst (SPX)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 5. Cascade (SPX, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  5/8  Cascade (SPX 0DTE puts, short)")
    print(f"       Allowed: Crash, Bear Trend, Distribution")
    print(f"{'=' * 70}")

    trades, report = run_bear_breakdown(regime_filter=rf)
    normalized = [normalize_trade(t, "Cascade") for t in trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Cascade (SPX)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 6. Fortress (SPX, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  6/8  Fortress (SPX put credit, long)")
    print(f"       Blocked: Crash (Panic), Distribution")
    print(f"{'=' * 70}")

    trades, report = run_bull_credit(regime_filter=rf)
    normalized = [normalize_trade(t, "Fortress") for t in trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Fortress (SPX)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 7. Crossfire (SPX, per-side regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  7/8  Crossfire (SPX credit spreads, both)")
    if ema_blocked_call or ema_blocked_put:
        print(f"       Blocked call: {ema_blocked_call or 'none'}")
        print(f"       Blocked put:  {ema_blocked_put or 'none'}")
    else:
        print(f"       No regime profile found — running unfiltered")
    print(f"{'=' * 70}")

    ema_trades, _ = run_ema_cross(regime_filter=rf)
    ema_trades = _post_hoc_filter_per_side(ema_trades, rf, ema_blocked_call, ema_blocked_put)
    normalized = [normalize_trade(t, "Crossfire") for t in ema_trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Crossfire (SPX)")
    per_strategy.append(stats)
    print(f"  {stats['trades']} trades | PF {stats['profit_factor']:.2f} | "
          f"P&L ${stats['pnl']:+,.0f}")

    # ── 8. Vector (SPX, regime-gated) ──
    print(f"\n{'=' * 70}")
    print(f"  8/8  Vector (SPX 0DTE puts, short)")
    print(f"{'=' * 70}")

    ema_dir_trades, _ = run_ema_cross_dir(regime_filter=rf)
    # Use same blocked regimes as call credit side (both are bearish/short)
    ema_dir_trades = _post_hoc_filter(ema_dir_trades, rf, ema_blocked_call, pnl_attr="net_pnl")
    normalized = [normalize_trade(t, "Vector") for t in ema_dir_trades]
    all_trades.extend(normalized)
    stats = quick_stats(normalized, "Vector (SPX)")
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

    save_results(per_strategy, combined, current["label"], regime_mode=regime_mode)

    print(f"\n{'=' * 90}")
    print(f"  MEDAL-LION FUND PORTFOLIO BACKTEST COMPLETE ({regime_mode} regimes, 8 strategies)")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
