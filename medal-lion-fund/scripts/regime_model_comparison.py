#!/usr/bin/env python3
"""Medal-Lion Fund — OLD vs NEW Regime Model Comparison

Runs the full portfolio backtest twice:
  OLD: 3 features (returns, range, volume_vol), 7-state HMM
  NEW: 5 features (returns, range, volume_vol, close_in_range, close_vs_open smoothed), 5-state HMM

Reports side-by-side strategy-by-strategy comparison with delta P&L.

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    source .venv/bin/activate
    python scripts/regime_model_comparison.py
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
#  MODEL BUILDING
# ═══════════════════════════════════════════════════════════════

def build_old_model():
    """Build the OLD 3-feature, 7-state regime model from scratch."""
    _swap_path(MEDALLION_ROOT)

    from models.hmm_regime import RegimeDetector
    from data.data_loader import download_ohlcv

    print("\n" + "=" * 70)
    print("  FITTING OLD MODEL: 3 features, 7 states")
    print("=" * 70)

    spy_hourly = download_ohlcv("SPY", "1h", "730d")

    # Compute only 3 features (NO close_in_range, NO close_vs_open)
    features_3 = pd.DataFrame(index=spy_hourly.index)
    features_3["returns"] = np.log(spy_hourly["Close"] / spy_hourly["Close"].shift(1))
    features_3["range"] = (spy_hourly["High"] - spy_hourly["Low"]) / spy_hourly["Close"]
    log_vol = np.log(spy_hourly["Volume"].replace(0, np.nan))
    features_3["volume_vol"] = log_vol.rolling(20).std()

    detector_old = RegimeDetector(
        n_regimes=7,
        covariance_type="full",
        n_iter=200,
        n_restarts=10,
        random_state=42,
        features=["returns", "range", "volume_vol"],
    )
    detector_old.fit(features_3, feature_cols=["returns", "range", "volume_vol"])

    # Build the full df for RegimeFilter (needs features + ohlcv index)
    full_df_old = spy_hourly.join(features_3)

    print(f"  States: {list(detector_old.regime_labels.values())}")
    current = detector_old.get_current_regime(features_3)
    print(f"  Current regime: {current['label']} ({current['confidence']:.1%})")

    return detector_old, full_df_old, features_3, spy_hourly


def build_new_model():
    """Build the NEW 5-feature, 5-state regime model from scratch."""
    _swap_path(MEDALLION_ROOT)

    from models.hmm_regime import RegimeDetector
    from data.data_loader import download_ohlcv, compute_hmm_features

    print("\n" + "=" * 70)
    print("  FITTING NEW MODEL: 5 features (smoothed), 5 states")
    print("=" * 70)

    spy_hourly = download_ohlcv("SPY", "1h", "730d")
    features_5 = compute_hmm_features(spy_hourly)

    detector_new = RegimeDetector(
        n_regimes=5,
        covariance_type="full",
        n_iter=200,
        n_restarts=10,
        random_state=42,
        features=["returns", "range", "volume_vol", "close_in_range", "close_vs_open"],
    )
    detector_new.fit(features_5, feature_cols=["returns", "range", "volume_vol", "close_in_range", "close_vs_open"])

    full_df_new = spy_hourly.join(features_5)

    print(f"  States: {list(detector_new.regime_labels.values())}")
    current = detector_new.get_current_regime(features_5)
    print(f"  Current regime: {current['label']} ({current['confidence']:.1%})")

    return detector_new, full_df_new, features_5, spy_hourly


def create_regime_filter(detector, full_df):
    """Create a RegimeFilter manually to avoid settings.py import issues."""
    _swap_path(MEDALLION_ROOT)
    from models.regime_api import RegimeFilter
    rf = RegimeFilter(detector, full_df)
    return rf


# ═══════════════════════════════════════════════════════════════
#  REGIME DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_regime_distribution(detector, features, label):
    """Analyze the distribution of regimes (days in each state)."""
    predictions = detector.predict(features)
    valid = predictions.dropna(subset=["regime_label"])

    # Group by trading day
    valid_copy = valid.copy()
    valid_copy["date"] = valid_copy.index.date
    daily = valid_copy.groupby("date")["regime_label"].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)

    total_days = len(daily)
    dist = {}
    for regime_label in detector.regime_labels.values():
        count = (daily == regime_label).sum()
        dist[regime_label] = {"count": count, "pct": count / total_days * 100}

    # Classify into bearish/neutral/bullish
    from config.settings import BULLISH_REGIMES, BEARISH_REGIMES
    bearish_days = sum(d["count"] for label, d in dist.items() if label in BEARISH_REGIMES)
    bullish_days = sum(d["count"] for label, d in dist.items() if label in BULLISH_REGIMES)
    neutral_days = total_days - bearish_days - bullish_days

    return {
        "label": label,
        "total_days": total_days,
        "distribution": dist,
        "bearish_pct": bearish_days / total_days * 100,
        "neutral_pct": neutral_days / total_days * 100,
        "bullish_pct": bullish_days / total_days * 100,
        "bearish_days": bearish_days,
        "neutral_days": neutral_days,
        "bullish_days": bullish_days,
    }


# ═══════════════════════════════════════════════════════════════
#  STRATEGY RUNNERS (same as portfolio_backtest.py)
# ═══════════════════════════════════════════════════════════════

def run_amt_tema(regime_filter=None, blocked_regimes=None):
    """Undertow (ES 5m, short-only + IB Rejection)."""
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
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
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
    """Nexus (ES 5m, both directions)."""
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.config import StrategyConfig
    from backtester.data_loader import load_tos_csv
    from backtester.engine import run_backtest
    from backtester.metrics import compute_metrics

    data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento_2yr.csv"
    df = load_tos_csv(str(data_path), instrument="ES")

    cfg = StrategyConfig()
    cfg.direction_filter = "both"
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
    return trades, metrics


def run_ema_cross(regime_filter=None):
    """Crossfire (SPX credit spreads both sides)."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.ema_cross import EMACross
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = EMACross()
    trades = strategy.run_backtest(df, vix_data=vix)
    return trades


def run_ema_cross_dir(regime_filter=None):
    """Vector (SPX 0DTE puts, short-only)."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.ema_cross_directional import EMACrossDirectionalBest
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    df_all_path = SPX_OPTIONS_ROOT / "data" / "spy_1m_all.parquet"
    df_all = None
    if df_all_path.exists():
        df_all = load_spy_data(str(df_all_path))
        df_all = tag_sessions(df_all)

    strategy = EMACrossDirectionalBest()
    trades = strategy.run_backtest(df, vix_data=vix, df_all_sessions=df_all)
    return trades


# ═══════════════════════════════════════════════════════════════
#  POST-HOC FILTERING
# ═══════════════════════════════════════════════════════════════

def _post_hoc_filter(trades, regime_filter, blocked_regimes):
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
#  TRADE NORMALIZATION & STATS
# ═══════════════════════════════════════════════════════════════

def normalize_trade(t, strategy_name):
    """Extract (exit_time, pnl) from any trade type."""
    exit_time = getattr(t, 'exit_time', None) or getattr(t, 'time_exit', None)
    entry_time = getattr(t, 'entry_time', None) or getattr(t, 'time_enter', None)
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
        return {"label": label, "trades": 0, "pnl": 0, "win_rate": 0,
                "profit_factor": 0, "avg_trade": 0, "max_dd": 0, "sharpe": 0}

    pnls = [t["pnl"] for t in trades_normalized]
    n = len(pnls)
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_trade = total_pnl / n

    sorted_trades = sorted(trades_normalized, key=lambda t: t["exit_time"])
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted_trades:
        equity += t["pnl"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

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
    }


# ═══════════════════════════════════════════════════════════════
#  STRATEGY CONFIG — Regime blocking rules per model
# ═══════════════════════════════════════════════════════════════

# OLD model (7-state) blocking rules
OLD_BLOCKING = {
    "Undertow":  {"blocked": {"Strong Bull (Trend)", "Bull Run (Trend)"}},
    "Sentinel":  {"blocked": set()},  # unfiltered
    "Nexus":     {"blocked": set()},  # profile has no blocked (all regimes profitable)
    "Catalyst":  {"allowed": {"Crash (Panic)", "Bear Trend", "Distribution"}},
    "Cascade":   {"allowed": {"Crash (Panic)", "Bear Trend", "Distribution"}},
    "Fortress":  {"blocked": {"Crash (Panic)", "Distribution"}},
    "Crossfire": {"blocked_call": {"Crash (Panic)"}, "blocked_put": set()},
    "Vector":    {"blocked_call": {"Crash (Panic)"}, "blocked_put": set()},
}

# NEW model (5-state) blocking rules
NEW_BLOCKING = {
    "Undertow":  {"blocked": {"Bull Run (Trend)"}},  # Strong Bull absorbed into Bull Run
    "Sentinel":  {"blocked": set()},  # unfiltered
    "Nexus":     {"blocked": set()},  # unfiltered (old profiles don't map)
    "Catalyst":  {"allowed": {"Crash (Panic)", "Bear Trend"}},  # Distribution absorbed
    "Cascade":   {"allowed": {"Crash (Panic)", "Bear Trend"}},
    "Fortress":  {"blocked": {"Crash (Panic)"}},  # Distribution absorbed into Bear Trend
    "Crossfire": {"blocked_call": set(), "blocked_put": set()},  # unfiltered for now
    "Vector":    {"blocked_call": set(), "blocked_put": set()},  # unfiltered for now
}


def _all_7state_labels():
    return {"Crash (Panic)", "Bear Trend", "Distribution", "Accumulation (Chop)",
            "Recovery", "Bull Run (Trend)", "Strong Bull (Trend)"}


def _all_5state_labels():
    return {"Crash (Panic)", "Bear Trend", "Accumulation (Chop)",
            "Recovery", "Bull Run (Trend)"}


def get_blocked_set(strategy_name, blocking_rules, all_labels):
    """Compute the blocked regime set from rules (handles 'allowed' inversion)."""
    rules = blocking_rules[strategy_name]
    if "allowed" in rules:
        return all_labels - rules["allowed"]
    return rules.get("blocked", set())


# ═══════════════════════════════════════════════════════════════
#  RUN FULL BACKTEST WITH ONE MODEL
# ═══════════════════════════════════════════════════════════════

def run_portfolio(rf, blocking_rules, all_labels, model_label):
    """Run all 8 strategies with the given regime filter and return per-strategy stats."""
    results = {}
    all_trades = []

    # ── 1. Undertow (ES, regime-gated) ──
    print(f"\n  [{model_label}] 1/8 Undertow...", end="", flush=True)
    blocked = get_blocked_set("Undertow", blocking_rules, all_labels)
    try:
        trades, metrics = run_amt_tema(regime_filter=rf, blocked_regimes=blocked)
        normalized = [normalize_trade(t, "Undertow") for t in trades]
        all_trades.extend(normalized)
        results["Undertow (ES)"] = quick_stats(normalized, "Undertow (ES)")
        print(f" {results['Undertow (ES)']['trades']} trades, ${results['Undertow (ES)']['pnl']:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results["Undertow (ES)"] = quick_stats([], "Undertow (ES)")

    # ── 2. Sentinel (ES, unfiltered) ──
    print(f"  [{model_label}] 2/8 Sentinel...", end="", flush=True)
    try:
        lvl_trades = run_lvl_v13()
        normalized = [normalize_trade(t, "Sentinel") for t in lvl_trades]
        all_trades.extend(normalized)
        results["Sentinel (ES)"] = quick_stats(normalized, "Sentinel (ES)")
        print(f" {results['Sentinel (ES)']['trades']} trades, ${results['Sentinel (ES)']['pnl']:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results["Sentinel (ES)"] = quick_stats([], "Sentinel (ES)")

    # ── 3. Nexus (ES, post-hoc) ──
    print(f"  [{model_label}] 3/8 Nexus...", end="", flush=True)
    blocked = get_blocked_set("Nexus", blocking_rules, all_labels)
    try:
        ms_trades, ms_metrics = run_ms_os(regime_filter=rf)
        ms_trades = _post_hoc_filter(ms_trades, rf, blocked)
        normalized = [normalize_trade(t, "Nexus") for t in ms_trades]
        all_trades.extend(normalized)
        results["Nexus (ES)"] = quick_stats(normalized, "Nexus (ES)")
        print(f" {results['Nexus (ES)']['trades']} trades, ${results['Nexus (ES)']['pnl']:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results["Nexus (ES)"] = quick_stats([], "Nexus (ES)")

    # ── 4. Catalyst (SPX, regime-gated) ──
    print(f"  [{model_label}] 4/8 Catalyst...", end="", flush=True)
    try:
        trades, report = run_home_run(regime_filter=rf)
        # Post-hoc filter: only keep trades in allowed regimes
        blocked = get_blocked_set("Catalyst", blocking_rules, all_labels)
        trades = _post_hoc_filter(trades, rf, blocked)
        normalized = [normalize_trade(t, "Catalyst") for t in trades]
        all_trades.extend(normalized)
        results["Catalyst (SPX)"] = quick_stats(normalized, "Catalyst (SPX)")
        print(f" {results['Catalyst (SPX)']['trades']} trades, ${results['Catalyst (SPX)']['pnl']:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results["Catalyst (SPX)"] = quick_stats([], "Catalyst (SPX)")

    # ── 5. Cascade (SPX, regime-gated) ──
    print(f"  [{model_label}] 5/8 Cascade...", end="", flush=True)
    try:
        trades, report = run_bear_breakdown(regime_filter=rf)
        blocked = get_blocked_set("Cascade", blocking_rules, all_labels)
        trades = _post_hoc_filter(trades, rf, blocked)
        normalized = [normalize_trade(t, "Cascade") for t in trades]
        all_trades.extend(normalized)
        results["Cascade (SPX)"] = quick_stats(normalized, "Cascade (SPX)")
        print(f" {results['Cascade (SPX)']['trades']} trades, ${results['Cascade (SPX)']['pnl']:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results["Cascade (SPX)"] = quick_stats([], "Cascade (SPX)")

    # ── 6. Fortress (SPX, regime-gated) ──
    print(f"  [{model_label}] 6/8 Fortress...", end="", flush=True)
    try:
        trades, report = run_bull_credit(regime_filter=rf)
        blocked = get_blocked_set("Fortress", blocking_rules, all_labels)
        trades = _post_hoc_filter(trades, rf, blocked)
        normalized = [normalize_trade(t, "Fortress") for t in trades]
        all_trades.extend(normalized)
        results["Fortress (SPX)"] = quick_stats(normalized, "Fortress (SPX)")
        print(f" {results['Fortress (SPX)']['trades']} trades, ${results['Fortress (SPX)']['pnl']:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results["Fortress (SPX)"] = quick_stats([], "Fortress (SPX)")

    # ── 7. Crossfire (SPX, per-side regime-gated) ──
    print(f"  [{model_label}] 7/8 Crossfire...", end="", flush=True)
    try:
        ema_trades = run_ema_cross(regime_filter=rf)
        rules = blocking_rules["Crossfire"]
        blocked_call = rules.get("blocked_call", set())
        blocked_put = rules.get("blocked_put", set())
        ema_trades = _post_hoc_filter_per_side(ema_trades, rf, blocked_call, blocked_put)
        normalized = [normalize_trade(t, "Crossfire") for t in ema_trades]
        all_trades.extend(normalized)
        results["Crossfire (SPX)"] = quick_stats(normalized, "Crossfire (SPX)")
        print(f" {results['Crossfire (SPX)']['trades']} trades, ${results['Crossfire (SPX)']['pnl']:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results["Crossfire (SPX)"] = quick_stats([], "Crossfire (SPX)")

    # ── 8. Vector (SPX, regime-gated) ──
    print(f"  [{model_label}] 8/8 Vector...", end="", flush=True)
    try:
        ema_dir_trades = run_ema_cross_dir(regime_filter=rf)
        rules = blocking_rules["Vector"]
        blocked_call = rules.get("blocked_call", set())
        ema_dir_trades = _post_hoc_filter(ema_dir_trades, rf, blocked_call)
        normalized = [normalize_trade(t, "Vector") for t in ema_dir_trades]
        all_trades.extend(normalized)
        results["Vector (SPX)"] = quick_stats(normalized, "Vector (SPX)")
        print(f" {results['Vector (SPX)']['trades']} trades, ${results['Vector (SPX)']['pnl']:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        results["Vector (SPX)"] = quick_stats([], "Vector (SPX)")

    # Portfolio total
    portfolio = quick_stats(all_trades, "PORTFOLIO")
    return results, portfolio


# ═══════════════════════════════════════════════════════════════
#  COMPARISON REPORT
# ═══════════════════════════════════════════════════════════════

def print_comparison(old_results, old_portfolio, new_results, new_portfolio,
                     old_dist, new_dist):
    """Print the side-by-side comparison table."""

    strategies = [
        "Undertow (ES)", "Sentinel (ES)", "Nexus (ES)",
        "Catalyst (SPX)", "Cascade (SPX)", "Fortress (SPX)",
        "Crossfire (SPX)", "Vector (SPX)",
    ]

    print("\n")
    print("=" * 120)
    print("  PORTFOLIO BACKTEST: OLD vs NEW REGIME MODEL")
    print("=" * 120)

    # Header
    print(f"\n  {'STRATEGY-BY-STRATEGY COMPARISON':^118s}")
    print(f"  {'':22s} |{'---- OLD (3feat/7state) ----':^30s}|{'---- NEW (5feat/5state) ----':^30s}|")
    print(f"  {'Strategy':<22s} {'Trades':>6s} {'P&L':>11s} {'PF':>6s} {'WR':>6s}  "
          f"{'Trades':>6s} {'P&L':>11s} {'PF':>6s} {'WR':>6s}  {'Delta P&L':>11s} {'DTrd':>5s}")
    print(f"  {'-' * 116}")

    for strat in strategies:
        old = old_results.get(strat, quick_stats([], strat))
        new = new_results.get(strat, quick_stats([], strat))

        delta_pnl = new["pnl"] - old["pnl"]
        delta_trades = new["trades"] - old["trades"]

        old_pf = f"{old['profit_factor']:.2f}" if old['profit_factor'] < 100 else "inf"
        new_pf = f"{new['profit_factor']:.2f}" if new['profit_factor'] < 100 else "inf"

        dpnl_str = f"+${delta_pnl:>8,.0f}" if delta_pnl >= 0 else f"-${abs(delta_pnl):>8,.0f}"
        dtrd_str = f"+{delta_trades:>4d}" if delta_trades >= 0 else f"{delta_trades:>5d}"

        print(f"  {strat:<22s} {old['trades']:>6d} ${old['pnl']:>9,.0f} {old_pf:>6s} "
              f"{old['win_rate']:>5.1f}%  "
              f"{new['trades']:>6d} ${new['pnl']:>9,.0f} {new_pf:>6s} "
              f"{new['win_rate']:>5.1f}%  "
              f"{dpnl_str} {dtrd_str}")

    print(f"  {'-' * 116}")

    # Portfolio totals
    old_p = old_portfolio
    new_p = new_portfolio
    delta_total = new_p["pnl"] - old_p["pnl"]
    delta_t = new_p["trades"] - old_p["trades"]

    old_pf = f"{old_p['profit_factor']:.2f}" if old_p['profit_factor'] < 100 else "inf"
    new_pf = f"{new_p['profit_factor']:.2f}" if new_p['profit_factor'] < 100 else "inf"

    dpnl_str = f"+${delta_total:>8,.0f}" if delta_total >= 0 else f"-${abs(delta_total):>8,.0f}"
    dtrd_str = f"+{delta_t:>4d}" if delta_t >= 0 else f"{delta_t:>5d}"

    print(f"  {'PORTFOLIO TOTAL':<22s} {old_p['trades']:>6d} ${old_p['pnl']:>9,.0f} {old_pf:>6s} "
          f"{old_p['win_rate']:>5.1f}%  "
          f"{new_p['trades']:>6d} ${new_p['pnl']:>9,.0f} {new_pf:>6s} "
          f"{new_p['win_rate']:>5.1f}%  "
          f"{dpnl_str} {dtrd_str}")

    # Sharpe comparison
    print(f"\n  {'':22s} {'Sharpe':>6s} {'MaxDD':>11s}                "
          f"{'Sharpe':>6s} {'MaxDD':>11s}")
    print(f"  {'PORTFOLIO':<22s} {old_p['sharpe']:>6.2f} ${old_p['max_dd']:>9,.0f}                "
          f"{new_p['sharpe']:>6.2f} ${new_p['max_dd']:>9,.0f}")

    # Regime distribution
    print(f"\n{'=' * 120}")
    print(f"  REGIME DISTRIBUTION (trading days)")
    print(f"{'=' * 120}")
    print(f"\n  OLD (7-state, 3 features):")
    print(f"    {old_dist['bearish_pct']:.1f}% bearish ({old_dist['bearish_days']}d), "
          f"{old_dist['neutral_pct']:.1f}% neutral ({old_dist['neutral_days']}d), "
          f"{old_dist['bullish_pct']:.1f}% bullish ({old_dist['bullish_days']}d)")
    for label, d in sorted(old_dist["distribution"].items(), key=lambda x: -x[1]["pct"]):
        bar = "#" * int(d["pct"] / 2)
        print(f"    {label:<25s} {d['count']:>4d}d ({d['pct']:>5.1f}%) {bar}")

    print(f"\n  NEW (5-state, 5 features):")
    print(f"    {new_dist['bearish_pct']:.1f}% bearish ({new_dist['bearish_days']}d), "
          f"{new_dist['neutral_pct']:.1f}% neutral ({new_dist['neutral_days']}d), "
          f"{new_dist['bullish_pct']:.1f}% bullish ({new_dist['bullish_days']}d)")
    for label, d in sorted(new_dist["distribution"].items(), key=lambda x: -x[1]["pct"]):
        bar = "#" * int(d["pct"] / 2)
        print(f"    {label:<25s} {d['count']:>4d}d ({d['pct']:>5.1f}%) {bar}")

    # Key changes summary
    print(f"\n{'=' * 120}")
    print(f"  KEY CHANGES")
    print(f"{'=' * 120}")

    gained = []
    lost = []
    for strat in strategies:
        old = old_results.get(strat, quick_stats([], strat))
        new = new_results.get(strat, quick_stats([], strat))
        delta = new["trades"] - old["trades"]
        if delta > 0:
            gained.append(f"{strat} (+{delta})")
        elif delta < 0:
            lost.append(f"{strat} ({delta})")

    net_sign = "+" if delta_total >= 0 else "-"
    net_tsign = "+" if delta_t >= 0 else ""
    print(f"\n  Strategies that GAINED trades: {', '.join(gained) if gained else 'none'}")
    print(f"  Strategies that LOST trades:   {', '.join(lost) if lost else 'none'}")
    print(f"  Net P&L impact:                {net_sign}${abs(delta_total):,.0f}")
    print(f"  Net trade count change:        {net_tsign}{delta_t}")

    # Per-strategy impact detail for the most affected
    print(f"\n  BIGGEST IMPACTS:")
    impacts = []
    for strat in strategies:
        old = old_results.get(strat, quick_stats([], strat))
        new = new_results.get(strat, quick_stats([], strat))
        impacts.append((strat, new["pnl"] - old["pnl"], new["trades"] - old["trades"]))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    for strat, dpnl, dtrd in impacts[:5]:
        if dpnl == 0 and dtrd == 0:
            continue
        dpnl_str = f"+${dpnl:>8,.0f}" if dpnl >= 0 else f"-${abs(dpnl):>8,.0f}"
        dtrd_str = f"+{dtrd}" if dtrd >= 0 else f"{dtrd}"
        print(f"    {strat:<22s} {dpnl_str} P&L  ({dtrd_str} trades)")

    print(f"\n{'=' * 120}\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MEDAL-LION FUND: OLD vs NEW REGIME MODEL COMPARISON")
    print("  OLD: 3 features (returns, range, vol_vol), 7 states")
    print("  NEW: 5 features (+ close_in_range, close_vs_open smoothed), 5 states")
    print("=" * 70)

    # ── Part A: Fit BOTH models ──
    detector_old, full_df_old, features_old, spy_hourly = build_old_model()
    detector_new, full_df_new, features_new, _ = build_new_model()

    # Create RegimeFilters
    rf_old = create_regime_filter(detector_old, full_df_old)
    rf_new = create_regime_filter(detector_new, full_df_new)

    # ── Part B: Regime distribution analysis ──
    old_dist = analyze_regime_distribution(detector_old, features_old, "OLD (3feat/7state)")
    new_dist = analyze_regime_distribution(detector_new, features_new, "NEW (5feat/5state)")

    # ── Part C: Run portfolio with BOTH models ──
    print("\n" + "=" * 70)
    print("  RUNNING PORTFOLIO WITH OLD MODEL (3 feat / 7 state)")
    print("=" * 70)
    old_results, old_portfolio = run_portfolio(
        rf_old, OLD_BLOCKING, _all_7state_labels(), "OLD"
    )

    print("\n" + "=" * 70)
    print("  RUNNING PORTFOLIO WITH NEW MODEL (5 feat / 5 state)")
    print("=" * 70)
    new_results, new_portfolio = run_portfolio(
        rf_new, NEW_BLOCKING, _all_5state_labels(), "NEW"
    )

    # ── Part D: Comparison report ──
    print_comparison(old_results, old_portfolio, new_results, new_portfolio,
                     old_dist, new_dist)


if __name__ == "__main__":
    main()
