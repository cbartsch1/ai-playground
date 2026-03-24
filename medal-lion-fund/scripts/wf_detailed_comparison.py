#!/usr/bin/env python3
"""Medal-Lion Fund — Walk-Forward Detailed Portfolio Backtest

Compares UNFILTERED (no regime filter) vs WALK-FORWARD FILTERED (new 5-state model)
across all 8 strategies. Uses bias-free walk-forward regime labels from parquet.

This is the definitive test: does the regime filter ADD or SUBTRACT value?

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    .venv/bin/python scripts/wf_detailed_comparison.py 2>&1
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

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
#  WALK-FORWARD REGIME FILTER
# ═══════════════════════════════════════════════════════════════

def load_wf_regime_filter():
    """Load the walk-forward regime filter from parquet."""
    _swap_path(MEDALLION_ROOT)
    from models.wf_regime_api import WalkForwardRegimeFilter

    wf_path = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"
    if not wf_path.exists():
        print(f"ERROR: Walk-forward regimes not found: {wf_path}")
        sys.exit(1)

    rf = WalkForwardRegimeFilter(str(wf_path))
    return rf


# ═══════════════════════════════════════════════════════════════
#  5-STATE BLOCKING RULES (from task specification)
# ═══════════════════════════════════════════════════════════════

ALL_5STATE = {"Crash (Panic)", "Bear Trend", "Accumulation (Chop)", "Recovery", "Bull Run (Trend)"}

WF_BLOCKING = {
    "Undertow":  {"blocked": {"Bull Run (Trend)"}},
    "Sentinel":  {"blocked": set()},          # unfiltered
    "Nexus":     {"blocked": set()},          # unfiltered
    "Catalyst":  {"allowed": {"Crash (Panic)", "Bear Trend"}},
    "Cascade":   {"allowed": {"Crash (Panic)", "Bear Trend"}},
    "Fortress":  {"blocked": {"Crash (Panic)"}},
    "Crossfire": {"blocked_call": set(), "blocked_put": set()},   # unfiltered
    "Vector":    {"blocked_call": set(), "blocked_put": set()},   # unfiltered
}


def get_blocked_set(strategy_name):
    """Compute the blocked regime set from WF blocking rules."""
    rules = WF_BLOCKING[strategy_name]
    if "allowed" in rules:
        return ALL_5STATE - rules["allowed"]
    return rules.get("blocked", set())


# ═══════════════════════════════════════════════════════════════
#  STRATEGY RUNNERS
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


def run_ms_os():
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


def run_ema_cross():
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


def run_ema_cross_dir():
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
#  TRADE NORMALIZATION & POST-HOC FILTERING
# ═══════════════════════════════════════════════════════════════

def normalize_trade(t, strategy_name):
    """Extract (entry_time, exit_time, pnl, direction) from any trade type."""
    exit_time = getattr(t, 'exit_time', None) or getattr(t, 'time_exit', None)
    entry_time = getattr(t, 'entry_time', None) or getattr(t, 'time_enter', None)
    pnl = getattr(t, 'pnl_dollar', None)
    if pnl is None or pnl == 0:
        pnl = getattr(t, 'net_pnl', 0)
    direction = getattr(t, 'direction', None)
    return {
        "strategy": strategy_name,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl": pnl,
        "direction": direction,
    }


def post_hoc_filter(trades_normalized, regime_filter, blocked_regimes):
    """Post-hoc filter normalized trades — remove those in blocked regimes."""
    if not blocked_regimes or regime_filter is None:
        return trades_normalized
    kept = []
    for t in trades_normalized:
        regime = regime_filter.get_regime_at(t["entry_time"])
        label = regime.get("label", "")
        if label not in blocked_regimes:
            kept.append(t)
    return kept


def post_hoc_filter_per_side(trades_normalized, regime_filter, blocked_call, blocked_put):
    """Post-hoc filter normalized trades per-side (for Crossfire)."""
    if regime_filter is None:
        return trades_normalized
    if not blocked_call and not blocked_put:
        return trades_normalized
    kept = []
    for t in trades_normalized:
        regime = regime_filter.get_regime_at(t["entry_time"])
        label = regime.get("label", "")
        if t["direction"] == "short" and label in blocked_call:
            continue
        if t["direction"] == "long" and label in blocked_put:
            continue
        kept.append(t)
    return kept


# ═══════════════════════════════════════════════════════════════
#  DETAILED STATS
# ═══════════════════════════════════════════════════════════════

def detailed_stats(trades_normalized, label):
    """Compute full detailed stats from normalized trade list."""
    empty = {
        "label": label, "trades": 0, "pnl": 0.0, "profit_factor": 0.0,
        "win_rate": 0.0, "sharpe": 0.0, "max_dd": 0.0,
        "winning_months": 0, "losing_months": 0, "total_months": 0,
        "monthly_wr": 0.0, "avg_trade": 0.0, "best_month": 0.0,
        "worst_month": 0.0, "trades_per_day": 0.0, "monthly_pnl": {},
    }
    if not trades_normalized:
        return empty

    pnls = [t["pnl"] for t in trades_normalized]
    n = len(pnls)
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100 if n > 0 else 0.0
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_trade = total_pnl / n if n > 0 else 0.0

    sorted_trades = sorted(trades_normalized, key=lambda t: t["exit_time"])

    # Equity curve + Max Drawdown
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

    # Monthly P&L
    monthly_pnl = defaultdict(float)
    for t in sorted_trades:
        et = t["exit_time"]
        if hasattr(et, "strftime"):
            month_key = et.strftime("%Y-%m")
        else:
            month_key = "unknown"
        monthly_pnl[month_key] += t["pnl"]

    monthly_values = list(monthly_pnl.values())
    winning_months = sum(1 for v in monthly_values if v > 0)
    losing_months = sum(1 for v in monthly_values if v <= 0)
    total_months = len(monthly_values)
    monthly_wr = (winning_months / total_months * 100) if total_months > 0 else 0.0
    best_month = max(monthly_values) if monthly_values else 0.0
    worst_month = min(monthly_values) if monthly_values else 0.0

    # Sharpe from daily P&L
    daily_pnl = defaultdict(float)
    for t in sorted_trades:
        et = t["exit_time"]
        if hasattr(et, "strftime"):
            day = et.strftime("%Y-%m-%d")
        else:
            day = "unknown"
        daily_pnl[day] += t["pnl"]
    daily_arr = np.array(list(daily_pnl.values()))
    sharpe = 0.0
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = (daily_arr.mean() / daily_arr.std()) * np.sqrt(252)

    # Trades per day
    if len(sorted_trades) >= 2:
        first_exit = sorted_trades[0]["exit_time"]
        last_exit = sorted_trades[-1]["exit_time"]
        if hasattr(first_exit, "date") and hasattr(last_exit, "date"):
            span_days = (last_exit.date() - first_exit.date()).days
            approx_trading_days = max(1, int(span_days * 252 / 365))
        else:
            approx_trading_days = 1
    else:
        approx_trading_days = 1
    trades_per_day = n / approx_trading_days if approx_trading_days > 0 else 0.0

    return {
        "label": label, "trades": n, "pnl": total_pnl,
        "profit_factor": pf, "win_rate": wr, "sharpe": sharpe,
        "max_dd": max_dd, "winning_months": winning_months,
        "losing_months": losing_months, "total_months": total_months,
        "monthly_wr": monthly_wr, "avg_trade": avg_trade,
        "best_month": best_month, "worst_month": worst_month,
        "trades_per_day": trades_per_day, "monthly_pnl": dict(monthly_pnl),
    }


# ═══════════════════════════════════════════════════════════════
#  RUN ALL STRATEGIES — returns raw normalized trades per strategy
# ═══════════════════════════════════════════════════════════════

STRATEGY_ORDER = [
    "Undertow", "Sentinel", "Nexus", "Catalyst",
    "Cascade", "Fortress", "Crossfire", "Vector",
]

STRATEGY_DESC = {
    "Undertow":  "ES 5m Short + IB Rejection",
    "Sentinel":  "ES 5m Short — Unfiltered",
    "Nexus":     "ES 5m Both — MS/OS",
    "Catalyst":  "SPX 0DTE Puts — Short",
    "Cascade":   "SPX 0DTE Puts — Short",
    "Fortress":  "SPX Put Credit — Long",
    "Crossfire": "SPX Credit Spreads — Both",
    "Vector":    "SPX 0DTE Puts — Short",
}

# Cache for raw trades — we run each strategy ONCE then filter
_raw_trades_cache = {}


def run_all_strategies_raw():
    """Run all 8 strategies with NO regime filter, return raw normalized trades."""
    raw = {}

    # ── 1. Undertow ──
    print(f"\n  Running 1/8 Undertow...", end="", flush=True)
    try:
        trades, _ = run_amt_tema(regime_filter=None, blocked_regimes=None)
        normalized = [normalize_trade(t, "Undertow") for t in trades]
        raw["Undertow"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw["Undertow"] = []

    # ── 2. Sentinel (always unfiltered) ──
    print(f"  Running 2/8 Sentinel...", end="", flush=True)
    try:
        lvl_trades = run_lvl_v13()
        normalized = [normalize_trade(t, "Sentinel") for t in lvl_trades]
        raw["Sentinel"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw["Sentinel"] = []

    # ── 3. Nexus ──
    print(f"  Running 3/8 Nexus...", end="", flush=True)
    try:
        ms_trades, _ = run_ms_os()
        normalized = [normalize_trade(t, "Nexus") for t in ms_trades]
        raw["Nexus"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw["Nexus"] = []

    # ── 4. Catalyst ──
    print(f"  Running 4/8 Catalyst...", end="", flush=True)
    try:
        trades, _ = run_home_run(regime_filter=None)
        normalized = [normalize_trade(t, "Catalyst") for t in trades]
        raw["Catalyst"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw["Catalyst"] = []

    # ── 5. Cascade ──
    print(f"  Running 5/8 Cascade...", end="", flush=True)
    try:
        trades, _ = run_bear_breakdown(regime_filter=None)
        normalized = [normalize_trade(t, "Cascade") for t in trades]
        raw["Cascade"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw["Cascade"] = []

    # ── 6. Fortress ──
    print(f"  Running 6/8 Fortress...", end="", flush=True)
    try:
        trades, _ = run_bull_credit(regime_filter=None)
        normalized = [normalize_trade(t, "Fortress") for t in trades]
        raw["Fortress"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw["Fortress"] = []

    # ── 7. Crossfire ──
    print(f"  Running 7/8 Crossfire...", end="", flush=True)
    try:
        trades = run_ema_cross()
        normalized = [normalize_trade(t, "Crossfire") for t in trades]
        raw["Crossfire"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw["Crossfire"] = []

    # ── 8. Vector ──
    print(f"  Running 8/8 Vector...", end="", flush=True)
    try:
        trades = run_ema_cross_dir()
        normalized = [normalize_trade(t, "Vector") for t in trades]
        raw["Vector"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw["Vector"] = []

    return raw


def apply_wf_filter(raw_trades, rf):
    """Apply walk-forward regime filter post-hoc to all raw trades."""
    filtered = {}

    for name in STRATEGY_ORDER:
        trades = raw_trades.get(name, [])
        if not trades:
            filtered[name] = []
            continue

        if name == "Undertow":
            # Undertow: engine-level filtering already handled if regime_filter passed
            # Since we ran unfiltered, we need to post-hoc filter
            blocked = get_blocked_set("Undertow")
            filtered[name] = post_hoc_filter(trades, rf, blocked)

        elif name == "Sentinel":
            # Always unfiltered
            filtered[name] = trades

        elif name == "Nexus":
            # Unfiltered in 5-state model
            blocked = get_blocked_set("Nexus")
            filtered[name] = post_hoc_filter(trades, rf, blocked)

        elif name in ("Catalyst", "Cascade"):
            blocked = get_blocked_set(name)
            filtered[name] = post_hoc_filter(trades, rf, blocked)

        elif name == "Fortress":
            blocked = get_blocked_set("Fortress")
            filtered[name] = post_hoc_filter(trades, rf, blocked)

        elif name == "Crossfire":
            rules = WF_BLOCKING["Crossfire"]
            bc = rules.get("blocked_call", set())
            bp = rules.get("blocked_put", set())
            filtered[name] = post_hoc_filter_per_side(trades, rf, bc, bp)

        elif name == "Vector":
            rules = WF_BLOCKING["Vector"]
            bc = rules.get("blocked_call", set())
            filtered[name] = post_hoc_filter(trades, rf, bc)

        else:
            filtered[name] = trades

    return filtered


# ═══════════════════════════════════════════════════════════════
#  FORMATTED OUTPUT
# ═══════════════════════════════════════════════════════════════

def fmt_d(v):
    """Format dollar value."""
    if v >= 0:
        return f"${v:,.0f}"
    return f"-${abs(v):,.0f}"


def fmt_dd(v):
    """Format delta dollar with sign."""
    if v >= 0:
        return f"+${v:,.0f}"
    return f"-${abs(v):,.0f}"


def fmt_pf(v):
    if v == 0:
        return "0.00"
    if v >= 100 or v == float("inf"):
        return "inf"
    return f"{v:.2f}"


def print_strategy_block(name, desc, uf, wf):
    """Print one strategy's detailed comparison block."""
    print(f"\n{name.upper()} ({desc})")
    print(f"{'=' * 65}")

    rows = [
        ("Trades:",       f"{uf['trades']}",            f"{wf['trades']}",            f"{wf['trades'] - uf['trades']:+d}"),
        ("P&L:",          fmt_d(uf['pnl']),             fmt_d(wf['pnl']),             fmt_dd(wf['pnl'] - uf['pnl'])),
        ("Profit Factor:", fmt_pf(uf['profit_factor']), fmt_pf(wf['profit_factor']),  ""),
        ("Win Rate:",     f"{uf['win_rate']:.1f}%",     f"{wf['win_rate']:.1f}%",     f"{wf['win_rate'] - uf['win_rate']:+.1f}%"),
        ("Sharpe:",       f"{uf['sharpe']:.2f}",        f"{wf['sharpe']:.2f}",        f"{wf['sharpe'] - uf['sharpe']:+.2f}"),
        ("Max Drawdown:", fmt_d(uf['max_dd']),          fmt_d(wf['max_dd']),          fmt_dd(wf['max_dd'] - uf['max_dd'])),
        ("Win/Lose Mo:",  f"{uf['winning_months']}W/{uf['losing_months']}L",
                          f"{wf['winning_months']}W/{wf['losing_months']}L",          ""),
        ("Monthly WR:",   f"{uf['monthly_wr']:.0f}%",   f"{wf['monthly_wr']:.0f}%",   f"{wf['monthly_wr'] - uf['monthly_wr']:+.0f}%"),
        ("Avg Trade:",    fmt_d(uf['avg_trade']),       fmt_d(wf['avg_trade']),       ""),
        ("Best Month:",   fmt_d(uf['best_month']),      fmt_d(wf['best_month']),      ""),
        ("Worst Month:",  fmt_d(uf['worst_month']),     fmt_d(wf['worst_month']),     ""),
        ("Trades/Day:",   f"{uf['trades_per_day']:.2f}", f"{wf['trades_per_day']:.2f}", ""),
    ]

    # Determine blocked regimes for display
    if name in WF_BLOCKING:
        rules = WF_BLOCKING[name]
        if "allowed" in rules:
            blocked = ALL_5STATE - rules["allowed"]
        elif "blocked" in rules:
            blocked = rules["blocked"]
        else:
            blocked = set()
        filter_desc = f"Blocked: {', '.join(sorted(blocked))}" if blocked else "Unfiltered"
        print(f"  Filter: {filter_desc}")
    print(f"  {'':20s} {'UNFILTERED':>14s}  {'WF-FILTERED':>14s}  {'Delta':>14s}")
    for label, uv, wv, dv in rows:
        print(f"  {label:20s} {uv:>14s}  {wv:>14s}  {dv:>14s}")


def print_full_report(uf_stats, wf_stats, uf_portfolio, wf_portfolio, wf_dist):
    """Print the complete report."""

    print("\n")
    print("=" * 70)
    print("  WALK-FORWARD PORTFOLIO BACKTEST")
    print("  UNFILTERED vs NEW 5-STATE REGIME FILTER (walk-forward labels)")
    print("=" * 70)

    # Regime distribution
    print(f"\n  WF Regime Distribution (hourly bars):")
    for label, d in sorted(wf_dist.items(), key=lambda x: -x[1]["pct"]):
        bar = "#" * int(d["pct"] / 2)
        print(f"    {label:<25s} {d['count']:>5d} bars ({d['pct']:>5.1f}%) {bar}")

    for name in STRATEGY_ORDER:
        desc = STRATEGY_DESC[name]
        uf = uf_stats.get(name, detailed_stats([], name))
        wf = wf_stats.get(name, detailed_stats([], name))
        print_strategy_block(name, desc, uf, wf)

    # Portfolio Total
    print(f"\n\n{'=' * 70}")
    print(f"  PORTFOLIO TOTAL (All 8 Strategies Combined)")
    print(f"{'=' * 70}")
    print_strategy_block("PORTFOLIO", "All 8 Strategies", uf_portfolio, wf_portfolio)

    # Impact Summary
    print(f"\n\n{'=' * 70}")
    print(f"  REGIME FILTER VALUE ASSESSMENT")
    print(f"{'=' * 70}")

    improved = []
    hurt = []
    unchanged = []

    for name in STRATEGY_ORDER:
        uf = uf_stats.get(name, detailed_stats([], name))
        wf = wf_stats.get(name, detailed_stats([], name))
        delta_pnl = wf["pnl"] - uf["pnl"]
        delta_trades = wf["trades"] - uf["trades"]

        if delta_trades == 0 and abs(delta_pnl) < 0.01:
            unchanged.append((name, delta_pnl, delta_trades))
        elif delta_pnl > 0:
            improved.append((name, delta_pnl, delta_trades))
        else:
            hurt.append((name, delta_pnl, delta_trades))

    improved.sort(key=lambda x: x[1], reverse=True)
    hurt.sort(key=lambda x: x[1])

    print(f"\n  Strategies IMPROVED by regime filter:")
    if improved:
        for name, dpnl, dtrd in improved:
            print(f"    {name:<20s}  {fmt_dd(dpnl)} P&L  ({dtrd:+d} trades)")
    else:
        print(f"    (none)")

    print(f"\n  Strategies HURT by regime filter:")
    if hurt:
        for name, dpnl, dtrd in hurt:
            print(f"    {name:<20s}  {fmt_dd(dpnl)} P&L  ({dtrd:+d} trades)")
    else:
        print(f"    (none)")

    print(f"\n  Strategies UNCHANGED by regime filter:")
    if unchanged:
        for name, dpnl, dtrd in unchanged:
            print(f"    {name:<20s}  (no filtering applied)")
    else:
        print(f"    (none)")

    net = wf_portfolio["pnl"] - uf_portfolio["pnl"]
    net_trades = wf_portfolio["trades"] - uf_portfolio["trades"]
    print(f"\n  Net portfolio impact: {fmt_dd(net)} P&L  ({net_trades:+d} trades)")
    print(f"  Sharpe: {uf_portfolio['sharpe']:.2f} -> {wf_portfolio['sharpe']:.2f}")
    print(f"  Max DD: {fmt_d(uf_portfolio['max_dd'])} -> {fmt_d(wf_portfolio['max_dd'])}")

    print(f"\n{'=' * 70}\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MEDAL-LION FUND: WALK-FORWARD DETAILED PORTFOLIO BACKTEST")
    print("  Comparing: UNFILTERED vs 5-STATE WF REGIME FILTER")
    print("  Labels: data/processed/walk_forward_regimes.parquet")
    print("=" * 70)

    # Load walk-forward regime filter
    print("\n  Loading walk-forward regime labels...")
    rf = load_wf_regime_filter()

    # Analyze WF regime distribution
    wf_df = pd.read_parquet(MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet")
    wf_dist = {}
    total_bars = len(wf_df)
    for label in wf_df["regime_label"].unique():
        count = (wf_df["regime_label"] == label).sum()
        wf_dist[label] = {"count": int(count), "pct": count / total_bars * 100}

    # Run all strategies ONCE (unfiltered)
    print(f"\n{'=' * 70}")
    print(f"  RUNNING ALL 8 STRATEGIES (UNFILTERED)")
    print(f"{'=' * 70}")
    raw_trades = run_all_strategies_raw()

    # Apply walk-forward filter post-hoc
    print(f"\n{'=' * 70}")
    print(f"  APPLYING WALK-FORWARD REGIME FILTER (post-hoc)")
    print(f"{'=' * 70}")
    filtered_trades = apply_wf_filter(raw_trades, rf)

    for name in STRATEGY_ORDER:
        uf_n = len(raw_trades.get(name, []))
        wf_n = len(filtered_trades.get(name, []))
        blocked = get_blocked_set(name) if "allowed" not in WF_BLOCKING[name] else ALL_5STATE - WF_BLOCKING[name].get("allowed", set())
        if blocked:
            print(f"  {name:<15s}: {uf_n} -> {wf_n} trades ({uf_n - wf_n} blocked)")
        else:
            print(f"  {name:<15s}: {uf_n} trades (unfiltered)")

    # Compute detailed stats
    uf_stats = {}
    wf_stats = {}
    uf_all = []
    wf_all = []

    for name in STRATEGY_ORDER:
        uf_stats[name] = detailed_stats(raw_trades.get(name, []), name)
        wf_stats[name] = detailed_stats(filtered_trades.get(name, []), name)
        uf_all.extend(raw_trades.get(name, []))
        wf_all.extend(filtered_trades.get(name, []))

    uf_portfolio = detailed_stats(uf_all, "PORTFOLIO")
    wf_portfolio = detailed_stats(wf_all, "PORTFOLIO")

    # Print the full report
    print_full_report(uf_stats, wf_stats, uf_portfolio, wf_portfolio, wf_dist)

    # Save results to JSON
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
        "regime_mode": "walk-forward",
        "n_states": 5,
        "wf_parquet": "data/processed/walk_forward_regimes.parquet",
        "unfiltered": {name: _clean(uf_stats[name]) for name in STRATEGY_ORDER},
        "wf_filtered": {name: _clean(wf_stats[name]) for name in STRATEGY_ORDER},
        "unfiltered_portfolio": _clean(uf_portfolio),
        "wf_filtered_portfolio": _clean(wf_portfolio),
    }

    out_path = out_dir / "wf_detailed_comparison.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
