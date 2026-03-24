#!/usr/bin/env python3
"""Medal-Lion Fund — WF Regime: UNFILTERED vs FILTERED Comparison

Runs all 8 strategies twice using walk-forward regime labels:
  1. UNFILTERED — no regime blocking at all
  2. FILTERED — original March 2 blocking rules mapped to 5-state

Focus: MAX DRAWDOWN per strategy and portfolio-level.

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    .venv/bin/python scripts/wf_filtered_vs_unfiltered.py
"""

import sys
from pathlib import Path
from collections import defaultdict

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
#  5-STATE BLOCKING RULES (original March 2 rules mapped)
# ═══════════════════════════════════════════════════════════════

ALL_5STATE = {"Crash (Panic)", "Bear Trend", "Accumulation (Chop)", "Recovery", "Bull Run (Trend)"}

# Original rules mapped to 5-state:
# Undertow: blocked = {Strong Bull, Bull Run} -> {Bull Run}
# Sentinel: unfiltered
# Nexus: unfiltered
# Catalyst: allowed = {Crash, Bear, Distribution} -> allowed = {Crash, Bear}
#           blocked = ALL - allowed = {Accumulation, Recovery, Bull Run}
# Cascade: same as Catalyst
# Fortress: blocked = {Crash, Distribution} -> blocked = {Crash}
# Crossfire: unfiltered (profiles were minimal)
# Vector: unfiltered

FILTERED_RULES = {
    "Undertow":  {"blocked": {"Bull Run (Trend)"}},
    "Sentinel":  {"blocked": set()},
    "Nexus":     {"blocked": set()},
    "Catalyst":  {"blocked": ALL_5STATE - {"Crash (Panic)", "Bear Trend"}},
    "Cascade":   {"blocked": ALL_5STATE - {"Crash (Panic)", "Bear Trend"}},
    "Fortress":  {"blocked": {"Crash (Panic)"}},
    "Crossfire": {"blocked": set()},
    "Vector":    {"blocked": set()},
}


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
    return trades


def run_lvl_v13():
    """Sentinel (ES 5m, short-only, always unfiltered)."""
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
    return trades


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
    return trades


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
    return trades


def run_ms_os():
    """Nexus (ES 5m, both directions). No internal regime filter."""
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.config import StrategyConfig
    from backtester.data_loader import load_tos_csv
    from backtester.engine import run_backtest

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
    return trades


def run_ema_cross():
    """Crossfire (SPX credit spreads both sides). No internal regime filter."""
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
    """Vector (SPX 0DTE puts, short-only). No internal regime filter."""
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
#  TRADE NORMALIZATION & POST-HOC FILTER
# ═══════════════════════════════════════════════════════════════

def normalize_trade(t, strategy_name):
    """Extract (exit_time, entry_time, pnl, direction) from any trade type."""
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


def post_hoc_filter(trades_raw, rf, blocked_regimes):
    """Post-hoc filter: remove trades whose entry falls in a blocked regime."""
    if not blocked_regimes or rf is None:
        return trades_raw
    kept = []
    for t in trades_raw:
        entry = getattr(t, 'entry_time', None) or getattr(t, 'time_enter', None)
        regime = rf.get_regime_at(entry)
        label = regime.get("label", "")
        if label not in blocked_regimes:
            kept.append(t)
    return kept


# ═══════════════════════════════════════════════════════════════
#  FULL STATS COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_full_stats(trades_normalized, label):
    """Compute all metrics from normalized trade list."""
    empty = {
        "label": label, "trades": 0, "pnl": 0.0, "profit_factor": 0.0,
        "win_rate": 0.0, "sharpe": 0.0, "max_dd": 0.0,
        "max_consec_losers": 0, "winning_months": 0, "losing_months": 0,
        "total_months": 0, "worst_month": 0.0, "worst_trade": 0.0,
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

    # Sort by exit time for equity curve
    sorted_trades = sorted(trades_normalized, key=lambda t: t["exit_time"])

    # Max Drawdown (peak-to-trough of cumulative P&L)
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

    # Max consecutive losers
    max_consec = 0
    current_consec = 0
    for t in sorted_trades:
        if t["pnl"] <= 0:
            current_consec += 1
            if current_consec > max_consec:
                max_consec = current_consec
        else:
            current_consec = 0

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
    worst_month = min(monthly_values) if monthly_values else 0.0

    # Worst single trade
    worst_trade = min(pnls) if pnls else 0.0

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

    return {
        "label": label,
        "trades": n,
        "pnl": total_pnl,
        "profit_factor": pf,
        "win_rate": wr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "max_consec_losers": max_consec,
        "winning_months": winning_months,
        "losing_months": losing_months,
        "total_months": total_months,
        "worst_month": worst_month,
        "worst_trade": worst_trade,
    }


# ═══════════════════════════════════════════════════════════════
#  RUN ALL STRATEGIES (returns raw trade objects per strategy)
# ═══════════════════════════════════════════════════════════════

def run_all_strategies_raw(rf, mode_label):
    """Run all 8 strategies and return raw trade objects per strategy.

    For UNFILTERED mode: pass rf=None to strategies that use internal
    regime_filter (Catalyst, Cascade, Fortress), and don't pass blocked
    regimes to Undertow.

    For FILTERED mode: use post-hoc filtering with the blocking rules.

    This function always runs UNFILTERED — filtering is applied afterward.
    """
    raw_trades = {}

    # ── 1. Undertow (ES) ──
    print(f"  [{mode_label}] 1/8 Undertow...", end="", flush=True)
    try:
        trades = run_amt_tema(regime_filter=None, blocked_regimes=None)
        raw_trades["Undertow"] = trades
        pnl = sum(getattr(t, 'pnl_dollar', 0) or 0 for t in trades)
        print(f" {len(trades)} trades, ${pnl:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw_trades["Undertow"] = []

    # ── 2. Sentinel (ES, always unfiltered) ──
    print(f"  [{mode_label}] 2/8 Sentinel...", end="", flush=True)
    try:
        trades = run_lvl_v13()
        raw_trades["Sentinel"] = trades
        pnl = sum(getattr(t, 'pnl_dollar', 0) or 0 for t in trades)
        print(f" {len(trades)} trades, ${pnl:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw_trades["Sentinel"] = []

    # ── 3. Nexus (ES, no internal regime filter) ──
    print(f"  [{mode_label}] 3/8 Nexus...", end="", flush=True)
    try:
        trades = run_ms_os()
        raw_trades["Nexus"] = trades
        pnl = sum(getattr(t, 'pnl_dollar', 0) or 0 for t in trades)
        print(f" {len(trades)} trades, ${pnl:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw_trades["Nexus"] = []

    # ── 4. Catalyst (SPX, no internal regime filter) ──
    print(f"  [{mode_label}] 4/8 Catalyst...", end="", flush=True)
    try:
        trades = run_home_run(regime_filter=None)
        raw_trades["Catalyst"] = trades
        pnl = sum(getattr(t, 'net_pnl', 0) or 0 for t in trades)
        print(f" {len(trades)} trades, ${pnl:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw_trades["Catalyst"] = []

    # ── 5. Cascade (SPX, no internal regime filter) ──
    print(f"  [{mode_label}] 5/8 Cascade...", end="", flush=True)
    try:
        trades = run_bear_breakdown(regime_filter=None)
        raw_trades["Cascade"] = trades
        pnl = sum(getattr(t, 'net_pnl', 0) or 0 for t in trades)
        print(f" {len(trades)} trades, ${pnl:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw_trades["Cascade"] = []

    # ── 6. Fortress (SPX, no internal regime filter) ──
    print(f"  [{mode_label}] 6/8 Fortress...", end="", flush=True)
    try:
        trades = run_bull_credit(regime_filter=None)
        raw_trades["Fortress"] = trades
        pnl = sum(getattr(t, 'net_pnl', 0) or 0 for t in trades)
        print(f" {len(trades)} trades, ${pnl:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw_trades["Fortress"] = []

    # ── 7. Crossfire (SPX, no internal regime filter) ──
    print(f"  [{mode_label}] 7/8 Crossfire...", end="", flush=True)
    try:
        trades = run_ema_cross()
        raw_trades["Crossfire"] = trades
        pnl = sum(getattr(t, 'net_pnl', 0) or 0 for t in trades)
        print(f" {len(trades)} trades, ${pnl:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw_trades["Crossfire"] = []

    # ── 8. Vector (SPX, no internal regime filter) ──
    print(f"  [{mode_label}] 8/8 Vector...", end="", flush=True)
    try:
        trades = run_ema_cross_dir()
        raw_trades["Vector"] = trades
        pnl = sum(getattr(t, 'net_pnl', 0) or 0 for t in trades)
        print(f" {len(trades)} trades, ${pnl:+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        raw_trades["Vector"] = []

    return raw_trades


# ═══════════════════════════════════════════════════════════════
#  APPLY FILTERING + NORMALIZE
# ═══════════════════════════════════════════════════════════════

def apply_filter_and_normalize(raw_trades, rf, blocking_rules):
    """Apply post-hoc regime blocking and normalize trades.

    If blocking_rules is None, no filtering is applied (unfiltered run).
    """
    per_strategy = {}
    all_normalized = []

    STRATEGY_ORDER = [
        "Catalyst", "Cascade", "Fortress", "Vector",
        "Undertow", "Crossfire", "Sentinel", "Nexus",
    ]

    for name in STRATEGY_ORDER:
        trades_raw = raw_trades.get(name, [])

        if blocking_rules is not None:
            blocked = blocking_rules[name]["blocked"]
            trades_raw = post_hoc_filter(trades_raw, rf, blocked)

        normalized = [normalize_trade(t, name) for t in trades_raw]
        per_strategy[name] = normalized
        all_normalized.extend(normalized)

    return per_strategy, all_normalized


# ═══════════════════════════════════════════════════════════════
#  FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════

def fmt_d(v):
    """Format dollar value."""
    if v >= 0:
        return f"${v:,.0f}"
    return f"-${abs(v):,.0f}"


def fmt_dd(v):
    """Format delta dollar value with sign."""
    if v >= 0:
        return f"+${v:,.0f}"
    return f"-${abs(v):,.0f}"


def fmt_pf(v):
    if v == 0:
        return "0.00"
    if v >= 100 or v == float("inf"):
        return "inf"
    return f"{v:.2f}"


# ═══════════════════════════════════════════════════════════════
#  PRINT REPORT
# ═══════════════════════════════════════════════════════════════

STRATEGY_ORDER = [
    "Catalyst", "Cascade", "Fortress", "Vector",
    "Undertow", "Crossfire", "Sentinel", "Nexus",
]


def print_report(unf_stats, flt_stats, unf_port, flt_port):
    """Print the full comparison report."""

    w = 100
    print()
    print("=" * w)
    print("  WALK-FORWARD PORTFOLIO: UNFILTERED vs FILTERED")
    print("  (5-state WF regime labels, original March 2 blocking rules)")
    print("=" * w)

    # ── Summary table ──
    print(f"\n  STRATEGY COMPARISON:")
    print(f"  {'':16s}  {'--- UNFILTERED ---':^28s}  {'--- FILTERED ---':^28s}")
    print(f"  {'Strategy':<16s} {'Trades':>6s} {'P&L':>11s} {'Max DD':>10s}  "
          f"{'Trades':>6s} {'P&L':>11s} {'Max DD':>10s}  {'DD Delta':>11s}")
    print(f"  {'-' * (w - 4)}")

    for name in STRATEGY_ORDER:
        u = unf_stats[name]
        f = flt_stats[name]
        dd_delta = f["max_dd"] - u["max_dd"]
        # Negative dd_delta = filter REDUCES drawdown (good)
        dd_sign = "-" if dd_delta < 0 else "+"
        print(f"  {name:<16s} {u['trades']:>6d} {fmt_d(u['pnl']):>11s} {fmt_d(u['max_dd']):>10s}  "
              f"{f['trades']:>6d} {fmt_d(f['pnl']):>11s} {fmt_d(f['max_dd']):>10s}  "
              f"{fmt_dd(-dd_delta) if dd_delta <= 0 else fmt_dd(-dd_delta):>11s}")

    print(f"  {'-' * (w - 4)}")
    dd_delta_port = flt_port["max_dd"] - unf_port["max_dd"]
    print(f"  {'PORTFOLIO':<16s} {unf_port['trades']:>6d} {fmt_d(unf_port['pnl']):>11s} "
          f"{fmt_d(unf_port['max_dd']):>10s}  "
          f"{flt_port['trades']:>6d} {fmt_d(flt_port['pnl']):>11s} "
          f"{fmt_d(flt_port['max_dd']):>10s}  "
          f"{fmt_dd(-dd_delta_port):>11s}")

    # ── Detailed per-strategy blocks ──
    print(f"\n\n{'=' * w}")
    print(f"  DETAILED PER-STRATEGY BREAKDOWN")
    print(f"{'=' * w}")

    for name in STRATEGY_ORDER:
        u = unf_stats[name]
        f = flt_stats[name]

        print(f"\n  {name.upper()}")
        print(f"  {'─' * 70}")
        print(f"  {'':24s} {'UNFILTERED':>14s} {'FILTERED':>14s} {'Delta':>14s}")

        rows = [
            ("Trades:",
             f"{u['trades']}",
             f"{f['trades']}",
             f"{f['trades'] - u['trades']:+d}"),
            ("P&L:",
             fmt_d(u['pnl']),
             fmt_d(f['pnl']),
             fmt_dd(f['pnl'] - u['pnl'])),
            ("Profit Factor:",
             fmt_pf(u['profit_factor']),
             fmt_pf(f['profit_factor']),
             ""),
            ("Win Rate:",
             f"{u['win_rate']:.1f}%",
             f"{f['win_rate']:.1f}%",
             f"{f['win_rate'] - u['win_rate']:+.1f}%"),
            ("Sharpe:",
             f"{u['sharpe']:.2f}",
             f"{f['sharpe']:.2f}",
             f"{f['sharpe'] - u['sharpe']:+.2f}"),
            ("MAX DRAWDOWN:",
             fmt_d(u['max_dd']),
             fmt_d(f['max_dd']),
             fmt_dd(-(f['max_dd'] - u['max_dd']))),
            ("Max Consec Losers:",
             f"{u['max_consec_losers']}",
             f"{f['max_consec_losers']}",
             f"{f['max_consec_losers'] - u['max_consec_losers']:+d}"),
            ("Win/Lose Months:",
             f"{u['winning_months']}W/{u['losing_months']}L",
             f"{f['winning_months']}W/{f['losing_months']}L",
             ""),
            ("Worst Month:",
             fmt_d(u['worst_month']),
             fmt_d(f['worst_month']),
             ""),
            ("Worst Trade:",
             fmt_d(u['worst_trade']),
             fmt_d(f['worst_trade']),
             ""),
        ]

        for lbl, uv, fv, dv in rows:
            # Highlight the MAX DRAWDOWN row
            marker = "  <-- KEY" if lbl == "MAX DRAWDOWN:" else ""
            print(f"  {lbl:24s} {uv:>14s} {fv:>14s} {dv:>14s}{marker}")

    # ── Portfolio detail ──
    print(f"\n\n{'=' * w}")
    print(f"  PORTFOLIO TOTALS")
    print(f"{'=' * w}")
    u = unf_port
    f = flt_port
    print(f"  {'':24s} {'UNFILTERED':>14s} {'FILTERED':>14s} {'Delta':>14s}")
    port_rows = [
        ("Trades:", f"{u['trades']}", f"{f['trades']}", f"{f['trades'] - u['trades']:+d}"),
        ("P&L:", fmt_d(u['pnl']), fmt_d(f['pnl']), fmt_dd(f['pnl'] - u['pnl'])),
        ("Profit Factor:", fmt_pf(u['profit_factor']), fmt_pf(f['profit_factor']), ""),
        ("Win Rate:", f"{u['win_rate']:.1f}%", f"{f['win_rate']:.1f}%",
         f"{f['win_rate'] - u['win_rate']:+.1f}%"),
        ("Sharpe:", f"{u['sharpe']:.2f}", f"{f['sharpe']:.2f}",
         f"{f['sharpe'] - u['sharpe']:+.2f}"),
        ("MAX DRAWDOWN:", fmt_d(u['max_dd']), fmt_d(f['max_dd']),
         fmt_dd(-(f['max_dd'] - u['max_dd']))),
        ("Max Consec Losers:", f"{u['max_consec_losers']}", f"{f['max_consec_losers']}",
         f"{f['max_consec_losers'] - u['max_consec_losers']:+d}"),
        ("Win/Lose Months:", f"{u['winning_months']}W/{u['losing_months']}L",
         f"{f['winning_months']}W/{f['losing_months']}L", ""),
        ("Worst Month:", fmt_d(u['worst_month']), fmt_d(f['worst_month']), ""),
        ("Worst Trade:", fmt_d(u['worst_trade']), fmt_d(f['worst_trade']), ""),
    ]
    for lbl, uv, fv, dv in port_rows:
        marker = "  <-- KEY" if lbl == "MAX DRAWDOWN:" else ""
        print(f"  {lbl:24s} {uv:>14s} {fv:>14s} {dv:>14s}{marker}")

    # ── Verdict ──
    print(f"\n\n{'=' * w}")
    print(f"  VERDICT")
    print(f"{'=' * w}")

    reduces = []
    minimal = []
    increases = []
    for name in STRATEGY_ORDER:
        u_dd = unf_stats[name]["max_dd"]
        f_dd = flt_stats[name]["max_dd"]
        if u_dd == 0 and f_dd == 0:
            minimal.append(name)
            continue
        reduction = u_dd - f_dd
        pct = (reduction / u_dd * 100) if u_dd > 0 else 0
        if reduction > 500:  # meaningful reduction
            reduces.append(f"{name} ({fmt_d(u_dd)} -> {fmt_d(f_dd)}, {pct:+.0f}%)")
        elif reduction < -500:  # filter increases DD
            increases.append(f"{name} ({fmt_d(u_dd)} -> {fmt_d(f_dd)}, {pct:+.0f}%)")
        else:
            minimal.append(name)

    print(f"\n  Filter REDUCES drawdown significantly:")
    if reduces:
        for r in reduces:
            print(f"    - {r}")
    else:
        print(f"    (none)")

    print(f"\n  Filter has MINIMAL DD impact:")
    if minimal:
        for m in minimal:
            print(f"    - {m}")
    else:
        print(f"    (none)")

    print(f"\n  Filter INCREASES drawdown (loses profitable trades):")
    if increases:
        for i in increases:
            print(f"    - {i}")
    else:
        print(f"    (none)")

    port_reduction = unf_port["max_dd"] - flt_port["max_dd"]
    port_pct = (port_reduction / unf_port["max_dd"] * 100) if unf_port["max_dd"] > 0 else 0
    print(f"\n  Portfolio DD: unfiltered {fmt_d(unf_port['max_dd'])} vs "
          f"filtered {fmt_d(flt_port['max_dd'])} "
          f"({fmt_dd(port_reduction)} reduction, {port_pct:+.0f}%)")

    pnl_cost = unf_port["pnl"] - flt_port["pnl"]
    print(f"  P&L cost of filtering: {fmt_dd(-pnl_cost)} "
          f"(unfiltered {fmt_d(unf_port['pnl'])} vs filtered {fmt_d(flt_port['pnl'])})")

    if flt_port["max_dd"] > 0 and unf_port["max_dd"] > 0:
        unf_return_dd = unf_port["pnl"] / unf_port["max_dd"]
        flt_return_dd = flt_port["pnl"] / flt_port["max_dd"]
        print(f"  Return/MaxDD ratio: unfiltered {unf_return_dd:.2f}x vs "
              f"filtered {flt_return_dd:.2f}x")

    print(f"\n{'=' * w}\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  MEDAL-LION FUND: UNFILTERED vs FILTERED (Walk-Forward Regimes)")
    print("  5-state WF labels, original March 2 blocking rules")
    print("=" * 80)

    # Load WF regime filter
    wf_path = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"
    if not wf_path.exists():
        print(f"ERROR: Walk-forward regimes not found at {wf_path}")
        sys.exit(1)

    _swap_path(MEDALLION_ROOT)
    from models.wf_regime_api import WalkForwardRegimeFilter
    rf = WalkForwardRegimeFilter(str(wf_path))

    # Run all 8 strategies once (unfiltered — no regime blocking at all)
    print(f"\n{'=' * 80}")
    print(f"  RUNNING ALL 8 STRATEGIES (UNFILTERED — raw trade generation)")
    print(f"{'=' * 80}")
    raw_trades = run_all_strategies_raw(rf, "RAW")

    # Build UNFILTERED results (no post-hoc filter)
    print(f"\n  Building UNFILTERED results (all trades kept)...")
    unf_per_strat, unf_all = apply_filter_and_normalize(raw_trades, rf, None)
    unf_stats = {}
    for name in STRATEGY_ORDER:
        unf_stats[name] = compute_full_stats(unf_per_strat[name], name)
    unf_port = compute_full_stats(unf_all, "PORTFOLIO")

    # Build FILTERED results (post-hoc regime blocking)
    print(f"  Building FILTERED results (original March 2 blocking rules)...")
    flt_per_strat, flt_all = apply_filter_and_normalize(raw_trades, rf, FILTERED_RULES)
    flt_stats = {}
    for name in STRATEGY_ORDER:
        flt_stats[name] = compute_full_stats(flt_per_strat[name], name)
    flt_port = compute_full_stats(flt_all, "PORTFOLIO")

    # Print the comparison report
    print_report(unf_stats, flt_stats, unf_port, flt_port)


if __name__ == "__main__":
    main()
