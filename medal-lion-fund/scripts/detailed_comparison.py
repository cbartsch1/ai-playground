#!/usr/bin/env python3
"""Medal-Lion Fund — DETAILED Per-Strategy Comparison: OLD vs NEW Regime Model.

Runs both HMM models (old 3feat/7state, new 5feat/5state) across all 8 strategies
and produces a full per-strategy breakdown including:
  - Trades, P&L, Profit Factor, Win Rate, Sharpe, Max Drawdown
  - Winning/Losing months, Monthly profitability rate
  - Average trade P&L, Best/Worst month, Trades per day

Usage:
    cd ~/projects/ai-playground/medal-lion-fund
    .venv/bin/python scripts/detailed_comparison.py 2>&1
"""

import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

# ── Import everything from the existing comparison script ──
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from regime_model_comparison import (
    build_old_model,
    build_new_model,
    create_regime_filter,
    analyze_regime_distribution,
    run_amt_tema,
    run_lvl_v13,
    run_ms_os,
    run_home_run,
    run_bear_breakdown,
    run_bull_credit,
    run_ema_cross,
    run_ema_cross_dir,
    normalize_trade,
    _post_hoc_filter,
    _post_hoc_filter_per_side,
    get_blocked_set,
    OLD_BLOCKING,
    NEW_BLOCKING,
    _all_7state_labels,
    _all_5state_labels,
)


# ═══════════════════════════════════════════════════════════════
#  DETAILED STATS — computes everything the user wants
# ═══════════════════════════════════════════════════════════════

def detailed_stats(trades_normalized, label):
    """Compute full detailed stats from normalized trade list.

    Returns dict with:
      trades, pnl, profit_factor, win_rate, sharpe, max_dd,
      winning_months, losing_months, total_months, monthly_wr,
      avg_trade, best_month, worst_month, trades_per_day,
      monthly_pnl (dict)
    """
    empty = {
        "label": label,
        "trades": 0,
        "pnl": 0.0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "max_dd": 0.0,
        "winning_months": 0,
        "losing_months": 0,
        "total_months": 0,
        "monthly_wr": 0.0,
        "avg_trade": 0.0,
        "best_month": 0.0,
        "worst_month": 0.0,
        "trades_per_day": 0.0,
        "monthly_pnl": {},
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

    # Sort by exit time
    sorted_trades = sorted(trades_normalized, key=lambda t: t["exit_time"])

    # ── Equity curve + Max Drawdown ──
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

    # ── Monthly P&L ──
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

    # ── Sharpe from daily P&L ──
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

    # ── Trades per day ──
    trading_days = set()
    for t in sorted_trades:
        et = t["exit_time"]
        if hasattr(et, "date"):
            trading_days.add(et.date())
        elif hasattr(et, "strftime"):
            trading_days.add(et.strftime("%Y-%m-%d"))
    # Use span of data (first exit to last exit) to get total trading days
    if len(sorted_trades) >= 2:
        first_exit = sorted_trades[0]["exit_time"]
        last_exit = sorted_trades[-1]["exit_time"]
        if hasattr(first_exit, "date") and hasattr(last_exit, "date"):
            span_days = (last_exit.date() - first_exit.date()).days
            # Approximate trading days as 252/365 * calendar days
            approx_trading_days = max(1, int(span_days * 252 / 365))
        else:
            approx_trading_days = max(1, len(trading_days))
    else:
        approx_trading_days = max(1, len(trading_days))
    trades_per_day = n / approx_trading_days if approx_trading_days > 0 else 0.0

    return {
        "label": label,
        "trades": n,
        "pnl": total_pnl,
        "profit_factor": pf,
        "win_rate": wr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "winning_months": winning_months,
        "losing_months": losing_months,
        "total_months": total_months,
        "monthly_wr": monthly_wr,
        "avg_trade": avg_trade,
        "best_month": best_month,
        "worst_month": worst_month,
        "trades_per_day": trades_per_day,
        "monthly_pnl": dict(monthly_pnl),
    }


# ═══════════════════════════════════════════════════════════════
#  RUN PORTFOLIO — returns normalized trades per strategy
# ═══════════════════════════════════════════════════════════════

STRATEGY_ORDER = [
    "Undertow",
    "Sentinel",
    "Nexus",
    "Catalyst",
    "Cascade",
    "Fortress",
    "Crossfire",
    "Vector",
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


def run_portfolio_detailed(rf, blocking_rules, all_labels, model_label):
    """Run all 8 strategies, return per-strategy normalized trade lists."""
    strat_trades = {}

    # ── 1. Undertow ──
    print(f"\n  [{model_label}] 1/8 Undertow...", end="", flush=True)
    blocked = get_blocked_set("Undertow", blocking_rules, all_labels)
    try:
        trades, _ = run_amt_tema(regime_filter=rf, blocked_regimes=blocked)
        normalized = [normalize_trade(t, "Undertow") for t in trades]
        strat_trades["Undertow"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        strat_trades["Undertow"] = []

    # ── 2. Sentinel (unfiltered) ──
    print(f"  [{model_label}] 2/8 Sentinel...", end="", flush=True)
    try:
        lvl_trades = run_lvl_v13()
        normalized = [normalize_trade(t, "Sentinel") for t in lvl_trades]
        strat_trades["Sentinel"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        strat_trades["Sentinel"] = []

    # ── 3. Nexus ──
    print(f"  [{model_label}] 3/8 Nexus...", end="", flush=True)
    blocked = get_blocked_set("Nexus", blocking_rules, all_labels)
    try:
        ms_trades, _ = run_ms_os(regime_filter=rf)
        ms_trades = _post_hoc_filter(ms_trades, rf, blocked)
        normalized = [normalize_trade(t, "Nexus") for t in ms_trades]
        strat_trades["Nexus"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        strat_trades["Nexus"] = []

    # ── 4. Catalyst ──
    print(f"  [{model_label}] 4/8 Catalyst...", end="", flush=True)
    try:
        trades, _ = run_home_run(regime_filter=rf)
        blocked = get_blocked_set("Catalyst", blocking_rules, all_labels)
        trades = _post_hoc_filter(trades, rf, blocked)
        normalized = [normalize_trade(t, "Catalyst") for t in trades]
        strat_trades["Catalyst"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        strat_trades["Catalyst"] = []

    # ── 5. Cascade ──
    print(f"  [{model_label}] 5/8 Cascade...", end="", flush=True)
    try:
        trades, _ = run_bear_breakdown(regime_filter=rf)
        blocked = get_blocked_set("Cascade", blocking_rules, all_labels)
        trades = _post_hoc_filter(trades, rf, blocked)
        normalized = [normalize_trade(t, "Cascade") for t in trades]
        strat_trades["Cascade"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        strat_trades["Cascade"] = []

    # ── 6. Fortress ──
    print(f"  [{model_label}] 6/8 Fortress...", end="", flush=True)
    try:
        trades, _ = run_bull_credit(regime_filter=rf)
        blocked = get_blocked_set("Fortress", blocking_rules, all_labels)
        trades = _post_hoc_filter(trades, rf, blocked)
        normalized = [normalize_trade(t, "Fortress") for t in trades]
        strat_trades["Fortress"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        strat_trades["Fortress"] = []

    # ── 7. Crossfire ──
    print(f"  [{model_label}] 7/8 Crossfire...", end="", flush=True)
    try:
        ema_trades = run_ema_cross(regime_filter=rf)
        rules = blocking_rules["Crossfire"]
        blocked_call = rules.get("blocked_call", set())
        blocked_put = rules.get("blocked_put", set())
        ema_trades = _post_hoc_filter_per_side(ema_trades, rf, blocked_call, blocked_put)
        normalized = [normalize_trade(t, "Crossfire") for t in ema_trades]
        strat_trades["Crossfire"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        strat_trades["Crossfire"] = []

    # ── 8. Vector ──
    print(f"  [{model_label}] 8/8 Vector...", end="", flush=True)
    try:
        ema_dir_trades = run_ema_cross_dir(regime_filter=rf)
        rules = blocking_rules["Vector"]
        blocked_call = rules.get("blocked_call", set())
        ema_dir_trades = _post_hoc_filter(ema_dir_trades, rf, blocked_call)
        normalized = [normalize_trade(t, "Vector") for t in ema_dir_trades]
        strat_trades["Vector"] = normalized
        print(f" {len(normalized)} trades, ${sum(t['pnl'] for t in normalized):+,.0f}")
    except Exception as e:
        print(f" FAILED: {e}")
        strat_trades["Vector"] = []

    return strat_trades


# ═══════════════════════════════════════════════════════════════
#  FORMATTED OUTPUT
# ═══════════════════════════════════════════════════════════════

def fmt_dollars(v):
    """Format dollar value with sign."""
    if v >= 0:
        return f"${v:,.0f}"
    return f"-${abs(v):,.0f}"


def fmt_dollars_delta(v):
    """Format delta dollar value with explicit sign."""
    if v >= 0:
        return f"+${v:,.0f}"
    return f"-${abs(v):,.0f}"


def fmt_pf(v):
    """Format profit factor."""
    if v == 0:
        return "0.00"
    if v >= 100 or v == float("inf"):
        return "inf"
    return f"{v:.2f}"


def fmt_pct(v):
    """Format percentage."""
    return f"{v:.1f}%"


def print_strategy_block(name, desc, old, new):
    """Print one strategy's detailed comparison block."""
    print(f"\n{name.upper()} ({desc})")
    print(f"{'─' * 60}")

    rows = [
        ("Trades:",      f"{old['trades']}",              f"{new['trades']}",              f"{new['trades'] - old['trades']:+d}"),
        ("P&L:",         fmt_dollars(old['pnl']),         fmt_dollars(new['pnl']),         fmt_dollars_delta(new['pnl'] - old['pnl'])),
        ("Profit Factor:", fmt_pf(old['profit_factor']),  fmt_pf(new['profit_factor']),    ""),
        ("Win Rate:",    fmt_pct(old['win_rate']),        fmt_pct(new['win_rate']),        f"{new['win_rate'] - old['win_rate']:+.1f}%"),
        ("Sharpe:",      f"{old['sharpe']:.2f}",         f"{new['sharpe']:.2f}",          f"{new['sharpe'] - old['sharpe']:+.2f}"),
        ("Max Drawdown:", fmt_dollars(old['max_dd']),     fmt_dollars(new['max_dd']),      fmt_dollars_delta(new['max_dd'] - old['max_dd'])),
        ("Win/Lose Mo:", f"{old['winning_months']}W / {old['losing_months']}L",
                         f"{new['winning_months']}W / {new['losing_months']}L",           ""),
        ("Monthly WR:",  fmt_pct(old['monthly_wr']),     fmt_pct(new['monthly_wr']),      f"{new['monthly_wr'] - old['monthly_wr']:+.1f}%"),
        ("Avg Trade:",   fmt_dollars(old['avg_trade']),   fmt_dollars(new['avg_trade']),   ""),
        ("Best Month:",  fmt_dollars(old['best_month']),  fmt_dollars(new['best_month']),  ""),
        ("Worst Month:", fmt_dollars(old['worst_month']), fmt_dollars(new['worst_month']), ""),
        ("Trades/Day:",  f"{old['trades_per_day']:.2f}",  f"{new['trades_per_day']:.2f}",  ""),
    ]

    print(f"  {'':20s} {'OLD':>14s}  {'NEW':>14s}  {'Delta':>14s}")
    for label, ov, nv, dv in rows:
        print(f"  {label:20s} {ov:>14s}  {nv:>14s}  {dv:>14s}")


def print_full_report(old_stats, new_stats, old_portfolio, new_portfolio, old_dist, new_dist):
    """Print the complete detailed comparison report."""

    print("\n")
    print("=" * 80)
    print("  DETAILED STRATEGY COMPARISON")
    print("  OLD: 3 features / 7 states    vs    NEW: 5 features / 5 states (3-bar smooth)")
    print("=" * 80)

    for name in STRATEGY_ORDER:
        desc = STRATEGY_DESC[name]
        old = old_stats.get(name, detailed_stats([], name))
        new = new_stats.get(name, detailed_stats([], name))
        print_strategy_block(name, desc, old, new)

    # ── Portfolio Total ──
    print(f"\n\n{'=' * 80}")
    print(f"  PORTFOLIO TOTAL")
    print(f"{'=' * 80}")
    print_strategy_block("PORTFOLIO", "All 8 Strategies Combined", old_portfolio, new_portfolio)

    # ── Regime Distribution ──
    print(f"\n\n{'=' * 80}")
    print(f"  REGIME DISTRIBUTION (trading days)")
    print(f"{'=' * 80}")

    print(f"\n  OLD (7-state, 3 features):")
    print(f"    Bearish: {old_dist['bearish_pct']:.1f}% ({old_dist['bearish_days']}d)  |  "
          f"Neutral: {old_dist['neutral_pct']:.1f}% ({old_dist['neutral_days']}d)  |  "
          f"Bullish: {old_dist['bullish_pct']:.1f}% ({old_dist['bullish_days']}d)")
    for label, d in sorted(old_dist["distribution"].items(), key=lambda x: -x[1]["pct"]):
        bar = "#" * int(d["pct"] / 2)
        print(f"    {label:<25s} {d['count']:>4d}d ({d['pct']:>5.1f}%) {bar}")

    print(f"\n  NEW (5-state, 5 features):")
    print(f"    Bearish: {new_dist['bearish_pct']:.1f}% ({new_dist['bearish_days']}d)  |  "
          f"Neutral: {new_dist['neutral_pct']:.1f}% ({new_dist['neutral_days']}d)  |  "
          f"Bullish: {new_dist['bullish_pct']:.1f}% ({new_dist['bullish_days']}d)")
    for label, d in sorted(new_dist["distribution"].items(), key=lambda x: -x[1]["pct"]):
        bar = "#" * int(d["pct"] / 2)
        print(f"    {label:<25s} {d['count']:>4d}d ({d['pct']:>5.1f}%) {bar}")

    # ── Winners / Losers Summary ──
    print(f"\n\n{'=' * 80}")
    print(f"  IMPACT SUMMARY")
    print(f"{'=' * 80}")

    impacts = []
    for name in STRATEGY_ORDER:
        old = old_stats.get(name, detailed_stats([], name))
        new = new_stats.get(name, detailed_stats([], name))
        delta_pnl = new["pnl"] - old["pnl"]
        delta_trades = new["trades"] - old["trades"]
        impacts.append((name, delta_pnl, delta_trades))

    impacts.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Strategy':<20s} {'P&L Delta':>14s}  {'Trade Delta':>12s}")
    print(f"  {'─' * 50}")
    for name, dpnl, dtrd in impacts:
        print(f"  {name:<20s} {fmt_dollars_delta(dpnl):>14s}  {dtrd:>+12d}")

    total_dpnl = new_portfolio["pnl"] - old_portfolio["pnl"]
    total_dtrd = new_portfolio["trades"] - old_portfolio["trades"]
    print(f"  {'─' * 50}")
    print(f"  {'TOTAL':<20s} {fmt_dollars_delta(total_dpnl):>14s}  {total_dtrd:>+12d}")

    print(f"\n{'=' * 80}\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  MEDAL-LION FUND: DETAILED OLD vs NEW REGIME MODEL COMPARISON")
    print("  OLD: 3 features (returns, range, vol_vol), 7 states")
    print("  NEW: 5 features (+ close_in_range, close_vs_open smoothed), 5 states")
    print("=" * 80)

    # ── Fit both models ──
    detector_old, full_df_old, features_old, spy_hourly = build_old_model()
    detector_new, full_df_new, features_new, _ = build_new_model()

    # Create regime filters
    rf_old = create_regime_filter(detector_old, full_df_old)
    rf_new = create_regime_filter(detector_new, full_df_new)

    # ── Regime distribution ──
    old_dist = analyze_regime_distribution(detector_old, features_old, "OLD")
    new_dist = analyze_regime_distribution(detector_new, features_new, "NEW")

    # ── Run portfolio with OLD model ──
    print("\n" + "=" * 80)
    print("  RUNNING ALL 8 STRATEGIES WITH OLD MODEL (3 feat / 7 state)")
    print("=" * 80)
    old_trades = run_portfolio_detailed(rf_old, OLD_BLOCKING, _all_7state_labels(), "OLD")

    # ── Run portfolio with NEW model ──
    print("\n" + "=" * 80)
    print("  RUNNING ALL 8 STRATEGIES WITH NEW MODEL (5 feat / 5 state)")
    print("=" * 80)
    new_trades = run_portfolio_detailed(rf_new, NEW_BLOCKING, _all_5state_labels(), "NEW")

    # ── Compute detailed stats for each strategy ──
    old_stats = {}
    new_stats = {}
    old_all_trades = []
    new_all_trades = []

    for name in STRATEGY_ORDER:
        old_stats[name] = detailed_stats(old_trades.get(name, []), name)
        new_stats[name] = detailed_stats(new_trades.get(name, []), name)
        old_all_trades.extend(old_trades.get(name, []))
        new_all_trades.extend(new_trades.get(name, []))

    old_portfolio = detailed_stats(old_all_trades, "PORTFOLIO")
    new_portfolio = detailed_stats(new_all_trades, "PORTFOLIO")

    # ── Print the full report ──
    print_full_report(old_stats, new_stats, old_portfolio, new_portfolio, old_dist, new_dist)


if __name__ == "__main__":
    main()
