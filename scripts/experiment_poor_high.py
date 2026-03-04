#!/usr/bin/env python3
"""Experiment: Poor High/Low Classification for ONH Level Rejection.

Hypothesis: When the overnight high (ONH) was formed by many bars near the
high ("poor high" in Market Profile terms), the level is weak and likely to
break on revisit. Skipping ONH rejections on poor-high days should improve PF.

Conversely, when ONH was formed by a single sharp spike ("excess"), the level
is strong and more likely to hold — our rejection trade should work better.

Classification:
  - Excess: <= 2 overnight bars within 3 pts of ONH (sharp spike, strong)
  - Poor:   >= 5 overnight bars within 3 pts of ONH (multiple touches, weak)
  - Normal: 3-4 bars near ONH

Tests v13 (ONH Level Rejection, 3x C3) with poor-high filter.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
from backtester.session import SessionState, update_session
from backtester.indicators import compute_indicators

DATA = "data/es_5m_databento_2yr.csv"
SPLIT_DATE = pd.Timestamp("2025-02-14", tz="America/New_York")


def v13_config():
    """v13 ONH Level Rejection config (baseline)."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_ib_break = True
    cfg.use_trend_filter = True
    cfg.use_ib_reject = True
    cfg.rej_wide_only = True
    cfg.rej_target = "ib_low"
    cfg.rej_stop_buffer = 8.0
    cfg.max_rej_trades = 8
    cfg.use_level_reject = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = False
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 7.0
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_rr = 0.5
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_target_skip = 2
    return cfg


def print_metrics(label, m, trades):
    pnl = sum(t.pnl_dollar for t in trades)
    print(f"  {label}: {m.total_trades:>4} trades, PF {m.profit_factor:>5.2f}, "
          f"WR {m.win_rate:>5.1f}%, ${pnl:>+10,.0f}")


def analyze_onh_quality(df, cfg):
    """Run bar-by-bar to collect ONH quality stats per session day."""
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                      tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                      atr_avg_len=cfg.atr_avg_len)
    state = SessionState()
    prev_bar = None
    daily_stats = []

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx

        old_frozen = state.on_frozen
        update_session(state, bar, prev_bar, cfg)

        # Capture on RTH open (transition to frozen)
        if bar["new_rth"]:
            daily_stats.append({
                "date": idx,
                "on_high": state.on_high,
                "bars_near_high": state.on_bars_near_high,
                "total_on_bars": state.on_high_bar_count,
                "is_poor": state.on_high_is_poor,
                "is_excess": state.on_high_is_excess,
            })

        prev_bar = bar

    return daily_stats


def main():
    print("Loading ES data...")
    df = load_tos_csv(DATA, instrument="ES")
    print(f"Loaded {len(df)} bars")

    cfg = v13_config()

    # ── Step 1: Analyze ONH quality distribution ──
    print("\nAnalyzing overnight high quality...")
    stats = analyze_onh_quality(df.copy(), cfg)
    print(f"Total trading days: {len(stats)}")

    excess_days = sum(1 for s in stats if s["is_excess"])
    poor_days = sum(1 for s in stats if s["is_poor"])
    normal_days = len(stats) - excess_days - poor_days
    print(f"  Excess (strong, <= 2 bars near high): {excess_days} days ({excess_days/len(stats)*100:.1f}%)")
    print(f"  Normal (3-4 bars near high):          {normal_days} days ({normal_days/len(stats)*100:.1f}%)")
    print(f"  Poor (weak, >= 5 bars near high):     {poor_days} days ({poor_days/len(stats)*100:.1f}%)")

    # Distribution of bars_near_high
    from collections import Counter
    bar_counts = Counter(s["bars_near_high"] for s in stats)
    print(f"\n  Bars-near-high distribution:")
    for count in sorted(bar_counts.keys()):
        print(f"    {count:>3} bars: {bar_counts[count]:>4} days ({bar_counts[count]/len(stats)*100:.1f}%)")

    # ── Step 2: Run backtest with post-hoc tagging ──
    print(f"\n{'='*70}")
    print(f"  v13 LVL Trades by ONH Quality")
    print(f"{'='*70}")

    # Run baseline and tag trades with ONH quality
    # We need to run the backtest and capture session state per trade
    # Since we can't easily tag during the backtest, let's build a date-indexed lookup
    quality_lookup = {}
    for s in stats:
        date = s["date"].date() if hasattr(s["date"], "date") else s["date"]
        quality_lookup[date] = s

    trades_all = run_backtest(df.copy(), cfg)
    lvl_trades = [t for t in trades_all if t.setup.startswith("LVL")]

    buckets = {"excess": [], "normal": [], "poor": [], "unknown": []}
    for t in lvl_trades:
        trade_date = t.entry_time.date() if hasattr(t.entry_time, "date") else None
        if trade_date and trade_date in quality_lookup:
            q = quality_lookup[trade_date]
            if q["is_excess"]:
                buckets["excess"].append(t)
            elif q["is_poor"]:
                buckets["poor"].append(t)
            else:
                buckets["normal"].append(t)
        else:
            buckets["unknown"].append(t)

    print(f"\n  LVL trades split by ONH quality (full 2yr):")
    print(f"  {'Quality':<12} {'Trades':>7} {'PF':>7} {'WR':>7} {'P&L':>12}")
    print(f"  {'-'*48}")

    for label in ["excess", "normal", "poor"]:
        trades = buckets[label]
        if len(trades) >= 5:
            m = compute_metrics(trades)
            pnl = sum(t.pnl_dollar for t in trades)
            print(f"  {label:<12} {len(trades):>7} {m.profit_factor:>7.2f} {m.win_rate:>6.1f}% ${pnl:>+10,.0f}")
        else:
            print(f"  {label:<12} {len(trades):>7} trades (too few)")

    if buckets["unknown"]:
        print(f"  {'unknown':<12} {len(buckets['unknown']):>7} trades (no quality data)")

    # ── Step 3: Skip poor high filter ──
    print(f"\n{'='*70}")
    print(f"  v13 Baseline vs Skip-Poor-High")
    print(f"{'='*70}")

    cfg_filter = v13_config()
    cfg_filter.lvl_skip_poor_high = True

    trades_base = run_backtest(df.copy(), cfg)
    trades_filter = run_backtest(df.copy(), cfg_filter)

    m_base = compute_metrics(trades_base)
    m_filter = compute_metrics(trades_filter)

    print(f"\n  --- Full 2-Year ---")
    print_metrics("Baseline         ", m_base, trades_base)
    print_metrics("Skip poor ONH    ", m_filter, trades_filter)

    lvl_base = [t for t in trades_base if t.setup.startswith("LVL")]
    lvl_filter = [t for t in trades_filter if t.setup.startswith("LVL")]
    if lvl_base:
        m_lb = compute_metrics(lvl_base)
        pnl_lb = sum(t.pnl_dollar for t in lvl_base)
        print(f"\n  LVL only:")
        print(f"    Baseline:       {len(lvl_base):>4} trades, PF {m_lb.profit_factor:.2f}, ${pnl_lb:>+10,.0f}")
    if lvl_filter:
        m_lf = compute_metrics(lvl_filter)
        pnl_lf = sum(t.pnl_dollar for t in lvl_filter)
        print(f"    Skip poor:      {len(lvl_filter):>4} trades, PF {m_lf.profit_factor:.2f}, ${pnl_lf:>+10,.0f}")

    # ── Step 4: IS/OOS ──
    print(f"\n  --- IS/OOS Split ---")
    df_is = df[df.index < SPLIT_DATE].copy()
    df_oos = df[df.index >= SPLIT_DATE].copy()

    for period_label, period_df in [("IS ", df_is), ("OOS", df_oos)]:
        t_b = run_backtest(period_df.copy(), cfg)
        t_f = run_backtest(period_df.copy(), cfg_filter)
        m_b = compute_metrics(t_b) if t_b else None
        m_f = compute_metrics(t_f) if t_f else None
        if m_b:
            pnl_b = sum(t.pnl_dollar for t in t_b)
            print(f"    {period_label} Baseline:        {m_b.total_trades:>4} trades, PF {m_b.profit_factor:.2f}, ${pnl_b:>+10,.0f}")
        if m_f:
            pnl_f = sum(t.pnl_dollar for t in t_f)
            print(f"    {period_label} Skip poor:       {m_f.total_trades:>4} trades, PF {m_f.profit_factor:.2f}, ${pnl_f:>+10,.0f}")

    # ── Step 5: Excess-only filter (opposite — only trade on strong highs) ──
    print(f"\n{'='*70}")
    print(f"  Bonus: Excess-Only ONH (trade ONLY when ONH is strong)")
    print(f"{'='*70}")

    # This is just the post-hoc analysis from Step 2, but broken out by IS/OOS
    for period_label, period_df in [("Full", df), ("IS ", df_is), ("OOS", df_oos)]:
        t_all = run_backtest(period_df.copy(), cfg)
        lvl_t = [t for t in t_all if t.setup.startswith("LVL")]
        excess_t = [t for t in lvl_t
                     if hasattr(t.entry_time, "date") and t.entry_time.date() in quality_lookup
                     and quality_lookup[t.entry_time.date()]["is_excess"]]
        if len(excess_t) >= 3:
            m = compute_metrics(excess_t)
            pnl = sum(t.pnl_dollar for t in excess_t)
            print(f"    {period_label} Excess-only LVL: {len(excess_t):>4} trades, PF {m.profit_factor:.2f}, ${pnl:>+10,.0f}")
        else:
            print(f"    {period_label} Excess-only LVL: {len(excess_t):>4} trades (too few)")

    print(f"\n{'='*70}")
    print("  Poor High Experiment Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
