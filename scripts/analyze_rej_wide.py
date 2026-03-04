#!/usr/bin/env python3
"""Analyze IB Rejection filtered to wide days only.

Tests multiple definitions of "wide" to find the clearest threshold.
Then runs full significance analysis on the best.
"""

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
from backtester.session import SessionState, update_session
from scipy import stats


def make_cfg():
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    return cfg


def get_ib_data_by_date(df, cfg):
    """Extract IB range and ratio for each trading session."""
    state = SessionState()
    prev_bar = None
    ib_by_date = {}

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)

        if state.ib_done and bar["is_rth"] and not bar.get("is_ib_period", True):
            date_key = idx.date()
            if date_key not in ib_by_date:
                ib_by_date[date_key] = {
                    "ib_range": state.ib_range,
                    "ib_ratio": state.ib_ratio,
                    "ib_high": state.ib_high,
                    "ib_low": state.ib_low,
                    "ib_range_avg": state.ib_range_avg,
                    "is_narrow": state.is_narrow_ib,
                    "is_wide": state.is_wide_ib,
                }
        prev_bar = bar

    return ib_by_date


def filter_trades_by_ib(trades, ib_by_date, min_ratio=None, min_range=None):
    """Filter trades to only those on days matching criteria."""
    filtered = []
    for t in trades:
        date_key = t.entry_time.date() if hasattr(t.entry_time, 'date') else None
        if date_key and date_key in ib_by_date:
            info = ib_by_date[date_key]
            if min_ratio is not None and info["ib_ratio"] < min_ratio:
                continue
            if min_range is not None and info["ib_range"] < min_range:
                continue
            filtered.append(t)
    return filtered


def significance_tests(trades, label):
    """Run t-test, permutation, bootstrap."""
    pnls = np.array([t.pnl_dollar for t in trades])
    n = len(pnls)
    if n < 5:
        print(f"  {label}: Too few trades ({n})")
        return

    t_stat, p_value = stats.ttest_1samp(pnls, 0)

    # Permutation
    observed = np.sum(pnls)
    np.random.seed(42)
    count_better = sum(1 for _ in range(10000)
                       if np.sum(np.abs(pnls) * np.random.choice([-1, 1], size=n)) >= observed)
    perm_p = count_better / 10000

    # Bootstrap
    boot_means = [np.mean(np.random.choice(pnls, size=n, replace=True)) for _ in range(10000)]
    boot_means = np.array(boot_means)
    p_profit = np.mean(boot_means > 0) * 100
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f"  {label}:")
    print(f"    t-test p={p_value:.4f}, permutation p={perm_p:.4f}, "
          f"bootstrap P(profit)={p_profit:.1f}%, CI [{ci_low:+.0f}, {ci_high:+.0f}]")
    stars = "***" if perm_p < 0.01 else "**" if perm_p < 0.05 else "*" if perm_p < 0.1 else ""
    if stars:
        print(f"    {stars}")


def main():
    print("Loading data...")
    df = load_tos_csv(sys.argv[1], instrument="ES")
    print(f"Loaded {len(df)} bars\n")

    cfg = make_cfg()
    trades = run_backtest(df.copy(), cfg)
    rej_trades = [t for t in trades if t.setup == "REJ"]
    ib_trades = [t for t in trades if t.setup == "IB"]
    ib_by_date = get_ib_data_by_date(df, cfg)

    # ── Part 1: Sweep "wide" definitions ──
    print(f"{'='*90}")
    print(f"  DEFINING 'WIDE' — Testing IB Ratio and IB Range thresholds")
    print(f"  (REJ trades only, filtered by day type)")
    print(f"{'='*90}")

    # IB Ratio thresholds (ratio = IB range / 20-day EMA of IB range)
    print(f"\n  --- By IB Ratio (IB Range / 20-day avg) ---")
    print(f"  {'Ratio >=':>10} {'REJ Trades':>10} {'WR%':>6} {'PF':>7} {'P&L':>11} {'Avg$':>8} "
          f"{'MaxDD':>9} {'Sharpe':>7} {'Days':>5}")
    print(f"  {'-'*80}")

    for ratio in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        filtered = filter_trades_by_ib(rej_trades, ib_by_date, min_ratio=ratio)
        if len(filtered) < 10:
            continue
        m = compute_metrics(filtered)
        days = len(set(t.entry_time.date() for t in filtered if hasattr(t.entry_time, 'date')))
        print(f"  {ratio:>10.1f} {m.total_trades:>10} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
              f"${m.net_pnl:>+10,.0f} ${m.avg_trade:>+7,.0f} ${m.max_drawdown:>8,.0f} "
              f"{m.sharpe:>7.2f} {days:>5}")

    # IB Range absolute thresholds
    print(f"\n  --- By Absolute IB Range (points) ---")
    print(f"  {'Range >=':>10} {'REJ Trades':>10} {'WR%':>6} {'PF':>7} {'P&L':>11} {'Avg$':>8} "
          f"{'MaxDD':>9} {'Sharpe':>7} {'Days':>5}")
    print(f"  {'-'*80}")

    for min_rng in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
        filtered = filter_trades_by_ib(rej_trades, ib_by_date, min_range=min_rng)
        if len(filtered) < 10:
            continue
        m = compute_metrics(filtered)
        days = len(set(t.entry_time.date() for t in filtered if hasattr(t.entry_time, 'date')))
        print(f"  {min_rng:>10} {m.total_trades:>10} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
              f"${m.net_pnl:>+10,.0f} ${m.avg_trade:>+7,.0f} ${m.max_drawdown:>8,.0f} "
              f"{m.sharpe:>7.2f} {days:>5}")

    # ── Part 2: IB Range as percentage of price (scales with ES level) ──
    print(f"\n  --- By IB Range as % of Price (basis points) ---")
    print(f"  {'Bps >=':>10} {'REJ Trades':>10} {'WR%':>6} {'PF':>7} {'P&L':>11} {'Avg$':>8} "
          f"{'MaxDD':>9} {'Sharpe':>7} {'Days':>5}")
    print(f"  {'-'*80}")

    # Add bps info to ib_by_date
    for date_key, info in ib_by_date.items():
        mid = (info["ib_high"] + info["ib_low"]) / 2 if info["ib_high"] > 0 else 1
        info["ib_range_bps"] = info["ib_range"] / mid * 10000

    for min_bps in [30, 40, 50, 60, 70, 80, 90, 100, 120, 150]:
        filtered = []
        for t in rej_trades:
            date_key = t.entry_time.date() if hasattr(t.entry_time, 'date') else None
            if date_key and date_key in ib_by_date:
                if ib_by_date[date_key]["ib_range_bps"] >= min_bps:
                    filtered.append(t)
        if len(filtered) < 10:
            continue
        m = compute_metrics(filtered)
        days = len(set(t.entry_time.date() for t in filtered if hasattr(t.entry_time, 'date')))
        print(f"  {min_bps:>10} {m.total_trades:>10} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
              f"${m.net_pnl:>+10,.0f} ${m.avg_trade:>+7,.0f} ${m.max_drawdown:>8,.0f} "
              f"{m.sharpe:>7.2f} {days:>5}")

    # ── Part 3: Pick best threshold, run full analysis ──
    # Use ratio >= 1.2 as the "wide" definition (matches Pine Script default)
    # Also test ratio >= 1.0 as a more inclusive alternative
    print(f"\n\n{'='*90}")
    print(f"  FULL ANALYSIS — BEST THRESHOLDS")
    print(f"{'='*90}")

    for label, min_ratio in [("IB Ratio >= 1.0 (above average)", 1.0),
                              ("IB Ratio >= 1.2 (wide — Pine default)", 1.2),
                              ("IB Ratio >= 1.4 (very wide)", 1.4)]:
        rej_filtered = filter_trades_by_ib(rej_trades, ib_by_date, min_ratio=min_ratio)
        ib_filtered = filter_trades_by_ib(ib_trades, ib_by_date, min_ratio=min_ratio)
        combined = rej_filtered + ib_filtered
        combined.sort(key=lambda t: t.entry_time)

        if len(rej_filtered) < 10:
            continue

        m_rej = compute_metrics(rej_filtered)
        m_all = compute_metrics(combined)

        print(f"\n  ── {label} ──")
        print(f"  REJ only: {m_rej.total_trades} trades, {m_rej.win_rate:.1f}% WR, "
              f"PF {m_rej.profit_factor:.3f}, P&L ${m_rej.net_pnl:+,.0f}, "
              f"DD ${m_rej.max_drawdown:,.0f}, Sharpe {m_rej.sharpe:.2f}")
        print(f"  Combined: {m_all.total_trades} trades, {m_all.win_rate:.1f}% WR, "
              f"PF {m_all.profit_factor:.3f}, P&L ${m_all.net_pnl:+,.0f}, "
              f"DD ${m_all.max_drawdown:,.0f}, Sharpe {m_all.sharpe:.2f}")

        # Exit reasons
        reasons = {}
        for t in rej_filtered:
            if t.exit_reason not in reasons:
                reasons[t.exit_reason] = {"count": 0, "pnl": 0, "wins": 0}
            reasons[t.exit_reason]["count"] += 1
            reasons[t.exit_reason]["pnl"] += t.pnl_dollar
            if t.pnl_dollar > 0:
                reasons[t.exit_reason]["wins"] += 1
        print(f"  REJ Exit Reasons:")
        for reason, r in sorted(reasons.items(), key=lambda x: -x[1]["pnl"]):
            wr = r["wins"] / r["count"] * 100
            print(f"    {reason:<10} {r['count']:>4} trades  {wr:>5.1f}% WR  ${r['pnl']:>+9,.0f}")

        # Year split
        if rej_filtered:
            mid_time = rej_filtered[0].entry_time + (rej_filtered[-1].entry_time - rej_filtered[0].entry_time) / 2
            y1 = [t for t in rej_filtered if t.entry_time < mid_time]
            y2 = [t for t in rej_filtered if t.entry_time >= mid_time]
            m1 = compute_metrics(y1) if y1 else None
            m2 = compute_metrics(y2) if y2 else None
            if m1 and m2:
                print(f"  Year split (REJ):")
                print(f"    Y1: {m1.total_trades} trades, PF {m1.profit_factor:.3f}, "
                      f"P&L ${m1.net_pnl:+,.0f}, Sharpe {m1.sharpe:.2f}")
                print(f"    Y2: {m2.total_trades} trades, PF {m2.profit_factor:.3f}, "
                      f"P&L ${m2.net_pnl:+,.0f}, Sharpe {m2.sharpe:.2f}")

        # Significance
        significance_tests(rej_filtered, f"REJ {label}")
        significance_tests(combined, f"Combined {label}")

    # ── Part 4: Monthly breakdown for best threshold ──
    best_rej = filter_trades_by_ib(rej_trades, ib_by_date, min_ratio=1.2)
    best_combined = best_rej + filter_trades_by_ib(ib_trades, ib_by_date, min_ratio=1.2)
    # Actually show monthly for ALL combined (REJ wide + IB all days)
    # because IB breakout runs on all days, only REJ is filtered
    all_combined = best_rej + ib_trades
    all_combined.sort(key=lambda t: t.entry_time)

    print(f"\n\n{'='*90}")
    print(f"  RECOMMENDED: REJ (wide days only, ratio >= 1.2) + IB Breakout (all days)")
    print(f"{'='*90}")

    m_rec = compute_metrics(all_combined)
    print(f"\n  Trades:       {m_rec.total_trades}")
    print(f"  Net P&L:      ${m_rec.net_pnl:+,.0f}")
    print(f"  Win Rate:     {m_rec.win_rate:.1f}%")
    print(f"  Profit Factor:{m_rec.profit_factor:.3f}")
    print(f"  Max Drawdown: ${m_rec.max_drawdown:,.0f} ({m_rec.max_drawdown_pct:.1f}%)")
    print(f"  Avg Trade:    ${m_rec.avg_trade:+,.0f}")
    print(f"  Sharpe:       {m_rec.sharpe:.2f}")
    print(f"  Calmar:       {m_rec.calmar:.2f}")
    print(f"  Trades/Day:   {m_rec.trades_per_day:.2f}")

    # Setup breakdown
    rej_in_combined = [t for t in all_combined if t.setup == "REJ"]
    ib_in_combined = [t for t in all_combined if t.setup == "IB"]
    m_rej_c = compute_metrics(rej_in_combined)
    m_ib_c = compute_metrics(ib_in_combined)
    print(f"\n  Per-Setup:")
    print(f"    IB:  {m_ib_c.total_trades:>4} trades, {m_ib_c.win_rate:>5.1f}% WR, "
          f"PF {m_ib_c.profit_factor:.3f}, P&L ${m_ib_c.net_pnl:>+9,.0f}")
    print(f"    REJ: {m_rej_c.total_trades:>4} trades, {m_rej_c.win_rate:>5.1f}% WR, "
          f"PF {m_rej_c.profit_factor:.3f}, P&L ${m_rej_c.net_pnl:>+9,.0f}")

    # Monthly
    print(f"\n  Monthly Breakdown:")
    by_month = {}
    for t in all_combined:
        key = t.entry_time.strftime("%Y-%m") if hasattr(t.entry_time, 'strftime') else "?"
        if key not in by_month:
            by_month[key] = {"count": 0, "pnl": 0, "wins": 0, "rej": 0, "ib": 0}
        by_month[key]["count"] += 1
        by_month[key]["pnl"] += t.pnl_dollar
        if t.pnl_dollar > 0:
            by_month[key]["wins"] += 1
        if t.setup == "REJ":
            by_month[key]["rej"] += 1
        else:
            by_month[key]["ib"] += 1

    print(f"  {'Month':<8} {'Total':>5} {'IB':>3} {'REJ':>3} {'WR%':>6} {'P&L':>10} {'Cum':>11}")
    print(f"  {'-'*52}")
    cum = 0
    win_months = 0
    for month in sorted(by_month.keys()):
        m = by_month[month]
        wr = m["wins"] / m["count"] * 100 if m["count"] > 0 else 0
        cum += m["pnl"]
        if m["pnl"] > 0:
            win_months += 1
        print(f"  {month:<8} {m['count']:>5} {m['ib']:>3} {m['rej']:>3} "
              f"{wr:>5.1f}% ${m['pnl']:>+9,.0f} ${cum:>+10,.0f}")
    print(f"  Winning months: {win_months}/{len(by_month)} ({win_months/len(by_month)*100:.0f}%)")

    # Year split for recommended
    mid_time = all_combined[0].entry_time + (all_combined[-1].entry_time - all_combined[0].entry_time) / 2
    y1 = [t for t in all_combined if t.entry_time < mid_time]
    y2 = [t for t in all_combined if t.entry_time >= mid_time]
    m1 = compute_metrics(y1)
    m2 = compute_metrics(y2)
    print(f"\n  Year Split:")
    print(f"    Y1: {m1.total_trades} trades, PF {m1.profit_factor:.3f}, P&L ${m1.net_pnl:+,.0f}, "
          f"Sharpe {m1.sharpe:.2f}, DD ${m1.max_drawdown:,.0f}")
    print(f"    Y2: {m2.total_trades} trades, PF {m2.profit_factor:.3f}, P&L ${m2.net_pnl:+,.0f}, "
          f"Sharpe {m2.sharpe:.2f}, DD ${m2.max_drawdown:,.0f}")

    # Significance
    print(f"\n  Significance:")
    significance_tests(all_combined, "Recommended combo")

    # ── Part 5: Distribution of IB ranges on wide days ──
    print(f"\n\n{'='*90}")
    print(f"  IB RANGE DISTRIBUTION (all sessions)")
    print(f"{'='*90}")
    ranges = [info["ib_range"] for info in ib_by_date.values() if info["ib_range"] > 0]
    ratios = [info["ib_ratio"] for info in ib_by_date.values() if info["ib_ratio"] > 0]
    print(f"  Sessions: {len(ranges)}")
    print(f"  IB Range:  mean={np.mean(ranges):.1f}pts, median={np.median(ranges):.1f}pts, "
          f"std={np.std(ranges):.1f}pts")
    print(f"  IB Ratio:  mean={np.mean(ratios):.2f}, median={np.median(ratios):.2f}")
    print(f"  Days >= 1.0x: {sum(1 for r in ratios if r >= 1.0)} ({sum(1 for r in ratios if r >= 1.0)/len(ratios)*100:.0f}%)")
    print(f"  Days >= 1.2x: {sum(1 for r in ratios if r >= 1.2)} ({sum(1 for r in ratios if r >= 1.2)/len(ratios)*100:.0f}%)")
    print(f"  Days >= 1.4x: {sum(1 for r in ratios if r >= 1.4)} ({sum(1 for r in ratios if r >= 1.4)/len(ratios)*100:.0f}%)")
    print(f"  Days >= 1.6x: {sum(1 for r in ratios if r >= 1.6)} ({sum(1 for r in ratios if r >= 1.6)/len(ratios)*100:.0f}%)")

    # Percentile
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  P{p}: {np.percentile(ranges, p):.1f}pts (ratio {np.percentile(ratios, p):.2f})")


if __name__ == "__main__":
    main()
