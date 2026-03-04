#!/usr/bin/env python3
"""Deep analysis of best IB Rejection combo: any/ib_low/TEMA=OFF/zone=5/stop=8/max=8."""

import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
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
    # Best REJ combo
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    return cfg


def analyze(trades, label, initial_capital=100_000.0):
    """Full analysis of a trade set."""
    if not trades:
        print(f"\n  {label}: No trades")
        return

    m = compute_metrics(trades, initial_capital)
    pnls = [t.pnl_dollar for t in trades]
    pts = [t.pnl_pts for t in trades]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Trades:       {m.total_trades}")
    print(f"  Net P&L:      ${m.net_pnl:+,.0f}")
    print(f"  Win Rate:     {m.win_rate:.1f}%")
    print(f"  Profit Factor:{m.profit_factor:.3f}")
    print(f"  Max Drawdown: ${m.max_drawdown:,.0f} ({m.max_drawdown_pct:.1f}%)")
    print(f"  Avg Trade:    ${m.avg_trade:+,.0f} ({np.mean(pts):+.1f} pts)")
    print(f"  Avg Win:      ${m.avg_win:+,.0f} ({np.mean([t.pnl_pts for t in trades if t.pnl_dollar > 0]):+.1f} pts)" if m.winners > 0 else "")
    print(f"  Avg Loss:     ${m.avg_loss:+,.0f} ({np.mean([t.pnl_pts for t in trades if t.pnl_dollar <= 0]):+.1f} pts)" if m.losers > 0 else "")
    print(f"  Sharpe:       {m.sharpe:.2f}")
    print(f"  Calmar:       {m.calmar:.2f}")
    print(f"  Trades/Day:   {m.trades_per_day:.2f}")
    print(f"  Win Streak:   {m.longest_win_streak}")
    print(f"  Lose Streak:  {m.longest_lose_streak}")

    # --- Exit Reasons ---
    print(f"\n  Exit Reasons:")
    reasons = {}
    for t in trades:
        if t.exit_reason not in reasons:
            reasons[t.exit_reason] = {"count": 0, "pnl": 0, "wins": 0, "pts": []}
        reasons[t.exit_reason]["count"] += 1
        reasons[t.exit_reason]["pnl"] += t.pnl_dollar
        reasons[t.exit_reason]["pts"].append(t.pnl_pts)
        if t.pnl_dollar > 0:
            reasons[t.exit_reason]["wins"] += 1
    for reason, r in sorted(reasons.items(), key=lambda x: -x[1]["pnl"]):
        wr = r["wins"] / r["count"] * 100 if r["count"] > 0 else 0
        avg_pts = np.mean(r["pts"])
        print(f"    {reason:<10} {r['count']:>5} trades  {wr:>5.1f}% WR  "
              f"avg {avg_pts:>+6.1f}pts  ${r['pnl']:>+10,.0f}")

    # --- P&L Distribution ---
    print(f"\n  P&L Distribution:")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(pnls, p)
        print(f"    P{p:>2}: ${val:>+8,.0f}")

    return m


def monthly_breakdown(trades, label):
    """Monthly P&L breakdown."""
    print(f"\n  Monthly Breakdown ({label}):")
    by_month = {}
    for t in trades:
        if hasattr(t.entry_time, 'strftime'):
            key = t.entry_time.strftime("%Y-%m")
        else:
            continue
        if key not in by_month:
            by_month[key] = {"count": 0, "pnl": 0, "wins": 0}
        by_month[key]["count"] += 1
        by_month[key]["pnl"] += t.pnl_dollar
        if t.pnl_dollar > 0:
            by_month[key]["wins"] += 1

    print(f"  {'Month':<8} {'Trades':>6} {'WR%':>6} {'P&L':>10} {'Cum P&L':>11}")
    print(f"  {'-'*45}")
    cum = 0
    win_months = 0
    for month in sorted(by_month.keys()):
        m = by_month[month]
        wr = m["wins"] / m["count"] * 100 if m["count"] > 0 else 0
        cum += m["pnl"]
        marker = "+" if m["pnl"] > 0 else "-"
        if m["pnl"] > 0:
            win_months += 1
        print(f"  {month:<8} {m['count']:>6} {wr:>5.1f}% ${m['pnl']:>+9,.0f} ${cum:>+10,.0f}  {marker}")

    total_months = len(by_month)
    print(f"  Winning months: {win_months}/{total_months} ({win_months/total_months*100:.0f}%)")


def yearly_split(trades):
    """Year 1 vs Year 2 breakdown."""
    # Split at midpoint
    if not trades:
        return
    mid = trades[0].entry_time + (trades[-1].entry_time - trades[0].entry_time) / 2
    y1 = [t for t in trades if t.entry_time < mid]
    y2 = [t for t in trades if t.entry_time >= mid]

    print(f"\n{'='*70}")
    print(f"  YEAR 1 vs YEAR 2 SPLIT")
    print(f"{'='*70}")
    for label, subset in [("Year 1 (early)", y1), ("Year 2 (recent)", y2)]:
        m = compute_metrics(subset)
        print(f"  {label}: {m.total_trades} trades, {m.win_rate:.1f}% WR, "
              f"PF {m.profit_factor:.3f}, P&L ${m.net_pnl:+,.0f}, "
              f"DD ${m.max_drawdown:,.0f}, Sharpe {m.sharpe:.2f}")


def statistical_significance(trades):
    """T-test and permutation test for significance."""
    pnls = np.array([t.pnl_dollar for t in trades])
    n = len(pnls)
    mean_pnl = np.mean(pnls)

    print(f"\n{'='*70}")
    print(f"  STATISTICAL SIGNIFICANCE")
    print(f"{'='*70}")

    # t-test: is mean P&L significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(pnls, 0)
    print(f"  t-test:        t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''}")

    # Permutation test: shuffle trade signs, compute P(random >= observed)
    observed = np.sum(pnls)
    n_perms = 10000
    np.random.seed(42)
    count_better = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=n)
        shuffled = np.abs(pnls) * signs
        if np.sum(shuffled) >= observed:
            count_better += 1
    perm_p = count_better / n_perms
    print(f"  Permutation:   p={perm_p:.4f} (10K shuffles) {'***' if perm_p < 0.01 else '**' if perm_p < 0.05 else '*' if perm_p < 0.1 else ''}")

    # Bootstrap: P(profitable) and 95% CI
    n_boot = 10000
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(pnls, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    p_profitable = np.mean(boot_means > 0) * 100
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"  Bootstrap:     P(profit)={p_profitable:.1f}%, 95% CI avg trade: [${ci_low:+,.0f}, ${ci_high:+,.0f}]")


def day_type_analysis(trades, df, cfg):
    """Breakdown by IB day type (narrow/normal/wide)."""
    # We need to re-run to get session state per trade
    # Simpler: compute IB range per session_date from the data
    from backtester.session import SessionState, update_session

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
                    "is_narrow": state.is_narrow_ib,
                    "is_wide": state.is_wide_ib,
                }
        prev_bar = bar

    print(f"\n{'='*70}")
    print(f"  DAY TYPE ANALYSIS (REJ trades only)")
    print(f"{'='*70}")

    rej_trades = [t for t in trades if t.setup == "REJ"]
    by_type = {"Narrow": [], "Normal": [], "Wide": []}
    for t in rej_trades:
        date_key = t.entry_time.date() if hasattr(t.entry_time, 'date') else None
        if date_key and date_key in ib_by_date:
            info = ib_by_date[date_key]
            if info["is_narrow"]:
                by_type["Narrow"].append(t)
            elif info["is_wide"]:
                by_type["Wide"].append(t)
            else:
                by_type["Normal"].append(t)

    for day_type, subset in by_type.items():
        if subset:
            m = compute_metrics(subset)
            print(f"  {day_type:<8}: {m.total_trades:>4} trades, {m.win_rate:>5.1f}% WR, "
                  f"PF {m.profit_factor:>6.3f}, P&L ${m.net_pnl:>+10,.0f}, "
                  f"Avg ${m.avg_trade:>+6,.0f}")


def holding_time_analysis(trades):
    """Analyze holding time for winners vs losers."""
    print(f"\n{'='*70}")
    print(f"  HOLDING TIME ANALYSIS (REJ trades)")
    print(f"{'='*70}")

    rej = [t for t in trades if t.setup == "REJ"]
    winners = [t for t in rej if t.pnl_dollar > 0]
    losers = [t for t in rej if t.pnl_dollar <= 0]

    def avg_hold(subset):
        holds = []
        for t in subset:
            if hasattr(t.entry_time, 'timestamp') and hasattr(t.exit_time, 'timestamp'):
                delta = (t.exit_time - t.entry_time).total_seconds() / 60
                holds.append(delta)
        return holds

    w_holds = avg_hold(winners)
    l_holds = avg_hold(losers)

    if w_holds:
        print(f"  Winners ({len(winners)}):  avg hold {np.mean(w_holds):.0f} min, "
              f"median {np.median(w_holds):.0f} min")
    if l_holds:
        print(f"  Losers ({len(losers)}):   avg hold {np.mean(l_holds):.0f} min, "
              f"median {np.median(l_holds):.0f} min")

    # By exit reason
    for reason in ["stop", "target", "flatten"]:
        subset = [t for t in rej if t.exit_reason == reason]
        if subset:
            holds = avg_hold(subset)
            if holds:
                print(f"  {reason:<10}: avg hold {np.mean(holds):.0f} min, "
                      f"median {np.median(holds):.0f} min, "
                      f"avg P&L ${np.mean([t.pnl_dollar for t in subset]):+,.0f}")


def trades_per_day_distribution(trades):
    """How many trades per day."""
    print(f"\n{'='*70}")
    print(f"  TRADES PER DAY DISTRIBUTION")
    print(f"{'='*70}")

    by_date = {}
    for t in trades:
        if hasattr(t.entry_time, 'date'):
            d = t.entry_time.date()
            by_date[d] = by_date.get(d, 0) + 1

    counts = list(by_date.values())
    if counts:
        print(f"  Days traded: {len(counts)}")
        print(f"  Avg trades/day: {np.mean(counts):.1f}")
        print(f"  Median trades/day: {np.median(counts):.0f}")
        print(f"  Max trades/day: {max(counts)}")
        # Distribution
        for n in range(1, max(counts) + 1):
            c = sum(1 for x in counts if x == n)
            if c > 0:
                print(f"    {n} trades/day: {c} days ({c/len(counts)*100:.1f}%)")


def main():
    print("Loading data...")
    df = load_tos_csv(sys.argv[1], instrument="ES")
    print(f"Loaded {len(df)} bars")

    cfg = make_cfg()
    trades = run_backtest(df.copy(), cfg)

    all_trades = trades
    rej_trades = [t for t in trades if t.setup == "REJ"]
    ib_trades = [t for t in trades if t.setup == "IB"]

    # Full analysis
    analyze(all_trades, "COMBINED (IB Breakout + IB Rejection)")
    analyze(rej_trades, "IB REJECTION ONLY")
    analyze(ib_trades, "IB BREAKOUT ONLY (for reference)")

    # Monthly breakdown
    monthly_breakdown(all_trades, "Combined")
    monthly_breakdown(rej_trades, "REJ Only")

    # Yearly split
    yearly_split(all_trades)
    yearly_split(rej_trades)

    # Statistical significance — combined and REJ standalone
    print(f"\n--- Combined Strategy ---")
    statistical_significance(all_trades)
    print(f"\n--- REJ Setup Standalone ---")
    statistical_significance(rej_trades)

    # Day type analysis
    day_type_analysis(trades, df, cfg)

    # Holding time
    holding_time_analysis(trades)

    # Trades per day
    trades_per_day_distribution(all_trades)


if __name__ == "__main__":
    main()
