#!/usr/bin/env python3
"""
Test whether the Friday filter should apply to IB Rejection trades
separately from IB Breakout trades.

Compares multiple configurations on the full 2yr dataset:
1. Baseline (current): skip_friday=True for ALL setups
2. All Fridays ON: skip_friday=False
3. Split analysis: breakdown by setup type x day-of-week
4. Simulated "split Friday" configs

Usage:
    cd /Users/chuck_mf_norris/projects/backtesting/es
    .venv/bin/python scripts/test_rej_friday.py data/es_5m_databento_2yr.csv
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


def make_base_config():
    """AMT-TEMA v8 + REJ wide-day settings."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
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


def bucket_metrics(trades):
    """Compute metrics for a list of trades. Returns dict with key stats."""
    if not trades:
        return {
            "count": 0, "winners": 0, "losers": 0,
            "win_rate": 0.0, "pf": 0.0, "net_pnl": 0.0,
            "avg_trade": 0.0, "gross_win": 0.0, "gross_loss": 0.0,
        }
    m = compute_metrics(trades)
    return {
        "count": m.total_trades,
        "winners": m.winners,
        "losers": m.losers,
        "win_rate": m.win_rate,
        "pf": m.profit_factor,
        "net_pnl": m.net_pnl,
        "avg_trade": m.avg_trade,
        "gross_win": m.gross_profit,
        "gross_loss": m.gross_loss,
    }


def print_header(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_row(label, d):
    """Print one row of the summary table."""
    print(f"  {label:<30s}  {d['count']:>5d}  {d['win_rate']:>6.1f}%  "
          f"{d['pf']:>6.2f}  ${d['net_pnl']:>+10,.0f}  ${d['avg_trade']:>+8,.0f}")


def print_table_header():
    print(f"  {'Bucket':<30s}  {'#':>5s}  {'WR':>6s}   {'PF':>6s}  {'Net P&L':>11s}  {'Avg':>9s}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*11}  {'-'*9}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_rej_friday.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]
    print(f"Loading data from {data_file} ...")
    df = load_tos_csv(data_file)
    print(f"Loaded {len(df):,} bars")

    # =========================================================================
    # CONFIG A: Baseline (current) - skip_friday=True for ALL setups
    # =========================================================================
    print("\nRunning Config A (skip_friday=True) ...")
    cfg_a = make_base_config()
    cfg_a.skip_friday = True
    trades_a = run_backtest(df, cfg_a)

    # =========================================================================
    # CONFIG B: All Fridays ON - skip_friday=False
    # =========================================================================
    print("Running Config B (skip_friday=False) ...")
    cfg_b = make_base_config()
    cfg_b.skip_friday = False
    trades_b = run_backtest(df, cfg_b)

    # =========================================================================
    # SECTION 1: High-level comparison (A vs B)
    # =========================================================================
    print_header("1. HIGH-LEVEL: skip_friday=True vs skip_friday=False")
    print_table_header()

    m_a = bucket_metrics(trades_a)
    m_b = bucket_metrics(trades_b)
    print_row("A: skip_friday=True (current)", m_a)
    print_row("B: skip_friday=False (all)", m_b)

    delta_pnl = m_b["net_pnl"] - m_a["net_pnl"]
    delta_trades = m_b["count"] - m_a["count"]
    print(f"\n  Delta (B - A): {delta_trades:+d} trades, ${delta_pnl:+,.0f} P&L")

    # =========================================================================
    # SECTION 2: Split analysis - breakdown by setup x day
    # =========================================================================
    print_header("2. SPLIT ANALYSIS: Performance by Setup x Day-of-Week")
    print("  (Using skip_friday=False run to see ALL trades including Fridays)")

    # Partition trades from Config B
    ib_nonfri  = [t for t in trades_b if t.setup == "IB" and t.entry_time.weekday() != 4]
    ib_fri     = [t for t in trades_b if t.setup == "IB" and t.entry_time.weekday() == 4]
    rej_nonfri = [t for t in trades_b if t.setup == "REJ" and t.entry_time.weekday() != 4]
    rej_fri    = [t for t in trades_b if t.setup == "REJ" and t.entry_time.weekday() == 4]

    print()
    print_table_header()
    print_row("IB Breakout: Mon-Thu", bucket_metrics(ib_nonfri))
    print_row("IB Breakout: Friday", bucket_metrics(ib_fri))
    print_row("REJ Rejection: Mon-Thu", bucket_metrics(rej_nonfri))
    print_row("REJ Rejection: Friday", bucket_metrics(rej_fri))

    print()
    print(f"  --- Totals by setup ---")
    print_table_header()
    all_ib = ib_nonfri + ib_fri
    all_rej = rej_nonfri + rej_fri
    print_row("IB Breakout: ALL days", bucket_metrics(all_ib))
    print_row("REJ Rejection: ALL days", bucket_metrics(all_rej))

    # =========================================================================
    # SECTION 3: Day-of-week breakdown for each setup
    # =========================================================================
    print_header("3. FULL DAY-OF-WEEK BREAKDOWN")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    for setup_name, setup_code in [("IB Breakout", "IB"), ("REJ Rejection", "REJ")]:
        print(f"\n  --- {setup_name} ---")
        print_table_header()
        setup_trades = [t for t in trades_b if t.setup == setup_code]
        for dow in range(5):
            day_trades = [t for t in setup_trades if t.entry_time.weekday() == dow]
            label = f"{day_names[dow]}"
            print_row(label, bucket_metrics(day_trades))
        print_row("TOTAL", bucket_metrics(setup_trades))

    # =========================================================================
    # SECTION 4: Simulated split-Friday configs
    # =========================================================================
    print_header("4. SIMULATED SPLIT-FRIDAY CONFIGURATIONS")
    print("  Since the engine applies skip_friday to ALL setups, we simulate")
    print("  split configs by filtering the skip_friday=False trades.\n")

    # Config C: IB skips Friday, REJ trades Friday
    # = IB non-Friday + REJ all days
    trades_c = ib_nonfri + all_rej
    trades_c.sort(key=lambda t: t.entry_time)

    # Config D: IB trades Friday, REJ skips Friday (unlikely but for completeness)
    trades_d = all_ib + rej_nonfri
    trades_d.sort(key=lambda t: t.entry_time)

    # Config E: Both skip Friday (filtered from B, should match A)
    trades_e = ib_nonfri + rej_nonfri
    trades_e.sort(key=lambda t: t.entry_time)

    m_e = bucket_metrics(trades_e)
    m_c = bucket_metrics(trades_c)
    m_d = bucket_metrics(trades_d)

    print_table_header()
    print_row("A: Both skip Fri (engine)", m_a)
    print_row("E: Both skip Fri (filtered)", m_e)
    print_row("B: Both trade Fri (engine)", m_b)
    print_row("C: IB skip Fri, REJ trade Fri", m_c)
    print_row("D: IB trade Fri, REJ skip Fri", m_d)

    # =========================================================================
    # SECTION 5: The key question answered
    # =========================================================================
    print_header("5. THE KEY QUESTION: Does adding REJ Fridays help?")

    rej_fri_m = bucket_metrics(rej_fri)
    ib_fri_m = bucket_metrics(ib_fri)

    print(f"\n  REJ trades on Friday alone:")
    print(f"    Trades:    {rej_fri_m['count']}")
    print(f"    Win Rate:  {rej_fri_m['win_rate']:.1f}%")
    print(f"    PF:        {rej_fri_m['pf']:.2f}")
    print(f"    Net P&L:   ${rej_fri_m['net_pnl']:+,.0f}")
    print(f"    Avg Trade:  ${rej_fri_m['avg_trade']:+,.0f}")

    print(f"\n  IB Breakout trades on Friday alone:")
    print(f"    Trades:    {ib_fri_m['count']}")
    print(f"    Win Rate:  {ib_fri_m['win_rate']:.1f}%")
    print(f"    PF:        {ib_fri_m['pf']:.2f}")
    print(f"    Net P&L:   ${ib_fri_m['net_pnl']:+,.0f}")
    print(f"    Avg Trade:  ${ib_fri_m['avg_trade']:+,.0f}")

    print(f"\n  Combined portfolio comparison:")
    print(f"    Current (both skip Fri):       {m_e['count']} trades, PF {m_e['pf']:.2f}, ${m_e['net_pnl']:+,.0f}")
    print(f"    + REJ Fridays only (Config C): {m_c['count']} trades, PF {m_c['pf']:.2f}, ${m_c['net_pnl']:+,.0f}")
    print(f"    + All Fridays (Config B):      {m_b['count']} trades, PF {m_b['pf']:.2f}, ${m_b['net_pnl']:+,.0f}")

    delta_c = m_c["net_pnl"] - m_e["net_pnl"]
    delta_b_from_e = m_b["net_pnl"] - m_e["net_pnl"]

    print(f"\n  Adding REJ Fridays only:  ${delta_c:+,.0f} P&L change ({m_c['count'] - m_e['count']:+d} trades)")
    print(f"  Adding ALL Fridays:       ${delta_b_from_e:+,.0f} P&L change ({m_b['count'] - m_e['count']:+d} trades)")

    # Recommendation
    print()
    print("-" * 80)
    if rej_fri_m["count"] == 0:
        print("  RESULT: No REJ trades on Friday -- filter has no impact on REJ.")
    elif rej_fri_m["pf"] >= 1.2 and rej_fri_m["net_pnl"] > 0 and delta_c > 0:
        print("  RESULT: REJ Fridays are PROFITABLE (PF >= 1.2, positive P&L).")
        print("  RECOMMENDATION: Consider allowing REJ trades on Fridays.")
        print("  This would require a per-setup Friday filter in the engine.")
    elif rej_fri_m["net_pnl"] > 0 and rej_fri_m["pf"] >= 1.0:
        print("  RESULT: REJ Fridays are marginally profitable but weak.")
        print("  RECOMMENDATION: Keep skip_friday=True for now (not worth the complexity).")
    else:
        print("  RESULT: REJ Fridays are UNPROFITABLE or break-even.")
        print("  RECOMMENDATION: Keep skip_friday=True for both setups (current behavior).")
    print("-" * 80)
    print()


if __name__ == "__main__":
    main()
