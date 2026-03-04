#!/usr/bin/env python3
"""Analyze Level Rejection System — market structure aware short entries.

4-stage analysis:
  Stage 1: Full system performance (all levels ON, state-aware)
  Stage 2: Parameter tuning (shared params, not per-level)
  Stage 3: Market structure analysis (how levels interact)
  Stage 4: Day flow analysis (learning market patterns)

Usage:
    python scripts/analyze_level_rejection.py data/es_5m_databento_2yr.csv
    python scripts/analyze_level_rejection.py data/es_5m_databento_2yr.csv --stage 1
    python scripts/analyze_level_rejection.py data/es_5m_databento_2yr.csv --stage 3
"""

import argparse
import sys
import os
import math
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown
from backtester.session import SessionState, update_session
from backtester.indicators import compute_indicators


# ── Shared config builders ──

def make_v8_cfg():
    """v8 baseline: short-only, pct stop, Friday/noon filters, no VA."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    return cfg


def make_lvl_cfg(**overrides):
    """Level Rejection config: v8 base + level rejection enabled."""
    cfg = make_v8_cfg()
    cfg.use_level_reject = True
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 8.0
    cfg.lvl_require_tema = False
    cfg.max_lvl_trades = 4
    cfg.lvl_ibh_wide_only = True
    cfg.lvl_max_tests = 3
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def print_metrics(label, trades):
    """Print a compact metrics line."""
    m = compute_metrics(trades)
    print(f"  {label:<35s}  {m.total_trades:>4d} trades  "
          f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+10,.0f}  DD ${m.max_drawdown:>8,.0f}  "
          f"Sharpe {m.sharpe:>5.2f}")
    return m


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ── Stage 1: Full System Performance ──

def stage1_full_system(df):
    """Run Level Rejection with all levels, compare vs v8 and v10."""
    print_header("STAGE 1: Full System Performance")

    # v8 baseline (IB Breakout only)
    cfg_v8 = make_v8_cfg()
    trades_v8 = run_backtest(df.copy(), cfg_v8)
    print("\n  --- Baselines ---")
    print_metrics("v8 (IB Breakout only)", trades_v8)

    # v10 (IB Rejection + IB Breakout)
    cfg_v10 = make_v8_cfg()
    cfg_v10.use_ib_reject = True
    cfg_v10.rej_trigger = "any"
    cfg_v10.rej_target = "ib_low"
    cfg_v10.rej_zone_pts = 5.0
    cfg_v10.rej_stop_buffer = 8.0
    cfg_v10.rej_require_tema = False
    cfg_v10.max_rej_trades = 8
    trades_v10 = run_backtest(df.copy(), cfg_v10)
    print_metrics("v10 (IBH Rejection, no state)", trades_v10)

    # Level Rejection (all levels, state-aware)
    cfg_lvl = make_lvl_cfg()
    trades_lvl = run_backtest(df.copy(), cfg_lvl)
    print("\n  --- Level Rejection System ---")
    m_lvl = print_metrics("Level Rejection (all levels)", trades_lvl)

    # Per-setup breakdown
    if trades_lvl:
        breakdown = per_setup_breakdown(trades_lvl)
        print(f"\n  Per-Setup Breakdown:")
        for setup, sm in sorted(breakdown.items()):
            print(f"    {setup:<10s}  {sm.total_trades:>4d} trades  "
                  f"WR {sm.win_rate:>5.1f}%  PF {sm.profit_factor:>6.3f}  "
                  f"P&L ${sm.net_pnl:>+10,.0f}")

    # Exit reason breakdown
    if trades_lvl:
        print(f"\n  Exit Reasons:")
        reasons = defaultdict(lambda: {"count": 0, "pnl": 0})
        for t in trades_lvl:
            reasons[t.exit_reason]["count"] += 1
            reasons[t.exit_reason]["pnl"] += t.pnl_dollar
        for reason, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            print(f"    {reason:<12s}  {d['count']:>4d} trades  ${d['pnl']:>+10,.0f}")


# ── Stage 2: Parameter Tuning ──

def stage2_param_sweep(df):
    """Sweep shared parameters to find best system config."""
    print_header("STAGE 2: Parameter Tuning")

    # --- Sweep max_lvl_trades ---
    print(f"\n  --- Max Trades per Day ---")
    for mt in [2, 3, 4, 5, 6]:
        cfg = make_lvl_cfg(max_lvl_trades=mt)
        trades = run_backtest(df.copy(), cfg)
        print_metrics(f"max_trades={mt}", trades)

    # --- Sweep zone_pts ---
    print(f"\n  --- Zone Points ---")
    for zp in [3.0, 5.0, 8.0, 10.0]:
        cfg = make_lvl_cfg(lvl_zone_pts=zp)
        trades = run_backtest(df.copy(), cfg)
        print_metrics(f"zone_pts={zp}", trades)

    # --- Sweep stop_buffer ---
    print(f"\n  --- Stop Buffer ---")
    for sb in [3.0, 5.0, 8.0, 12.0]:
        cfg = make_lvl_cfg(lvl_stop_buffer=sb)
        trades = run_backtest(df.copy(), cfg)
        print_metrics(f"stop_buffer={sb}", trades)

    # --- Sweep max_tests ---
    print(f"\n  --- Max Tests (level exhaustion) ---")
    for mt in [2, 3, 4, 99]:
        label = f"max_tests={mt}" if mt < 99 else "max_tests=unlimited"
        cfg = make_lvl_cfg(lvl_max_tests=mt)
        trades = run_backtest(df.copy(), cfg)
        print_metrics(label, trades)

    # --- IBH wide-only vs all days ---
    print(f"\n  --- IBH Wide Filter ---")
    for wide_only in [True, False]:
        label = "IBH wide-only" if wide_only else "IBH all days"
        cfg = make_lvl_cfg(lvl_ibh_wide_only=wide_only)
        trades = run_backtest(df.copy(), cfg)
        print_metrics(label, trades)

    # --- Trigger types ---
    print(f"\n  --- Trigger Types ---")
    for trigger in ["any", "bearish_close", "wick", "failed_break"]:
        cfg = make_lvl_cfg(lvl_trigger=trigger)
        trades = run_backtest(df.copy(), cfg)
        print_metrics(f"trigger={trigger}", trades)

    # --- TEMA filter ---
    print(f"\n  --- TEMA Filter ---")
    for tema in [False, True]:
        label = "TEMA required" if tema else "TEMA off"
        cfg = make_lvl_cfg(lvl_require_tema=tema)
        trades = run_backtest(df.copy(), cfg)
        print_metrics(label, trades)


# ── Stage 3: Market Structure Analysis ──

def stage3_market_structure(df):
    """Analyze how levels interact with market flow."""
    print_header("STAGE 3: Market Structure Analysis")

    # Run full backtest to get trades with setup tags
    cfg = make_lvl_cfg()
    trades = run_backtest(df.copy(), cfg)

    if not trades:
        print("  No trades to analyze.")
        return

    # --- Per-level breakdown ---
    print(f"\n  --- Per-Level Breakdown (understanding, not elimination) ---")
    breakdown = per_setup_breakdown(trades)
    for setup, sm in sorted(breakdown.items()):
        avg = sm.net_pnl / sm.total_trades if sm.total_trades > 0 else 0
        print(f"    {setup:<10s}  {sm.total_trades:>4d} trades  "
              f"WR {sm.win_rate:>5.1f}%  PF {sm.profit_factor:>6.3f}  "
              f"P&L ${sm.net_pnl:>+10,.0f}  Avg ${avg:>+7,.0f}")

    # --- Test-count analysis ---
    print(f"\n  --- Test-Count Analysis ---")
    print(f"  (How does test # at entry time correlate with outcomes?)")

    # We need to track test count at entry time — re-run with instrumentation
    cfg2 = make_lvl_cfg()
    test_count_trades = _run_with_level_info(df.copy(), cfg2)

    if test_count_trades:
        by_test = defaultdict(list)
        for t, info in test_count_trades:
            tc = info.get("test_count", 0)
            by_test[tc].append(t.pnl_dollar)

        for tc in sorted(by_test.keys()):
            pnls = by_test[tc]
            n = len(pnls)
            wr = sum(1 for p in pnls if p > 0) / n * 100 if n > 0 else 0
            total = sum(pnls)
            gp = sum(p for p in pnls if p > 0) or 0
            gl = abs(sum(p for p in pnls if p <= 0)) or 1
            pf = gp / gl if gl > 0 else float("inf")
            print(f"    Test #{tc:<2d}  {n:>4d} trades  WR {wr:>5.1f}%  PF {pf:>6.3f}  P&L ${total:>+10,.0f}")

    # --- Level confluence analysis ---
    print(f"\n  --- Level Confluence Analysis ---")
    _analyze_confluence(df, cfg)

    # --- Target completion analysis ---
    print(f"\n  --- Target Completion ---")
    if trades:
        target_hits = sum(1 for t in trades if t.exit_reason == "target")
        pct = target_hits / len(trades) * 100
        print(f"    Target reached: {target_hits}/{len(trades)} ({pct:.1f}%)")

        # Target distances
        distances = [abs(t.entry_price - t.target) for t in trades]
        print(f"    Target distance: min={min(distances):.1f}  "
              f"median={np.median(distances):.1f}  "
              f"max={max(distances):.1f}  mean={np.mean(distances):.1f}")


def _run_with_level_info(df, cfg):
    """Re-run backtest capturing test count at entry time.

    Returns list of (Trade, info_dict) tuples.
    """
    from backtester.indicators import compute_indicators
    from backtester.session import SessionState, update_session
    from backtester.position import Position
    from backtester.setups import ib_breakout, ib_rejection, level_rejection, va_fade, eighty_rule, tema_cross

    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    pos = Position()
    results = []
    prev_bar = None
    prev_position_size = 0
    pending_info = None

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx

        update_session(state, bar, prev_bar, cfg)
        level_rejection.update_level_state(bar, state, cfg)

        current_position_size = 0 if pos.is_flat else pos.direction
        if current_position_size == 0 and prev_position_size != 0:
            state.bars_since_exit = 0
        elif current_position_size == 0:
            state.bars_since_exit += 1

        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            if pending_info:
                results.append((trade, pending_info))
                pending_info = None

        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                if pending_info:
                    results.append((trade, pending_info))
                    pending_info = None

        if pos.is_flat:
            signal = None
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = ib_rejection.check_signal(bar, prev_bar, state, cfg)

                if signal is None:
                    # Capture level info BEFORE check_signal increments counter
                    resistance = level_rejection._get_resistance_map(state)
                    level_info = {}
                    for name, price in resistance:
                        if bar["high"] >= price - cfg.lvl_zone_pts:
                            level_info[name] = {
                                "test_count": state.lvl_test_count.get(name, 0),
                                "broken": state.lvl_broken.get(name, False),
                            }

                    signal = level_rejection.check_signal(bar, prev_bar, state, cfg)
                    if signal is not None:
                        # Find which level was used
                        setup_name = signal["setup"].replace("LVL_", "")
                        tc = level_info.get(setup_name, {}).get("test_count", 0)
                        pending_info = {"test_count": tc, "level": setup_name}

                if signal is None:
                    signal = va_fade.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = eighty_rule.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = tema_cross.check_signal(bar, prev_bar, state, cfg)

            if signal is not None and cfg.direction_filter != "both":
                if cfg.direction_filter == "short" and signal["direction"] == 1:
                    signal = None

            if signal is not None:
                pos.enter(
                    direction=signal["direction"],
                    price=bar["close"],
                    stop=signal["stop"],
                    target=signal["target"],
                    setup=signal["setup"],
                    time=idx,
                    slippage=cfg.slippage_pts,
                )

        prev_position_size = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    return results


def _analyze_confluence(df, cfg):
    """Analyze how often resistance levels cluster within 5pts."""
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    prev_bar = None
    confluence_days = 0
    total_days = 0
    confluence_counts = defaultdict(int)  # {2: N, 3: N, 4: N}

    last_date = None

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)

        # Check once per session after IB done
        current_date = idx.date()
        if state.ib_done and bar["is_rth"] and current_date != last_date:
            last_date = current_date
            total_days += 1

            from backtester.setups.level_rejection import _get_resistance_map
            resistance = _get_resistance_map(state)
            if len(resistance) >= 2:
                # Check pairwise clustering within 5pts
                prices = [p for _, p in resistance]
                cluster_size = 1
                for i in range(len(prices) - 1):
                    if abs(prices[i] - prices[i + 1]) <= 5.0:
                        cluster_size += 1
                if cluster_size >= 2:
                    confluence_days += 1
                    confluence_counts[cluster_size] += 1

        prev_bar = bar

    if total_days > 0:
        pct = confluence_days / total_days * 100
        print(f"    Confluence (2+ levels within 5pts): {confluence_days}/{total_days} days ({pct:.1f}%)")
        for size, count in sorted(confluence_counts.items()):
            print(f"      {size} levels clustered: {count} days")
    else:
        print(f"    No sessions found.")


# ── Stage 4: Day Flow Analysis ──

def stage4_day_flow(df):
    """Analyze daily market flow patterns through the level map."""
    print_header("STAGE 4: Day Flow Analysis")

    cfg = make_lvl_cfg()
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    prev_bar = None

    # Track per-day info
    daily_first_hit = {}   # {date: level_name}
    daily_levels_tested = defaultdict(set)  # {date: {level_names}}
    daily_breaks = defaultdict(list)  # {date: [level_names broken]}
    daily_ib_type = {}  # {date: "wide"/"normal"/"narrow"}

    last_date = None
    first_hit_found = {}

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)
        level_rejection.update_level_state(bar, state, cfg)

        current_date = idx.date()

        if state.ib_done and bar["is_rth"]:
            # Track IB type
            if current_date not in daily_ib_type:
                if state.is_wide_ib:
                    daily_ib_type[current_date] = "wide"
                elif state.is_narrow_ib:
                    daily_ib_type[current_date] = "narrow"
                else:
                    daily_ib_type[current_date] = "normal"

            # Track which levels are tested
            from backtester.setups.level_rejection import _get_resistance_map
            resistance = _get_resistance_map(state)
            zone = cfg.lvl_zone_pts

            for name, price in resistance:
                if bar["high"] >= price - zone:
                    daily_levels_tested[current_date].add(name)
                    # First hit of the day
                    if current_date not in first_hit_found:
                        first_hit_found[current_date] = True
                        daily_first_hit[current_date] = name

                    # Track breaks
                    if bar["close"] > price:
                        if name not in daily_breaks[current_date]:
                            daily_breaks[current_date].append(name)

        prev_bar = bar

    # --- Which resistance level is hit FIRST each session? ---
    print(f"\n  --- First Resistance Hit Per Session ---")
    first_counts = defaultdict(int)
    for date, name in daily_first_hit.items():
        first_counts[name] += 1
    total = sum(first_counts.values())
    for name in sorted(first_counts, key=first_counts.get, reverse=True):
        pct = first_counts[name] / total * 100 if total > 0 else 0
        print(f"    {name:<6s}  {first_counts[name]:>4d} days ({pct:>5.1f}%)")

    # --- Levels tested per session (activity breadth) ---
    print(f"\n  --- Levels Tested Per Session ---")
    level_counts = [len(v) for v in daily_levels_tested.values()]
    if level_counts:
        print(f"    Mean: {np.mean(level_counts):.1f}  Median: {np.median(level_counts):.0f}  "
              f"Min: {min(level_counts)}  Max: {max(level_counts)}")
        # Distribution
        for n in range(0, max(level_counts) + 1):
            count = sum(1 for c in level_counts if c == n)
            if count > 0:
                print(f"    {n} levels: {count} days ({count/len(level_counts)*100:.1f}%)")

    # --- Wide vs Normal vs Narrow days ---
    print(f"\n  --- Performance by IB Day Type ---")
    # Get trades with dates
    trades = run_backtest(df.copy(), cfg)
    if trades:
        by_type = defaultdict(list)
        for t in trades:
            if hasattr(t.entry_time, 'date'):
                d = t.entry_time.date()
                ib_type = daily_ib_type.get(d, "unknown")
                by_type[ib_type].append(t)

        for ib_type in ["wide", "normal", "narrow"]:
            type_trades = by_type.get(ib_type, [])
            if type_trades:
                m = compute_metrics(type_trades)
                print(f"    {ib_type:<8s}  {m.total_trades:>4d} trades  "
                      f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
                      f"P&L ${m.net_pnl:>+10,.0f}")
            else:
                print(f"    {ib_type:<8s}  No trades")

    # --- Level breaks per day ---
    print(f"\n  --- Level Breaks Per Session ---")
    break_counts = [len(v) for v in daily_breaks.values() if v]
    no_break_days = sum(1 for v in daily_breaks.values() if not v)
    total_days = len(daily_levels_tested)
    if break_counts:
        print(f"    Days with breaks: {len(break_counts)}/{total_days} ({len(break_counts)/total_days*100:.1f}%)")
        print(f"    Avg breaks/day: {np.mean(break_counts):.1f}")
        # Which levels break most?
        break_freq = defaultdict(int)
        for breaks in daily_breaks.values():
            for name in breaks:
                break_freq[name] += 1
        for name in sorted(break_freq, key=break_freq.get, reverse=True):
            pct = break_freq[name] / total_days * 100 if total_days > 0 else 0
            print(f"    {name:<6s} breaks: {break_freq[name]:>4d} days ({pct:>5.1f}%)")


# ── Year-split analysis ──

def year_split_analysis(df):
    """Run Level Rejection on Year 1 and Year 2 separately."""
    print_header("YEAR SPLIT ANALYSIS")

    # Split data: Year 2 = first ~70K bars, Year 1 = last ~70K bars
    midpoint = len(df) // 2
    df_y2 = df.iloc[:midpoint].copy()
    df_y1 = df.iloc[midpoint:].copy()

    print(f"\n  Year 2 (older): {df_y2.index[0].date()} to {df_y2.index[-1].date()} ({len(df_y2)} bars)")
    print(f"  Year 1 (recent): {df_y1.index[0].date()} to {df_y1.index[-1].date()} ({len(df_y1)} bars)")

    cfg = make_lvl_cfg()

    print(f"\n  --- Full 2-Year ---")
    trades_all = run_backtest(df.copy(), cfg)
    print_metrics("Level Rejection (combined)", trades_all)

    print(f"\n  --- Year 2 (older, ~ES 5000) ---")
    trades_y2 = run_backtest(df_y2, cfg)
    m2 = print_metrics("Year 2", trades_y2)

    print(f"\n  --- Year 1 (recent, ~ES 6800) ---")
    trades_y1 = run_backtest(df_y1, cfg)
    m1 = print_metrics("Year 1", trades_y1)

    # Both years profitable?
    if m1.net_pnl > 0 and m2.net_pnl > 0:
        print(f"\n  BOTH YEARS PROFITABLE")
    else:
        losers = []
        if m1.net_pnl <= 0:
            losers.append("Year 1")
        if m2.net_pnl <= 0:
            losers.append("Year 2")
        print(f"\n  WARNING: {', '.join(losers)} losing")


# ── Main ──

def main():
    from backtester.setups import level_rejection as lr_mod
    # Make level_rejection available globally for stage4
    global level_rejection
    level_rejection = lr_mod

    parser = argparse.ArgumentParser(description="Level Rejection Analysis")
    parser.add_argument("csv_file", help="Path to CSV data file")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run specific stage (1-4), 0=all")
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    rth_bars = df["is_rth"].sum()
    sessions = df["new_rth"].sum()
    print(f"RTH bars: {rth_bars}, Trading sessions: {sessions}")

    if args.stage == 0 or args.stage == 1:
        stage1_full_system(df)

    if args.stage == 0 or args.stage == 2:
        stage2_param_sweep(df)

    if args.stage == 0 or args.stage == 3:
        stage3_market_structure(df)

    if args.stage == 0 or args.stage == 4:
        stage4_day_flow(df)

    # Always show year split
    if args.stage == 0:
        year_split_analysis(df)


if __name__ == "__main__":
    main()
