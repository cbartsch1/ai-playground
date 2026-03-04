#!/usr/bin/env python3
"""Daily MA Filter for MS Short Trades — tests if SMA(5) or SMA(20) on daily 
closes acts as support that blocks short targets.

Hypothesis: Losing MS short trades are catching support at the 5-day or 20-day
daily moving average. If the daily MA sits between entry and target, the short
is more likely to fail.

Tests:
  1. Diagnostic: split all MS trades by whether a daily MA was between entry/target
  2. Applied filter: skip shorts when daily SMA(5) or SMA(20) is within N pts below entry
     (acts as support shelf). Test N = 3, 5, 8, 10, 15, 20 pts.

Usage:
    python scripts/ms_daily_ma_filter.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: compute daily SMAs and map to 5-min bars
# ═══════════════════════════════════════════════════════════════════════════════

def add_daily_sma_columns(df):
    """Add daily_sma5 and daily_sma20 columns to the 5-min DataFrame.
    
    Uses PREVIOUS day's close SMA to avoid lookahead bias (shift(1)).
    Extracts daily closes from the last RTH bar of each day.
    """
    # Extract daily closes from RTH data — last RTH bar each day
    rth_bars = df[df['is_rth']].copy()
    daily_closes = rth_bars.resample('D')['close'].last().dropna()
    
    # Compute daily SMAs
    daily_sma5 = daily_closes.rolling(5).mean()
    daily_sma20 = daily_closes.rolling(20).mean()
    
    # Shift by 1 to use PREVIOUS day's MA (no lookahead)
    daily_sma5_shifted = daily_sma5.shift(1)
    daily_sma20_shifted = daily_sma20.shift(1)
    
    # Map back to 5-min bars using the date portion of the index
    # Create a mapping from date -> SMA value
    sma5_map = daily_sma5_shifted.to_dict()
    sma20_map = daily_sma20_shifted.to_dict()
    
    # For each 5-min bar, find its session_date and look up the daily SMA
    # Use normalize() to get the date part of the datetime index
    bar_dates = df.index.normalize()
    
    df['daily_sma5'] = bar_dates.map(sma5_map)
    df['daily_sma20'] = bar_dates.map(sma20_map)
    
    # Forward-fill within each day (handles overnight bars that don't match a daily close date)
    df['daily_sma5'] = df['daily_sma5'].ffill()
    df['daily_sma20'] = df['daily_sma20'].ffill()
    
    # Report coverage
    total = len(df)
    sma5_valid = df['daily_sma5'].notna().sum()
    sma20_valid = df['daily_sma20'].notna().sum()
    print(f"  Daily SMA coverage: SMA5={sma5_valid}/{total} ({100*sma5_valid/total:.1f}%), "
          f"SMA20={sma20_valid}/{total} ({100*sma20_valid/total:.1f}%)")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: check if daily MA is between entry and target for a trade
# ═══════════════════════════════════════════════════════════════════════════════

def daily_ma_between_entry_target(trade, df):
    """Check if SMA5 or SMA20 was between entry and target at trade entry time.
    
    For shorts: entry is above target, so MA "between" = target < MA < entry
    For longs: entry is below target, so MA "between" = entry < MA < target
    
    Returns: (sma5_between, sma20_between, sma5_val, sma20_val)
    """
    entry_time = trade.entry_time
    
    # Look up the daily SMA at entry time
    if entry_time not in df.index:
        # Find nearest bar
        idx = df.index.get_indexer([entry_time], method='nearest')[0]
        entry_time = df.index[idx]
    
    sma5 = df.loc[entry_time, 'daily_sma5'] if 'daily_sma5' in df.columns else np.nan
    sma20 = df.loc[entry_time, 'daily_sma20'] if 'daily_sma20' in df.columns else np.nan
    
    entry = trade.entry_price
    target = trade.target
    
    sma5_between = False
    sma20_between = False
    
    if trade.direction == -1:  # Short: target < MA < entry means MA is support
        if not np.isnan(sma5):
            sma5_between = target < sma5 < entry
        if not np.isnan(sma20):
            sma20_between = target < sma20 < entry
    else:  # Long: entry < MA < target means MA is resistance
        if not np.isnan(sma5):
            sma5_between = entry < sma5 < target
        if not np.isnan(sma20):
            sma20_between = entry < sma20 < target
    
    return sma5_between, sma20_between, sma5, sma20


def daily_ma_proximity_below(trade, df, distance):
    """Check if daily SMA5 or SMA20 is within `distance` points below entry.
    
    For shorts: MA below entry by less than `distance` points = potential support.
    Returns True if the trade should be BLOCKED.
    """
    if trade.direction != -1:
        return False  # Only filter shorts
    
    entry_time = trade.entry_time
    if entry_time not in df.index:
        idx = df.index.get_indexer([entry_time], method='nearest')[0]
        entry_time = df.index[idx]
    
    sma5 = df.loc[entry_time, 'daily_sma5'] if 'daily_sma5' in df.columns else np.nan
    sma20 = df.loc[entry_time, 'daily_sma20'] if 'daily_sma20' in df.columns else np.nan
    
    entry = trade.entry_price
    
    # Block if SMA is within `distance` below entry (acting as nearby support)
    if not np.isnan(sma5):
        gap5 = entry - sma5
        if 0 < gap5 <= distance:
            return True
    
    if not np.isnan(sma20):
        gap20 = entry - sma20
        if 0 < gap20 <= distance:
            return True
    
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Config: MS ON-only (strongest edge from analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def make_ms_on_only():
    """MS with ON-only levels, SMA 8/24, direction=both."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"
    
    # Disable all other setups
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
    
    # Enable MS
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 5.0
    cfg.ms_min_target_pts = 4.0
    cfg.ms_min_rr = 0.5
    cfg.ms_max_risk = 15.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    
    # ON-only levels (strongest edge)
    cfg.ms_use_prev_va = False
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False
    
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def print_trade_group(label, trades, indent=4):
    """Print metrics for a group of trades."""
    prefix = " " * indent
    if not trades:
        print(f"{prefix}{label:<50s}  NO TRADES")
        return None
    
    m = compute_metrics(trades)
    pnls = [t.pnl_dollar for t in trades]
    avg = np.mean(pnls)
    
    # T-test
    if len(pnls) >= 5:
        t_stat, p_val = stats.ttest_1samp(pnls, 0)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
    else:
        p_val = float('nan')
        sig = "n/a"
    
    print(f"{prefix}{label:<50s}  {m.total_trades:>4d} trades  "
          f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+10,.0f}  Avg ${avg:>+8,.0f}  "
          f"p={p_val:.4f} {sig}")
    
    return m


def print_direction_breakdown(label_prefix, trades, indent=6):
    """Print long/short breakdown for a group of trades."""
    prefix = " " * indent
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    
    if longs:
        m = compute_metrics(longs)
        pnls = [t.pnl_dollar for t in longs]
        avg = np.mean(pnls)
        print(f"{prefix}LONGS:  {m.total_trades:>4d} trades  "
              f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
              f"P&L ${m.net_pnl:>+10,.0f}  Avg ${avg:>+8,.0f}")
    else:
        print(f"{prefix}LONGS:  NO TRADES")
    
    if shorts:
        m = compute_metrics(shorts)
        pnls = [t.pnl_dollar for t in shorts]
        avg = np.mean(pnls)
        print(f"{prefix}SHORTS: {m.total_trades:>4d} trades  "
              f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
              f"P&L ${m.net_pnl:>+10,.0f}  Avg ${avg:>+8,.0f}")
    else:
        print(f"{prefix}SHORTS: NO TRADES")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "es_5m_databento_2yr.csv")
    
    print("=" * 120)
    print("  DAILY MA FILTER TEST FOR MS SHORT TRADES")
    print("  Hypothesis: Daily SMA(5) or SMA(20) between entry and target blocks short targets")
    print("=" * 120)
    
    # Load data
    print(f"\nLoading {data_path}...")
    df = load_tos_csv(data_path, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Compute daily SMAs
    print("\nComputing daily SMAs (using previous day's close, no lookahead)...")
    add_daily_sma_columns(df)
    
    # Run MS backtest with ON-only levels
    print("\nRunning MS backtest (ON-only levels, SMA 8/24, both directions)...")
    cfg = make_ms_on_only()
    trades = run_backtest(df.copy(), cfg)
    print(f"Total trades: {len(trades)}")
    
    # ═══════════════════════════════════════════════════════════════
    #  PART 1: Diagnostic — split by MA between entry/target
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 120)
    print("  PART 1: DIAGNOSTIC — Daily MA Between Entry and Target")
    print("  For shorts: does SMA sit between entry (above) and target (below)?")
    print("  For longs: does SMA sit between entry (below) and target (above)?")
    print("=" * 120)
    
    # Classify each trade
    trades_no_ma = []        # No daily MA between entry/target
    trades_with_ma = []      # At least one daily MA between entry/target
    trades_sma5_only = []    # SMA5 between (not SMA20)
    trades_sma20_only = []   # SMA20 between (not SMA5)
    trades_both_ma = []      # Both SMA5 and SMA20 between
    trades_skipped = 0       # Could not classify (NaN SMA)
    
    for trade in trades:
        sma5_between, sma20_between, sma5_val, sma20_val = daily_ma_between_entry_target(trade, df)
        
        if np.isnan(sma5_val if sma5_val is not None else np.nan) and np.isnan(sma20_val if sma20_val is not None else np.nan):
            trades_skipped += 1
            continue
        
        any_between = sma5_between or sma20_between
        
        if any_between:
            trades_with_ma.append(trade)
            if sma5_between and sma20_between:
                trades_both_ma.append(trade)
            elif sma5_between:
                trades_sma5_only.append(trade)
            else:
                trades_sma20_only.append(trade)
        else:
            trades_no_ma.append(trade)
    
    if trades_skipped > 0:
        print(f"\n  (Skipped {trades_skipped} trades with no daily SMA data)")
    
    print(f"\n  --- ALL TRADES (baseline) ---")
    print_trade_group("All trades", trades)
    print_direction_breakdown("All", trades)
    
    print(f"\n  --- NO DAILY MA between entry/target ---")
    print_trade_group("No MA blocking target path", trades_no_ma)
    print_direction_breakdown("No MA", trades_no_ma)
    
    print(f"\n  --- DAILY MA IS between entry/target ---")
    print_trade_group("MA blocking target path (any)", trades_with_ma)
    print_direction_breakdown("MA blocking", trades_with_ma)
    
    # Sub-breakdown of which MA is between
    if trades_sma5_only or trades_sma20_only or trades_both_ma:
        print(f"\n  --- MA breakdown (which MA is between entry/target) ---")
        print_trade_group("SMA5 only between", trades_sma5_only)
        print_trade_group("SMA20 only between", trades_sma20_only)
        print_trade_group("BOTH SMA5 and SMA20 between", trades_both_ma)
    
    # Shorts-only diagnostic (the user's observation)
    print(f"\n  --- SHORTS ONLY: Diagnostic ---")
    all_shorts = [t for t in trades if t.direction == -1]
    shorts_no_ma = [t for t in trades_no_ma if t.direction == -1]
    shorts_with_ma = [t for t in trades_with_ma if t.direction == -1]
    
    print_trade_group("All shorts", all_shorts)
    print_trade_group("Shorts — no MA blocking", shorts_no_ma)
    print_trade_group("Shorts — MA blocking target", shorts_with_ma)
    
    # Win rate comparison
    if shorts_no_ma and shorts_with_ma:
        wr_no = sum(1 for t in shorts_no_ma if t.pnl_dollar > 0) / len(shorts_no_ma) * 100
        wr_with = sum(1 for t in shorts_with_ma if t.pnl_dollar > 0) / len(shorts_with_ma) * 100
        print(f"\n    Win Rate delta: {wr_no:.1f}% (no MA) vs {wr_with:.1f}% (MA blocking) = {wr_no - wr_with:+.1f}pp")
        
        # Are the two groups statistically different?
        pnls_no = [t.pnl_dollar for t in shorts_no_ma]
        pnls_with = [t.pnl_dollar for t in shorts_with_ma]
        if len(pnls_no) >= 5 and len(pnls_with) >= 5:
            t_stat, p_val = stats.ttest_ind(pnls_no, pnls_with, equal_var=False)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"    Two-sample t-test (no-MA vs MA-blocking): t={t_stat:.3f}, p={p_val:.4f} {sig}")
    
    # Exit reason breakdown for MA-blocked shorts
    if shorts_with_ma:
        print(f"\n    Exit reasons for MA-blocked shorts:")
        reasons = {}
        for t in shorts_with_ma:
            r = t.exit_reason
            if r not in reasons:
                reasons[r] = {"count": 0, "pnl": 0}
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t.pnl_dollar
        for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            print(f"      {r:<12s}  {d['count']:>4d}  ${d['pnl']:>+10,.0f}")
    
    # ═══════════════════════════════════════════════════════════════
    #  PART 2: Applied Filter — Skip shorts when daily MA is close below entry
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 120)
    print("  PART 2: APPLIED FILTER — Skip shorts when daily SMA is within N pts below entry")
    print("  Logic: if SMA5 or SMA20 is 0 < (entry - SMA) <= N, skip the short")
    print("=" * 120)
    
    filter_distances = [3, 5, 8, 10, 15, 20]
    
    print(f"\n  {'Filter':<25s} {'Total':>6s} {'Shorts':>7s} {'Longs':>6s} "
          f"{'Blocked':>8s} {'WR':>6s} {'PF':>7s} {'P&L':>12s} {'DD':>10s} "
          f"{'Sharpe':>7s} {'p-val':>8s}")
    print(f"  {'-'*115}")
    
    # Baseline (no filter)
    m_base = compute_metrics(trades)
    pnls_base = [t.pnl_dollar for t in trades]
    _, p_base = stats.ttest_1samp(pnls_base, 0) if len(pnls_base) >= 5 else (0, float('nan'))
    n_shorts = len([t for t in trades if t.direction == -1])
    n_longs = len([t for t in trades if t.direction == 1])
    print(f"  {'No filter (baseline)':<25s} {m_base.total_trades:>6d} {n_shorts:>7d} {n_longs:>6d} "
          f"{'0':>8s} {m_base.win_rate:>5.1f}% {m_base.profit_factor:>7.3f} "
          f"{'${:,.0f}'.format(m_base.net_pnl):>12s} {'${:,.0f}'.format(m_base.max_drawdown):>10s} "
          f"{m_base.sharpe:>7.2f} {p_base:>8.4f}")
    
    best_pf = m_base.profit_factor
    best_dist = 0
    best_trades = trades
    
    for dist in filter_distances:
        # Filter: remove shorts blocked by daily MA proximity
        filtered_trades = []
        blocked_count = 0
        
        for trade in trades:
            if daily_ma_proximity_below(trade, df, dist):
                blocked_count += 1
            else:
                filtered_trades.append(trade)
        
        if not filtered_trades:
            print(f"  {'N=' + str(dist) + ' pts':<25s}  ALL TRADES BLOCKED")
            continue
        
        m = compute_metrics(filtered_trades)
        pnls = [t.pnl_dollar for t in filtered_trades]
        _, p_val = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, float('nan'))
        
        n_s = len([t for t in filtered_trades if t.direction == -1])
        n_l = len([t for t in filtered_trades if t.direction == 1])
        
        marker = ""
        if m.profit_factor > best_pf:
            best_pf = m.profit_factor
            best_dist = dist
            best_trades = filtered_trades
            marker = " <-- BEST PF"
        
        print(f"  {'N=' + str(dist) + ' pts':<25s} {m.total_trades:>6d} {n_s:>7d} {n_l:>6d} "
              f"{blocked_count:>8d} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
              f"{'${:,.0f}'.format(m.net_pnl):>12s} {'${:,.0f}'.format(m.max_drawdown):>10s} "
              f"{m.sharpe:>7.2f} {p_val:>8.4f}{marker}")
    
    # ═══════════════════════════════════════════════════════════════
    #  PART 3: Detailed breakdown of best filter
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n" + "=" * 120)
    if best_dist > 0:
        print(f"  PART 3: BEST FILTER DETAIL — N={best_dist} pts")
    else:
        print(f"  PART 3: No filter improved PF — showing baseline detail")
    print("=" * 120)
    
    if best_dist > 0:
        print(f"\n  Filtered trades (N={best_dist}):")
        print_trade_group("Filtered total", best_trades, indent=4)
        print_direction_breakdown("Filtered", best_trades)
        
        # What was blocked?
        blocked = [t for t in trades if t not in best_trades]
        print(f"\n  Blocked trades:")
        print_trade_group("Blocked total", blocked, indent=4)
        if blocked:
            bl_m = compute_metrics(blocked)
            print(f"    -> These {len(blocked)} blocked trades had avg P&L of ${np.mean([t.pnl_dollar for t in blocked]):+,.0f}")
            print(f"    -> If blocked trades were NET LOSERS, the filter is working correctly")
        
        # Setup breakdown for filtered trades
        print(f"\n  Setup breakdown (filtered):")
        setup_groups = {}
        for t in best_trades:
            setup_groups.setdefault(t.setup, []).append(t)
        for setup, strades in sorted(setup_groups.items()):
            print_trade_group(f"  {setup}", strades, indent=4)
    else:
        print(f"\n  Baseline trades (no filter):")
        print_trade_group("All trades", trades, indent=4)
        print_direction_breakdown("All", trades)
    
    # ═══════════════════════════════════════════════════════════════
    #  PART 4: Longs diagnostic (do daily MAs also matter for longs?)
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n" + "=" * 120)
    print(f"  PART 4: LONGS DIAGNOSTIC — Do daily MAs act as resistance for longs?")
    print("=" * 120)
    
    all_longs = [t for t in trades if t.direction == 1]
    longs_no_ma = [t for t in trades_no_ma if t.direction == 1]
    longs_with_ma = [t for t in trades_with_ma if t.direction == 1]
    
    print_trade_group("All longs", all_longs)
    print_trade_group("Longs — no MA blocking", longs_no_ma)
    print_trade_group("Longs — MA blocking target", longs_with_ma)
    
    if longs_no_ma and longs_with_ma:
        wr_no = sum(1 for t in longs_no_ma if t.pnl_dollar > 0) / len(longs_no_ma) * 100
        wr_with = sum(1 for t in longs_with_ma if t.pnl_dollar > 0) / len(longs_with_ma) * 100
        print(f"\n    Win Rate delta: {wr_no:.1f}% (no MA) vs {wr_with:.1f}% (MA blocking) = {wr_no - wr_with:+.1f}pp")
    
    # ═══════════════════════════════════════════════════════════════
    #  PART 5: Sample blocked trades (for manual inspection)
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n" + "=" * 120)
    print(f"  PART 5: SAMPLE TRADES — Shorts with daily MA between entry/target")
    print("=" * 120)
    
    if shorts_with_ma:
        # Show up to 15 examples
        print(f"\n  {'Date':<22s} {'Setup':<12s} {'Dir':>4s} {'Entry':>8s} {'Target':>8s} "
              f"{'SMA5':>8s} {'SMA20':>8s} {'Exit':>8s} {'Reason':<10s} {'P&L':>10s}")
        print(f"  {'-'*110}")
        
        for t in shorts_with_ma[:15]:
            sma5_btwn, sma20_btwn, sma5_val, sma20_val = daily_ma_between_entry_target(t, df)
            sma5_str = f"{sma5_val:.1f}" if not np.isnan(sma5_val) else "N/A"
            sma20_str = f"{sma20_val:.1f}" if not np.isnan(sma20_val) else "N/A"
            dir_str = "S" if t.direction == -1 else "L"
            
            # Mark which MA is between
            markers = []
            if sma5_btwn:
                markers.append("5d")
            if sma20_btwn:
                markers.append("20d")
            marker_str = "+".join(markers)
            
            print(f"  {str(t.entry_time):<22s} {t.setup:<12s} {dir_str:>4s} {t.entry_price:>8.1f} "
                  f"{t.target:>8.1f} {sma5_str:>8s} {sma20_str:>8s} {t.exit_price:>8.1f} "
                  f"{t.exit_reason:<10s} ${t.pnl_dollar:>+9,.0f}  [{marker_str}]")
        
        if len(shorts_with_ma) > 15:
            print(f"  ... and {len(shorts_with_ma) - 15} more")
    else:
        print(f"\n  No shorts found with daily MA between entry and target.")
    
    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n" + "=" * 120)
    print(f"  SUMMARY")
    print("=" * 120)
    
    print(f"\n  MS ON-only baseline: {len(trades)} trades, WR {m_base.win_rate:.1f}%, "
          f"PF {m_base.profit_factor:.3f}, P&L ${m_base.net_pnl:+,.0f}")
    
    if trades_with_ma:
        pct_blocked = len(trades_with_ma) / len(trades) * 100
        print(f"\n  Trades with daily MA between entry/target: {len(trades_with_ma)} ({pct_blocked:.1f}%)")
        
        if shorts_with_ma:
            sh_m = compute_metrics(shorts_with_ma)
            print(f"  Shorts with MA blocking: {len(shorts_with_ma)} trades, "
                  f"WR {sh_m.win_rate:.1f}%, PF {sh_m.profit_factor:.3f}, "
                  f"P&L ${sh_m.net_pnl:+,.0f}, Avg ${np.mean([t.pnl_dollar for t in shorts_with_ma]):+,.0f}/trade")
    
    if best_dist > 0:
        bm = compute_metrics(best_trades)
        print(f"\n  Best proximity filter: N={best_dist} pts")
        print(f"  Filtered: {bm.total_trades} trades, WR {bm.win_rate:.1f}%, "
              f"PF {bm.profit_factor:.3f}, P&L ${bm.net_pnl:+,.0f}")
        
        blocked_trades = [t for t in trades if t not in best_trades]
        if blocked_trades:
            print(f"  Blocked: {len(blocked_trades)} trades, "
                  f"Avg P&L ${np.mean([t.pnl_dollar for t in blocked_trades]):+,.0f}/trade")
    else:
        print(f"\n  No proximity filter improved PF over baseline.")
    
    print(f"\n  VERDICT: ", end="")
    if best_dist > 0 and compute_metrics(best_trades).profit_factor > m_base.profit_factor * 1.05:
        print(f"Daily MA filter at N={best_dist} pts IMPROVES the strategy.")
        print(f"  Implement: skip shorts when daily SMA(5) or SMA(20) is within {best_dist} pts below entry.")
    elif trades_with_ma and compute_metrics(trades_with_ma).profit_factor < 0.9:
        print(f"Daily MA DOES act as support — MA-blocked trades underperform significantly.")
        print(f"  But proximity filter may need tuning. Consider a tighter zone or per-MA filter.")
    else:
        print(f"Daily MA filter does NOT show a clear edge improvement.")
        print(f"  The hypothesis may not hold, or the sample size is too small.")
    
    print()


if __name__ == "__main__":
    main()
