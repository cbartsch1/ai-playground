#!/usr/bin/env python3
"""Catalyst ES — Port of SPX Catalyst strategy to ES futures.

Catalyst detects 30m structure breaks:
  - Lower-high + lower-low on 30m bars
  - Range >= threshold (22 SPX pts ~ 22 ES pts)
  - Bar closes below prior 30m bar's low
  - SHORT ONLY
  - Exit: first qualified green 5m bar (skip doji greens + first 2 greens after entry)
  - OR fixed-point target alternative exit
  - Entry window: configurable, max 1/day
  - Stop: percentage-based (bps)

Standalone script — does NOT modify any existing backtester files.
Imports data_loader and metrics only.
"""

import os, sys, math, time as _time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtester.data_loader import load_tos_csv
from backtester.indicators import compute_indicators
from backtester.metrics import compute_metrics, Metrics
from backtester.position import Trade

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# Pre-compute 30m bars ONCE from the 5m DataFrame
# =============================================================================

@dataclass
class Bar30m:
    open: float; high: float; low: float; close: float
    bar_count: int; et_start: int; session_date: object
    first_5m_idx: int  # index into original 5m DataFrame for entry timing

def build_30m_bars(df: pd.DataFrame) -> List[Bar30m]:
    """Pre-compute all 30m bars from 5m data. Run once."""
    bars_30m = []
    cur_high = cur_low = cur_open = cur_close = float("nan")
    cur_count = 0
    cur_period = -1
    cur_et_start = 0
    cur_session = None
    cur_first_idx = 0

    for i in range(len(df)):
        row = df.iloc[i]
        et_time = int(row["et_time"])
        is_rth = bool(row["is_rth"])
        if not is_rth or et_time < 930 or et_time >= 1600:
            continue

        minutes_from_open = (et_time // 100 - 9) * 60 + (et_time % 100) - 30
        period = minutes_from_open // 30
        session_date = row.get("session_date", None)

        # Session change or period boundary -> finalize
        new_session = (session_date != cur_session and cur_session is not None)
        new_period = (period != cur_period and cur_period >= 0)

        if new_session or new_period:
            if cur_count > 0 and not math.isnan(cur_high):
                bars_30m.append(Bar30m(
                    open=cur_open, high=cur_high, low=cur_low, close=cur_close,
                    bar_count=cur_count, et_start=cur_et_start,
                    session_date=cur_session, first_5m_idx=cur_first_idx,
                ))
            if new_session:
                cur_session = session_date
            cur_open = float(row["open"])
            cur_high = float(row["high"])
            cur_low = float(row["low"])
            cur_close = float(row["close"])
            cur_count = 1
            cur_period = period
            cur_et_start = et_time
            cur_first_idx = i
        elif cur_period < 0:
            # First bar
            cur_session = session_date
            cur_open = float(row["open"])
            cur_high = float(row["high"])
            cur_low = float(row["low"])
            cur_close = float(row["close"])
            cur_count = 1
            cur_period = period
            cur_et_start = et_time
            cur_first_idx = i
        else:
            # Accumulate
            cur_high = max(cur_high, float(row["high"]))
            cur_low = min(cur_low, float(row["low"]))
            cur_close = float(row["close"])
            cur_count += 1

    # Finalize last
    if cur_count > 0 and not math.isnan(cur_high):
        bars_30m.append(Bar30m(
            open=cur_open, high=cur_high, low=cur_low, close=cur_close,
            bar_count=cur_count, et_start=cur_et_start,
            session_date=cur_session, first_5m_idx=cur_first_idx,
        ))

    return bars_30m


def find_signal_bars(bars_30m: List[Bar30m], range_threshold: float,
                     require_close_below: bool = True) -> List[int]:
    """Find all 30m bar indices where Catalyst signal fires.

    Returns list of indices into bars_30m.
    """
    signals = []
    for i in range(1, len(bars_30m)):
        curr = bars_30m[i]
        prev = bars_30m[i - 1]

        if curr.session_date != prev.session_date:
            continue

        # LH + LL
        if not (curr.high < prev.high and curr.low < prev.low):
            continue

        # Range threshold
        if (curr.high - curr.low) < range_threshold:
            continue

        # Close below prior 30m bar's low
        if require_close_below and curr.close >= prev.low:
            continue

        signals.append(i)

    return signals


# =============================================================================
# Backtest on pre-computed data
# =============================================================================

POINT_VALUE = 50.0
COMMISSION = 2.50
SLIPPAGE_PTS = 0.25

def run_backtest_green_exit(df: pd.DataFrame, bars_30m: List[Bar30m],
                            signal_indices: List[int],
                            stop_bps: float = 30.0,
                            entry_start: int = 1000,
                            entry_end: int = 1300,
                            flatten_time: int = 1555,
                            max_per_day: int = 1,
                            skip_friday: bool = True,
                            blackout_start: int = 0,
                            blackout_end: int = 0,
                            doji_threshold: float = 1.0,
                            skip_first_n_greens: int = 2) -> List[Trade]:
    """Run Catalyst with green-bar exit on pre-computed signals."""
    trades = []
    # Build signal lookup: session_date -> list of (30m_bar_index, 30m_bar)
    signals_by_session = defaultdict(list)
    for si in signal_indices:
        b = bars_30m[si]
        signals_by_session[b.session_date].append((si, b))

    et_times = df["et_time"].values
    is_rth = df["is_rth"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    session_dates = df["session_date"].values
    weekdays = df["weekday"].values
    dt_index = df.index

    n = len(df)
    i = 0
    while i < n:
        if not is_rth[i]:
            i += 1
            continue

        et = int(et_times[i])
        sd = session_dates[i]
        wd = int(weekdays[i])

        # Check if this session has any signals
        if sd not in signals_by_session:
            i += 1
            continue

        # Find if a signal has fired by now (30m bar completed before this 5m bar)
        signal_available = False
        for si, sb in signals_by_session[sd]:
            # The signal bar's first_5m_idx is where the 30m bar STARTED collecting.
            # The signal fires when the 30m bar COMPLETES. That means the next 30m
            # period's first bar is when we can trade. We approximate: the 30m bar's
            # last 5m bar index ~ first_5m_idx + bar_count - 1.
            # We can enter on any 5m bar after that.
            signal_complete_idx = sb.first_5m_idx + sb.bar_count
            if i >= signal_complete_idx:
                signal_available = True
                break

        if not signal_available:
            i += 1
            continue

        # Entry filters
        if et < entry_start or et >= entry_end:
            i += 1
            continue
        if skip_friday and wd == 4:
            i += 1
            continue
        if blackout_start > 0 and blackout_end > 0 and blackout_start <= et < blackout_end:
            i += 1
            continue

        # ENTER SHORT
        entry_price = closes[i] + SLIPPAGE_PTS
        stop_price = entry_price + (entry_price * stop_bps / 10000.0)
        entry_time = dt_index[i]
        entry_session = sd
        green_count = 0

        # Mark signal as consumed for this session
        # (max 1/day: remove all signals for this session)
        del signals_by_session[sd]

        # Walk forward through bars for exit
        j = i + 1
        trade_done = False
        while j < n:
            if not is_rth[j]:
                j += 1
                continue
            # Different session? Force close at last RTH bar
            if session_dates[j] != entry_session:
                # Close at previous bar's close
                k = j - 1
                while k > i and not is_rth[k]:
                    k -= 1
                exit_price = closes[k]
                pnl = entry_price - exit_price
                trades.append(Trade(
                    setup="CAT", direction=-1,
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=dt_index[k], exit_price=exit_price,
                    exit_reason="flatten", pnl_pts=pnl,
                    pnl_dollar=pnl * POINT_VALUE - COMMISSION * 2,
                    stop=stop_price, target=0.0,
                ))
                trade_done = True
                i = j
                break

            et_j = int(et_times[j])

            # Stop check
            if highs[j] >= stop_price:
                pnl = entry_price - stop_price
                trades.append(Trade(
                    setup="CAT", direction=-1,
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=dt_index[j], exit_price=stop_price,
                    exit_reason="stop", pnl_pts=pnl,
                    pnl_dollar=pnl * POINT_VALUE - COMMISSION * 2,
                    stop=stop_price, target=0.0,
                ))
                trade_done = True
                i = j + 1
                break

            # Green bar exit
            is_green = closes[j] > opens[j]
            bar_range = highs[j] - lows[j]
            is_doji = bar_range < doji_threshold

            if is_green and not is_doji:
                green_count += 1
                if green_count > skip_first_n_greens:
                    exit_price = closes[j]
                    pnl = entry_price - exit_price
                    trades.append(Trade(
                        setup="CAT", direction=-1,
                        entry_time=entry_time, entry_price=entry_price,
                        exit_time=dt_index[j], exit_price=exit_price,
                        exit_reason="green_bar", pnl_pts=pnl,
                        pnl_dollar=pnl * POINT_VALUE - COMMISSION * 2,
                        stop=stop_price, target=0.0,
                    ))
                    trade_done = True
                    i = j + 1
                    break

            # Flatten
            if et_j >= flatten_time:
                exit_price = closes[j]
                pnl = entry_price - exit_price
                trades.append(Trade(
                    setup="CAT", direction=-1,
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=dt_index[j], exit_price=exit_price,
                    exit_reason="flatten", pnl_pts=pnl,
                    pnl_dollar=pnl * POINT_VALUE - COMMISSION * 2,
                    stop=stop_price, target=0.0,
                ))
                trade_done = True
                i = j + 1
                break

            j += 1

        if not trade_done:
            # End of data
            exit_price = closes[n - 1]
            pnl = entry_price - exit_price
            trades.append(Trade(
                setup="CAT", direction=-1,
                entry_time=entry_time, entry_price=entry_price,
                exit_time=dt_index[n - 1], exit_price=exit_price,
                exit_reason="data_end", pnl_pts=pnl,
                pnl_dollar=pnl * POINT_VALUE - COMMISSION * 2,
                stop=stop_price, target=0.0,
            ))
            break
        if i <= j:
            i = j

    return trades


def run_backtest_target_exit(df: pd.DataFrame, bars_30m: List[Bar30m],
                             signal_indices: List[int],
                             target_pts: float = 15.0,
                             stop_bps: float = 30.0,
                             entry_start: int = 1000,
                             entry_end: int = 1300,
                             flatten_time: int = 1555,
                             max_per_day: int = 1,
                             skip_friday: bool = True,
                             blackout_start: int = 0,
                             blackout_end: int = 0) -> List[Trade]:
    """Run Catalyst with fixed-point target exit."""
    trades = []
    signals_by_session = defaultdict(list)
    for si in signal_indices:
        b = bars_30m[si]
        signals_by_session[b.session_date].append((si, b))

    et_times = df["et_time"].values
    is_rth = df["is_rth"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    session_dates = df["session_date"].values
    weekdays = df["weekday"].values
    dt_index = df.index

    n = len(df)
    i = 0
    while i < n:
        if not is_rth[i]:
            i += 1
            continue

        et = int(et_times[i])
        sd = session_dates[i]
        wd = int(weekdays[i])

        if sd not in signals_by_session:
            i += 1
            continue

        signal_available = False
        for si, sb in signals_by_session[sd]:
            signal_complete_idx = sb.first_5m_idx + sb.bar_count
            if i >= signal_complete_idx:
                signal_available = True
                break

        if not signal_available:
            i += 1
            continue

        if et < entry_start or et >= entry_end:
            i += 1
            continue
        if skip_friday and wd == 4:
            i += 1
            continue
        if blackout_start > 0 and blackout_end > 0 and blackout_start <= et < blackout_end:
            i += 1
            continue

        entry_price = closes[i] + SLIPPAGE_PTS
        stop_price = entry_price + (entry_price * stop_bps / 10000.0)
        target_price = entry_price - target_pts
        entry_time = dt_index[i]
        entry_session = sd

        del signals_by_session[sd]

        j = i + 1
        trade_done = False
        while j < n:
            if not is_rth[j]:
                j += 1
                continue
            if session_dates[j] != entry_session:
                k = j - 1
                while k > i and not is_rth[k]:
                    k -= 1
                exit_price = closes[k]
                pnl = entry_price - exit_price
                trades.append(Trade(
                    setup="CAT_T", direction=-1,
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=dt_index[k], exit_price=exit_price,
                    exit_reason="flatten", pnl_pts=pnl,
                    pnl_dollar=pnl * POINT_VALUE - COMMISSION * 2,
                    stop=stop_price, target=target_price,
                ))
                trade_done = True
                i = j
                break

            et_j = int(et_times[j])

            stop_hit = highs[j] >= stop_price
            target_hit = lows[j] <= target_price

            if stop_hit and target_hit:
                # Pessimistic: stop wins
                exit_price = stop_price
                reason = "stop"
            elif stop_hit:
                exit_price = stop_price
                reason = "stop"
            elif target_hit:
                exit_price = target_price
                reason = "target"
            elif et_j >= flatten_time:
                exit_price = closes[j]
                reason = "flatten"
            else:
                j += 1
                continue

            pnl = entry_price - exit_price
            trades.append(Trade(
                setup="CAT_T", direction=-1,
                entry_time=entry_time, entry_price=entry_price,
                exit_time=dt_index[j], exit_price=exit_price,
                exit_reason=reason, pnl_pts=pnl,
                pnl_dollar=pnl * POINT_VALUE - COMMISSION * 2,
                stop=stop_price, target=target_price,
            ))
            trade_done = True
            i = j + 1
            break

        if not trade_done:
            exit_price = closes[n - 1]
            pnl = entry_price - exit_price
            trades.append(Trade(
                setup="CAT_T", direction=-1,
                entry_time=entry_time, entry_price=entry_price,
                exit_time=dt_index[n - 1], exit_price=exit_price,
                exit_reason="data_end", pnl_pts=pnl,
                pnl_dollar=pnl * POINT_VALUE - COMMISSION * 2,
                stop=stop_price, target=target_price,
            ))
            break
        if i <= j:
            i = j

    return trades


# =============================================================================
# Significance + Walk-Forward
# =============================================================================

WF_SPLIT = "2025-02-16"

def run_significance(trades, seed=42):
    pnls = np.array([t.pnl_dollar for t in trades])
    n = len(pnls)
    if n < 5:
        return 1.0, 1.0, 0.0
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    if np.isnan(t_pval):
        t_pval = 1.0
    rng = np.random.default_rng(seed)
    obs = np.sum(pnls)
    abs_p = np.abs(pnls)
    count = sum(1 for _ in range(5000) if np.dot(rng.choice([-1.0, 1.0], size=n), abs_p) >= obs)
    perm_pval = count / 5000
    boot = np.array([np.sum(rng.choice(pnls, size=n, replace=True)) for _ in range(5000)])
    prob = float(np.mean(boot > 0))
    return t_pval, perm_pval, prob


def print_row(label, m):
    pf = f"{m.profit_factor:.3f}" if m.total_trades > 0 else "  N/A"
    wr = f"{m.win_rate:.1f}%" if m.total_trades > 0 else " N/A"
    sh = f"{m.sharpe:.2f}" if m.total_trades > 0 else " N/A"
    dd = f"${m.max_drawdown:,.0f}" if m.total_trades > 0 else "N/A"
    print(f"  {label:<40} {m.total_trades:>5} {wr:>7} {pf:>8} ${m.net_pnl:>10,.0f} {sh:>7} {dd:>9}")


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = _time.time()
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "es_5m_databento_2yr.csv"
    )
    print(f"Loading {data_path}...")
    df = load_tos_csv(data_path, instrument="ES")
    print(f"Loaded {len(df):,} bars: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    compute_indicators(df)
    print(f"Data loaded + indicators in {_time.time()-t0:.1f}s")

    # Pre-compute 30m bars once
    print("Building 30m bars...")
    bars_30m = build_30m_bars(df)
    print(f"Built {len(bars_30m)} 30m bars")

    # Pre-compute signals for various thresholds
    print("Pre-computing signals for all thresholds...")
    thresholds = [8, 10, 12, 15, 18, 20, 22, 25, 28, 30]
    signals_with_close = {}
    signals_no_close = {}
    for th in thresholds:
        signals_with_close[th] = find_signal_bars(bars_30m, th, require_close_below=True)
        signals_no_close[th] = find_signal_bars(bars_30m, th, require_close_below=False)
        print(f"  Range>={th}pt: {len(signals_with_close[th])} signals (close<low) | {len(signals_no_close[th])} (any close)")

    # =========================================================================
    # PHASE 1: Range Threshold Sweep (Green Bar Exit, original Catalyst params)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 1: RANGE THRESHOLD SWEEP — Green Bar Exit, 10:00-13:00")
    print(f"{'='*80}")
    print(f"  {'Config':<40} {'Trds':>5} {'WR':>7} {'PF':>8} {'Net P&L':>11} {'Sharpe':>7} {'DD':>9}")
    print(f"  {'-'*85}")

    for th in thresholds:
        trades = run_backtest_green_exit(df, bars_30m, signals_with_close[th],
                                         stop_bps=30, entry_start=1000, entry_end=1300)
        m = compute_metrics(trades)
        print_row(f"Range>={th}pt close<low GreenExit", m)

    # Also test without close < prior low
    print(f"\n  --- Without close < prior low requirement ---")
    for th in [15, 18, 20, 22, 25]:
        trades = run_backtest_green_exit(df, bars_30m, signals_no_close[th],
                                         stop_bps=30, entry_start=1000, entry_end=1300)
        m = compute_metrics(trades)
        print_row(f"Range>={th}pt ANY close GreenExit", m)

    # =========================================================================
    # PHASE 2: Stop BPS Sweep (best range threshold from Phase 1)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 2: STOP BPS SWEEP — Green Bar Exit")
    print(f"{'='*80}")
    print(f"  {'Config':<40} {'Trds':>5} {'WR':>7} {'PF':>8} {'Net P&L':>11} {'Sharpe':>7} {'DD':>9}")
    print(f"  {'-'*85}")

    # Test multiple thresholds x stops
    for th in [15, 18, 20, 22]:
        for stop in [20, 25, 30, 35, 40, 50]:
            trades = run_backtest_green_exit(df, bars_30m, signals_with_close[th],
                                             stop_bps=stop, entry_start=1000, entry_end=1300)
            m = compute_metrics(trades)
            print_row(f"R{th} S{stop}bps close<low GreenExit", m)

    # Also without close requirement
    for th in [15, 18, 20]:
        for stop in [25, 30, 35]:
            trades = run_backtest_green_exit(df, bars_30m, signals_no_close[th],
                                             stop_bps=stop, entry_start=1000, entry_end=1300)
            m = compute_metrics(trades)
            print_row(f"R{th} S{stop}bps ANY GreenExit", m)

    # =========================================================================
    # PHASE 3: Entry Window Sweep
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 3: ENTRY WINDOW SWEEP — Green Bar Exit, Stop=30bps")
    print(f"{'='*80}")
    print(f"  {'Config':<40} {'Trds':>5} {'WR':>7} {'PF':>8} {'Net P&L':>11} {'Sharpe':>7} {'DD':>9}")
    print(f"  {'-'*85}")

    windows = [(935, 1300), (935, 1400), (935, 1500), (1000, 1300),
               (1000, 1400), (1000, 1500), (1030, 1300), (1030, 1400)]

    for th in [15, 18, 20, 22]:
        for start, end in windows:
            for close_req, sigs in [("CL", signals_with_close), ("ANY", signals_no_close)]:
                trades = run_backtest_green_exit(df, bars_30m, sigs[th],
                                                 stop_bps=30, entry_start=start, entry_end=end)
                m = compute_metrics(trades)
                if m.total_trades > 0:
                    wlabel = f"{start//100}:{start%100:02d}-{end//100}:{end%100:02d}"
                    print_row(f"R{th} {close_req} {wlabel} GreenExit", m)

    # =========================================================================
    # PHASE 4: Target-Based Exit Sweep
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 4: TARGET EXIT SWEEP — Fixed target, Stop=30bps, 10:00-13:00")
    print(f"{'='*80}")
    print(f"  {'Config':<40} {'Trds':>5} {'WR':>7} {'PF':>8} {'Net P&L':>11} {'Sharpe':>7} {'DD':>9}")
    print(f"  {'-'*85}")

    for th in [15, 18, 20, 22]:
        for tgt in [8, 10, 12, 15, 20, 25]:
            for close_req, sigs in [("CL", signals_with_close), ("ANY", signals_no_close)]:
                trades = run_backtest_target_exit(df, bars_30m, sigs[th],
                                                  target_pts=tgt, stop_bps=30,
                                                  entry_start=1000, entry_end=1300)
                m = compute_metrics(trades)
                if m.total_trades > 0:
                    print_row(f"R{th} {close_req} Tgt{tgt} S30 TgtExit", m)

    # Wider windows with target exit
    for th in [15, 18, 20]:
        for tgt in [10, 15, 20]:
            for start, end in [(935, 1400), (935, 1500), (1000, 1400)]:
                for close_req, sigs in [("CL", signals_with_close), ("ANY", signals_no_close)]:
                    trades = run_backtest_target_exit(df, bars_30m, sigs[th],
                                                      target_pts=tgt, stop_bps=30,
                                                      entry_start=start, entry_end=end)
                    m = compute_metrics(trades)
                    if m.total_trades > 0:
                        wlabel = f"{start//100}:{start%100:02d}-{end//100}:{end%100:02d}"
                        print_row(f"R{th} {close_req} Tgt{tgt} {wlabel}", m)

    # =========================================================================
    # PHASE 5: Collect ALL results, rank, walk-forward validate top 10
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 5: TOP CONFIGS — WALK-FORWARD VALIDATION")
    print(f"{'='*80}")

    # Re-run top configs and collect results
    all_configs = []

    for th in [10, 12, 15, 18, 20, 22, 25]:
        for stop in [25, 30, 35]:
            for start, end in [(935, 1300), (935, 1400), (1000, 1300), (1000, 1400), (935, 1500)]:
                for close_req, sigs in [("CL", signals_with_close), ("ANY", signals_no_close)]:
                    # Green exit
                    trades = run_backtest_green_exit(df, bars_30m, sigs[th],
                                                     stop_bps=stop, entry_start=start, entry_end=end)
                    m = compute_metrics(trades)
                    if m.total_trades >= 10:
                        all_configs.append({
                            "label": f"R{th} {close_req} S{stop} {start}-{end} Green",
                            "trades": trades, "m": m, "th": th, "stop": stop,
                            "start": start, "end": end, "close_req": close_req,
                            "exit": "green", "tgt": 0,
                        })

                    # Target exits
                    for tgt in [10, 15, 20]:
                        trades = run_backtest_target_exit(df, bars_30m, sigs[th],
                                                          target_pts=tgt, stop_bps=stop,
                                                          entry_start=start, entry_end=end)
                        m = compute_metrics(trades)
                        if m.total_trades >= 10:
                            all_configs.append({
                                "label": f"R{th} {close_req} S{stop} {start}-{end} Tgt{tgt}",
                                "trades": trades, "m": m, "th": th, "stop": stop,
                                "start": start, "end": end, "close_req": close_req,
                                "exit": "target", "tgt": tgt,
                            })

    # Sort by net P&L
    all_configs.sort(key=lambda x: x["m"].net_pnl, reverse=True)

    print(f"\n  Found {len(all_configs)} configs with >= 10 trades")
    print(f"\n  TOP 15 by Net P&L:")
    print(f"  {'Config':<45} {'Trds':>5} {'WR':>7} {'PF':>8} {'Net P&L':>11} {'Sharpe':>7}")
    print(f"  {'-'*85}")
    for c in all_configs[:15]:
        m = c["m"]
        print_row(c["label"], m)

    # Walk-forward on top 10
    print(f"\n  WALK-FORWARD on top 10:")
    print(f"  {'Config':<40} {'IS PF':>7} {'OOS PF':>8} {'Ratio':>7} {'OOS p':>8} {'Result':>8}")
    print(f"  {'-'*80}")

    wf_passing = []
    df_is = df[df.index < WF_SPLIT]
    df_oos = df[df.index >= WF_SPLIT]
    bars_30m_is = build_30m_bars(df_is)
    bars_30m_oos = build_30m_bars(df_oos)

    for c in all_configs[:10]:
        th = c["th"]
        close_below = (c["close_req"] == "CL")

        sigs_is = find_signal_bars(bars_30m_is, th, require_close_below=close_below)
        sigs_oos = find_signal_bars(bars_30m_oos, th, require_close_below=close_below)

        if c["exit"] == "green":
            t_is = run_backtest_green_exit(df_is, bars_30m_is, sigs_is,
                                           stop_bps=c["stop"], entry_start=c["start"], entry_end=c["end"])
            t_oos = run_backtest_green_exit(df_oos, bars_30m_oos, sigs_oos,
                                            stop_bps=c["stop"], entry_start=c["start"], entry_end=c["end"])
        else:
            t_is = run_backtest_target_exit(df_is, bars_30m_is, sigs_is,
                                            target_pts=c["tgt"], stop_bps=c["stop"],
                                            entry_start=c["start"], entry_end=c["end"])
            t_oos = run_backtest_target_exit(df_oos, bars_30m_oos, sigs_oos,
                                             target_pts=c["tgt"], stop_bps=c["stop"],
                                             entry_start=c["start"], entry_end=c["end"])

        m_is = compute_metrics(t_is)
        m_oos = compute_metrics(t_oos)

        pf_ratio = m_oos.profit_factor / m_is.profit_factor if m_is.profit_factor > 0 else 0
        oos_pval = 1.0
        if len(t_oos) >= 5:
            pnls = np.array([t.pnl_dollar for t in t_oos])
            _, oos_pval = scipy_stats.ttest_1samp(pnls, 0)
            if np.isnan(oos_pval):
                oos_pval = 1.0

        passed = pf_ratio >= 0.7 and m_oos.profit_factor > 1.0
        status = "PASS" if passed else "FAIL"

        print(f"  {c['label']:<40} {m_is.profit_factor:>7.3f} {m_oos.profit_factor:>8.3f} "
              f"{pf_ratio:>7.3f} {oos_pval:>8.4f} {status:>8}")

        if passed:
            wf_passing.append({
                **c, "m_is": m_is, "m_oos": m_oos, "pf_ratio": pf_ratio,
                "oos_pval": oos_pval, "t_is": t_is, "t_oos": t_oos,
            })

    # =========================================================================
    # PHASE 6: Full analysis on best walk-forward passing config
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 6: FINAL ANALYSIS")
    print(f"{'='*80}")

    if wf_passing:
        best = max(wf_passing, key=lambda x: x["m_oos"].net_pnl)
        m = best["m"]
        m_is = best["m_is"]
        m_oos = best["m_oos"]
        trades = best["trades"]

        print(f"\n  BEST CONFIG: {best['label']}")
        print(f"  Full Period: {m.total_trades} trades | WR {m.win_rate:.1f}% | PF {m.profit_factor:.3f} | "
              f"${m.net_pnl:+,.0f} | Sharpe {m.sharpe:.2f} | DD ${m.max_drawdown:,.0f}")
        print(f"  Avg Trade: ${m.avg_trade:,.2f} | Avg Win: ${m.avg_win:,.2f} | Avg Loss: ${m.avg_loss:,.2f}")

        # Exit breakdown
        reasons = defaultdict(lambda: {"count": 0, "pnl": 0.0})
        for t in trades:
            reasons[t.exit_reason]["count"] += 1
            reasons[t.exit_reason]["pnl"] += t.pnl_dollar
        print(f"  Exits: ", end="")
        print(" | ".join(f"{r}={d['count']}(${d['pnl']:+,.0f})" for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"])))

        print(f"\n  Walk-Forward:")
        print(f"  IS:  {m_is.total_trades} trades | PF {m_is.profit_factor:.3f} | WR {m_is.win_rate:.1f}% | ${m_is.net_pnl:+,.0f}")
        print(f"  OOS: {m_oos.total_trades} trades | PF {m_oos.profit_factor:.3f} | WR {m_oos.win_rate:.1f}% | ${m_oos.net_pnl:+,.0f}")
        print(f"  PF Ratio: {best['pf_ratio']:.3f} PASS")
        print(f"  OOS p-value: {best['oos_pval']:.4f}")

        # Full-period significance
        t_pval, perm_pval, boot_prob = run_significance(trades)
        print(f"\n  Statistical Significance (full period):")
        stars = '***' if t_pval < 0.01 else '**' if t_pval < 0.05 else '*' if t_pval < 0.10 else ''
        print(f"  t-test p={t_pval:.4f} {stars}")
        print(f"  Permutation p={perm_pval:.4f}")
        print(f"  Bootstrap P(profit)={boot_prob:.1%}")

        # Monthly
        monthly = defaultdict(float)
        monthly_cnt = defaultdict(int)
        for t in trades:
            mo = t.entry_time.strftime("%Y-%m") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:7]
            monthly[mo] += t.pnl_dollar
            monthly_cnt[mo] += 1

        print(f"\n  Monthly P&L:")
        win_m = 0
        for mo in sorted(monthly):
            p = monthly[mo]
            if p > 0: win_m += 1
            print(f"    {mo}: ${p:>+10,.0f} ({monthly_cnt[mo]} trades)")
        tot_m = len(monthly)
        print(f"  Winning months: {win_m}/{tot_m} ({win_m/tot_m*100:.0f}%)" if tot_m else "")

        # Parameter sensitivity
        print(f"\n  Parameter Sensitivity (nearby configs):")
        nearby = [c for c in all_configs if abs(c["th"] - best["th"]) <= 3
                  and abs(c["stop"] - best["stop"]) <= 5
                  and c["exit"] == best["exit"]][:10]
        for c in nearby:
            cm = c["m"]
            print(f"    {c['label']:<45} {cm.total_trades:>4} trades PF {cm.profit_factor:.3f} ${cm.net_pnl:+,.0f}")

        # VERDICT
        significant = t_pval < 0.05
        print(f"\n  {'='*60}")
        if significant and best["pf_ratio"] >= 0.7:
            print(f"  VERDICT: PASS")
            print(f"    Statistically significant (p={t_pval:.4f})")
            print(f"    Walk-forward ratio {best['pf_ratio']:.3f} >= 0.7")
            print(f"    Net P&L ${m.net_pnl:+,.0f} over {m.total_trades} trades")
        elif best["pf_ratio"] >= 0.7 and m.net_pnl > 0:
            print(f"  VERDICT: MARGINAL")
            print(f"    Profitable with WF pass but p={t_pval:.4f} (need < 0.05)")
        else:
            print(f"  VERDICT: FAIL")
            print(f"    p={t_pval:.4f} | PF ratio={best['pf_ratio']:.3f} | Net=${m.net_pnl:+,.0f}")
        print(f"  {'='*60}")

    else:
        print(f"\n  NO CONFIG PASSED WALK-FORWARD VALIDATION")
        print(f"  (OOS PF > 1.0 AND PF ratio >= 0.7)")

        if all_configs:
            best = all_configs[0]
            m = best["m"]
            print(f"\n  Best overall (full period): {best['label']}")
            print(f"  {m.total_trades} trades | WR {m.win_rate:.1f}% | PF {m.profit_factor:.3f} | "
                  f"${m.net_pnl:+,.0f} | Sharpe {m.sharpe:.2f}")

            if m.total_trades >= 5:
                t_pval, _, _ = run_significance(best["trades"])
                print(f"  p-value: {t_pval:.4f}")
        else:
            print(f"\n  NO CONFIGS WITH >= 10 TRADES")
            print(f"  Signal counts by threshold:")
            for th in thresholds:
                print(f"    Range>={th}pt: {len(signals_with_close[th])} (close<low) | {len(signals_no_close[th])} (any)")

        print(f"\n  {'='*60}")
        print(f"  VERDICT: FAIL — Catalyst does not port to ES futures")
        print(f"  {'='*60}")

    print(f"\n  Total runtime: {_time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
