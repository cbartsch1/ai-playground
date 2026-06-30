#!/usr/bin/env python3
"""Cascade ES — 30m Bearish Engulfing ported from SPX BearBreakdown.

Standalone backtest: aggregates 5m bars into 30m, detects bearish engulfing,
enters short on ES futures with percentage-based stop.

Source: ~/projects/backtesting/spx/backtester/strategies/bear_breakdown.py
- 30m bearish engulfing: current red bar body engulfs prior green bar body
- Range >= threshold (20 SPX pts ~ 2 ES pts at 10x ratio, but ES has its
  own volatility profile so we sweep 15-25 ES pts)
- Short only, max 1/day, exit on first green 5m bar or 15:55 flatten

Translation notes:
- SPX strategy buys puts (short exposure). ES version shorts futures directly.
- SPX 20pt range ~ ES 2pt, but ES has different vol characteristics.
  We sweep range thresholds independently.
- Stop: percentage-based (bps), not structure-based.
- No options pricing — futures P&L = (entry - exit) * $50/pt.
"""

import os
import sys
import math
import itertools
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics
from backtester.position import Trade

# ============================================================================
# Constants
# ============================================================================

POINT_VALUE = 50.0       # $50/pt for ES
COMMISSION_RT = 5.0      # $2.50/side = $5 round trip
WF_SPLIT = "2025-02-16"  # Walk-forward split date
FLATTEN_TIME = 1555      # Flatten at 15:55 ET
SEED = 42

# ============================================================================
# 30m Bar Aggregator
# ============================================================================

@dataclass
class Bar30m:
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    et_start: int = 0
    session_date: object = None
    bar_count: int = 0


@dataclass
class Aggregator30m:
    """Accumulates 5m bars into clock-aligned 30m bars.

    30m boundaries: 9:30, 10:00, 10:30, 11:00, ...
    """
    current_open: float = float("nan")
    current_high: float = float("nan")
    current_low: float = float("nan")
    current_close: float = float("nan")
    current_count: int = 0
    current_period: int = -1
    current_et_start: int = 0
    current_session: object = None
    completed: list = field(default_factory=list)

    def feed(self, bar: dict) -> Optional[Bar30m]:
        et_time = bar.get("et_time", 0)
        if et_time < 930 or et_time >= 1600:
            return None

        minutes_from_open = (et_time // 100 - 9) * 60 + (et_time % 100) - 30
        period = minutes_from_open // 30

        session_date = bar.get("session_date", None)

        # Session change
        if session_date != self.current_session and self.current_session is not None:
            completed_bar = self._finalize()
            self._reset()
            self.current_session = session_date
            self._start_new(bar, period, et_time)
            return completed_bar

        self.current_session = session_date

        # Period boundary
        if period != self.current_period and self.current_period >= 0:
            completed_bar = self._finalize()
            self._start_new(bar, period, et_time)
            return completed_bar

        # Same period — accumulate
        if self.current_period < 0:
            self._start_new(bar, period, et_time)
        else:
            self._accumulate(bar)

        return None

    def _start_new(self, bar, period, et_time):
        self.current_period = period
        self.current_open = bar["open"]
        self.current_high = bar["high"]
        self.current_low = bar["low"]
        self.current_close = bar["close"]
        self.current_count = 1
        self.current_et_start = et_time

    def _accumulate(self, bar):
        self.current_high = max(self.current_high, bar["high"])
        self.current_low = min(self.current_low, bar["low"])
        self.current_close = bar["close"]
        self.current_count += 1

    def _finalize(self) -> Optional[Bar30m]:
        if self.current_count == 0 or math.isnan(self.current_high):
            return None
        b = Bar30m(
            open=self.current_open,
            high=self.current_high,
            low=self.current_low,
            close=self.current_close,
            et_start=self.current_et_start,
            session_date=self.current_session,
            bar_count=self.current_count,
        )
        self.completed.append(b)
        if len(self.completed) > 5:
            self.completed = self.completed[-5:]
        return b

    def _reset(self):
        self.current_open = float("nan")
        self.current_high = float("nan")
        self.current_low = float("nan")
        self.current_close = float("nan")
        self.current_count = 0
        self.current_period = -1
        self.current_et_start = 0


# ============================================================================
# Cascade ES Backtest
# ============================================================================

@dataclass
class CascadeConfig:
    range_threshold: float = 20.0   # Min 30m bar range in ES points
    stop_bps: float = 30.0          # Stop in basis points
    entry_start: int = 935          # Earliest entry (HHMM ET)
    entry_end: int = 1400           # Latest entry (HHMM ET)
    max_trades_per_day: int = 1
    flatten_time: int = 1555


def run_cascade(df: pd.DataFrame, cfg: CascadeConfig) -> List[Trade]:
    """Run Cascade ES backtest bar-by-bar.

    Logic:
    1. Aggregate 5m bars into 30m bars
    2. Detect bearish engulfing on completed 30m bars
    3. Enter short at 5m bar close when signal fires
    4. Exit: first green 5m bar OR flatten at 15:55
    5. Stop: percentage-based (cfg.stop_bps)
    """
    agg = Aggregator30m()
    trades: List[Trade] = []

    # State
    engulfing_signal = False   # True when a qualifying engulfing just completed
    in_position = False
    entry_price = 0.0
    entry_time = None
    stop_price = 0.0
    session_date_current = None
    trades_today = 0

    prev_bar = None
    bars = df.to_dict("records")
    times = df.index.tolist()

    for i, bar in enumerate(bars):
        bar["_time"] = times[i]
        et_time = bar.get("et_time", 0)
        is_rth = bar.get("is_rth", False)
        session_date = bar.get("session_date", None)

        # New session reset
        if session_date != session_date_current:
            # Flatten if still in position at session boundary
            if in_position and prev_bar is not None:
                pnl_pts = entry_price - prev_bar["close"]
                pnl_dollar = pnl_pts * POINT_VALUE - COMMISSION_RT
                trades.append(Trade(
                    setup="CASCADE",
                    direction=-1,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=prev_bar["_time"],
                    exit_price=prev_bar["close"],
                    exit_reason="session_end",
                    pnl_pts=pnl_pts,
                    pnl_dollar=pnl_dollar,
                    stop=stop_price,
                    target=0.0,
                ))
                in_position = False

            session_date_current = session_date
            trades_today = 0
            engulfing_signal = False
            agg = Aggregator30m()
            agg.current_session = session_date

        if not is_rth:
            prev_bar = bar
            continue

        # Feed 5m bar to 30m aggregator
        completed_30m = agg.feed(bar)

        # Check for bearish engulfing on completed 30m bar
        if completed_30m is not None:
            engulfing_signal = False  # Reset — only fire on fresh signals
            bars_30m = agg.completed
            if len(bars_30m) >= 2:
                curr = bars_30m[-1]
                prev30 = bars_30m[-2]

                # Same session
                if curr.session_date == prev30.session_date:
                    # Previous bar green
                    prev_green = prev30.close > prev30.open
                    # Current bar red
                    curr_red = curr.close < curr.open
                    # Body engulfing
                    body_engulf = (curr.open > prev30.close) and (curr.close < prev30.open)
                    # Range filter
                    bar_range = curr.high - curr.low
                    range_ok = bar_range >= cfg.range_threshold

                    if prev_green and curr_red and body_engulf and range_ok:
                        engulfing_signal = True

        # Position management
        if in_position:
            # Check stop
            if bar["high"] >= stop_price:
                pnl_pts = entry_price - stop_price
                pnl_dollar = pnl_pts * POINT_VALUE - COMMISSION_RT
                trades.append(Trade(
                    setup="CASCADE",
                    direction=-1,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=bar["_time"],
                    exit_price=stop_price,
                    exit_reason="stop",
                    pnl_pts=pnl_pts,
                    pnl_dollar=pnl_dollar,
                    stop=stop_price,
                    target=0.0,
                ))
                in_position = False
                prev_bar = bar
                continue

            # Check flatten
            if et_time >= cfg.flatten_time:
                pnl_pts = entry_price - bar["close"]
                pnl_dollar = pnl_pts * POINT_VALUE - COMMISSION_RT
                trades.append(Trade(
                    setup="CASCADE",
                    direction=-1,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=bar["_time"],
                    exit_price=bar["close"],
                    exit_reason="flatten",
                    pnl_pts=pnl_pts,
                    pnl_dollar=pnl_dollar,
                    stop=stop_price,
                    target=0.0,
                ))
                in_position = False
                prev_bar = bar
                continue

            # Check first green 5m bar (close > open) — the SPX Cascade exit
            if bar["close"] > bar["open"]:
                pnl_pts = entry_price - bar["close"]
                pnl_dollar = pnl_pts * POINT_VALUE - COMMISSION_RT
                trades.append(Trade(
                    setup="CASCADE",
                    direction=-1,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=bar["_time"],
                    exit_price=bar["close"],
                    exit_reason="green_bar",
                    pnl_pts=pnl_pts,
                    pnl_dollar=pnl_dollar,
                    stop=stop_price,
                    target=0.0,
                ))
                in_position = False
                prev_bar = bar
                continue

        else:
            # Entry logic — only when flat
            if engulfing_signal and trades_today < cfg.max_trades_per_day:
                if cfg.entry_start <= et_time < cfg.entry_end:
                    entry_price = bar["close"]
                    entry_time = bar["_time"]
                    stop_price = entry_price * (1 + cfg.stop_bps / 10000.0)
                    in_position = True
                    trades_today += 1
                    engulfing_signal = False  # Consumed

        prev_bar = bar

    # Close any open position at end of data
    if in_position and prev_bar is not None:
        pnl_pts = entry_price - prev_bar["close"]
        pnl_dollar = pnl_pts * POINT_VALUE - COMMISSION_RT
        trades.append(Trade(
            setup="CASCADE",
            direction=-1,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=prev_bar["_time"],
            exit_price=prev_bar["close"],
            exit_reason="data_end",
            pnl_pts=pnl_pts,
            pnl_dollar=pnl_dollar,
            stop=stop_price,
            target=0.0,
        ))

    return trades


# ============================================================================
# Analysis Helpers
# ============================================================================

def run_significance(pnls: np.ndarray, seed=SEED):
    """T-test, permutation test, bootstrap profit probability."""
    n = len(pnls)
    if n < 5:
        return 1.0, 1.0, 0.0

    # T-test (one-sided: mean > 0)
    t_stat, t_pval_two = scipy_stats.ttest_1samp(pnls, 0)
    t_pval = t_pval_two / 2 if t_stat > 0 else 1 - t_pval_two / 2

    # Permutation test
    obs_pnl = np.sum(pnls)
    abs_pnls = np.abs(pnls)
    rng = np.random.default_rng(seed)
    n_perm = 10000
    count_better = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if np.dot(signs, abs_pnls) >= obs_pnl:
            count_better += 1
    perm_pval = count_better / n_perm

    # Bootstrap
    n_boot = 10000
    boot_pnl = np.array([
        np.sum(rng.choice(pnls, size=n, replace=True))
        for _ in range(n_boot)
    ])
    prob_profit = float(np.mean(boot_pnl > 0))

    return t_pval, perm_pval, prob_profit


def walk_forward_metrics(trades: List[Trade], split_date: str):
    """Split trades into IS/OOS and compute metrics for each."""
    split_dt = pd.Timestamp(split_date, tz="America/New_York")
    is_trades = [t for t in trades if t.entry_time < split_dt]
    oos_trades = [t for t in trades if t.entry_time >= split_dt]
    m_is = compute_metrics(is_trades) if is_trades else None
    m_oos = compute_metrics(oos_trades) if oos_trades else None
    return m_is, m_oos, is_trades, oos_trades


# ============================================================================
# Parameter Sweep
# ============================================================================

def run_sweep(df: pd.DataFrame):
    """Sweep range threshold, stop bps, entry window. Print results."""

    range_thresholds = [10, 12, 15, 18, 20, 22, 25]
    stop_bps_values = [20, 25, 30, 35, 40]
    entry_windows = [
        (935, 1400, "935-1400"),
        (935, 1300, "935-1300"),
        (935, 1500, "935-1500"),
    ]

    results = []

    total_combos = len(range_thresholds) * len(stop_bps_values) * len(entry_windows)
    print(f"\nSweeping {total_combos} parameter combinations...")
    print(f"{'Range':>6} {'Stop':>6} {'Window':>10} | {'Tr':>4} {'WR%':>6} {'PF':>7} {'NetP&L':>10} {'Sharpe':>7} {'MaxDD':>10} | {'IS_PF':>6} {'OOS_PF':>7} {'PFR':>6} | {'t-pval':>8}")
    print("-" * 120)

    for rng in range_thresholds:
        for stop in stop_bps_values:
            for entry_start, entry_end, win_label in entry_windows:
                cfg = CascadeConfig(
                    range_threshold=rng,
                    stop_bps=stop,
                    entry_start=entry_start,
                    entry_end=entry_end,
                )

                trades = run_cascade(df.copy(), cfg)
                m = compute_metrics(trades) if trades else None

                if not trades or m.total_trades < 5:
                    print(f"{rng:>6} {stop:>5}b {win_label:>10} | {0:>4} {'---':>6} {'---':>7} {'---':>10} {'---':>7} {'---':>10} | {'---':>6} {'---':>7} {'---':>6} | {'---':>8}")
                    continue

                pnls = np.array([t.pnl_dollar for t in trades])
                t_pval, perm_pval, boot_prob = run_significance(pnls)

                # Walk-forward
                m_is, m_oos, _, _ = walk_forward_metrics(trades, WF_SPLIT)
                is_pf = m_is.profit_factor if m_is and m_is.total_trades >= 3 else float("nan")
                oos_pf = m_oos.profit_factor if m_oos and m_oos.total_trades >= 3 else float("nan")
                pf_ratio = oos_pf / is_pf if is_pf > 0 and not math.isnan(is_pf) and not math.isnan(oos_pf) else float("nan")

                print(f"{rng:>6} {stop:>5}b {win_label:>10} | "
                      f"{m.total_trades:>4} {m.win_rate:>5.1f}% {m.profit_factor:>7.2f} "
                      f"${m.net_pnl:>9,.0f} {m.sharpe:>7.2f} ${m.max_drawdown:>9,.0f} | "
                      f"{is_pf:>6.2f} {oos_pf:>7.2f} {pf_ratio:>6.2f} | "
                      f"{t_pval:>8.4f}")

                results.append({
                    "range": rng,
                    "stop_bps": stop,
                    "window": win_label,
                    "trades": m.total_trades,
                    "win_rate": m.win_rate,
                    "pf": m.profit_factor,
                    "net_pnl": m.net_pnl,
                    "sharpe": m.sharpe,
                    "max_dd": m.max_drawdown,
                    "is_pf": is_pf,
                    "oos_pf": oos_pf,
                    "pf_ratio": pf_ratio,
                    "t_pval": t_pval,
                    "perm_pval": perm_pval,
                    "boot_prob": boot_prob,
                })

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "es_5m_databento_2yr.csv"
    )

    print(f"Loading {data_path}...")
    df = load_tos_csv(data_path, instrument="ES")
    print(f"Loaded {len(df):,} bars: {df.index[0].date()} to {df.index[-1].date()}")
    rth_bars = df["is_rth"].sum()
    sessions = df["new_rth"].sum()
    print(f"RTH bars: {rth_bars:,} | Sessions: {sessions}")

    # ── Phase 1: Parameter Sweep ──
    print("\n" + "=" * 120)
    print("  PHASE 1: PARAMETER SWEEP")
    print("=" * 120)

    results = run_sweep(df)

    if not results:
        print("\nNo trades generated across any parameter combination. FAIL.")
        return

    # ── Phase 2: Find Best Config ──
    # Filter: trades >= 10, PF > 1.0, t_pval < 0.10 (relaxed for selection)
    viable = [r for r in results
              if r["trades"] >= 10
              and r["pf"] > 1.0
              and r["t_pval"] < 0.10]

    if not viable:
        # Relax criteria
        viable = [r for r in results if r["trades"] >= 5 and r["pf"] > 1.0]

    if not viable:
        print("\n" + "=" * 80)
        print("  NO VIABLE CONFIGURATION FOUND")
        print("=" * 80)
        # Show best PF anyway
        best_pf = max(results, key=lambda r: r["pf"] if r["trades"] >= 5 else 0)
        print(f"  Best PF (>= 5 trades): range={best_pf['range']}, stop={best_pf['stop_bps']}bps, "
              f"window={best_pf['window']}")
        print(f"  PF={best_pf['pf']:.2f}, Trades={best_pf['trades']}, p={best_pf['t_pval']:.4f}")
        print(f"\n  VERDICT: FAIL")
        return

    # Rank by: primary = net P&L, secondary = PF ratio (WF stability)
    viable.sort(key=lambda r: r["net_pnl"], reverse=True)
    best = viable[0]

    print("\n" + "=" * 80)
    print("  PHASE 2: BEST CONFIGURATION")
    print("=" * 80)
    print(f"  Range threshold: {best['range']} ES pts")
    print(f"  Stop:            {best['stop_bps']} bps")
    print(f"  Entry window:    {best['window']}")
    print(f"  Trades:          {best['trades']}")
    print(f"  Win Rate:        {best['win_rate']:.1f}%")
    print(f"  Profit Factor:   {best['pf']:.2f}")
    print(f"  Net P&L:         ${best['net_pnl']:,.0f}")
    print(f"  Sharpe:          {best['sharpe']:.2f}")
    print(f"  Max Drawdown:    ${best['max_dd']:,.0f}")

    # ── Phase 3: Detailed Analysis of Best Config ──
    print("\n" + "=" * 80)
    print("  PHASE 3: DETAILED ANALYSIS — BEST CONFIG")
    print("=" * 80)

    cfg = CascadeConfig(
        range_threshold=best["range"],
        stop_bps=best["stop_bps"],
        entry_start=int(best["window"].split("-")[0]),
        entry_end=int(best["window"].split("-")[1]),
    )

    trades = run_cascade(df.copy(), cfg)
    m = compute_metrics(trades)
    pnls = np.array([t.pnl_dollar for t in trades])

    print(f"\n  --- Full-Period Metrics ---")
    print(f"  Total Trades:   {m.total_trades}")
    print(f"  Net P&L:        ${m.net_pnl:,.2f}")
    print(f"  Win Rate:       {m.win_rate:.1f}%")
    print(f"  Profit Factor:  {m.profit_factor:.3f}")
    print(f"  Sharpe:         {m.sharpe:.2f}")
    print(f"  Max Drawdown:   ${m.max_drawdown:,.2f}")
    print(f"  Avg Trade:      ${m.avg_trade:,.2f}")
    print(f"  Avg Win:        ${m.avg_win:,.2f}")
    print(f"  Avg Loss:       ${m.avg_loss:,.2f}")
    print(f"  Win Streak:     {m.longest_win_streak}")
    print(f"  Lose Streak:    {m.longest_lose_streak}")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons.setdefault(t.exit_reason, {"count": 0, "pnl": 0.0})
        reasons[t.exit_reason]["count"] += 1
        reasons[t.exit_reason]["pnl"] += t.pnl_dollar
    print(f"\n  --- Exit Reasons ---")
    for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        avg = data["pnl"] / data["count"]
        print(f"  {reason:<16} {data['count']:>4} trades  ${data['pnl']:>+10,.0f}  (avg ${avg:>+8,.0f})")

    # Monthly breakdown
    print(f"\n  --- Monthly Breakdown ---")
    monthly = {}
    for t in trades:
        key = t.entry_time.strftime("%Y-%m") if hasattr(t.entry_time, "strftime") else "unknown"
        monthly.setdefault(key, {"count": 0, "pnl": 0.0})
        monthly[key]["count"] += 1
        monthly[key]["pnl"] += t.pnl_dollar
    for month in sorted(monthly.keys()):
        data = monthly[month]
        bar = "+" * int(abs(data["pnl"]) / 200) if data["pnl"] > 0 else "-" * int(abs(data["pnl"]) / 200)
        print(f"  {month}  {data['count']:>3} trades  ${data['pnl']:>+10,.0f}  {bar}")

    # ── Phase 4: Statistical Significance ──
    print(f"\n  --- Statistical Significance ---")
    t_pval, perm_pval, boot_prob = run_significance(pnls)
    print(f"  t-test p-value:        {t_pval:.6f} {'***' if t_pval < 0.01 else '**' if t_pval < 0.05 else '*' if t_pval < 0.10 else ''}")
    print(f"  Permutation p-value:   {perm_pval:.6f}")
    print(f"  Bootstrap P(profit):   {boot_prob:.2%}")

    # ── Phase 5: Walk-Forward Validation ──
    print(f"\n  --- Walk-Forward Validation (split: {WF_SPLIT}) ---")
    m_is, m_oos, is_trades, oos_trades = walk_forward_metrics(trades, WF_SPLIT)

    if m_is and m_is.total_trades >= 3:
        pnls_is = np.array([t.pnl_dollar for t in is_trades])
        t_pval_is, _, _ = run_significance(pnls_is)
        print(f"  IS:  {m_is.total_trades:>4} trades | PF {m_is.profit_factor:.3f} | "
              f"WR {m_is.win_rate:.1f}% | ${m_is.net_pnl:>+10,.0f} | p={t_pval_is:.4f}")
    else:
        print(f"  IS:  Too few trades ({m_is.total_trades if m_is else 0})")

    if m_oos and m_oos.total_trades >= 3:
        pnls_oos = np.array([t.pnl_dollar for t in oos_trades])
        t_pval_oos, _, _ = run_significance(pnls_oos)
        print(f"  OOS: {m_oos.total_trades:>4} trades | PF {m_oos.profit_factor:.3f} | "
              f"WR {m_oos.win_rate:.1f}% | ${m_oos.net_pnl:>+10,.0f} | p={t_pval_oos:.4f}")
    else:
        print(f"  OOS: Too few trades ({m_oos.total_trades if m_oos else 0})")

    if m_is and m_oos and m_is.profit_factor > 0 and m_oos.profit_factor > 0:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor
        print(f"  PF ratio (OOS/IS):     {pf_ratio:.3f} {'PASS (>= 0.70)' if pf_ratio >= 0.70 else 'FAIL (< 0.70)'}")
    elif m_is and m_oos:
        pf_ratio = 0
        print(f"  PF ratio: N/A (IS or OOS PF <= 0)")
    else:
        pf_ratio = 0

    # ── Phase 6: Parameter Sensitivity ──
    print(f"\n  --- Parameter Sensitivity (range threshold) ---")
    for rng in [15, 18, 20, 22, 25]:
        cfg_s = CascadeConfig(
            range_threshold=rng,
            stop_bps=best["stop_bps"],
            entry_start=int(best["window"].split("-")[0]),
            entry_end=int(best["window"].split("-")[1]),
        )
        t_s = run_cascade(df.copy(), cfg_s)
        m_s = compute_metrics(t_s) if t_s else None
        if m_s and m_s.total_trades >= 3:
            p_s = np.array([t.pnl_dollar for t in t_s])
            tp, _, _ = run_significance(p_s)
            print(f"  range={rng:>2}: {m_s.total_trades:>4} trades | PF {m_s.profit_factor:.2f} | "
                  f"${m_s.net_pnl:>+9,.0f} | p={tp:.4f}")
        else:
            print(f"  range={rng:>2}: {m_s.total_trades if m_s else 0} trades (insufficient)")

    print(f"\n  --- Parameter Sensitivity (stop bps) ---")
    for stop in [25, 28, 30, 32, 35]:
        cfg_s = CascadeConfig(
            range_threshold=best["range"],
            stop_bps=stop,
            entry_start=int(best["window"].split("-")[0]),
            entry_end=int(best["window"].split("-")[1]),
        )
        t_s = run_cascade(df.copy(), cfg_s)
        m_s = compute_metrics(t_s) if t_s else None
        if m_s and m_s.total_trades >= 3:
            p_s = np.array([t.pnl_dollar for t in t_s])
            tp, _, _ = run_significance(p_s)
            print(f"  stop={stop:>2}bps: {m_s.total_trades:>4} trades | PF {m_s.profit_factor:.2f} | "
                  f"${m_s.net_pnl:>+9,.0f} | p={tp:.4f}")
        else:
            print(f"  stop={stop:>2}bps: {m_s.total_trades if m_s else 0} trades (insufficient)")

    # ── FINAL VERDICT ──
    print("\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)

    sig_pass = t_pval < 0.05
    wf_pass = (m_is is not None and m_oos is not None
               and m_is.total_trades >= 5 and m_oos.total_trades >= 5
               and m_oos.profit_factor > 1.0
               and (m_oos.profit_factor / m_is.profit_factor >= 0.70 if m_is.profit_factor > 0 else False))
    pf_pass = m.profit_factor > 1.0
    trade_pass = m.total_trades >= 10

    checks = [
        ("Trades >= 10", trade_pass),
        ("PF > 1.0", pf_pass),
        ("p < 0.05 (t-test)", sig_pass),
        ("WF PF ratio >= 0.70", wf_pass),
    ]

    all_pass = all(v for _, v in checks)

    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n  OVERALL: {verdict}")
    print("=" * 80)


if __name__ == "__main__":
    main()
