#!/usr/bin/env python3
"""Vector ES — EMA Cross Directional ported from SPX to ES futures.

Source: SPX backtester strategies/ema_cross_directional.py (Vector)
Signal: EMA 8/24 bearish cross on 5m ES bars → SHORT ONLY
Filters: 30m range threshold, TEMA bearish, ON inventory (skip net-short days)
Stop: 30bps percentage-based
Exit: opposite cross, time stop 15:55, max hold bars, target (next support)

Sweep: range threshold, EMA periods, stop bps, with/without ON filter, max hold
Walk-forward + t-test validation.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics
from backtester.position import Trade


# ── Configuration ──

@dataclass
class VectorConfig:
    """Vector ES parameters."""
    # EMA
    ema_fast: int = 8
    ema_slow: int = 24

    # 30m range filter (ES points)
    min_30m_range: float = 15.0

    # TEMA bearish filter (use precomputed tema_fast < tema_slow from indicators)
    require_tema_bearish: bool = True

    # ON inventory filter: skip net-short ON days
    skip_net_short_on: bool = False

    # Stop: percentage-based (basis points)
    stop_bps: float = 30.0

    # Max hold bars (5m bars, 0=unlimited)
    max_hold_bars: int = 0

    # Entry window (ET time as HHMM)
    entry_start: int = 935
    entry_end: int = 1500
    time_stop: int = 1555

    # Max trades per day
    max_trades_per_day: int = 2

    # Target: fixed pts below entry (0=no target, use opposite cross / time stop)
    target_pts: float = 0.0

    # Instrument
    point_value: float = 50.0
    commission_rt: float = 5.0   # $2.50 per side
    slippage_ticks: int = 1
    tick_size: float = 0.25


# ── Data Loading ──

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "es_5m_databento_2yr.csv",
)
WF_SPLIT = "2025-02-16"
INITIAL_CAPITAL = 100_000.0


def load_data():
    """Load ES 5m data with session tags and indicators."""
    print(f"Loading {DATA_PATH}...")
    df = load_tos_csv(DATA_PATH, instrument="ES")
    print(f"  {len(df):,} bars | {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    # Compute indicators (TEMA for filter confirmation)
    from backtester.indicators import compute_indicators
    compute_indicators(df)

    return df


# ── 30-Minute Bar Aggregation ──

def build_30m_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5m bars into 30m bars for range filter.

    Groups by session_date + 30-minute period to get proper 30m OHLCV.
    Returns DataFrame indexed by period end time with columns:
        open, high, low, close, volume, range, session_date, et_time
    """
    rth = df[df["is_rth"]].copy()
    if rth.empty:
        return pd.DataFrame()

    # 30-minute period index: 930-1000=0, 1000-1030=1, etc.
    et = rth["et_time"].values
    minutes_from_open = ((et // 100 - 9) * 60 + (et % 100)) - 30
    rth["period_30m"] = minutes_from_open // 30

    bars_30m = rth.groupby(["session_date", "period_30m"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        et_time=("et_time", "last"),
    ).reset_index()
    bars_30m["range"] = bars_30m["high"] - bars_30m["low"]

    return bars_30m


# ── Overnight Inventory ──

def compute_on_inventory(df: pd.DataFrame) -> dict:
    """Compute overnight inventory per session date.

    Net short ON: on_low < prev_day_close (overnight traded below prior close).
    Returns dict: {session_date: "net_short" | "net_long" | "mixed"}
    """
    inventory = {}

    # Get session dates and their first RTH bars
    rth = df[df["new_rth"]].copy()
    if rth.empty:
        return inventory

    dates = sorted(rth["session_date"].unique())

    # For each session, we need on_high, on_low, prev_day_close
    # These come from the session state built by data_loader
    # We'll compute manually from the raw data

    for i, date in enumerate(dates):
        if i == 0:
            continue

        prev_date = dates[i - 1]

        # Previous day close: last RTH bar of prev_date
        prev_rth = df[(df["session_date"] == prev_date) & df["is_rth"]]
        if prev_rth.empty:
            continue
        prev_close = prev_rth.iloc[-1]["close"]

        # Overnight bars: between prev RTH end and current RTH start
        # Globex: bars where session_date == current_date and is_rth == False (before 9:30)
        # OR bars after prev_date RTH (18:00+)
        globex = df[(df["session_date"] == date) & ~df["is_rth"]]
        if globex.empty:
            # No globex data — can't classify
            inventory[date] = "mixed"
            continue

        on_high = globex["high"].max()
        on_low = globex["low"].min()

        # Classification based on Vector SPX logic:
        # net_short: on_low < prev_close (overnight sold below close)
        on_above = on_high > prev_close
        on_below = on_low < prev_close

        if on_above and not on_below:
            inventory[date] = "net_long"
        elif on_below and not on_above:
            inventory[date] = "net_short"
        else:
            inventory[date] = "mixed"

    return inventory


# ── EMA Cross Detection ──

def detect_ema_crosses(df: pd.DataFrame, fast_period: int, slow_period: int):
    """Compute EMAs and detect bearish crosses on 5m bars.

    Returns (ema_fast_arr, ema_slow_arr, cross_bear_arr) as numpy arrays.
    Cross only fires within same session_date (no overnight crossovers).
    """
    close = df["close"]
    ema_fast = close.ewm(span=fast_period, adjust=False).mean().values
    ema_slow = close.ewm(span=slow_period, adjust=False).mean().values

    n = len(df)
    cross_bear = np.zeros(n, dtype=bool)
    cross_bull = np.zeros(n, dtype=bool)
    sess = df["session_date"].values

    for i in range(1, n):
        if sess[i] != sess[i - 1]:
            continue
        if ema_fast[i - 1] >= ema_slow[i - 1] and ema_fast[i] < ema_slow[i]:
            cross_bear[i] = True
        if ema_fast[i - 1] <= ema_slow[i - 1] and ema_fast[i] > ema_slow[i]:
            cross_bull[i] = True

    return ema_fast, ema_slow, cross_bear, cross_bull


# ── 30m Range Lookup ──

def build_range_lookup(bars_30m: pd.DataFrame) -> dict:
    """Build lookup: (session_date) -> max 30m range seen so far up to current bar.

    For the range filter, we check: has ANY completed 30m bar today had range >= threshold?
    Returns dict: session_date -> list of (et_time, cumulative_max_range).
    """
    lookup = {}  # session_date -> max_range_so_far at each 30m completion

    for _, row in bars_30m.iterrows():
        sd = row["session_date"]
        rng = row["range"]
        et = row["et_time"]

        if sd not in lookup:
            lookup[sd] = []

        prev_max = lookup[sd][-1][1] if lookup[sd] else 0.0
        lookup[sd].append((et, max(prev_max, rng)))

    return lookup


def get_max_30m_range(range_lookup: dict, session_date, et_time: int) -> float:
    """Get the maximum 30m range for this session up to the given ET time.

    Uses the most recently completed 30m bar (not the one still forming).
    """
    entries = range_lookup.get(session_date)
    if not entries:
        return 0.0

    max_range = 0.0
    for bar_et, cum_max in entries:
        if bar_et <= et_time:
            max_range = cum_max
        else:
            break

    return max_range


# ── Backtest Engine ──

def run_vector_backtest(df: pd.DataFrame, cfg: VectorConfig) -> List[Trade]:
    """Run Vector ES backtest.

    EMA cross bearish → short entry on ES futures.
    """
    # Build 30m bars and range lookup
    bars_30m = build_30m_bars(df)
    range_lookup = build_range_lookup(bars_30m)

    # ON inventory
    on_inventory = compute_on_inventory(df) if cfg.skip_net_short_on else {}

    # Detect EMA crosses
    ema_fast, ema_slow, cross_bear, cross_bull = detect_ema_crosses(
        df, cfg.ema_fast, cfg.ema_slow
    )

    # Work on RTH + ETH data but only enter during RTH window
    et = df["et_time"].values
    sess = df["session_date"].values
    cl = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    is_rth = df["is_rth"].values
    timestamps = df.index
    tema_bearish = df["tema_bearish"].values if "tema_bearish" in df.columns else np.ones(len(df), dtype=bool)
    n = len(df)

    trades = []
    position = None  # dict: entry_idx, entry_price, stop, target, session, trades_today
    trades_today = 0
    current_session = None
    slippage = cfg.slippage_ticks * cfg.tick_size

    warmup = max(cfg.ema_slow, 30) + 5

    for i in range(warmup, n):
        s = sess[i]

        # Session reset
        if s != current_session:
            # Close any open position at session boundary
            if position is not None:
                exit_price = cl[i - 1] + slippage  # close at previous bar's close (short → buy to close)
                pnl_pts = position["entry_price"] - exit_price
                pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
                trades.append(Trade(
                    setup="VEC_ES",
                    direction=-1,
                    entry_time=timestamps[position["entry_idx"]],
                    entry_price=position["entry_price"],
                    exit_time=timestamps[i - 1],
                    exit_price=exit_price,
                    exit_reason="session_end",
                    pnl_pts=pnl_pts,
                    pnl_dollar=pnl_dollar,
                    stop=position["stop"],
                    target=position["target"],
                ))
                position = None

            current_session = s
            trades_today = 0

        # ── EXIT CHECKS (before entry) ──
        if position is not None:
            exit_reason = None
            exit_price = None

            # Stop loss (high hits stop)
            if hi[i] >= position["stop"]:
                exit_reason = "stop"
                exit_price = position["stop"] + slippage

            # Target hit (if we have a target)
            if exit_reason is None and position["target"] > 0 and lo[i] <= position["target"]:
                exit_reason = "target"
                exit_price = position["target"] - slippage

            # Max hold bars
            if exit_reason is None and cfg.max_hold_bars > 0:
                bars_held = i - position["entry_idx"]
                if bars_held >= cfg.max_hold_bars:
                    exit_reason = "max_hold"
                    exit_price = cl[i]

            # Time stop
            if exit_reason is None and et[i] >= cfg.time_stop and is_rth[i]:
                exit_reason = "time_stop"
                exit_price = cl[i]

            # Opposite cross (bullish EMA cross)
            if exit_reason is None and cross_bull[i]:
                exit_reason = "opposite_cross"
                exit_price = cl[i]

            if exit_reason is not None:
                pnl_pts = position["entry_price"] - exit_price
                pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
                trades.append(Trade(
                    setup="VEC_ES",
                    direction=-1,
                    entry_time=timestamps[position["entry_idx"]],
                    entry_price=position["entry_price"],
                    exit_time=timestamps[i],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pts=pnl_pts,
                    pnl_dollar=pnl_dollar,
                    stop=position["stop"],
                    target=position["target"],
                ))
                position = None

        # ── ENTRY CHECKS ──
        if position is None and is_rth[i]:
            # Time window
            if et[i] < cfg.entry_start or et[i] >= cfg.entry_end:
                continue

            # Max trades per day
            if trades_today >= cfg.max_trades_per_day:
                continue

            # Bearish EMA cross
            if not cross_bear[i]:
                continue

            # TEMA bearish filter (from indicators.py: tema_fast < tema_slow)
            if cfg.require_tema_bearish and not tema_bearish[i]:
                continue

            # 30m range filter: require at least one 30m bar today with range >= threshold
            max_range = get_max_30m_range(range_lookup, s, et[i])
            if max_range < cfg.min_30m_range:
                continue

            # ON inventory filter
            if cfg.skip_net_short_on and on_inventory.get(s) == "net_short":
                continue

            # ── ENTER SHORT ──
            entry_price = cl[i] - slippage  # short entry → sell at slightly worse price
            stop_pts = entry_price * cfg.stop_bps / 10000.0
            stop_price = entry_price + stop_pts

            if cfg.target_pts > 0:
                target_price = entry_price - cfg.target_pts
            else:
                target_price = 0.0  # no fixed target → exit on cross/time

            position = {
                "entry_idx": i,
                "entry_price": entry_price,
                "stop": stop_price,
                "target": target_price,
                "session": s,
            }
            trades_today += 1

    # Close any remaining position
    if position is not None:
        exit_price = cl[-1] + slippage
        pnl_pts = position["entry_price"] - exit_price
        pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
        trades.append(Trade(
            setup="VEC_ES",
            direction=-1,
            entry_time=timestamps[position["entry_idx"]],
            entry_price=position["entry_price"],
            exit_time=timestamps[-1],
            exit_price=exit_price,
            exit_reason="data_end",
            pnl_pts=pnl_pts,
            pnl_dollar=pnl_dollar,
            stop=position["stop"],
            target=position["target"],
        ))

    return trades


# ── Reporting ──

def print_metrics(trades: List[Trade], label: str = "", capital: float = INITIAL_CAPITAL):
    """Print metrics summary."""
    m = compute_metrics(trades, capital)
    tag = f"  [{label}]" if label else ""
    print(f"{tag}  {m.total_trades} trades | WR {m.win_rate:.1f}% | "
          f"PF {m.profit_factor:.3f} | ${m.net_pnl:,.0f} | "
          f"Sharpe {m.sharpe:.2f} | DD ${m.max_drawdown:,.0f}")
    return m


def run_significance(trades: List[Trade], seed: int = 42) -> Tuple[float, float, float]:
    """T-test + permutation + bootstrap."""
    pnls = np.array([t.pnl_dollar for t in trades])
    n = len(pnls)

    if n < 5:
        return 1.0, 1.0, 0.0

    # T-test (one-sample, H0: mean=0)
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    t_pval_one = t_pval / 2 if t_stat > 0 else 1 - t_pval / 2  # one-tailed

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

    # Bootstrap P(profit)
    n_boot = 10000
    boot_pnl = np.array([
        np.sum(rng.choice(pnls, size=n, replace=True))
        for _ in range(n_boot)
    ])
    prob_profit = float(np.mean(boot_pnl > 0))

    return t_pval_one, perm_pval, prob_profit


def walk_forward(df: pd.DataFrame, cfg: VectorConfig,
                 split_date: str = WF_SPLIT) -> Tuple[Optional[object], Optional[object]]:
    """Walk-forward: IS before split_date, OOS after."""
    df_is = df[df.index < split_date].copy()
    df_oos = df[df.index >= split_date].copy()

    trades_is = run_vector_backtest(df_is, cfg)
    trades_oos = run_vector_backtest(df_oos, cfg)

    m_is = compute_metrics(trades_is, INITIAL_CAPITAL) if trades_is else None
    m_oos = compute_metrics(trades_oos, INITIAL_CAPITAL) if trades_oos else None

    return m_is, m_oos, trades_is, trades_oos


# ── Parameter Sweep ──

def run_sweep(df: pd.DataFrame):
    """Sweep key parameters and report results."""
    print("\n" + "=" * 90)
    print("  VECTOR ES — PARAMETER SWEEP")
    print("=" * 90)

    # Sweep dimensions
    ema_pairs = [(8, 24), (9, 21), (8, 21), (12, 24)]
    range_thresholds = [0, 8, 10, 12, 15, 18, 20, 25]
    stop_bps_list = [20, 25, 30, 35, 40]
    on_filter_options = [False, True]
    max_hold_list = [0, 6, 12, 24]  # 0=no limit, 6=30min, 12=60min, 24=120min
    target_pts_list = [0, 5, 8, 10]

    results = []

    # Phase 1: Range threshold sweep (fix other params at baseline)
    print("\n  --- Phase 1: Range Threshold Sweep (EMA 8/24, 30bps stop, no ON filter) ---")
    print(f"  {'Range':>6} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>12} {'Sharpe':>8} {'AvgTrd':>8}")
    print(f"  {'-'*58}")

    for rng in range_thresholds:
        cfg = VectorConfig(min_30m_range=rng)
        trades = run_vector_backtest(df, cfg)
        m = compute_metrics(trades, INITIAL_CAPITAL)
        print(f"  {rng:>6.0f} {m.total_trades:>7} {m.win_rate:>6.1f}% "
              f"{m.profit_factor:>8.3f} ${m.net_pnl:>11,.0f} {m.sharpe:>8.2f} ${m.avg_trade:>7,.0f}")
        results.append({"range": rng, "ema": "8/24", "stop_bps": 30, "on_filter": False,
                        "max_hold": 0, "target_pts": 0, "trades": m.total_trades,
                        "wr": m.win_rate, "pf": m.profit_factor, "pnl": m.net_pnl,
                        "sharpe": m.sharpe, "dd": m.max_drawdown})

    # Phase 2: EMA period sweep (fix range at best from phase 1)
    # Find best range (highest PF with >= 30 trades)
    viable = [r for r in results if r["trades"] >= 30]
    best_range = max(viable, key=lambda x: x["pf"])["range"] if viable else 15

    print(f"\n  --- Phase 2: EMA Period Sweep (range >= {best_range}, 30bps stop) ---")
    print(f"  {'EMA':>8} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*52}")

    ema_results = []
    for fast, slow in ema_pairs:
        cfg = VectorConfig(ema_fast=fast, ema_slow=slow, min_30m_range=best_range)
        trades = run_vector_backtest(df, cfg)
        m = compute_metrics(trades, INITIAL_CAPITAL)
        label = f"{fast}/{slow}"
        print(f"  {label:>8} {m.total_trades:>7} {m.win_rate:>6.1f}% "
              f"{m.profit_factor:>8.3f} ${m.net_pnl:>11,.0f} {m.sharpe:>8.2f}")
        ema_results.append({"ema": label, "fast": fast, "slow": slow, "pf": m.profit_factor,
                            "trades": m.total_trades, "pnl": m.net_pnl})

    # Best EMA
    viable_ema = [r for r in ema_results if r["trades"] >= 20]
    best_ema = max(viable_ema, key=lambda x: x["pf"]) if viable_ema else {"fast": 8, "slow": 24}

    # Phase 3: Stop BPS sweep
    print(f"\n  --- Phase 3: Stop BPS Sweep (EMA {best_ema['fast']}/{best_ema['slow']}, range >= {best_range}) ---")
    print(f"  {'StopBPS':>8} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*52}")

    stop_results = []
    for sbps in stop_bps_list:
        cfg = VectorConfig(
            ema_fast=best_ema["fast"], ema_slow=best_ema["slow"],
            min_30m_range=best_range, stop_bps=sbps,
        )
        trades = run_vector_backtest(df, cfg)
        m = compute_metrics(trades, INITIAL_CAPITAL)
        print(f"  {sbps:>8} {m.total_trades:>7} {m.win_rate:>6.1f}% "
              f"{m.profit_factor:>8.3f} ${m.net_pnl:>11,.0f} {m.sharpe:>8.2f}")
        stop_results.append({"stop_bps": sbps, "pf": m.profit_factor,
                             "trades": m.total_trades, "pnl": m.net_pnl})

    viable_stop = [r for r in stop_results if r["trades"] >= 20]
    best_stop = max(viable_stop, key=lambda x: x["pf"])["stop_bps"] if viable_stop else 30

    # Phase 4: Max hold sweep
    print(f"\n  --- Phase 4: Max Hold Sweep (EMA {best_ema['fast']}/{best_ema['slow']}, "
          f"range >= {best_range}, stop {best_stop}bps) ---")
    print(f"  {'MaxHold':>8} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*52}")

    hold_results = []
    for mh in max_hold_list:
        cfg = VectorConfig(
            ema_fast=best_ema["fast"], ema_slow=best_ema["slow"],
            min_30m_range=best_range, stop_bps=best_stop, max_hold_bars=mh,
        )
        trades = run_vector_backtest(df, cfg)
        m = compute_metrics(trades, INITIAL_CAPITAL)
        label = f"{mh*5}min" if mh > 0 else "none"
        print(f"  {label:>8} {m.total_trades:>7} {m.win_rate:>6.1f}% "
              f"{m.profit_factor:>8.3f} ${m.net_pnl:>11,.0f} {m.sharpe:>8.2f}")
        hold_results.append({"max_hold": mh, "pf": m.profit_factor,
                             "trades": m.total_trades, "pnl": m.net_pnl})

    viable_hold = [r for r in hold_results if r["trades"] >= 20]
    best_hold = max(viable_hold, key=lambda x: x["pf"])["max_hold"] if viable_hold else 0

    # Phase 5: Target pts sweep
    print(f"\n  --- Phase 5: Target Sweep (EMA {best_ema['fast']}/{best_ema['slow']}, "
          f"range >= {best_range}, stop {best_stop}bps, hold {best_hold*5 if best_hold else 'none'}min) ---")
    print(f"  {'Target':>8} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*52}")

    tgt_results = []
    for tgt in target_pts_list:
        cfg = VectorConfig(
            ema_fast=best_ema["fast"], ema_slow=best_ema["slow"],
            min_30m_range=best_range, stop_bps=best_stop,
            max_hold_bars=best_hold, target_pts=tgt,
        )
        trades = run_vector_backtest(df, cfg)
        m = compute_metrics(trades, INITIAL_CAPITAL)
        label = f"{tgt}pts" if tgt > 0 else "none"
        print(f"  {label:>8} {m.total_trades:>7} {m.win_rate:>6.1f}% "
              f"{m.profit_factor:>8.3f} ${m.net_pnl:>11,.0f} {m.sharpe:>8.2f}")
        tgt_results.append({"target_pts": tgt, "pf": m.profit_factor,
                            "trades": m.total_trades, "pnl": m.net_pnl})

    viable_tgt = [r for r in tgt_results if r["trades"] >= 20]
    best_tgt = max(viable_tgt, key=lambda x: x["pf"])["target_pts"] if viable_tgt else 0

    # Phase 6: ON inventory filter (final toggle)
    print(f"\n  --- Phase 6: ON Inventory Filter ---")
    print(f"  {'ON Filter':>10} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*54}")

    on_results = []
    for on_filt in [False, True]:
        cfg = VectorConfig(
            ema_fast=best_ema["fast"], ema_slow=best_ema["slow"],
            min_30m_range=best_range, stop_bps=best_stop,
            max_hold_bars=best_hold, target_pts=best_tgt,
            skip_net_short_on=on_filt,
        )
        trades = run_vector_backtest(df, cfg)
        m = compute_metrics(trades, INITIAL_CAPITAL)
        label = "ON=skip" if on_filt else "ON=off"
        print(f"  {label:>10} {m.total_trades:>7} {m.win_rate:>6.1f}% "
              f"{m.profit_factor:>8.3f} ${m.net_pnl:>11,.0f} {m.sharpe:>8.2f}")
        on_results.append({"on_filter": on_filt, "pf": m.profit_factor,
                           "trades": m.total_trades, "pnl": m.net_pnl})

    viable_on = [r for r in on_results if r["trades"] >= 20]
    best_on = max(viable_on, key=lambda x: x["pf"])["on_filter"] if viable_on else False

    # Phase 7: TEMA filter ablation
    print(f"\n  --- Phase 7: TEMA Bearish Filter Ablation ---")
    print(f"  {'TEMA':>10} {'Trades':>7} {'WR':>7} {'PF':>8} {'P&L':>12} {'Sharpe':>8}")
    print(f"  {'-'*54}")

    for tema_on in [True, False]:
        cfg = VectorConfig(
            ema_fast=best_ema["fast"], ema_slow=best_ema["slow"],
            min_30m_range=best_range, stop_bps=best_stop,
            max_hold_bars=best_hold, target_pts=best_tgt,
            skip_net_short_on=best_on, require_tema_bearish=tema_on,
        )
        trades = run_vector_backtest(df, cfg)
        m = compute_metrics(trades, INITIAL_CAPITAL)
        label = "TEMA=on" if tema_on else "TEMA=off"
        print(f"  {label:>10} {m.total_trades:>7} {m.win_rate:>6.1f}% "
              f"{m.profit_factor:>8.3f} ${m.net_pnl:>11,.0f} {m.sharpe:>8.2f}")

    # ── BEST CONFIG ──
    best_cfg = VectorConfig(
        ema_fast=best_ema["fast"], ema_slow=best_ema["slow"],
        min_30m_range=best_range, stop_bps=best_stop,
        max_hold_bars=best_hold, target_pts=best_tgt,
        skip_net_short_on=best_on,
    )

    print(f"\n{'='*90}")
    print(f"  BEST CONFIG")
    print(f"{'='*90}")
    print(f"  EMA:          {best_ema['fast']}/{best_ema['slow']}")
    print(f"  30m Range:    >= {best_range} ES pts")
    print(f"  Stop:         {best_stop} bps")
    print(f"  Max Hold:     {'none' if best_hold == 0 else f'{best_hold*5} min'}")
    print(f"  Target:       {'none' if best_tgt == 0 else f'{best_tgt} pts'}")
    print(f"  ON Filter:    {'ON' if best_on else 'OFF'}")
    print(f"  TEMA Filter:  ON")

    return best_cfg


def run_full_validation(df: pd.DataFrame, cfg: VectorConfig):
    """Run full validation: full backtest + walk-forward + significance."""
    print(f"\n{'='*90}")
    print(f"  VECTOR ES — FULL VALIDATION")
    print(f"{'='*90}")
    print(f"  Config: EMA {cfg.ema_fast}/{cfg.ema_slow} | Range >= {cfg.min_30m_range} | "
          f"Stop {cfg.stop_bps}bps | Hold {'none' if cfg.max_hold_bars == 0 else f'{cfg.max_hold_bars*5}min'} | "
          f"Target {'none' if cfg.target_pts == 0 else f'{cfg.target_pts}pts'} | "
          f"ON {'skip' if cfg.skip_net_short_on else 'off'}")

    # Full backtest
    print(f"\n  --- Full Period ---")
    trades_all = run_vector_backtest(df, cfg)
    m_all = print_metrics(trades_all, "FULL")

    # Exit reason breakdown
    if trades_all:
        reasons = {}
        for t in trades_all:
            reasons.setdefault(t.exit_reason, {"count": 0, "pnl": 0})
            reasons[t.exit_reason]["count"] += 1
            reasons[t.exit_reason]["pnl"] += t.pnl_dollar

        print(f"\n  --- Exit Reasons ---")
        for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            avg = data["pnl"] / data["count"] if data["count"] else 0
            print(f"  {reason:<18} {data['count']:>4} trades  ${data['pnl']:>+10,.0f}  (avg ${avg:>+7,.0f})")

    # Walk-forward
    print(f"\n  --- Walk-Forward (split: {WF_SPLIT}) ---")
    m_is, m_oos, trades_is, trades_oos = walk_forward(df, cfg)

    if m_is:
        print(f"  IS:   {m_is.total_trades:>4} trades | WR {m_is.win_rate:.1f}% | "
              f"PF {m_is.profit_factor:.3f} | ${m_is.net_pnl:,.0f} | Sharpe {m_is.sharpe:.2f}")
    if m_oos:
        print(f"  OOS:  {m_oos.total_trades:>4} trades | WR {m_oos.win_rate:.1f}% | "
              f"PF {m_oos.profit_factor:.3f} | ${m_oos.net_pnl:,.0f} | Sharpe {m_oos.sharpe:.2f}")
    if m_is and m_oos and m_is.profit_factor > 0:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor
        wf_pass = pf_ratio >= 0.70
        print(f"  PF Ratio (OOS/IS): {pf_ratio:.3f} {'PASS' if wf_pass else 'FAIL'} (threshold: 0.70)")

    # Significance on full period
    print(f"\n  --- Statistical Significance (Full Period) ---")
    t_pval, perm_pval, prob_profit = run_significance(trades_all)
    sig_pass = t_pval < 0.05
    print(f"  t-test p-value:      {t_pval:.6f} {'***' if t_pval < 0.01 else '**' if t_pval < 0.05 else '*' if t_pval < 0.10 else ''}")
    print(f"  Permutation p-value: {perm_pval:.6f}")
    print(f"  Bootstrap P(profit): {prob_profit:.2%}")

    # Significance on OOS only
    if trades_oos and len(trades_oos) >= 10:
        print(f"\n  --- Statistical Significance (OOS Only) ---")
        t_pval_oos, perm_pval_oos, prob_profit_oos = run_significance(trades_oos, seed=123)
        print(f"  t-test p-value:      {t_pval_oos:.6f}")
        print(f"  Permutation p-value: {perm_pval_oos:.6f}")
        print(f"  Bootstrap P(profit): {prob_profit_oos:.2%}")

    # Monthly breakdown
    if trades_all:
        print(f"\n  --- Monthly Breakdown ---")
        monthly = {}
        for t in trades_all:
            month = t.entry_time.strftime("%Y-%m") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:7]
            monthly.setdefault(month, {"count": 0, "pnl": 0})
            monthly[month]["count"] += 1
            monthly[month]["pnl"] += t.pnl_dollar

        print(f"  {'Month':<10} {'Trades':>7} {'P&L':>12}")
        print(f"  {'-'*30}")
        win_months = 0
        for month in sorted(monthly.keys()):
            data = monthly[month]
            print(f"  {month:<10} {data['count']:>7} ${data['pnl']:>11,.0f}")
            if data["pnl"] > 0:
                win_months += 1
        total_months = len(monthly)
        print(f"\n  Winning months: {win_months}/{total_months} ({win_months/total_months*100:.0f}%)")

    # ── FINAL VERDICT ──
    print(f"\n{'='*90}")
    print(f"  FINAL VERDICT")
    print(f"{'='*90}")

    checks = []
    if m_all:
        checks.append(("Trades >= 30", m_all.total_trades >= 30))
        checks.append(("PF > 1.0", m_all.profit_factor > 1.0))
        checks.append(("p < 0.05 (t-test)", t_pval < 0.05))
    if m_oos and m_is and m_is.profit_factor > 0:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor
        checks.append(("WF PF ratio > 0.70", pf_ratio >= 0.70))
    if m_oos:
        checks.append(("OOS PF > 1.0", m_oos.profit_factor > 1.0))

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n  *** OVERALL: {verdict} ***")
    print(f"{'='*90}")

    return verdict


# ── Main ──

def main():
    t0 = time.time()
    df = load_data()

    # Run sweep
    best_cfg = run_sweep(df)

    # Full validation on best config
    verdict = run_full_validation(df, best_cfg)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
