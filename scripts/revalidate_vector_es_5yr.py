#!/usr/bin/env python3
"""Vector ES — 5yr revalidation on clean databento ES futures data.

Production config (SOURCE OF TRUTH = dashboard vector_es.py):
  - EMA 9 / EMA 21 bearish cross on 5m ES bars
  - SHORT ONLY
  - ON gap filter: prev_close - onl >= 15 pts
  - Stop: 20 bps above entry
  - Target: overnight low (ONL)
  - Max 2 trades/day
  - Entry window: 09:35 – 15:00 ET
  - 1 contract, $50/pt, $5 RT commission, 1 tick slippage
  - Time stop: 15:55 (session end)

Datasets:
  - 5yr: es_5m_5yr.csv (2017-01-02 → 2021-12-31, 350,527 bars)
  - 2yr (clean): es_5m_databento_2yr.csv (2024-02 → 2026-02, 140,149 bars)

Walk-forward splits:
  - 5yr: IS 2017-2019 / OOS 2020-2021 (split 2020-01-01)
  - 2yr: IS ~1yr / OOS ~1yr (split 2025-02-16)

Verdict thresholds:
  HOLD:    5yr PF > 1.3, OOS PF > 1.0, OOS p < 0.05
  DEGRADE: 5yr PF in 1.0-1.3, OOS PF > 0.9 (marginal)
  BREAK:   5yr PF < 1.0 or OOS PF < 0.9 or OOS p > 0.10
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics
from backtester.position import Trade


# ── PRODUCTION CONFIG (from dashboard vector_es.py) ──

@dataclass
class VectorProdConfig:
    ema_fast: int = 9
    ema_slow: int = 21
    on_gap_threshold: float = 15.0   # prev_close - onl >= 15
    stop_bps: float = 20.0
    max_trades_per_day: int = 2
    entry_start: int = 935
    entry_end: int = 1500            # exclusive
    time_stop: int = 1555            # force flatten at 15:55 ET

    point_value: float = 50.0
    commission_rt: float = 5.0
    slippage_ticks: int = 1
    tick_size: float = 0.25

    # Warmup bars before EMA valid (matches dashboard MIN_CLOSES_FOR_WARM)
    min_closes_for_warm: int = 24

    initial_capital: float = 100_000.0


# ── ONH/ONL/prev_close computation (matches dashboard semantics) ──

def compute_session_levels(df: pd.DataFrame) -> dict:
    """For each RTH session_date, compute:
        onh = max high of Globex (prev 18:00 ET) bars BEFORE RTH open
        onl = min low of same Globex bars
        prev_close = close of last RTH bar of prior RTH session

    Returns: {session_date (date obj): (onh, onl, prev_close)}

    Semantics match dashboard: values frozen at RTH open (new_rth bar).
    """
    levels = {}

    # Sort by time just in case
    df = df.sort_index()
    rth_mask = df["is_rth"].values
    globex_mask = df["is_globex"].values
    session_dates = df["session_date"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # Walk through, tracking globex accumulator between RTH sessions
    # Each RTH session_date has its own overnight session preceding its first RTH bar.
    # The data_loader sets session_date for Globex bars to the SAME RTH date they precede
    # (confirmed by data_loader.tag_sessions ffill logic — but ffill assigns prior RTH date to
    # following Globex bars. So Globex bars for Monday RTH have Friday's session_date until
    # new_rth fires Monday morning).

    # Safer approach: iterate and build ON high/low from the stretch of non-RTH bars that
    # IMMEDIATELY PRECEDE each RTH session.
    # We'll do it by iterating chronologically. Reset ON accumulators on new_globex / after RTH
    # close; freeze on new_rth.

    new_rth = df["new_rth"].values
    new_globex = df["new_globex"].values

    on_high = np.nan
    on_low = np.nan
    last_rth_close = np.nan
    prev_rth_last_close = np.nan  # the close we'll use for NEXT session's prev_close

    for i in range(len(df)):
        if new_rth[i]:
            # Session opens: freeze ON levels and assign prev_close
            sd = session_dates[i]
            if not isinstance(sd, (pd.Timestamp,)) and sd is not None:
                try:
                    sd_key = pd.Timestamp(sd).date() if not isinstance(sd, type(pd.Timestamp('2017-01-03').date())) else sd
                except Exception:
                    sd_key = sd
            else:
                sd_key = sd
            levels[sd_key] = (
                float(on_high) if not np.isnan(on_high) else None,
                float(on_low) if not np.isnan(on_low) else None,
                float(prev_rth_last_close) if not np.isnan(prev_rth_last_close) else None,
            )
            # Reset ON accumulators for next overnight
            on_high = np.nan
            on_low = np.nan
            # Begin tracking the RTH close for THIS session
            last_rth_close = closes[i]
        elif rth_mask[i]:
            # Inside RTH: update running close (will become prev_close for next session)
            last_rth_close = closes[i]
        else:
            # Non-RTH bar (globex / after-hours). Accumulate into ON range.
            # At the end of the RTH session (transition from RTH to non-RTH), we should
            # snapshot last_rth_close to prev_rth_last_close so it's ready for NEXT new_rth.
            if i > 0 and rth_mask[i - 1] and not rth_mask[i]:
                # Just left RTH — snapshot close
                if not np.isnan(last_rth_close):
                    prev_rth_last_close = last_rth_close
            # Accumulate ON high/low (only for truly Globex bars; safer to use all non-RTH)
            h = highs[i]
            l = lows[i]
            on_high = h if np.isnan(on_high) else max(on_high, h)
            on_low = l if np.isnan(on_low) else min(on_low, l)

    return levels


# ── Backtest engine (production logic, single config) ──

def run_prod_backtest(df: pd.DataFrame, cfg: VectorProdConfig) -> List[Trade]:
    """Run Vector ES production config on df. Returns list of Trade."""
    levels = compute_session_levels(df)

    # Compute EMAs (continuous, not reset per session — matches dashboard)
    close = df["close"].values
    k_fast = 2.0 / (cfg.ema_fast + 1)
    k_slow = 2.0 / (cfg.ema_slow + 1)
    n = len(df)

    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    ema_count = 0
    f = 0.0
    s = 0.0
    for i in range(n):
        ema_count += 1
        if ema_count == 1:
            f = close[i]
            s = close[i]
        else:
            f = close[i] * k_fast + f * (1 - k_fast)
            s = close[i] * k_slow + s * (1 - k_slow)
        ema_fast[i] = f
        ema_slow[i] = s

    # Detect bearish crosses (prev_fast >= prev_slow AND fast < slow)
    cross_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if ema_fast[i - 1] >= ema_slow[i - 1] and ema_fast[i] < ema_slow[i]:
            cross_bear[i] = True

    # Iterate bars
    et = df["et_time"].values
    sess = df["session_date"].values
    cl = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    is_rth = df["is_rth"].values
    timestamps = df.index
    slippage = cfg.slippage_ticks * cfg.tick_size

    trades: List[Trade] = []
    position = None
    trades_today = 0
    current_session = None

    for i in range(cfg.min_closes_for_warm, n):
        s_i = sess[i]

        # Session rollover: flatten any open position
        if s_i != current_session:
            if position is not None:
                exit_price = cl[i - 1] + slippage
                pnl_pts = position["entry_price"] - exit_price
                pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
                trades.append(Trade(
                    setup="VEC_ES_PROD",
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
            current_session = s_i
            trades_today = 0

        # ── EXITS ──
        if position is not None:
            exit_reason = None
            exit_price = None

            # Stop (high hits stop)
            if hi[i] >= position["stop"]:
                exit_reason = "stop"
                exit_price = position["stop"] + slippage
            # Target (ONL) hit
            elif position["target"] > 0 and lo[i] <= position["target"]:
                exit_reason = "target"
                exit_price = position["target"] - slippage
            # Time stop at 15:55 ET
            elif is_rth[i] and et[i] >= cfg.time_stop:
                exit_reason = "time_stop"
                exit_price = cl[i]

            if exit_reason is not None:
                pnl_pts = position["entry_price"] - exit_price
                pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
                trades.append(Trade(
                    setup="VEC_ES_PROD",
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

        # ── ENTRY ──
        if position is None and is_rth[i]:
            # Time window
            if et[i] < cfg.entry_start or et[i] >= cfg.entry_end:
                continue
            if trades_today >= cfg.max_trades_per_day:
                continue
            if not cross_bear[i]:
                continue

            # ON gap filter
            lvl = levels.get(s_i)
            if lvl is None:
                continue
            onh, onl, prev_close = lvl
            if onl is None or prev_close is None:
                continue
            on_gap = prev_close - onl
            if on_gap < cfg.on_gap_threshold:
                continue

            # Enter short
            entry_price = cl[i] - slippage
            stop_price = entry_price * (1 + cfg.stop_bps / 10000.0)
            target_price = onl

            position = {
                "entry_idx": i,
                "entry_price": entry_price,
                "stop": stop_price,
                "target": target_price,
                "session": s_i,
            }
            trades_today += 1

    # Close any remaining position at end of data
    if position is not None:
        exit_price = cl[-1] + slippage
        pnl_pts = position["entry_price"] - exit_price
        pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
        trades.append(Trade(
            setup="VEC_ES_PROD",
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


# ── Stats ──

def significance(trades: List[Trade], seed: int = 42) -> Tuple[float, float, float]:
    """Returns (t_pval_one_tailed, perm_pval, prob_profit)."""
    pnls = np.array([t.pnl_dollar for t in trades])
    n = len(pnls)
    if n < 5:
        return 1.0, 1.0, 0.0

    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    t_pval_one = t_pval / 2 if t_stat > 0 else 1 - t_pval / 2

    obs_pnl = np.sum(pnls)
    abs_pnls = np.abs(pnls)
    rng = np.random.default_rng(seed)
    n_perm = 5000
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if np.dot(signs, abs_pnls) >= obs_pnl:
            count += 1
    perm_pval = count / n_perm

    n_boot = 5000
    boot = np.array([np.sum(rng.choice(pnls, size=n, replace=True)) for _ in range(n_boot)])
    prob_profit = float(np.mean(boot > 0))

    return t_pval_one, perm_pval, prob_profit


def yearly_breakdown(trades: List[Trade]) -> dict:
    """Group by entry year, return {year: (n_trades, pnl, wr)}."""
    yr = {}
    for t in trades:
        y = t.entry_time.year
        yr.setdefault(y, []).append(t)
    result = {}
    for y, tlist in sorted(yr.items()):
        pnls = [x.pnl_dollar for x in tlist]
        wins = [p for p in pnls if p > 0]
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        result[y] = (len(tlist), sum(pnls), wr)
    return result


def print_summary(label: str, trades: List[Trade], capital: float):
    m = compute_metrics(trades, capital)
    t_p, perm_p, prob = significance(trades)
    print(f"\n  ── {label} ──")
    print(f"  Trades: {m.total_trades}  |  Winners: {m.winners}  |  Losers: {m.losers}")
    print(f"  Win rate: {m.win_rate:.1f}%")
    print(f"  Profit factor: {m.profit_factor:.3f}")
    print(f"  Net P&L: ${m.net_pnl:,.0f}")
    print(f"  Avg trade: ${m.avg_trade:,.0f}")
    print(f"  Max DD: ${m.max_drawdown:,.0f}  ({m.max_drawdown_pct:.1f}%)")
    print(f"  Sharpe (ann): {m.sharpe:.2f}")
    print(f"  t-test p (one-tailed): {t_p:.4f}")
    print(f"  Permutation p: {perm_p:.4f}")
    print(f"  Bootstrap P(profit): {prob*100:.1f}%")
    return m, t_p, perm_p, prob


def run_dataset(path: str, label: str, wf_split: str, cfg: VectorProdConfig):
    print("\n" + "=" * 90)
    print(f"  DATASET: {label}")
    print(f"  File: {path}")
    print(f"  WF split: {wf_split}")
    print("=" * 90)

    df = load_tos_csv(path, instrument="ES")
    print(f"  {len(df):,} bars | {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    # Full backtest
    trades_all = run_prod_backtest(df, cfg)
    m_all, t_all, perm_all, prob_all = print_summary(f"{label} — FULL", trades_all, cfg.initial_capital)

    # Yearly breakdown
    print(f"\n  Yearly breakdown:")
    print(f"  {'Year':>6} {'Trades':>8} {'P&L':>14} {'WR':>8}")
    for y, (n, pnl, wr) in yearly_breakdown(trades_all).items():
        print(f"  {y:>6} {n:>8} ${pnl:>12,.0f} {wr:>7.1f}%")

    # Walk-forward
    print(f"\n  Walk-forward split at {wf_split}:")
    df_is = df[df.index < wf_split].copy()
    df_oos = df[df.index >= wf_split].copy()
    t_is = run_prod_backtest(df_is, cfg)
    t_oos = run_prod_backtest(df_oos, cfg)
    m_is, t_is_p, _, _ = print_summary(f"{label} — IS ({df_is.index[0].date() if len(df_is) else '?'} → {df_is.index[-1].date() if len(df_is) else '?'})", t_is, cfg.initial_capital)
    m_oos, t_oos_p, perm_oos_p, prob_oos = print_summary(f"{label} — OOS ({df_oos.index[0].date() if len(df_oos) else '?'} → {df_oos.index[-1].date() if len(df_oos) else '?'})", t_oos, cfg.initial_capital)

    return {
        "all": (trades_all, m_all, t_all, perm_all),
        "is": (t_is, m_is, t_is_p),
        "oos": (t_oos, m_oos, t_oos_p, perm_oos_p, prob_oos),
    }


def main():
    cfg = VectorProdConfig()
    print("Vector ES 5yr revalidation — production config")
    print(f"  EMA {cfg.ema_fast}/{cfg.ema_slow}  |  ON gap >= {cfg.on_gap_threshold}pt")
    print(f"  Stop {cfg.stop_bps}bps  |  Target = ONL  |  Max {cfg.max_trades_per_day}/day")
    print(f"  Window {cfg.entry_start}-{cfg.entry_end} ET  |  Time stop {cfg.time_stop}")
    print(f"  $50/pt, ${cfg.commission_rt} RT commission, {cfg.slippage_ticks} tick slippage")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 5yr dataset: 2017-2021
    path_5yr = os.path.join(base, "data", "es_5m_5yr.csv")
    res_5yr = run_dataset(path_5yr, "5yr 2017-2021", "2020-01-01", cfg)

    # 2yr clean dataset (for comparison to prior "PF 3.44, 252t")
    path_2yr = os.path.join(base, "data", "es_5m_databento_2yr.csv")
    res_2yr = run_dataset(path_2yr, "2yr 2024-2026 (clean)", "2025-02-16", cfg)

    # Verdict
    print("\n" + "=" * 90)
    print("  VERDICT")
    print("=" * 90)

    m_5yr_all = res_5yr["all"][1]
    m_5yr_oos = res_5yr["oos"][1]
    oos_5yr_pf = m_5yr_oos.profit_factor
    oos_5yr_p = res_5yr["oos"][2]

    full_pf = m_5yr_all.profit_factor
    print(f"\n  5yr Full  PF: {full_pf:.3f}  |  P&L: ${m_5yr_all.net_pnl:,.0f}  |  Trades: {m_5yr_all.total_trades}")
    print(f"  5yr OOS   PF: {oos_5yr_pf:.3f}  |  P&L: ${m_5yr_oos.net_pnl:,.0f}  |  Trades: {m_5yr_oos.total_trades}  |  p: {oos_5yr_p:.4f}")

    if full_pf > 1.3 and oos_5yr_pf > 1.0 and oos_5yr_p < 0.05:
        verdict = "HOLD — edge confirmed on 5yr clean data"
    elif full_pf >= 1.0 and full_pf <= 1.3 and oos_5yr_pf > 0.9:
        verdict = "DEGRADE — edge exists but weaker than 2yr"
    elif full_pf < 1.0 or oos_5yr_pf < 0.9 or oos_5yr_p > 0.10:
        verdict = "BREAK — edge does not hold on 5yr clean data"
    else:
        verdict = "DEGRADE (mixed signals)"

    print(f"\n  >>> {verdict} <<<")


if __name__ == "__main__":
    main()
