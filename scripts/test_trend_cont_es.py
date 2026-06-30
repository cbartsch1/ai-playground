#!/usr/bin/env python3
"""Trend Continuation — ES Futures Port (v2: ES-adapted exits).

Ported from SPX TrendLowerCloseOptimized:
  Signal: 30m close < prior 30m LOW (sellers pushed through entire prior bar range)
  Direction: SHORT ONLY
  Stop: Percentage-based (bps)

Key adaptation for ES: SPX version uses options (gamma amplifies small moves), so
"first green bar" exit works there. ES is linear P&L — need bigger moves to overcome
commission/slippage. Exit modes tested:
  1. green_bar      — original (first green 5m bar)
  2. fixed_target   — fixed point target (e.g., 5, 8, 10, 15 pts)
  3. trail_stop     — trailing stop after trigger
  4. green_bar+min  — green bar exit but only after minimum hold (e.g., 3 bars)

Parameter sweep over: entry window, max hold, stop bps, max trades/day, exit mode.
Walk-forward validation + t-test on best config.

Usage:
    python3 scripts/test_trend_cont_es.py
"""

import sys
import os
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics
from backtester.position import Trade

# ── Constants ──
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "es_5m_databento_2yr.csv")
WF_SPLIT = "2025-02-16"
ES_POINT_VALUE = 50.0
COMMISSION_RT = 4.62
SLIPPAGE_PTS = 0.50
CONTRACTS = 1
INITIAL_CAPITAL = 100_000.0
FLATTEN_TIME = 1555


# ── 30m Aggregation ──

def aggregate_30m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5m RTH bars into clock-aligned 30m bars."""
    rth = df_5m[df_5m["is_rth"]].copy()
    if rth.empty:
        return pd.DataFrame()

    rth["m30_bucket"] = rth.index.floor("30min")
    grouped = rth.groupby([rth["session_date"], rth["m30_bucket"]])

    bars_30m = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    bars_30m = bars_30m.droplevel(0).sort_index()
    bars_30m["session_date"] = grouped["session_date"].first().droplevel(0)
    bars_30m["et_time"] = bars_30m.index.hour * 100 + bars_30m.index.minute
    return bars_30m


def compute_30m_lower_close(df_30m: pd.DataFrame) -> pd.DataFrame:
    """Flag 30m bars where close < prior 30m bar's LOW (within same session)."""
    df = df_30m.copy()
    c = df["close"].values
    l = df["low"].values
    s = df["session_date"].values
    n = len(df)

    lower_close = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if s[i] == s[i-1] and c[i] < l[i-1]:
            lower_close[i] = True

    df["lower_close"] = lower_close
    return df


def compute_atr(df_5m: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR on 5m bars for trailing stop sizing."""
    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values
    n = len(df_5m)

    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    tr[0] = h[0] - l[0]

    atr = np.zeros(n)
    atr[:period] = np.mean(tr[:period]) if period <= n else tr[0]
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]

    return atr


# ── Strategy Config ──

@dataclass
class TrendContConfig:
    entry_start: int = 935
    entry_end: int = 1300
    max_hold_bars: int = 9
    stop_bps: float = 30.0
    max_trades_day: int = 2
    min_gap_bars: int = 6
    flatten_time: int = FLATTEN_TIME
    # Exit mode
    exit_mode: str = "green_bar"      # green_bar, fixed_target, trail, green_bar_min
    target_pts: float = 10.0          # for fixed_target mode
    trail_trigger_pts: float = 5.0    # for trail mode: profit to activate
    trail_dist_pts: float = 8.0       # for trail mode: distance from best
    green_bar_min_hold: int = 3       # for green_bar_min: min bars before green exit
    # TEMA filter
    use_tema: bool = False
    tema_fast: int = 9
    tema_slow: int = 21

    @property
    def label(self):
        parts = [
            f"{self.entry_start}-{self.entry_end}",
            f"h{self.max_hold_bars}",
            f"s{self.stop_bps:.0f}",
            f"m{self.max_trades_day}",
            f"{self.exit_mode}",
        ]
        if self.exit_mode == "fixed_target":
            parts.append(f"t{self.target_pts:.0f}")
        elif self.exit_mode == "trail":
            parts.append(f"tr{self.trail_trigger_pts:.0f}/{self.trail_dist_pts:.0f}")
        elif self.exit_mode == "green_bar_min":
            parts.append(f"gm{self.green_bar_min_hold}")
        return "_".join(parts)


# ── Strategy Engine ──

def run_trend_cont(df_5m: pd.DataFrame, df_30m: pd.DataFrame,
                    cfg: TrendContConfig) -> List[Trade]:
    """Run Trend Continuation backtest on ES 5m data."""
    rth_5m = df_5m[df_5m["is_rth"]].copy()
    if rth_5m.empty or df_30m.empty:
        return []

    # Compute 30m lower_close signals
    df_30m_sig = compute_30m_lower_close(df_30m)

    # Build signal lookup — shift 30m timestamp forward to avoid lookahead
    signal_df = df_30m_sig[["lower_close", "session_date"]].copy()
    signal_df.columns = ["m30_lower_close", "m30_session"]
    signal_df.index = signal_df.index + pd.Timedelta(minutes=30)

    merged = pd.merge_asof(
        rth_5m.reset_index(),
        signal_df.reset_index(),
        left_on="datetime",
        right_on=signal_df.index.name or "datetime",
        direction="backward",
    ).set_index("datetime")
    merged["m30_lower_close"] = merged["m30_lower_close"].fillna(False)

    # Compute ATR for trailing stop
    atr = compute_atr(merged, period=14)

    # Extract arrays for speed
    n = len(merged)
    et = merged["et_time"].values
    sess = merged["session_date"].values
    cl = merged["close"].values
    op = merged["open"].values
    hi = merged["high"].values
    lo = merged["low"].values
    sig = merged["m30_lower_close"].values
    times = merged.index

    # Optional TEMA filter
    tema_bearish = np.ones(n, dtype=bool)  # default: always pass
    if cfg.use_tema:
        # Compute TEMA on close
        ema1_f = pd.Series(cl).ewm(span=cfg.tema_fast, adjust=False).mean().values
        ema2_f = pd.Series(ema1_f).ewm(span=cfg.tema_fast, adjust=False).mean().values
        ema3_f = pd.Series(ema2_f).ewm(span=cfg.tema_fast, adjust=False).mean().values
        tema_f = 3 * ema1_f - 3 * ema2_f + ema3_f

        ema1_s = pd.Series(cl).ewm(span=cfg.tema_slow, adjust=False).mean().values
        ema2_s = pd.Series(ema1_s).ewm(span=cfg.tema_slow, adjust=False).mean().values
        ema3_s = pd.Series(ema2_s).ewm(span=cfg.tema_slow, adjust=False).mean().values
        tema_s = 3 * ema1_s - 3 * ema2_s + ema3_s

        tema_bearish = tema_f < tema_s

    # Bar-by-bar simulation
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    entry_idx = 0
    stop_price = 0.0
    target_price = 0.0
    best_price = 0.0
    trail_active = False

    trade_count = {}
    last_entry_bar = {}

    def make_trade(exit_time, exit_price, exit_reason):
        pnl_pts = entry_price - exit_price  # short
        pnl_dollar = (pnl_pts * ES_POINT_VALUE - COMMISSION_RT) * CONTRACTS - SLIPPAGE_PTS * ES_POINT_VALUE * CONTRACTS
        return Trade(
            setup="TC", direction=-1,
            entry_time=entry_time, entry_price=entry_price,
            exit_time=exit_time, exit_price=exit_price,
            exit_reason=exit_reason, pnl_pts=pnl_pts,
            pnl_dollar=pnl_dollar, stop=stop_price, target=target_price,
        )

    for i in range(n):
        s = sess[i]
        t = et[i]

        # ── Check exits first ──
        if in_position:
            entry_sess = sess[entry_idx]
            bars_held = i - entry_idx

            # Session changed
            if s != entry_sess:
                trades.append(make_trade(times[i-1] if i > 0 else times[i], cl[i-1] if i > 0 else cl[i], "session_end"))
                in_position = False

            # Stop hit (high >= stop for short)
            elif hi[i] >= stop_price:
                trades.append(make_trade(times[i], stop_price, "stop"))
                in_position = False
                continue

            # Flatten time
            elif t >= cfg.flatten_time:
                trades.append(make_trade(times[i], cl[i], "flatten"))
                in_position = False
                continue

            # Fixed target hit (low <= target for short)
            elif cfg.exit_mode == "fixed_target" and target_price > 0 and lo[i] <= target_price:
                trades.append(make_trade(times[i], target_price, "target"))
                in_position = False
                continue

            # Trailing stop logic
            elif cfg.exit_mode == "trail":
                best_price = min(best_price, lo[i])
                profit = entry_price - best_price
                if profit >= cfg.trail_trigger_pts:
                    trail_active = True
                if trail_active:
                    trail_level = best_price + cfg.trail_dist_pts
                    if hi[i] >= trail_level:
                        trades.append(make_trade(times[i], trail_level, "trail"))
                        in_position = False
                        continue

            # Max hold
            if in_position and bars_held >= cfg.max_hold_bars:
                trades.append(make_trade(times[i], cl[i], "max_hold"))
                in_position = False
                continue

            # Green bar exit (with optional min hold)
            if in_position:
                if cfg.exit_mode == "green_bar":
                    if cl[i] > op[i]:
                        trades.append(make_trade(times[i], cl[i], "green_bar"))
                        in_position = False
                        continue
                elif cfg.exit_mode == "green_bar_min":
                    if bars_held >= cfg.green_bar_min_hold and cl[i] > op[i]:
                        trades.append(make_trade(times[i], cl[i], "green_bar"))
                        in_position = False
                        continue

            if in_position:
                continue

        # ── Check entry ──
        if in_position:
            continue

        if t < cfg.entry_start or t >= cfg.entry_end:
            continue
        if trade_count.get(s, 0) >= cfg.max_trades_day:
            continue
        if s in last_entry_bar and (i - last_entry_bar[s]) < cfg.min_gap_bars:
            continue
        if not sig[i]:
            continue
        if not tema_bearish[i]:
            continue

        # ENTER SHORT
        entry_price = cl[i]
        entry_time = times[i]
        entry_idx = i
        stop_price = entry_price * (1.0 + cfg.stop_bps / 10000.0)
        best_price = entry_price
        trail_active = False
        in_position = True

        if cfg.exit_mode == "fixed_target":
            target_price = entry_price - cfg.target_pts
        else:
            target_price = 0.0

        trade_count[s] = trade_count.get(s, 0) + 1
        last_entry_bar[s] = i

    # Close any open position
    if in_position:
        trades.append(make_trade(times[-1], cl[-1], "data_end"))

    return trades


# ── Parameter Sweep ──

def parameter_sweep(df_5m, df_30m):
    """Sweep all parameter combinations. Return results DataFrame."""

    configs = []

    # Core grid
    windows = [(935, 1300), (935, 1400), (1000, 1300)]
    holds = [6, 9, 12, 18]                    # 30, 45, 60, 90 min
    stops = [20, 25, 30, 40]
    max_trades_list = [1, 2, 3]

    # Exit mode configurations
    exit_configs = [
        {"exit_mode": "green_bar"},
        {"exit_mode": "green_bar_min", "green_bar_min_hold": 2},
        {"exit_mode": "green_bar_min", "green_bar_min_hold": 3},
        {"exit_mode": "green_bar_min", "green_bar_min_hold": 4},
        {"exit_mode": "fixed_target", "target_pts": 5.0},
        {"exit_mode": "fixed_target", "target_pts": 8.0},
        {"exit_mode": "fixed_target", "target_pts": 10.0},
        {"exit_mode": "fixed_target", "target_pts": 15.0},
        {"exit_mode": "trail", "trail_trigger_pts": 3.0, "trail_dist_pts": 5.0},
        {"exit_mode": "trail", "trail_trigger_pts": 5.0, "trail_dist_pts": 8.0},
        {"exit_mode": "trail", "trail_trigger_pts": 8.0, "trail_dist_pts": 10.0},
    ]

    # Build all configs
    for (es, ee), hold, stop, mt in itertools.product(windows, holds, stops, max_trades_list):
        for ex_cfg in exit_configs:
            cfg = TrendContConfig(
                entry_start=es, entry_end=ee,
                max_hold_bars=hold, stop_bps=stop,
                max_trades_day=mt, **ex_cfg,
            )
            configs.append(cfg)

    # Also test with TEMA filter on the best exit modes
    for (es, ee), hold, stop, mt in itertools.product(
        [(935, 1300), (935, 1400)], [9, 12], [25, 30], [1, 2]
    ):
        for ex_cfg in exit_configs:
            cfg = TrendContConfig(
                entry_start=es, entry_end=ee,
                max_hold_bars=hold, stop_bps=stop,
                max_trades_day=mt, use_tema=True, **ex_cfg,
            )
            configs.append(cfg)

    total = len(configs)
    print(f"\nParameter sweep: {total} combinations")
    print("=" * 80)

    results = []
    for count, cfg in enumerate(configs, 1):
        trades = run_trend_cont(df_5m, df_30m, cfg)

        if len(trades) < 15:
            if count % 200 == 0:
                print(f"  ... {count}/{total} done")
            continue

        m = compute_metrics(trades, INITIAL_CAPITAL)
        pnls = [t.pnl_dollar for t in trades]

        if len(pnls) > 1 and np.std(pnls) > 0:
            t_stat, p_val = stats.ttest_1samp(pnls, 0)
            p_val = p_val / 2 if t_stat > 0 else 1.0
        else:
            p_val = 1.0

        exit_label = cfg.exit_mode
        if cfg.exit_mode == "fixed_target":
            exit_label = f"target_{cfg.target_pts:.0f}"
        elif cfg.exit_mode == "trail":
            exit_label = f"trail_{cfg.trail_trigger_pts:.0f}/{cfg.trail_dist_pts:.0f}"
        elif cfg.exit_mode == "green_bar_min":
            exit_label = f"gb_min{cfg.green_bar_min_hold}"

        results.append({
            "window": f"{cfg.entry_start}-{cfg.entry_end}",
            "max_hold": cfg.max_hold_bars,
            "stop_bps": cfg.stop_bps,
            "max_trades": cfg.max_trades_day,
            "exit_mode": exit_label,
            "tema": cfg.use_tema,
            "trades": m.total_trades,
            "win_rate": m.win_rate,
            "pf": m.profit_factor,
            "net_pnl": m.net_pnl,
            "avg_trade": m.avg_trade,
            "max_dd": m.max_drawdown,
            "sharpe": m.sharpe,
            "p_value": p_val,
        })

        if count % 200 == 0:
            print(f"  ... {count}/{total} done")

    print(f"  ... {total}/{total} done")
    return pd.DataFrame(results)


# ── Walk-Forward ──

def walk_forward_validate(df_5m, df_30m, cfg, split_date=WF_SPLIT):
    split_ts = pd.Timestamp(split_date, tz=df_5m.index.tz)

    df_5m_is = df_5m[df_5m.index < split_ts]
    df_5m_oos = df_5m[df_5m.index >= split_ts]
    df_30m_is = df_30m[df_30m.index < split_ts]
    df_30m_oos = df_30m[df_30m.index >= split_ts]

    trades_is = run_trend_cont(df_5m_is, df_30m_is, cfg)
    trades_oos = run_trend_cont(df_5m_oos, df_30m_oos, cfg)

    m_is = compute_metrics(trades_is, INITIAL_CAPITAL) if trades_is else None
    m_oos = compute_metrics(trades_oos, INITIAL_CAPITAL) if trades_oos else None

    def ttest_pval(pnls):
        if len(pnls) > 1 and np.std(pnls) > 0:
            t, p = stats.ttest_1samp(pnls, 0)
            return p / 2 if t > 0 else 1.0
        return 1.0

    pnls_is = [t.pnl_dollar for t in trades_is]
    pnls_oos = [t.pnl_dollar for t in trades_oos]

    pf_ratio = 0.0
    if m_is and m_oos and m_is.profit_factor > 0:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor

    return {
        "is_trades": len(trades_is),
        "is_wr": m_is.win_rate if m_is else 0,
        "is_pf": m_is.profit_factor if m_is else 0,
        "is_pnl": m_is.net_pnl if m_is else 0,
        "is_dd": m_is.max_drawdown if m_is else 0,
        "is_sharpe": m_is.sharpe if m_is else 0,
        "is_p": ttest_pval(pnls_is),
        "oos_trades": len(trades_oos),
        "oos_wr": m_oos.win_rate if m_oos else 0,
        "oos_pf": m_oos.profit_factor if m_oos else 0,
        "oos_pnl": m_oos.net_pnl if m_oos else 0,
        "oos_dd": m_oos.max_drawdown if m_oos else 0,
        "oos_sharpe": m_oos.sharpe if m_oos else 0,
        "oos_p": ttest_pval(pnls_oos),
        "pf_ratio": pf_ratio,
        "trades_is_list": trades_is,
        "trades_oos_list": trades_oos,
    }


# ── Reporting ──

def print_metrics(label, trades):
    if not trades:
        print(f"\n  {label}: 0 trades")
        return

    m = compute_metrics(trades, INITIAL_CAPITAL)
    pnls = [t.pnl_dollar for t in trades]

    if len(pnls) > 1 and np.std(pnls) > 0:
        t_stat, p_val = stats.ttest_1samp(pnls, 0)
        p_val = p_val / 2 if t_stat > 0 else 1.0
    else:
        p_val = 1.0

    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    reason_str = ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))

    print(f"\n  {label}")
    print(f"  {'─'*55}")
    print(f"  Trades:       {m.total_trades}")
    print(f"  Win Rate:     {m.win_rate:.1f}%")
    print(f"  Profit Factor: {m.profit_factor:.2f}")
    print(f"  Net P&L:      ${m.net_pnl:,.0f}")
    print(f"  Avg Trade:    ${m.avg_trade:,.0f}")
    print(f"  Avg Win:      ${m.avg_win:,.0f}")
    print(f"  Avg Loss:     ${m.avg_loss:,.0f}")
    print(f"  Max Drawdown: ${m.max_drawdown:,.0f} ({m.max_drawdown_pct:.1f}%)")
    print(f"  Sharpe:       {m.sharpe:.2f}")
    print(f"  p-value:      {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")
    print(f"  Win Streak:   {m.longest_win_streak}  |  Loss Streak: {m.longest_lose_streak}")
    print(f"  Exits:        {reason_str}")


def print_top_configs(df_results, top_n=20):
    viable = df_results[(df_results["trades"] >= 30) & (df_results["pf"] > 1.0)].copy()

    if viable.empty:
        print("\n  No configs with PF > 1.0 and >= 30 trades")
        viable = df_results[df_results["trades"] >= 20].sort_values("pf", ascending=False).head(top_n)
        if viable.empty:
            return
        print(f"\n  Showing top {min(top_n, len(viable))} by PF (min 20 trades):")
    else:
        viable = viable.sort_values("pf", ascending=False).head(top_n)
        print(f"\n  TOP {min(top_n, len(viable))} CONFIGS (PF > 1.0, >= 30 trades)")

    print(f"  {'─'*120}")
    print(f"  {'Window':<12} {'Hold':<5} {'Stop':<5} {'M/D':<4} {'Exit':<16} {'TEMA':<5} "
          f"{'Trades':>6} {'WR%':>6} {'PF':>6} {'Net P&L':>10} {'Avg':>8} {'DD':>8} {'Sharpe':>7} {'p':>7}")
    print(f"  {'─'*120}")

    for _, r in viable.iterrows():
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        tema_str = "Y" if r["tema"] else ""
        print(f"  {r['window']:<12} {r['max_hold']:<5} {r['stop_bps']:<5.0f} {r['max_trades']:<4} "
              f"{r['exit_mode']:<16} {tema_str:<5} "
              f"{r['trades']:>6} {r['win_rate']:>5.1f}% {r['pf']:>6.2f} "
              f"${r['net_pnl']:>9,.0f} ${r['avg_trade']:>7,.0f} "
              f"${r['max_dd']:>7,.0f} {r['sharpe']:>7.2f} {r['p_value']:>6.4f}{sig}")


def print_sensitivity(df_results):
    print(f"\n\n{'='*70}")
    print(f"  PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*70}")

    # Only PF > 0 and enough trades
    viable = df_results[df_results["trades"] >= 20].copy()
    if viable.empty:
        print("  Not enough data")
        return

    # By exit mode
    print(f"\n  BY EXIT MODE:")
    print(f"  {'Exit':<20} {'Avg PF':>8} {'Avg Trades':>10} {'Avg P&L':>10} {'Best PF':>8} {'Configs':>8}")
    for mode in sorted(viable["exit_mode"].unique()):
        sub = viable[viable["exit_mode"] == mode]
        print(f"  {mode:<20} {sub['pf'].mean():>8.2f} {sub['trades'].mean():>10.0f} "
              f"${sub['net_pnl'].mean():>9,.0f} {sub['pf'].max():>8.2f} {len(sub):>8}")

    # By entry window
    print(f"\n  BY ENTRY WINDOW:")
    print(f"  {'Window':<12} {'Avg PF':>8} {'Best PF':>8} {'Avg Trades':>10}")
    for window in sorted(viable["window"].unique()):
        sub = viable[viable["window"] == window]
        print(f"  {window:<12} {sub['pf'].mean():>8.2f} {sub['pf'].max():>8.2f} {sub['trades'].mean():>10.0f}")

    # By max hold
    print(f"\n  BY MAX HOLD:")
    print(f"  {'Hold':<10} {'Avg PF':>8} {'Best PF':>8}")
    for hold in sorted(viable["max_hold"].unique()):
        sub = viable[viable["max_hold"] == hold]
        print(f"  {hold:<4} ({hold*5}m)  {sub['pf'].mean():>8.2f} {sub['pf'].max():>8.2f}")

    # By stop bps
    print(f"\n  BY STOP BPS:")
    print(f"  {'Stop':<8} {'Avg PF':>8} {'Best PF':>8}")
    for stop in sorted(viable["stop_bps"].unique()):
        sub = viable[viable["stop_bps"] == stop]
        print(f"  {stop:<8.0f} {sub['pf'].mean():>8.2f} {sub['pf'].max():>8.2f}")

    # By max trades/day
    print(f"\n  BY MAX TRADES/DAY:")
    print(f"  {'Max/D':<8} {'Avg PF':>8} {'Best PF':>8}")
    for mt in sorted(viable["max_trades"].unique()):
        sub = viable[viable["max_trades"] == mt]
        print(f"  {mt:<8} {sub['pf'].mean():>8.2f} {sub['pf'].max():>8.2f}")

    # TEMA vs no TEMA
    print(f"\n  TEMA FILTER:")
    for tema in [False, True]:
        sub = viable[viable["tema"] == tema]
        if not sub.empty:
            label = "With TEMA" if tema else "No TEMA"
            print(f"  {label:<12} Avg PF {sub['pf'].mean():.2f}, Best PF {sub['pf'].max():.2f}, "
                  f"Avg trades {sub['trades'].mean():.0f}")


# ── Main ──

def main():
    print("=" * 70)
    print("  TREND CONTINUATION — ES FUTURES PORT (v2)")
    print("  Signal: 30m close < prior 30m LOW | SHORT ONLY")
    print("  Exit modes: green_bar, green_bar+min, fixed_target, trail")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df_5m = load_tos_csv(DATA_PATH)
    print(f"  Loaded {len(df_5m):,} bars, {df_5m.index[0]} to {df_5m.index[-1]}")

    # Aggregate 30m
    print("Aggregating 30m bars...")
    df_30m = aggregate_30m(df_5m)
    print(f"  Generated {len(df_30m):,} 30m bars")

    # ── Phase 1: Full sweep ──
    print(f"\n{'='*70}")
    print(f"  PHASE 1: FULL-DATASET PARAMETER SWEEP")
    print(f"{'='*70}")

    df_results = parameter_sweep(df_5m, df_30m)

    if df_results.empty:
        print("\n  FAIL: No parameter combinations produced trades.")
        return

    print_top_configs(df_results)
    print_sensitivity(df_results)

    # ── Phase 2: Best config ──
    viable = df_results[(df_results["trades"] >= 30) & (df_results["pf"] > 1.0)]
    if viable.empty:
        viable = df_results[df_results["trades"] >= 20]

    if viable.empty:
        print("\n  FAIL: No viable configs.")
        return

    best = viable.sort_values("pf", ascending=False).iloc[0]

    print(f"\n\n{'='*70}")
    print(f"  PHASE 2: BEST CONFIG — DETAILED ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  Best: window={best['window']}, hold={best['max_hold']}, "
          f"stop={best['stop_bps']:.0f}bps, max_trades={best['max_trades']}, "
          f"exit={best['exit_mode']}, tema={best['tema']}")

    # Reconstruct config
    w = best["window"].split("-")
    exit_kwargs = {}
    em = best["exit_mode"]
    if em.startswith("target_"):
        exit_kwargs = {"exit_mode": "fixed_target", "target_pts": float(em.split("_")[1])}
    elif em.startswith("trail_"):
        parts = em.replace("trail_", "").split("/")
        exit_kwargs = {"exit_mode": "trail", "trail_trigger_pts": float(parts[0]), "trail_dist_pts": float(parts[1])}
    elif em.startswith("gb_min"):
        exit_kwargs = {"exit_mode": "green_bar_min", "green_bar_min_hold": int(em.replace("gb_min", ""))}
    else:
        exit_kwargs = {"exit_mode": "green_bar"}

    best_cfg = TrendContConfig(
        entry_start=int(w[0]), entry_end=int(w[1]),
        max_hold_bars=int(best["max_hold"]),
        stop_bps=float(best["stop_bps"]),
        max_trades_day=int(best["max_trades"]),
        use_tema=bool(best["tema"]),
        **exit_kwargs,
    )

    all_trades = run_trend_cont(df_5m, df_30m, best_cfg)
    print_metrics("FULL DATASET (2yr)", all_trades)

    # Monthly breakdown
    if all_trades:
        print(f"\n  MONTHLY BREAKDOWN:")
        monthly = {}
        for t in all_trades:
            if hasattr(t.entry_time, 'strftime'):
                month = t.entry_time.strftime("%Y-%m")
            else:
                month = str(t.entry_time)[:7]
            monthly.setdefault(month, []).append(t.pnl_dollar)

        print(f"  {'Month':<10} {'Trades':>7} {'Net P&L':>10} {'Avg':>8} {'WR%':>6}")
        win_months = 0
        for month in sorted(monthly.keys()):
            pnls = monthly[month]
            net = sum(pnls)
            avg = np.mean(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            if net > 0:
                win_months += 1
            print(f"  {month:<10} {len(pnls):>7} ${net:>9,.0f} ${avg:>7,.0f} {wr:>5.1f}%")
        print(f"  Winning months: {win_months}/{len(monthly)} ({win_months/len(monthly)*100:.0f}%)")

    # ── Phase 3: Walk-forward on best config ──
    print(f"\n\n{'='*70}")
    print(f"  PHASE 3: WALK-FORWARD VALIDATION (split: {WF_SPLIT})")
    print(f"{'='*70}")

    wf = walk_forward_validate(df_5m, df_30m, best_cfg)

    print(f"\n  {'Metric':<20} {'In-Sample':>15} {'Out-of-Sample':>15}")
    print(f"  {'─'*52}")
    print(f"  {'Trades':<20} {wf['is_trades']:>15} {wf['oos_trades']:>15}")
    print(f"  {'Win Rate':<20} {wf['is_wr']:>14.1f}% {wf['oos_wr']:>14.1f}%")
    print(f"  {'Profit Factor':<20} {wf['is_pf']:>15.2f} {wf['oos_pf']:>15.2f}")
    print(f"  {'Net P&L':<20} ${wf['is_pnl']:>14,.0f} ${wf['oos_pnl']:>14,.0f}")
    print(f"  {'Max Drawdown':<20} ${wf['is_dd']:>14,.0f} ${wf['oos_dd']:>14,.0f}")
    print(f"  {'Sharpe':<20} {wf['is_sharpe']:>15.2f} {wf['oos_sharpe']:>15.2f}")
    print(f"  {'p-value':<20} {wf['is_p']:>15.4f} {wf['oos_p']:>15.4f}")
    print(f"\n  PF Ratio (OOS/IS): {wf['pf_ratio']:.2f} {'PASS (>0.7)' if wf['pf_ratio'] > 0.7 else 'FAIL (<0.7)'}")

    print_metrics("IN-SAMPLE", wf["trades_is_list"])
    print_metrics("OUT-OF-SAMPLE", wf["trades_oos_list"])

    # ── Phase 4: Walk-forward top 5 ──
    print(f"\n\n{'='*70}")
    print(f"  PHASE 4: WALK-FORWARD ON TOP 5 CONFIGS")
    print(f"{'='*70}")

    top5 = viable.sort_values("pf", ascending=False).head(5)

    print(f"\n  {'Window':<12} {'Hold':<5} {'Stop':<5} {'M/D':<4} {'Exit':<16} "
          f"{'IS PF':>7} {'OOS PF':>8} {'Ratio':>7} {'OOS p':>7} {'Result':>8}")
    print(f"  {'─'*90}")

    pass_count = 0
    for _, r in top5.iterrows():
        w2 = r["window"].split("-")
        ek = {}
        em2 = r["exit_mode"]
        if em2.startswith("target_"):
            ek = {"exit_mode": "fixed_target", "target_pts": float(em2.split("_")[1])}
        elif em2.startswith("trail_"):
            parts = em2.replace("trail_", "").split("/")
            ek = {"exit_mode": "trail", "trail_trigger_pts": float(parts[0]), "trail_dist_pts": float(parts[1])}
        elif em2.startswith("gb_min"):
            ek = {"exit_mode": "green_bar_min", "green_bar_min_hold": int(em2.replace("gb_min", ""))}
        else:
            ek = {"exit_mode": "green_bar"}

        cfg = TrendContConfig(
            entry_start=int(w2[0]), entry_end=int(w2[1]),
            max_hold_bars=int(r["max_hold"]),
            stop_bps=float(r["stop_bps"]),
            max_trades_day=int(r["max_trades"]),
            use_tema=bool(r["tema"]),
            **ek,
        )
        wf_r = walk_forward_validate(df_5m, df_30m, cfg)

        result = "PASS" if wf_r["pf_ratio"] > 0.7 and wf_r["oos_pf"] > 1.0 else "FAIL"
        if result == "PASS":
            pass_count += 1
        sig = "*" if wf_r["oos_p"] < 0.05 else ""

        print(f"  {r['window']:<12} {r['max_hold']:<5} {r['stop_bps']:<5.0f} {r['max_trades']:<4} "
              f"{r['exit_mode']:<16} "
              f"{wf_r['is_pf']:>7.2f} {wf_r['oos_pf']:>8.2f} {wf_r['pf_ratio']:>7.2f} "
              f"{wf_r['oos_p']:>6.4f}{sig} {result:>8}")

    # ── Final Verdict ──
    print(f"\n\n{'='*70}")
    print(f"  FINAL VERDICT")
    print(f"{'='*70}")

    full_pf = best["pf"]
    full_p = best["p_value"]
    oos_pf = wf["oos_pf"]
    oos_p = wf["oos_p"]
    pf_ratio = wf["pf_ratio"]

    checks = [
        ("Full dataset p < 0.05", full_p < 0.05, f"p = {full_p:.4f}"),
        ("Full dataset PF > 1.0", full_pf > 1.0, f"PF = {full_pf:.2f}"),
        ("OOS PF > 1.0", oos_pf > 1.0, f"OOS PF = {oos_pf:.2f}"),
        ("WF PF ratio > 0.7", pf_ratio > 0.7, f"ratio = {pf_ratio:.2f}"),
        ("OOS p < 0.10", oos_p < 0.10, f"OOS p = {oos_p:.4f}"),
        ("Robustness (3/5 top pass WF)", pass_count >= 3, f"{pass_count}/5 pass"),
    ]

    all_pass = all(c[1] for c in checks)

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} — {detail}")

    print(f"\n  {'='*50}")
    passed_count = sum(1 for c in checks if c[1])
    if all_pass:
        print(f"  RESULT: PASS — Trend Continuation ES is validated")
    elif passed_count >= 4:
        print(f"  RESULT: CONDITIONAL PASS — {passed_count}/{len(checks)} checks passed")
    else:
        print(f"  RESULT: FAIL — {passed_count}/{len(checks)} checks passed")
    print(f"  {'='*50}")

    print(f"\n  Best config: {best['window']} window, {best['max_hold']:.0f} bar hold, "
          f"{best['stop_bps']:.0f}bps stop, {best['max_trades']:.0f}/day, "
          f"exit={best['exit_mode']}, tema={best['tema']}")
    print(f"  Full dataset: {best['trades']:.0f} trades, PF {best['pf']:.2f}, "
          f"${best['net_pnl']:,.0f}, Sharpe {best['sharpe']:.2f}")
    if wf:
        print(f"  Walk-forward: IS PF {wf['is_pf']:.2f}, OOS PF {wf['oos_pf']:.2f}, "
              f"ratio {wf['pf_ratio']:.2f}")


if __name__ == "__main__":
    main()
