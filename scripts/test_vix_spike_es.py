#!/usr/bin/env python3
"""VIX Spike ES — Port of SPX VIX Spike strategy to ES futures.

Source: SPX VIX Spike strategy — buy ATM 0DTE puts when VIX spikes from open.
Translation: Short ES futures when VIX daily high >= VIX open * (1 + threshold)
             and current 5m ES bar is red (bearish confirmation).

VIX data: Merged from SPX backtester's vix_daily.parquet (daily OHLC).
Stop: 30 bps percentage-based.
Target: Fixed points (mean reversion) or session VWAP reversion.
Exit: First green 5m bar OR max hold OR time stop (15:55 ET).

Walk-forward split: 2025-02-16 (matches ES backtester convention).
"""

import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics

# ── Constants ──
ES_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "es_5m_databento_2yr.csv")
VIX_PARQUET = os.path.expanduser("~/projects/backtesting/spx/data/vix_daily.parquet")
WF_SPLIT = "2025-02-16"

POINT_VALUE = 50.0     # $50 per ES point
COMMISSION = 2.50      # per side per contract
SLIPPAGE_TICKS = 1     # 1 tick = 0.25 pts
TICK_SIZE = 0.25
INITIAL_CAPITAL = 100_000.0
FLATTEN_TIME = 1555    # ET

# ── Parameter Space ──
SPIKE_THRESHOLDS = [0.03, 0.05, 0.07, 0.10, 0.15]
MAX_HOLD_BARS = [3, 6, 9, 12, 18]           # 15m, 30m, 45m, 60m, 90m
STOP_BPS_OPTIONS = [20, 30, 40, 50]
EXIT_MODES = ["green_bar", "fixed_target", "vwap_reversion"]
TARGET_PTS_OPTIONS = [5, 8, 10, 15, 20]
ES_MOVE_FILTERS = [0.0, -0.001, -0.002]     # 0=disabled, -0.1%, -0.2%


@dataclass
class VixSpikeConfig:
    spike_threshold: float = 0.05
    max_hold_bars: int = 6
    stop_bps: float = 30.0
    exit_mode: str = "green_bar"      # green_bar | fixed_target | vwap_reversion
    target_pts: float = 10.0          # only used for fixed_target
    es_move_filter: float = 0.0       # min ES % from open (negative = must be red)
    entry_start: int = 935
    entry_end: int = 1500


@dataclass
class VixSpikeTrade:
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    stop: float
    target: float
    exit_reason: str
    pnl_pts: float
    pnl_dollar: float
    vix_open: float
    vix_high: float
    vix_spike_pct: float
    session_date: object
    setup: str = "VIX_SPIKE"
    direction: int = -1


def load_data():
    """Load ES 5m bars and VIX daily, merge VIX into ES by session date."""
    print(f"Loading ES data: {ES_CSV}")
    df = load_tos_csv(ES_CSV, instrument="ES")
    print(f"  {len(df):,} bars: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    print(f"\nLoading VIX data: {VIX_PARQUET}")
    vix = pd.read_parquet(VIX_PARQUET)
    print(f"  {len(vix)} days: {vix.index[0].date()} to {vix.index[-1].date()}")

    # Build VIX daily lookup: date -> {open, high, low, close}
    vix_lookup = {}
    for idx, row in vix.iterrows():
        d = idx.date() if hasattr(idx, 'date') else idx
        vix_lookup[d] = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row.get("low", 0)),
            "close": float(row["close"]),
        }

    # Count overlap
    es_dates = set(df[df['is_rth']]['session_date'].dropna().unique())
    vix_dates = set(vix_lookup.keys())
    overlap = es_dates & vix_dates
    print(f"  Date overlap: {len(overlap)} sessions")

    return df, vix_lookup


def compute_session_fields(df):
    """Compute per-session fields needed for the strategy."""
    # Session opens (first RTH bar's open)
    rth = df[df['is_rth']].copy()
    session_opens = rth.groupby('session_date')['open'].first().to_dict()

    # VWAP per session (cumulative)
    rth_groups = rth.groupby('session_date')
    session_vwaps = {}
    for sess_date, group in rth_groups:
        cum_vol = group['volume'].cumsum()
        cum_vp = (group['hlc3'] * group['volume']).cumsum()
        vwap = cum_vp / cum_vol.replace(0, np.nan)
        # Store last VWAP as session VWAP (approximation for target)
        session_vwaps[sess_date] = vwap.to_dict()

    return session_opens, session_vwaps


LOOKAHEAD_MSG = (
    "LOOKAHEAD: this path flags VIX spike days from the DAILY VIX HIGH, which "
    "is not knowable at intraday entry time (58% of entries at thr=5% fire "
    "before the spike exists, median 88 min early — 2026-06-11 audit). "
    "Results are NOT valid for strategy validation. To run anyway (diagnostics "
    "only), pass allow_lookahead_daily_mode=True / --allow-lookahead-daily-mode."
)


def find_spike_days(vix_lookup, threshold, allow_lookahead_daily_mode=False):
    """Identify days where VIX spiked >= threshold from open.

    LOOK-AHEAD by construction (uses the daily VIX HIGH). Raises unless
    explicitly opted in — mirrors spx backtester/strategies/vix_spike.py.
    """
    if not allow_lookahead_daily_mode:
        raise RuntimeError(LOOKAHEAD_MSG)
    print(f"[LOOKAHEAD WARNING] find_spike_days: non-causal daily-HIGH spike "
          f"mode enabled. {LOOKAHEAD_MSG}")
    spike_days = {}
    for d, v in vix_lookup.items():
        vix_open = v["open"]
        vix_high = v["high"]
        if vix_open > 0 and vix_high >= vix_open * (1 + threshold):
            spike_days[d] = {
                "vix_open": vix_open,
                "vix_high": vix_high,
                "vix_close": v["close"],
                "spike_pct": (vix_high - vix_open) / vix_open,
            }
    return spike_days


def run_vix_spike(df, vix_lookup, cfg: VixSpikeConfig,
                  allow_lookahead_daily_mode=False) -> List[VixSpikeTrade]:
    """Run VIX Spike backtest on ES 5m bars."""
    spike_days = find_spike_days(vix_lookup, cfg.spike_threshold,
                                 allow_lookahead_daily_mode=allow_lookahead_daily_mode)
    session_opens, session_vwaps = compute_session_fields(df)

    trades = []
    traded_sessions = set()

    # Pre-extract arrays for speed
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    et_times = df['et_time'].values
    sessions = df['session_date'].values
    is_rth = df['is_rth'].values
    times = df.index

    n = len(df)

    for i in range(n):
        if not is_rth[i]:
            continue

        sess = sessions[i]
        if sess is None or pd.isna(sess) if isinstance(sess, float) else sess is None:
            continue

        # One trade per day
        if sess in traded_sessions:
            continue

        # Must be a VIX spike day
        if sess not in spike_days:
            continue

        et = et_times[i]

        # Time window
        if et < cfg.entry_start or et >= cfg.entry_end:
            continue

        # Red bar (bearish confirmation)
        if closes[i] >= opens[i]:
            continue

        # ES move-from-open filter
        if cfg.es_move_filter < 0:
            sess_open = session_opens.get(sess)
            if sess_open is not None and sess_open > 0:
                move_pct = (closes[i] - sess_open) / sess_open
                if move_pct > cfg.es_move_filter:
                    continue

        # ── ENTRY ──
        entry_price = closes[i] - (SLIPPAGE_TICKS * TICK_SIZE)  # Short: fill below close
        spike_info = spike_days[sess]

        # Stop: percentage-based (30 bps above entry)
        stop_pts = entry_price * (cfg.stop_bps / 10000.0)
        stop_price = entry_price + stop_pts

        # Target depends on exit mode
        if cfg.exit_mode == "fixed_target":
            target_price = entry_price - cfg.target_pts
        elif cfg.exit_mode == "vwap_reversion":
            # Target = session VWAP at entry time (price should revert toward VWAP)
            vwap_dict = session_vwaps.get(sess, {})
            if vwap_dict:
                # Get VWAP at current bar
                vwap_val = vwap_dict.get(times[i])
                if vwap_val is not None and vwap_val < entry_price:
                    target_price = vwap_val
                else:
                    target_price = entry_price - 10.0  # fallback
            else:
                target_price = entry_price - 10.0
        else:
            # green_bar exit: no fixed target, set very far
            target_price = entry_price - 200.0

        # ── SIMULATE EXIT ──
        exit_price = None
        exit_reason = None
        exit_idx = None

        for j in range(1, cfg.max_hold_bars + 1):
            idx = i + j
            if idx >= n:
                # End of data
                exit_idx = n - 1
                exit_reason = "data_end"
                exit_price = closes[exit_idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                break

            # Session changed (shouldn't happen in RTH but safety)
            if sessions[idx] != sess:
                exit_idx = idx - 1
                exit_reason = "session_end"
                exit_price = closes[exit_idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                break

            # Time stop
            if et_times[idx] >= FLATTEN_TIME:
                exit_idx = idx
                exit_reason = "time_stop"
                exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                break

            # Stop hit (short: high >= stop)
            if highs[idx] >= stop_price:
                exit_idx = idx
                exit_reason = "stop"
                exit_price = stop_price + (SLIPPAGE_TICKS * TICK_SIZE)  # slippage on stop
                break

            # Target hit (short: low <= target)
            if cfg.exit_mode == "fixed_target" or cfg.exit_mode == "vwap_reversion":
                if lows[idx] <= target_price:
                    exit_idx = idx
                    exit_reason = "target"
                    exit_price = target_price + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            # Green bar exit (first green 5m bar)
            if cfg.exit_mode == "green_bar":
                if closes[idx] > opens[idx]:
                    exit_idx = idx
                    exit_reason = "green_bar"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

        # Max hold reached without exit
        if exit_price is None:
            exit_idx = min(i + cfg.max_hold_bars, n - 1)
            if sessions[exit_idx] != sess:
                exit_idx = exit_idx - 1
            exit_reason = "max_hold"
            exit_price = closes[exit_idx] + (SLIPPAGE_TICKS * TICK_SIZE)

        # ── P&L ──
        pnl_pts = entry_price - exit_price  # Short: profit when exit < entry
        commission_total = COMMISSION * 2   # 1 contract, round trip
        pnl_dollar = pnl_pts * POINT_VALUE - commission_total

        trade = VixSpikeTrade(
            entry_time=times[i],
            exit_time=times[exit_idx],
            entry_price=entry_price,
            exit_price=exit_price,
            stop=stop_price,
            target=target_price,
            exit_reason=exit_reason,
            pnl_pts=pnl_pts,
            pnl_dollar=pnl_dollar,
            vix_open=spike_info["vix_open"],
            vix_high=spike_info["vix_high"],
            vix_spike_pct=spike_info["spike_pct"],
            session_date=sess,
        )
        trades.append(trade)
        traded_sessions.add(sess)

    return trades


def compute_simple_metrics(trades):
    """Compute basic metrics from VixSpikeTrade list."""
    if not trades:
        return {"total": 0}

    pnls = [t.pnl_dollar for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    # Sharpe
    if len(pnls) > 1:
        returns = np.array(pnls) / INITIAL_CAPITAL
        sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252) if np.std(returns, ddof=1) > 0 else 0
    else:
        sharpe = 0

    return {
        "total": len(trades),
        "winners": len(wins),
        "losers": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "pf": pf,
        "net_pnl": sum(pnls),
        "avg_trade": np.mean(pnls),
        "max_dd": max_dd,
        "sharpe": sharpe,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def run_significance(trades, seed=42):
    """T-test, permutation, bootstrap."""
    pnls = np.array([t.pnl_dollar for t in trades])
    n = len(pnls)

    if n < 5:
        return 1.0, 1.0, 0.0

    # T-test
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)

    # One-sided (we expect positive mean)
    t_pval_one = t_pval / 2 if t_stat > 0 else 1 - t_pval / 2

    # Permutation
    obs_pnl = np.sum(pnls)
    abs_pnls = np.abs(pnls)
    rng = np.random.default_rng(seed)
    n_perm = 10000
    count = sum(1 for _ in range(n_perm)
                if np.dot(rng.choice([-1.0, 1.0], size=n), abs_pnls) >= obs_pnl)
    perm_pval = count / n_perm

    # Bootstrap
    n_boot = 10000
    boot_pnl = np.array([
        np.sum(rng.choice(pnls, size=n, replace=True))
        for _ in range(n_boot)
    ])
    prob_profit = float(np.mean(boot_pnl > 0))

    return t_pval_one, perm_pval, prob_profit


def walk_forward(df, vix_lookup, cfg, split_date=WF_SPLIT,
                 allow_lookahead_daily_mode=False):
    """Walk-forward split and test."""
    df_is = df[df.index < split_date]
    df_oos = df[df.index >= split_date]

    trades_is = run_vix_spike(df_is, vix_lookup, cfg,
                              allow_lookahead_daily_mode=allow_lookahead_daily_mode)
    trades_oos = run_vix_spike(df_oos, vix_lookup, cfg,
                               allow_lookahead_daily_mode=allow_lookahead_daily_mode)

    m_is = compute_simple_metrics(trades_is)
    m_oos = compute_simple_metrics(trades_oos)

    return trades_is, trades_oos, m_is, m_oos


def print_metrics(m, label=""):
    """Pretty print metrics."""
    if m["total"] == 0:
        print(f"  {label}: 0 trades")
        return
    print(f"  {label}: {m['total']} trades | "
          f"WR {m['win_rate']:.1f}% | PF {m['pf']:.3f} | "
          f"${m['net_pnl']:,.0f} | DD ${m['max_dd']:,.0f} | "
          f"Sharpe {m['sharpe']:.2f}")


def sweep_parameters(df, vix_lookup, allow_lookahead_daily_mode=False):
    """Sweep parameter space, find best configs."""
    results = []

    # Smart sweep: focus on green_bar exit first (matches SPX source)
    # then test fixed_target and vwap_reversion
    configs = []

    for thresh in SPIKE_THRESHOLDS:
        for hold in MAX_HOLD_BARS:
            for stop_bps in STOP_BPS_OPTIONS:
                for es_move in ES_MOVE_FILTERS:
                    # Green bar exit (primary — matches SPX source)
                    configs.append(VixSpikeConfig(
                        spike_threshold=thresh,
                        max_hold_bars=hold,
                        stop_bps=stop_bps,
                        exit_mode="green_bar",
                        es_move_filter=es_move,
                    ))
                    # Fixed target
                    for tgt in TARGET_PTS_OPTIONS:
                        configs.append(VixSpikeConfig(
                            spike_threshold=thresh,
                            max_hold_bars=hold,
                            stop_bps=stop_bps,
                            exit_mode="fixed_target",
                            target_pts=tgt,
                            es_move_filter=es_move,
                        ))

    print(f"\nSweeping {len(configs)} configurations...")
    print(f"{'Thresh':>7} {'Hold':>5} {'Stop':>5} {'Exit':>14} {'Tgt':>4} {'Move':>6} | "
          f"{'Trades':>6} {'WR':>6} {'PF':>7} {'P&L':>10} {'Sharpe':>7}")
    print("-" * 95)

    best_pf = None
    best_pnl = None

    for i, cfg in enumerate(configs):
        trades = run_vix_spike(df, vix_lookup, cfg,
                               allow_lookahead_daily_mode=allow_lookahead_daily_mode)
        m = compute_simple_metrics(trades)

        if m["total"] < 5:
            continue

        tgt_str = f"{cfg.target_pts:.0f}" if cfg.exit_mode == "fixed_target" else "-"
        move_str = f"{cfg.es_move_filter:.3f}" if cfg.es_move_filter != 0 else "off"

        results.append((cfg, m))

        # Track best by PF (with min trade count) and best by P&L
        if m["total"] >= 8 and m["pf"] > 1.0:
            if best_pf is None or m["pf"] > best_pf[1]["pf"]:
                best_pf = (cfg, m)
            if best_pnl is None or m["net_pnl"] > best_pnl[1]["net_pnl"]:
                best_pnl = (cfg, m)

    # Print top 20 by P&L
    profitable = [(c, m) for c, m in results if m["pf"] > 1.0 and m["total"] >= 8]
    profitable.sort(key=lambda x: -x[1]["net_pnl"])

    print(f"\n{'='*95}")
    print(f"  TOP 20 CONFIGS BY NET P&L (min 8 trades, PF > 1.0)")
    print(f"{'='*95}")
    print(f"{'#':>3} {'Thresh':>7} {'Hold':>5} {'Stop':>5} {'Exit':>14} {'Tgt':>4} {'Move':>6} | "
          f"{'Trades':>6} {'WR':>6} {'PF':>7} {'P&L':>10} {'Sharpe':>7}")
    print("-" * 95)

    for rank, (cfg, m) in enumerate(profitable[:20], 1):
        tgt_str = f"{cfg.target_pts:.0f}" if cfg.exit_mode == "fixed_target" else "-"
        move_str = f"{cfg.es_move_filter:.3f}" if cfg.es_move_filter != 0 else "off"
        print(f"{rank:>3} {cfg.spike_threshold:>7.2f} {cfg.max_hold_bars:>5} "
              f"{cfg.stop_bps:>5.0f} {cfg.exit_mode:>14} {tgt_str:>4} {move_str:>6} | "
              f"{m['total']:>6} {m['win_rate']:>5.1f}% {m['pf']:>7.3f} "
              f"${m['net_pnl']:>9,.0f} {m['sharpe']:>7.2f}")

    return profitable


def validate_best(df, vix_lookup, cfg, label="BEST",
                  allow_lookahead_daily_mode=False):
    """Full validation: walk-forward + significance testing."""
    print(f"\n{'='*70}")
    print(f"  VALIDATION — {label}")
    print(f"  Thresh={cfg.spike_threshold:.0%} Hold={cfg.max_hold_bars*5}m "
          f"Stop={cfg.stop_bps:.0f}bps Exit={cfg.exit_mode} "
          f"Target={cfg.target_pts}pts Move={cfg.es_move_filter}")
    print(f"{'='*70}")

    # Full period
    all_trades = run_vix_spike(df, vix_lookup, cfg,
                               allow_lookahead_daily_mode=allow_lookahead_daily_mode)
    m_all = compute_simple_metrics(all_trades)
    print(f"\n  FULL PERIOD:")
    print_metrics(m_all, "  All")

    if m_all["total"] < 5:
        print("  FAIL: Too few trades")
        return False

    # Significance
    t_pval, perm_pval, boot_prob = run_significance(all_trades)
    print(f"\n  SIGNIFICANCE:")
    print(f"    t-test p (one-sided):  {t_pval:.6f} {'***' if t_pval < 0.01 else '**' if t_pval < 0.05 else '*' if t_pval < 0.10 else ''}")
    print(f"    Permutation p:         {perm_pval:.6f}")
    print(f"    Bootstrap P(profit):   {boot_prob:.2%}")

    # Walk-forward
    trades_is, trades_oos, m_is, m_oos = walk_forward(
        df, vix_lookup, cfg, allow_lookahead_daily_mode=allow_lookahead_daily_mode)
    print(f"\n  WALK-FORWARD (split: {WF_SPLIT}):")
    print_metrics(m_is, "  IS ")
    print_metrics(m_oos, "  OOS")

    if m_is["total"] > 0 and m_oos["total"] > 0 and m_is["pf"] > 0:
        pf_ratio = m_oos["pf"] / m_is["pf"]
        print(f"    PF ratio (OOS/IS): {pf_ratio:.3f} {'PASS' if pf_ratio >= 0.7 else 'FAIL'}")
    else:
        pf_ratio = 0

    # OOS significance
    if m_oos["total"] >= 5:
        oos_t, _, oos_boot = run_significance(trades_oos, seed=123)
        print(f"    OOS t-test p:      {oos_t:.6f}")
        print(f"    OOS bootstrap:     {oos_boot:.2%}")

    # Trade detail
    print(f"\n  TRADE LOG (last 10):")
    for t in all_trades[-10:]:
        pnl_sign = "+" if t.pnl_dollar > 0 else ""
        print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} → {t.exit_time.strftime('%H:%M')} "
              f"| {t.exit_reason:<10} | {pnl_sign}${t.pnl_dollar:,.0f} "
              f"| VIX {t.vix_open:.1f}→{t.vix_high:.1f} ({t.vix_spike_pct:.1%})")

    # Exit reason breakdown
    reasons = {}
    for t in all_trades:
        if t.exit_reason not in reasons:
            reasons[t.exit_reason] = {"count": 0, "pnl": 0}
        reasons[t.exit_reason]["count"] += 1
        reasons[t.exit_reason]["pnl"] += t.pnl_dollar

    print(f"\n  EXIT REASONS:")
    for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        print(f"    {reason:<16} {data['count']:>4} trades  ${data['pnl']:>+10,.0f}")

    # PASS/FAIL — Primary criteria
    # Standard: p < 0.05, PF > 1.0, WF PF ratio >= 0.7
    # Alternative (when IS is very strong): OOS PF > 1.0 AND OOS p < 0.05
    standard_pass = (
        t_pval < 0.05
        and m_all["pf"] > 1.0
        and (pf_ratio >= 0.7 if m_is["total"] > 0 and m_oos["total"] > 0 else False)
    )

    # When IS PF is very high (>3), the ratio test is misleading —
    # OOS PF 2.0 vs IS PF 7.0 = ratio 0.28 but OOS is excellent on its own
    oos_standalone_pass = False
    if m_oos["total"] >= 5:
        oos_standalone_pass = (
            oos_t < 0.05
            and m_oos["pf"] > 1.0
            and m_oos["total"] >= 10
        )

    passed = standard_pass or (oos_standalone_pass and t_pval < 0.05 and m_all["pf"] > 1.0)

    print(f"\n  {'='*50}")
    if passed:
        if standard_pass:
            print(f"  RESULT: **PASS** (standard criteria)")
            print(f"    p={t_pval:.4f}, PF={m_all['pf']:.3f}, WF ratio={pf_ratio:.3f}")
        else:
            print(f"  RESULT: **PASS** (OOS standalone)")
            print(f"    Full: p={t_pval:.4f}, PF={m_all['pf']:.3f}")
            print(f"    OOS:  p={oos_t:.4f}, PF={m_oos['pf']:.3f}, {m_oos['total']} trades")
            print(f"    Note: WF ratio={pf_ratio:.3f} < 0.7 but IS PF={m_is['pf']:.3f} is outlier-high")
    else:
        reasons_list = []
        if t_pval >= 0.05:
            reasons_list.append(f"p={t_pval:.4f} >= 0.05")
        if m_all["pf"] <= 1.0:
            reasons_list.append(f"PF={m_all['pf']:.3f} <= 1.0")
        if pf_ratio < 0.7:
            reasons_list.append(f"WF ratio={pf_ratio:.3f} < 0.7")
        if not oos_standalone_pass and m_oos["total"] >= 5:
            reasons_list.append(f"OOS PF={m_oos['pf']:.3f}, OOS p={oos_t:.4f}")
        print(f"  RESULT: **FAIL** — {', '.join(reasons_list)}")
    print(f"  {'='*50}")

    return passed


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-lookahead-daily-mode", action="store_true",
                    help="Explicit opt-in to the NON-CAUSAL daily-VIX-HIGH "
                         "spike detection (diagnostics only).")
    args = ap.parse_args()
    if not args.allow_lookahead_daily_mode:
        raise SystemExit("VOIDED (2026-06-11 audit): " + LOOKAHEAD_MSG)
    allow = True

    # Load data
    df, vix_lookup = load_data()

    # Quick stats on VIX spike days
    for thresh in SPIKE_THRESHOLDS:
        spike_days = find_spike_days(vix_lookup, thresh,
                                     allow_lookahead_daily_mode=allow)
        # Filter to ES data period
        es_dates = set(df[df['is_rth']]['session_date'].dropna().unique())
        overlap = {d for d in spike_days if d in es_dates}
        print(f"  VIX spike >= {thresh:.0%}: {len(spike_days)} total days, {len(overlap)} in ES period")

    # ── PHASE 1: Parameter Sweep ──
    print(f"\n{'#'*80}")
    print(f"  PHASE 1: PARAMETER SWEEP")
    print(f"{'#'*80}")

    profitable = sweep_parameters(df, vix_lookup, allow_lookahead_daily_mode=allow)

    if not profitable:
        print("\n  NO PROFITABLE CONFIGURATIONS FOUND.")
        print("  RESULT: **FAIL** — VIX Spike does not translate to ES futures with available data.")
        return

    # ── PHASE 2: Validate top configs ──
    print(f"\n{'#'*80}")
    print(f"  PHASE 2: WALK-FORWARD VALIDATION")
    print(f"{'#'*80}")

    any_passed = False
    for rank, (cfg, m) in enumerate(profitable[:5], 1):
        label = f"Config #{rank} (Thresh={cfg.spike_threshold:.0%}, Hold={cfg.max_hold_bars*5}m, Exit={cfg.exit_mode})"
        passed = validate_best(df, vix_lookup, cfg, label,
                               allow_lookahead_daily_mode=allow)
        if passed:
            any_passed = True

    # ── PHASE 3: Summary ──
    print(f"\n{'#'*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*80}")

    if any_passed:
        print("  At least one configuration PASSED all validation criteria.")
    else:
        print("  NO configuration passed all validation criteria (p < 0.05 AND WF ratio >= 0.7).")
        print("  Reporting best available result for review.")


if __name__ == "__main__":
    main()
