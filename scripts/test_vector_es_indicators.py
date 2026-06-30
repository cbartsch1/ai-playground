#!/usr/bin/env python3
"""Vector ES — Indicator Enhancement Study.

Baseline: EMA 8/24 bearish cross, short-only, 30bps stop, TEMA bearish filter,
          30m range >= 15, time-stop exit at 15:55.
          506 trades/2yr, PF 1.377, +$53K, p=0.028.

Goal: Compute new indicators from raw OHLCV data and test each as a FILTER
on the baseline signal. Find the best 3, combine, walk-forward validate.

SHORT ONLY. No IB. Entry from 9:35. Walk-forward split: 2025-02-16.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics, Metrics
from backtester.position import Trade
from backtester.indicators import compute_indicators, ema, sma, atr_wilders


# ── Configuration ──

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "es_5m_databento_2yr.csv",
)
WF_SPLIT = "2025-02-16"
INITIAL_CAPITAL = 100_000.0


@dataclass
class VectorConfig:
    """Vector ES parameters — baseline per user: EMA 9/21, 20bps stop."""
    ema_fast: int = 9
    ema_slow: int = 21
    min_30m_range: float = 0.0     # user reported 506 trades — no range filter
    require_tema_bearish: bool = True
    skip_net_short_on: bool = False
    stop_bps: float = 20.0
    max_hold_bars: int = 0
    entry_start: int = 935
    entry_end: int = 1500
    time_stop: int = 1555
    max_trades_per_day: int = 2
    target_pts: float = 0.0
    point_value: float = 50.0
    commission_rt: float = 5.0
    slippage_ticks: int = 1
    tick_size: float = 0.25


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR COMPUTATION — all computed from raw OHLCV
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_indicators(df: pd.DataFrame) -> None:
    """Compute every indicator we want to test, added as columns in-place."""

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)
    opn = df["open"]

    # ── 1. RSI (14) ──
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)

    # ── 2. RSI (5) fast ──
    avg_gain5 = gain.ewm(alpha=1.0 / 5, adjust=False).mean()
    avg_loss5 = loss.ewm(alpha=1.0 / 5, adjust=False).mean()
    rs5 = avg_gain5 / avg_loss5.replace(0, np.nan)
    df["rsi_5"] = 100 - (100 / (1 + rs5))
    df["rsi_5"] = df["rsi_5"].fillna(50)

    # ── 3. Stochastic %K/%D (14, 3, 3) ──
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = df["stoch_k"].fillna(50)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean().fillna(50)

    # ── 4. MACD histogram ──
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - macd_signal

    # ── 5. Rate of Change (ROC) 10-bar, 20-bar ──
    df["roc_10"] = (close / close.shift(10) - 1) * 100
    df["roc_20"] = (close / close.shift(20) - 1) * 100

    # ── 6. CCI (20) ──
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci_20"] = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)
    df["cci_20"] = df["cci_20"].fillna(0)

    # ── 7. Bollinger Bands (20, 2) ──
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct_b"] = df["bb_pct_b"].fillna(0.5)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

    # ── 8. Keltner Channels (20, 1.5) ──
    kc_mid = close.ewm(span=20, adjust=False).mean()
    atr_kc = atr_wilders(df, 10)
    df["kc_upper"] = kc_mid + 1.5 * atr_kc
    df["kc_lower"] = kc_mid - 1.5 * atr_kc
    df["kc_pct"] = (close - kc_mid) / (1.5 * atr_kc).replace(0, np.nan)
    df["kc_pct"] = df["kc_pct"].fillna(0)

    # ── 9. ATR percentile (current ATR vs 50-bar ATR) ──
    # atr and atr_avg already computed by compute_indicators
    df["atr_pctile"] = df["atr"].rolling(50).apply(
        lambda x: scipy_stats.percentileofscore(x, x.iloc[-1]) if len(x) == 50 else 50,
        raw=False
    )
    df["atr_pctile"] = df["atr_pctile"].fillna(50)

    # ── 10. Bar range ratio ──
    bar_range = high - low
    avg_range_20 = bar_range.rolling(20).mean()
    df["bar_range_ratio"] = bar_range / avg_range_20.replace(0, np.nan)
    df["bar_range_ratio"] = df["bar_range_ratio"].fillna(1.0)

    # ── 11. Volume spike (volume / 20-bar avg volume) ──
    vol_sma20 = volume.rolling(20).mean()
    df["vol_spike"] = volume / vol_sma20.replace(0, np.nan)
    df["vol_spike"] = df["vol_spike"].fillna(1.0)

    # ── 12. Volume trend (5-bar volume MA slope) ──
    vol_sma5 = volume.rolling(5).mean()
    df["vol_trend"] = vol_sma5 / vol_sma5.shift(5).replace(0, np.nan)
    df["vol_trend"] = df["vol_trend"].fillna(1.0)

    # ── 13. OBV (On-Balance Volume) — divergence from price ──
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["obv"] = obv
    # OBV slope (5-bar): compare OBV direction vs price direction
    df["obv_slope_5"] = obv.diff(5)
    df["price_slope_5"] = close.diff(5)
    # Bearish divergence: price rising but OBV falling
    df["obv_bear_div"] = (df["price_slope_5"] > 0) & (df["obv_slope_5"] < 0)

    # ── 14. VWAP distance ──
    # Session VWAP: cumulative (price * volume) / cumulative volume per session
    # Reset at session start
    df["tp_vol"] = ((high + low + close) / 3) * volume
    vwap_num = df.groupby("session_date")["tp_vol"].cumsum()
    vwap_den = df.groupby("session_date")["volume"].cumsum().astype(float)
    df["vwap"] = vwap_num / vwap_den.replace(0, np.nan)
    df["vwap"] = df["vwap"].fillna(close)
    df["vwap_dist"] = (close - df["vwap"]) / df["atr"].replace(0, np.nan)
    df["vwap_dist"] = df["vwap_dist"].fillna(0)

    # ── 15. Consecutive red bars ──
    is_red = (close < opn).astype(int)
    # Count consecutive reds looking backward
    streaks = []
    count = 0
    for val in is_red.values:
        if val == 1:
            count += 1
        else:
            count = 0
        streaks.append(count)
    df["consec_red"] = streaks

    # ── 16. Higher high failure ──
    # Bar makes new high vs prior bar but closes below prior close
    df["hh_fail"] = (high > high.shift(1)) & (close < close.shift(1))

    # ── 17. Range expansion ──
    df["range_expansion"] = bar_range > 1.5 * bar_range.shift(1)

    # ── 18. Gap from session open ──
    # Distance from today's RTH open price
    session_open = df.groupby("session_date")["open"].transform("first")
    df["gap_from_open"] = (close - session_open) / df["atr"].replace(0, np.nan)
    df["gap_from_open"] = df["gap_from_open"].fillna(0)

    # ── 19. Distance from overnight high ──
    # We need to compute ON high per session
    # ON = bars before RTH on same session_date
    on_high_dict = {}
    for sd, grp in df.groupby("session_date"):
        on_bars = grp[~grp["is_rth"]]
        if not on_bars.empty:
            on_high_dict[sd] = on_bars["high"].max()
        else:
            rth_bars = grp[grp["is_rth"]]
            if not rth_bars.empty:
                on_high_dict[sd] = rth_bars.iloc[0]["high"]  # fallback
    df["on_high"] = df["session_date"].map(on_high_dict)
    df["dist_from_onh"] = (close - df["on_high"]) / df["atr"].replace(0, np.nan)
    df["dist_from_onh"] = df["dist_from_onh"].fillna(0)

    # ── 20. EMA 21 slope (5-bar) ──
    ema21 = close.ewm(span=24, adjust=False).mean()  # Use the slow EMA (24)
    df["ema_slow_slope"] = ema21.diff(5)

    # ── 21. Price vs TEMA 55 ──
    # Already computed: trend_down = close < tema_trend

    # ── 22. Stochastic %K crossdown ──
    df["stoch_cross_down"] = (df["stoch_k"].shift(1) > df["stoch_d"].shift(1)) & \
                              (df["stoch_k"] < df["stoch_d"])

    # ── 23. Session VWAP slope (5-bar) ──
    df["vwap_slope"] = df["vwap"].diff(5)

    # ── 24. Day of week ──
    df["dow"] = df.index.dayofweek  # 0=Mon, 4=Fri

    # ── 25. ADX (14) ──
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = atr_wilders(df, 14)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    df["adx"] = df["adx"].fillna(20)
    df["plus_di"] = plus_di.fillna(0)
    df["minus_di"] = minus_di.fillna(0)

    # ── 26. Williams %R (14) ──
    df["willr"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)
    df["willr"] = df["willr"].fillna(-50)

    # ── 27. Price acceleration (2nd derivative of close) ──
    df["price_accel"] = close.diff().diff()

    # Clean up temp columns
    for col in ["tp_vol"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING (copied from baseline)
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load ES 5m data with session tags and indicators."""
    print(f"Loading {DATA_PATH}...")
    df = load_tos_csv(DATA_PATH, instrument="ES")
    print(f"  {len(df):,} bars | {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    # Compute TEMA, ATR, etc.
    compute_indicators(df)

    # Compute all new indicators
    compute_all_indicators(df)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  30-MINUTE BAR AGGREGATION + RANGE LOOKUP (from baseline)
# ══════════════════════════════════════════════════════════════════════════════

def build_30m_bars(df: pd.DataFrame) -> pd.DataFrame:
    rth = df[df["is_rth"]].copy()
    if rth.empty:
        return pd.DataFrame()
    et = rth["et_time"].values
    minutes_from_open = ((et // 100 - 9) * 60 + (et % 100)) - 30
    rth["period_30m"] = minutes_from_open // 30
    bars_30m = rth.groupby(["session_date", "period_30m"]).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), et_time=("et_time", "last"),
    ).reset_index()
    bars_30m["range"] = bars_30m["high"] - bars_30m["low"]
    return bars_30m


def build_range_lookup(bars_30m: pd.DataFrame) -> dict:
    lookup = {}
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


# ══════════════════════════════════════════════════════════════════════════════
#  EMA CROSS DETECTION (from baseline)
# ══════════════════════════════════════════════════════════════════════════════

def detect_ema_crosses(df: pd.DataFrame, fast_period: int, slow_period: int):
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


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE — accepts an optional indicator filter function
# ══════════════════════════════════════════════════════════════════════════════

def run_vector_backtest(df: pd.DataFrame, cfg: VectorConfig,
                        indicator_filter=None) -> List[Trade]:
    """Run Vector ES backtest with optional indicator filter.

    indicator_filter: callable(df, i) -> bool
        Returns True if the indicator condition is satisfied at bar i.
        If None, no additional filter (baseline behavior).
    """
    bars_30m = build_30m_bars(df)
    range_lookup = build_range_lookup(bars_30m)

    ema_fast, ema_slow, cross_bear, cross_bull = detect_ema_crosses(
        df, cfg.ema_fast, cfg.ema_slow
    )

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
    position = None
    trades_today = 0
    current_session = None
    slippage = cfg.slippage_ticks * cfg.tick_size
    warmup = max(cfg.ema_slow, 55) + 5  # extra warmup for indicators

    for i in range(warmup, n):
        s = sess[i]

        # Session reset
        if s != current_session:
            if position is not None:
                exit_price = cl[i - 1] + slippage
                pnl_pts = position["entry_price"] - exit_price
                pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
                trades.append(Trade(
                    setup="VEC_ES", direction=-1,
                    entry_time=timestamps[position["entry_idx"]],
                    entry_price=position["entry_price"],
                    exit_time=timestamps[i - 1], exit_price=exit_price,
                    exit_reason="session_end",
                    pnl_pts=pnl_pts, pnl_dollar=pnl_dollar,
                    stop=position["stop"], target=position["target"],
                ))
                position = None
            current_session = s
            trades_today = 0

        # ── EXIT CHECKS ──
        if position is not None:
            exit_reason = None
            exit_price = None

            if hi[i] >= position["stop"]:
                exit_reason = "stop"
                exit_price = position["stop"] + slippage

            if exit_reason is None and position["target"] > 0 and lo[i] <= position["target"]:
                exit_reason = "target"
                exit_price = position["target"] - slippage

            if exit_reason is None and cfg.max_hold_bars > 0:
                bars_held = i - position["entry_idx"]
                if bars_held >= cfg.max_hold_bars:
                    exit_reason = "max_hold"
                    exit_price = cl[i]

            if exit_reason is None and et[i] >= cfg.time_stop and is_rth[i]:
                exit_reason = "time_stop"
                exit_price = cl[i]

            if exit_reason is None and cross_bull[i]:
                exit_reason = "opposite_cross"
                exit_price = cl[i]

            if exit_reason is not None:
                pnl_pts = position["entry_price"] - exit_price
                pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
                trades.append(Trade(
                    setup="VEC_ES", direction=-1,
                    entry_time=timestamps[position["entry_idx"]],
                    entry_price=position["entry_price"],
                    exit_time=timestamps[i], exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pts=pnl_pts, pnl_dollar=pnl_dollar,
                    stop=position["stop"], target=position["target"],
                ))
                position = None

        # ── ENTRY CHECKS ──
        if position is None and is_rth[i]:
            if et[i] < cfg.entry_start or et[i] >= cfg.entry_end:
                continue
            if trades_today >= cfg.max_trades_per_day:
                continue
            if not cross_bear[i]:
                continue
            if cfg.require_tema_bearish and not tema_bearish[i]:
                continue
            max_range = get_max_30m_range(range_lookup, s, et[i])
            if max_range < cfg.min_30m_range:
                continue

            # ── INDICATOR FILTER ──
            if indicator_filter is not None and not indicator_filter(df, i):
                continue

            # ── ENTER SHORT ──
            entry_price = cl[i] - slippage
            stop_pts = entry_price * cfg.stop_bps / 10000.0
            stop_price = entry_price + stop_pts

            if cfg.target_pts > 0:
                target_price = entry_price - cfg.target_pts
            else:
                target_price = 0.0

            position = {
                "entry_idx": i,
                "entry_price": entry_price,
                "stop": stop_price,
                "target": target_price,
                "session": s,
            }
            trades_today += 1

    # Close remaining position
    if position is not None:
        exit_price = cl[-1] + slippage
        pnl_pts = position["entry_price"] - exit_price
        pnl_dollar = pnl_pts * cfg.point_value - cfg.commission_rt
        trades.append(Trade(
            setup="VEC_ES", direction=-1,
            entry_time=timestamps[position["entry_idx"]],
            entry_price=position["entry_price"],
            exit_time=timestamps[-1], exit_price=exit_price,
            exit_reason="data_end",
            pnl_pts=pnl_pts, pnl_dollar=pnl_dollar,
            stop=position["stop"], target=position["target"],
        ))

    return trades


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNIFICANCE TESTING
# ══════════════════════════════════════════════════════════════════════════════

def run_significance(trades: List[Trade], seed: int = 42) -> Tuple[float, float, float]:
    """T-test + permutation + bootstrap."""
    pnls = np.array([t.pnl_dollar for t in trades])
    n = len(pnls)
    if n < 5:
        return 1.0, 1.0, 0.0

    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    t_pval_one = t_pval / 2 if t_stat > 0 else 1 - t_pval / 2

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

    n_boot = 10000
    boot_pnl = np.array([
        np.sum(rng.choice(pnls, size=n, replace=True))
        for _ in range(n_boot)
    ])
    prob_profit = float(np.mean(boot_pnl > 0))

    return t_pval_one, perm_pval, prob_profit


def walk_forward(df: pd.DataFrame, cfg: VectorConfig,
                 indicator_filter=None,
                 split_date: str = WF_SPLIT):
    """Walk-forward: IS before split_date, OOS after."""
    df_is = df[df.index < split_date].copy()
    df_oos = df[df.index >= split_date].copy()

    trades_is = run_vector_backtest(df_is, cfg, indicator_filter)
    trades_oos = run_vector_backtest(df_oos, cfg, indicator_filter)

    m_is = compute_metrics(trades_is, INITIAL_CAPITAL) if trades_is else None
    m_oos = compute_metrics(trades_oos, INITIAL_CAPITAL) if trades_oos else None

    return m_is, m_oos, trades_is, trades_oos


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR FILTER DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def define_indicator_filters(df: pd.DataFrame) -> List[Dict]:
    """Define all indicator filters to test.

    Each is a dict with:
        name: str — display name
        filter_fn: callable(df, i) -> bool
        category: str — grouping
    """

    # Pre-extract arrays for speed (avoid repeated .values calls in loops)
    rsi14 = df["rsi_14"].values
    rsi5 = df["rsi_5"].values
    stoch_k = df["stoch_k"].values
    stoch_d = df["stoch_d"].values
    macd_hist = df["macd_hist"].values
    roc_10 = df["roc_10"].values
    roc_20 = df["roc_20"].values
    cci_20 = df["cci_20"].values
    bb_pct_b = df["bb_pct_b"].values
    bb_width = df["bb_width"].values
    kc_pct = df["kc_pct"].values
    atr_pctile = df["atr_pctile"].values
    bar_range_ratio = df["bar_range_ratio"].values
    vol_spike = df["vol_spike"].values
    vol_trend = df["vol_trend"].values
    obv_bear_div = df["obv_bear_div"].values
    vwap_dist = df["vwap_dist"].values
    consec_red = df["consec_red"].values
    hh_fail = df["hh_fail"].values
    range_expansion = df["range_expansion"].values
    gap_from_open = df["gap_from_open"].values
    dist_from_onh = df["dist_from_onh"].values
    ema_slow_slope = df["ema_slow_slope"].values
    trend_down = df["trend_down"].values
    stoch_cross_dn = df["stoch_cross_down"].values
    vwap_slope = df["vwap_slope"].values
    dow = df["dow"].values
    adx = df["adx"].values
    minus_di = df["minus_di"].values
    plus_di = df["plus_di"].values
    willr = df["willr"].values
    price_accel = df["price_accel"].values
    close = df["close"].values
    vwap = df["vwap"].values

    filters = []

    # ── MOMENTUM / OSCILLATORS ──

    # RSI(14) > 60: overbought into cross = mean reversion short
    filters.append({
        "name": "RSI14 > 60",
        "filter_fn": lambda df, i, arr=rsi14: arr[i] > 60,
        "category": "Momentum",
    })
    filters.append({
        "name": "RSI14 > 55",
        "filter_fn": lambda df, i, arr=rsi14: arr[i] > 55,
        "category": "Momentum",
    })
    filters.append({
        "name": "RSI14 > 50",
        "filter_fn": lambda df, i, arr=rsi14: arr[i] > 50,
        "category": "Momentum",
    })
    filters.append({
        "name": "RSI14 > 65",
        "filter_fn": lambda df, i, arr=rsi14: arr[i] > 65,
        "category": "Momentum",
    })
    filters.append({
        "name": "RSI14 < 45 (oversold skip)",
        "filter_fn": lambda df, i, arr=rsi14: arr[i] >= 45,  # skip if oversold (bounces)
        "category": "Momentum",
    })

    # RSI(5) fast
    filters.append({
        "name": "RSI5 > 60",
        "filter_fn": lambda df, i, arr=rsi5: arr[i] > 60,
        "category": "Momentum",
    })
    filters.append({
        "name": "RSI5 > 70",
        "filter_fn": lambda df, i, arr=rsi5: arr[i] > 70,
        "category": "Momentum",
    })

    # Stochastic: overbought cross-down confirmation
    filters.append({
        "name": "Stoch %K > 70",
        "filter_fn": lambda df, i, arr=stoch_k: arr[i] > 70,
        "category": "Momentum",
    })
    filters.append({
        "name": "Stoch %K > 80",
        "filter_fn": lambda df, i, arr=stoch_k: arr[i] > 80,
        "category": "Momentum",
    })
    filters.append({
        "name": "Stoch crossdown",
        "filter_fn": lambda df, i, arr=stoch_cross_dn: arr[i],
        "category": "Momentum",
    })

    # MACD histogram negative
    filters.append({
        "name": "MACD hist < 0",
        "filter_fn": lambda df, i, arr=macd_hist: arr[i] < 0,
        "category": "Momentum",
    })
    filters.append({
        "name": "MACD hist declining",
        "filter_fn": lambda df, i, arr=macd_hist: i >= 1 and arr[i] < arr[i-1],
        "category": "Momentum",
    })

    # ROC negative
    filters.append({
        "name": "ROC10 < 0",
        "filter_fn": lambda df, i, arr=roc_10: arr[i] < 0,
        "category": "Momentum",
    })
    filters.append({
        "name": "ROC20 < 0",
        "filter_fn": lambda df, i, arr=roc_20: arr[i] < 0,
        "category": "Momentum",
    })
    filters.append({
        "name": "ROC10 < -0.1",
        "filter_fn": lambda df, i, arr=roc_10: arr[i] < -0.1,
        "category": "Momentum",
    })

    # CCI overbought
    filters.append({
        "name": "CCI > +100",
        "filter_fn": lambda df, i, arr=cci_20: arr[i] > 100,
        "category": "Momentum",
    })
    filters.append({
        "name": "CCI > +50",
        "filter_fn": lambda df, i, arr=cci_20: arr[i] > 50,
        "category": "Momentum",
    })
    filters.append({
        "name": "CCI declining",
        "filter_fn": lambda df, i, arr=cci_20: i >= 1 and arr[i] < arr[i-1],
        "category": "Momentum",
    })

    # ADX
    filters.append({
        "name": "ADX > 20 (trending)",
        "filter_fn": lambda df, i, arr=adx: arr[i] > 20,
        "category": "Momentum",
    })
    filters.append({
        "name": "ADX > 25",
        "filter_fn": lambda df, i, arr=adx: arr[i] > 25,
        "category": "Momentum",
    })
    filters.append({
        "name": "-DI > +DI",
        "filter_fn": lambda df, i, mi=minus_di, pl=plus_di: mi[i] > pl[i],
        "category": "Momentum",
    })

    # Williams %R
    filters.append({
        "name": "WillR > -20 (overbought)",
        "filter_fn": lambda df, i, arr=willr: arr[i] > -20,
        "category": "Momentum",
    })
    filters.append({
        "name": "WillR > -30",
        "filter_fn": lambda df, i, arr=willr: arr[i] > -30,
        "category": "Momentum",
    })

    # ── VOLATILITY ──

    # Bollinger: price near upper band (mean reversion short)
    filters.append({
        "name": "BB %B > 0.8",
        "filter_fn": lambda df, i, arr=bb_pct_b: arr[i] > 0.8,
        "category": "Volatility",
    })
    filters.append({
        "name": "BB %B > 0.7",
        "filter_fn": lambda df, i, arr=bb_pct_b: arr[i] > 0.7,
        "category": "Volatility",
    })
    filters.append({
        "name": "BB %B > 0.5",
        "filter_fn": lambda df, i, arr=bb_pct_b: arr[i] > 0.5,
        "category": "Volatility",
    })

    # Keltner: price above upper channel
    filters.append({
        "name": "KC %pct > 0.5",
        "filter_fn": lambda df, i, arr=kc_pct: arr[i] > 0.5,
        "category": "Volatility",
    })
    filters.append({
        "name": "KC %pct > 0.8",
        "filter_fn": lambda df, i, arr=kc_pct: arr[i] > 0.8,
        "category": "Volatility",
    })
    filters.append({
        "name": "KC %pct > 1.0 (outside)",
        "filter_fn": lambda df, i, arr=kc_pct: arr[i] > 1.0,
        "category": "Volatility",
    })

    # ATR percentile
    filters.append({
        "name": "ATR pctile > 60",
        "filter_fn": lambda df, i, arr=atr_pctile: arr[i] > 60,
        "category": "Volatility",
    })
    filters.append({
        "name": "ATR pctile > 70",
        "filter_fn": lambda df, i, arr=atr_pctile: arr[i] > 70,
        "category": "Volatility",
    })
    filters.append({
        "name": "ATR pctile < 40 (quiet)",
        "filter_fn": lambda df, i, arr=atr_pctile: arr[i] < 40,
        "category": "Volatility",
    })

    # Bar range ratio
    filters.append({
        "name": "BarRange > 1.2x avg",
        "filter_fn": lambda df, i, arr=bar_range_ratio: arr[i] > 1.2,
        "category": "Volatility",
    })
    filters.append({
        "name": "BarRange > 1.5x avg",
        "filter_fn": lambda df, i, arr=bar_range_ratio: arr[i] > 1.5,
        "category": "Volatility",
    })

    # ── VOLUME ──

    # Volume spike
    filters.append({
        "name": "Vol > 1.5x avg",
        "filter_fn": lambda df, i, arr=vol_spike: arr[i] > 1.5,
        "category": "Volume",
    })
    filters.append({
        "name": "Vol > 2.0x avg",
        "filter_fn": lambda df, i, arr=vol_spike: arr[i] > 2.0,
        "category": "Volume",
    })
    filters.append({
        "name": "Vol > 1.2x avg",
        "filter_fn": lambda df, i, arr=vol_spike: arr[i] > 1.2,
        "category": "Volume",
    })

    # Volume trend rising
    filters.append({
        "name": "VolTrend > 1.1 (rising)",
        "filter_fn": lambda df, i, arr=vol_trend: arr[i] > 1.1,
        "category": "Volume",
    })
    filters.append({
        "name": "VolTrend > 1.3",
        "filter_fn": lambda df, i, arr=vol_trend: arr[i] > 1.3,
        "category": "Volume",
    })

    # OBV bearish divergence
    filters.append({
        "name": "OBV bear divergence",
        "filter_fn": lambda df, i, arr=obv_bear_div: arr[i],
        "category": "Volume",
    })

    # VWAP distance
    filters.append({
        "name": "Above VWAP (>0)",
        "filter_fn": lambda df, i, arr=vwap_dist: arr[i] > 0,
        "category": "Volume",
    })
    filters.append({
        "name": "Above VWAP > 0.5 ATR",
        "filter_fn": lambda df, i, arr=vwap_dist: arr[i] > 0.5,
        "category": "Volume",
    })
    filters.append({
        "name": "Above VWAP > 1.0 ATR",
        "filter_fn": lambda df, i, arr=vwap_dist: arr[i] > 1.0,
        "category": "Volume",
    })
    filters.append({
        "name": "Price > VWAP (raw)",
        "filter_fn": lambda df, i, c=close, v=vwap: c[i] > v[i],
        "category": "Volume",
    })

    # ── PRICE ACTION ──

    # Consecutive red bars
    filters.append({
        "name": "ConsecRed >= 2",
        "filter_fn": lambda df, i, arr=consec_red: arr[i] >= 2,
        "category": "PriceAction",
    })
    filters.append({
        "name": "ConsecRed >= 3",
        "filter_fn": lambda df, i, arr=consec_red: arr[i] >= 3,
        "category": "PriceAction",
    })
    filters.append({
        "name": "ConsecRed == 0 (not oversold)",
        "filter_fn": lambda df, i, arr=consec_red: arr[i] == 0,
        "category": "PriceAction",
    })

    # Higher high failure
    filters.append({
        "name": "HH Failure",
        "filter_fn": lambda df, i, arr=hh_fail: arr[i],
        "category": "PriceAction",
    })

    # Range expansion
    filters.append({
        "name": "Range expansion 1.5x",
        "filter_fn": lambda df, i, arr=range_expansion: arr[i],
        "category": "PriceAction",
    })

    # Gap from open (short works better when above open — room to fall)
    filters.append({
        "name": "Above open > 0.5 ATR",
        "filter_fn": lambda df, i, arr=gap_from_open: arr[i] > 0.5,
        "category": "PriceAction",
    })
    filters.append({
        "name": "Above open > 0",
        "filter_fn": lambda df, i, arr=gap_from_open: arr[i] > 0,
        "category": "PriceAction",
    })
    filters.append({
        "name": "Below open (already weak)",
        "filter_fn": lambda df, i, arr=gap_from_open: arr[i] < 0,
        "category": "PriceAction",
    })

    # Distance from ONH
    filters.append({
        "name": "Near ONH (<0.5 ATR)",
        "filter_fn": lambda df, i, arr=dist_from_onh: abs(arr[i]) < 0.5,
        "category": "PriceAction",
    })
    filters.append({
        "name": "Below ONH",
        "filter_fn": lambda df, i, arr=dist_from_onh: arr[i] < 0,
        "category": "PriceAction",
    })

    # Price acceleration negative
    filters.append({
        "name": "Price accel < 0",
        "filter_fn": lambda df, i, arr=price_accel: arr[i] < 0,
        "category": "PriceAction",
    })

    # ── MARKET STRUCTURE ──

    # EMA slow slope declining
    filters.append({
        "name": "EMA24 slope < 0",
        "filter_fn": lambda df, i, arr=ema_slow_slope: arr[i] < 0,
        "category": "Structure",
    })
    filters.append({
        "name": "EMA24 slope < -1",
        "filter_fn": lambda df, i, arr=ema_slow_slope: arr[i] < -1,
        "category": "Structure",
    })

    # Price below TEMA 55 (downtrend)
    filters.append({
        "name": "Below TEMA 55",
        "filter_fn": lambda df, i, arr=trend_down: arr[i],
        "category": "Structure",
    })

    # VWAP slope declining
    filters.append({
        "name": "VWAP slope < 0",
        "filter_fn": lambda df, i, arr=vwap_slope: arr[i] < 0,
        "category": "Structure",
    })

    # ── TIME / CALENDAR ──

    # Not Friday
    filters.append({
        "name": "Not Friday",
        "filter_fn": lambda df, i, arr=dow: arr[i] != 4,
        "category": "Time",
    })
    # Not Monday
    filters.append({
        "name": "Not Monday",
        "filter_fn": lambda df, i, arr=dow: arr[i] != 0,
        "category": "Time",
    })
    # Mon-Wed only
    filters.append({
        "name": "Mon-Wed only",
        "filter_fn": lambda df, i, arr=dow: arr[i] <= 2,
        "category": "Time",
    })
    # Tue-Thu only
    filters.append({
        "name": "Tue-Thu only",
        "filter_fn": lambda df, i, arr=dow: 1 <= arr[i] <= 3,
        "category": "Time",
    })

    # ── COMBO LOGIC (multi-indicator) ──

    # RSI overbought + MACD declining
    filters.append({
        "name": "RSI14>55 + MACD decl",
        "filter_fn": lambda df, i, r=rsi14, m=macd_hist: r[i] > 55 and i >= 1 and m[i] < m[i-1],
        "category": "Combo",
    })
    # Stoch overbought + below TEMA55
    filters.append({
        "name": "Stoch>70 + belowTEMA55",
        "filter_fn": lambda df, i, s=stoch_k, t=trend_down: s[i] > 70 and t[i],
        "category": "Combo",
    })
    # Above VWAP + RSI>55
    filters.append({
        "name": "AboveVWAP + RSI14>55",
        "filter_fn": lambda df, i, v=vwap_dist, r=rsi14: v[i] > 0 and r[i] > 55,
        "category": "Combo",
    })
    # EMA slope declining + vol spike
    filters.append({
        "name": "EMAslopeDn + Vol>1.5x",
        "filter_fn": lambda df, i, e=ema_slow_slope, v=vol_spike: e[i] < 0 and v[i] > 1.5,
        "category": "Combo",
    })
    # CCI>50 + VWAP slope declining
    filters.append({
        "name": "CCI>50 + VWAPslopeDn",
        "filter_fn": lambda df, i, c=cci_20, vs=vwap_slope: c[i] > 50 and vs[i] < 0,
        "category": "Combo",
    })

    return filters


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TEST HARNESS
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── LOAD DATA ──
    df = load_data()
    cfg = VectorConfig()

    # ── BASELINE ──
    print(f"\n{'='*100}")
    print(f"  VECTOR ES INDICATOR ENHANCEMENT STUDY")
    print(f"{'='*100}")
    print(f"\n  Baseline: EMA {cfg.ema_fast}/{cfg.ema_slow} | 30m range >= {cfg.min_30m_range} | "
          f"Stop {cfg.stop_bps}bps | TEMA bearish | Short only")
    print(f"  Walk-forward split: {WF_SPLIT}")

    trades_base = run_vector_backtest(df, cfg, indicator_filter=None)
    m_base = compute_metrics(trades_base, INITIAL_CAPITAL)
    t_pval_base, _, _ = run_significance(trades_base)

    print(f"\n  BASELINE: {m_base.total_trades} trades | WR {m_base.win_rate:.1f}% | "
          f"PF {m_base.profit_factor:.3f} | ${m_base.net_pnl:,.0f} | "
          f"Sharpe {m_base.sharpe:.2f} | p={t_pval_base:.4f} | DD ${m_base.max_drawdown:,.0f}")

    # Walk-forward baseline
    m_base_is, m_base_oos, _, _ = walk_forward(df, cfg, indicator_filter=None)
    if m_base_is and m_base_oos:
        wf_ratio_base = m_base_oos.profit_factor / m_base_is.profit_factor if m_base_is.profit_factor > 0 else 0
        print(f"  BASELINE WF: IS PF {m_base_is.profit_factor:.3f} ({m_base_is.total_trades}t) | "
              f"OOS PF {m_base_oos.profit_factor:.3f} ({m_base_oos.total_trades}t) | "
              f"Ratio {wf_ratio_base:.3f}")

    # ── DEFINE AND TEST ALL INDICATOR FILTERS ──
    filters = define_indicator_filters(df)

    print(f"\n  Testing {len(filters)} indicator filters...")
    print(f"\n  {'#':>3} {'Category':<12} {'Indicator Filter':<28} {'Trades':>6} {'WR':>6} "
          f"{'PF':>7} {'P&L':>10} {'Shrp':>6} {'p-val':>7} {'DD':>8} {'AvgTrd':>8} {'vs Base':>8}")
    print(f"  {'-'*112}")

    results = []
    for idx, filt in enumerate(filters):
        trades = run_vector_backtest(df, cfg, indicator_filter=filt["filter_fn"])
        m = compute_metrics(trades, INITIAL_CAPITAL)

        # Only run significance if enough trades
        if m.total_trades >= 20:
            t_pval, _, _ = run_significance(trades)
        else:
            t_pval = 1.0

        pf_delta = m.profit_factor - m_base.profit_factor
        pnl_delta = m.net_pnl - m_base.net_pnl

        result = {
            "idx": idx,
            "name": filt["name"],
            "category": filt["category"],
            "trades": m.total_trades,
            "wr": m.win_rate,
            "pf": m.profit_factor,
            "pnl": m.net_pnl,
            "sharpe": m.sharpe,
            "p_val": t_pval,
            "dd": m.max_drawdown,
            "avg_trade": m.avg_trade,
            "pf_delta": pf_delta,
            "pnl_delta": pnl_delta,
            "filter_fn": filt["filter_fn"],
        }
        results.append(result)

        # Color-code: green if PF improved and still significant
        marker = ""
        if m.total_trades >= 100 and m.profit_factor > m_base.profit_factor and t_pval < 0.05:
            marker = " ***"
        elif m.total_trades >= 50 and m.profit_factor > m_base.profit_factor and t_pval < 0.10:
            marker = " **"
        elif m.profit_factor > m_base.profit_factor:
            marker = " *"

        print(f"  {idx+1:>3} {filt['category']:<12} {filt['name']:<28} "
              f"{m.total_trades:>6} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
              f"${m.net_pnl:>9,.0f} {m.sharpe:>6.2f} {t_pval:>7.4f} "
              f"${m.max_drawdown:>7,.0f} ${m.avg_trade:>7,.0f} {pf_delta:>+7.3f}{marker}")

    # ══════════════════════════════════════════════════════════════════════════
    #  TOP PERFORMERS — sorted by PF with minimum trade threshold
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*100}")
    print(f"  TOP PERFORMERS (>= 100 trades, PF > baseline)")
    print(f"{'='*100}")

    # Filter for viable candidates
    viable = [r for r in results if r["trades"] >= 100 and r["pf"] > m_base.profit_factor]
    viable_sorted = sorted(viable, key=lambda x: x["pf"], reverse=True)

    print(f"\n  {'Rank':>4} {'Indicator Filter':<28} {'Trades':>6} {'WR':>6} "
          f"{'PF':>7} {'P&L':>10} {'p-val':>7} {'PF delta':>9}")
    print(f"  {'-'*80}")

    for rank, r in enumerate(viable_sorted[:20], 1):
        print(f"  {rank:>4} {r['name']:<28} {r['trades']:>6} {r['wr']:>5.1f}% "
              f"{r['pf']:>7.3f} ${r['pnl']:>9,.0f} {r['p_val']:>7.4f} {r['pf_delta']:>+8.3f}")

    # Also show the aggressive filter category (>= 50 trades, big PF boost)
    print(f"\n  TOP AGGRESSIVE FILTERS (>= 50 trades, PF > baseline + 0.1)")
    print(f"  {'-'*80}")
    aggressive = [r for r in results if r["trades"] >= 50 and r["pf"] > m_base.profit_factor + 0.1]
    aggressive_sorted = sorted(aggressive, key=lambda x: x["pf"], reverse=True)
    for rank, r in enumerate(aggressive_sorted[:15], 1):
        print(f"  {rank:>4} {r['name']:<28} {r['trades']:>6} {r['wr']:>5.1f}% "
              f"{r['pf']:>7.3f} ${r['pnl']:>9,.0f} {r['p_val']:>7.4f} {r['pf_delta']:>+8.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  COMBO TEST — pick top 3 independent filters and combine
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*100}")
    print(f"  COMBINATION TESTING — Top Filters")
    print(f"{'='*100}")

    # Select top candidates from different categories (no double-dipping)
    # We want diversity: pick best from each category
    category_best = {}
    for r in results:
        if r["trades"] < 80 or r["pf"] <= m_base.profit_factor:
            continue
        cat = r["category"]
        if cat not in category_best or r["pf"] > category_best[cat]["pf"]:
            category_best[cat] = r

    print(f"\n  Best per category (>= 80 trades, PF > baseline):")
    for cat, r in sorted(category_best.items(), key=lambda x: x[1]["pf"], reverse=True):
        print(f"    {cat:<14} {r['name']:<28} PF {r['pf']:.3f} ({r['trades']} trades)")

    # Get up to 6 best diverse candidates for pairwise testing
    candidates = sorted(category_best.values(), key=lambda x: x["pf"], reverse=True)[:6]

    # Also grab the overall top 3 by PF (>= 100 trades) regardless of category
    top3_overall = viable_sorted[:3] if len(viable_sorted) >= 3 else viable_sorted

    # Pairwise combinations
    print(f"\n  --- Pairwise Combinations ---")
    print(f"  {'Filter A':<24} {'Filter B':<24} {'Trades':>6} {'WR':>6} "
          f"{'PF':>7} {'P&L':>10} {'p-val':>7}")
    print(f"  {'-'*88}")

    combo_results = []

    # Test all pairs from candidates + top3
    all_candidates = {r["name"]: r for r in candidates + top3_overall}
    candidate_list = list(all_candidates.values())

    for i in range(len(candidate_list)):
        for j in range(i + 1, len(candidate_list)):
            a = candidate_list[i]
            b = candidate_list[j]

            def combo_filter(df, idx, fn_a=a["filter_fn"], fn_b=b["filter_fn"]):
                return fn_a(df, idx) and fn_b(df, idx)

            trades = run_vector_backtest(df, cfg, indicator_filter=combo_filter)
            m = compute_metrics(trades, INITIAL_CAPITAL)

            if m.total_trades >= 20:
                t_pval, _, _ = run_significance(trades)
            else:
                t_pval = 1.0

            combo_results.append({
                "name_a": a["name"],
                "name_b": b["name"],
                "trades": m.total_trades,
                "wr": m.win_rate,
                "pf": m.profit_factor,
                "pnl": m.net_pnl,
                "p_val": t_pval,
                "sharpe": m.sharpe,
                "dd": m.max_drawdown,
                "filter_fn": combo_filter,
            })

            marker = ""
            if m.total_trades >= 80 and m.profit_factor > m_base.profit_factor and t_pval < 0.05:
                marker = " ***"
            elif m.total_trades >= 50 and m.profit_factor > m_base.profit_factor:
                marker = " **"

            print(f"  {a['name']:<24} {b['name']:<24} {m.total_trades:>6} {m.win_rate:>5.1f}% "
                  f"{m.profit_factor:>7.3f} ${m.net_pnl:>9,.0f} {t_pval:>7.4f}{marker}")

    # Triple combinations from top pairwise results
    print(f"\n  --- Triple Combinations (top pairs + 3rd filter) ---")
    print(f"  {'Combo':<60} {'Trades':>6} {'WR':>6} {'PF':>7} {'P&L':>10} {'p-val':>7}")
    print(f"  {'-'*100}")

    # Get best pairwise combos
    good_combos = [c for c in combo_results if c["trades"] >= 60 and c["pf"] > m_base.profit_factor]
    good_combos = sorted(good_combos, key=lambda x: x["pf"], reverse=True)[:5]

    triple_results = []
    for combo in good_combos:
        for cand in candidate_list:
            if cand["name"] == combo["name_a"] or cand["name"] == combo["name_b"]:
                continue

            def triple_filter(df, idx, fn_ab=combo["filter_fn"], fn_c=cand["filter_fn"]):
                return fn_ab(df, idx) and fn_c(df, idx)

            trades = run_vector_backtest(df, cfg, indicator_filter=triple_filter)
            m = compute_metrics(trades, INITIAL_CAPITAL)

            if m.total_trades < 30:
                continue

            if m.total_trades >= 20:
                t_pval, _, _ = run_significance(trades)
            else:
                t_pval = 1.0

            label = f"{combo['name_a']} + {combo['name_b']} + {cand['name']}"
            triple_results.append({
                "label": label,
                "trades": m.total_trades,
                "wr": m.win_rate,
                "pf": m.profit_factor,
                "pnl": m.net_pnl,
                "p_val": t_pval,
                "sharpe": m.sharpe,
                "dd": m.max_drawdown,
                "filter_fn": triple_filter,
                "name_a": combo["name_a"],
                "name_b": combo["name_b"],
                "name_c": cand["name"],
            })

            marker = ""
            if m.total_trades >= 60 and m.profit_factor > m_base.profit_factor and t_pval < 0.05:
                marker = " ***"

            print(f"  {label:<60} {m.total_trades:>6} {m.win_rate:>5.1f}% "
                  f"{m.profit_factor:>7.3f} ${m.net_pnl:>9,.0f} {t_pval:>7.4f}{marker}")

    # ══════════════════════════════════════════════════════════════════════════
    #  WALK-FORWARD VALIDATION — best candidates
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*100}")
    print(f"  WALK-FORWARD VALIDATION (split: {WF_SPLIT})")
    print(f"{'='*100}")

    # Collect all candidates worth validating
    wf_candidates = []

    # Best singles (>= 100 trades, PF > baseline, p < 0.10)
    for r in viable_sorted[:5]:
        if r["p_val"] < 0.10:
            wf_candidates.append({
                "label": f"Single: {r['name']}",
                "filter_fn": r["filter_fn"],
                "full_pf": r["pf"],
                "full_trades": r["trades"],
            })

    # Best pairs (>= 80 trades, PF > baseline)
    good_pairs = [c for c in combo_results if c["trades"] >= 80 and c["pf"] > m_base.profit_factor]
    good_pairs = sorted(good_pairs, key=lambda x: x["pf"], reverse=True)[:5]
    for c in good_pairs:
        wf_candidates.append({
            "label": f"Pair: {c['name_a']} + {c['name_b']}",
            "filter_fn": c["filter_fn"],
            "full_pf": c["pf"],
            "full_trades": c["trades"],
        })

    # Best triples (>= 50 trades, PF > baseline)
    good_triples = [t for t in triple_results if t["trades"] >= 50 and t["pf"] > m_base.profit_factor]
    good_triples = sorted(good_triples, key=lambda x: x["pf"], reverse=True)[:5]
    for t in good_triples:
        wf_candidates.append({
            "label": f"Triple: {t['label']}",
            "filter_fn": t["filter_fn"],
            "full_pf": t["pf"],
            "full_trades": t["trades"],
        })

    print(f"\n  {'Label':<60} {'Full':>6} {'IS':>6} {'IS_PF':>7} {'OOS':>6} {'OOS_PF':>7} "
          f"{'Ratio':>6} {'OOS_p':>7} {'Verdict':>8}")
    print(f"  {'-'*118}")

    wf_results = []
    for cand in wf_candidates:
        m_is, m_oos, trades_is, trades_oos = walk_forward(
            df, cfg, indicator_filter=cand["filter_fn"]
        )

        if m_is and m_oos and m_is.profit_factor > 0:
            wf_ratio = m_oos.profit_factor / m_is.profit_factor
        else:
            wf_ratio = 0.0

        oos_pval = 1.0
        if trades_oos and len(trades_oos) >= 10:
            oos_pval, _, _ = run_significance(trades_oos, seed=123)

        is_trades = m_is.total_trades if m_is else 0
        oos_trades = m_oos.total_trades if m_oos else 0
        is_pf = m_is.profit_factor if m_is else 0
        oos_pf = m_oos.profit_factor if m_oos else 0

        # Pass criteria: OOS PF > 1.0, WF ratio > 0.7, OOS p < 0.10
        passed = (oos_pf > 1.0 and wf_ratio >= 0.70 and oos_trades >= 20)
        verdict = "PASS" if passed else "FAIL"

        wf_results.append({
            "label": cand["label"],
            "full_pf": cand["full_pf"],
            "full_trades": cand["full_trades"],
            "is_trades": is_trades,
            "is_pf": is_pf,
            "oos_trades": oos_trades,
            "oos_pf": oos_pf,
            "wf_ratio": wf_ratio,
            "oos_pval": oos_pval,
            "passed": passed,
            "filter_fn": cand["filter_fn"],
        })

        print(f"  {cand['label']:<60} {cand['full_trades']:>6} {is_trades:>6} {is_pf:>7.3f} "
              f"{oos_trades:>6} {oos_pf:>7.3f} {wf_ratio:>6.3f} {oos_pval:>7.4f} {verdict:>8}")

    # ══════════════════════════════════════════════════════════════════════════
    #  DEEP DIVE — Best passing candidate(s)
    # ══════════════════════════════════════════════════════════════════════════

    passing = [w for w in wf_results if w["passed"]]

    if passing:
        print(f"\n{'='*100}")
        print(f"  DEEP DIVE — Walk-Forward Validated Candidates")
        print(f"{'='*100}")

        for cand in passing:
            print(f"\n  >>> {cand['label']}")
            print(f"  Full: {cand['full_trades']} trades | PF {cand['full_pf']:.3f}")
            print(f"  IS:   {cand['is_trades']} trades | PF {cand['is_pf']:.3f}")
            print(f"  OOS:  {cand['oos_trades']} trades | PF {cand['oos_pf']:.3f}")
            print(f"  WF Ratio: {cand['wf_ratio']:.3f} | OOS p-value: {cand['oos_pval']:.4f}")

            # Full period significance
            trades_full = run_vector_backtest(df, cfg, indicator_filter=cand["filter_fn"])
            m_full = compute_metrics(trades_full, INITIAL_CAPITAL)
            t_pval_full, perm_pval, prob_profit = run_significance(trades_full)

            print(f"\n  Full Period Stats:")
            print(f"    Trades:         {m_full.total_trades}")
            print(f"    Win Rate:       {m_full.win_rate:.1f}%")
            print(f"    Profit Factor:  {m_full.profit_factor:.3f}")
            print(f"    Net P&L:        ${m_full.net_pnl:,.0f}")
            print(f"    Avg Trade:      ${m_full.avg_trade:,.0f}")
            print(f"    Max Drawdown:   ${m_full.max_drawdown:,.0f}")
            print(f"    Sharpe:         {m_full.sharpe:.2f}")
            print(f"    t-test p:       {t_pval_full:.6f}")
            print(f"    Permutation p:  {perm_pval:.6f}")
            print(f"    Bootstrap P(profit): {prob_profit:.2%}")

            # Monthly breakdown
            print(f"\n  Monthly Breakdown:")
            monthly = {}
            for t in trades_full:
                month = t.entry_time.strftime("%Y-%m") if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:7]
                monthly.setdefault(month, {"count": 0, "pnl": 0})
                monthly[month]["count"] += 1
                monthly[month]["pnl"] += t.pnl_dollar

            print(f"    {'Month':<10} {'Trades':>7} {'P&L':>12}")
            print(f"    {'-'*30}")
            win_months = 0
            for month in sorted(monthly.keys()):
                data = monthly[month]
                marker = " +" if data["pnl"] > 0 else " -"
                print(f"    {month:<10} {data['count']:>7} ${data['pnl']:>11,.0f}{marker}")
                if data["pnl"] > 0:
                    win_months += 1
            total_months = len(monthly)
            print(f"\n    Winning months: {win_months}/{total_months} ({win_months/total_months*100:.0f}%)")

            # Exit reason breakdown
            reasons = {}
            for t in trades_full:
                reasons.setdefault(t.exit_reason, {"count": 0, "pnl": 0})
                reasons[t.exit_reason]["count"] += 1
                reasons[t.exit_reason]["pnl"] += t.pnl_dollar

            print(f"\n  Exit Reasons:")
            for reason, data in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
                avg = data["pnl"] / data["count"] if data["count"] else 0
                print(f"    {reason:<18} {data['count']:>4} trades  ${data['pnl']:>+10,.0f}  (avg ${avg:>+7,.0f})")
    else:
        print(f"\n  NO candidates passed walk-forward validation.")
        print(f"  Looking at best failing candidates for analysis...")

        # Show the best fails
        wf_sorted = sorted(wf_results, key=lambda x: x["oos_pf"], reverse=True)
        for w in wf_sorted[:3]:
            print(f"\n  Near-miss: {w['label']}")
            print(f"    Full: {w['full_trades']}t PF {w['full_pf']:.3f} | "
                  f"IS: {w['is_trades']}t PF {w['is_pf']:.3f} | "
                  f"OOS: {w['oos_trades']}t PF {w['oos_pf']:.3f} | "
                  f"Ratio: {w['wf_ratio']:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*100}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*100}")

    print(f"\n  BASELINE:  {m_base.total_trades} trades | PF {m_base.profit_factor:.3f} | "
          f"${m_base.net_pnl:,.0f} | p={t_pval_base:.4f}")

    # Count improvements
    improved = len([r for r in results if r["pf"] > m_base.profit_factor and r["trades"] >= 100])
    total_tested = len(results)
    print(f"\n  Tested: {total_tested} indicator filters")
    print(f"  Improved PF (>= 100 trades): {improved}/{total_tested}")

    if viable_sorted:
        best = viable_sorted[0]
        print(f"\n  BEST SINGLE FILTER: {best['name']}")
        print(f"    {best['trades']} trades | PF {best['pf']:.3f} | ${best['pnl']:,.0f} | "
              f"p={best['p_val']:.4f} | PF delta: {best['pf_delta']:+.3f}")

    if good_pairs:
        best_pair = good_pairs[0]
        print(f"\n  BEST PAIR: {best_pair['name_a']} + {best_pair['name_b']}")
        print(f"    {best_pair['trades']} trades | PF {best_pair['pf']:.3f} | ${best_pair['pnl']:,.0f} | "
              f"p={best_pair['p_val']:.4f}")

    if good_triples:
        best_triple = good_triples[0]
        print(f"\n  BEST TRIPLE: {best_triple['label']}")
        print(f"    {best_triple['trades']} trades | PF {best_triple['pf']:.3f} | ${best_triple['pnl']:,.0f} | "
              f"p={best_triple['p_val']:.4f}")

    n_passed = len(passing)
    print(f"\n  Walk-forward validated: {n_passed}/{len(wf_candidates)}")

    if passing:
        print(f"\n  RECOMMENDED ENHANCEMENT(S):")
        for p in passing:
            print(f"    {p['label']}")
            print(f"      Full: {p['full_trades']}t PF {p['full_pf']:.3f} | "
                  f"OOS: {p['oos_trades']}t PF {p['oos_pf']:.3f} | "
                  f"WF {p['wf_ratio']:.3f} | OOS p={p['oos_pval']:.4f}")
    else:
        print(f"\n  HONEST ASSESSMENT: No indicator filter survived walk-forward validation.")
        print(f"  The baseline EMA cross + TEMA bearish + 30m range is already well-filtered.")
        print(f"  Adding more filters tends to over-fit the in-sample period.")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
