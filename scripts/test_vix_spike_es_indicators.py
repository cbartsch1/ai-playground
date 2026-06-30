#!/usr/bin/env python3
"""VIX Spike ES — Indicator Enhancement Study.

Goal: Add indicators/filters to improve PF, expand trade universe, improve exits.
Baseline: 107 trades, PF 2.641, +$23K, p=0.0012 (VIX spike >= 7%, ES down 0.2%, green bar exit).

Tests 24 indicator ideas across 4 categories:
  A. Expand trade universe (ES-native signals without VIX)
  B. Confirm VIX spike signal (improve PF)
  C. VIX-specific enhancements
  D. Exit improvements
  E. Combined universe (VIX + vol_ratio union/intersection)
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtester.data_loader import load_tos_csv

# ── Constants ──
ES_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "es_5m_databento_2yr.csv")
VIX_PARQUET = os.path.expanduser("~/projects/backtesting/spx/data/vix_daily.parquet")
WF_SPLIT = "2025-02-16"

POINT_VALUE = 50.0
COMMISSION = 2.50
SLIPPAGE_TICKS = 1
TICK_SIZE = 0.25
INITIAL_CAPITAL = 100_000.0
FLATTEN_TIME = 1555

# ════════════════════════════════════════════════════════════════
#   DATA LOADING & INDICATOR COMPUTATION
# ════════════════════════════════════════════════════════════════

def load_data():
    """Load ES 5m bars and VIX daily."""
    print(f"Loading ES data: {ES_CSV}")
    df = load_tos_csv(ES_CSV, instrument="ES")
    print(f"  {len(df):,} bars: {df.index[0].date()} to {df.index[-1].date()}")

    print(f"Loading VIX data: {VIX_PARQUET}")
    vix = pd.read_parquet(VIX_PARQUET)
    print(f"  {len(vix)} days")

    vix_lookup = {}
    for idx, row in vix.iterrows():
        d = idx.date() if hasattr(idx, 'date') else idx
        vix_lookup[d] = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row.get("low", 0)),
            "close": float(row["close"]),
        }

    return df, vix_lookup


def compute_all_indicators(df, vix_lookup):
    """Compute all indicators needed for the study. Adds columns to session_data dict."""

    rth = df[df['is_rth']].copy()
    sessions = sorted(rth['session_date'].dropna().unique())

    # ── Per-session computations ──
    session_data = {}

    # Pre-compute ATR series (14-period on daily closes)
    daily_bars = []
    for sess in sessions:
        mask = rth['session_date'] == sess
        grp = rth[mask]
        if len(grp) == 0:
            continue
        daily_bars.append({
            'date': sess,
            'open': grp['open'].iloc[0],
            'high': grp['high'].max(),
            'low': grp['low'].min(),
            'close': grp['close'].iloc[-1],
            'volume': grp['volume'].sum(),
            'first_close': grp['close'].iloc[0],
            'bar_count': len(grp),
        })

    daily = pd.DataFrame(daily_bars).set_index('date')

    # True Range & ATR
    daily['prev_close'] = daily['close'].shift(1)
    daily['tr'] = np.maximum(
        daily['high'] - daily['low'],
        np.maximum(
            abs(daily['high'] - daily['prev_close']),
            abs(daily['low'] - daily['prev_close'])
        )
    )
    daily['atr_14'] = daily['tr'].rolling(14, min_periods=5).mean()
    daily['atr_avg'] = daily['atr_14'].rolling(20, min_periods=10).mean()
    daily['vol_ratio'] = daily['atr_14'] / daily['atr_avg']

    # 20-day low
    daily['low_20'] = daily['low'].rolling(20, min_periods=10).min()

    # TEMA (9, 21, 55) on daily closes
    for period in [9, 21, 55]:
        ema1 = daily['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        daily[f'tema_{period}'] = 3 * ema1 - 3 * ema2 + ema3

    # VIX data per session
    daily['vix_open'] = np.nan
    daily['vix_high'] = np.nan
    daily['vix_close'] = np.nan
    daily['vix_low'] = np.nan
    for d in daily.index:
        if d in vix_lookup:
            daily.loc[d, 'vix_open'] = vix_lookup[d]['open']
            daily.loc[d, 'vix_high'] = vix_lookup[d]['high']
            daily.loc[d, 'vix_close'] = vix_lookup[d]['close']
            daily.loc[d, 'vix_low'] = vix_lookup[d]['low']

    daily['vix_spike_pct'] = (daily['vix_high'] - daily['vix_open']) / daily['vix_open']
    daily['vix_prev_close'] = daily['vix_close'].shift(1)
    daily['vix_up_days'] = 0
    for i in range(1, len(daily)):
        if daily['vix_close'].iloc[i] > daily['vix_close'].iloc[i-1]:
            daily.iloc[i, daily.columns.get_loc('vix_up_days')] = daily['vix_up_days'].iloc[i-1] + 1

    # Overnight range (globex high - globex low)
    for sess in sessions:
        # Get globex bars for this session (bars before RTH of this date)
        sess_rth = rth[rth['session_date'] == sess]
        if len(sess_rth) == 0:
            continue

        rth_start_time = sess_rth.index[0]

        # Get globex bars: from previous RTH close to this RTH open
        globex_mask = (df.index < rth_start_time) & (df['is_globex'] == True)
        # Limit to last ~16 hours (overnight only)
        cutoff = rth_start_time - pd.Timedelta(hours=16)
        globex_bars = df[globex_mask & (df.index >= cutoff)]

        on_high = globex_bars['high'].max() if len(globex_bars) > 0 else np.nan
        on_low = globex_bars['low'].min() if len(globex_bars) > 0 else np.nan
        on_range = on_high - on_low if not np.isnan(on_high) else np.nan

        if sess in daily.index:
            daily.loc[sess, 'on_high'] = on_high
            daily.loc[sess, 'on_low'] = on_low
            daily.loc[sess, 'on_range'] = on_range

    # Overnight range average
    daily['on_range_avg'] = daily['on_range'].rolling(20, min_periods=10).mean()
    daily['on_range_ratio'] = daily['on_range'] / daily['on_range_avg']

    # Gap: open vs prev close
    daily['gap_pct'] = (daily['open'] - daily['prev_close']) / daily['prev_close']

    # First 30 min range (first 6 bars of RTH)
    for sess in sessions:
        mask = rth['session_date'] == sess
        grp = rth[mask]
        if len(grp) < 6:
            if sess in daily.index:
                daily.loc[sess, 'first_30m_range'] = np.nan
            continue
        first_6 = grp.iloc[:6]
        r = first_6['high'].max() - first_6['low'].min()
        if sess in daily.index:
            daily.loc[sess, 'first_30m_range'] = r

    # Price below 20-day low
    daily['below_20d_low'] = daily['close'] < daily['low_20']

    # ES move from open (at each bar — computed per-bar in backtest)
    # For daily: day's close vs open
    daily['day_move_pct'] = (daily['close'] - daily['open']) / daily['open']

    print(f"  Computed indicators for {len(daily)} sessions")
    return df, daily, vix_lookup


def compute_5m_indicators(df):
    """Compute 5m bar-level indicators: RSI, TEMA, VWAP, volume ratio."""
    rth = df[df['is_rth']].copy()

    # RSI-14 on 5m closes (all bars, not just RTH, for continuity)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # TEMA 9/21/55 on 5m closes
    for period in [9, 21, 55]:
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        df[f'tema_{period}'] = 3 * ema1 - 3 * ema2 + ema3

    # Session VWAP
    df['vwap'] = np.nan
    for sess in rth['session_date'].dropna().unique():
        mask = (df['session_date'] == sess) & df['is_rth']
        grp = df[mask]
        if len(grp) == 0:
            continue
        cum_vol = grp['volume'].cumsum()
        cum_vp = (grp['hlc3'] * grp['volume']).cumsum()
        vwap = cum_vp / cum_vol.replace(0, np.nan)
        df.loc[mask, 'vwap'] = vwap

    # Volume moving average (20-bar on 5m)
    df['vol_ma_20'] = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio_5m'] = df['volume'] / df['vol_ma_20']

    # Consecutive red bars
    df['is_red'] = df['close'] < df['open']
    df['consec_red'] = 0
    consec = 0
    for i in range(len(df)):
        if df['is_red'].iloc[i]:
            consec += 1
        else:
            consec = 0
        df.iloc[i, df.columns.get_loc('consec_red')] = consec

    print(f"  Computed 5m indicators (RSI, TEMA, VWAP, vol_ratio, consec_red)")
    return df


# ════════════════════════════════════════════════════════════════
#   BACKTEST ENGINE (supports all indicator filters and exit modes)
# ════════════════════════════════════════════════════════════════

@dataclass
class EnhancedConfig:
    """All possible entry/filter/exit parameters."""
    # Entry signal type
    signal_type: str = "vix_spike"  # vix_spike | vol_ratio | gap_down | first_30m | below_20d | combined_score | vix_or_vol | vix_and_vol

    # VIX params
    spike_threshold: float = 0.07
    es_move_filter: float = -0.002

    # Vol ratio params (ES-native)
    vol_ratio_threshold: float = 2.0

    # Gap-down params
    gap_threshold: float = -0.005  # -0.5%

    # First 30m range params
    first_30m_threshold: float = 20.0  # points

    # Score-based params
    min_score: int = 3

    # VIX-specific enhancements
    require_vix_level: float = 0.0       # VIX must be > this (0 = disabled)
    require_vix_close_above_open: bool = False
    require_vix_multi_day: int = 0       # consecutive up days (0 = disabled)
    require_prev_vix_above: float = 0.0  # prev day VIX close (0 = disabled)
    vix_spike_min_if_level: float = 0.05 # lower spike threshold when VIX level is high

    # Confirmation filters
    require_tema_bearish: bool = False    # TEMA 9 < TEMA 21 on 5m at entry
    require_rsi_below: float = 0.0       # RSI < this (0 = disabled)
    require_vol_3x: bool = False         # entry bar volume > 3x average
    require_below_vwap: bool = False     # price below VWAP
    require_consec_red: int = 0          # min consecutive red bars (0 = disabled)

    # Exit params
    exit_mode: str = "green_bar"  # green_bar | delayed_green | green_after_hold | green_vol_drop | atr_target | support_target | momentum_decay | hold_all_day
    max_hold_bars: int = 18       # 90 min
    stop_bps: float = 30.0
    skip_green_bars: int = 0      # for delayed_green: skip first N green bars
    min_hold_bars: int = 0        # for green_after_hold: min bars before green exit
    atr_target_mult: float = 3.0  # for atr_target
    entry_start: int = 935
    entry_end: int = 1500


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    pnl_pts: float
    pnl_dollar: float
    exit_reason: str
    session_date: object
    signal_type: str = ""


def run_backtest(df, daily, vix_lookup, cfg: EnhancedConfig) -> List[Trade]:
    """Universal backtester supporting all indicator combinations."""

    # Build set of qualifying days based on signal_type
    qualifying_days = build_qualifying_days(daily, vix_lookup, cfg)

    if len(qualifying_days) == 0:
        return []

    # Pre-extract arrays
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    et_times = df['et_time'].values
    sessions = df['session_date'].values
    is_rth = df['is_rth'].values
    times = df.index

    # 5m indicators (may not exist for all modes)
    has_rsi = 'rsi_14' in df.columns
    has_tema = 'tema_9' in df.columns
    has_vwap = 'vwap' in df.columns
    has_vol = 'vol_ratio_5m' in df.columns
    has_consec = 'consec_red' in df.columns

    rsi_vals = df['rsi_14'].values if has_rsi else None
    tema9_vals = df['tema_9'].values if has_tema else None
    tema21_vals = df['tema_21'].values if has_tema else None
    vwap_vals = df['vwap'].values if has_vwap else None
    vol_ratio_vals = df['vol_ratio_5m'].values if has_vol else None
    consec_red_vals = df['consec_red'].values if has_consec else None
    volumes = df['volume'].values

    # Per-session data lookup
    daily_dict = {}
    for d in daily.index:
        daily_dict[d] = daily.loc[d]

    # Session opens
    rth_df = df[df['is_rth']]
    session_opens = rth_df.groupby('session_date')['open'].first().to_dict()

    trades = []
    traded_sessions = set()
    n = len(df)

    for i in range(n):
        if not is_rth[i]:
            continue

        sess = sessions[i]
        if sess is None or (isinstance(sess, float) and pd.isna(sess)):
            continue

        if sess in traded_sessions:
            continue

        if sess not in qualifying_days:
            continue

        et = et_times[i]
        if et < cfg.entry_start or et >= cfg.entry_end:
            continue

        # Red bar required
        if closes[i] >= opens[i]:
            continue

        # ES move filter
        if cfg.es_move_filter < 0:
            sess_open = session_opens.get(sess)
            if sess_open is not None and sess_open > 0:
                move_pct = (closes[i] - sess_open) / sess_open
                if move_pct > cfg.es_move_filter:
                    continue

        # ── Confirmation filters ──
        if cfg.require_tema_bearish and has_tema:
            if tema9_vals[i] >= tema21_vals[i]:
                continue

        if cfg.require_rsi_below > 0 and has_rsi:
            if rsi_vals[i] >= cfg.require_rsi_below:
                continue

        if cfg.require_vol_3x and has_vol:
            if vol_ratio_vals[i] < 3.0:
                continue

        if cfg.require_below_vwap and has_vwap:
            if not np.isnan(vwap_vals[i]) and closes[i] >= vwap_vals[i]:
                continue

        if cfg.require_consec_red > 0 and has_consec:
            if consec_red_vals[i] < cfg.require_consec_red:
                continue

        # ── ENTRY ──
        entry_price = closes[i] - (SLIPPAGE_TICKS * TICK_SIZE)
        stop_pts = entry_price * (cfg.stop_bps / 10000.0)
        stop_price = entry_price + stop_pts

        # ATR for exit modes that need it
        dd = daily_dict.get(sess)
        atr_val = dd['atr_14'] if dd is not None and not np.isnan(dd['atr_14']) else 20.0

        # Support levels for support_target exit
        on_low = dd['on_low'] if dd is not None and 'on_low' in dd.index and not np.isnan(dd.get('on_low', np.nan)) else None
        prev_day_low = None
        if dd is not None:
            # Find previous session
            sess_idx = list(daily.index).index(sess) if sess in daily.index else -1
            if sess_idx > 0:
                prev_day_low = daily.iloc[sess_idx - 1]['low']

        # ── SIMULATE EXIT ──
        exit_price = None
        exit_reason = None
        exit_idx = None
        green_count = 0

        for j in range(1, 200):  # large max to allow hold_all_day
            idx = i + j
            if idx >= n:
                exit_idx = n - 1
                exit_reason = "data_end"
                exit_price = closes[exit_idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                break

            if sessions[idx] != sess:
                exit_idx = idx - 1
                exit_reason = "session_end"
                exit_price = closes[exit_idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                break

            # Time stop (flatten)
            if et_times[idx] >= FLATTEN_TIME:
                exit_idx = idx
                exit_reason = "flatten"
                exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                break

            # Stop hit
            if highs[idx] >= stop_price:
                exit_idx = idx
                exit_reason = "stop"
                exit_price = stop_price + (SLIPPAGE_TICKS * TICK_SIZE)
                break

            # ── EXIT MODES ──
            bars_held = j

            if cfg.exit_mode == "green_bar":
                if bars_held > cfg.max_hold_bars:
                    exit_idx = idx
                    exit_reason = "max_hold"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                if closes[idx] > opens[idx]:
                    exit_idx = idx
                    exit_reason = "green_bar"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "delayed_green":
                if bars_held > cfg.max_hold_bars:
                    exit_idx = idx
                    exit_reason = "max_hold"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                if closes[idx] > opens[idx]:
                    green_count += 1
                    if green_count > cfg.skip_green_bars:
                        exit_idx = idx
                        exit_reason = "delayed_green"
                        exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                        break

            elif cfg.exit_mode == "green_after_hold":
                if bars_held > cfg.max_hold_bars:
                    exit_idx = idx
                    exit_reason = "max_hold"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                if bars_held >= cfg.min_hold_bars and closes[idx] > opens[idx]:
                    exit_idx = idx
                    exit_reason = "green_after_hold"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "green_vol_drop":
                if bars_held > cfg.max_hold_bars:
                    exit_idx = idx
                    exit_reason = "max_hold"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                if closes[idx] > opens[idx] and has_vol and vol_ratio_vals[idx] < 1.0:
                    exit_idx = idx
                    exit_reason = "green_vol_drop"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "atr_target":
                target = entry_price - (atr_val * cfg.atr_target_mult)
                if lows[idx] <= target:
                    exit_idx = idx
                    exit_reason = "atr_target"
                    exit_price = target + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                if bars_held > cfg.max_hold_bars:
                    exit_idx = idx
                    exit_reason = "max_hold"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "support_target":
                # Use overnight low or previous day low as target
                target = None
                if on_low is not None and on_low < entry_price:
                    target = on_low
                elif prev_day_low is not None and prev_day_low < entry_price:
                    target = prev_day_low
                if target is not None and lows[idx] <= target:
                    exit_idx = idx
                    exit_reason = "support_target"
                    exit_price = target + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                # Also allow green bar exit as backup
                if closes[idx] > opens[idx]:
                    exit_idx = idx
                    exit_reason = "green_bar"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                if bars_held > cfg.max_hold_bars:
                    exit_idx = idx
                    exit_reason = "max_hold"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "momentum_decay":
                # Exit when 5m vol_ratio drops below 1.0
                if has_vol and vol_ratio_vals[idx] < 1.0 and bars_held >= 2:
                    exit_idx = idx
                    exit_reason = "vol_decay"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                if bars_held > cfg.max_hold_bars:
                    exit_idx = idx
                    exit_reason = "max_hold"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "hold_all_day":
                # Only exit on flatten or stop
                pass  # loop continues until flatten_time or stop

        # If nothing triggered (shouldn't happen with flatten, but safety)
        if exit_price is None:
            exit_idx = min(i + 200, n - 1)
            exit_reason = "safety"
            exit_price = closes[exit_idx] + (SLIPPAGE_TICKS * TICK_SIZE)

        pnl_pts = entry_price - exit_price
        pnl_dollar = pnl_pts * POINT_VALUE - COMMISSION * 2

        trades.append(Trade(
            entry_time=times[i],
            exit_time=times[exit_idx],
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pts=pnl_pts,
            pnl_dollar=pnl_dollar,
            exit_reason=exit_reason,
            session_date=sess,
            signal_type=cfg.signal_type,
        ))
        traded_sessions.add(sess)

    return trades


def build_qualifying_days(daily, vix_lookup, cfg) -> set:
    """Determine which days qualify for entry based on signal_type."""
    qualifying = set()

    for d in daily.index:
        row = daily.loc[d]

        if cfg.signal_type == "vix_spike":
            if d not in vix_lookup:
                continue
            v = vix_lookup[d]
            spike = (v['high'] - v['open']) / v['open'] if v['open'] > 0 else 0
            if spike < cfg.spike_threshold:
                continue
            # VIX-specific enhancement filters
            if cfg.require_vix_level > 0 and v['open'] < cfg.require_vix_level:
                continue
            if cfg.require_vix_close_above_open and v['close'] <= v['open']:
                continue
            if cfg.require_vix_multi_day > 0 and row.get('vix_up_days', 0) < cfg.require_vix_multi_day:
                continue
            if cfg.require_prev_vix_above > 0:
                prev_c = row.get('vix_prev_close', 0)
                if np.isnan(prev_c) or prev_c < cfg.require_prev_vix_above:
                    continue
            qualifying.add(d)

        elif cfg.signal_type == "vol_ratio":
            vr = row.get('vol_ratio', np.nan)
            if np.isnan(vr) or vr < cfg.vol_ratio_threshold:
                continue
            qualifying.add(d)

        elif cfg.signal_type == "gap_down":
            gap = row.get('gap_pct', np.nan)
            if np.isnan(gap) or gap > cfg.gap_threshold:
                continue
            qualifying.add(d)

        elif cfg.signal_type == "first_30m":
            r = row.get('first_30m_range', np.nan)
            if np.isnan(r) or r < cfg.first_30m_threshold:
                continue
            qualifying.add(d)

        elif cfg.signal_type == "below_20d":
            if not row.get('below_20d_low', False):
                continue
            qualifying.add(d)

        elif cfg.signal_type == "vix_or_vol":
            # Union: VIX spike OR vol_ratio
            is_vix = False
            if d in vix_lookup:
                v = vix_lookup[d]
                spike = (v['high'] - v['open']) / v['open'] if v['open'] > 0 else 0
                if spike >= cfg.spike_threshold:
                    is_vix = True
            is_vol = False
            vr = row.get('vol_ratio', np.nan)
            if not np.isnan(vr) and vr >= cfg.vol_ratio_threshold:
                is_vol = True
            if is_vix or is_vol:
                qualifying.add(d)

        elif cfg.signal_type == "vix_and_vol":
            # Intersection: VIX spike AND vol_ratio
            is_vix = False
            if d in vix_lookup:
                v = vix_lookup[d]
                spike = (v['high'] - v['open']) / v['open'] if v['open'] > 0 else 0
                if spike >= cfg.spike_threshold:
                    is_vix = True
            vr = row.get('vol_ratio', np.nan)
            is_vol = not np.isnan(vr) and vr >= cfg.vol_ratio_threshold
            if is_vix and is_vol:
                qualifying.add(d)

        elif cfg.signal_type == "combined_score":
            score = 0
            # VIX spike = 2 pts
            if d in vix_lookup:
                v = vix_lookup[d]
                spike = (v['high'] - v['open']) / v['open'] if v['open'] > 0 else 0
                if spike >= 0.05:
                    score += 2
            # vol_ratio > 1.5 = 1 pt
            vr = row.get('vol_ratio', np.nan)
            if not np.isnan(vr) and vr >= 1.5:
                score += 1
            # Gap down > 0.3% = 1 pt
            gap = row.get('gap_pct', np.nan)
            if not np.isnan(gap) and gap <= -0.003:
                score += 1
            # First 30m range > 15 = 1 pt
            r = row.get('first_30m_range', np.nan)
            if not np.isnan(r) and r >= 15.0:
                score += 1
            # Below 20d low = 1 pt
            if row.get('below_20d_low', False):
                score += 1

            if score >= cfg.min_score:
                qualifying.add(d)

        elif cfg.signal_type == "vix_level_combo":
            # VIX > 25 AND spike > 5%
            if d not in vix_lookup:
                continue
            v = vix_lookup[d]
            spike = (v['high'] - v['open']) / v['open'] if v['open'] > 0 else 0
            if v['open'] >= cfg.require_vix_level and spike >= cfg.vix_spike_min_if_level:
                qualifying.add(d)

    return qualifying


# ════════════════════════════════════════════════════════════════
#   METRICS & SIGNIFICANCE
# ════════════════════════════════════════════════════════════════

def compute_metrics(trades):
    if not trades:
        return {"total": 0, "pf": 0, "net_pnl": 0, "win_rate": 0, "avg_trade": 0, "max_dd": 0, "sharpe": 0}

    pnls = [t.pnl_dollar for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_profit / gross_loss

    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    if len(pnls) > 1:
        returns = np.array(pnls) / INITIAL_CAPITAL
        sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252) if np.std(returns, ddof=1) > 0 else 0
    else:
        sharpe = 0

    return {
        "total": len(trades),
        "winners": len(wins),
        "losers": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pf": pf,
        "net_pnl": sum(pnls),
        "avg_trade": np.mean(pnls),
        "max_dd": max_dd,
        "sharpe": sharpe,
    }


def t_test(trades):
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t.pnl_dollar for t in trades])
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    return t_pval / 2 if t_stat > 0 else 1 - t_pval / 2


def walk_forward_test(df, daily, vix_lookup, cfg):
    """Walk-forward split test."""
    df_is = df[df.index < WF_SPLIT]
    df_oos = df[df.index >= WF_SPLIT]
    daily_is = daily[daily.index < pd.Timestamp(WF_SPLIT).date()]
    daily_oos = daily[daily.index >= pd.Timestamp(WF_SPLIT).date()]

    trades_is = run_backtest(df_is, daily_is, vix_lookup, cfg)
    trades_oos = run_backtest(df_oos, daily_oos, vix_lookup, cfg)

    m_is = compute_metrics(trades_is)
    m_oos = compute_metrics(trades_oos)

    return m_is, m_oos, trades_is, trades_oos


# ════════════════════════════════════════════════════════════════
#   TEST SUITE
# ════════════════════════════════════════════════════════════════

def fmt_result(name, m, p_val=None):
    """Format a single test result line."""
    if m['total'] == 0:
        return f"  {name:<50} |   0 trades"
    p_str = f"p={p_val:.4f}" if p_val is not None else "      "
    sig = ""
    if p_val is not None:
        if p_val < 0.001:
            sig = " ***"
        elif p_val < 0.01:
            sig = " **"
        elif p_val < 0.05:
            sig = " *"
    return (f"  {name:<50} | {m['total']:>4} trades | WR {m['win_rate']:>5.1f}% | "
            f"PF {m['pf']:>6.3f} | ${m['net_pnl']:>+10,.0f} | "
            f"Avg ${m['avg_trade']:>+7,.0f} | DD ${m['max_dd']:>8,.0f} | {p_str}{sig}")


def run_all_tests(df, daily, vix_lookup):
    """Run all 24 indicator tests."""

    results = []

    def test(name, cfg):
        trades = run_backtest(df, daily, vix_lookup, cfg)
        m = compute_metrics(trades)
        p = t_test(trades) if m['total'] >= 5 else 1.0
        results.append((name, m, p, cfg, trades))
        print(fmt_result(name, m, p))

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("  BASELINE")
    print("=" * 130)

    test("BASELINE: VIX>=7%, ES down 0.2%, green bar",
         EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002))

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("  CATEGORY A: EXPAND TRADE UNIVERSE (ES-native, no VIX required)")
    print("=" * 130)

    # A1: Vol ratio standalone
    for vr in [1.5, 1.8, 2.0, 2.5]:
        test(f"A1: vol_ratio > {vr} (standalone)",
             EnhancedConfig(signal_type="vol_ratio", vol_ratio_threshold=vr, es_move_filter=-0.002))

    # A2: Overnight range expansion
    # Signal based on on_range_ratio. We use first_30m as proxy since overnight is computed daily.
    # Actually, let's use vol_ratio which captures the same concept on a daily ATR basis.
    # We'll test gap_down separately.

    # A3: Gap down
    for gap in [-0.003, -0.005, -0.007, -0.01]:
        test(f"A3: gap_down > {gap*100:.1f}% (standalone)",
             EnhancedConfig(signal_type="gap_down", gap_threshold=gap, es_move_filter=0))

    # A4: First 30min range
    for rng in [15, 20, 25, 30]:
        test(f"A4: first_30m_range > {rng}pts (standalone)",
             EnhancedConfig(signal_type="first_30m", first_30m_threshold=rng, es_move_filter=-0.002))

    # A5: Below 20-day low
    test("A5: close below 20-day low (standalone)",
         EnhancedConfig(signal_type="below_20d", es_move_filter=-0.002))

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("  CATEGORY B: CONFIRM VIX SPIKE SIGNAL (add filter to baseline)")
    print("=" * 130)

    # B6: TEMA bearish
    test("B6: + TEMA 9 < TEMA 21 (bearish trend)",
         EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                        require_tema_bearish=True))

    # B7: RSI < 30
    for rsi_lvl in [30, 35, 40]:
        test(f"B7: + RSI < {rsi_lvl}",
             EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                            require_rsi_below=rsi_lvl))

    # B8: Volume > 3x average
    test("B8: + Volume > 3x average",
         EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                        require_vol_3x=True))

    # B9: Below VWAP
    test("B9: + Below VWAP",
         EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                        require_below_vwap=True))

    # B10: Consecutive red bars
    for n_red in [2, 3, 4]:
        test(f"B10: + {n_red}+ consecutive red bars",
             EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                            require_consec_red=n_red))

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("  CATEGORY C: VIX-SPECIFIC ENHANCEMENTS")
    print("=" * 130)

    # C11: VIX level + spike combo
    for vix_lvl in [20, 25, 30]:
        test(f"C11: VIX > {vix_lvl} AND spike > 5%",
             EnhancedConfig(signal_type="vix_level_combo", require_vix_level=vix_lvl,
                            vix_spike_min_if_level=0.05, es_move_filter=-0.002))

    # C12: VIX close > open (real fear)
    test("C12: + VIX close > open (sustained fear)",
         EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                        require_vix_close_above_open=True))

    # C13: VIX up 2+ consecutive days
    for ndays in [2, 3]:
        test(f"C13: + VIX up {ndays}+ consecutive days",
             EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                            require_vix_multi_day=ndays))

    # C14: Previous day VIX close > 20
    for prev_vix in [18, 20, 25]:
        test(f"C14: + prev day VIX close > {prev_vix}",
             EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                            require_prev_vix_above=prev_vix))

    # Lowering VIX threshold when level is elevated (expanding universe)
    test("C14b: VIX>25 + spike>=5% (lower thresh, elevated base)",
         EnhancedConfig(signal_type="vix_level_combo", require_vix_level=25,
                        vix_spike_min_if_level=0.05, es_move_filter=-0.002))

    test("C14c: VIX>20 + spike>=5% (lower thresh, moderate base)",
         EnhancedConfig(signal_type="vix_level_combo", require_vix_level=20,
                        vix_spike_min_if_level=0.05, es_move_filter=-0.002))

    test("C14d: VIX>20 + spike>=3% (lowest thresh, moderate base)",
         EnhancedConfig(signal_type="vix_level_combo", require_vix_level=20,
                        vix_spike_min_if_level=0.03, es_move_filter=-0.002))

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("  CATEGORY D: EXIT IMPROVEMENTS")
    print("=" * 130)

    base_entry = dict(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002)

    # D15: Delayed green bar (skip first 1-2 green bars)
    for skip in [1, 2, 3]:
        test(f"D15: delayed green (skip {skip} greens)",
             EnhancedConfig(**base_entry, exit_mode="delayed_green", skip_green_bars=skip, max_hold_bars=36))

    # D16: Green bar only after min hold
    for hold in [3, 6, 9, 12]:
        test(f"D16: green after {hold*5}min hold",
             EnhancedConfig(**base_entry, exit_mode="green_after_hold", min_hold_bars=hold, max_hold_bars=36))

    # D17: Green bar + volume drop
    test("D17: green bar + volume < avg (weak bounce)",
         EnhancedConfig(**base_entry, exit_mode="green_vol_drop", max_hold_bars=36))

    # D18: ATR target
    for mult in [2.0, 3.0, 4.0, 5.0]:
        test(f"D18: ATR target {mult}x",
             EnhancedConfig(**base_entry, exit_mode="atr_target", atr_target_mult=mult, max_hold_bars=36))

    # D19: Support level target
    test("D19: support target (ON low / prev day low)",
         EnhancedConfig(**base_entry, exit_mode="support_target", max_hold_bars=36))

    # D20: Momentum decay
    test("D20: exit on volume decay (vol_ratio < 1.0)",
         EnhancedConfig(**base_entry, exit_mode="momentum_decay", max_hold_bars=36))

    # D21: Hold all day (flatten at 15:55)
    test("D21: HOLD ALL DAY (flatten 15:55)",
         EnhancedConfig(**base_entry, exit_mode="hold_all_day", max_hold_bars=999, stop_bps=30))

    # D21b: Hold all day with wider stop
    for stop in [40, 50, 60, 80]:
        test(f"D21b: HOLD ALL DAY, stop {stop}bps",
             EnhancedConfig(**base_entry, exit_mode="hold_all_day", max_hold_bars=999, stop_bps=stop))

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("  CATEGORY E: COMBINED UNIVERSE (VIX + ES-native)")
    print("=" * 130)

    # E22: VIX spike OR vol_ratio (union)
    for vr in [1.5, 2.0, 2.5]:
        test(f"E22: VIX>=7% OR vol_ratio>{vr} (union)",
             EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.07, vol_ratio_threshold=vr,
                            es_move_filter=-0.002))

    # E23: VIX spike AND vol_ratio (intersection)
    for vr in [1.2, 1.5, 2.0]:
        test(f"E23: VIX>=7% AND vol_ratio>{vr} (intersection)",
             EnhancedConfig(signal_type="vix_and_vol", spike_threshold=0.07, vol_ratio_threshold=vr,
                            es_move_filter=-0.002))

    # E24: Score-based
    for min_s in [2, 3, 4]:
        test(f"E24: score>={min_s} (VIX=2, vol/gap/30m/20dLow=1 each)",
             EnhancedConfig(signal_type="combined_score", min_score=min_s, es_move_filter=-0.002))

    # E: VIX OR vol_ratio with best exit mode from D
    test("E25: VIX>=7% OR vol_ratio>2.0, delayed green skip 2",
         EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.07, vol_ratio_threshold=2.0,
                        es_move_filter=-0.002, exit_mode="delayed_green", skip_green_bars=2, max_hold_bars=36))

    test("E26: VIX>=7% OR vol_ratio>2.0, hold all day",
         EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.07, vol_ratio_threshold=2.0,
                        es_move_filter=-0.002, exit_mode="hold_all_day", max_hold_bars=999))

    # Lower VIX threshold + vol_ratio union (maximum trade expansion)
    test("E27: VIX>=5% OR vol_ratio>1.5 (max expansion)",
         EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.05, vol_ratio_threshold=1.5,
                        es_move_filter=-0.002))

    test("E28: VIX>=5% OR vol_ratio>1.5, hold all day",
         EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.05, vol_ratio_threshold=1.5,
                        es_move_filter=-0.002, exit_mode="hold_all_day", max_hold_bars=999))

    # Score-based with hold all day
    test("E29: score>=3, hold all day",
         EnhancedConfig(signal_type="combined_score", min_score=3, es_move_filter=-0.002,
                        exit_mode="hold_all_day", max_hold_bars=999))

    test("E30: score>=2, hold all day",
         EnhancedConfig(signal_type="combined_score", min_score=2, es_move_filter=-0.002,
                        exit_mode="hold_all_day", max_hold_bars=999))

    return results


def walk_forward_winners(df, daily, vix_lookup, results):
    """Walk-forward validate the top results."""
    print("\n" + "#" * 130)
    print("  WALK-FORWARD VALIDATION (top results)")
    print("#" * 130)

    # Sort by PF * sqrt(trades) (balancing quality and quantity)
    candidates = [(n, m, p, cfg, tr) for n, m, p, cfg, tr in results
                  if m['total'] >= 10 and m['pf'] > 1.5 and p < 0.10]
    candidates.sort(key=lambda x: -x[1]['pf'] * np.sqrt(x[1]['total']))

    # Also grab best by trade count (above 150 trades with PF > 1.5)
    volume_candidates = [(n, m, p, cfg, tr) for n, m, p, cfg, tr in results
                         if m['total'] >= 150 and m['pf'] > 1.3 and p < 0.10]
    volume_candidates.sort(key=lambda x: -x[1]['net_pnl'])

    # Combine, deduplicate
    seen = set()
    to_validate = []
    for item in candidates[:10] + volume_candidates[:5]:
        if item[0] not in seen:
            seen.add(item[0])
            to_validate.append(item)

    if not to_validate:
        print("  No candidates meet walk-forward criteria (PF > 1.5, p < 0.10, >= 10 trades)")
        # Relax and try again
        candidates = [(n, m, p, cfg, tr) for n, m, p, cfg, tr in results
                      if m['total'] >= 8 and m['pf'] > 1.2 and p < 0.15]
        candidates.sort(key=lambda x: -x[1]['net_pnl'])
        to_validate = candidates[:10]
        if not to_validate:
            print("  Still no candidates. All results below threshold.")
            return

    for name, m_full, p_full, cfg, trades_full in to_validate:
        print(f"\n{'─' * 130}")
        print(f"  {name}")
        print(f"  Full: {m_full['total']} trades | PF {m_full['pf']:.3f} | ${m_full['net_pnl']:>+,.0f} | p={p_full:.4f}")

        m_is, m_oos, trades_is, trades_oos = walk_forward_test(df, daily, vix_lookup, cfg)

        p_is = t_test(trades_is) if m_is['total'] >= 5 else 1.0
        p_oos = t_test(trades_oos) if m_oos['total'] >= 5 else 1.0

        print(f"  IS:   {m_is['total']:>4} trades | PF {m_is['pf']:>6.3f} | ${m_is['net_pnl']:>+10,.0f} | p={p_is:.4f}")
        print(f"  OOS:  {m_oos['total']:>4} trades | PF {m_oos['pf']:>6.3f} | ${m_oos['net_pnl']:>+10,.0f} | p={p_oos:.4f}")

        if m_is['total'] > 0 and m_oos['total'] > 0 and m_is['pf'] > 0:
            pf_ratio = m_oos['pf'] / m_is['pf']
            print(f"  PF ratio (OOS/IS): {pf_ratio:.3f} {'PASS' if pf_ratio >= 0.5 else 'WEAK' if pf_ratio >= 0.3 else 'FAIL'}")
        else:
            pf_ratio = 0

        # Verdict
        if m_oos['total'] >= 5 and m_oos['pf'] > 1.0 and p_oos < 0.10:
            print(f"  >>> OOS PROFITABLE: PF {m_oos['pf']:.3f}, p={p_oos:.4f}")
        elif m_oos['total'] >= 5 and m_oos['pf'] > 1.0:
            print(f"  >>> OOS PROFITABLE but p={p_oos:.4f} (weak significance)")
        else:
            print(f"  >>> OOS {'UNPROFITABLE' if m_oos['total'] > 0 else 'NO TRADES'}")

        # Exit reason breakdown for full period
        reasons = {}
        for t in trades_full:
            reasons.setdefault(t.exit_reason, {'count': 0, 'pnl': 0})
            reasons[t.exit_reason]['count'] += 1
            reasons[t.exit_reason]['pnl'] += t.pnl_dollar
        reason_str = " | ".join(f"{r}: {d['count']} (${d['pnl']:+,.0f})" for r, d in sorted(reasons.items(), key=lambda x: -x[1]['count']))
        print(f"  Exits: {reason_str}")


# ════════════════════════════════════════════════════════════════
#   SUPPLEMENTAL: COMBINATION TESTS (best filters + best exits + expanded universe)
# ════════════════════════════════════════════════════════════════

def run_combination_tests(df, daily, vix_lookup):
    """Run targeted combination tests based on what performed well individually."""
    print("\n" + "#" * 130)
    print("  COMBINATION TESTS (best filters x best exits x universe expansion)")
    print("#" * 130)

    results = []

    def test(name, cfg):
        trades = run_backtest(df, daily, vix_lookup, cfg)
        m = compute_metrics(trades)
        p = t_test(trades) if m['total'] >= 5 else 1.0
        results.append((name, m, p, cfg, trades))
        print(fmt_result(name, m, p))

    # ── Best exit on baseline ──
    # Test what we think the best exits will be
    print("\n  --- Best Exit on Baseline ---")

    test("COMBO1: baseline + hold all day + 50bps stop",
         EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                        exit_mode="hold_all_day", max_hold_bars=999, stop_bps=50))

    test("COMBO2: baseline + delayed green(2) + TEMA bearish",
         EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                        exit_mode="delayed_green", skip_green_bars=2, max_hold_bars=36,
                        require_tema_bearish=True))

    test("COMBO3: baseline + green after 30min hold",
         EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                        exit_mode="green_after_hold", min_hold_bars=6, max_hold_bars=36))

    # ── VIX threshold sweep with hold_all_day ──
    print("\n  --- VIX Threshold Sweep with Hold All Day ---")
    for thresh in [0.03, 0.05, 0.07, 0.10]:
        test(f"VIX>={thresh*100:.0f}% + ES dn 0.2% + hold all day",
             EnhancedConfig(signal_type="vix_spike", spike_threshold=thresh, es_move_filter=-0.002,
                            exit_mode="hold_all_day", max_hold_bars=999, stop_bps=40))

    # ── VIX threshold sweep with green_bar (baseline exit) ──
    print("\n  --- VIX Threshold Sweep with Green Bar Exit ---")
    for thresh in [0.03, 0.05, 0.10]:
        test(f"VIX>={thresh*100:.0f}% + ES dn 0.2% + green bar",
             EnhancedConfig(signal_type="vix_spike", spike_threshold=thresh, es_move_filter=-0.002,
                            exit_mode="green_bar", max_hold_bars=18))

    # ── ES move filter sweep ──
    print("\n  --- ES Move Filter Sweep ---")
    for es_mv in [0, -0.001, -0.002, -0.003, -0.005]:
        test(f"VIX>=7% + ES move {es_mv*100:.1f}% + green bar",
             EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=es_mv))

    # ── Expanded universe + best filters ──
    print("\n  --- Expanded Universe + Best Filters ---")

    test("EXPAND1: VIX>=5% OR vol>2.0, TEMA bearish",
         EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.05, vol_ratio_threshold=2.0,
                        es_move_filter=-0.002, require_tema_bearish=True))

    test("EXPAND2: VIX>=5% OR vol>2.0, below VWAP",
         EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.05, vol_ratio_threshold=2.0,
                        es_move_filter=-0.002, require_below_vwap=True))

    test("EXPAND3: VIX>=5% OR vol>1.5, TEMA bearish + below VWAP",
         EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.05, vol_ratio_threshold=1.5,
                        es_move_filter=-0.002, require_tema_bearish=True, require_below_vwap=True))

    test("EXPAND4: score>=2, TEMA bearish",
         EnhancedConfig(signal_type="combined_score", min_score=2, es_move_filter=-0.002,
                        require_tema_bearish=True))

    test("EXPAND5: score>=2, below VWAP",
         EnhancedConfig(signal_type="combined_score", min_score=2, es_move_filter=-0.002,
                        require_below_vwap=True))

    # ── Expanded universe + hold all day ──
    print("\n  --- Expanded Universe + Hold All Day ---")

    test("EXPHOLD1: VIX>=5% OR vol>2.0, TEMA bear, hold all day",
         EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.05, vol_ratio_threshold=2.0,
                        es_move_filter=-0.002, require_tema_bearish=True,
                        exit_mode="hold_all_day", max_hold_bars=999, stop_bps=50))

    test("EXPHOLD2: score>=2, TEMA bear, hold all day",
         EnhancedConfig(signal_type="combined_score", min_score=2, es_move_filter=-0.002,
                        require_tema_bearish=True,
                        exit_mode="hold_all_day", max_hold_bars=999, stop_bps=50))

    test("EXPHOLD3: score>=3, hold all day, 40bps stop",
         EnhancedConfig(signal_type="combined_score", min_score=3, es_move_filter=-0.002,
                        exit_mode="hold_all_day", max_hold_bars=999, stop_bps=40))

    test("EXPHOLD4: VIX>=5% OR vol>2.0, hold all day, 40bps",
         EnhancedConfig(signal_type="vix_or_vol", spike_threshold=0.05, vol_ratio_threshold=2.0,
                        es_move_filter=-0.002,
                        exit_mode="hold_all_day", max_hold_bars=999, stop_bps=40))

    # ── Stop sweep on hold all day ──
    print("\n  --- Stop Sweep on Hold All Day ---")
    for stop in [30, 40, 50, 60, 80, 100]:
        test(f"VIX>=7%, hold all day, stop {stop}bps",
             EnhancedConfig(signal_type="vix_spike", spike_threshold=0.07, es_move_filter=-0.002,
                            exit_mode="hold_all_day", max_hold_bars=999, stop_bps=stop))

    return results


# ════════════════════════════════════════════════════════════════
#   MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 130)
    print("  VIX SPIKE ES — INDICATOR ENHANCEMENT STUDY")
    print("  Baseline: 107 trades, PF 2.641, +$23,047, p=0.0012")
    print("=" * 130)

    # Load and compute
    df, vix_lookup = load_data()
    df, daily, vix_lookup = compute_all_indicators(df, vix_lookup)
    df = compute_5m_indicators(df)

    # Run all individual indicator tests
    results = run_all_tests(df, daily, vix_lookup)

    # Run combination tests
    combo_results = run_combination_tests(df, daily, vix_lookup)

    # Combine all results
    all_results = results + combo_results

    # Walk-forward validate the best
    walk_forward_winners(df, daily, vix_lookup, all_results)

    # ── FINAL SUMMARY ──
    print("\n" + "#" * 130)
    print("  FINAL RANKING (all tests, sorted by PF * sqrt(trades))")
    print("#" * 130)

    ranked = [(n, m, p) for n, m, p, _, _ in all_results if m['total'] >= 5]
    ranked.sort(key=lambda x: -x[1]['pf'] * np.sqrt(x[1]['total']))

    print(f"\n  {'Rank':<5} {'Name':<55} {'Trades':>6} {'WR':>6} {'PF':>7} {'Net P&L':>11} {'Avg':>8} {'p-val':>8}")
    print("  " + "-" * 115)
    for i, (name, m, p) in enumerate(ranked[:30], 1):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {i:<5} {name:<55} {m['total']:>6} {m['win_rate']:>5.1f}% {m['pf']:>7.3f} "
              f"${m['net_pnl']:>+10,.0f} ${m['avg_trade']:>+7,.0f}  {p:.4f} {sig}")

    # Also show best by trade count
    print(f"\n  BEST BY TRADE COUNT (min PF > 1.3):")
    by_count = [(n, m, p) for n, m, p in ranked if m['pf'] > 1.3]
    by_count.sort(key=lambda x: -x[1]['total'])
    for i, (name, m, p) in enumerate(by_count[:10], 1):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {i:<5} {name:<55} {m['total']:>6} {m['win_rate']:>5.1f}% {m['pf']:>7.3f} "
              f"${m['net_pnl']:>+10,.0f} ${m['avg_trade']:>+7,.0f}  {p:.4f} {sig}")


if __name__ == "__main__":
    main()
