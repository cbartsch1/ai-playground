#!/usr/bin/env python3
"""VIX Spike ES v2 — Structural improvement search.

Baseline: VIX daily high >= open * 1.07, ES down 0.2% from open,
          first red 5m bar, green bar exit, 30bps stop, 45min max hold.
          107 trades, PF 2.641, +$23K, p=0.0012, WF 1.119.

Goal: Find structurally better version — more trades without killing edge,
      or smarter exits to capture more of the move.

Tests:
  A. Signal variants (VIX spike detection)
  B. Entry variants (what triggers the short)
  C. Exit variants (how we leave the trade)
  D. Filter variants (what we skip)
  E. Combined best-of-breed
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data_loader import load_tos_csv
from backtester.indicators import compute_indicators

# ── Constants ──
ES_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "es_5m_databento_2yr.csv")
VIX_PARQUET = os.path.expanduser("~/projects/backtesting/spx/data/vix_daily.parquet")
WF_SPLIT = "2025-02-16"

POINT_VALUE = 50.0
COMMISSION = 2.50      # per side per contract
SLIPPAGE_TICKS = 1     # 1 tick = 0.25 pts
TICK_SIZE = 0.25
INITIAL_CAPITAL = 100_000.0
FLATTEN_TIME = 1555


@dataclass
class Config:
    """All tunable knobs for a VIX spike variant."""
    # -- Signal: how we detect VIX spike days --
    signal_mode: str = "daily_spike"    # daily_spike | overnight_jump | vix_level | multi_day | combo
    spike_threshold: float = 0.07       # for daily_spike: VIX high >= open * (1 + threshold)
    overnight_jump_pct: float = 0.03    # for overnight_jump: VIX open > prev close by X%
    vix_level: float = 25.0             # for vix_level: VIX open > level
    multi_day_count: int = 2            # for multi_day: VIX rising N consecutive days
    require_vol_ratio: float = 0.0      # >0 means require ES vol_ratio above this on entry bar

    # -- ES move filter --
    es_move_filter: float = -0.002      # min ES % from session open (negative = must be down)

    # -- Entry: what triggers the short --
    entry_mode: str = "red_bar"         # red_bar | any_bar | first_bar | tema_confirm | break_low
    max_entries_per_day: int = 1         # 1, 2, or 3 entries per spike day
    entry_start: int = 935
    entry_end: int = 1500

    # -- Exit: how we leave --
    exit_mode: str = "green_bar"        # green_bar | delayed_green | green_with_profit | atr_target |
                                        # time_stop_early | trailing | momentum_exit | hold_all_day |
                                        # fixed_target
    stop_bps: float = 30.0
    max_hold_bars: int = 9              # 45 min at 5m bars

    # Exit-specific params
    skip_green_count: int = 1           # delayed_green: skip first N green bars
    min_profit_for_green: float = 10.0  # green_with_profit: min pts profit before green bar exit
    atr_target_mult: float = 3.0        # atr_target: N x ATR as target
    early_time_stop: int = 1400         # time_stop_early: flatten at this time
    trail_trigger_atr: float = 1.0      # trailing: trigger trail after N x ATR profit
    trail_distance_atr: float = 1.5     # trailing: trail distance in ATR
    momentum_vol_floor: float = 1.0     # momentum_exit: exit when vol_ratio drops below this
    target_pts: float = 20.0            # fixed_target: absolute points target from entry

    # -- Filters --
    skip_gap_up: bool = False           # skip if ES opens above prev close on VIX spike day
    require_below_prev_low: bool = False  # require ES below yesterday's low
    morning_only: bool = False          # entry only 9:35-12:00
    tema_trend_filter: bool = False     # only short if close < TEMA 55
    skip_friday: bool = False           # skip Fridays
    skip_monday: bool = False           # skip Mondays

    # -- Look-ahead gate (2026-06-11 audit) --
    # daily_spike / combo / overnight_or_daily / overnight_and_daily read the
    # DAILY VIX HIGH (not knowable intraday); multi_day reads the SAME-DAY
    # close. All are LOOK-AHEAD. Explicit opt-in required; never valid for
    # validation. Mirrors spx backtester/strategies/vix_spike.py.
    allow_lookahead_daily_mode: bool = False

    label: str = ""                     # descriptive name for output


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    stop: float
    pnl_pts: float
    pnl_dollar: float
    exit_reason: str
    session_date: object
    vix_open: float = 0.0
    vix_high: float = 0.0
    vix_spike_pct: float = 0.0
    direction: int = -1
    setup: str = "VIX_SPIKE"


# ────────────────────────────────────────────────────────
#  DATA LOADING
# ────────────────────────────────────────────────────────

def load_data():
    """Load ES 5m bars with indicators, and VIX daily."""
    print(f"Loading ES data: {ES_CSV}")
    df = load_tos_csv(ES_CSV, instrument="ES")
    compute_indicators(df)
    print(f"  {len(df):,} bars: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    print(f"Loading VIX data: {VIX_PARQUET}")
    vix = pd.read_parquet(VIX_PARQUET)

    # Build VIX daily lookup with prev close for overnight jump calc
    vix_lookup = {}
    prev_close = None
    prev_high = None
    prev_low = None
    for idx, row in vix.iterrows():
        d = idx.date() if hasattr(idx, 'date') else idx
        vix_lookup[d] = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row.get("low", 0)),
            "close": float(row["close"]),
            "prev_close": prev_close,
            "prev_high": prev_high,
            "prev_low": prev_low,
        }
        prev_close = float(row["close"])
        prev_high = float(row["high"])
        prev_low = float(row.get("low", 0))

    es_dates = set(df[df['is_rth']]['session_date'].dropna().unique())
    vix_dates = set(vix_lookup.keys())
    overlap = es_dates & vix_dates
    print(f"  VIX/ES overlap: {len(overlap)} sessions")

    # Precompute session fields
    rth = df[df['is_rth']].copy()
    session_opens = rth.groupby('session_date')['open'].first().to_dict()
    session_highs = rth.groupby('session_date')['high'].max().to_dict()
    session_lows = rth.groupby('session_date')['low'].min().to_dict()

    # Previous session close (last RTH bar's close)
    session_closes = rth.groupby('session_date')['close'].last().to_dict()

    # Build prev_session_close and prev_session_low lookup
    sorted_dates = sorted(session_closes.keys())
    prev_session_data = {}
    for i, d in enumerate(sorted_dates):
        if i > 0:
            prev_d = sorted_dates[i-1]
            prev_session_data[d] = {
                "prev_close": session_closes[prev_d],
                "prev_low": session_lows.get(prev_d, 0),
            }

    return df, vix_lookup, session_opens, prev_session_data


# ────────────────────────────────────────────────────────
#  SIGNAL DETECTION: Is today a VIX spike day?
# ────────────────────────────────────────────────────────

LOOKAHEAD_MODES = {"daily_spike", "combo", "overnight_or_daily",
                   "overnight_and_daily", "multi_day"}

LOOKAHEAD_MSG = (
    "LOOKAHEAD: signal_mode '{mode}' reads the daily VIX HIGH (or same-day "
    "close), which is not knowable at intraday entry time. Results are NOT "
    "valid for strategy validation (2026-06-11 audit; the original "
    "vix_spike_es validation is VOID for this reason). To run anyway "
    "(diagnostics only), set Config(allow_lookahead_daily_mode=True)."
)

_lookahead_warned = set()


def is_spike_day(sess_date, vix_lookup, cfg: Config) -> Optional[Dict]:
    """Check if session qualifies as a VIX spike day per config."""
    if cfg.signal_mode in LOOKAHEAD_MODES:
        if not cfg.allow_lookahead_daily_mode:
            raise RuntimeError(LOOKAHEAD_MSG.format(mode=cfg.signal_mode))
        if cfg.signal_mode not in _lookahead_warned:
            _lookahead_warned.add(cfg.signal_mode)
            print(f"[LOOKAHEAD WARNING] is_spike_day: non-causal mode "
                  f"'{cfg.signal_mode}' enabled by explicit opt-in. "
                  f"{LOOKAHEAD_MSG.format(mode=cfg.signal_mode)}")

    vix = vix_lookup.get(sess_date)
    if not vix or vix["open"] <= 0:
        return None

    mode = cfg.signal_mode

    if mode == "daily_spike":
        # Original: VIX high >= open * (1 + threshold)
        if vix["high"] >= vix["open"] * (1 + cfg.spike_threshold):
            return {
                "vix_open": vix["open"],
                "vix_high": vix["high"],
                "spike_pct": (vix["high"] - vix["open"]) / vix["open"],
            }

    elif mode == "overnight_jump":
        # VIX opens higher than previous close (gap up in fear)
        if vix["prev_close"] and vix["prev_close"] > 0:
            jump = (vix["open"] - vix["prev_close"]) / vix["prev_close"]
            if jump >= cfg.overnight_jump_pct:
                return {
                    "vix_open": vix["open"],
                    "vix_high": vix["high"],
                    "spike_pct": jump,
                }

    elif mode == "vix_level":
        # VIX open above absolute level
        if vix["open"] >= cfg.vix_level:
            return {
                "vix_open": vix["open"],
                "vix_high": vix["high"],
                "spike_pct": (vix["high"] - vix["open"]) / vix["open"],
            }

    elif mode == "multi_day":
        # VIX closing higher for N consecutive days
        d = sess_date
        count = 0
        for _ in range(cfg.multi_day_count):
            v = vix_lookup.get(d)
            if not v or v["prev_close"] is None:
                break
            if v["close"] > v["prev_close"]:
                count += 1
            else:
                break
            # Go to previous day (approximate — skip weekends)
            import datetime
            d = d - datetime.timedelta(days=1)
            while d not in vix_lookup and d > sess_date - datetime.timedelta(days=10):
                d = d - datetime.timedelta(days=1)

        if count >= cfg.multi_day_count:
            return {
                "vix_open": vix["open"],
                "vix_high": vix["high"],
                "spike_pct": (vix["high"] - vix["open"]) / vix["open"],
            }

    elif mode == "combo":
        # VIX spike AND vol_ratio requirement (vol_ratio checked at entry bar level)
        if vix["high"] >= vix["open"] * (1 + cfg.spike_threshold):
            return {
                "vix_open": vix["open"],
                "vix_high": vix["high"],
                "spike_pct": (vix["high"] - vix["open"]) / vix["open"],
            }

    elif mode == "overnight_or_daily":
        # EITHER overnight jump OR intraday spike (union = more trades)
        triggered = False
        spike_pct = 0
        if vix["prev_close"] and vix["prev_close"] > 0:
            jump = (vix["open"] - vix["prev_close"]) / vix["prev_close"]
            if jump >= cfg.overnight_jump_pct:
                triggered = True
                spike_pct = jump
        if vix["high"] >= vix["open"] * (1 + cfg.spike_threshold):
            triggered = True
            spike_pct = max(spike_pct, (vix["high"] - vix["open"]) / vix["open"])
        if triggered:
            return {
                "vix_open": vix["open"],
                "vix_high": vix["high"],
                "spike_pct": spike_pct,
            }

    elif mode == "overnight_and_daily":
        # BOTH overnight jump AND intraday spike (intersection = highest conviction)
        if vix["prev_close"] and vix["prev_close"] > 0:
            jump = (vix["open"] - vix["prev_close"]) / vix["prev_close"]
            if jump >= cfg.overnight_jump_pct and vix["high"] >= vix["open"] * (1 + cfg.spike_threshold):
                return {
                    "vix_open": vix["open"],
                    "vix_high": vix["high"],
                    "spike_pct": (vix["high"] - vix["open"]) / vix["open"],
                }

    return None


# ────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ────────────────────────────────────────────────────────

def run_backtest(df, vix_lookup, session_opens, prev_session_data, cfg: Config) -> List[Trade]:
    """Run VIX Spike v2 backtest."""

    # Pre-extract arrays for speed
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    et_times = df['et_time'].values
    sessions = df['session_date'].values
    is_rth = df['is_rth'].values
    weekdays = df['weekday'].values
    times = df.index
    n = len(df)

    # Indicator arrays (may be NaN early)
    has_indicators = 'vol_ratio' in df.columns
    if has_indicators:
        vol_ratios = df['vol_ratio'].values
        tema_trends = df['tema_trend'].values if 'tema_trend' in df.columns else None
        tema_bearish = df['tema_bearish'].values if 'tema_bearish' in df.columns else None
        atr_vals = df['atr'].values if 'atr' in df.columns else None
    else:
        vol_ratios = np.ones(n)
        tema_trends = None
        tema_bearish = None
        atr_vals = None

    trades = []
    entries_today = {}  # sess -> count of entries today
    prev_bar_low = {}   # sess -> lowest low of previous bar for break_low entry

    # Effective entry window
    entry_start = cfg.entry_start
    entry_end = 1200 if cfg.morning_only else cfg.entry_end

    for i in range(n):
        if not is_rth[i]:
            continue

        sess = sessions[i]
        if sess is None or (isinstance(sess, float) and pd.isna(sess)):
            continue

        et = et_times[i]

        # Max entries per day check
        today_entries = entries_today.get(sess, 0)
        if today_entries >= cfg.max_entries_per_day:
            continue

        # Check if VIX spike day
        spike = is_spike_day(sess, vix_lookup, cfg)
        if spike is None:
            continue

        # Time window
        if et < entry_start or et >= entry_end:
            continue

        # ── DAY-LEVEL FILTERS ──

        # Friday filter
        if cfg.skip_friday and weekdays[i] == 4:
            continue

        # Monday filter
        if cfg.skip_monday and weekdays[i] == 0:
            continue

        # Gap-up filter: skip if ES opened above previous session close
        if cfg.skip_gap_up:
            prev_data = prev_session_data.get(sess)
            sess_open = session_opens.get(sess)
            if prev_data and sess_open and sess_open > prev_data["prev_close"]:
                continue

        # Below previous session low filter
        if cfg.require_below_prev_low:
            prev_data = prev_session_data.get(sess)
            if prev_data and closes[i] > prev_data["prev_low"]:
                continue

        # ── BAR-LEVEL FILTERS ──

        # ES move from open filter
        if cfg.es_move_filter < 0:
            sess_open = session_opens.get(sess)
            if sess_open and sess_open > 0:
                move_pct = (closes[i] - sess_open) / sess_open
                if move_pct > cfg.es_move_filter:
                    continue

        # Vol ratio filter (for combo mode or standalone)
        if cfg.require_vol_ratio > 0:
            if np.isnan(vol_ratios[i]) or vol_ratios[i] < cfg.require_vol_ratio:
                continue

        # TEMA trend filter: only short if close < TEMA 55
        if cfg.tema_trend_filter and tema_trends is not None:
            if np.isnan(tema_trends[i]) or closes[i] >= tema_trends[i]:
                continue

        # ── ENTRY LOGIC ──

        entered = False

        if cfg.entry_mode == "red_bar":
            # Original: only enter on red (bearish) 5m bar
            if closes[i] < opens[i]:
                entered = True

        elif cfg.entry_mode == "any_bar":
            # Enter on any bar (skip red filter)
            entered = True

        elif cfg.entry_mode == "first_bar":
            # Enter only on the first RTH bar (9:30 or 9:35)
            if et == 930 or et == 935:
                entered = True

        elif cfg.entry_mode == "tema_confirm":
            # Enter when TEMA bearish (fast < slow)
            if tema_bearish is not None and not np.isnan(tema_bearish[i]) and tema_bearish[i]:
                if closes[i] < opens[i]:  # still require red bar
                    entered = True

        elif cfg.entry_mode == "break_low":
            # Enter when bar breaks below prior bar's low
            if i > 0 and closes[i] < lows[i-1] and closes[i] < opens[i]:
                entered = True

        if not entered:
            continue

        # ── ENTRY PRICE ──
        entry_price = closes[i] - (SLIPPAGE_TICKS * TICK_SIZE)

        # Stop: percentage-based
        stop_pts = entry_price * (cfg.stop_bps / 10000.0)
        stop_price = entry_price + stop_pts

        # ATR at entry (for ATR-based exits)
        entry_atr = atr_vals[i] if atr_vals is not None and not np.isnan(atr_vals[i]) else 10.0

        # ── SIMULATE EXIT ──
        exit_price = None
        exit_reason = None
        exit_idx = None
        green_bars_seen = 0
        trail_active = False
        trail_stop = None
        best_price = entry_price  # best (lowest for short) price seen

        for j in range(1, cfg.max_hold_bars + 1):
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

            # Time stop
            flatten_time = cfg.early_time_stop if cfg.exit_mode == "time_stop_early" else FLATTEN_TIME
            if et_times[idx] >= flatten_time:
                exit_idx = idx
                exit_reason = "time_stop"
                exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                break

            # Stop hit (short: high >= stop)
            if highs[idx] >= stop_price:
                exit_idx = idx
                exit_reason = "stop"
                exit_price = stop_price + (SLIPPAGE_TICKS * TICK_SIZE)
                break

            # Trailing stop check (if active)
            if trail_active and trail_stop is not None:
                if highs[idx] >= trail_stop:
                    exit_idx = idx
                    exit_reason = "trail_stop"
                    exit_price = trail_stop + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            # Track best price for trailing
            if lows[idx] < best_price:
                best_price = lows[idx]

            # ── EXIT MODE LOGIC ──

            if cfg.exit_mode == "green_bar":
                # Original: exit on first green bar
                if closes[idx] > opens[idx]:
                    exit_idx = idx
                    exit_reason = "green_bar"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "delayed_green":
                # Skip first N green bars
                if closes[idx] > opens[idx]:
                    green_bars_seen += 1
                    if green_bars_seen > cfg.skip_green_count:
                        exit_idx = idx
                        exit_reason = "delayed_green"
                        exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                        break

            elif cfg.exit_mode == "green_with_profit":
                # Only exit on green bar if already in profit by min_profit_for_green pts
                current_profit = entry_price - closes[idx]
                if closes[idx] > opens[idx] and current_profit >= cfg.min_profit_for_green:
                    exit_idx = idx
                    exit_reason = "green_profit"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                elif closes[idx] > opens[idx] and current_profit < 0:
                    # Exit losing green bars immediately (don't hold losers)
                    exit_idx = idx
                    exit_reason = "green_loss_cut"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "atr_target":
                # Fixed ATR-multiple target
                target_price = entry_price - (entry_atr * cfg.atr_target_mult)
                if lows[idx] <= target_price:
                    exit_idx = idx
                    exit_reason = "atr_target"
                    exit_price = target_price + (SLIPPAGE_TICKS * TICK_SIZE)
                    break
                # Also exit on green bar as backup
                if closes[idx] > opens[idx]:
                    exit_idx = idx
                    exit_reason = "green_bar"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "time_stop_early":
                # Time stop handled above; otherwise use green bar
                if closes[idx] > opens[idx]:
                    exit_idx = idx
                    exit_reason = "green_bar"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "trailing":
                # Activate trailing stop after N*ATR profit
                profit_pts = entry_price - lows[idx]
                trigger_level = entry_atr * cfg.trail_trigger_atr
                if profit_pts >= trigger_level and not trail_active:
                    trail_active = True
                    trail_distance = entry_atr * cfg.trail_distance_atr
                    trail_stop = best_price + trail_distance

                if trail_active:
                    trail_distance = entry_atr * cfg.trail_distance_atr
                    trail_stop = best_price + trail_distance

                # Also check trail stop on this bar (in case just activated)
                if trail_active and trail_stop is not None and highs[idx] >= trail_stop:
                    exit_idx = idx
                    exit_reason = "trail_stop"
                    exit_price = trail_stop + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "momentum_exit":
                # Exit when vol_ratio drops below threshold (panic subsiding)
                if has_indicators and not np.isnan(vol_ratios[idx]):
                    if vol_ratios[idx] < cfg.momentum_vol_floor:
                        exit_idx = idx
                        exit_reason = "momentum_fade"
                        exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                        break

            elif cfg.exit_mode == "hold_all_day":
                # Hold until flatten time — no early exit (handled above via time stop)
                pass

            elif cfg.exit_mode == "fixed_target":
                # Absolute points target from entry. Short: target = entry - target_pts.
                # First bar where low <= target → exit at target. Stop and time/session
                # exits handled above; max_hold_bars semantics preserved.
                target_price = entry_price - cfg.target_pts
                if lows[idx] <= target_price:
                    exit_idx = idx
                    exit_reason = "fixed_target"
                    exit_price = target_price + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

            elif cfg.exit_mode == "green_bar_or_trail":
                # Hybrid: green bar exit but with a trailing stop as safety net
                profit_pts = entry_price - lows[idx]
                trigger_level = entry_atr * cfg.trail_trigger_atr
                if profit_pts >= trigger_level and not trail_active:
                    trail_active = True
                trail_distance = entry_atr * cfg.trail_distance_atr
                if trail_active:
                    trail_stop = best_price + trail_distance
                    if highs[idx] >= trail_stop:
                        exit_idx = idx
                        exit_reason = "trail_stop"
                        exit_price = trail_stop + (SLIPPAGE_TICKS * TICK_SIZE)
                        break

                if closes[idx] > opens[idx]:
                    exit_idx = idx
                    exit_reason = "green_bar"
                    exit_price = closes[idx] + (SLIPPAGE_TICKS * TICK_SIZE)
                    break

        # Max hold reached without exit
        if exit_price is None:
            exit_idx = min(i + cfg.max_hold_bars, n - 1)
            if sessions[exit_idx] != sess:
                exit_idx -= 1
            exit_reason = "max_hold"
            exit_price = closes[exit_idx] + (SLIPPAGE_TICKS * TICK_SIZE)

        # ── P&L ──
        pnl_pts = entry_price - exit_price
        commission_total = COMMISSION * 2
        pnl_dollar = pnl_pts * POINT_VALUE - commission_total

        trade = Trade(
            entry_time=times[i],
            exit_time=times[exit_idx],
            entry_price=entry_price,
            exit_price=exit_price,
            stop=stop_price,
            pnl_pts=pnl_pts,
            pnl_dollar=pnl_dollar,
            exit_reason=exit_reason,
            session_date=sess,
            vix_open=spike.get("vix_open", 0),
            vix_high=spike.get("vix_high", 0),
            vix_spike_pct=spike.get("spike_pct", 0),
        )
        trades.append(trade)
        entries_today[sess] = entries_today.get(sess, 0) + 1

    return trades


# ────────────────────────────────────────────────────────
#  METRICS & STATS
# ────────────────────────────────────────────────────────

def compute_metrics(trades: List[Trade]) -> Dict:
    if not trades:
        return {"total": 0, "pf": 0, "net_pnl": 0, "win_rate": 0,
                "sharpe": 0, "max_dd": 0, "avg_trade": 0, "winners": 0, "losers": 0,
                "gross_profit": 0, "gross_loss": 0}

    pnls = [t.pnl_dollar for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    # Sharpe
    if len(pnls) > 1:
        returns = np.array(pnls) / INITIAL_CAPITAL
        std = np.std(returns, ddof=1)
        sharpe = np.mean(returns) / std * np.sqrt(252) if std > 0 else 0
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


def t_test(trades: List[Trade]) -> float:
    """One-sided t-test p-value (we expect positive mean)."""
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t.pnl_dollar for t in trades])
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    return t_pval / 2 if t_stat > 0 else 1 - t_pval / 2


def permutation_test(trades: List[Trade], seed=42) -> float:
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t.pnl_dollar for t in trades])
    obs = np.sum(pnls)
    abs_pnls = np.abs(pnls)
    rng = np.random.default_rng(seed)
    n_perm = 10000
    count = sum(1 for _ in range(n_perm)
                if np.dot(rng.choice([-1.0, 1.0], size=len(pnls)), abs_pnls) >= obs)
    return count / n_perm


def bootstrap_profit_prob(trades: List[Trade], seed=42) -> float:
    if len(trades) < 5:
        return 0.0
    pnls = np.array([t.pnl_dollar for t in trades])
    rng = np.random.default_rng(seed)
    boot = np.array([np.sum(rng.choice(pnls, size=len(pnls), replace=True)) for _ in range(10000)])
    return float(np.mean(boot > 0))


# ────────────────────────────────────────────────────────
#  WALK-FORWARD
# ────────────────────────────────────────────────────────

def walk_forward(df, vix_lookup, session_opens, prev_session_data, cfg: Config):
    """Walk-forward split and test. Returns (trades_is, trades_oos, m_is, m_oos)."""
    df_is = df[df.index < WF_SPLIT]
    df_oos = df[df.index >= WF_SPLIT]

    trades_is = run_backtest(df_is, vix_lookup, session_opens, prev_session_data, cfg)
    trades_oos = run_backtest(df_oos, vix_lookup, session_opens, prev_session_data, cfg)

    return (trades_is, trades_oos,
            compute_metrics(trades_is), compute_metrics(trades_oos))


# ────────────────────────────────────────────────────────
#  VARIANT DEFINITIONS
# ────────────────────────────────────────────────────────

def build_variants() -> List[Config]:
    """Build all structural variants to test."""
    variants = []

    # ══════════════════════════════════════════════════════
    #  BASELINE (for comparison)
    # ══════════════════════════════════════════════════════
    variants.append(Config(
        label="BASELINE (7% spike, red bar, green exit, 30bps, 45m)",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
    ))

    # ══════════════════════════════════════════════════════
    #  A. SIGNAL VARIANTS — Different VIX spike detection
    # ══════════════════════════════════════════════════════

    # A1: Lower VIX spike thresholds (more trades?)
    for thresh in [0.03, 0.04, 0.05, 0.06]:
        variants.append(Config(
            label=f"A1: Daily spike {thresh:.0%}",
            signal_mode="daily_spike", spike_threshold=thresh,
            entry_mode="red_bar", exit_mode="green_bar",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        ))

    # A2: Overnight VIX jump (VIX opens above prev close)
    for jump in [0.02, 0.03, 0.05, 0.07, 0.10]:
        variants.append(Config(
            label=f"A2: Overnight jump {jump:.0%}",
            signal_mode="overnight_jump", overnight_jump_pct=jump,
            entry_mode="red_bar", exit_mode="green_bar",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        ))

    # A3: VIX absolute level
    for level in [18, 20, 22, 25, 30]:
        variants.append(Config(
            label=f"A3: VIX level >= {level}",
            signal_mode="vix_level", vix_level=level,
            entry_mode="red_bar", exit_mode="green_bar",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        ))

    # A4: Multi-day VIX rise
    for days in [2, 3]:
        variants.append(Config(
            label=f"A4: VIX rising {days} days",
            signal_mode="multi_day", multi_day_count=days,
            entry_mode="red_bar", exit_mode="green_bar",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        ))

    # A5: VIX spike + vol_ratio combo
    for vr in [1.2, 1.5, 2.0]:
        variants.append(Config(
            label=f"A5: 7% spike + vol_ratio > {vr}",
            signal_mode="combo", spike_threshold=0.07, require_vol_ratio=vr,
            entry_mode="red_bar", exit_mode="green_bar",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        ))

    # A6: Union signal (overnight OR daily spike) — more trades
    variants.append(Config(
        label="A6: Overnight 3% OR daily 7% (union)",
        signal_mode="overnight_or_daily", spike_threshold=0.07, overnight_jump_pct=0.03,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
    ))
    variants.append(Config(
        label="A6: Overnight 5% OR daily 7% (union)",
        signal_mode="overnight_or_daily", spike_threshold=0.07, overnight_jump_pct=0.05,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
    ))

    # A7: Intersection (overnight AND daily spike) — highest conviction
    variants.append(Config(
        label="A7: Overnight 3% AND daily 5% (intersection)",
        signal_mode="overnight_and_daily", spike_threshold=0.05, overnight_jump_pct=0.03,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
    ))

    # ══════════════════════════════════════════════════════
    #  B. ES MOVE THRESHOLD VARIANTS
    # ══════════════════════════════════════════════════════

    for move in [0.0, -0.001, -0.0015, -0.003, -0.005]:
        variants.append(Config(
            label=f"B: ES move filter {move:.1%}",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="green_bar",
            stop_bps=30, max_hold_bars=9, es_move_filter=move,
        ))

    # ══════════════════════════════════════════════════════
    #  C. ENTRY VARIANTS
    # ══════════════════════════════════════════════════════

    # C1: Any bar (not just red) — does red filter help or hurt?
    variants.append(Config(
        label="C1: Any bar entry (no red filter)",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="any_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
    ))

    # C2: First bar only (9:35)
    variants.append(Config(
        label="C2: First bar only (9:35)",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="first_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
    ))

    # C3: TEMA confirmation
    variants.append(Config(
        label="C3: TEMA bearish confirmation",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="tema_confirm", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
    ))

    # C4: Break below prior bar's low
    variants.append(Config(
        label="C4: Break below prior bar low",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="break_low", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
    ))

    # C5: Multiple entries per day (2 or 3)
    for max_e in [2, 3]:
        variants.append(Config(
            label=f"C5: Up to {max_e} entries/day",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="green_bar",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
            max_entries_per_day=max_e,
        ))

    # ══════════════════════════════════════════════════════
    #  D. EXIT VARIANTS
    # ══════════════════════════════════════════════════════

    # D1: Delayed green bar (skip first 1-2 green bars)
    for skip in [1, 2]:
        variants.append(Config(
            label=f"D1: Skip first {skip} green bar(s)",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="delayed_green",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
            skip_green_count=skip,
        ))

    # D2: Green bar with minimum profit
    for min_p in [5, 8, 10, 15]:
        variants.append(Config(
            label=f"D2: Green exit only if +{min_p}pts",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="green_with_profit",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
            min_profit_for_green=min_p,
        ))

    # D3: ATR-scaled targets
    for mult in [2.0, 3.0, 4.0]:
        variants.append(Config(
            label=f"D3: {mult}x ATR target",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="atr_target",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
            atr_target_mult=mult,
        ))

    # D4: Early time stop (14:00 instead of 15:55)
    for t in [1300, 1400]:
        variants.append(Config(
            label=f"D4: Time stop at {t}",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="time_stop_early",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
            early_time_stop=t,
        ))

    # D5: Trailing stop
    for trigger, distance in [(0.5, 1.0), (1.0, 1.0), (1.0, 1.5), (1.5, 2.0)]:
        variants.append(Config(
            label=f"D5: Trail (trigger {trigger}x, dist {distance}x ATR)",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="trailing",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
            trail_trigger_atr=trigger, trail_distance_atr=distance,
        ))

    # D6: Momentum exit (vol_ratio fade)
    for floor in [0.8, 1.0, 1.2]:
        variants.append(Config(
            label=f"D6: Momentum exit (vol_ratio < {floor})",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="momentum_exit",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
            momentum_vol_floor=floor,
        ))

    # D7: Hold all day — flatten at 15:55
    variants.append(Config(
        label="D7: Hold ALL DAY (flatten 15:55)",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="hold_all_day",
        stop_bps=30, max_hold_bars=200, es_move_filter=-0.002,
    ))

    # D8: Longer max hold with green bar
    for hold in [12, 18, 24, 36]:
        variants.append(Config(
            label=f"D8: Green bar exit, max hold {hold*5}min",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="green_bar",
            stop_bps=30, max_hold_bars=hold, es_move_filter=-0.002,
        ))

    # D9: Green bar + trailing stop hybrid
    for trigger, distance in [(1.0, 1.5), (1.5, 2.0)]:
        variants.append(Config(
            label=f"D9: Green+Trail (trig {trigger}x, dist {distance}x)",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="green_bar_or_trail",
            stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
            trail_trigger_atr=trigger, trail_distance_atr=distance,
        ))

    # ══════════════════════════════════════════════════════
    #  E. STOP VARIANTS
    # ══════════════════════════════════════════════════════

    for bps in [20, 25, 35, 40, 50]:
        variants.append(Config(
            label=f"E: Stop {bps}bps",
            signal_mode="daily_spike", spike_threshold=0.07,
            entry_mode="red_bar", exit_mode="green_bar",
            stop_bps=bps, max_hold_bars=9, es_move_filter=-0.002,
        ))

    # ══════════════════════════════════════════════════════
    #  F. FILTER VARIANTS
    # ══════════════════════════════════════════════════════

    # F1: Skip gap-up days
    variants.append(Config(
        label="F1: Skip gap-up days",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        skip_gap_up=True,
    ))

    # F2: Require ES below prev session low
    variants.append(Config(
        label="F2: ES below prev session low",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        require_below_prev_low=True,
    ))

    # F3: Morning only (9:35-12:00)
    variants.append(Config(
        label="F3: Morning only (9:35-12:00)",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        morning_only=True,
    ))

    # F4: TEMA trend filter (close < TEMA 55)
    variants.append(Config(
        label="F4: TEMA trend filter (close < T55)",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        tema_trend_filter=True,
    ))

    # F5: Skip Friday
    variants.append(Config(
        label="F5: Skip Friday",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        skip_friday=True,
    ))

    # F6: Skip Monday
    variants.append(Config(
        label="F6: Skip Monday",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        skip_monday=True,
    ))

    # F7: Remove ES move filter entirely (just VIX spike + red bar)
    variants.append(Config(
        label="F7: No ES move filter",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=0.0,
    ))

    return variants


# ────────────────────────────────────────────────────────
#  MAIN — RUN ALL VARIANTS
# ────────────────────────────────────────────────────────

def print_divider(char="=", width=120):
    print(char * width)


def main():
    df, vix_lookup, session_opens, prev_session_data = load_data()

    variants = build_variants()
    print(f"\nTesting {len(variants)} structural variants...\n")

    # ── PHASE 1: Screen all variants (full period) ──
    print_divider("#")
    print("  PHASE 1: FULL-PERIOD SCREEN")
    print_divider("#")

    header = (f"{'#':>3} {'Trades':>6} {'WR':>6} {'PF':>7} {'Net P&L':>10} "
              f"{'Sharpe':>7} {'DD':>8} {'p-val':>8} | {'Label'}")
    print(header)
    print_divider("-")

    results = []
    for idx, cfg in enumerate(variants):
        trades = run_backtest(df, vix_lookup, session_opens, prev_session_data, cfg)
        m = compute_metrics(trades)
        p = t_test(trades)

        results.append((cfg, trades, m, p))

        flag = ""
        if m["total"] >= 10 and m["pf"] > 1.5 and p < 0.05:
            flag = " ***"
        elif m["total"] >= 10 and m["pf"] > 1.0 and p < 0.10:
            flag = " **"
        elif m["total"] >= 8 and m["pf"] > 1.0:
            flag = " *"

        print(f"{idx:>3} {m['total']:>6} {m['win_rate']:>5.1f}% {m['pf']:>7.3f} "
              f"${m['net_pnl']:>9,.0f} {m['sharpe']:>7.2f} ${m['max_dd']:>7,.0f} "
              f"{p:>8.4f} | {cfg.label}{flag}")

    # ── PHASE 2: Rank candidates ──
    print(f"\n")
    print_divider("#")
    print("  PHASE 2: RANKED CANDIDATES (min 10 trades, PF > 1.0, p < 0.10)")
    print_divider("#")

    candidates = [(cfg, trades, m, p) for cfg, trades, m, p in results
                   if m["total"] >= 10 and m["pf"] > 1.0 and p < 0.10]

    # Sort by net P&L (primary), then PF (secondary)
    candidates.sort(key=lambda x: (-x[2]["net_pnl"]))

    print(f"\n{'Rank':>4} {'Trades':>6} {'WR':>6} {'PF':>7} {'Net P&L':>10} "
          f"{'Sharpe':>7} {'p-val':>8} | {'Label'}")
    print_divider("-")

    for rank, (cfg, trades, m, p) in enumerate(candidates[:30], 1):
        beat = ""
        if m["net_pnl"] > 23047:
            beat = " ^^BEATS BASELINE P&L"
        elif m["pf"] > 2.641:
            beat = " ^^BEATS BASELINE PF"
        print(f"{rank:>4} {m['total']:>6} {m['win_rate']:>5.1f}% {m['pf']:>7.3f} "
              f"${m['net_pnl']:>9,.0f} {m['sharpe']:>7.2f} {p:>8.4f} | {cfg.label}{beat}")

    # ── PHASE 3: Walk-forward validation on top candidates ──
    print(f"\n")
    print_divider("#")
    print("  PHASE 3: WALK-FORWARD VALIDATION (top candidates)")
    print_divider("#")

    validated = []
    # Test top 15 by P&L, plus any that beat baseline PF
    top_set = candidates[:15]
    # Also add any with PF > baseline that aren't already in top_set
    pf_beats = [x for x in candidates if x[2]["pf"] > 2.641 and x not in top_set]
    test_set = top_set + pf_beats[:5]

    for cfg, trades_full, m_full, p_full in test_set:
        trades_is, trades_oos, m_is, m_oos = walk_forward(
            df, vix_lookup, session_opens, prev_session_data, cfg)

        p_oos = t_test(trades_oos) if trades_oos else 1.0

        # WF ratio
        if m_is["total"] > 0 and m_oos["total"] > 0 and m_is["pf"] > 0:
            wf_ratio = m_oos["pf"] / m_is["pf"]
        else:
            wf_ratio = 0

        # Pass criteria
        standard_pass = (p_full < 0.05 and m_full["pf"] > 1.0 and wf_ratio >= 0.7)
        oos_pass = (m_oos["total"] >= 5 and m_oos["pf"] > 1.0 and p_oos < 0.10)
        passed = standard_pass or (oos_pass and p_full < 0.05)

        status = "PASS" if passed else "FAIL"

        print(f"\n  {cfg.label}")
        print(f"    FULL:  {m_full['total']} trades | PF {m_full['pf']:.3f} | "
              f"${m_full['net_pnl']:,.0f} | p={p_full:.4f}")
        print(f"    IS:    {m_is['total']} trades | PF {m_is['pf']:.3f} | "
              f"${m_is['net_pnl']:,.0f}")
        print(f"    OOS:   {m_oos['total']} trades | PF {m_oos['pf']:.3f} | "
              f"${m_oos['net_pnl']:,.0f} | p={p_oos:.4f}")
        print(f"    WF ratio: {wf_ratio:.3f} | {status}")

        if passed:
            validated.append({
                "cfg": cfg,
                "m_full": m_full,
                "m_is": m_is,
                "m_oos": m_oos,
                "p_full": p_full,
                "p_oos": p_oos,
                "wf_ratio": wf_ratio,
                "trades_full": trades_full,
            })

    # ── PHASE 4: Deep validation on PASSED variants ──
    print(f"\n")
    print_divider("#")
    print("  PHASE 4: DEEP VALIDATION — PASSED VARIANTS")
    print_divider("#")

    if not validated:
        print("\n  No variants passed walk-forward. Reporting best available.\n")
        # Show the best attempt anyway
        if candidates:
            cfg, trades, m, p = candidates[0]
            trades_is, trades_oos, m_is, m_oos = walk_forward(
                df, vix_lookup, session_opens, prev_session_data, cfg)
            wf_ratio = m_oos["pf"] / m_is["pf"] if m_is["pf"] > 0 else 0
            print(f"  Best attempt: {cfg.label}")
            print(f"    FULL: {m['total']} trades | PF {m['pf']:.3f} | ${m['net_pnl']:,.0f} | p={p:.4f}")
            print(f"    WF ratio: {wf_ratio:.3f}")
    else:
        # Sort validated by net P&L
        validated.sort(key=lambda x: -x["m_full"]["net_pnl"])

        for rank, v in enumerate(validated, 1):
            cfg = v["cfg"]
            m = v["m_full"]
            trades = v["trades_full"]
            p = v["p_full"]
            m_oos = v["m_oos"]

            print(f"\n  {'='*100}")
            print(f"  #{rank}: {cfg.label}")
            print(f"  {'='*100}")

            print(f"\n  FULL PERIOD:")
            print(f"    Trades: {m['total']} | Winners: {m['winners']} | Losers: {m['losers']}")
            print(f"    Win Rate: {m['win_rate']:.1f}%")
            print(f"    Profit Factor: {m['pf']:.3f}")
            print(f"    Net P&L: ${m['net_pnl']:,.0f}")
            print(f"    Avg Trade: ${m['avg_trade']:,.0f}")
            print(f"    Max DD: ${m['max_dd']:,.0f}")
            print(f"    Sharpe: {m['sharpe']:.2f}")

            print(f"\n  SIGNIFICANCE:")
            perm_p = permutation_test(trades)
            boot_p = bootstrap_profit_prob(trades)
            print(f"    t-test p (one-sided):  {p:.6f} {'***' if p < 0.01 else '**' if p < 0.05 else ''}")
            print(f"    Permutation p:         {perm_p:.6f}")
            print(f"    Bootstrap P(profit):   {boot_p:.2%}")

            print(f"\n  WALK-FORWARD (split: {WF_SPLIT}):")
            print(f"    IS:  {v['m_is']['total']} trades | PF {v['m_is']['pf']:.3f} | ${v['m_is']['net_pnl']:,.0f}")
            print(f"    OOS: {v['m_oos']['total']} trades | PF {v['m_oos']['pf']:.3f} | ${v['m_oos']['net_pnl']:,.0f} | p={v['p_oos']:.4f}")
            print(f"    WF ratio: {v['wf_ratio']:.3f}")

            # Exit reasons
            reasons = {}
            for t in trades:
                reasons.setdefault(t.exit_reason, {"count": 0, "pnl": 0})
                reasons[t.exit_reason]["count"] += 1
                reasons[t.exit_reason]["pnl"] += t.pnl_dollar
            print(f"\n  EXIT REASONS:")
            for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
                print(f"    {r:<16} {d['count']:>4} trades  ${d['pnl']:>+10,.0f}")

            # Last 10 trades
            print(f"\n  LAST 10 TRADES:")
            for t in trades[-10:]:
                sign = "+" if t.pnl_dollar > 0 else ""
                print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} -> "
                      f"{t.exit_time.strftime('%H:%M')} | {t.exit_reason:<14} | "
                      f"{sign}${t.pnl_dollar:,.0f} | VIX {t.vix_open:.1f}->{t.vix_high:.1f}")

            # vs baseline comparison
            print(f"\n  VS BASELINE (107 trades, PF 2.641, $23,047):")
            pnl_delta = m["net_pnl"] - 23047
            trade_delta = m["total"] - 107
            pf_delta = m["pf"] - 2.641
            print(f"    Trades: {trade_delta:+d} ({m['total']})")
            print(f"    PF:     {pf_delta:+.3f} ({m['pf']:.3f})")
            print(f"    P&L:    ${pnl_delta:+,.0f} (${m['net_pnl']:,.0f})")

    # ── PHASE 5: Combined best-of-breed ──
    print(f"\n")
    print_divider("#")
    print("  PHASE 5: COMBINED BEST-OF-BREED")
    print_divider("#")

    # Build combinations from the best individual improvements
    combos = []

    # Combo 1: Lower threshold (5%) + all other baseline params
    # This tests: can we get more trades with a slightly lower bar?
    combos.append(Config(
        label="COMBO1: 5% spike + baseline",
        signal_mode="daily_spike", spike_threshold=0.05,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
    ))

    # Combo 2: Overnight OR daily + skip gap up
    combos.append(Config(
        label="COMBO2: Overnight 5% OR daily 7% + skip gap up",
        signal_mode="overnight_or_daily", spike_threshold=0.07, overnight_jump_pct=0.05,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        skip_gap_up=True,
    ))

    # Combo 3: Baseline signal + delayed green exit (hold longer on trend days)
    combos.append(Config(
        label="COMBO3: Baseline + skip 1 green bar",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="delayed_green",
        stop_bps=30, max_hold_bars=18, es_move_filter=-0.002,
        skip_green_count=1,
    ))

    # Combo 4: Baseline + longer hold + wider stop
    combos.append(Config(
        label="COMBO4: Baseline + 90min hold + 40bps stop",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=40, max_hold_bars=18, es_move_filter=-0.002,
    ))

    # Combo 5: 5% spike + 2 entries/day + skip gap up
    combos.append(Config(
        label="COMBO5: 5% spike + 2 entries/day + skip gap up",
        signal_mode="daily_spike", spike_threshold=0.05,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        max_entries_per_day=2, skip_gap_up=True,
    ))

    # Combo 6: Overnight jump + TEMA trend filter
    combos.append(Config(
        label="COMBO6: Overnight 5% + TEMA trend filter",
        signal_mode="overnight_jump", overnight_jump_pct=0.05,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        tema_trend_filter=True,
    ))

    # Combo 7: 5% spike + green with profit (hold for bigger moves)
    combos.append(Config(
        label="COMBO7: 5% spike + green exit only if +8pts",
        signal_mode="daily_spike", spike_threshold=0.05,
        entry_mode="red_bar", exit_mode="green_with_profit",
        stop_bps=30, max_hold_bars=18, es_move_filter=-0.002,
        min_profit_for_green=8.0,
    ))

    # Combo 8: Union signal + morning only
    combos.append(Config(
        label="COMBO8: Overnight 3% OR daily 7% + morning only",
        signal_mode="overnight_or_daily", spike_threshold=0.07, overnight_jump_pct=0.03,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        morning_only=True,
    ))

    # Combo 9: Baseline + 2 entries + wider hold
    combos.append(Config(
        label="COMBO9: 7% spike + 2 entries + 90min hold",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=18, es_move_filter=-0.002,
        max_entries_per_day=2,
    ))

    # Combo 10: 5% spike + ES move -0.1% (less restrictive) + 2 entries
    combos.append(Config(
        label="COMBO10: 5% spike + ES -0.1% + 2 entries",
        signal_mode="daily_spike", spike_threshold=0.05,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.001,
        max_entries_per_day=2,
    ))

    # Combo 11: Overnight 3% + daily 5% union + lower ES filter + skip gap up
    combos.append(Config(
        label="COMBO11: ON 3% OR daily 5% + ES -0.1% + skip gap up",
        signal_mode="overnight_or_daily", spike_threshold=0.05, overnight_jump_pct=0.03,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.001,
        skip_gap_up=True,
    ))

    # Combo 12: Hold all day with tighter signal
    combos.append(Config(
        label="COMBO12: 7% spike + hold all day",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="hold_all_day",
        stop_bps=30, max_hold_bars=200, es_move_filter=-0.002,
    ))

    # Combo 13: Lower spike + TEMA confirm + skip gap up
    combos.append(Config(
        label="COMBO13: 5% spike + TEMA confirm + skip gap up",
        signal_mode="daily_spike", spike_threshold=0.05,
        entry_mode="tema_confirm", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
        skip_gap_up=True,
    ))

    # Combo 14: Multi-entry with trailing stop
    combos.append(Config(
        label="COMBO14: 7% spike + 2 entries + trailing (1x trig, 1.5x dist)",
        signal_mode="daily_spike", spike_threshold=0.07,
        entry_mode="red_bar", exit_mode="trailing",
        stop_bps=30, max_hold_bars=18, es_move_filter=-0.002,
        max_entries_per_day=2,
        trail_trigger_atr=1.0, trail_distance_atr=1.5,
    ))

    # Combo 15: Best signal expansion attempt
    combos.append(Config(
        label="COMBO15: 5% spike + ES -0.15% + 2 entries + longer hold",
        signal_mode="daily_spike", spike_threshold=0.05,
        entry_mode="red_bar", exit_mode="green_bar",
        stop_bps=30, max_hold_bars=18, es_move_filter=-0.0015,
        max_entries_per_day=2,
    ))

    print(f"\nTesting {len(combos)} combined configurations...\n")
    print(f"{'#':>3} {'Trades':>6} {'WR':>6} {'PF':>7} {'Net P&L':>10} "
          f"{'Sharpe':>7} {'p-val':>8} {'WF':>6} | {'Label'}")
    print_divider("-")

    combo_results = []
    for idx, cfg in enumerate(combos):
        trades = run_backtest(df, vix_lookup, session_opens, prev_session_data, cfg)
        m = compute_metrics(trades)
        p = t_test(trades)

        # Quick WF
        _, _, m_is, m_oos = walk_forward(
            df, vix_lookup, session_opens, prev_session_data, cfg)
        wf = m_oos["pf"] / m_is["pf"] if m_is["pf"] > 0 and m_oos["total"] > 0 else 0

        combo_results.append((cfg, trades, m, p, wf))

        flag = ""
        if m["pf"] > 2.641:
            flag = " ^^PF"
        if m["net_pnl"] > 23047:
            flag += " ^^P&L"
        if wf >= 0.7 and p < 0.05:
            flag += " PASS"

        print(f"{idx:>3} {m['total']:>6} {m['win_rate']:>5.1f}% {m['pf']:>7.3f} "
              f"${m['net_pnl']:>9,.0f} {m['sharpe']:>7.2f} {p:>8.4f} {wf:>6.3f} | {cfg.label}{flag}")

    # Deep validate any combos that pass
    combo_passed = [(c, t, m, p, w) for c, t, m, p, w in combo_results
                     if m["total"] >= 10 and p < 0.05 and w >= 0.7 and m["pf"] > 1.0]

    if combo_passed:
        combo_passed.sort(key=lambda x: -x[2]["net_pnl"])
        print(f"\n  COMBO WINNERS ({len(combo_passed)} passed all criteria):")
        for cfg, trades, m, p, wf in combo_passed:
            perm = permutation_test(trades)
            boot = bootstrap_profit_prob(trades)
            print(f"\n    {cfg.label}")
            print(f"    {m['total']} trades | PF {m['pf']:.3f} | ${m['net_pnl']:,.0f} | "
                  f"Sharpe {m['sharpe']:.2f}")
            print(f"    t-test: {p:.4f} | perm: {perm:.4f} | bootstrap: {boot:.2%} | WF: {wf:.3f}")

            # vs baseline
            pnl_d = m["net_pnl"] - 23047
            pf_d = m["pf"] - 2.641
            td = m["total"] - 107
            print(f"    vs BASELINE: trades {td:+d}, PF {pf_d:+.3f}, P&L ${pnl_d:+,.0f}")

    # ── FINAL SUMMARY ──
    print(f"\n")
    print_divider("=")
    print("  FINAL SUMMARY")
    print_divider("=")

    # Collect all passing results
    all_passed = []
    for v in validated:
        all_passed.append({
            "label": v["cfg"].label,
            "trades": v["m_full"]["total"],
            "pf": v["m_full"]["pf"],
            "pnl": v["m_full"]["net_pnl"],
            "sharpe": v["m_full"]["sharpe"],
            "p": v["p_full"],
            "wf": v["wf_ratio"],
            "source": "individual",
        })
    for cfg, trades, m, p, wf in combo_passed:
        all_passed.append({
            "label": cfg.label,
            "trades": m["total"],
            "pf": m["pf"],
            "pnl": m["net_pnl"],
            "sharpe": m["sharpe"],
            "p": p,
            "wf": wf,
            "source": "combo",
        })

    all_passed.sort(key=lambda x: -x["pnl"])

    print(f"\n  BASELINE: 107 trades | PF 2.641 | $23,047 | Sharpe 4.76 | p=0.0012 | WF 1.119")
    print(f"\n  VARIANTS THAT PASSED (p<0.05, WF>=0.7, PF>1.0):")

    if all_passed:
        for r in all_passed:
            beat = []
            if r["pnl"] > 23047:
                beat.append("P&L")
            if r["pf"] > 2.641:
                beat.append("PF")
            if r["trades"] > 107:
                beat.append("TRADES")
            beat_str = f" [BEATS: {', '.join(beat)}]" if beat else ""

            print(f"    {r['label']}")
            print(f"      {r['trades']} trades | PF {r['pf']:.3f} | ${r['pnl']:,.0f} | "
                  f"Sharpe {r['sharpe']:.2f} | p={r['p']:.4f} | WF {r['wf']:.3f}{beat_str}")
    else:
        print("    NONE — baseline is already near-optimal for this signal class.")
        print("    The 7% VIX spike with green bar exit and 30bps stop is the structural sweet spot.")

    print(f"\n  TOTAL VARIANTS TESTED: {len(variants) + len(combos)}")
    print(f"  INDIVIDUAL PASSES: {len(validated)}")
    print(f"  COMBO PASSES: {len(combo_passed)}")
    print_divider("=")


if __name__ == "__main__":
    main()
