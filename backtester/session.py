"""Session state — IB tracking, day type classification, VA levels, trade counters.

All state is tracked bar-by-bar via SessionState, matching Pine Script var variables.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SessionState:
    """Mutable state that persists across bars within a session and resets on new_rth.

    Maps 1:1 to Pine Script `var` variables.
    """

    # --- IB ---
    ib_high: float = float("nan")
    ib_low: float = float("nan")
    ib_done: bool = False

    # --- Day Type ---
    ib_range_avg: float = float("nan")  # Rolling EMA of IB range

    # --- Value Area (previous session) ---
    prev_poc: float = float("nan")
    prev_vah: float = float("nan")
    prev_val: float = float("nan")

    # VWAP accumulators (current RTH session)
    rth_vwap_sum: float = 0.0
    rth_vol_sum: float = 0.0
    rth_sq_dev: float = 0.0
    rth_bars: int = 0

    # RTH session range
    rth_hi: float = float("nan")
    rth_lo: float = float("nan")

    # --- RTH Open ---
    rth_open: float = float("nan")
    open_above_va: bool = False
    open_below_va: bool = False

    # --- Trade counters (reset daily) ---
    ib_trades_l: int = 0
    ib_trades_s: int = 0
    va_trades_l: int = 0
    va_trades_s: int = 0
    eighty_trades_l: int = 0
    eighty_trades_s: int = 0
    tx_trades_s: int = 0
    rej_trades_s: int = 0

    # --- 80% Rule state ---
    eighty_reentered: bool = False
    eighty_inside_count: int = 0
    eighty_confirmed: bool = False

    # --- Overnight levels (6 PM → 9:30 AM, frozen at RTH open) ---
    on_high: float = float("nan")
    on_low: float = float("nan")
    on_frozen: bool = False

    # --- Previous day levels (stored on new_rth before reset) ---
    prev_day_high: float = float("nan")
    prev_day_low: float = float("nan")

    # --- Level state tracking (reset daily) ---
    lvl_test_count: dict = field(default_factory=dict)    # {"PDH": 2, "ONH": 1, ...}
    lvl_broken: dict = field(default_factory=dict)        # {"PDH": False, "ONH": True, ...}
    lvl_broken_count: dict = field(default_factory=dict)  # {"PDH": 0, ...} — consecutive closes above level
    lvl_in_zone: dict = field(default_factory=dict)       # {"PDH": True, ...} — was bar in zone last bar?
    lvl_first_test_bar: dict = field(default_factory=dict)  # {"PDH": 42, ...} — bar index of first test
    lvl_trades_s: int = 0                                 # Shared trade counter across all SHORT level rejection entries
    lvl_trades_l: int = 0                                 # Shared trade counter across all LONG level rejection entries
    lvl_bar_index: int = 0                                # Bar counter within RTH session

    # Support level state tracking (for longs — mirrors resistance tracking)
    sup_test_count: dict = field(default_factory=dict)    # {"PDL": 2, "ONL": 1, ...}
    sup_broken: dict = field(default_factory=dict)        # {"PDL": True, ...} — broken = close below level
    sup_broken_count: dict = field(default_factory=dict)  # consecutive closes below level
    sup_in_zone: dict = field(default_factory=dict)       # zone presence for dedup

    # --- Cooldown ---
    bars_since_exit: int = 100

    # --- Derived (recomputed each bar) ---
    ib_range: float = 0.0
    ib_mid: float = float("nan")

    # Day type
    ib_ratio: float = 1.0
    is_narrow_ib: bool = False
    is_wide_ib: bool = False
    is_normal_ib: bool = True

    # VA position
    above_va: bool = False
    below_va: bool = False
    inside_va: bool = False

    # Vol filter
    vol_ok: bool = True


def update_session(state: SessionState, bar: dict, prev_bar: Optional[dict],
                   cfg) -> None:
    """Update session state for current bar. Call once per bar in sequence.

    bar: dict with keys from DataFrame row (open, high, low, close, volume,
         is_rth, is_ib_period, is_trading_window, new_rth, hlc3, et_time, etc.)
    prev_bar: previous bar dict (or None for first bar)
    cfg: StrategyConfig
    """
    new_rth = bar["new_rth"]

    # ─── Overnight (Globex) tracking ───
    is_globex = bar.get("is_globex", False)
    new_globex = bar.get("new_globex", False)

    if new_globex:
        # New globex session — reset overnight levels
        state.on_high = bar["high"]
        state.on_low = bar["low"]
        state.on_frozen = False
    elif is_globex and not state.on_frozen:
        # Accumulate overnight range
        if math.isnan(state.on_high):
            state.on_high = bar["high"]
            state.on_low = bar["low"]
        else:
            state.on_high = max(state.on_high, bar["high"])
            state.on_low = min(state.on_low, bar["low"])

    # ─── New RTH session reset ───
    if new_rth:
        # Freeze overnight levels at RTH open
        state.on_frozen = True

        # Store previous day high/low BEFORE reset
        if not math.isnan(state.rth_hi):
            state.prev_day_high = state.rth_hi
            state.prev_day_low = state.rth_lo

        # Store previous session VA before reset
        if state.rth_vol_sum > 0:
            session_vwap = state.rth_vwap_sum / state.rth_vol_sum
            session_stdev = math.sqrt(state.rth_sq_dev / state.rth_bars) if state.rth_bars > 1 else 0.0
            state.prev_poc = session_vwap
            state.prev_vah = session_vwap + session_stdev * cfg.va_stdev_mult
            state.prev_val = session_vwap - session_stdev * cfg.va_stdev_mult

        # Reset VWAP accumulators
        state.rth_vwap_sum = 0.0
        state.rth_vol_sum = 0.0
        state.rth_sq_dev = 0.0
        state.rth_bars = 0
        state.rth_hi = bar["high"]
        state.rth_lo = bar["low"]

        # Reset IB
        state.ib_high = bar["high"]
        state.ib_low = bar["low"]
        state.ib_done = False

        # Reset trade counters
        state.ib_trades_l = 0
        state.ib_trades_s = 0
        state.va_trades_l = 0
        state.va_trades_s = 0
        state.eighty_trades_l = 0
        state.eighty_trades_s = 0
        state.tx_trades_s = 0
        state.rej_trades_s = 0
        state.lvl_trades_s = 0
        state.lvl_trades_l = 0

        # Reset level state tracking (resistance — shorts)
        state.lvl_test_count = {}
        state.lvl_broken = {}
        state.lvl_broken_count = {}
        state.lvl_in_zone = {}
        state.lvl_first_test_bar = {}
        state.lvl_bar_index = 0

        # Reset support state tracking (support — longs)
        state.sup_test_count = {}
        state.sup_broken = {}
        state.sup_broken_count = {}
        state.sup_in_zone = {}

        # Reset 80% Rule
        state.eighty_reentered = False
        state.eighty_inside_count = 0
        state.eighty_confirmed = False

        # RTH open
        state.rth_open = bar["open"]

        # Check if open is outside previous VA
        if not math.isnan(state.prev_vah):
            state.open_above_va = bar["open"] > state.prev_vah
        else:
            state.open_above_va = False
        if not math.isnan(state.prev_val):
            state.open_below_va = bar["open"] < state.prev_val
        else:
            state.open_below_va = False

    elif bar["is_ib_period"]:
        # Update IB range during IB period
        ib_h = state.ib_high if not math.isnan(state.ib_high) else bar["high"]
        ib_l = state.ib_low if not math.isnan(state.ib_low) else bar["low"]
        state.ib_high = max(ib_h, bar["high"])
        state.ib_low = min(ib_l, bar["low"])

    elif not state.ib_done and bar["is_rth"] and not bar["is_ib_period"]:
        # First bar after IB period ends
        state.ib_done = True

    # ─── IB Range EMA (updates when IB completes) ───
    prev_ib_done = False
    if prev_bar is not None:
        # Approximate prev ib_done — we track this via the transition
        pass
    # We detect ib_done transition: ib_done is True and was just set this bar
    if state.ib_done and bar["is_rth"] and not bar["is_ib_period"]:
        # Check if this is the first bar after IB (transition bar)
        if prev_bar is not None and prev_bar.get("is_ib_period", False):
            ib_range = state.ib_high - state.ib_low if not math.isnan(state.ib_high) else 0
            if ib_range > 0:
                alpha = 2.0 / (cfg.ib_avg_len + 1.0)
                if math.isnan(state.ib_range_avg):
                    state.ib_range_avg = ib_range
                else:
                    state.ib_range_avg = state.ib_range_avg * (1 - alpha) + ib_range * alpha

    # ─── Bar counter (RTH only) ───
    if bar["is_rth"]:
        state.lvl_bar_index += 1

    # ─── VWAP accumulation (RTH only) ───
    if bar["is_rth"]:
        vol = bar["volume"] if bar["volume"] > 0 else 1  # avoid division by zero
        state.rth_vwap_sum += bar["hlc3"] * vol
        state.rth_vol_sum += vol
        rth_hi = state.rth_hi if not math.isnan(state.rth_hi) else bar["high"]
        rth_lo = state.rth_lo if not math.isnan(state.rth_lo) else bar["low"]
        state.rth_hi = max(rth_hi, bar["high"])
        state.rth_lo = min(rth_lo, bar["low"])

        # Running stdev from VWAP
        if state.rth_vol_sum > 0:
            current_vwap = state.rth_vwap_sum / state.rth_vol_sum
            state.rth_sq_dev += (bar["close"] - current_vwap) ** 2
            state.rth_bars += 1

    # ─── Derived values ───
    if not math.isnan(state.ib_high) and not math.isnan(state.ib_low):
        state.ib_range = state.ib_high - state.ib_low
        state.ib_mid = (state.ib_high + state.ib_low) / 2
    else:
        state.ib_range = 0
        state.ib_mid = bar["close"]

    # Day type
    if not math.isnan(state.ib_range_avg) and state.ib_range_avg > 0:
        state.ib_ratio = state.ib_range / state.ib_range_avg
    else:
        state.ib_ratio = 1.0
    state.is_narrow_ib = state.ib_ratio < cfg.ib_narrow_ratio
    state.is_wide_ib = state.ib_ratio > cfg.ib_wide_ratio
    state.is_normal_ib = not state.is_narrow_ib and not state.is_wide_ib

    # VA position — use fallbacks matching Pine nz()
    prev_vah = state.prev_vah if not math.isnan(state.prev_vah) else bar["close"] + 10
    prev_val = state.prev_val if not math.isnan(state.prev_val) else bar["close"] - 10

    state.above_va = bar["close"] > prev_vah
    state.below_va = bar["close"] < prev_val
    state.inside_va = not state.above_va and not state.below_va

    # ─── 80% Rule counting ───
    if bar["is_rth"] and (state.open_above_va or state.open_below_va) and not state.eighty_confirmed:
        if state.inside_va:
            if not state.eighty_reentered:
                state.eighty_reentered = True
            state.eighty_inside_count += 1
        else:
            state.eighty_inside_count = 0
        if state.eighty_inside_count >= cfg.eighty_conf_bars:
            state.eighty_confirmed = True

    # Vol filter
    vol_ratio = bar.get("vol_ratio", 1.0)
    state.vol_ok = (not cfg.use_vol_filter or
                    (vol_ratio >= cfg.vol_low_ratio and vol_ratio <= cfg.vol_high_ratio))
