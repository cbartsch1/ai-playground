"""Session state — IB tracking, day type classification, VA levels, trade counters.

All state is tracked bar-by-bar via SessionState, matching Pine Script var variables.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .volume_profile import VolumeProfile


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

    # Overnight high/low quality classification (poor vs excess)
    on_bars_near_high: int = 0       # Bars where high >= on_high - threshold
    on_high_bar_count: int = 0       # Total Globex bars
    on_high_is_poor: bool = False    # True = many bars near high (weak, likely to break)
    on_high_is_excess: bool = False  # True = single spike at high (strong, likely to hold)

    # --- Previous day levels (stored on new_rth before reset) ---
    prev_day_high: float = float("nan")
    prev_day_low: float = float("nan")
    prev_day_close: float = float("nan")   # Previous day's RTH close (last bar close)
    prev_day_range: float = float("nan")    # Full RTH range of previous day
    day_range_avg: float = float("nan")     # Rolling 20-day EMA of daily RTH range
    prev_day_is_wide: bool = False          # Previous day range > 1.2x average
    prev_day_is_narrow: bool = False        # Previous day range < 0.8x average

    # --- Overnight Sweep / Gap state ---
    last_rth_close: float = float("nan")    # Running: close of most recent RTH bar (used to set prev_day_close)
    gap_size: float = 0.0                   # RTH open - prev_close (positive = gap up)
    gap_direction: int = 0                  # +1 gap up, -1 gap down, 0 no meaningful gap
    opening_print_high: float = float("nan")  # High of first RTH bar (stop reference for shorts)
    opening_print_low: float = float("nan")   # Low of first RTH bar (stop reference for longs)
    on_above_prev_close: bool = False       # Did ON high trade above prev close? (overhead supply)
    on_below_prev_close: bool = False       # Did ON low trade below prev close? (demand below)
    os_trades: int = 0                      # Overnight Sweep trade counter

    # --- Level state tracking (reset daily) ---
    lvl_test_count: dict = field(default_factory=dict)    # {"PDH": 2, "ONH": 1, ...}
    lvl_broken: dict = field(default_factory=dict)        # {"PDH": False, "ONH": True, ...}
    lvl_broken_count: dict = field(default_factory=dict)  # {"PDH": 0, ...} — consecutive closes above level
    lvl_in_zone: dict = field(default_factory=dict)       # {"PDH": True, ...} — was bar in zone last bar?
    lvl_first_test_bar: dict = field(default_factory=dict)  # {"PDH": 42, ...} — bar index of first test
    lvl_trades_s: int = 0                                 # Shared trade counter across all SHORT level rejection entries
    lvl_trades_l: int = 0                                 # Shared trade counter across all LONG level rejection entries
    lvl_bar_index: int = 0                                # Bar counter within RTH session

    # Absorption tracking: bar range/volume while in zone (for quality filter)
    lvl_zone_ranges: dict = field(default_factory=dict)   # {"ONH": [12.5, 8.0, 6.25]} — ranges of bars at level
    lvl_zone_volumes: dict = field(default_factory=dict)  # {"ONH": [1250, 1680, 990]} — volumes of bars at level

    # Support level state tracking (for longs — mirrors resistance tracking)
    sup_test_count: dict = field(default_factory=dict)    # {"PDL": 2, "ONL": 1, ...}
    sup_broken: dict = field(default_factory=dict)        # {"PDL": True, ...} — broken = close below level
    sup_broken_count: dict = field(default_factory=dict)  # consecutive closes below level
    sup_in_zone: dict = field(default_factory=dict)       # zone presence for dedup

    # --- TPO Day Type Classification ---
    # One-Time-Framing (OTF): consecutive 30-min periods making new directional extremes
    otf_up_count: int = 0          # consecutive periods with higher high AND higher low
    otf_down_count: int = 0        # consecutive periods with lower low AND lower high
    otf_max_up: int = 0            # max consecutive OTF-up periods today
    otf_max_down: int = 0          # max consecutive OTF-down periods today
    prev_period_high: float = float("nan")  # previous 30-min period's high
    prev_period_low: float = float("nan")   # previous 30-min period's low

    # Previous day classification (computed at new_rth from yesterday's data)
    prev_day_otf_max_up: int = 0   # yesterday's max OTF-up streak
    prev_day_otf_max_down: int = 0 # yesterday's max OTF-down streak
    prev_day_was_trend: bool = False  # OTF streak >= 4 in either direction
    prev_day_single_print_high: float = float("nan")  # top of thin structure from trend day
    prev_day_single_print_low: float = float("nan")   # bottom of thin structure

    # Developing value area (from VolumeProfile, updated every bar)
    dev_poc: float = float("nan")
    dev_vah: float = float("nan")
    dev_val: float = float("nan")

    # TPO strategy trade counters
    var_trades: int = 0            # Value Area Rotation trades today
    ptf_trades: int = 0           # Post-Trend Fade trades today

    # Volume Profile-derived previous day levels (REAL VA, not VWAP proxy)
    prev_vp_poc: float = float("nan")   # Previous day's actual POC from volume profile
    prev_vp_vah: float = float("nan")   # Previous day's actual VAH (68% value area)
    prev_vp_val: float = float("nan")   # Previous day's actual VAL (68% value area)

    # Market Structure trade tracking
    ms_trades: int = 0                  # Market Structure trades today
    ms_traded_levels: set = field(default_factory=set)  # Levels already traded today

    # --- Failed Auction (FA) state tracking ---
    fa_ib_broken_above: bool = False       # has price closed above IBH today?
    fa_ib_broken_below: bool = False       # has price closed below IBL today?
    fa_break_above_bar: int = 0            # bar index when break above occurred
    fa_break_below_bar: int = 0            # bar index when break below occurred
    fa_break_above_extreme: float = float("nan")  # highest price during upside break
    fa_break_below_extreme: float = float("nan")  # lowest price during downside break
    fa_trades: int = 0                     # trade counter (reset at new_rth)
    fa_short_fired: bool = False           # already took the short side today
    fa_long_fired: bool = False            # already took the long side today

    # --- Volume Profile (multi-day composite) ---
    current_vp: VolumeProfile = field(default_factory=VolumeProfile)
    recent_profiles: list = field(default_factory=list)  # Last N days' finalized profiles
    prev_vp: Optional[VolumeProfile] = None  # Most recent day's profile (backward compat)

    # 30-minute bar aggregator (TPO standard timeframe)
    # Accumulates 5-min bars into 30-min periods before adding to VP
    vp_30m_high: float = float("nan")
    vp_30m_low: float = float("nan")
    vp_30m_vol: float = 0.0
    vp_30m_period: int = -1  # Current 30-min period index (0=930-1000, 1=1000-1030, etc.)

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
        state.on_bars_near_high = 1  # First bar is always "near" the high
        state.on_high_bar_count = 1
    elif is_globex and not state.on_frozen:
        # Accumulate overnight range
        if math.isnan(state.on_high):
            state.on_high = bar["high"]
            state.on_low = bar["low"]
            state.on_bars_near_high = 1
            state.on_high_bar_count = 1
        else:
            # If this bar sets a NEW high, recount bars near the new high
            if bar["high"] > state.on_high:
                state.on_high = bar["high"]
                # Recount not possible without bar history, but we can reset
                # and count this bar (conservative: slightly overcounts "excess")
                state.on_bars_near_high = 1
            state.on_low = min(state.on_low, bar["low"])
            state.on_high_bar_count += 1

            # Count bars near the current high (within 3 pts)
            if not math.isnan(state.on_high) and bar["high"] >= state.on_high - 3.0:
                state.on_bars_near_high += 1

    # ─── New RTH session reset ───
    if new_rth:
        # Freeze overnight levels at RTH open
        state.on_frozen = True

        # Classify overnight high quality (poor vs excess)
        if state.on_high_bar_count > 0:
            near_ratio = state.on_bars_near_high / state.on_high_bar_count
            state.on_high_is_excess = state.on_bars_near_high <= 2
            state.on_high_is_poor = state.on_bars_near_high >= 5
        else:
            state.on_high_is_excess = False
            state.on_high_is_poor = False

        # Store previous day high/low/close BEFORE reset
        if not math.isnan(state.rth_hi):
            state.prev_day_high = state.rth_hi
            state.prev_day_low = state.rth_lo
            # prev_close = close of the last RTH bar (tracked as running value)
            if not math.isnan(state.last_rth_close):
                state.prev_day_close = state.last_rth_close

            # Daily range classification (NR4/WR)
            day_range = state.rth_hi - state.rth_lo
            state.prev_day_range = day_range
            alpha = 2.0 / 21.0  # 20-day EMA
            if math.isnan(state.day_range_avg):
                state.day_range_avg = day_range
            else:
                state.day_range_avg = state.day_range_avg * (1 - alpha) + day_range * alpha
            if state.day_range_avg > 0:
                ratio = day_range / state.day_range_avg
                state.prev_day_is_wide = ratio > 1.2
                state.prev_day_is_narrow = ratio < 0.8
            else:
                state.prev_day_is_wide = False
                state.prev_day_is_narrow = False

        # Store previous session VA before reset
        if state.rth_vol_sum > 0:
            session_vwap = state.rth_vwap_sum / state.rth_vol_sum
            session_stdev = math.sqrt(state.rth_sq_dev / state.rth_bars) if state.rth_bars > 1 else 0.0
            state.prev_poc = session_vwap
            state.prev_vah = session_vwap + session_stdev * cfg.va_stdev_mult
            state.prev_val = session_vwap - session_stdev * cfg.va_stdev_mult

        # Store OTF classification before reset (for next day's PTF strategy)
        # Also handle last completed 30-min period's OTF
        if state.vp_30m_period >= 0 and not math.isnan(state.vp_30m_high):
            if not math.isnan(state.prev_period_high):
                hh = state.vp_30m_high > state.prev_period_high
                hl = state.vp_30m_low > state.prev_period_low
                ll = state.vp_30m_low < state.prev_period_low
                lh = state.vp_30m_high < state.prev_period_high
                if hh and hl:
                    state.otf_up_count += 1
                    state.otf_down_count = 0
                elif ll and lh:
                    state.otf_down_count += 1
                    state.otf_up_count = 0
                else:
                    state.otf_up_count = 0
                    state.otf_down_count = 0
                state.otf_max_up = max(state.otf_max_up, state.otf_up_count)
                state.otf_max_down = max(state.otf_max_down, state.otf_down_count)

        state.prev_day_otf_max_up = state.otf_max_up
        state.prev_day_otf_max_down = state.otf_max_down
        state.prev_day_was_trend = max(state.otf_max_up, state.otf_max_down) >= 4

        # Flush last 30-min bar before finalizing VP
        if state.vp_30m_period >= 0 and not math.isnan(state.vp_30m_high):
            state.current_vp.add_bar(state.vp_30m_high, state.vp_30m_low, state.vp_30m_vol)
        state.vp_30m_high = float("nan")
        state.vp_30m_low = float("nan")
        state.vp_30m_vol = 0.0
        state.vp_30m_period = -1

        # Finalize and rotate volume profile (multi-day rolling window)
        if not state.current_vp.is_empty():
            state.current_vp.finalize()
            state.prev_vp = state.current_vp

            # Extract REAL VP-derived levels (not VWAP proxy)
            vp_poc = state.current_vp.get_poc()
            if vp_poc is not None:
                state.prev_vp_poc = vp_poc
            vp_va = state.current_vp.get_value_area(0.68)  # Freeman's 68%
            if vp_va is not None:
                state.prev_vp_val, state.prev_vp_vah = vp_va

            # Single print detection on trend days
            if state.prev_day_was_trend:
                _detect_single_prints(state, state.current_vp)
            else:
                state.prev_day_single_print_high = float("nan")
                state.prev_day_single_print_low = float("nan")

            state.recent_profiles.append(state.current_vp)
            # Keep only last 5 days
            if len(state.recent_profiles) > 5:
                state.recent_profiles = state.recent_profiles[-5:]
        else:
            state.prev_day_single_print_high = float("nan")
            state.prev_day_single_print_low = float("nan")
        state.current_vp = VolumeProfile()

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
        state.var_trades = 0
        state.ptf_trades = 0
        state.ms_trades = 0
        state.ms_traded_levels = set()

        # Reset Failed Auction state
        state.fa_ib_broken_above = False
        state.fa_ib_broken_below = False
        state.fa_break_above_bar = 0
        state.fa_break_below_bar = 0
        state.fa_break_above_extreme = float("nan")
        state.fa_break_below_extreme = float("nan")
        state.fa_trades = 0
        state.fa_short_fired = False
        state.fa_long_fired = False

        # Reset OTF counters
        state.otf_up_count = 0
        state.otf_down_count = 0
        state.otf_max_up = 0
        state.otf_max_down = 0
        state.prev_period_high = float("nan")
        state.prev_period_low = float("nan")

        # Reset developing VA
        state.dev_poc = float("nan")
        state.dev_vah = float("nan")
        state.dev_val = float("nan")

        # Reset level state tracking (resistance — shorts)
        state.lvl_test_count = {}
        state.lvl_broken = {}
        state.lvl_broken_count = {}
        state.lvl_in_zone = {}
        state.lvl_first_test_bar = {}
        state.lvl_bar_index = 0
        state.lvl_zone_ranges = {}
        state.lvl_zone_volumes = {}

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

        # Opening print (first RTH bar's extremes — stop reference for gap fades)
        state.opening_print_high = bar["high"]
        state.opening_print_low = bar["low"]

        # Gap detection: open vs prev_close
        if not math.isnan(state.prev_day_close):
            state.gap_size = bar["open"] - state.prev_day_close
            if state.gap_size > 2.0:  # meaningful gap up (> 2 pts)
                state.gap_direction = 1
            elif state.gap_size < -2.0:  # meaningful gap down
                state.gap_direction = -1
            else:
                state.gap_direction = 0
        else:
            state.gap_size = 0.0
            state.gap_direction = 0

        # Overnight sweep: did overnight session trade above/below prev close?
        if not math.isnan(state.prev_day_close) and not math.isnan(state.on_high):
            state.on_above_prev_close = state.on_high > state.prev_day_close
            state.on_below_prev_close = state.on_low < state.prev_day_close
        else:
            state.on_above_prev_close = False
            state.on_below_prev_close = False

        # Reset overnight sweep trade counter
        state.os_trades = 0

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

    # ─── VWAP + Volume Profile accumulation (RTH only) ───
    if bar["is_rth"]:
        vol = bar["volume"] if bar["volume"] > 0 else 1  # avoid division by zero
        state.rth_vwap_sum += bar["hlc3"] * vol
        state.rth_vol_sum += vol

        # 30-minute TPO aggregation for volume profile
        # Market Profile standard: each 30-min period = one TPO letter
        # Periods: 930-1000=0, 1000-1030=1, 1030-1100=2, ..., 1530-1600=13
        et_time = bar.get("et_time", 0)
        if et_time >= 930:
            minutes_from_open = (et_time // 100 - 9) * 60 + (et_time % 100) - 30
            current_period = minutes_from_open // 30
        else:
            current_period = -1

        if current_period >= 0:
            if current_period != state.vp_30m_period:
                # Period boundary crossed — finalize previous 30-min bar
                if state.vp_30m_period >= 0 and not math.isnan(state.vp_30m_high):
                    state.current_vp.add_bar(state.vp_30m_high, state.vp_30m_low, state.vp_30m_vol)

                    # OTF detection: compare completed period with previous
                    if not math.isnan(state.prev_period_high):
                        hh = state.vp_30m_high > state.prev_period_high
                        hl = state.vp_30m_low > state.prev_period_low
                        ll = state.vp_30m_low < state.prev_period_low
                        lh = state.vp_30m_high < state.prev_period_high

                        if hh and hl:
                            state.otf_up_count += 1
                            state.otf_down_count = 0
                        elif ll and lh:
                            state.otf_down_count += 1
                            state.otf_up_count = 0
                        else:
                            state.otf_up_count = 0
                            state.otf_down_count = 0

                        state.otf_max_up = max(state.otf_max_up, state.otf_up_count)
                        state.otf_max_down = max(state.otf_max_down, state.otf_down_count)

                    state.prev_period_high = state.vp_30m_high
                    state.prev_period_low = state.vp_30m_low

                # Start new period
                state.vp_30m_period = current_period
                state.vp_30m_high = bar["high"]
                state.vp_30m_low = bar["low"]
                state.vp_30m_vol = vol
            else:
                # Same period — accumulate
                if math.isnan(state.vp_30m_high):
                    state.vp_30m_high = bar["high"]
                    state.vp_30m_low = bar["low"]
                else:
                    state.vp_30m_high = max(state.vp_30m_high, bar["high"])
                    state.vp_30m_low = min(state.vp_30m_low, bar["low"])
                state.vp_30m_vol += vol

        rth_hi = state.rth_hi if not math.isnan(state.rth_hi) else bar["high"]
        rth_lo = state.rth_lo if not math.isnan(state.rth_lo) else bar["low"]
        state.rth_hi = max(rth_hi, bar["high"])
        state.rth_lo = min(rth_lo, bar["low"])

        # Track last RTH close (used for prev_day_close at next session reset)
        state.last_rth_close = bar["close"]

        # Running stdev from VWAP
        if state.rth_vol_sum > 0:
            current_vwap = state.rth_vwap_sum / state.rth_vol_sum
            state.rth_sq_dev += (bar["close"] - current_vwap) ** 2
            state.rth_bars += 1

        # Developing value area from today's volume profile
        if not state.current_vp.is_empty():
            poc = state.current_vp.get_poc()
            if poc is not None:
                state.dev_poc = poc
            va = state.current_vp.get_value_area(0.68)
            if va is not None:
                state.dev_val, state.dev_vah = va

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


def _detect_single_prints(state: SessionState, vp: VolumeProfile) -> None:
    """Detect single print zones from a trend day's volume profile.

    Single prints = thin structure where only 1 TPO period visited.
    On trend days, price one-time-frames through these zones, leaving
    thin volume that tends to get revisited (filled) the next day.

    Scans from the direction of the trend:
      - OTF-up trend: scan from bottom up (single prints are in the lower extension)
      - OTF-down trend: scan from top down (single prints are in the upper extension)
    """
    bins = vp.get_bins()
    if not bins:
        return

    volumes = list(bins.values())
    mean_vol = sum(volumes) / len(volumes) if volumes else 0
    thin_cutoff = mean_vol * 0.3

    sorted_bins = sorted(bins.keys())
    if not sorted_bins:
        return

    # Determine scan direction based on which OTF was stronger
    scan_down = state.prev_day_otf_max_up >= state.prev_day_otf_max_down

    if scan_down:
        # OTF-up trend day: single prints are near the top (extension)
        # Scan from top down to find thin structure
        scan_order = sorted_bins[::-1]
    else:
        # OTF-down trend day: single prints are near the bottom (extension)
        # Scan from bottom up to find thin structure
        scan_order = sorted_bins

    sp_bins = []
    for b in scan_order:
        vol = bins.get(b, 0)
        if vol <= thin_cutoff:
            sp_bins.append(b)
        elif sp_bins:
            break  # Stop at first thick structure after finding thin

    if sp_bins:
        bin_size = vp.bin_size
        sp_min = min(sp_bins)
        sp_max = max(sp_bins)
        state.prev_day_single_print_low = sp_min * bin_size
        state.prev_day_single_print_high = (sp_max + 1) * bin_size
    else:
        state.prev_day_single_print_high = float("nan")
        state.prev_day_single_print_low = float("nan")
