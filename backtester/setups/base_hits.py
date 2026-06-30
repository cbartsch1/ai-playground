"""Base Hits — ES Futures Scalping from SPX Options Signals.

Every proven SPX short signal = short ES at market. Fixed or level-aware
stop/target. On target hit, re-evaluate signal. If still valid, re-enter.

SPX Signal Types Detected (all on 30m bars aggregated from 5m):
  1. Structure Break  — lower-high + lower-low after uptrend (HH+HL context)
  2. Bearish Engulfing — prev green, current red body fully engulfs prev body
  3. Trend Continuation — 30m close below prior 30m low (momentum continuation)
  4. EMA Cross          — EMA 8/24 bearish crossover on 5m bars
  5. VIX Spike          — VIX daily high >= open * (1 + threshold), bar is red

Exit: Fixed point stop / Fixed point target / Signal invalidation / 15:55 ET flatten.
On target hit: check if signal still valid. If yes, re-enter immediately.

Interface: check_signal(bar, prev_bar, session, cfg) -> Optional[dict]
Matches the existing ES backtester setup contract exactly.

Session state (VA, POC, IB, ON levels) available for level-aware exits
when cfg.bh_use_level_targets is True.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# 30m Bar Aggregator — accumulates 5m bars into 30m candles
# ============================================================================

@dataclass
class Bar30m:
    """A completed 30m bar."""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    bar_count: int = 0    # number of 5m bars aggregated
    et_start: int = 0     # ET time of first 5m bar (HHMM)
    session_date: object = None


@dataclass
class Aggregator30m:
    """Accumulates 5m bars into 30m bars. Emits completed bars.

    A 30m boundary is at minutes 00 and 30 (i.e., 09:30, 10:00, 10:30, ...).
    When et_time crosses a boundary, the current accumulation finalizes
    and a new period starts.
    """
    current_high: float = float("nan")
    current_low: float = float("nan")
    current_open: float = float("nan")
    current_close: float = float("nan")
    current_count: int = 0
    current_period: int = -1   # 30-min period index (0=930-1000, 1=1000-1030, ...)
    current_et_start: int = 0
    current_session: object = None

    # Completed bar ring buffer (last 3 bars for structure detection)
    completed: list = field(default_factory=list)

    def feed(self, bar: dict) -> Optional[Bar30m]:
        """Feed a 5m bar. Returns a completed 30m bar or None.

        IMPORTANT: The returned bar is the JUST-COMPLETED bar.
        The current bar has already started a new period.
        """
        et_time = bar.get("et_time", 0)
        if et_time < 930 or et_time >= 1600:
            return None

        # Compute period index: 930-959=0, 1000-1029=1, 1030-1059=2, ...
        minutes_from_open = (et_time // 100 - 9) * 60 + (et_time % 100) - 30
        period = minutes_from_open // 30

        session_date = bar.get("session_date", None)

        # Session change = reset
        if session_date != self.current_session and self.current_session is not None:
            completed_bar = self._finalize()
            self._reset()
            self.current_session = session_date
            self._start_new(bar, period, et_time)
            return completed_bar

        self.current_session = session_date

        # Period boundary = finalize old, start new
        if period != self.current_period and self.current_period >= 0:
            completed_bar = self._finalize()
            self._start_new(bar, period, et_time)
            return completed_bar

        # Same period = accumulate
        if self.current_period < 0:
            self._start_new(bar, period, et_time)
        else:
            self._accumulate(bar)

        return None

    def _start_new(self, bar: dict, period: int, et_time: int):
        self.current_period = period
        self.current_open = bar["open"]
        self.current_high = bar["high"]
        self.current_low = bar["low"]
        self.current_close = bar["close"]
        self.current_count = 1
        self.current_et_start = et_time

    def _accumulate(self, bar: dict):
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
            bar_count=self.current_count,
            et_start=self.current_et_start,
            session_date=self.current_session,
        )
        self.completed.append(b)
        # Keep only the last 5 bars (need 2 for engulfing, 3+ for structure)
        if len(self.completed) > 5:
            self.completed = self.completed[-5:]
        return b

    def _reset(self):
        self.current_high = float("nan")
        self.current_low = float("nan")
        self.current_open = float("nan")
        self.current_close = float("nan")
        self.current_count = 0
        self.current_period = -1
        self.current_et_start = 0


# ============================================================================
# Signal State — tracks which signals are active for re-entry
# ============================================================================

@dataclass
class SignalState:
    """Per-session state for Base Hits signal tracking."""
    # Structure break state
    uptrend_count: int = 0           # consecutive HH+HL 30m bars
    had_uptrend: bool = False        # saw at least 1 HH+HL in this session
    structure_break_active: bool = False  # LH+LL detected after uptrend
    sb_bar_range: float = 0.0        # range of the structure break bar (for threshold)

    # Engulfing state
    engulfing_active: bool = False

    # Trend continuation state
    lower_close_active: bool = False

    # EMA cross state (tracked on 5m bars directly)
    ema_8: float = float("nan")
    ema_24: float = float("nan")
    prev_ema_8: float = float("nan")
    prev_ema_24: float = float("nan")
    ema_cross_active: bool = False    # bearish cross fired this session

    # VIX spike state (requires external VIX data or vol_ratio proxy)
    vix_spike_active: bool = False

    # Composite: ANY signal currently active
    any_signal_active: bool = False

    # The setup name of the most recent signal (for trade labeling)
    active_setup: str = ""

    # Trade management
    trades_today: int = 0
    last_entry_bar: int = -100       # bar index of last entry (for min gap)
    bar_index: int = 0               # running 5m bar counter within session

    # Re-entry tracking
    last_target_hit_bar: int = -100  # bar index when target was last hit

    def reset_session(self):
        """Reset all state for a new trading day."""
        self.uptrend_count = 0
        self.had_uptrend = False
        self.structure_break_active = False
        self.sb_bar_range = 0.0
        self.engulfing_active = False
        self.lower_close_active = False
        self.ema_cross_active = False
        self.vix_spike_active = False
        self.any_signal_active = False
        self.active_setup = ""
        self.trades_today = 0
        self.last_entry_bar = -100
        self.bar_index = 0
        self.last_target_hit_bar = -100
        # EMAs do NOT reset (they're continuous)


# ============================================================================
# Module-level persistent state (survives across bars, resets per session)
# ============================================================================

_aggregator = Aggregator30m()
_signal_state = SignalState()
_initialized = False


def _ensure_init():
    """Lazy init — only called once."""
    global _initialized
    if not _initialized:
        _initialized = True


def _reset_for_new_session():
    """Called when new_rth is detected."""
    global _aggregator
    _signal_state.reset_session()
    _aggregator = Aggregator30m()


# ============================================================================
# 30m Signal Detection
# ============================================================================

def _update_structure_break(completed_bar: Bar30m, state: SignalState, cfg):
    """Check for 30m structure break: LH + LL after uptrend.

    Requires at least 2 completed 30m bars. The most recent completed
    bar is checked against its predecessor.
    """
    bars = _aggregator.completed
    if len(bars) < 2:
        return

    curr = bars[-1]
    prev = bars[-2]

    # Same session check
    if curr.session_date != prev.session_date:
        state.uptrend_count = 0
        state.had_uptrend = False
        state.structure_break_active = False
        return

    # Higher-high + higher-low = uptrend bar
    hh = curr.high > prev.high
    hl = curr.low > prev.low

    # Lower-high + lower-low = structure break
    lh = curr.high < prev.high
    ll = curr.low < prev.low

    if hh and hl:
        state.uptrend_count += 1
        if state.uptrend_count >= 1:
            state.had_uptrend = True
    elif lh and ll:
        state.uptrend_count = 0
        # Structure break only valid if preceded by uptrend
        if state.had_uptrend:
            bar_range = curr.high - curr.low
            if bar_range >= cfg.bh_sb_range_threshold:
                state.structure_break_active = True
                state.sb_bar_range = bar_range
    else:
        # Indeterminate: decay uptrend counter
        state.uptrend_count = max(0, state.uptrend_count - 1)


def _update_engulfing(completed_bar: Bar30m, state: SignalState, cfg):
    """Check for 30m bearish engulfing.

    Previous bar green (close > open), current bar red (close < open),
    current body fully engulfs previous body:
        current open > prev close AND current close < prev open
    """
    bars = _aggregator.completed
    if len(bars) < 2:
        return

    curr = bars[-1]
    prev = bars[-2]

    if curr.session_date != prev.session_date:
        return

    # Previous bar must be green
    if prev.close <= prev.open:
        return

    # Current bar must be red
    if curr.close >= curr.open:
        return

    # Body engulfing
    if curr.open > prev.close and curr.close < prev.open:
        bar_range = curr.high - curr.low
        if bar_range >= cfg.bh_engulf_range_threshold:
            state.engulfing_active = True


def _update_trend_continuation(completed_bar: Bar30m, state: SignalState, cfg):
    """Check for trend continuation: 30m close below prior 30m low.

    This is the WINNING variant from the SPX trend continuation strategy.
    """
    bars = _aggregator.completed
    if len(bars) < 2:
        return

    curr = bars[-1]
    prev = bars[-2]

    if curr.session_date != prev.session_date:
        return

    # 30m close below prior 30m low = strong momentum continuation
    if curr.close < prev.low:
        state.lower_close_active = True


def _update_ema_cross(bar: dict, state: SignalState, cfg):
    """Check for EMA 8/24 bearish crossover on 5m bars.

    This runs on every 5m bar (not 30m). Uses the EMA values
    already computed by the indicators module if available,
    otherwise computes incrementally.
    """
    # Use pre-computed EMAs from indicators.py
    ema_8 = bar.get("ema_8", float("nan"))
    ema_24 = bar.get("sma_24", float("nan"))  # backtester computes sma_24

    # Fallback: use ema_9/ema_21 which are always computed
    if math.isnan(ema_8):
        ema_8 = bar.get("ema_9", float("nan"))
    if math.isnan(ema_24):
        ema_24 = bar.get("ema_21", float("nan"))

    if math.isnan(ema_8) or math.isnan(ema_24):
        return

    prev_8 = state.prev_ema_8
    prev_24 = state.prev_ema_24

    state.prev_ema_8 = state.ema_8
    state.prev_ema_24 = state.ema_24
    state.ema_8 = ema_8
    state.ema_24 = ema_24

    if math.isnan(prev_8) or math.isnan(prev_24):
        return

    # Bearish crossover: was above, now below
    if prev_8 >= prev_24 and ema_8 < ema_24:
        state.ema_cross_active = True


def _update_vix_spike(bar: dict, state: SignalState, cfg):
    """Check for VIX spike proxy using ATR volatility ratio.

    Real VIX intraday data isn't available in the ES backtester.
    Proxy: vol_ratio (ATR / ATR_avg) as a local volatility spike detector.
    When vol_ratio >= threshold AND bar is red, treat as VIX spike equivalent.

    This is conservative but captures the same market dynamics:
    expanding ATR = expanding realized vol = VIX is spiking.
    """
    vol_ratio = bar.get("vol_ratio", 1.0)
    is_red = bar["close"] < bar["open"]

    if vol_ratio >= cfg.bh_vix_vol_ratio_threshold and is_red:
        state.vix_spike_active = True


# ============================================================================
# Signal Validity Check (for re-entry)
# ============================================================================

def _is_signal_still_valid(bar: dict, state: SignalState, cfg) -> bool:
    """Check if any signal condition still holds for re-entry.

    Called when a target is hit. If still valid, the engine will re-enter.
    """
    # TEMA bearish state must still hold
    if not bar.get("tema_bearish", False):
        return False

    # Must still be below TEMA trend
    if cfg.bh_require_trend and not bar.get("trend_down", False):
        return False

    # At least one signal type must still be active
    if state.structure_break_active:
        return True
    if state.engulfing_active:
        return True
    if state.lower_close_active:
        return True
    if state.ema_cross_active:
        return True
    if state.vix_spike_active:
        return True

    return False


# ============================================================================
# Stop / Target Computation
# ============================================================================

def _compute_stop(bar: dict, session, cfg) -> float:
    """Compute stop for a short entry.

    If bh_use_level_stops is True, uses session structural levels
    (IB mid, IBH, ONH) as stop placement. Otherwise, fixed points.
    Stop is always capped at pct_stop_bps from entry.
    """
    entry = bar["close"]

    # Percentage cap (always applied)
    pct_cap = entry * cfg.bh_stop_bps / 10000.0

    if cfg.bh_use_level_stops:
        # Level-aware: nearest resistance above entry as stop
        candidates = []

        # IB high
        if session.ib_done and not math.isnan(session.ib_high):
            if session.ib_high > entry:
                candidates.append(session.ib_high + 1.0)

        # IB mid (if we're below IB)
        if session.ib_done and not math.isnan(session.ib_mid):
            if session.ib_mid > entry:
                candidates.append(session.ib_mid)

        # Overnight high
        if not math.isnan(session.on_high) and session.on_high > entry:
            candidates.append(session.on_high + 1.0)

        # Previous day high
        if not math.isnan(session.prev_day_high) and session.prev_day_high > entry:
            candidates.append(session.prev_day_high + 1.0)

        if candidates:
            # Use nearest level above entry
            raw_stop = min(candidates)
            # Cap by percentage
            return min(raw_stop, entry + pct_cap)

    # Fixed stop
    return entry + min(cfg.bh_stop_pts, pct_cap)


def _compute_target(bar: dict, session, cfg) -> float:
    """Compute target for a short entry.

    If bh_use_level_targets is True, uses session structural levels
    (prev_poc, prev_val, dev_val, PDL) as target. Otherwise, fixed points.
    """
    entry = bar["close"]

    if cfg.bh_use_level_targets:
        # Level-aware: nearest support below entry as target
        candidates = []

        # Developing POC
        if not math.isnan(session.dev_poc) and session.dev_poc < entry:
            dist = entry - session.dev_poc
            if dist >= cfg.bh_min_target_pts:
                candidates.append(session.dev_poc)

        # Developing VAL
        if not math.isnan(session.dev_val) and session.dev_val < entry:
            dist = entry - session.dev_val
            if dist >= cfg.bh_min_target_pts:
                candidates.append(session.dev_val)

        # Previous POC (real VP if available, else VWAP proxy)
        prev_poc = session.prev_vp_poc if not math.isnan(session.prev_vp_poc) else session.prev_poc
        if not math.isnan(prev_poc) and prev_poc < entry:
            dist = entry - prev_poc
            if dist >= cfg.bh_min_target_pts:
                candidates.append(prev_poc)

        # Previous VAL
        prev_val = session.prev_vp_val if not math.isnan(session.prev_vp_val) else session.prev_val
        if not math.isnan(prev_val) and prev_val < entry:
            dist = entry - prev_val
            if dist >= cfg.bh_min_target_pts:
                candidates.append(prev_val)

        # Previous day low
        if not math.isnan(session.prev_day_low) and session.prev_day_low < entry:
            dist = entry - session.prev_day_low
            if dist >= cfg.bh_min_target_pts:
                candidates.append(session.prev_day_low)

        if candidates:
            # Nearest support (closest to entry = most likely to hit)
            candidates.sort(reverse=True)  # highest first = nearest below entry
            return candidates[0]

    # Fixed target
    return entry - cfg.bh_target_pts


# ============================================================================
# Main Entry: check_signal() — matches ES backtester interface
# ============================================================================

def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Base Hits — check for ES short entry from SPX option signal types.

    This function is called once per 5m bar by the engine when position is flat.
    It:
      1. Manages the 30m aggregator (feeds 5m bars, gets completed 30m bars)
      2. Updates all 5 signal detectors
      3. Checks if any signal is active + filters pass
      4. Returns an entry signal dict or None

    Signal dict format:
        {"direction": -1, "stop": float, "target": float, "setup": str}

    Args:
        bar: Current 5m bar dict with all indicator columns
        prev_bar: Previous 5m bar dict (or None)
        session: SessionState object
        cfg: StrategyConfig with bh_* fields

    Returns:
        Signal dict for a short entry, or None
    """
    _ensure_init()

    if not cfg.bh_enabled:
        return None

    # ── Session reset ──
    if bar.get("new_rth", False):
        _reset_for_new_session()

    if not bar.get("is_rth", False):
        return None

    _signal_state.bar_index += 1

    # ── Feed 5m bar to 30m aggregator ──
    completed_30m = _aggregator.feed(bar)

    # ── Update 30m-based signals when a 30m bar completes ──
    if completed_30m is not None:
        if cfg.bh_use_structure_break:
            _update_structure_break(completed_30m, _signal_state, cfg)
        if cfg.bh_use_engulfing:
            _update_engulfing(completed_30m, _signal_state, cfg)
        if cfg.bh_use_trend_cont:
            _update_trend_continuation(completed_30m, _signal_state, cfg)

    # ── Update 5m-based signals every bar ──
    if cfg.bh_use_ema_cross:
        _update_ema_cross(bar, _signal_state, cfg)
    if cfg.bh_use_vix_spike:
        _update_vix_spike(bar, _signal_state, cfg)

    # ── Signal invalidation: reset signals when TEMA turns bullish ──
    if bar.get("tema_bullish", False):
        _signal_state.structure_break_active = False
        _signal_state.engulfing_active = False
        _signal_state.lower_close_active = False
        _signal_state.ema_cross_active = False
        _signal_state.vix_spike_active = False

    # ── Composite signal check ──
    _signal_state.any_signal_active = (
        _signal_state.structure_break_active
        or _signal_state.engulfing_active
        or _signal_state.lower_close_active
        or _signal_state.ema_cross_active
        or _signal_state.vix_spike_active
    )

    # Determine which signal triggered (for labeling)
    if _signal_state.structure_break_active:
        _signal_state.active_setup = "BH_SB"
    elif _signal_state.engulfing_active:
        _signal_state.active_setup = "BH_ENG"
    elif _signal_state.lower_close_active:
        _signal_state.active_setup = "BH_TC"
    elif _signal_state.ema_cross_active:
        _signal_state.active_setup = "BH_EMA"
    elif _signal_state.vix_spike_active:
        _signal_state.active_setup = "BH_VIX"
    else:
        _signal_state.active_setup = ""

    if not _signal_state.any_signal_active:
        return None

    # ── Time window filter ──
    et_time = bar.get("et_time", 0)
    if et_time < cfg.bh_entry_start or et_time >= cfg.bh_entry_end:
        return None

    # ── Must be in trading window ──
    if not bar.get("is_trading_window", False):
        # Allow early entries for VIX spike (fires before IB completes)
        if not (_signal_state.vix_spike_active and et_time >= 935):
            return None

    # ── Max trades per day ──
    if _signal_state.trades_today >= cfg.bh_max_trades_per_day:
        return None

    # ── Minimum bars between entries ──
    if (_signal_state.bar_index - _signal_state.last_entry_bar) < cfg.bh_min_entry_gap:
        return None

    # ── Cooldown from last exit ──
    if session.bars_since_exit < cfg.cooldown_bars:
        return None

    # ── TEMA filter: require bearish MA state ──
    if cfg.bh_require_tema and not bar.get("tema_bearish", False):
        return None

    # ── Trend filter: require close below TEMA 55 ──
    if cfg.bh_require_trend and not bar.get("trend_down", False):
        return None

    # ── Volatility filter ──
    if cfg.bh_use_vol_filter and not session.vol_ok:
        return None

    # ── Compute stop and target ──
    stop = _compute_stop(bar, session, cfg)
    target = _compute_target(bar, session, cfg)

    # ── Validate risk/reward ──
    entry = bar["close"]
    risk = stop - entry
    reward = entry - target
    if risk <= 0 or reward <= 0:
        return None
    if cfg.bh_min_rr > 0 and (reward / risk) < cfg.bh_min_rr:
        return None

    # ── Record entry ──
    _signal_state.trades_today += 1
    _signal_state.last_entry_bar = _signal_state.bar_index

    return {
        "direction": -1,
        "stop": stop,
        "target": target,
        "setup": _signal_state.active_setup,
    }


def check_reentry(bar: dict, prev_bar: Optional[dict], session, cfg) -> bool:
    """Called by the engine after a target hit to check for re-entry.

    Returns True if signal conditions still hold and a new short
    entry should be placed. The engine handles the actual position
    management — this just validates the signal.
    """
    if not cfg.bh_allow_reentry:
        return False

    if _signal_state.trades_today >= cfg.bh_max_trades_per_day:
        return False

    et_time = bar.get("et_time", 0)
    if et_time >= cfg.bh_entry_end:
        return False

    return _is_signal_still_valid(bar, _signal_state, cfg)


def on_target_hit(bar: dict, session, cfg):
    """Notification from engine that our target was hit.

    Updates internal state for re-entry tracking.
    """
    _signal_state.last_target_hit_bar = _signal_state.bar_index


def on_stop_hit(bar: dict, session, cfg):
    """Notification from engine that our stop was hit.

    Invalidates all signals — stop hit means thesis is wrong.
    """
    _signal_state.structure_break_active = False
    _signal_state.engulfing_active = False
    _signal_state.lower_close_active = False
    _signal_state.ema_cross_active = False
    _signal_state.vix_spike_active = False
    _signal_state.any_signal_active = False


def reset_module():
    """Full reset of module state. Call between independent backtests."""
    global _aggregator, _signal_state, _initialized
    _aggregator = Aggregator30m()
    _signal_state = SignalState()
    _initialized = False


# ============================================================================
# Strategy Config Preset
# ============================================================================

def get_config():
    """Return a StrategyConfig with Base Hits defaults.

    Disables all other setups so only BH signals fire.
    """
    from backtester.config import StrategyConfig

    cfg = StrategyConfig()

    # Enable Base Hits
    cfg.bh_enabled = True

    # Disable all other setups
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_ib_reject = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_tema_cross = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_ms = False
    cfg.use_fa = False
    cfg.use_os = False

    # Direction: short only (all SPX signals are short/puts)
    cfg.direction_filter = "short"

    # Standard ES settings
    cfg.pct_stop_mode = False  # BH uses its own stop logic
    cfg.skip_friday = False
    cfg.blackout_start = 1200  # Match v8 noon blackout
    cfg.blackout_end = 1300

    # --- Tunable parameters (explicit for AutoResearch cfg_attr extraction) ---
    # Signal toggles
    cfg.bh_use_structure_break = True
    cfg.bh_use_engulfing = True
    cfg.bh_use_trend_cont = True
    cfg.bh_use_ema_cross = True
    cfg.bh_use_vix_spike = True

    # Signal thresholds
    cfg.bh_sb_range_threshold = 2.0
    cfg.bh_engulf_range_threshold = 2.0
    cfg.bh_vix_vol_ratio_threshold = 1.8

    # Entry filters
    cfg.bh_entry_start = 935
    cfg.bh_entry_end = 1500
    cfg.bh_require_tema = True
    cfg.bh_require_trend = True

    # Stop / target (validated: 4pt/4pt)
    cfg.bh_stop_pts = 4.5
    cfg.bh_target_pts = 4.5
    cfg.bh_stop_bps = 30.0
    cfg.bh_min_rr = 0.5

    # Trade management
    cfg.bh_max_trades_per_day = 10
    cfg.bh_min_entry_gap = 2
    cfg.bh_allow_reentry = True

    return cfg
