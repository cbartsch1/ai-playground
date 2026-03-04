"""Setup 10: Failed Auction — IB breakout that fails and reverses.

Dalton's key pattern: when the IB extension fails, the auction has found a
boundary. The reversal back inside IB has high conviction because:
  - Breakout traders are trapped
  - Their stops fuel the reversal
  - The "failed" direction is now validated as rejected

Entry: Short when price breaks above IBH then closes back inside IB.
       Long when price breaks below IBL then closes back inside IB.
Stop: Beyond the failed breakout extreme + buffer
Target: Opposite IB edge (IBH->IBL or IBL->IBH)
"""

import math
from typing import Optional


def update_state(bar: dict, session, cfg) -> None:
    """Track IB breakout / failure state every bar.

    Must be called EVERY bar (even when in position) to maintain accurate state.
    Only tracks during RTH after IB is done.
    """
    if not cfg.use_fa:
        return
    if not bar["is_rth"] or not session.ib_done:
        return

    close = bar["close"]
    high = bar["high"]
    low = bar["low"]
    ib_high = session.ib_high
    ib_low = session.ib_low

    if math.isnan(ib_high) or math.isnan(ib_low):
        return

    # --- Upside break tracking ---
    if close > ib_high and not session.fa_ib_broken_above:
        # First close above IBH — mark the break
        session.fa_ib_broken_above = True
        session.fa_break_above_bar = session.lvl_bar_index
        session.fa_break_above_extreme = high

    if session.fa_ib_broken_above and high > session.fa_break_above_extreme:
        # Update the extreme high reached during the upside break
        session.fa_break_above_extreme = high

    # --- Downside break tracking ---
    if close < ib_low and not session.fa_ib_broken_below:
        # First close below IBL — mark the break
        session.fa_ib_broken_below = True
        session.fa_break_below_bar = session.lvl_bar_index
        session.fa_break_below_extreme = low

    if session.fa_ib_broken_below and low < session.fa_break_below_extreme:
        # Update the extreme low reached during the downside break
        session.fa_break_below_extreme = low


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for Failed Auction signal.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_fa:
        return None

    # Gate 1: RTH, after IB is complete
    if not bar["is_rth"] or not session.ib_done:
        return None

    # Gate 2: Trade counter
    if session.fa_trades >= cfg.max_fa_trades:
        return None

    # Gate 3: IB must be valid (non-NaN, range >= 4 pts)
    ib_high = session.ib_high
    ib_low = session.ib_low
    if math.isnan(ib_high) or math.isnan(ib_low):
        return None
    ib_range = ib_high - ib_low
    if ib_range < 4.0:
        return None

    close = bar["close"]

    # ═══════════════════════════════════════════════════════════════
    # Check for FAILED UPSIDE BREAK → SHORT signal
    # ═══════════════════════════════════════════════════════════════
    if (session.fa_ib_broken_above
            and close < ib_high
            and not session.fa_short_fired):
        bars_since_break = session.lvl_bar_index - session.fa_break_above_bar
        if bars_since_break <= cfg.fa_max_break_bars and bars_since_break > 0:
            # Optional MA confirmation: SMA 8 < SMA 24 (bearish)
            if cfg.fa_require_ma:
                if cfg.fa_ma_type == "sma":
                    ma_fast = bar.get("sma_8", float("nan"))
                    ma_slow = bar.get("sma_24", float("nan"))
                elif cfg.fa_ma_type == "tema":
                    ma_fast = bar.get("tema_fast", float("nan"))
                    ma_slow = bar.get("tema_slow", float("nan"))
                else:
                    ma_fast = bar.get("sma_8", float("nan"))
                    ma_slow = bar.get("sma_24", float("nan"))
                if math.isnan(ma_fast) or math.isnan(ma_slow):
                    pass  # skip this signal path
                elif ma_fast >= ma_slow:
                    # MA is not bearish — skip SHORT
                    return _check_long_signal(bar, session, cfg)

            stop = session.fa_break_above_extreme + cfg.fa_stop_buffer
            target = ib_low  # Opposite IB edge

            risk = stop - close
            reward = close - target

            if risk <= 0 or reward <= 0:
                return _check_long_signal(bar, session, cfg)
            if risk > cfg.fa_max_risk:
                return _check_long_signal(bar, session, cfg)
            if (reward / risk) < cfg.fa_min_rr:
                return _check_long_signal(bar, session, cfg)

            # Direction filter
            if cfg.direction_filter == "long":
                return _check_long_signal(bar, session, cfg)

            session.fa_trades += 1
            session.fa_short_fired = True
            return {
                "direction": -1,
                "stop": stop,
                "target": target,
                "setup": "FA_short",
            }

    # ═══════════════════════════════════════════════════════════════
    # Check for FAILED DOWNSIDE BREAK → LONG signal
    # ═══════════════════════════════════════════════════════════════
    return _check_long_signal(bar, session, cfg)


def _check_long_signal(bar: dict, session, cfg) -> Optional[dict]:
    """Check for failed downside break -> LONG signal."""
    close = bar["close"]
    ib_high = session.ib_high
    ib_low = session.ib_low

    if math.isnan(ib_high) or math.isnan(ib_low):
        return None

    if (session.fa_ib_broken_below
            and close > ib_low
            and not session.fa_long_fired):
        bars_since_break = session.lvl_bar_index - session.fa_break_below_bar
        if bars_since_break <= cfg.fa_max_break_bars and bars_since_break > 0:
            # Optional MA confirmation: SMA 8 > SMA 24 (bullish)
            if cfg.fa_require_ma:
                if cfg.fa_ma_type == "sma":
                    ma_fast = bar.get("sma_8", float("nan"))
                    ma_slow = bar.get("sma_24", float("nan"))
                elif cfg.fa_ma_type == "tema":
                    ma_fast = bar.get("tema_fast", float("nan"))
                    ma_slow = bar.get("tema_slow", float("nan"))
                else:
                    ma_fast = bar.get("sma_8", float("nan"))
                    ma_slow = bar.get("sma_24", float("nan"))
                if math.isnan(ma_fast) or math.isnan(ma_slow):
                    return None
                if ma_fast <= ma_slow:
                    return None  # MA is not bullish — skip LONG

            stop = session.fa_break_below_extreme - cfg.fa_stop_buffer
            target = ib_high  # Opposite IB edge

            risk = close - stop
            reward = target - close

            if risk <= 0 or reward <= 0:
                return None
            if risk > cfg.fa_max_risk:
                return None
            if (reward / risk) < cfg.fa_min_rr:
                return None

            # Direction filter
            if cfg.direction_filter == "short":
                return None

            session.fa_trades += 1
            session.fa_long_fired = True
            return {
                "direction": 1,
                "stop": stop,
                "target": target,
                "setup": "FA_long",
            }

    return None
