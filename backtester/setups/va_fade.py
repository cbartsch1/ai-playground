"""Setup 2: Value Area Fade + TEMA Slope — mean reversion at VA edges.

Matches Pine Script lines 278-321 exactly.
"""

import math
from typing import Optional


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for VA Fade signal.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_va_fade:
        return None

    if not bar["is_trading_window"]:
        return None

    if session.bars_since_exit < cfg.cooldown_bars:
        return None

    # Need previous VA levels
    if math.isnan(session.prev_vah) or math.isnan(session.prev_val) or math.isnan(session.prev_poc):
        return None

    # Vol filter required for VA Fade (unlike IB)
    if not session.vol_ok:
        return None

    # Day type gate: skip on narrow IB (trending, fades get run over)
    if cfg.use_day_type and session.is_narrow_ib:
        return None

    # Fallback values matching Pine nz()
    prev_vah = session.prev_vah
    prev_val = session.prev_val
    prev_poc = session.prev_poc if not math.isnan(session.prev_poc) else bar["close"]

    buffer_pts = cfg.va_buffer_pts
    atr = bar["atr"]

    # Touch detection
    touch_low = bar["low"] <= prev_val + buffer_pts and bar["close"] > prev_val
    touch_high = bar["high"] >= prev_vah - buffer_pts and bar["close"] < prev_vah

    if not touch_low and not touch_high:
        return None

    if touch_low:
        # Long at VAL
        if not bar["slope_rising"]:
            return None
        if session.va_trades_l >= cfg.max_va_trades:
            return None

        stop = prev_val - atr * cfg.va_stop_mult
        target = prev_poc

        # R:R check
        reward = abs(target - bar["close"])
        risk = abs(bar["close"] - stop)
        if risk <= 0 or reward / risk < cfg.va_min_rr:
            return None

        session.va_trades_l += 1
        return {"direction": 1, "stop": stop, "target": target, "setup": "VA"}

    else:  # touch_high
        # Short at VAH
        if not bar["slope_falling"]:
            return None
        if session.va_trades_s >= cfg.max_va_trades:
            return None

        stop = prev_vah + atr * cfg.va_stop_mult
        target = prev_poc

        # R:R check
        reward = abs(bar["close"] - target)
        risk = abs(stop - bar["close"])
        if risk <= 0 or reward / risk < cfg.va_min_rr:
            return None

        session.va_trades_s += 1
        return {"direction": -1, "stop": stop, "target": target, "setup": "VA"}
