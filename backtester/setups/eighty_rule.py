"""Setup 3: 80% Rule + TEMA — open outside VA, re-enter and traverse.

Matches Pine Script lines 323-372 exactly.
Default OFF (use_eighty=False).
"""

import math
from typing import Optional


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for 80% Rule signal.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_eighty:
        return None

    if not bar["is_trading_window"]:
        return None

    if session.bars_since_exit < cfg.cooldown_bars:
        return None

    # Need previous VA levels
    if math.isnan(session.prev_vah) or math.isnan(session.prev_val):
        return None

    # Vol filter required
    if not session.vol_ok:
        return None

    atr = bar["atr"]
    prev_vah = session.prev_vah
    prev_val = session.prev_val

    # 80% Rule: opened below VA, confirmed re-entry → long to VAH
    if session.open_below_va and session.eighty_confirmed:
        if not (bar["tema_bullish"] or bar["slope_rising"]):
            return None
        if session.eighty_trades_l >= cfg.max_eighty_trades:
            return None

        stop = prev_val - atr * cfg.eighty_stop_buf
        target = prev_vah

        session.eighty_trades_l += 1
        return {"direction": 1, "stop": stop, "target": target, "setup": "80%"}

    # 80% Rule: opened above VA, confirmed re-entry → short to VAL
    if session.open_above_va and session.eighty_confirmed:
        if not (bar["tema_bearish"] or bar["slope_falling"]):
            return None
        if session.eighty_trades_s >= cfg.max_eighty_trades:
            return None

        stop = prev_vah + atr * cfg.eighty_stop_buf
        target = prev_val

        session.eighty_trades_s += 1
        return {"direction": -1, "stop": stop, "target": target, "setup": "80%"}

    return None
