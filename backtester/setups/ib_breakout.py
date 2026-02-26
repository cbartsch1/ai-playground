"""Setup 1: IB Breakout + TEMA — trend day entry after IB range break.

v8: One-shot crossover/crossunder (ta.crossover/ta.crossunder)
v9: Continuous re-arm (signal fires every bar price is beyond IB)
"""

import math
from typing import Optional, Tuple


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for IB Breakout signal.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_ib_break:
        return None

    if not session.ib_done:
        return None

    if not bar["is_trading_window"]:
        return None

    if session.bars_since_exit < cfg.cooldown_bars:
        return None

    # IB range validation
    ib_valid = session.ib_range >= cfg.min_ib_range and session.ib_range <= cfg.max_ib_range
    if not ib_valid:
        return None

    if prev_bar is None:
        return None

    ib_high = session.ib_high
    ib_low = session.ib_low

    # v9: Continuous re-arm — signal fires every bar beyond IB
    # v8: One-shot crossover/crossunder
    if cfg.tp_atr_mult > 0:
        # v9 mode: continuous check
        cross_up = bar["close"] > ib_high
        cross_down = bar["close"] < ib_low
    else:
        # v8 mode: one-shot crossover/crossunder
        cross_up = bar["close"] > ib_high and prev_bar["close"] <= ib_high
        cross_down = bar["close"] < ib_low and prev_bar["close"] >= ib_low

    if not cross_up and not cross_down:
        return None

    # Dynamic max stop: scale with price if pct_stop_mode is on
    max_stop = cfg.ib_max_stop_pts
    if cfg.pct_stop_mode:
        max_stop = bar["close"] * cfg.pct_stop_bps / 10000.0

    atr = bar["atr"]

    # Direction-specific gates
    if cross_up:
        if not bar["tema_bullish"]:
            return None
        if cfg.use_trend_filter and not bar["trend_up"]:
            return None
        if session.ib_trades_l >= cfg.max_ib_trades:
            return None

        if cfg.ib_stop_type == "IB Mid":
            raw_sl = session.ib_mid
        elif cfg.ib_stop_type == "IB Edge":
            raw_sl = ib_low
        else:  # ATR
            raw_sl = bar["close"] - atr * 1.5

        stop = max(raw_sl, bar["close"] - max_stop)

        # v9 scaled TP: cap at ATR multiple
        if cfg.tp_atr_mult > 0:
            tp_pts = max(cfg.ib_min_target, min(session.ib_range, atr * cfg.tp_atr_mult))
        else:
            tp_pts = max(session.ib_range, cfg.ib_min_target)
        target = bar["close"] + tp_pts

        session.ib_trades_l += 1
        return {"direction": 1, "stop": stop, "target": target, "setup": "IB"}

    else:  # cross_down
        if not bar["tema_bearish"]:
            return None
        if cfg.use_trend_filter and not bar["trend_down"]:
            return None
        if session.ib_trades_s >= cfg.max_ib_trades:
            return None

        if cfg.ib_stop_type == "IB Mid":
            raw_sl = session.ib_mid
        elif cfg.ib_stop_type == "IB Edge":
            raw_sl = ib_high
        else:  # ATR
            raw_sl = bar["close"] + atr * 1.5

        stop = min(raw_sl, bar["close"] + max_stop)

        # v9 scaled TP: cap at ATR multiple
        if cfg.tp_atr_mult > 0:
            tp_pts = max(cfg.ib_min_target, min(session.ib_range, atr * cfg.tp_atr_mult))
        else:
            tp_pts = max(session.ib_range, cfg.ib_min_target)
        target = bar["close"] - tp_pts

        session.ib_trades_s += 1
        return {"direction": -1, "stop": stop, "target": target, "setup": "IB"}
