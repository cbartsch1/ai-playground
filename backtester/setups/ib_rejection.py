"""Setup 5: IB Rejection Short — fade IB high resistance.

Short when price tests IB high zone and shows rejection.
Level-to-level targeting: VWAP, IB mid, IB low, previous POC, or fixed points.
Stop tight above IB high — let the stop do the risk management.

Trigger types (from most to least aggressive):
  - "any":           bar high reaches IB high zone → short
  - "bearish_close": bar high in zone + close < open
  - "wick":          bar high in zone + close < open + close in lower half
  - "failed_break":  high pierces above IB high + close back below IB high
"""

import math
from typing import Optional


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for IB Rejection Short signal.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_ib_reject:
        return None

    if not session.ib_done:
        return None

    if not bar["is_trading_window"]:
        return None

    if session.bars_since_exit < cfg.cooldown_bars:
        return None

    # Short-only setup — skip if direction filter blocks shorts
    if cfg.direction_filter == "long":
        return None

    # Max trades per day
    if session.rej_trades_s >= cfg.max_rej_trades:
        return None

    # Wide IB day filter — rejection alpha only exists on wide days
    if cfg.rej_wide_only and not session.is_wide_ib:
        return None

    ib_high = session.ib_high
    zone_floor = ib_high - cfg.rej_zone_pts

    # Bar must reach into the IB high zone (high touches/enters zone)
    if bar["high"] < zone_floor:
        return None

    # TEMA bearish state filter (optional — not a crossover, just current state)
    if cfg.rej_require_tema and not bar["tema_bearish"]:
        return None

    # ── Trigger check ──
    trigger = cfg.rej_trigger
    triggered = False

    if trigger == "any":
        # Most aggressive: touching zone is enough
        triggered = True

    elif trigger == "bearish_close":
        # Bearish candle in zone
        triggered = bar["close"] < bar["open"]

    elif trigger == "wick":
        # Bearish candle closing in lower half of bar
        bar_mid = (bar["high"] + bar["low"]) / 2
        triggered = bar["close"] < bar["open"] and bar["close"] < bar_mid

    elif trigger == "failed_break":
        # High pierces above IB high, close back below
        triggered = bar["high"] > ib_high and bar["close"] <= ib_high

    if not triggered:
        return None

    # ── Stop: just above IB high ──
    stop = ib_high + cfg.rej_stop_buffer

    # ── Target: level-to-level ──
    target = _compute_target(bar, session, cfg)
    if target is None:
        return None

    # Target must be below entry price
    if target >= bar["close"]:
        return None

    # Stop must be above entry price
    if stop <= bar["close"]:
        stop = bar["close"] + 2.0

    session.rej_trades_s += 1
    return {"direction": -1, "stop": stop, "target": target, "setup": "REJ"}


def _compute_target(bar: dict, session, cfg) -> Optional[float]:
    """Compute target price based on target type."""
    target_type = cfg.rej_target

    if target_type == "vwap":
        if session.rth_vol_sum > 0:
            vwap = session.rth_vwap_sum / session.rth_vol_sum
            return vwap
        return None

    elif target_type == "ib_mid":
        if not math.isnan(session.ib_mid):
            return session.ib_mid
        return None

    elif target_type == "ib_low":
        if not math.isnan(session.ib_low):
            return session.ib_low
        return None

    elif target_type == "prev_poc":
        if not math.isnan(session.prev_poc):
            return session.prev_poc
        return None

    elif target_type == "fixed":
        return bar["close"] - cfg.rej_target_pts

    return None
