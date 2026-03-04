"""Setup 4: TEMA Cross Short — momentum short on TEMA fast/slow crossdown (v9).

Short-only on ta.crossunder(temaFast, temaSlow), day-type gated.
Narrow IB = trend day = TEMA cross works. Wide IB = rotational = gets chopped.
"""

from typing import Optional


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for TEMA Cross Short signal.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_tema_cross:
        return None

    if not session.ib_done:
        return None

    if not bar["is_trading_window"]:
        return None

    if session.bars_since_exit < cfg.cooldown_bars:
        return None

    # TEMA crossunder detection (transition, not state)
    if not bar.get("tema_cross_down", False):
        return None

    # Trend filter: require close < TEMA trend
    if not bar["trend_down"]:
        return None

    # Day type filter
    day_type = cfg.tx_day_type_filter.lower()
    if day_type == "narrow":
        if not session.is_narrow_ib:
            return None
    elif day_type == "narrow+normal":
        if session.is_wide_ib:
            return None
    # "all" — no day type filter

    # Volatility filter (required for TX, unlike IB)
    if not session.vol_ok:
        return None

    # Trade limit
    if session.tx_trades_s >= cfg.max_tx_trades:
        return None

    # Stop and target
    stop_pts = bar["close"] * cfg.tx_stop_bps / 10000.0
    stop = bar["close"] + stop_pts
    target = bar["close"] - bar["atr"] * cfg.tx_tp_atr_mult

    session.tx_trades_s += 1
    return {"direction": -1, "stop": stop, "target": target, "setup": "TX"}
