"""Setup 7: Value Area Rotation (VAR) — fade developing VA edges on rotation days.

On rotation days (no active OTF), price oscillates between the developing
Value Area High and Value Area Low. This setup shorts at dev_VAH and goes
long at dev_VAL, targeting the developing POC (the center of accepted value).

Key design decisions:
  - Uses DEVELOPING VA (today's evolving profile), not yesterday's static VA
  - Both longs AND shorts (rotation = both directions)
  - Target is dev_POC (dynamic — moves as profile develops)
  - No TEMA filter (rotation days don't trend, TEMA adds noise)
  - Min 4 periods (~2 hours) ensures enough data for reliable VA
  - High frequency: up to 8 trades/day on rotation days
"""

import math
from typing import Optional


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for Value Area Rotation signal at developing VA edges.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_var:
        return None

    # Gate 1: Must be RTH, after IB (wait for profile to develop)
    if not bar["is_rth"] or not session.ib_done:
        return None

    # Gate 2: Must have enough 30-min periods for reliable developing VA
    periods_elapsed = session.vp_30m_period
    if periods_elapsed < cfg.var_min_ib_periods:
        return None

    # Gate 3: Trade counter
    if session.var_trades >= cfg.max_var_trades:
        return None

    # Gate 4: Rotation check — no active OTF streak
    if cfg.var_require_rotation:
        if session.otf_up_count > cfg.var_max_otf or session.otf_down_count > cfg.var_max_otf:
            return None

    # Gate 5: Developing VA must exist and have reasonable width
    if math.isnan(session.dev_vah) or math.isnan(session.dev_val) or math.isnan(session.dev_poc):
        return None
    va_width = session.dev_vah - session.dev_val
    if va_width < 4.0:  # too narrow = no room to trade
        return None

    close = bar["close"]
    high = bar["high"]
    low = bar["low"]

    # SHORT at developing VAH
    if high >= session.dev_vah - cfg.var_zone_pts and close < session.dev_vah:
        target = session.dev_poc if cfg.var_target_pts == 0 else close - cfg.var_target_pts
        stop = session.dev_vah + cfg.var_stop_buffer
        distance = close - target
        risk = stop - close
        if distance > 0 and risk > 0 and (distance / risk) >= cfg.var_min_rr:
            session.var_trades += 1
            return {"direction": -1, "stop": stop, "target": target, "setup": "VAR"}

    # LONG at developing VAL
    if low <= session.dev_val + cfg.var_zone_pts and close > session.dev_val:
        target = session.dev_poc if cfg.var_target_pts == 0 else close + cfg.var_target_pts
        stop = session.dev_val - cfg.var_stop_buffer
        distance = target - close
        risk = close - stop
        if distance > 0 and risk > 0 and (distance / risk) >= cfg.var_min_rr:
            session.var_trades += 1
            return {"direction": 1, "stop": stop, "target": target, "setup": "VAR"}

    return None
