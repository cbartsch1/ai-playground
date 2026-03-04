"""Setup 8: Post-Trend Day Fade (PTF) — fade into yesterday's single prints.

After a trend day leaves thin single-print structure, the next day tends to
revisit (fill) those thin zones. This setup enters when price reaches the
single print zone and targets the previous day's POC (where value was accepted).

Key design decisions:
  - Only fires the day AFTER a trend day (prev_day_was_trend)
  - Entry zone = yesterday's single prints (thin structure that gets filled)
  - Target = prev_poc (where value was accepted — the "magnet")
  - Optional reversal confirmation: wait for OTF in opposite direction
  - Both longs AND shorts depending on trend day direction
  - Lower frequency (max 2/day) but larger targets (8+ pts)
"""

import math
from typing import Optional


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for Post-Trend Day Fade signal in yesterday's single prints.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_ptf:
        return None

    # Gate 1: Previous day must have been a trend day
    if not session.prev_day_was_trend:
        return None

    # Gate 2: Must be RTH after IB
    if not bar["is_rth"] or not session.ib_done:
        return None

    # Gate 3: Trade counter
    if session.ptf_trades >= cfg.max_ptf_trades:
        return None

    # Gate 4: Single print zone must be defined
    sp_high = session.prev_day_single_print_high
    sp_low = session.prev_day_single_print_low
    if math.isnan(sp_high) or math.isnan(sp_low):
        return None

    close = bar["close"]

    # Determine target
    if cfg.ptf_target == "prev_poc":
        target = session.prev_poc
    elif cfg.ptf_target == "composite_poc":
        poc = session.current_vp.get_poc()
        target = poc if poc is not None else session.prev_poc
    else:  # single_print_mid
        target = (sp_high + sp_low) / 2

    if math.isnan(target):
        return None

    # AFTER OTF-UP trend day: SHORT into single prints
    if session.prev_day_otf_max_up >= cfg.ptf_min_otf:
        # Price should be near or inside the single print zone
        if close >= sp_low and close <= sp_high + cfg.ptf_stop_buffer:
            if target < close:  # target must be below for shorts
                stop = sp_high + cfg.ptf_stop_buffer
                distance = close - target
                risk = stop - close
                if distance >= cfg.ptf_min_target_pts and risk > 0:
                    if cfg.ptf_require_reversal and session.otf_down_count < 1:
                        return None
                    session.ptf_trades += 1
                    return {"direction": -1, "stop": stop, "target": target, "setup": "PTF"}

    # AFTER OTF-DOWN trend day: LONG into single prints
    if session.prev_day_otf_max_down >= cfg.ptf_min_otf:
        if close <= sp_high and close >= sp_low - cfg.ptf_stop_buffer:
            if target > close:  # target must be above for longs
                stop = sp_low - cfg.ptf_stop_buffer
                distance = target - close
                risk = close - stop
                if distance >= cfg.ptf_min_target_pts and risk > 0:
                    if cfg.ptf_require_reversal and session.otf_up_count < 1:
                        return None
                    session.ptf_trades += 1
                    return {"direction": 1, "stop": stop, "target": target, "setup": "PTF"}

    return None
