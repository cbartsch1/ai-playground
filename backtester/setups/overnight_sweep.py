"""Overnight Sweep / Gap Fade — short gap-up opens after overnight supply builds.

Auction Theory Logic:
  - Overnight session trades above previous day's close → weak-handed longs
    accumulated at premium prices become overhead supply
  - Gap up at RTH open → those overnight longs need to sell
  - First RTH candle establishes the opening print (high = natural stop)
  - Price tends to rotate down toward accepted value:
    prev_close → prev_vah → prev_poc → prev_val (cascade of support levels)

Mirror logic for gap-down opens:
  - Overnight below prev close → overhead demand
  - Gap down at open → weak shorts cover → price rotates up

Entry:
  - First bar(s) of RTH session when gap detected + overnight sweep confirmed
  - SHORT on gap up, LONG on gap down
  - Stop at opening print high/low + buffer
  - Target cascades through support/resistance levels
"""

import math
from typing import Optional


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for overnight sweep / gap fade entry signal.

    Args:
        bar: Current bar dict
        prev_bar: Previous bar dict
        session: SessionState
        cfg: StrategyConfig

    Returns:
        Signal dict or None
    """
    if not cfg.use_os:
        return None

    # Must be RTH
    if not bar["is_rth"]:
        return None

    # Trade counter
    if session.os_trades >= cfg.max_os_trades:
        return None

    # Entry window: only trade in first N bars of RTH (opening range)
    # Bar index 1 = first RTH bar (new_rth), 2 = second, etc.
    if session.lvl_bar_index > cfg.os_entry_window:
        return None

    # Need previous day close for gap calculation
    if math.isnan(session.prev_day_close):
        return None

    # Need opening print reference
    if math.isnan(session.opening_print_high) or math.isnan(session.opening_print_low):
        return None

    close = bar["close"]
    gap = session.gap_size

    # ═══════════════════════════════════════════════════
    # GAP UP → SHORT (overnight supply above prev close)
    # ═══════════════════════════════════════════════════
    if gap >= cfg.os_min_gap and gap <= cfg.os_max_gap:
        # Require overnight session to have swept above prev close
        if cfg.os_require_on_sweep and not session.on_above_prev_close:
            return None

        # MA confirmation (optional)
        if cfg.os_require_ma:
            if cfg.os_ma_type == "sma":
                sma_fast = bar.get("sma_fast", float("nan"))
                sma_slow = bar.get("sma_slow", float("nan"))
                if math.isnan(sma_fast) or math.isnan(sma_slow):
                    return None
                if sma_fast >= sma_slow:  # Need bearish alignment
                    return None
            elif cfg.os_ma_type == "tema":
                tema_fast = bar.get("tema_fast", float("nan"))
                tema_slow = bar.get("tema_slow", float("nan"))
                if math.isnan(tema_fast) or math.isnan(tema_slow):
                    return None
                if tema_fast >= tema_slow:
                    return None

        # Stop placement
        stop = _compute_stop_short(session, cfg)
        if math.isnan(stop):
            return None

        # Target: cascade through support levels
        target = _find_target_short(close, session, cfg)
        if target is None:
            return None

        # Risk/reward checks
        risk = stop - close
        distance = close - target
        if risk <= 0 or distance <= 0:
            return None
        if risk > cfg.os_max_risk:
            return None
        if distance < cfg.os_min_target_pts:
            return None
        if distance / risk < cfg.os_min_rr:
            return None

        session.os_trades += 1
        return {
            "direction": -1,
            "stop": stop,
            "target": target,
            "setup": "OS_GAP_UP",
        }

    # ═══════════════════════════════════════════════════
    # GAP DOWN → LONG (demand built below prev close)
    # ═══════════════════════════════════════════════════
    if gap <= -cfg.os_min_gap and gap >= -cfg.os_max_gap:
        # Require overnight to have traded below prev close
        if cfg.os_require_on_sweep and not session.on_below_prev_close:
            return None

        # MA confirmation (optional)
        if cfg.os_require_ma:
            if cfg.os_ma_type == "sma":
                sma_fast = bar.get("sma_fast", float("nan"))
                sma_slow = bar.get("sma_slow", float("nan"))
                if math.isnan(sma_fast) or math.isnan(sma_slow):
                    return None
                if sma_fast <= sma_slow:  # Need bullish alignment
                    return None
            elif cfg.os_ma_type == "tema":
                tema_fast = bar.get("tema_fast", float("nan"))
                tema_slow = bar.get("tema_slow", float("nan"))
                if math.isnan(tema_fast) or math.isnan(tema_slow):
                    return None
                if tema_fast <= tema_slow:
                    return None

        # Stop placement
        stop = _compute_stop_long(session, cfg)
        if math.isnan(stop):
            return None

        # Target: cascade through resistance levels
        target = _find_target_long(close, session, cfg)
        if target is None:
            return None

        # Risk/reward checks
        risk = close - stop
        distance = target - close
        if risk <= 0 or distance <= 0:
            return None
        if risk > cfg.os_max_risk:
            return None
        if distance < cfg.os_min_target_pts:
            return None
        if distance / risk < cfg.os_min_rr:
            return None

        session.os_trades += 1
        return {
            "direction": 1,
            "stop": stop,
            "target": target,
            "setup": "OS_GAP_DN",
        }

    return None


def _compute_stop_short(session, cfg) -> float:
    """Compute stop for short gap-up fade."""
    if cfg.os_stop_mode == "opening_print":
        return session.opening_print_high + cfg.os_stop_buffer
    elif cfg.os_stop_mode == "on_extreme":
        if math.isnan(session.on_high):
            return session.opening_print_high + cfg.os_stop_buffer
        return max(session.on_high, session.opening_print_high) + cfg.os_stop_buffer
    elif cfg.os_stop_mode == "fixed":
        return session.rth_open + cfg.os_fixed_stop
    return float("nan")


def _compute_stop_long(session, cfg) -> float:
    """Compute stop for long gap-down fade."""
    if cfg.os_stop_mode == "opening_print":
        return session.opening_print_low - cfg.os_stop_buffer
    elif cfg.os_stop_mode == "on_extreme":
        if math.isnan(session.on_low):
            return session.opening_print_low - cfg.os_stop_buffer
        return min(session.on_low, session.opening_print_low) - cfg.os_stop_buffer
    elif cfg.os_stop_mode == "fixed":
        return session.rth_open - cfg.os_fixed_stop
    return float("nan")


def _find_target_short(close: float, session, cfg) -> Optional[float]:
    """Find downside target for gap-up short.

    Cascade mode tries each support level in order:
    prev_close → prev_vah → prev_poc → prev_val
    Returns the nearest one that gives enough distance.
    """
    targets = []

    if cfg.os_target_mode == "prev_close":
        if not math.isnan(session.prev_day_close):
            return session.prev_day_close
        return None

    if cfg.os_target_mode == "prev_vah":
        # Use real VP levels if available
        vah = session.prev_vp_vah if not math.isnan(session.prev_vp_vah) else session.prev_vah
        if not math.isnan(vah):
            return vah
        return None

    if cfg.os_target_mode == "prev_poc":
        poc = session.prev_vp_poc if not math.isnan(session.prev_vp_poc) else session.prev_poc
        if not math.isnan(poc):
            return poc
        return None

    # Cascade mode: try levels in order, pick the nearest that works
    if cfg.os_target_mode == "cascade":
        # Level 1: previous day close (gap fill)
        if not math.isnan(session.prev_day_close) and session.prev_day_close < close:
            targets.append(session.prev_day_close)

        # Level 2: prev VAH (real VP if available)
        vah = session.prev_vp_vah if not math.isnan(session.prev_vp_vah) else session.prev_vah
        if not math.isnan(vah) and vah < close:
            targets.append(vah)

        # Level 3: prev POC (real VP if available)
        poc = session.prev_vp_poc if not math.isnan(session.prev_vp_poc) else session.prev_poc
        if not math.isnan(poc) and poc < close:
            targets.append(poc)

        # Level 4: prev VAL (real VP if available)
        val = session.prev_vp_val if not math.isnan(session.prev_vp_val) else session.prev_val
        if not math.isnan(val) and val < close:
            targets.append(val)

        if not targets:
            return None

        # Sort by distance from close (nearest first)
        targets.sort(key=lambda t: close - t)

        # Return nearest target that meets minimum distance
        for t in targets:
            if close - t >= cfg.os_min_target_pts:
                return t

        # If no target meets min distance, return the farthest one
        # (it might still pass the R:R check)
        return targets[-1] if targets else None

    return None


def _find_target_long(close: float, session, cfg) -> Optional[float]:
    """Find upside target for gap-down long.

    Cascade: prev_close → prev_val → prev_poc → prev_vah
    """
    targets = []

    if cfg.os_target_mode == "prev_close":
        if not math.isnan(session.prev_day_close):
            return session.prev_day_close
        return None

    if cfg.os_target_mode == "prev_vah":
        vah = session.prev_vp_vah if not math.isnan(session.prev_vp_vah) else session.prev_vah
        if not math.isnan(vah):
            return vah
        return None

    if cfg.os_target_mode == "prev_poc":
        poc = session.prev_vp_poc if not math.isnan(session.prev_vp_poc) else session.prev_poc
        if not math.isnan(poc):
            return poc
        return None

    if cfg.os_target_mode == "cascade":
        # Level 1: previous day close (gap fill)
        if not math.isnan(session.prev_day_close) and session.prev_day_close > close:
            targets.append(session.prev_day_close)

        # Level 2: prev VAL (real VP if available)
        val = session.prev_vp_val if not math.isnan(session.prev_vp_val) else session.prev_val
        if not math.isnan(val) and val > close:
            targets.append(val)

        # Level 3: prev POC
        poc = session.prev_vp_poc if not math.isnan(session.prev_vp_poc) else session.prev_poc
        if not math.isnan(poc) and poc > close:
            targets.append(poc)

        # Level 4: prev VAH
        vah = session.prev_vp_vah if not math.isnan(session.prev_vp_vah) else session.prev_vah
        if not math.isnan(vah) and vah > close:
            targets.append(vah)

        if not targets:
            return None

        # Sort by distance from close (nearest first)
        targets.sort(key=lambda t: t - close)

        for t in targets:
            if t - close >= cfg.os_min_target_pts:
                return t

        return targets[-1] if targets else None

    return None
