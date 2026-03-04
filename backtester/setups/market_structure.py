"""Setup 9: Market Structure (MS) — Dalton's structural levels + SMA 8/24 timing.

Implements core Dalton Market Profile setups with modern MA confirmation:
  - VA Fade: responsive trading at prev day's REAL VP-derived VAH/VAL
  - IB Level Trade: fade or break at IB high/low
  - Overnight Level Trade: fade or break at ON high/low
  - Developing VA Rotation: fade at today's evolving VA edges
  - POC Pivot: trade at previous/developing POC as pivot

The KEY insight: structure tells you WHERE, the MA tells you WHEN.
  - Structural levels (VA, IB, ON, POC) = WHERE to enter
  - SMA 8/24 state = WHEN to enter (direction confirmation)
  - Entry lag = prevents algorithm shakeouts

Previous VA Fade failed because prev_poc/prev_vah/prev_val were computed from
a VWAP proxy (Gaussian approximation). Now uses REAL volume profile-derived
levels from yesterday's finalized VolumeProfile with 68% value area.
"""

import math
from typing import Optional, List, Tuple




def _fib_target_short(price: float, session) -> Optional[float]:
    """Compute Fibonacci retracement target for shorts.
    
    Target = prev_day_low + (prev_day_range * 0.382) — the 38.2% retracement from the low.
    This targets the 0.382 level of the prior day's range.
    """
    if math.isnan(session.prev_day_low) or math.isnan(session.prev_day_range):
        return None
    if session.prev_day_range <= 0:
        return None
    target = session.prev_day_low + (session.prev_day_range * 0.382)
    if target < price - 2:
        return target
    return None


def _fib_target_long(price: float, session) -> Optional[float]:
    """Compute Fibonacci retracement target for longs.
    
    Target = prev_day_low + (prev_day_range * 0.618) — the 61.8% retracement from the low.
    This targets the 0.618 level of the prior day's range.
    """
    if math.isnan(session.prev_day_low) or math.isnan(session.prev_day_range):
        return None
    if session.prev_day_range <= 0:
        return None
    target = session.prev_day_low + (session.prev_day_range * 0.618)
    if target > price + 2:
        return target
    return None


def _level_allowed(label: str, direction: str, level_directions: dict) -> bool:
    """Check if a level is allowed in the given direction per ms_level_directions.

    Args:
        label: The level label (e.g., "MS_ONH", "MS_pPOC_L")
        direction: "short" or "long"
        level_directions: Dict mapping base label to allowed direction
                         e.g., {"MS_ONH": "short", "MS_pPOC": "long"}

    Returns True if allowed (default when dict is empty or label not in dict).
    For POC long labels (ending in _L), strips _L to find the base key.
    """
    if not level_directions:
        return True  # empty dict = allow all (backward compatible)

    # Normalize: MS_pPOC_L -> MS_pPOC, MS_dPOC_L -> MS_dPOC
    base_label = label.rstrip("_L") if label.endswith("_L") else label

    allowed = level_directions.get(base_label)
    if allowed is None:
        return False  # level not in dict = NOT allowed (whitelist mode)

    return allowed == "both" or allowed == direction


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for Market Structure signal at any structural level.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_ms:
        return None

    # Gate 1: Must be RTH after IB
    if not bar["is_rth"] or not session.ib_done:
        return None

    # Gate 2: Trade counter
    if session.ms_trades >= cfg.max_ms_trades:
        return None

    close = bar["close"]
    high = bar["high"]
    low = bar["low"]

    # Get MA state for direction confirmation
    if cfg.ms_ma_type == "sma":
        ma_fast = bar.get("sma_8", float("nan"))
        ma_slow = bar.get("sma_24", float("nan"))
    else:  # tema
        ma_fast = bar.get("tema_fast", float("nan"))
        ma_slow = bar.get("tema_slow", float("nan"))

    if math.isnan(ma_fast) or math.isnan(ma_slow):
        return None

    ma_bearish = ma_fast < ma_slow
    ma_bullish = ma_fast > ma_slow

    # Entry lag: require MA state for N consecutive bars
    if cfg.ms_ma_confirm_bars > 0 and prev_bar is not None:
        if cfg.ms_ma_type == "sma":
            prev_fast = prev_bar.get("sma_8", float("nan"))
            prev_slow = prev_bar.get("sma_24", float("nan"))
        else:
            prev_fast = prev_bar.get("tema_fast", float("nan"))
            prev_slow = prev_bar.get("tema_slow", float("nan"))
        if not math.isnan(prev_fast) and not math.isnan(prev_slow):
            # For confirm_bars=1, require prev bar also in same state
            if ma_bearish and not (prev_fast < prev_slow):
                ma_bearish = False  # MA just crossed, too fresh
            if ma_bullish and not (prev_fast > prev_slow):
                ma_bullish = False

    zone = cfg.ms_zone_pts

    # Use VP-derived levels if available, fall back to VWAP proxy
    if cfg.ms_use_vp_levels:
        prev_vah = session.prev_vp_vah if not math.isnan(session.prev_vp_vah) else session.prev_vah
        prev_val = session.prev_vp_val if not math.isnan(session.prev_vp_val) else session.prev_val
        prev_poc = session.prev_vp_poc if not math.isnan(session.prev_vp_poc) else session.prev_poc
    else:
        prev_vah = session.prev_vah
        prev_val = session.prev_val
        prev_poc = session.prev_poc

    # ═══════════════════════════════════════════════════════════════
    #  Build level lists
    # ═══════════════════════════════════════════════════════════════
    sell_levels = []  # (price, label)
    buy_levels = []

    # Per-level direction filter (empty = allow all, backward compatible)
    lvl_dirs = getattr(cfg, 'ms_level_directions', {})

    # Previous day VA (Dalton's VA Fade — responsive trading)
    if cfg.ms_use_prev_va:
        if not math.isnan(prev_vah) and _level_allowed("MS_pVAH", "short", lvl_dirs):
            sell_levels.append((prev_vah, "MS_pVAH"))
        if not math.isnan(prev_val) and _level_allowed("MS_pVAL", "long", lvl_dirs):
            buy_levels.append((prev_val, "MS_pVAL"))

    # Overnight levels (session extremes)
    if cfg.ms_use_on_levels:
        if not math.isnan(session.on_high) and _level_allowed("MS_ONH", "short", lvl_dirs):
            sell_levels.append((session.on_high, "MS_ONH"))
        if not math.isnan(session.on_low) and _level_allowed("MS_ONL", "long", lvl_dirs):
            buy_levels.append((session.on_low, "MS_ONL"))

    # IB levels (initial balance edges)
    if cfg.ms_use_ib_levels:
        if not math.isnan(session.ib_high) and _level_allowed("MS_IBH", "short", lvl_dirs):
            sell_levels.append((session.ib_high, "MS_IBH"))
        if not math.isnan(session.ib_low) and _level_allowed("MS_IBL", "long", lvl_dirs):
            buy_levels.append((session.ib_low, "MS_IBL"))

    # Developing VA (today's evolving value area)
    if cfg.ms_use_dev_va:
        if not math.isnan(session.dev_vah) and _level_allowed("MS_dVAH", "short", lvl_dirs):
            sell_levels.append((session.dev_vah, "MS_dVAH"))
        if not math.isnan(session.dev_val) and _level_allowed("MS_dVAL", "long", lvl_dirs):
            buy_levels.append((session.dev_val, "MS_dVAL"))

    # POC as pivot (both directions based on MA state)
    if cfg.ms_use_poc:
        if not math.isnan(prev_poc):
            if _level_allowed("MS_pPOC", "short", lvl_dirs):
                sell_levels.append((prev_poc, "MS_pPOC"))
            if _level_allowed("MS_pPOC", "long", lvl_dirs):
                buy_levels.append((prev_poc, "MS_pPOC_L"))
        if not math.isnan(session.dev_poc):
            if _level_allowed("MS_dPOC", "short", lvl_dirs):
                sell_levels.append((session.dev_poc, "MS_dPOC"))
            if _level_allowed("MS_dPOC", "long", lvl_dirs):
                buy_levels.append((session.dev_poc, "MS_dPOC_L"))

    # ═══════════════════════════════════════════════════════════════
    #  Check SELL levels (price near level + MA bearish → SHORT)
    # ═══════════════════════════════════════════════════════════════
    if ma_bearish:
        for level, label in sell_levels:
            if label in session.ms_traded_levels:
                continue  # Each level trades once per day

            # Price must be in the zone
            if high >= level - zone and close <= level + zone:
                target = _find_target_below(close, session, prev_val, prev_poc)
                if target is None:
                    continue
                # Fibonacci target override: use fib target if it gives better R:R
                if getattr(cfg, 'ms_use_fib_targets', False):
                    fib_t = _fib_target_short(close, session)
                    if fib_t is not None:
                        fib_dist = close - fib_t
                        struct_dist = close - target
                        # Use fib target if it gives more reward (further from entry)
                        if fib_dist > struct_dist:
                            target = fib_t
                stop = level + cfg.ms_stop_buffer
                distance = close - target
                risk = stop - close
                if (distance >= cfg.ms_min_target_pts and risk > 0
                        and risk <= cfg.ms_max_risk
                        and (distance / risk) >= cfg.ms_min_rr):
                    session.ms_trades += 1
                    session.ms_traded_levels.add(label)
                    return {"direction": -1, "stop": stop, "target": target, "setup": label}

    # ═══════════════════════════════════════════════════════════════
    #  Check BUY levels (price near level + MA bullish → LONG)
    # ═══════════════════════════════════════════════════════════════
    if ma_bullish:
        for level, label in buy_levels:
            if label in session.ms_traded_levels:
                continue

            if low <= level + zone and close >= level - zone:
                target = _find_target_above(close, session, prev_vah, prev_poc)
                if target is None:
                    continue
                # POC overhead filter: skip longs when prev POC is between entry and target
                if getattr(cfg, 'ms_skip_long_poc_overhead', False):
                    if not math.isnan(prev_poc) and prev_poc > close and prev_poc < target:
                        continue  # POC acts as resistance, skip this long
                # Fibonacci target override: use fib target if it gives better R:R
                if getattr(cfg, 'ms_use_fib_targets', False):
                    fib_t = _fib_target_long(close, session)
                    if fib_t is not None:
                        fib_dist = fib_t - close
                        struct_dist = target - close
                        # Use fib target if it gives more reward (further from entry)
                        if fib_dist > struct_dist:
                            target = fib_t
                stop = level - cfg.ms_stop_buffer
                distance = target - close
                risk = close - stop
                if (distance >= cfg.ms_min_target_pts and risk > 0
                        and risk <= cfg.ms_max_risk
                        and (distance / risk) >= cfg.ms_min_rr):
                    session.ms_trades += 1
                    session.ms_traded_levels.add(label)
                    return {"direction": 1, "stop": stop, "target": target, "setup": label}

    return None


def _find_target_below(price: float, session, prev_val: float, prev_poc: float) -> Optional[float]:
    """Find nearest structural level below price as a short target.

    Dalton: price seeks fair value (POC) or next accepted area.
    """
    candidates = []
    if not math.isnan(prev_poc) and prev_poc < price - 2:
        candidates.append(prev_poc)
    if not math.isnan(prev_val) and prev_val < price - 2:
        candidates.append(prev_val)
    if not math.isnan(session.dev_poc) and session.dev_poc < price - 2:
        candidates.append(session.dev_poc)
    if not math.isnan(session.dev_val) and session.dev_val < price - 2:
        candidates.append(session.dev_val)
    if not math.isnan(session.on_low) and session.on_low < price - 2:
        candidates.append(session.on_low)
    if not math.isnan(session.ib_low) and session.ib_low < price - 2:
        candidates.append(session.ib_low)

    if not candidates:
        return None
    # Nearest target below = highest of the candidates below price
    return max(candidates)


def _find_target_above(price: float, session, prev_vah: float, prev_poc: float) -> Optional[float]:
    """Find nearest structural level above price as a long target.

    Dalton: price seeks fair value (POC) or next accepted area.
    """
    candidates = []
    if not math.isnan(prev_poc) and prev_poc > price + 2:
        candidates.append(prev_poc)
    if not math.isnan(prev_vah) and prev_vah > price + 2:
        candidates.append(prev_vah)
    if not math.isnan(session.dev_poc) and session.dev_poc > price + 2:
        candidates.append(session.dev_poc)
    if not math.isnan(session.dev_vah) and session.dev_vah > price + 2:
        candidates.append(session.dev_vah)
    if not math.isnan(session.on_high) and session.on_high > price + 2:
        candidates.append(session.on_high)
    if not math.isnan(session.ib_high) and session.ib_high > price + 2:
        candidates.append(session.ib_high)

    if not candidates:
        return None
    # Nearest target above = lowest of the candidates above price
    return min(candidates)
