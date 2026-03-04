"""Setup 6: Level Rejection Short — market structure aware short entries.

Short when price tests ANY resistance level (PDH, ONH, IBH, prev_VAH) and shows
rejection. Targets the next support level below (level-to-level trading).

Key design principles:
  - Levels have state: untested → tested (1x, 2x, 3x+) → broken
  - Broken levels are skipped (resistance flipped to support)
  - After N tests (default 3), level is skipped (defenders exhausted, likely to break)
  - Highest resistance level wins when multiple are in zone (confluence)
  - Shared trade counter across ALL levels prevents chop

Trigger types (same as ib_rejection):
  - "any":           bar high reaches level zone → short
  - "bearish_close": bar high in zone + close < open
  - "wick":          bar high in zone + close < open + close in lower half
  - "failed_break":  high pierces above level + close back below level
"""

import math
from typing import Optional, List, Tuple


# ── Resistance level names (short entry candidates, checked highest first) ──
RESISTANCE_LEVELS = ["PDH", "ONH", "IBH", "VAH"]

# ── Support level names (target candidates, checked from entry price downward) ──
SUPPORT_LEVELS = ["VWAP", "PREV_POC", "IB_MID", "IB_LOW", "VAL", "ONL", "PDL"]


def _get_resistance_map(session) -> List[Tuple[str, float]]:
    """Build sorted list of (name, price) for resistance levels. Highest first.

    Skips any level that is NaN (not yet established).
    """
    levels = []

    if not math.isnan(session.prev_day_high):
        levels.append(("PDH", session.prev_day_high))

    if not math.isnan(session.on_high):
        levels.append(("ONH", session.on_high))

    if session.ib_done and not math.isnan(session.ib_high):
        levels.append(("IBH", session.ib_high))

    if not math.isnan(session.prev_vah):
        levels.append(("VAH", session.prev_vah))

    # Sort descending by price (highest resistance first)
    levels.sort(key=lambda x: x[1], reverse=True)
    return levels


def _get_support_map(bar: dict, session) -> List[Tuple[str, float]]:
    """Build sorted list of (name, price) for support levels. Highest first.

    Used for level-to-level targeting: find the first support below entry.
    """
    levels = []

    # Session VWAP
    if session.rth_vol_sum > 0:
        vwap = session.rth_vwap_sum / session.rth_vol_sum
        levels.append(("VWAP", vwap))

    if not math.isnan(session.prev_poc):
        levels.append(("PREV_POC", session.prev_poc))

    if session.ib_done:
        if not math.isnan(session.ib_mid):
            levels.append(("IB_MID", session.ib_mid))
        if not math.isnan(session.ib_low):
            levels.append(("IB_LOW", session.ib_low))

    if not math.isnan(session.prev_val):
        levels.append(("VAL", session.prev_val))

    if not math.isnan(session.on_low):
        levels.append(("ONL", session.on_low))

    if not math.isnan(session.prev_day_low):
        levels.append(("PDL", session.prev_day_low))

    # Sort descending by price (so we can walk down from entry)
    levels.sort(key=lambda x: x[1], reverse=True)
    return levels


def update_level_state(bar: dict, session, cfg) -> None:
    """Track level state every bar — test count, broken flag, zone presence.

    Must be called EVERY bar (even when in position) to maintain accurate state.
    Only tracks during RTH after IB is done.

    Test counting uses proper deduplication: a "test" is a VISIT to the zone,
    not every individual bar inside it. If the previous bar was already in the
    zone, this bar does not start a new test. A new test only begins when price
    RE-ENTERS the zone after having left it.
    """
    if not cfg.use_level_reject:
        return

    if not bar["is_rth"] or not session.ib_done:
        return

    resistance = _get_resistance_map(session)
    zone = cfg.lvl_zone_pts

    for name, price in resistance:
        in_zone_now = bar["high"] >= price - zone
        was_in_zone = session.lvl_in_zone.get(name, False)

        if in_zone_now and not was_in_zone:
            # New visit to the zone — count as a new test
            count = session.lvl_test_count.get(name, 0) + 1
            session.lvl_test_count[name] = count

            # Record first test bar index (for speed of rejection tracking)
            if name not in session.lvl_first_test_bar:
                session.lvl_first_test_bar[name] = session.lvl_bar_index

        # Accumulate bar range/volume while in zone (for absorption proxy)
        if in_zone_now:
            bar_range = bar["high"] - bar["low"]
            bar_vol = bar.get("volume", 0)
            if name not in session.lvl_zone_ranges:
                session.lvl_zone_ranges[name] = []
                session.lvl_zone_volumes[name] = []
            session.lvl_zone_ranges[name].append(bar_range)
            session.lvl_zone_volumes[name].append(bar_vol)

        # Update zone presence for next bar's dedup
        session.lvl_in_zone[name] = in_zone_now

        # Level broken tracking: count consecutive closes above level
        if bar["close"] > price:
            session.lvl_broken_count[name] = session.lvl_broken_count.get(name, 0) + 1
        else:
            # Price back below level — reset broken count (failed breakout!)
            session.lvl_broken_count[name] = 0
            session.lvl_broken[name] = False

        # Only mark broken after N consecutive closes above (default 2)
        if session.lvl_broken_count.get(name, 0) >= cfg.lvl_broken_bars:
            session.lvl_broken[name] = True


def _check_trigger(bar: dict, trigger: str, level_price: float) -> bool:
    """Check if bar satisfies the trigger condition near a resistance level."""
    if trigger == "any":
        return True

    elif trigger == "bearish_close":
        return bar["close"] < bar["open"]

    elif trigger == "wick":
        bar_mid = (bar["high"] + bar["low"]) / 2
        return bar["close"] < bar["open"] and bar["close"] < bar_mid

    elif trigger == "failed_break":
        # High pierces above level, close pulls back below
        return bar["high"] > level_price and bar["close"] <= level_price

    return False


def _confluence_score(name: str, price: float, resistance: List[Tuple[str, float]],
                      zone: float) -> int:
    """Count how many resistance levels cluster within `zone` points of this level.

    Each distinct level within the zone adds +1. The level itself counts as 1.
    For example, if PDH=6050, ONH=6048, VAH=6053 and zone=5,
    all three cluster together → score=3 for any of them.
    """
    score = 0
    for other_name, other_price in resistance:
        if abs(other_price - price) <= zone:
            score += 1
    return score


def _bar_quality(bar: dict) -> dict:
    """Compute bar quality metrics for rejection detection.

    Returns dict with body_ratio, upper_wick_ratio, clv.
    """
    bar_range = bar["high"] - bar["low"]
    if bar_range <= 0:
        return {"body_ratio": 0.0, "upper_wick_ratio": 0.0, "clv": 0.0}

    body = abs(bar["close"] - bar["open"])
    upper_wick = bar["high"] - max(bar["close"], bar["open"])

    body_ratio = body / bar_range
    upper_wick_ratio = upper_wick / bar_range
    clv = (2 * bar["close"] - bar["low"] - bar["high"]) / bar_range

    return {
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "clv": clv,
    }


def _check_absorption(name: str, session, cfg) -> bool:
    """Check if bars at this level show absorption pattern (institutional defense).

    Absorption = 3+ bars at level with decreasing range and elevated volume.
    This signals that a large player is absorbing sell orders, increasing
    the probability that the level holds and price rejects.

    Returns True if absorption detected (pass filter), False if not.
    """
    ranges = session.lvl_zone_ranges.get(name, [])
    volumes = session.lvl_zone_volumes.get(name, [])

    if len(ranges) < cfg.lvl_absorption_min_bars:
        return False  # Not enough bars at level

    # Check decreasing range: last N bars should show compression
    recent = ranges[-cfg.lvl_absorption_min_bars:]
    decreasing = True
    for i in range(1, len(recent)):
        if recent[i] >= recent[i - 1]:
            decreasing = False
            break

    if not decreasing:
        return False

    # Check elevated volume: average volume at level vs session average
    if cfg.lvl_absorption_vol_mult > 0 and session.rth_bars > 0:
        avg_zone_vol = sum(volumes[-cfg.lvl_absorption_min_bars:]) / cfg.lvl_absorption_min_bars
        session_avg_vol = session.rth_vol_sum / session.rth_bars
        if session_avg_vol > 0 and avg_zone_vol < session_avg_vol * cfg.lvl_absorption_vol_mult:
            return False

    return True


def check_signal(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for Level Rejection Short signal at any resistance level.

    Walks resistance levels from highest to lowest — first match wins.
    Applies confluence scoring, R:R filter, and bar quality metrics.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_level_reject:
        return None

    if not session.ib_done:
        return None

    if not bar["is_trading_window"]:
        return None

    if session.bars_since_exit < cfg.cooldown_bars:
        return None

    # Short-only setup
    if cfg.direction_filter == "long":
        return None

    # Shared trade counter across all levels
    if session.lvl_trades_s >= cfg.max_lvl_trades:
        return None

    # MA bearish filter (optional) — supports TEMA, EMA, SMA variants
    if cfg.lvl_require_tema:
        ma_filter = cfg.lvl_ma_filter
        if ma_filter == "tema":
            ma_bearish = bar.get("tema_bearish", False)
        elif ma_filter == "ema_9_21":
            ma_bearish = bar.get("ema_bearish_9_21", False)
        elif ma_filter == "ema_8_21":
            ma_bearish = bar.get("ema_bearish_8_21", False)
        elif ma_filter == "sma_8_21":
            ma_bearish = bar.get("sma_bearish_8_21", False)
        else:
            ma_bearish = bar.get("tema_bearish", False)
        if not ma_bearish:
            return None

    resistance = _get_resistance_map(session)
    zone = cfg.lvl_zone_pts

    # Pre-compute bar quality metrics (once per bar, not per level)
    bq = _bar_quality(bar) if cfg.lvl_use_bar_metrics else None

    for name, price in resistance:
        # Skip levels not in enabled list (if configured)
        if cfg.lvl_enabled_levels and name not in cfg.lvl_enabled_levels:
            continue

        # Bar must reach into the zone
        if bar["high"] < price - zone:
            continue

        # Skip broken levels (resistance defeated, role flipped to support)
        if session.lvl_broken.get(name, False):
            continue

        # Skip exhausted levels (tested too many times, defenders tired)
        if session.lvl_test_count.get(name, 0) >= cfg.lvl_max_tests:
            continue

        # IBH: optionally restrict to wide IB days only (proven edge)
        if name == "IBH" and cfg.lvl_ibh_wide_only and not session.is_wide_ib:
            continue

        # ONH: optionally skip when overnight high is "poor" (weak, likely to break)
        if name == "ONH" and cfg.lvl_skip_poor_high and session.on_high_is_poor:
            continue

        # ── Confluence scoring ──
        confluence = _confluence_score(name, price, resistance, cfg.lvl_confluence_zone)
        if confluence < cfg.lvl_min_confluence:
            continue

        # Check trigger
        if not _check_trigger(bar, cfg.lvl_trigger, price):
            continue

        # ── Bar quality metrics (optional) ──
        if bq is not None and cfg.lvl_use_bar_metrics:
            # Require meaningful upper wick (rejection signature)
            if bq["upper_wick_ratio"] < cfg.lvl_min_wick_ratio:
                continue

        # ── Absorption proxy filter (optional) ──
        if cfg.lvl_use_absorption:
            if not _check_absorption(name, session, cfg):
                continue

        # ── Stop: level + buffer, capped by pct stop ──
        stop = price + cfg.lvl_stop_buffer

        # Cap by pct stop if enabled
        if cfg.pct_stop_mode:
            max_stop_pts = bar["close"] * cfg.pct_stop_bps / 10000.0
            pct_stop = bar["close"] + max_stop_pts
            stop = min(stop, pct_stop)

        # Stop must be above entry
        if stop <= bar["close"]:
            stop = bar["close"] + 2.0

        # ── Target: next support level below entry (level-to-level) ──
        target = _find_target(bar, session, skip=cfg.lvl_target_skip)
        if target is None:
            continue

        # Target must be below entry
        if target >= bar["close"]:
            continue

        # Minimum target distance filter
        target_distance = bar["close"] - target
        if cfg.lvl_min_target_pts > 0 and target_distance < cfg.lvl_min_target_pts:
            continue

        # ── R:R filter ──
        if cfg.lvl_min_rr > 0:
            stop_distance = stop - bar["close"]
            target_distance = bar["close"] - target
            if stop_distance > 0 and target_distance / stop_distance < cfg.lvl_min_rr:
                continue

        # Found a valid signal — first match wins (highest level priority)
        session.lvl_trades_s += 1
        setup_tag = f"LVL_{name}"
        return {
            "direction": -1,
            "stop": stop,
            "target": target,
            "setup": setup_tag,
        }

    return None


def check_signal_multi(bar: dict, prev_bar: Optional[dict], session, cfg,
                       n_contracts: int = 3, uniform_skip: int = -1) -> List[dict]:
    """Check for Level Rejection Short and return up to n_contracts signals with staggered targets.

    Same filter logic as check_signal(), but finds targets for skip=0,1,...,n_contracts-1
    and returns a list of valid signals. Increments session.lvl_trades_s by 1 (not N)
    when any signals are returned.

    Args:
        uniform_skip: If >= 0, ALL contracts target the same skip level (e.g., 2 = all target
                      3rd support). Default -1 = normal stagger (skip 0,1,2,...).

    Returns list of signal dicts with keys: direction, stop, target, setup, contract
    """
    if not cfg.use_level_reject:
        return []

    if not session.ib_done:
        return []

    if not bar["is_trading_window"]:
        return []

    if session.bars_since_exit < cfg.cooldown_bars:
        return []

    if cfg.direction_filter == "long":
        return []

    if session.lvl_trades_s >= cfg.max_lvl_trades:
        return []

    # MA bearish filter
    if cfg.lvl_require_tema:
        ma_filter = cfg.lvl_ma_filter
        if ma_filter == "tema":
            ma_bearish = bar.get("tema_bearish", False)
        elif ma_filter == "ema_9_21":
            ma_bearish = bar.get("ema_bearish_9_21", False)
        elif ma_filter == "ema_8_21":
            ma_bearish = bar.get("ema_bearish_8_21", False)
        elif ma_filter == "sma_8_21":
            ma_bearish = bar.get("sma_bearish_8_21", False)
        else:
            ma_bearish = bar.get("tema_bearish", False)
        if not ma_bearish:
            return []

    resistance = _get_resistance_map(session)
    zone = cfg.lvl_zone_pts

    bq = _bar_quality(bar) if cfg.lvl_use_bar_metrics else None

    for name, price in resistance:
        if cfg.lvl_enabled_levels and name not in cfg.lvl_enabled_levels:
            continue
        if bar["high"] < price - zone:
            continue
        if session.lvl_broken.get(name, False):
            continue
        if session.lvl_test_count.get(name, 0) >= cfg.lvl_max_tests:
            continue
        if name == "IBH" and cfg.lvl_ibh_wide_only and not session.is_wide_ib:
            continue

        # ONH poor high filter (same as check_signal)
        if name == "ONH" and cfg.lvl_skip_poor_high and session.on_high_is_poor:
            continue

        confluence = _confluence_score(name, price, resistance, cfg.lvl_confluence_zone)
        if confluence < cfg.lvl_min_confluence:
            continue

        if not _check_trigger(bar, cfg.lvl_trigger, price):
            continue

        if bq is not None and cfg.lvl_use_bar_metrics:
            if bq["upper_wick_ratio"] < cfg.lvl_min_wick_ratio:
                continue

        # Absorption proxy filter (same as check_signal)
        if cfg.lvl_use_absorption:
            if not _check_absorption(name, session, cfg):
                continue

        # Stop calculation (same as check_signal)
        stop = price + cfg.lvl_stop_buffer
        if cfg.pct_stop_mode:
            max_stop_pts = bar["close"] * cfg.pct_stop_bps / 10000.0
            pct_stop = bar["close"] + max_stop_pts
            stop = min(stop, pct_stop)
        if stop <= bar["close"]:
            stop = bar["close"] + 2.0

        # Find staggered targets
        signals = []

        for contract_idx in range(n_contracts):
            skip = uniform_skip if uniform_skip >= 0 else contract_idx
            target = _find_target(bar, session, skip=skip)
            if target is None:
                continue
            if target >= bar["close"]:
                continue

            target_distance = bar["close"] - target
            if cfg.lvl_min_target_pts > 0 and target_distance < cfg.lvl_min_target_pts:
                continue

            if cfg.lvl_min_rr > 0:
                stop_distance = stop - bar["close"]
                if stop_distance > 0 and target_distance / stop_distance < cfg.lvl_min_rr:
                    continue

            signals.append({
                "direction": -1,
                "stop": stop,
                "target": target,
                "setup": f"LVL_{name}",
                "contract": contract_idx + 1,
            })

        if signals:
            session.lvl_trades_s += 1  # Count as 1 trade, not N
            return signals

    return []


def _find_target(bar: dict, session, skip: int = 0) -> Optional[float]:
    """Find the Nth support level below entry price (level-to-level targeting).

    skip=0: first support below entry (default)
    skip=1: second support below entry
    skip=2: third support below entry
    """
    entry = bar["close"]
    support = _get_support_map(bar, session)

    found = 0
    for name, price in support:
        if price < entry:
            if found == skip:
                return price
            found += 1

    # Fallback: IB low
    if session.ib_done and not math.isnan(session.ib_low) and session.ib_low < entry:
        if found == skip:
            return session.ib_low

    return None


# ══════════════════════════════════════════════════════════════════
#  LONG SIDE — Support Level Rejection (mirror of short side)
# ══════════════════════════════════════════════════════════════════

# Support levels for long entry candidates (checked lowest first)
SUPPORT_ENTRY_LEVELS = ["PDL", "ONL", "VAL", "IBL"]


def _get_support_entry_map(session) -> List[Tuple[str, float]]:
    """Build sorted list of (name, price) for support levels. Lowest first.

    These are levels where we BUY the bounce (long entry candidates).
    """
    levels = []

    if not math.isnan(session.prev_day_low):
        levels.append(("PDL", session.prev_day_low))

    if not math.isnan(session.on_low):
        levels.append(("ONL", session.on_low))

    if session.ib_done and not math.isnan(session.ib_low):
        levels.append(("IBL", session.ib_low))

    if not math.isnan(session.prev_val):
        levels.append(("VAL", session.prev_val))

    # Sort ascending by price (lowest support first — strongest levels)
    levels.sort(key=lambda x: x[1])
    return levels


def _get_resistance_target_map(bar: dict, session) -> List[Tuple[str, float]]:
    """Build sorted list of resistance levels for long targeting. Lowest first.

    Used for level-to-level targeting: find the first resistance ABOVE entry.
    """
    levels = []

    # Session VWAP
    if session.rth_vol_sum > 0:
        vwap = session.rth_vwap_sum / session.rth_vol_sum
        levels.append(("VWAP", vwap))

    if not math.isnan(session.prev_poc):
        levels.append(("PREV_POC", session.prev_poc))

    if session.ib_done:
        if not math.isnan(session.ib_mid):
            levels.append(("IB_MID", session.ib_mid))
        if not math.isnan(session.ib_high):
            levels.append(("IB_HIGH", session.ib_high))

    if not math.isnan(session.prev_vah):
        levels.append(("VAH", session.prev_vah))

    if not math.isnan(session.on_high):
        levels.append(("ONH", session.on_high))

    if not math.isnan(session.prev_day_high):
        levels.append(("PDH", session.prev_day_high))

    # Sort ascending by price (so we can walk UP from entry)
    levels.sort(key=lambda x: x[1])
    return levels


def update_support_level_state(bar: dict, session, cfg) -> None:
    """Track support level state every bar — mirror of update_level_state for longs."""
    if not cfg.use_level_reject_long:
        return

    if not bar["is_rth"] or not session.ib_done:
        return

    support = _get_support_entry_map(session)
    zone = cfg.lvl_long_zone_pts

    for name, price in support:
        # Support zone: bar LOW dips into zone below level
        in_zone_now = bar["low"] <= price + zone
        was_in_zone = session.sup_in_zone.get(name, False)

        if in_zone_now and not was_in_zone:
            # New visit to the support zone — count as a test
            count = session.sup_test_count.get(name, 0) + 1
            session.sup_test_count[name] = count

        session.sup_in_zone[name] = in_zone_now

        # Support broken tracking: consecutive closes BELOW level
        if bar["close"] < price:
            session.sup_broken_count[name] = session.sup_broken_count.get(name, 0) + 1
        else:
            # Price back above — support held (failed breakdown!)
            session.sup_broken_count[name] = 0
            session.sup_broken[name] = False

        if session.sup_broken_count.get(name, 0) >= cfg.lvl_long_broken_bars:
            session.sup_broken[name] = True


def _check_long_trigger(bar: dict, trigger: str, level_price: float) -> bool:
    """Check if bar satisfies trigger condition near a support level (long entry)."""
    if trigger == "any":
        return True

    elif trigger == "bullish_close":
        return bar["close"] > bar["open"]

    elif trigger == "wick":
        bar_mid = (bar["high"] + bar["low"]) / 2
        return bar["close"] > bar["open"] and bar["close"] > bar_mid

    elif trigger == "failed_break":
        # Low pierces below level, close pulls back above
        return bar["low"] < level_price and bar["close"] >= level_price

    return False


def check_signal_long(bar: dict, prev_bar: Optional[dict], session, cfg) -> Optional[dict]:
    """Check for Level Rejection LONG signal at any support level.

    Mirror of check_signal (short side). Buys the bounce at support.
    Walks support levels from lowest to highest — first match wins.

    Returns dict with keys: direction, stop, target, setup
    or None if no signal.
    """
    if not cfg.use_level_reject_long:
        return None

    if not session.ib_done:
        return None

    if not bar["is_trading_window"]:
        return None

    if session.bars_since_exit < cfg.cooldown_bars:
        return None

    # Long-only setup — skip if direction filter is short-only
    if cfg.direction_filter == "short":
        return None

    # Shared long trade counter
    if session.lvl_trades_l >= cfg.max_lvl_long_trades:
        return None

    # MA bullish filter (optional) — mirror of bearish filter for shorts
    if cfg.lvl_long_require_tema:
        ma_filter = cfg.lvl_long_ma_filter
        if ma_filter == "tema":
            ma_bullish = bar.get("tema_bullish", False)
        elif ma_filter == "ema_9_21":
            ma_bullish = not bar.get("ema_bearish_9_21", True)
        elif ma_filter == "ema_8_21":
            ma_bullish = not bar.get("ema_bearish_8_21", True)
        elif ma_filter == "sma_8_21":
            ma_bullish = not bar.get("sma_bearish_8_21", True)
        else:
            ma_bullish = bar.get("tema_bullish", False)
        if not ma_bullish:
            return None

    support = _get_support_entry_map(session)
    zone = cfg.lvl_long_zone_pts

    for name, price in support:
        # Skip levels not in enabled list
        if cfg.lvl_long_enabled_levels and name not in cfg.lvl_long_enabled_levels:
            continue

        # Bar low must dip into the support zone
        if bar["low"] > price + zone:
            continue

        # Skip broken support (price fell through — support defeated)
        if session.sup_broken.get(name, False):
            continue

        # Skip exhausted levels
        if session.sup_test_count.get(name, 0) >= cfg.lvl_long_max_tests:
            continue

        # Confluence scoring (reuse same function — works for any level list)
        confluence = _confluence_score(name, price, support, cfg.lvl_confluence_zone)
        if confluence < cfg.lvl_min_confluence:
            continue

        # Check trigger
        if not _check_long_trigger(bar, cfg.lvl_long_trigger, price):
            continue

        # ── Stop: level - buffer, capped by pct stop ──
        stop = price - cfg.lvl_long_stop_buffer

        # Cap by pct stop if enabled
        if cfg.pct_stop_mode:
            max_stop_pts = bar["close"] * cfg.pct_stop_bps / 10000.0
            pct_stop = bar["close"] - max_stop_pts
            stop = max(stop, pct_stop)

        # Stop must be below entry
        if stop >= bar["close"]:
            stop = bar["close"] - 2.0

        # ── Target: next resistance level ABOVE entry ──
        target = _find_long_target(bar, session)
        if target is None:
            continue

        # Target must be above entry
        if target <= bar["close"]:
            continue

        # Minimum target distance
        target_distance = target - bar["close"]
        if cfg.lvl_long_min_target_pts > 0 and target_distance < cfg.lvl_long_min_target_pts:
            continue

        # R:R filter
        if cfg.lvl_long_min_rr > 0:
            stop_distance = bar["close"] - stop
            if stop_distance > 0 and target_distance / stop_distance < cfg.lvl_long_min_rr:
                continue

        # Found a valid long signal
        session.lvl_trades_l += 1
        setup_tag = f"LVLB_{name}"  # B = bounce/buy
        return {
            "direction": 1,
            "stop": stop,
            "target": target,
            "setup": setup_tag,
        }

    return None


def _find_long_target(bar: dict, session) -> Optional[float]:
    """Find the next resistance level ABOVE entry price (long targeting).

    Walks resistance levels from lowest to highest, returns first one above entry.
    """
    entry = bar["close"]
    resistance = _get_resistance_target_map(bar, session)

    for name, price in resistance:
        if price > entry:
            return price

    # Fallback: IB high
    if session.ib_done and not math.isnan(session.ib_high) and session.ib_high > entry:
        return session.ib_high

    return None
