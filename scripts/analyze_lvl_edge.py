#!/usr/bin/env python3
"""Hunt for edge in the Level Rejection infrastructure.

Three approaches:
  A. Different targets for LVL trades (level-to-level may be too tight)
  B. Confluence filtering on IB Breakout (use level map to improve existing setup)
  C. Level map as context filter (broken levels, overnight range, etc.)

Each approach tested with year split.
"""

import sys, os, math
from collections import defaultdict
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown
from backtester.indicators import compute_indicators
from backtester.session import SessionState, update_session
from backtester.position import Position, Trade
from backtester.setups import ib_breakout, ib_rejection, level_rejection, va_fade, eighty_rule, tema_cross


def make_v8():
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    return cfg


def pm(label, trades, show=True):
    m = compute_metrics(trades)
    if show:
        print(f"  {label:<45s}  {m.total_trades:>4d} trades  "
              f"WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
              f"P&L ${m.net_pnl:>+10,.0f}  DD ${m.max_drawdown:>8,.0f}  "
              f"Sharpe {m.sharpe:>5.2f}")
    return m


def pm_year(label, trades_all, trades_y2, trades_y1):
    ma = compute_metrics(trades_all)
    m2 = compute_metrics(trades_y2)
    m1 = compute_metrics(trades_y1)
    both = "BOTH" if m1.net_pnl > 0 and m2.net_pnl > 0 else "FAIL"
    print(f"  {label:<35s}  {ma.total_trades:>4d}t  PF {ma.profit_factor:>5.3f}  "
          f"${ma.net_pnl:>+9,.0f}  |  Y2 PF {m2.profit_factor:>5.3f} ${m2.net_pnl:>+9,.0f}  "
          f"Y1 PF {m1.profit_factor:>5.3f} ${m1.net_pnl:>+9,.0f}  "
          f"Sharpe {ma.sharpe:>5.2f}  {both}")
    return ma, m2, m1


def hdr(title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH A: Different targets for LVL trades
# Level-to-level median target is 11.8pts — too tight. Try wider targets.
# ═══════════════════════════════════════════════════════════════════════════════

def approach_a_targets(df, df_y2, df_y1):
    """Test LVL trades with different target strategies."""
    hdr("APPROACH A: Different Targets for LVL Trades")
    print("  (Level-to-level median = 11.8pts. Testing wider targets.)")

    # We need a custom engine run that overrides LVL targets.
    # Strategy: use level_rejection but replace _find_target with fixed/ib_low/vwap

    for target_type, target_label in [
        ("level", "Level-to-level (default)"),
        ("ib_low", "IB Low"),
        ("vwap", "Session VWAP"),
        ("fixed_20", "Fixed 20pt"),
        ("fixed_25", "Fixed 25pt"),
        ("fixed_30", "Fixed 30pt"),
        ("ib_mid", "IB Mid"),
    ]:
        trades_all = _run_lvl_custom_target(df.copy(), target_type)
        trades_y2 = _run_lvl_custom_target(df_y2.copy(), target_type)
        trades_y1 = _run_lvl_custom_target(df_y1.copy(), target_type)

        # Filter to LVL trades only for analysis
        lvl_all = [t for t in trades_all if t.setup.startswith("LVL_")]
        lvl_y2 = [t for t in trades_y2 if t.setup.startswith("LVL_")]
        lvl_y1 = [t for t in trades_y1 if t.setup.startswith("LVL_")]

        if lvl_all:
            ma = compute_metrics(lvl_all)
            m2 = compute_metrics(lvl_y2) if lvl_y2 else compute_metrics([])
            m1 = compute_metrics(lvl_y1) if lvl_y1 else compute_metrics([])
            both = "BOTH" if m1.net_pnl > 0 and m2.net_pnl > 0 else "FAIL"
            print(f"  {target_label:<25s}  LVL: {ma.total_trades:>4d}t  PF {ma.profit_factor:>5.3f}  "
                  f"${ma.net_pnl:>+9,.0f}  |  Y2 {m2.total_trades:>3d}t ${m2.net_pnl:>+8,.0f}  "
                  f"Y1 {m1.total_trades:>3d}t ${m1.net_pnl:>+8,.0f}  {both}")

        # Also show full system (IB + LVL combined)
        ma_full = compute_metrics(trades_all)
        m2_full = compute_metrics(trades_y2)
        m1_full = compute_metrics(trades_y1)
        both_full = "BOTH" if m1_full.net_pnl > 0 and m2_full.net_pnl > 0 else "FAIL"
        print(f"  {'':<25s}  ALL: {ma_full.total_trades:>4d}t  PF {ma_full.profit_factor:>5.3f}  "
              f"${ma_full.net_pnl:>+9,.0f}  |  Y2 {m2_full.total_trades:>3d}t ${m2_full.net_pnl:>+8,.0f}  "
              f"Y1 {m1_full.total_trades:>3d}t ${m1_full.net_pnl:>+8,.0f}  {both_full}")

    # Also test trigger=any (more trades) with best targets
    print(f"\n  --- trigger=any (more LVL trades) with wider targets ---")
    for target_type, target_label in [
        ("ib_low", "IB Low"),
        ("fixed_25", "Fixed 25pt"),
        ("fixed_30", "Fixed 30pt"),
    ]:
        trades_all = _run_lvl_custom_target(df.copy(), target_type, trigger="any")
        trades_y2 = _run_lvl_custom_target(df_y2.copy(), target_type, trigger="any")
        trades_y1 = _run_lvl_custom_target(df_y1.copy(), target_type, trigger="any")

        lvl_all = [t for t in trades_all if t.setup.startswith("LVL_")]
        lvl_y2 = [t for t in trades_y2 if t.setup.startswith("LVL_")]
        lvl_y1 = [t for t in trades_y1 if t.setup.startswith("LVL_")]

        if lvl_all:
            ma = compute_metrics(lvl_all)
            m2 = compute_metrics(lvl_y2) if lvl_y2 else compute_metrics([])
            m1 = compute_metrics(lvl_y1) if lvl_y1 else compute_metrics([])
            both = "BOTH" if m1.net_pnl > 0 and m2.net_pnl > 0 else "FAIL"
            print(f"  any+{target_label:<20s}  LVL: {ma.total_trades:>4d}t  PF {ma.profit_factor:>5.3f}  "
                  f"${ma.net_pnl:>+9,.0f}  |  Y2 {m2.total_trades:>3d}t ${m2.net_pnl:>+8,.0f}  "
                  f"Y1 {m1.total_trades:>3d}t ${m1.net_pnl:>+8,.0f}  {both}")

    # Test bearish_close trigger with wider targets
    print(f"\n  --- trigger=bearish_close with wider targets ---")
    for target_type, target_label in [
        ("ib_low", "IB Low"),
        ("fixed_25", "Fixed 25pt"),
    ]:
        trades_all = _run_lvl_custom_target(df.copy(), target_type, trigger="bearish_close")
        trades_y2 = _run_lvl_custom_target(df_y2.copy(), target_type, trigger="bearish_close")
        trades_y1 = _run_lvl_custom_target(df_y1.copy(), target_type, trigger="bearish_close")

        lvl_all = [t for t in trades_all if t.setup.startswith("LVL_")]
        lvl_y2 = [t for t in trades_y2 if t.setup.startswith("LVL_")]
        lvl_y1 = [t for t in trades_y1 if t.setup.startswith("LVL_")]

        if lvl_all:
            ma = compute_metrics(lvl_all)
            m2 = compute_metrics(lvl_y2) if lvl_y2 else compute_metrics([])
            m1 = compute_metrics(lvl_y1) if lvl_y1 else compute_metrics([])
            both = "BOTH" if m1.net_pnl > 0 and m2.net_pnl > 0 else "FAIL"
            print(f"  bc+{target_label:<21s}  LVL: {ma.total_trades:>4d}t  PF {ma.profit_factor:>5.3f}  "
                  f"${ma.net_pnl:>+9,.0f}  |  Y2 {m2.total_trades:>3d}t ${m2.net_pnl:>+8,.0f}  "
                  f"Y1 {m1.total_trades:>3d}t ${m1.net_pnl:>+8,.0f}  {both}")


def _compute_custom_target(bar, session, target_type):
    """Compute target based on target_type string."""
    entry = bar["close"]

    if target_type == "level":
        return None  # Use default level-to-level

    elif target_type == "ib_low":
        if session.ib_done and not math.isnan(session.ib_low) and session.ib_low < entry:
            return session.ib_low
        return None

    elif target_type == "vwap":
        if session.rth_vol_sum > 0:
            vwap = session.rth_vwap_sum / session.rth_vol_sum
            if vwap < entry:
                return vwap
        return None

    elif target_type == "ib_mid":
        if session.ib_done and not math.isnan(session.ib_mid) and session.ib_mid < entry:
            return session.ib_mid
        return None

    elif target_type.startswith("fixed_"):
        pts = float(target_type.split("_")[1])
        return entry - pts

    return None


def _run_lvl_custom_target(df, target_type, trigger="failed_break", tema=True):
    """Run backtest with custom LVL targets. Monkey-patches level_rejection._find_target."""
    cfg = make_v8()
    cfg.use_level_reject = True
    cfg.lvl_trigger = trigger
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 8.0
    cfg.lvl_require_tema = tema
    cfg.max_lvl_trades = 4
    cfg.lvl_ibh_wide_only = True
    cfg.lvl_max_tests = 3

    if target_type == "level":
        return run_backtest(df, cfg)

    # Monkey-patch _find_target
    original_find_target = level_rejection._find_target

    def patched_find_target(bar, session):
        result = _compute_custom_target(bar, session, target_type)
        if result is not None:
            return result
        return original_find_target(bar, session)

    level_rejection._find_target = patched_find_target
    try:
        trades = run_backtest(df, cfg)
    finally:
        level_rejection._find_target = original_find_target

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH B: Confluence filtering on IB Breakout
# Use the level map to improve the existing IB Breakout setup.
# ═══════════════════════════════════════════════════════════════════════════════

def approach_b_confluence(df, df_y2, df_y1):
    """Test IB Breakout filtered by level map context."""
    hdr("APPROACH B: Level Map as IB Breakout Filter")

    cfg = make_v8()
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    # Collect per-trade context about nearby levels
    trade_contexts = _collect_trade_context(df, cfg)

    if not trade_contexts:
        print("  No trades with context found.")
        return

    print(f"\n  Total IB trades with context: {len(trade_contexts)}")

    # --- Filter: resistance above entry ---
    print(f"\n  --- IB Short trades: resistance above entry? ---")
    _analyze_filter(trade_contexts, "short", "resistance_above",
                    "Resistance within 15pts above entry")

    # --- Filter: confluence (2+ resistance levels within 10pts of each other) ---
    print(f"\n  --- IB Short trades: resistance confluence? ---")
    _analyze_filter(trade_contexts, "short", "confluence",
                    "2+ resistance levels clustered within 10pts")

    # --- Filter: overnight range ---
    print(f"\n  --- IB Short trades: overnight range size ---")
    _analyze_filter(trade_contexts, "short", "wide_overnight",
                    "Overnight range > 20pts (volatile night)")

    # --- Filter: PDH relationship ---
    print(f"\n  --- IB Short trades: below vs above PDH ---")
    _analyze_filter(trade_contexts, "short", "below_pdh",
                    "Entry price below previous day high")

    # Now run the best filters as actual backtests with year split
    print(f"\n  --- Year-Split: Custom Engine with Confluence Filter ---")
    for filter_name, label in [
        ("none", "v8 baseline (no filter)"),
        ("resistance_near", "Skip short if resistance >20pts above IB"),
        ("below_pdh", "Short only when below PDH"),
        ("on_range_wide", "Short only when ON range > 15pts"),
        ("no_nearby_support", "Skip short if support within 5pts below"),
    ]:
        t_all = _run_ib_with_context_filter(df.copy(), filter_name)
        t_y2 = _run_ib_with_context_filter(df_y2.copy(), filter_name)
        t_y1 = _run_ib_with_context_filter(df_y1.copy(), filter_name)
        pm_year(label, t_all, t_y2, t_y1)


def _collect_trade_context(df, cfg):
    """Run v8 backtest and collect level map context for each IB trade."""
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    pos = Position()
    results = []
    prev_bar = None
    prev_ps = 0
    pending_ctx = None

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)

        cur_ps = 0 if pos.is_flat else pos.direction
        if cur_ps == 0 and prev_ps != 0:
            state.bars_since_exit = 0
        elif cur_ps == 0:
            state.bars_since_exit += 1

        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            if pending_ctx:
                results.append((trade, pending_ctx))
                pending_ctx = None

        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                if pending_ctx:
                    results.append((trade, pending_ctx))
                    pending_ctx = None

        if pos.is_flat:
            signal = None
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)

            if signal is not None and cfg.direction_filter == "short" and signal["direction"] == 1:
                signal = None

            if signal is not None:
                # Capture context
                resistance = level_rejection._get_resistance_map(state)
                support = level_rejection._get_support_map(bar, state)
                entry = bar["close"]

                # Nearest resistance above
                res_above = [(n, p) for n, p in resistance if p > entry]
                nearest_res = min(p - entry for _, p in res_above) if res_above else 999

                # Confluence: how many resistance levels within 10pts of each other
                res_prices = sorted([p for _, p in resistance], reverse=True)
                max_cluster = 1
                for i in range(len(res_prices)):
                    cluster = 1
                    for j in range(i + 1, len(res_prices)):
                        if res_prices[i] - res_prices[j] <= 10:
                            cluster += 1
                    max_cluster = max(max_cluster, cluster)

                # Overnight range
                on_range = 0
                if not math.isnan(state.on_high) and not math.isnan(state.on_low):
                    on_range = state.on_high - state.on_low

                # Below PDH?
                below_pdh = not math.isnan(state.prev_day_high) and entry < state.prev_day_high

                # Nearest support below
                sup_below = [(n, p) for n, p in support if p < entry]
                nearest_sup = min(entry - p for _, p in sup_below) if sup_below else 999

                pending_ctx = {
                    "direction": signal["direction"],
                    "nearest_res_above": nearest_res,
                    "confluence": max_cluster,
                    "on_range": on_range,
                    "below_pdh": below_pdh,
                    "nearest_sup_below": nearest_sup,
                    "n_resistance": len(resistance),
                    "n_support": len(support),
                }

                pos.enter(direction=signal["direction"], price=bar["close"],
                          stop=signal["stop"], target=signal["target"],
                          setup=signal["setup"], time=idx, slippage=cfg.slippage_pts)

        prev_ps = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    return results


def _analyze_filter(contexts, direction, filter_key, desc):
    """Analyze trades split by a boolean filter."""
    shorts = [(t, c) for t, c in contexts if c["direction"] == (-1 if direction == "short" else 1)]

    if not shorts:
        print(f"    No {direction} trades.")
        return

    if filter_key == "resistance_above":
        group_a = [(t, c) for t, c in shorts if c["nearest_res_above"] <= 15]
        group_b = [(t, c) for t, c in shorts if c["nearest_res_above"] > 15]
        label_a, label_b = "Res <=15pt above", "Res >15pt above"
    elif filter_key == "confluence":
        group_a = [(t, c) for t, c in shorts if c["confluence"] >= 2]
        group_b = [(t, c) for t, c in shorts if c["confluence"] < 2]
        label_a, label_b = "Confluence (2+)", "No confluence"
    elif filter_key == "wide_overnight":
        group_a = [(t, c) for t, c in shorts if c["on_range"] > 20]
        group_b = [(t, c) for t, c in shorts if c["on_range"] <= 20]
        label_a, label_b = "ON range >20", "ON range <=20"
    elif filter_key == "below_pdh":
        group_a = [(t, c) for t, c in shorts if c["below_pdh"]]
        group_b = [(t, c) for t, c in shorts if not c["below_pdh"]]
        label_a, label_b = "Below PDH", "At/Above PDH"
    else:
        return

    for label, group in [(label_a, group_a), (label_b, group_b)]:
        trades = [t for t, c in group]
        if trades:
            m = compute_metrics(trades)
            avg = m.net_pnl / m.total_trades if m.total_trades > 0 else 0
            print(f"    {label:<25s}  {m.total_trades:>4d}t  WR {m.win_rate:>5.1f}%  "
                  f"PF {m.profit_factor:>6.3f}  P&L ${m.net_pnl:>+9,.0f}  Avg ${avg:>+6,.0f}")
        else:
            print(f"    {label:<25s}    0t")


def _run_ib_with_context_filter(df, filter_name):
    """Run v8 IB Breakout with level-map based context filter."""
    cfg = make_v8()
    # We need level state tracking for context
    cfg.use_level_reject = False  # Don't take LVL trades, just track state

    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    pos = Position()
    trades = []
    prev_bar = None
    prev_ps = 0

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)
        level_rejection.update_level_state(bar, state, cfg)

        cur_ps = 0 if pos.is_flat else pos.direction
        if cur_ps == 0 and prev_ps != 0:
            state.bars_since_exit = 0
        elif cur_ps == 0:
            state.bars_since_exit += 1

        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                trades.append(trade)

        if pos.is_flat:
            signal = None
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)

            if signal is not None and cfg.direction_filter == "short" and signal["direction"] == 1:
                signal = None

            # Apply context filter
            if signal is not None and filter_name != "none":
                entry = bar["close"]
                if filter_name == "resistance_near":
                    # Skip if nearest resistance is > 20pts above (no ceiling to bounce off)
                    resistance = level_rejection._get_resistance_map(state)
                    res_above = [p for _, p in resistance if p > entry]
                    if res_above and min(p - entry for p in res_above) > 20:
                        signal = None
                elif filter_name == "below_pdh":
                    if math.isnan(state.prev_day_high) or entry >= state.prev_day_high:
                        signal = None
                elif filter_name == "on_range_wide":
                    on_range = 0
                    if not math.isnan(state.on_high) and not math.isnan(state.on_low):
                        on_range = state.on_high - state.on_low
                    if on_range <= 15:
                        signal = None
                elif filter_name == "no_nearby_support":
                    support = level_rejection._get_support_map(bar, state)
                    sup_below = [p for _, p in support if p < entry]
                    if sup_below and min(entry - p for p in sup_below) <= 5:
                        signal = None

            if signal is not None:
                pos.enter(direction=signal["direction"], price=bar["close"],
                          stop=signal["stop"], target=signal["target"],
                          setup=signal["setup"], time=idx, slippage=cfg.slippage_pts)

        prev_ps = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    if not pos.is_flat and prev_bar:
        trade = pos.flatten(prev_bar)
        if trade:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH C: Level map as context — broken levels, ON range, level density
# ═══════════════════════════════════════════════════════════════════════════════

def approach_c_context(df, df_y2, df_y1):
    """Use level infrastructure for context analysis without LVL trades."""
    hdr("APPROACH C: Level Map as Context / New Ideas")

    cfg = make_v8()
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    # --- Idea 1: ON high rejection (not in level_rejection — standalone) ---
    print(f"\n  --- Idea 1: Overnight High Rejection (standalone, like v10 REJ but at ONH) ---")
    for trigger in ["any", "failed_break", "bearish_close"]:
        t_all = _run_on_high_rejection(df.copy(), trigger=trigger)
        t_y2 = _run_on_high_rejection(df_y2.copy(), trigger=trigger)
        t_y1 = _run_on_high_rejection(df_y1.copy(), trigger=trigger)
        onh_all = [t for t in t_all if t.setup == "ONH_REJ"]
        onh_y2 = [t for t in t_y2 if t.setup == "ONH_REJ"]
        onh_y1 = [t for t in t_y1 if t.setup == "ONH_REJ"]
        if onh_all:
            ma = compute_metrics(onh_all)
            m2 = compute_metrics(onh_y2) if onh_y2 else compute_metrics([])
            m1 = compute_metrics(onh_y1) if onh_y1 else compute_metrics([])
            both = "BOTH" if m1.net_pnl > 0 and m2.net_pnl > 0 else "FAIL"
            print(f"    ONH_REJ trigger={trigger:<15s}  {ma.total_trades:>4d}t  PF {ma.profit_factor:>5.3f}  "
                  f"${ma.net_pnl:>+9,.0f}  |  Y2 ${m2.net_pnl:>+8,.0f}  Y1 ${m1.net_pnl:>+8,.0f}  {both}")

    # --- Idea 2: PDH rejection (standalone) ---
    print(f"\n  --- Idea 2: Previous Day High Rejection (standalone) ---")
    for trigger in ["any", "failed_break", "bearish_close"]:
        t_all = _run_pdh_rejection(df.copy(), trigger=trigger)
        t_y2 = _run_pdh_rejection(df_y2.copy(), trigger=trigger)
        t_y1 = _run_pdh_rejection(df_y1.copy(), trigger=trigger)
        pdh_all = [t for t in t_all if t.setup == "PDH_REJ"]
        pdh_y2 = [t for t in t_y2 if t.setup == "PDH_REJ"]
        pdh_y1 = [t for t in t_y1 if t.setup == "PDH_REJ"]
        if pdh_all:
            ma = compute_metrics(pdh_all)
            m2 = compute_metrics(pdh_y2) if pdh_y2 else compute_metrics([])
            m1 = compute_metrics(pdh_y1) if pdh_y1 else compute_metrics([])
            both = "BOTH" if m1.net_pnl > 0 and m2.net_pnl > 0 else "FAIL"
            print(f"    PDH_REJ trigger={trigger:<15s}  {ma.total_trades:>4d}t  PF {ma.profit_factor:>5.3f}  "
                  f"${ma.net_pnl:>+9,.0f}  |  Y2 ${m2.net_pnl:>+8,.0f}  Y1 ${m1.net_pnl:>+8,.0f}  {both}")

    # --- Idea 3: IB Breakout only on days where ON range is significant ---
    print(f"\n  --- Idea 3: IB Breakout gated by ON range ---")
    for on_min in [10, 15, 20, 25, 30]:
        t_all = _run_ib_with_context_filter(df.copy(), "on_range_wide")
        # Re-implement with parameterized threshold
        t_all = _run_ib_on_range_filter(df.copy(), on_min)
        t_y2 = _run_ib_on_range_filter(df_y2.copy(), on_min)
        t_y1 = _run_ib_on_range_filter(df_y1.copy(), on_min)
        pm_year(f"IB short + ON range >= {on_min}pt", t_all, t_y2, t_y1)

    # --- Idea 4: Combined IB Breakout + ONH Rejection (best targets) ---
    print(f"\n  --- Idea 4: v8 + ONH Rejection combined ---")
    for target in ["ib_low", "fixed_25"]:
        for trigger in ["any", "failed_break"]:
            t_all = _run_v8_plus_onh(df.copy(), trigger=trigger, target=target)
            t_y2 = _run_v8_plus_onh(df_y2.copy(), trigger=trigger, target=target)
            t_y1 = _run_v8_plus_onh(df_y1.copy(), trigger=trigger, target=target)
            pm_year(f"v8+ONH {trigger} tgt={target}", t_all, t_y2, t_y1)


def _run_on_high_rejection(df, trigger="any"):
    """Run v8 + standalone overnight high rejection."""
    cfg = make_v8()
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    pos = Position()
    trades = []
    prev_bar = None
    prev_ps = 0
    onh_trades_today = 0
    last_date = None

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)

        cur_date = idx.date()
        if cur_date != last_date:
            onh_trades_today = 0
            last_date = cur_date

        cur_ps = 0 if pos.is_flat else pos.direction
        if cur_ps == 0 and prev_ps != 0:
            state.bars_since_exit = 0
        elif cur_ps == 0:
            state.bars_since_exit += 1

        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                trades.append(trade)

        if pos.is_flat:
            signal = None
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)

                # ONH rejection — check if no IB signal
                if signal is None and state.ib_done and bar["is_trading_window"]:
                    if state.bars_since_exit >= cfg.cooldown_bars and onh_trades_today < 3:
                        onh = state.on_high
                        if not math.isnan(onh) and state.on_frozen:
                            zone = 5.0
                            if bar["high"] >= onh - zone:
                                triggered = False
                                if trigger == "any":
                                    triggered = True
                                elif trigger == "failed_break":
                                    triggered = bar["high"] > onh and bar["close"] <= onh
                                elif trigger == "bearish_close":
                                    triggered = bar["close"] < bar["open"]

                                if triggered:
                                    stop = onh + 8.0
                                    if cfg.pct_stop_mode:
                                        max_s = bar["close"] * cfg.pct_stop_bps / 10000.0
                                        stop = min(stop, bar["close"] + max_s)
                                    if stop <= bar["close"]:
                                        stop = bar["close"] + 2.0
                                    # Target: IB low
                                    target = None
                                    if state.ib_done and not math.isnan(state.ib_low) and state.ib_low < bar["close"]:
                                        target = state.ib_low
                                    if target is not None:
                                        signal = {"direction": -1, "stop": stop,
                                                  "target": target, "setup": "ONH_REJ"}
                                        onh_trades_today += 1

            if signal is not None and cfg.direction_filter == "short" and signal["direction"] == 1:
                signal = None

            if signal is not None:
                pos.enter(direction=signal["direction"], price=bar["close"],
                          stop=signal["stop"], target=signal["target"],
                          setup=signal["setup"], time=idx, slippage=cfg.slippage_pts)

        prev_ps = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    if not pos.is_flat and prev_bar:
        trade = pos.flatten(prev_bar)
        if trade:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

    return trades


def _run_pdh_rejection(df, trigger="any"):
    """Run v8 + standalone PDH rejection."""
    cfg = make_v8()
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    pos = Position()
    trades = []
    prev_bar = None
    prev_ps = 0
    pdh_trades_today = 0
    last_date = None

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)

        cur_date = idx.date()
        if cur_date != last_date:
            pdh_trades_today = 0
            last_date = cur_date

        cur_ps = 0 if pos.is_flat else pos.direction
        if cur_ps == 0 and prev_ps != 0:
            state.bars_since_exit = 0
        elif cur_ps == 0:
            state.bars_since_exit += 1

        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                trades.append(trade)

        if pos.is_flat:
            signal = None
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)

                if signal is None and state.ib_done and bar["is_trading_window"]:
                    if state.bars_since_exit >= cfg.cooldown_bars and pdh_trades_today < 3:
                        pdh = state.prev_day_high
                        if not math.isnan(pdh):
                            zone = 5.0
                            if bar["high"] >= pdh - zone:
                                triggered = False
                                if trigger == "any":
                                    triggered = True
                                elif trigger == "failed_break":
                                    triggered = bar["high"] > pdh and bar["close"] <= pdh
                                elif trigger == "bearish_close":
                                    triggered = bar["close"] < bar["open"]

                                if triggered:
                                    stop = pdh + 8.0
                                    if cfg.pct_stop_mode:
                                        max_s = bar["close"] * cfg.pct_stop_bps / 10000.0
                                        stop = min(stop, bar["close"] + max_s)
                                    if stop <= bar["close"]:
                                        stop = bar["close"] + 2.0
                                    target = None
                                    if state.ib_done and not math.isnan(state.ib_low) and state.ib_low < bar["close"]:
                                        target = state.ib_low
                                    if target is not None:
                                        signal = {"direction": -1, "stop": stop,
                                                  "target": target, "setup": "PDH_REJ"}
                                        pdh_trades_today += 1

            if signal is not None and cfg.direction_filter == "short" and signal["direction"] == 1:
                signal = None

            if signal is not None:
                pos.enter(direction=signal["direction"], price=bar["close"],
                          stop=signal["stop"], target=signal["target"],
                          setup=signal["setup"], time=idx, slippage=cfg.slippage_pts)

        prev_ps = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    if not pos.is_flat and prev_bar:
        trade = pos.flatten(prev_bar)
        if trade:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

    return trades


def _run_ib_on_range_filter(df, on_min):
    """Run IB Breakout gated by overnight range minimum."""
    cfg = make_v8()
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    pos = Position()
    trades = []
    prev_bar = None
    prev_ps = 0

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)

        cur_ps = 0 if pos.is_flat else pos.direction
        if cur_ps == 0 and prev_ps != 0:
            state.bars_since_exit = 0
        elif cur_ps == 0:
            state.bars_since_exit += 1

        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                trades.append(trade)

        if pos.is_flat:
            signal = None
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)

            if signal is not None and cfg.direction_filter == "short" and signal["direction"] == 1:
                signal = None

            # Gate: ON range must be >= threshold
            if signal is not None:
                on_range = 0
                if not math.isnan(state.on_high) and not math.isnan(state.on_low):
                    on_range = state.on_high - state.on_low
                if on_range < on_min:
                    signal = None

            if signal is not None:
                pos.enter(direction=signal["direction"], price=bar["close"],
                          stop=signal["stop"], target=signal["target"],
                          setup=signal["setup"], time=idx, slippage=cfg.slippage_pts)

        prev_ps = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    if not pos.is_flat and prev_bar:
        trade = pos.flatten(prev_bar)
        if trade:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

    return trades


def _run_v8_plus_onh(df, trigger="any", target="ib_low"):
    """Run v8 IB Breakout + ONH Rejection combined."""
    cfg = make_v8()
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    pos = Position()
    trades = []
    prev_bar = None
    prev_ps = 0
    onh_trades_today = 0
    last_date = None

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)

        cur_date = idx.date()
        if cur_date != last_date:
            onh_trades_today = 0
            last_date = cur_date

        cur_ps = 0 if pos.is_flat else pos.direction
        if cur_ps == 0 and prev_ps != 0:
            state.bars_since_exit = 0
        elif cur_ps == 0:
            state.bars_since_exit += 1

        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                trades.append(trade)

        if pos.is_flat:
            signal = None
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)

                if signal is None and state.ib_done and bar["is_trading_window"]:
                    if state.bars_since_exit >= cfg.cooldown_bars and onh_trades_today < 4:
                        onh = state.on_high
                        if not math.isnan(onh) and state.on_frozen:
                            zone = 5.0
                            if bar["high"] >= onh - zone:
                                triggered = False
                                if trigger == "any":
                                    triggered = True
                                elif trigger == "failed_break":
                                    triggered = bar["high"] > onh and bar["close"] <= onh

                                if triggered:
                                    stop = onh + 8.0
                                    if cfg.pct_stop_mode:
                                        max_s = bar["close"] * cfg.pct_stop_bps / 10000.0
                                        stop = min(stop, bar["close"] + max_s)
                                    if stop <= bar["close"]:
                                        stop = bar["close"] + 2.0
                                    tgt = None
                                    if target == "ib_low" and not math.isnan(state.ib_low) and state.ib_low < bar["close"]:
                                        tgt = state.ib_low
                                    elif target == "fixed_25":
                                        tgt = bar["close"] - 25.0
                                    if tgt is not None:
                                        signal = {"direction": -1, "stop": stop,
                                                  "target": tgt, "setup": "ONH_REJ"}
                                        onh_trades_today += 1

            if signal is not None and cfg.direction_filter == "short" and signal["direction"] == 1:
                signal = None

            if signal is not None:
                pos.enter(direction=signal["direction"], price=bar["close"],
                          stop=signal["stop"], target=signal["target"],
                          setup=signal["setup"], time=idx, slippage=cfg.slippage_pts)

        prev_ps = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    if not pos.is_flat and prev_bar:
        trade = pos.flatten(prev_bar)
        if trade:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

    return trades


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/es_5m_databento_2yr.csv"
    print(f"Loading {csv_file}...")
    df = load_tos_csv(csv_file, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    midpoint = len(df) // 2
    df_y2 = df.iloc[:midpoint].copy()
    df_y1 = df.iloc[midpoint:].copy()
    print(f"Year 2: {df_y2.index[0].date()} → {df_y2.index[-1].date()}")
    print(f"Year 1: {df_y1.index[0].date()} → {df_y1.index[-1].date()}")

    # v8 reference
    hdr("V8 BASELINE REFERENCE")
    cfg_v8 = make_v8()
    t_v8 = run_backtest(df.copy(), cfg_v8)
    t_v8_y2 = run_backtest(df_y2.copy(), cfg_v8)
    t_v8_y1 = run_backtest(df_y1.copy(), cfg_v8)
    pm_year("v8 (IB Breakout only)", t_v8, t_v8_y2, t_v8_y1)

    approach_a_targets(df, df_y2, df_y1)
    approach_b_confluence(df, df_y2, df_y1)
    approach_c_context(df, df_y2, df_y1)


if __name__ == "__main__":
    main()
