#!/usr/bin/env python3
"""Deep dive analysis of v8 + ONH Rejection setup.

Base: 881 trades, PF 1.225, +$39.8K, both years profitable.
Goal: find tweaks to improve entries or exits.

Explores:
  ENTRY: trigger type, TEMA filter, zone size, max trades/day, wide IB, ONH>IBH
  EXIT:  target type, stop buffer, time-based exit
  COMBINED: best entry + best exit
"""

import sys, os, math, time
from collections import defaultdict
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics
from backtester.indicators import compute_indicators
from backtester.session import SessionState, update_session
from backtester.position import Position, Trade
from backtester.setups import ib_breakout


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


def hdr(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")


def pm_year(label, trades_all, trades_y2, trades_y1):
    """Print compact year-split line. Returns (metrics_all, metrics_y2, metrics_y1)."""
    ma = compute_metrics(trades_all)
    m2 = compute_metrics(trades_y2) if trades_y2 else compute_metrics([])
    m1 = compute_metrics(trades_y1) if trades_y1 else compute_metrics([])
    both = "BOTH" if m1.net_pnl > 0 and m2.net_pnl > 0 else "FAIL"
    print(f"  {label:<38s}  {ma.total_trades:>4d}t  PF {ma.profit_factor:>5.3f}  "
          f"${ma.net_pnl:>+9,.0f}  |  "
          f"Y2 {m2.total_trades:>3d}t PF {m2.profit_factor:>5.3f} ${m2.net_pnl:>+8,.0f}  "
          f"Y1 {m1.total_trades:>3d}t PF {m1.profit_factor:>5.3f} ${m1.net_pnl:>+8,.0f}  "
          f"Sharpe {ma.sharpe:>5.2f}  {both}")
    return ma, m2, m1


def split_by_setup(trades, setup_prefix):
    """Split trades into those matching setup prefix and the rest."""
    matched = [t for t in trades if t.setup.startswith(setup_prefix)]
    other = [t for t in trades if not t.setup.startswith(setup_prefix)]
    return matched, other


def split_year(trades, midpoint_date):
    """Split trades into Y2 (before midpoint) and Y1 (after midpoint)."""
    y2 = [t for t in trades if t.entry_time.date() < midpoint_date]
    y1 = [t for t in trades if t.entry_time.date() >= midpoint_date]
    return y2, y1


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE: v8 IB Breakout + ONH Rejection (parameterized)
# ─────────────────────────────────────────────────────────────────────────────

def _check_onh_trigger(bar, trigger, onh_price):
    """Check if bar satisfies the ONH trigger condition."""
    if trigger == "any":
        return True
    elif trigger == "bearish_close":
        return bar["close"] < bar["open"]
    elif trigger == "wick":
        bar_mid = (bar["high"] + bar["low"]) / 2
        return bar["close"] < bar["open"] and bar["close"] < bar_mid
    elif trigger == "failed_break":
        return bar["high"] > onh_price and bar["close"] <= onh_price
    return False


def _compute_onh_target(bar, state, target_type):
    """Compute ONH rejection target based on target_type."""
    entry = bar["close"]

    if target_type == "ib_low":
        if state.ib_done and not math.isnan(state.ib_low) and state.ib_low < entry:
            return state.ib_low
    elif target_type == "ib_mid":
        if state.ib_done and not math.isnan(state.ib_mid) and state.ib_mid < entry:
            return state.ib_mid
    elif target_type == "vwap":
        if state.rth_vol_sum > 0:
            vwap = state.rth_vwap_sum / state.rth_vol_sum
            if vwap < entry:
                return vwap
    elif target_type == "prev_poc":
        if not math.isnan(state.prev_poc) and state.prev_poc < entry:
            return state.prev_poc
    elif target_type.startswith("fixed_"):
        pts = float(target_type.split("_")[1])
        return entry - pts

    return None


def run_v8_onh(df, trigger="any", target="ib_low", zone=5.0, stop_buffer=8.0,
               max_onh=4, require_tema=False, wide_only=False, onh_above_ibh=False,
               time_exit_bars=0):
    """Run v8 IB Breakout + ONH Rejection with full parameter control.

    Args:
        df: DataFrame (indicators already computed)
        trigger: "any", "bearish_close", "wick", "failed_break"
        target: "ib_low", "ib_mid", "vwap", "prev_poc", "fixed_15", "fixed_20", etc.
        zone: ONH zone size in points
        stop_buffer: points above ONH for stop
        max_onh: max ONH trades per day
        require_tema: require tema_bearish for ONH entries
        wide_only: only take ONH trades on wide IB days
        onh_above_ibh: require ONH > IBH (overnight pushed above IB = stronger resistance)
        time_exit_bars: if > 0, flatten ONH trades after N bars (0=disabled)

    Returns:
        List of Trade objects (all trades: IB + ONH)
    """
    cfg = make_v8()

    state = SessionState()
    pos = Position()
    trades = []
    prev_bar = None
    prev_ps = 0
    onh_trades_today = 0
    last_date = None
    onh_entry_bar_count = 0  # bars since ONH entry
    is_onh_position = False

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

        # Track bars in ONH position
        if is_onh_position and not pos.is_flat:
            onh_entry_bar_count += 1

        # Check exits (stop/target)
        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)
            is_onh_position = False
            onh_entry_bar_count = 0

        # Time-based exit for ONH trades
        if time_exit_bars > 0 and is_onh_position and not pos.is_flat:
            if onh_entry_bar_count >= time_exit_bars:
                trade = pos.close_at_market(bar, "time_exit")
                if trade is not None:
                    trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                    trades.append(trade)
                    is_onh_position = False
                    onh_entry_bar_count = 0

        # Session flatten
        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                trades.append(trade)
                is_onh_position = False
                onh_entry_bar_count = 0

        # Entry signals
        if pos.is_flat:
            signal = None
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                # Priority 1: IB Breakout
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)

                # Priority 2: ONH Rejection (only if IB didn't fire)
                if signal is None and state.ib_done and bar["is_trading_window"]:
                    if state.bars_since_exit >= cfg.cooldown_bars and onh_trades_today < max_onh:
                        onh = state.on_high
                        if not math.isnan(onh) and state.on_frozen:
                            # Zone check
                            if bar["high"] >= onh - zone:
                                # Wide IB filter
                                if wide_only and not state.is_wide_ib:
                                    pass
                                # ONH above IBH filter
                                elif onh_above_ibh and (math.isnan(state.ib_high) or onh <= state.ib_high):
                                    pass
                                # TEMA filter
                                elif require_tema and not bar.get("tema_bearish", False):
                                    pass
                                else:
                                    # Trigger check
                                    if _check_onh_trigger(bar, trigger, onh):
                                        # Stop
                                        stop = onh + stop_buffer
                                        if cfg.pct_stop_mode:
                                            max_s = bar["close"] * cfg.pct_stop_bps / 10000.0
                                            stop = min(stop, bar["close"] + max_s)
                                        if stop <= bar["close"]:
                                            stop = bar["close"] + 2.0

                                        # Target
                                        tgt = _compute_onh_target(bar, state, target)
                                        if tgt is not None and tgt < bar["close"]:
                                            signal = {"direction": -1, "stop": stop,
                                                      "target": tgt, "setup": "ONH_REJ"}
                                            onh_trades_today += 1

            # Direction filter
            if signal is not None and cfg.direction_filter == "short" and signal["direction"] == 1:
                signal = None

            if signal is not None:
                pos.enter(direction=signal["direction"], price=bar["close"],
                          stop=signal["stop"], target=signal["target"],
                          setup=signal["setup"], time=idx, slippage=cfg.slippage_pts)
                if signal["setup"] == "ONH_REJ":
                    is_onh_position = True
                    onh_entry_bar_count = 0

        prev_ps = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    # Close remaining position
    if not pos.is_flat and prev_bar:
        trade = pos.flatten(prev_bar)
        if trade:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def show_results(label, trades, midpoint_date, show_onh=True, show_all=True):
    """Show ONH-only and ALL trades with year split."""
    onh, ib = split_by_setup(trades, "ONH_REJ")
    onh_y2, onh_y1 = split_year(onh, midpoint_date)
    all_y2, all_y1 = split_year(trades, midpoint_date)

    if show_onh and onh:
        pm_year(f"  ONH: {label}", onh, onh_y2, onh_y1)
    if show_all:
        pm_year(f"  ALL: {label}", trades, all_y2, all_y1)
    return onh, trades


def section_1_trigger(df, midpoint_date):
    """1. Trigger type sweep."""
    hdr("1. ENTRY: Trigger Type Sweep")
    print("  (Base: trigger=any, target=ib_low, zone=5, stop_buf=8, max=4)")

    best_pf = 0
    best_trigger = "any"

    for trigger in ["any", "bearish_close", "wick", "failed_break"]:
        trades = run_v8_onh(df, trigger=trigger)
        onh, _ = show_results(trigger, trades, midpoint_date)
        onh_m = compute_metrics(onh) if onh else compute_metrics([])
        if onh_m.profit_factor > best_pf and onh_m.total_trades >= 30:
            best_pf = onh_m.profit_factor
            best_trigger = trigger

    print(f"\n  >> Best trigger: {best_trigger} (ONH PF {best_pf:.3f})")
    return best_trigger


def section_2_tema(df, midpoint_date):
    """2. TEMA filter for ONH entries."""
    hdr("2. ENTRY: TEMA Bearish Filter")
    print("  (Base: trigger=any, target=ib_low, zone=5, stop_buf=8, max=4)")

    best_pf = 0
    best_tema = False

    for tema in [False, True]:
        label = f"tema={'ON' if tema else 'OFF'}"
        trades = run_v8_onh(df, require_tema=tema)
        onh, _ = show_results(label, trades, midpoint_date)
        onh_m = compute_metrics(onh) if onh else compute_metrics([])
        if onh_m.profit_factor > best_pf and onh_m.total_trades >= 30:
            best_pf = onh_m.profit_factor
            best_tema = tema

    print(f"\n  >> Best TEMA: {'ON' if best_tema else 'OFF'} (ONH PF {best_pf:.3f})")
    return best_tema


def section_3_zone(df, midpoint_date):
    """3. Zone size sweep."""
    hdr("3. ENTRY: Zone Size (pts)")
    print("  (Base: trigger=any, target=ib_low, stop_buf=8, max=4)")

    best_pf = 0
    best_zone = 5.0

    for zone in [3.0, 5.0, 8.0]:
        label = f"zone={zone:.0f}pt"
        trades = run_v8_onh(df, zone=zone)
        onh, _ = show_results(label, trades, midpoint_date)
        onh_m = compute_metrics(onh) if onh else compute_metrics([])
        if onh_m.profit_factor > best_pf and onh_m.total_trades >= 30:
            best_pf = onh_m.profit_factor
            best_zone = zone

    print(f"\n  >> Best zone: {best_zone:.0f}pt (ONH PF {best_pf:.3f})")
    return best_zone


def section_4_max_trades(df, midpoint_date):
    """4. Max ONH trades/day sweep."""
    hdr("4. ENTRY: Max ONH Trades/Day")
    print("  (Base: trigger=any, target=ib_low, zone=5, stop_buf=8)")

    best_pf = 0
    best_max = 4

    for mx in [2, 3, 4, 6]:
        label = f"max={mx}/day"
        trades = run_v8_onh(df, max_onh=mx)
        onh, _ = show_results(label, trades, midpoint_date)
        onh_m = compute_metrics(onh) if onh else compute_metrics([])
        if onh_m.profit_factor > best_pf and onh_m.total_trades >= 30:
            best_pf = onh_m.profit_factor
            best_max = mx

    print(f"\n  >> Best max trades: {best_max}/day (ONH PF {best_pf:.3f})")
    return best_max


def section_5_wide_ib(df, midpoint_date):
    """5. Wide IB only for ONH (like IBH rejection proven edge)."""
    hdr("5. ENTRY: Wide IB Filter")
    print("  (Base: trigger=any, target=ib_low, zone=5, stop_buf=8, max=4)")

    best_pf = 0
    best_wide = False

    for wide in [False, True]:
        label = f"wide_only={'ON' if wide else 'OFF'}"
        trades = run_v8_onh(df, wide_only=wide)
        onh, _ = show_results(label, trades, midpoint_date)
        onh_m = compute_metrics(onh) if onh else compute_metrics([])
        if onh_m.profit_factor > best_pf and onh_m.total_trades >= 20:
            best_pf = onh_m.profit_factor
            best_wide = wide

    print(f"\n  >> Best wide_only: {'ON' if best_wide else 'OFF'} (ONH PF {best_pf:.3f})")
    return best_wide


def section_6_onh_above_ibh(df, midpoint_date):
    """6. ONH must be above IB high (overnight pushed higher)."""
    hdr("6. ENTRY: ONH > IBH Filter")
    print("  (Base: trigger=any, target=ib_low, zone=5, stop_buf=8, max=4)")

    best_pf = 0
    best_above = False

    for above in [False, True]:
        label = f"onh_above_ibh={'ON' if above else 'OFF'}"
        trades = run_v8_onh(df, onh_above_ibh=above)
        onh, _ = show_results(label, trades, midpoint_date)
        onh_m = compute_metrics(onh) if onh else compute_metrics([])
        if onh_m.profit_factor > best_pf and onh_m.total_trades >= 20:
            best_pf = onh_m.profit_factor
            best_above = above

    print(f"\n  >> Best onh_above_ibh: {'ON' if best_above else 'OFF'} (ONH PF {best_pf:.3f})")
    return best_above


def section_7_target(df, midpoint_date):
    """7. Target sweep for ONH trades."""
    hdr("7. EXIT: Target Type Sweep")
    print("  (Base: trigger=any, zone=5, stop_buf=8, max=4)")

    best_pf = 0
    best_target = "ib_low"

    for target in ["ib_low", "ib_mid", "vwap", "fixed_15", "fixed_20",
                    "fixed_25", "fixed_30", "prev_poc"]:
        label = f"target={target}"
        trades = run_v8_onh(df, target=target)
        onh, _ = show_results(label, trades, midpoint_date)
        onh_m = compute_metrics(onh) if onh else compute_metrics([])
        if onh_m.profit_factor > best_pf and onh_m.total_trades >= 30:
            best_pf = onh_m.profit_factor
            best_target = target

    print(f"\n  >> Best target: {best_target} (ONH PF {best_pf:.3f})")
    return best_target


def section_8_stop(df, midpoint_date):
    """8. Stop buffer sweep."""
    hdr("8. EXIT: Stop Buffer (pts above ONH)")
    print("  (Base: trigger=any, target=ib_low, zone=5, max=4)")

    best_pf = 0
    best_stop = 8.0

    for sb in [3.0, 5.0, 8.0, 12.0]:
        label = f"stop_buf={sb:.0f}pt"
        trades = run_v8_onh(df, stop_buffer=sb)
        onh, _ = show_results(label, trades, midpoint_date)
        onh_m = compute_metrics(onh) if onh else compute_metrics([])
        if onh_m.profit_factor > best_pf and onh_m.total_trades >= 30:
            best_pf = onh_m.profit_factor
            best_stop = sb

    print(f"\n  >> Best stop buffer: {best_stop:.0f}pt (ONH PF {best_pf:.3f})")
    return best_stop


def section_9_time_exit(df, midpoint_date):
    """9. Time-based exit for ONH trades."""
    hdr("9. EXIT: Time-Based Exit (flatten ONH after N bars)")
    print("  (Base: trigger=any, target=ib_low, zone=5, stop_buf=8, max=4)")

    best_pf = 0
    best_bars = 0

    for bars in [0, 6, 12, 24]:
        label_map = {0: "disabled", 6: "30min", 12: "1hr", 24: "2hr"}
        label = f"time_exit={label_map[bars]}"
        trades = run_v8_onh(df, time_exit_bars=bars)
        onh, _ = show_results(label, trades, midpoint_date)
        onh_m = compute_metrics(onh) if onh else compute_metrics([])
        if onh_m.profit_factor > best_pf and onh_m.total_trades >= 30:
            best_pf = onh_m.profit_factor
            best_bars = bars

    label_map = {0: "disabled", 6: "30min", 12: "1hr", 24: "2hr"}
    print(f"\n  >> Best time exit: {label_map[best_bars]} (ONH PF {best_pf:.3f})")
    return best_bars


def section_10_combined(df, midpoint_date, best_params):
    """10. Combine best entry tweak + best exit tweak."""
    hdr("10. COMBINED: Best Entry + Best Exit")

    # Unpack best params
    trigger = best_params.get("trigger", "any")
    tema = best_params.get("tema", False)
    zone = best_params.get("zone", 5.0)
    max_onh = best_params.get("max_onh", 4)
    wide = best_params.get("wide", False)
    above = best_params.get("above", False)
    target = best_params.get("target", "ib_low")
    stop_buf = best_params.get("stop_buf", 8.0)
    time_bars = best_params.get("time_bars", 0)

    print(f"  Best entry params: trigger={trigger}, tema={'ON' if tema else 'OFF'}, "
          f"zone={zone:.0f}, max={max_onh}, wide={'ON' if wide else 'OFF'}, "
          f"above={'ON' if above else 'OFF'}")
    label_map = {0: "disabled", 6: "30min", 12: "1hr", 24: "2hr"}
    print(f"  Best exit params:  target={target}, stop_buf={stop_buf:.0f}, "
          f"time_exit={label_map.get(time_bars, str(time_bars))}")
    print()

    # Run baseline first
    print("  --- Baseline (all defaults) ---")
    trades_base = run_v8_onh(df)
    onh_base, _ = split_by_setup(trades_base, "ONH_REJ")
    all_y2, all_y1 = split_year(trades_base, midpoint_date)
    onh_y2, onh_y1 = split_year(onh_base, midpoint_date)
    pm_year("ONH baseline", onh_base, onh_y2, onh_y1)
    pm_year("ALL baseline", trades_base, all_y2, all_y1)

    # Run best entry only (exits at default)
    print("\n  --- Best entry only (default exits) ---")
    trades_entry = run_v8_onh(df, trigger=trigger, require_tema=tema, zone=zone,
                               max_onh=max_onh, wide_only=wide, onh_above_ibh=above)
    show_results("best_entry", trades_entry, midpoint_date)

    # Run best exit only (entries at default)
    print("\n  --- Best exit only (default entries) ---")
    trades_exit = run_v8_onh(df, target=target, stop_buffer=stop_buf,
                              time_exit_bars=time_bars)
    show_results("best_exit", trades_exit, midpoint_date)

    # Run combined
    print("\n  --- Combined best entry + best exit ---")
    trades_combo = run_v8_onh(df, trigger=trigger, require_tema=tema, zone=zone,
                               max_onh=max_onh, wide_only=wide, onh_above_ibh=above,
                               target=target, stop_buffer=stop_buf,
                               time_exit_bars=time_bars)
    onh_combo, _ = split_by_setup(trades_combo, "ONH_REJ")
    onh_y2c, onh_y1c = split_year(onh_combo, midpoint_date)
    all_y2c, all_y1c = split_year(trades_combo, midpoint_date)
    pm_year("ONH combined", onh_combo, onh_y2c, onh_y1c)
    pm_year("ALL combined", trades_combo, all_y2c, all_y1c)

    # Also show exit reason breakdown for ONH combined
    if onh_combo:
        print("\n  --- ONH Combined exit reason breakdown ---")
        by_reason = defaultdict(list)
        for t in onh_combo:
            by_reason[t.exit_reason].append(t)
        for reason, tlist in sorted(by_reason.items()):
            m = compute_metrics(tlist)
            print(f"    {reason:<12s}  {m.total_trades:>4d}t  WR {m.win_rate:>5.1f}%  "
                  f"PF {m.profit_factor:>5.3f}  ${m.net_pnl:>+9,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/es_5m_databento_2yr.csv"
    print(f"Loading {csv_file}...")
    df = load_tos_csv(csv_file, instrument="ES")
    print(f"Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

    # Pre-compute indicators once on the full dataset
    cfg = make_v8()
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    # Year split: first half = Y2 (older), second half = Y1 (recent)
    midpoint = len(df) // 2
    midpoint_date = df.index[midpoint].date()
    print(f"Midpoint: {midpoint_date}")
    print(f"  Y2: {df.index[0].date()} -> {df.index[midpoint-1].date()}")
    print(f"  Y1: {midpoint_date} -> {df.index[-1].date()}")

    # ── Baseline ──
    hdr("BASELINE: v8 + ONH Rejection (defaults)")
    print("  trigger=any, target=ib_low, zone=5, stop_buf=8, max=4, tema=OFF, wide=OFF")
    t0 = time.time()
    trades_base = run_v8_onh(df)
    onh_base, ib_base = split_by_setup(trades_base, "ONH_REJ")
    onh_y2, onh_y1 = split_year(onh_base, midpoint_date)
    ib_y2, ib_y1 = split_year(ib_base, midpoint_date)
    all_y2, all_y1 = split_year(trades_base, midpoint_date)
    pm_year("IB only", ib_base, ib_y2, ib_y1)
    pm_year("ONH only", onh_base, onh_y2, onh_y1)
    pm_year("ALL (IB + ONH)", trades_base, all_y2, all_y1)
    dt = time.time() - t0
    print(f"\n  Baseline engine time: {dt:.1f}s")

    # Exit reason breakdown for ONH baseline
    if onh_base:
        print("\n  ONH exit reason breakdown:")
        by_reason = defaultdict(list)
        for t in onh_base:
            by_reason[t.exit_reason].append(t)
        for reason, tlist in sorted(by_reason.items()):
            m = compute_metrics(tlist)
            print(f"    {reason:<12s}  {m.total_trades:>4d}t  WR {m.win_rate:>5.1f}%  "
                  f"PF {m.profit_factor:>5.3f}  ${m.net_pnl:>+9,.0f}")

    # ── Run all sections ──
    best_params = {}

    best_params["trigger"] = section_1_trigger(df, midpoint_date)
    best_params["tema"] = section_2_tema(df, midpoint_date)
    best_params["zone"] = section_3_zone(df, midpoint_date)
    best_params["max_onh"] = section_4_max_trades(df, midpoint_date)
    best_params["wide"] = section_5_wide_ib(df, midpoint_date)
    best_params["above"] = section_6_onh_above_ibh(df, midpoint_date)
    best_params["target"] = section_7_target(df, midpoint_date)
    best_params["stop_buf"] = section_8_stop(df, midpoint_date)
    best_params["time_bars"] = section_9_time_exit(df, midpoint_date)

    # ── Combined best ──
    section_10_combined(df, midpoint_date, best_params)

    # ── Summary ──
    hdr("SUMMARY OF BEST PARAMETERS")
    label_map = {0: "disabled", 6: "30min", 12: "1hr", 24: "2hr"}
    print(f"  trigger:       {best_params['trigger']}")
    print(f"  tema_filter:   {'ON' if best_params['tema'] else 'OFF'}")
    print(f"  zone:          {best_params['zone']:.0f} pts")
    print(f"  max_onh/day:   {best_params['max_onh']}")
    print(f"  wide_only:     {'ON' if best_params['wide'] else 'OFF'}")
    print(f"  onh_above_ibh: {'ON' if best_params['above'] else 'OFF'}")
    print(f"  target:        {best_params['target']}")
    print(f"  stop_buffer:   {best_params['stop_buf']:.0f} pts")
    print(f"  time_exit:     {label_map.get(best_params['time_bars'], str(best_params['time_bars']))}")

    total_time = time.time() - t0
    print(f"\n  Total analysis time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
