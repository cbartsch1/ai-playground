#!/usr/bin/env python3
"""Debug script: investigate why no REJ trades fired on Feb 23, 2026.

NOTE: The 2yr data file (es_5m_databento_2yr.csv) ends at Feb 13, 2026.
Feb 23, 2026 is NOT in the data. This script:
  1. Reports that finding clearly
  2. Shows IB/REJ analysis for the last available week (Feb 5-13, 2026)
     so you can see the rolling IB avg context and how REJ zones look
  3. Picks a sample day and walks through the full REJ signal check

Usage:
    cd /Users/chuck_mf_norris/projects/backtesting/es
    python scripts/debug_feb23.py data/es_5m_databento_2yr.csv
"""

import sys
import math
import pandas as pd

# Add project root to path
sys.path.insert(0, "/Users/chuck_mf_norris/projects/backtesting/es")

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.indicators import compute_indicators
from backtester.session import SessionState, update_session
from backtester.setups import ib_rejection


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_feb23.py <data_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # -- Config: v8 + REJ settings --
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8

    # -- Load data --
    print(f"Loading data from {csv_path} ...")
    df = load_tos_csv(csv_path)
    print(f"  Loaded {len(df)} bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # -- Compute indicators --
    compute_indicators(
        df,
        tema_fast=cfg.tema_fast,
        tema_slow=cfg.tema_slow,
        tema_trend=cfg.tema_trend,
        atr_len=cfg.atr_len,
        atr_avg_len=cfg.atr_avg_len,
    )

    # -- Check what dates exist --
    all_session_dates = sorted(df["session_date"].dropna().unique())
    last_date = all_session_dates[-1]
    print(f"  Last session date in data: {last_date}")

    # Check if Feb 23 exists
    feb23 = pd.Timestamp("2026-02-23").date()
    if feb23 not in all_session_dates:
        print(f"\n  *** Feb 23, 2026 is NOT in the data file. ***")
        print(f"  The data ends at {last_date}.")
        print(f"  To debug Feb 23 specifically, you need to download")
        print(f"  updated data from Databento covering that date.")
        print(f"\n  Proceeding with analysis of the LAST AVAILABLE WEEK")
        print(f"  to show IB context and how REJ signals would work.")

    # Use the last 7 trading days available
    target_dates = all_session_dates[-7:]
    print(f"\n  Analyzing dates: {[str(d) for d in target_dates]}")

    # -- Run session state bar-by-bar --
    state = SessionState()
    prev_bar = None

    daily_ib_info = {}
    daily_tw_bars = {}  # date -> list of trading window bars
    daily_rth_bars = {}  # date -> list of all RTH bars

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx

        update_session(state, bar, prev_bar, cfg)

        session_date = bar.get("session_date")

        if session_date in target_dates:
            date_key = str(session_date)

            # Capture state right after IB completes
            if state.ib_done and date_key not in daily_ib_info:
                daily_ib_info[date_key] = {
                    "ib_high": state.ib_high,
                    "ib_low": state.ib_low,
                    "ib_range": state.ib_high - state.ib_low if not math.isnan(state.ib_high) else 0,
                    "ib_range_avg": state.ib_range_avg,
                    "ib_ratio": state.ib_ratio,
                    "is_wide_ib": state.is_wide_ib,
                    "is_narrow_ib": state.is_narrow_ib,
                    "is_normal_ib": state.is_normal_ib,
                    "weekday": session_date.weekday(),
                }

            if date_key not in daily_tw_bars:
                daily_tw_bars[date_key] = []
                daily_rth_bars[date_key] = []

            if bar["is_rth"]:
                daily_rth_bars[date_key].append({
                    "time": idx,
                    "et_time": bar["et_time"],
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "is_trading_window": bar["is_trading_window"],
                    "is_ib_period": bar["is_ib_period"],
                    "tema_bearish": bar.get("tema_bearish", None),
                })

            if bar["is_trading_window"]:
                daily_tw_bars[date_key].append({
                    "time": idx,
                    "et_time": bar["et_time"],
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "tema_bearish": bar.get("tema_bearish", None),
                })

        prev_bar = bar

    # ==================================================================
    # REPORT
    # ==================================================================

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    print("\n" + "=" * 80)
    print("IB VALUES -- LAST 7 TRADING DAYS")
    print("=" * 80)
    print(f"\n  {'Date':>12s}  {'Day':>9s}  {'IB High':>8s}  {'IB Low':>8s}  {'IB Rng':>7s}  "
          f"{'IB Avg':>7s}  {'Ratio':>6s}  {'Type':>8s}  {'Wide?':>5s}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*5}")

    for date_str in sorted(daily_ib_info.keys()):
        info = daily_ib_info[date_str]
        wd = info["weekday"]
        day_type = "WIDE" if info["is_wide_ib"] else ("NARROW" if info["is_narrow_ib"] else "Normal")
        wide_flag = "YES" if info["is_wide_ib"] else ""
        print(f"  {date_str:>12s}  {day_names[wd]:>9s}  {info['ib_high']:>8.2f}  {info['ib_low']:>8.2f}  "
              f"{info['ib_range']:>7.2f}  {info['ib_range_avg']:>7.2f}  {info['ib_ratio']:>6.3f}  "
              f"{day_type:>8s}  {wide_flag:>5s}")

    # -- For each day, show REJ zone analysis --
    print("\n" + "=" * 80)
    print("REJ ZONE ANALYSIS -- EACH DAY")
    print("=" * 80)
    print(f"\n  REJ Config: trigger={cfg.rej_trigger}, target={cfg.rej_target}, "
          f"zone={cfg.rej_zone_pts}pts, stop_buf={cfg.rej_stop_buffer}pts, "
          f"require_tema={cfg.rej_require_tema}, max_trades={cfg.max_rej_trades}")

    for date_str in sorted(daily_ib_info.keys()):
        info = daily_ib_info[date_str]
        wd = info["weekday"]
        ib_high = info["ib_high"]
        ib_low = info["ib_low"]
        rej_zone_floor = ib_high - cfg.rej_zone_pts

        tw_bars = daily_tw_bars.get(date_str, [])
        if not tw_bars:
            continue

        max_high = max(b["high"] for b in tw_bars)
        bars_in_zone = sum(1 for b in tw_bars if b["high"] >= rej_zone_floor)
        bars_in_blackout = sum(1 for b in tw_bars
                               if cfg.blackout_start <= b["et_time"] < cfg.blackout_end)
        bars_in_zone_not_blackout = sum(
            1 for b in tw_bars
            if b["high"] >= rej_zone_floor
            and not (cfg.blackout_start <= b["et_time"] < cfg.blackout_end)
        )
        is_friday = wd == 4

        print(f"\n  {date_str} ({day_names[wd]}):")
        print(f"    IB High={ib_high:.2f}, IB Low={ib_low:.2f}, IB Range={info['ib_range']:.2f}")
        print(f"    REJ Zone: {rej_zone_floor:.2f} to {ib_high:.2f}")
        print(f"    Stop: {ib_high + cfg.rej_stop_buffer:.2f}, Target (IB Low): {ib_low:.2f}")
        print(f"    TW bars: {len(tw_bars)}, Bars in zone: {bars_in_zone}, "
              f"Zone (excl blackout): {bars_in_zone_not_blackout}")
        print(f"    Max high in TW: {max_high:.2f}")
        if max_high < rej_zone_floor:
            print(f"    Gap to zone: {rej_zone_floor - max_high:.2f} pts BELOW zone floor")
        else:
            print(f"    Penetration: {max_high - rej_zone_floor:.2f} pts INTO zone")

        if is_friday and cfg.skip_friday:
            print(f"    FRIDAY FILTER: BLOCKED (skip_friday=True)")
        elif bars_in_zone == 0:
            print(f"    RESULT: No REJ signal -- price never reached zone")
        elif bars_in_zone_not_blackout == 0:
            print(f"    RESULT: No REJ signal -- zone bars all in blackout")
        else:
            print(f"    RESULT: {bars_in_zone_not_blackout} bar(s) eligible for REJ signal")

    # -- Deep dive: pick last non-Friday day with zone bars for detailed walkthrough --
    print("\n" + "=" * 80)
    print("DETAILED BAR-BY-BAR WALKTHROUGH -- LAST DAY WITH ZONE ACTIVITY")
    print("=" * 80)

    deep_dive_date = None
    for date_str in sorted(daily_ib_info.keys(), reverse=True):
        info = daily_ib_info[date_str]
        if info["weekday"] == 4:
            continue  # skip Friday
        ib_high = info["ib_high"]
        rej_zone_floor = ib_high - cfg.rej_zone_pts
        tw_bars = daily_tw_bars.get(date_str, [])
        if any(b["high"] >= rej_zone_floor for b in tw_bars):
            deep_dive_date = date_str
            break

    if deep_dive_date is None:
        # Just use the last non-Friday day
        for date_str in sorted(daily_ib_info.keys(), reverse=True):
            if daily_ib_info[date_str]["weekday"] != 4:
                deep_dive_date = date_str
                break

    if deep_dive_date:
        info = daily_ib_info[deep_dive_date]
        ib_high = info["ib_high"]
        ib_low = info["ib_low"]
        rej_zone_floor = ib_high - cfg.rej_zone_pts
        tw_bars = daily_tw_bars[deep_dive_date]

        print(f"\n  Date: {deep_dive_date} ({day_names[info['weekday']]})")
        print(f"  IB High: {ib_high:.2f}, IB Low: {ib_low:.2f}")
        print(f"  REJ Zone: {rej_zone_floor:.2f} to {ib_high:.2f}")
        print(f"\n  {'Time':>20s}  {'ET':>6s}  {'Open':>8s}  {'High':>8s}  {'Low':>8s}  "
              f"{'Close':>8s}  {'TEMA':>6s}  {'InZone':>6s}  {'Notes':s}")

        for b in tw_bars:
            in_zone = b["high"] >= rej_zone_floor
            in_blackout = cfg.blackout_start <= b["et_time"] < cfg.blackout_end
            zone_tag = "YES" if in_zone else "no"
            tema_tag = "bear" if b["tema_bearish"] else "BULL"
            notes = ""
            if in_blackout:
                notes += "[BLACKOUT] "
            if in_zone and not in_blackout:
                # Check target validity
                target_ok = ib_low < b["close"]
                stop = ib_high + cfg.rej_stop_buffer
                stop_ok = stop > b["close"]
                if target_ok and stop_ok:
                    notes += "[REJ SIGNAL OK]"
                elif not target_ok:
                    notes += f"[target {ib_low:.0f} >= close {b['close']:.0f}]"
                elif not stop_ok:
                    notes += f"[stop {stop:.0f} <= close {b['close']:.0f}]"

            print(f"  {str(b['time']):>20s}  {b['et_time']:>6d}  {b['open']:>8.2f}  {b['high']:>8.2f}  "
                  f"{b['low']:>8.2f}  {b['close']:>8.2f}  {tema_tag:>6s}  {zone_tag:>6s}  {notes}")

    # -- Final: run actual engine signal check on the deep-dive day --
    if deep_dive_date:
        print(f"\n  {'_' * 76}")
        print(f"  ACTUAL ENGINE SIGNAL CHECK -- {deep_dive_date}")
        print(f"  {'_' * 76}")

        state2 = SessionState()
        prev_bar2 = None
        signal_count = 0
        dd_date = pd.Timestamp(deep_dive_date).date()

        for idx, row in df.iterrows():
            bar = row.to_dict()
            bar["_time"] = idx
            update_session(state2, bar, prev_bar2, cfg)

            session_date = bar.get("session_date")
            if session_date == dd_date and bar["is_trading_window"]:
                # Check blackout
                et_time = bar["et_time"]
                in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                               and cfg.blackout_start <= et_time < cfg.blackout_end)

                if not in_blackout:
                    signal = ib_rejection.check_signal(bar, prev_bar2, state2, cfg)
                    if signal:
                        signal_count += 1
                        print(f"\n    SIGNAL #{signal_count} at {idx} (ET {et_time}):")
                        print(f"      Direction: {'SHORT' if signal['direction'] == -1 else 'LONG'}")
                        print(f"      Entry (close): {bar['close']:.2f}")
                        print(f"      Stop: {signal['stop']:.2f}")
                        print(f"      Target: {signal['target']:.2f}")
                        print(f"      Setup: {signal['setup']}")
                        risk = signal['stop'] - bar['close']
                        reward = bar['close'] - signal['target']
                        rr = reward / risk if risk > 0 else 0
                        print(f"      Risk: {risk:.2f} pts, Reward: {reward:.2f} pts, R:R = {rr:.2f}")

            prev_bar2 = bar

        if signal_count == 0:
            print(f"\n    No REJ signals fired on {deep_dive_date}.")
            print(f"    (This is expected if price never reached the zone,")
            print(f"     or if target >= close on all zone bars.)")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"\n  The data file ends at {df.index[-1].date()}.")
    print(f"  Feb 23, 2026 is NOT in the data.")
    print(f"  To debug Feb 23 specifically, download fresh Databento data:")
    print(f"    - Use the Databento Python client")
    print(f"    - Request ES.FUT from CME GLBX.MDP3")
    print(f"    - Date range through at least 2026-02-23")
    print(f"    - Resample to 5min OHLCV, save as CSV")
    print(f"    - Then re-run this script with the updated file")
    print(f"\n  The analysis above shows how the REJ setup behaves on the")
    print(f"  last available days, so you can see the IB context and")
    print(f"  whether the zone is being reached in recent trading.")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
