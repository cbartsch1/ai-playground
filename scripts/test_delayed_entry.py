#!/usr/bin/env python3
"""Test delayed/limit entry: wait for price to push X pts above signal before entering short.

Concept: if winners need 6-8 pts of room before working, why not wait for that push
and enter at a better price with a tighter stop?

Approach:
  - Normal signal fires at bar close (same filters as always)
  - Instead of entering immediately, set a limit entry at close + offset pts
  - On subsequent bars, if high >= limit price, enter short there
  - Stop = entry + stop_buffer (tight, e.g. 3pt)
  - Target = same 3rd support as before
  - Limit expires after N bars if never filled

All 3 contracts target the 3rd support (uniform_skip=2).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import List, Optional

from backtester.config import StrategyConfig
from backtester.indicators import compute_indicators
from backtester.session import SessionState, update_session
from backtester.position import Position, Trade
from backtester.setups import ib_breakout, ib_rejection, level_rejection


def make_cfg(stop_buffer):
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = True
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = stop_buffer
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = True
    cfg.lvl_ma_filter = "tema"
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = stop_buffer
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_min_rr = 0.5
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3
    return cfg


def _finalize_trade(trade, cfg):
    trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)


def run_delayed_entry(df, cfg, n_contracts=3, entry_offset=5.0,
                      limit_stop_buffer=3.0, timeout_bars=6):
    """Run backtest with delayed/limit entry for LVL trades.

    When LVL signal fires:
      - Place limit entry at signal_price + entry_offset (for shorts, higher = better)
      - If bar high reaches limit within timeout_bars, enter at limit price
      - Stop = entry_price + limit_stop_buffer
      - Target = same 3rd support

    Baseline IB/REJ trades still enter immediately (no change).
    """
    compute_indicators(
        df,
        tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
        tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
        atr_avg_len=cfg.atr_avg_len,
    )

    state = SessionState()
    baseline_pos = Position()
    lvl_positions = [Position() for _ in range(n_contracts)]
    trades: List[Trade] = []
    prev_bar = None
    was_any_active = False

    # Pending limit orders: list of dicts with limit_price, target, timeout_remaining, etc.
    pending_limits = []

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx

        update_session(state, bar, prev_bar, cfg)
        level_rejection.update_level_state(bar, state, cfg)

        # Cooldown tracking
        any_active_now = (not baseline_pos.is_flat or
                          any(not p.is_flat for p in lvl_positions))
        if not any_active_now and was_any_active:
            state.bars_since_exit = 0
        elif not any_active_now:
            state.bars_since_exit += 1

        # Check exits — baseline
        trade = baseline_pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            _finalize_trade(trade, cfg)
            trades.append(trade)

        # Check exits — LVL
        for pos in lvl_positions:
            trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
            if trade is not None:
                _finalize_trade(trade, cfg)
                trades.append(trade)

        # Session flatten
        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time:
            if not baseline_pos.is_flat:
                trade = baseline_pos.flatten(bar)
                if trade is not None:
                    _finalize_trade(trade, cfg)
                    trades.append(trade)
            for pos in lvl_positions:
                if not pos.is_flat:
                    trade = pos.flatten(bar)
                    if trade is not None:
                        _finalize_trade(trade, cfg)
                        trades.append(trade)
            # Cancel pending limits at flatten time
            pending_limits.clear()

        # Check if any pending limit orders fill on this bar
        all_flat = (baseline_pos.is_flat and
                    all(p.is_flat for p in lvl_positions))

        if pending_limits and all_flat:
            filled = False
            for pending in pending_limits:
                if bar["high"] >= pending["limit_price"]:
                    # Filled! Enter all contracts
                    entry_price = pending["limit_price"]
                    stop = entry_price + limit_stop_buffer
                    # Cap stop with pct stop
                    if cfg.pct_stop_mode:
                        max_stop_pts = entry_price * cfg.pct_stop_bps / 10000.0
                        pct_stop = entry_price + max_stop_pts
                        stop = min(stop, pct_stop)

                    for c_idx in range(n_contracts):
                        target = pending["targets"][c_idx] if c_idx < len(pending["targets"]) else None
                        if target is not None:
                            lvl_positions[c_idx].enter(
                                direction=-1,
                                price=entry_price,
                                stop=stop,
                                target=target,
                                setup=pending["setup"],
                                time=idx,
                                slippage=cfg.slippage_pts,
                                contract=c_idx + 1,
                            )
                    filled = True
                    break

            if filled:
                pending_limits.clear()
            else:
                # Decrement timeouts, remove expired
                for p in pending_limits:
                    p["timeout"] -= 1
                pending_limits = [p for p in pending_limits if p["timeout"] > 0]

        # Check for new entry signals (only if ALL flat and no pending limits)
        all_flat = (baseline_pos.is_flat and
                    all(p.is_flat for p in lvl_positions))

        if all_flat and not pending_limits:
            signal = None

            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4
            time_ok = not (in_blackout or (cfg.skip_friday and is_friday))

            if time_ok:
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = ib_rejection.check_signal(bar, prev_bar, state, cfg)

            if signal is not None:
                if cfg.direction_filter != "both":
                    if cfg.direction_filter == "short" and signal["direction"] == 1:
                        signal = None
                    elif cfg.direction_filter == "long" and signal["direction"] == -1:
                        signal = None

                if signal is not None:
                    baseline_pos.enter(
                        direction=signal["direction"],
                        price=bar["close"],
                        stop=signal["stop"],
                        target=signal["target"],
                        setup=signal["setup"],
                        time=idx,
                        slippage=cfg.slippage_pts,
                    )
            else:
                # Check LVL — but create pending limit instead of immediate entry
                lvl_ok = time_ok or cfg.lvl_own_filters

                if lvl_ok:
                    lvl_signals = level_rejection.check_signal_multi(
                        bar, prev_bar, state, cfg,
                        n_contracts=n_contracts, uniform_skip=2,
                    )

                    if lvl_signals:
                        limit_price = bar["close"] + entry_offset
                        targets = [sig["target"] for sig in lvl_signals]

                        pending_limits.append({
                            "limit_price": limit_price,
                            "targets": targets,
                            "setup": lvl_signals[0]["setup"],
                            "timeout": timeout_bars,
                        })

        was_any_active = (not baseline_pos.is_flat or
                          any(not p.is_flat for p in lvl_positions))
        prev_bar = bar

    # Close remaining
    if prev_bar is not None:
        if not baseline_pos.is_flat:
            trade = baseline_pos.flatten(prev_bar)
            if trade is not None:
                _finalize_trade(trade, cfg)
                trades.append(trade)
        for pos in lvl_positions:
            if not pos.is_flat:
                trade = pos.flatten(prev_bar)
                if trade is not None:
                    _finalize_trade(trade, cfg)
                    trades.append(trade)

    return trades


def analyze(trades, label):
    lvl = [t for t in trades if t.setup.startswith("LVL")]
    base = [t for t in trades if not t.setup.startswith("LVL")]

    if not lvl:
        return None

    pnls = [t.pnl_dollar for t in lvl]
    total = sum(pnls)
    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gw / gl if gl > 0 else float("inf")
    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100

    targets = [t for t in lvl if t.exit_reason == "target"]
    stops = [t for t in lvl if t.exit_reason == "stop"]
    flattens = [t for t in lvl if t.exit_reason == "flatten"]

    avg_tgt_pts = np.mean([t.entry_price - t.target for t in lvl]) if lvl else 0
    avg_stp_pts = np.mean([t.stop - t.entry_price for t in lvl]) if lvl else 0
    avg_rr = avg_tgt_pts / avg_stp_pts if avg_stp_pts > 0 else 0
    avg_stop_loss = np.mean([t.pnl_dollar for t in stops]) if stops else 0
    avg_tgt_win = np.mean([t.pnl_dollar for t in targets]) if targets else 0
    base_pnl = sum(t.pnl_dollar for t in base)

    return {
        "label": label,
        "lvl_fills": len(lvl),
        "pf": pf,
        "wr": wr,
        "lvl_pnl": total,
        "target_gain": sum(t.pnl_dollar for t in targets),
        "stop_loss": sum(t.pnl_dollar for t in stops),
        "flatten_pnl": sum(t.pnl_dollar for t in flattens),
        "tgt_count": len(targets),
        "stp_count": len(stops),
        "flt_count": len(flattens),
        "avg_tgt_pts": avg_tgt_pts,
        "avg_stp_pts": avg_stp_pts,
        "avg_rr": avg_rr,
        "avg_stop_loss": avg_stop_loss,
        "avg_tgt_win": avg_tgt_win,
        "base_pnl": base_pnl,
        "combined": total + base_pnl,
    }


def print_row(r):
    print(f"  {r['label']:>22}  {r['lvl_fills']:>5}  {r['pf']:>5.2f}  {r['wr']:>5.1f}%  "
          f"${r['lvl_pnl']:>+9,.0f}  "
          f"{r['tgt_count']:>4}  {r['stp_count']:>4}  {r['flt_count']:>4}  "
          f"{r['avg_tgt_pts']:>6.1f}  {r['avg_stp_pts']:>5.1f}  {r['avg_rr']:>5.2f}  "
          f"${r['avg_tgt_win']:>+7,.0f}  ${r['avg_stop_loss']:>+7,.0f}  "
          f"${r['combined']:>+9,.0f}")


def print_header():
    print(f"  {'Config':>22}  {'Fills':>5}  {'PF':>5}  {'WR':>6}  "
          f"{'LVL P&L':>10}  "
          f"{'Tgt#':>4}  {'Stp#':>4}  {'Flt#':>4}  "
          f"{'TgtPt':>6}  {'StpPt':>5}  {'R:R':>5}  "
          f"{'AvgWin$':>8}  {'AvgStp$':>8}  "
          f"{'Combined':>10}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*5}  {'-'*6}  "
          f"{'-'*10}  "
          f"{'-'*4}  {'-'*4}  {'-'*4}  "
          f"{'-'*6}  {'-'*5}  {'-'*5}  "
          f"{'-'*8}  {'-'*8}  "
          f"{'-'*10}")


def main():
    from backtester.data_loader import load_tos_csv
    from backtester.stagger_engine import run_backtest_stagger

    df = load_tos_csv("data/es_5m_databento_2yr.csv")
    split = "2025-02-14"
    df_is = df[df.index < split]
    df_oos = df[df.index >= split]

    print("=" * 145)
    print("  DELAYED ENTRY TEST — Wait for price push, enter at better price, tight stop")
    print("  All configs: 3 contracts targeting 3rd support")
    print("=" * 145)

    # Configs: (label, entry_offset, limit_stop_buffer, timeout_bars, is_delayed)
    configs = [
        # Baselines (immediate entry)
        ("Immediate / 7pt stop", 0, 7, 0, False),
        ("Immediate / 3pt stop", 0, 3, 0, False),
        # Delayed entry variants
        ("Wait +3pt / 3pt stop", 3, 3, 6, True),
        ("Wait +5pt / 3pt stop", 5, 3, 6, True),
        ("Wait +6pt / 3pt stop", 6, 3, 6, True),
        ("Wait +8pt / 3pt stop", 8, 3, 6, True),
        ("Wait +5pt / 5pt stop", 5, 5, 6, True),
        ("Wait +8pt / 5pt stop", 8, 5, 6, True),
    ]

    for period_label, df_slice in [("OUT-OF-SAMPLE", df_oos),
                                    ("IN-SAMPLE", df_is),
                                    ("FULL 2-YEAR", df)]:
        print(f"\n  {period_label}:")
        print_header()

        results = {}
        for label, offset, stop_buf, timeout, is_delayed in configs:
            if is_delayed:
                cfg = make_cfg(stop_buf)  # stop_buf used as lvl_stop_buffer for signal generation
                trades = run_delayed_entry(df_slice.copy(), cfg,
                                           n_contracts=3,
                                           entry_offset=offset,
                                           limit_stop_buffer=stop_buf,
                                           timeout_bars=timeout)
            else:
                cfg = make_cfg(stop_buf)
                trades = run_backtest_stagger(df_slice.copy(), cfg,
                                             n_contracts=3, uniform_skip=2)
            r = analyze(trades, label)
            if r:
                results[label] = r
                print_row(r)

        if period_label == "OUT-OF-SAMPLE":
            oos_results = results
        elif period_label == "IN-SAMPLE":
            is_results = results

    # Walk-forward
    print("\n" + "=" * 145)
    print("  WALK-FORWARD PF RATIOS")
    print("=" * 145)
    for label, _, _, _, _ in configs:
        oos_r = oos_results.get(label)
        is_r = is_results.get(label)
        if oos_r and is_r and is_r["pf"] > 0:
            ratio = oos_r["pf"] / is_r["pf"]
            grade = "ROBUST" if ratio >= 0.7 else "ACCEPTABLE" if ratio >= 0.5 else "WEAK"
            print(f"  {label:>22}:  IS PF {is_r['pf']:.2f}  ->  OOS PF {oos_r['pf']:.2f}  "
                  f"->  Ratio {ratio:.2f}  {grade}")

    # Summary
    print("\n" + "=" * 145)
    print("  KEY COMPARISON")
    print("=" * 145)

    baseline = oos_results.get("Immediate / 7pt stop")
    best_delayed = None
    best_delayed_pnl = -float("inf")
    for label, offset, stop_buf, timeout, is_delayed in configs:
        if is_delayed and label in oos_results:
            r = oos_results[label]
            if r["lvl_pnl"] > best_delayed_pnl:
                best_delayed_pnl = r["lvl_pnl"]
                best_delayed = r

    if baseline and best_delayed:
        print(f"\n  Current best (immediate / 7pt):")
        print(f"    {baseline['lvl_fills']} fills | PF {baseline['pf']:.2f} | "
              f"WR {baseline['wr']:.0f}% | LVL P&L ${baseline['lvl_pnl']:+,.0f} | "
              f"R:R {baseline['avg_rr']:.1f}")
        print(f"\n  Best delayed entry:")
        print(f"    {best_delayed['lvl_fills']} fills | PF {best_delayed['pf']:.2f} | "
              f"WR {best_delayed['wr']:.0f}% | LVL P&L ${best_delayed['lvl_pnl']:+,.0f} | "
              f"R:R {best_delayed['avg_rr']:.1f}")
        diff = best_delayed['lvl_pnl'] - baseline['lvl_pnl']
        print(f"\n  Difference: ${diff:+,.0f} "
              f"({'delayed wins' if diff > 0 else 'immediate wins'})")

        # Fill rate analysis
        if baseline["lvl_fills"] > 0:
            fill_pct = best_delayed["lvl_fills"] / baseline["lvl_fills"] * 100
            print(f"  Fill rate: {best_delayed['lvl_fills']}/{baseline['lvl_fills']} = "
                  f"{fill_pct:.0f}% of signals actually filled")

    print()


if __name__ == "__main__":
    main()
