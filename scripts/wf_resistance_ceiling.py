#!/usr/bin/env python3
"""Walk-forward validation: Resistance Ceiling filter on IB Breakout shorts.

The "resistance ceiling" filter:
  - On each IB Breakout short signal, check for resistance levels above entry
  - If the nearest resistance level is > 20 pts above entry price, SKIP the trade
  - Rationale: nearby resistance acts as a ceiling that limits upside risk for shorts

Validation:
  1. 5-fold sequential walk-forward (each fold is OOS, rest is IS)
  2. 50/50 year split (first half = IS, second half = OOS)
  3. Compare: v8 baseline vs v8 + resistance ceiling filter

Usage:
    .venv/bin/python scripts/wf_resistance_ceiling.py data/es_5m_databento_2yr.csv
"""

import argparse
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
from backtester.indicators import compute_indicators
from backtester.session import SessionState, update_session
from backtester.position import Position
from backtester.setups import ib_breakout, level_rejection


# ── Default threshold ─────────────────────────────────────────────────────

DEFAULT_CEILING_PTS = 20.0  # Max distance to nearest resistance above entry


def make_v8_config():
    """v8 baseline — short-only, 30bps pct stop, skip Friday, noon blackout."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    return cfg


def make_v8_ceiling_config():
    """v8 + resistance ceiling. Same as v8, but we also enable level state tracking."""
    cfg = make_v8_config()
    # Enable level state tracking only (not level_rejection signal generation)
    cfg.use_level_reject = True
    return cfg


# ── Custom Engine: v8 + Resistance Ceiling Filter ────────────────────────

def run_backtest_with_ceiling(df, cfg, ceiling_pts):
    """Custom engine loop that applies resistance ceiling filter to IB Breakout shorts.

    Identical to the standard engine EXCEPT:
      - After IB Breakout generates a short signal, we check the resistance map
      - If the nearest resistance level above entry is > ceiling_pts away, SKIP
      - If no resistance level exists above entry, also SKIP (no ceiling = unbounded risk)
    """
    from backtester.setups import ib_rejection, va_fade, eighty_rule, tema_cross

    # Compute indicators on full DataFrame first
    compute_indicators(
        df,
        tema_fast=cfg.tema_fast,
        tema_slow=cfg.tema_slow,
        tema_trend=cfg.tema_trend,
        atr_len=cfg.atr_len,
        atr_avg_len=cfg.atr_avg_len,
    )

    state = SessionState()
    pos = Position()
    trades = []
    prev_bar = None
    prev_position_size = 0
    skipped_no_ceiling = 0
    skipped_too_far = 0

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx

        # ── Update session state ──
        update_session(state, bar, prev_bar, cfg)

        # ── Level state tracking (must run every bar) ──
        level_rejection.update_level_state(bar, state, cfg)

        # ── Cooldown tracking ──
        current_position_size = 0 if pos.is_flat else pos.direction
        if current_position_size == 0 and prev_position_size != 0:
            state.bars_since_exit = 0
        elif current_position_size == 0:
            state.bars_since_exit += 1

        # ── Check exits ──
        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            _finalize_trade(trade, cfg)
            trades.append(trade)

        # ── TEMA Exit (v9) ──
        if not pos.is_flat and cfg.use_tema_exit:
            if pos.direction == -1 and bar.get("tema_cross_up", False):
                trade = pos.close_at_market(bar, "tema_exit")
                if trade is not None:
                    _finalize_trade(trade, cfg)
                    trades.append(trade)
            elif pos.direction == 1 and bar.get("tema_cross_down", False):
                trade = pos.close_at_market(bar, "tema_exit")
                if trade is not None:
                    _finalize_trade(trade, cfg)
                    trades.append(trade)

        # ── Session flatten ──
        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                _finalize_trade(trade, cfg)
                trades.append(trade)

        # ── Check entry signals (only if flat) ──
        if pos.is_flat:
            signal = None

            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                # Priority: IB Breakout > IB Rejection > VA Fade > 80% Rule > TEMA Cross
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)

                # ── RESISTANCE CEILING FILTER ──
                # Only applies to IB Breakout short signals
                if signal is not None and signal["setup"] == "IB" and signal["direction"] == -1:
                    entry_price = bar["close"]
                    resistance_map = level_rejection._get_resistance_map(state)

                    # Find resistance levels ABOVE entry price
                    levels_above = [(name, price) for name, price in resistance_map
                                    if price > entry_price]

                    if not levels_above:
                        # No resistance above entry — no ceiling, skip the trade
                        skipped_no_ceiling += 1
                        signal = None
                    else:
                        # Find nearest resistance above entry
                        nearest_name, nearest_price = min(levels_above, key=lambda x: x[1])
                        distance = nearest_price - entry_price

                        if distance > ceiling_pts:
                            # Resistance too far above — no effective ceiling
                            skipped_too_far += 1
                            signal = None

                # If IB Breakout was filtered out, try remaining setups
                if signal is None:
                    signal = ib_rejection.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = level_rejection.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = va_fade.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = eighty_rule.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = tema_cross.check_signal(bar, prev_bar, state, cfg)

            # Direction filter
            if signal is not None and cfg.direction_filter != "both":
                if cfg.direction_filter == "short" and signal["direction"] == 1:
                    signal = None
                elif cfg.direction_filter == "long" and signal["direction"] == -1:
                    signal = None

            if signal is not None:
                trail_trigger = 0.0
                trail_dist = 0.0
                if cfg.use_trail_stop:
                    trail_trigger = bar["close"] * cfg.trail_trigger_bps / 10000.0
                    trail_dist = bar["close"] * cfg.trail_dist_bps / 10000.0

                pos.enter(
                    direction=signal["direction"],
                    price=bar["close"],
                    stop=signal["stop"],
                    target=signal["target"],
                    setup=signal["setup"],
                    time=idx,
                    slippage=cfg.slippage_pts,
                    trail_trigger_pts=trail_trigger,
                    trail_dist_pts=trail_dist,
                )

        prev_position_size = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    # Close any remaining position
    if not pos.is_flat and prev_bar is not None:
        trade = pos.flatten(prev_bar)
        if trade is not None:
            _finalize_trade(trade, cfg)
            trades.append(trade)

    return trades, skipped_no_ceiling, skipped_too_far


def _finalize_trade(trade, cfg):
    """Compute dollar P&L including commission and slippage."""
    trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)


# ── Helpers ───────────────────────────────────────────────────────────────

def run_baseline(df_slice):
    """Run v8 baseline on a slice, return trades list."""
    cfg = make_v8_config()
    trades = run_backtest(df_slice.copy(), cfg)
    return trades


def run_ceiling(df_slice, ceiling_pts):
    """Run v8 + resistance ceiling on a slice, return trades + skip counts."""
    cfg = make_v8_ceiling_config()
    trades, skip_none, skip_far = run_backtest_with_ceiling(df_slice.copy(), cfg, ceiling_pts)
    return trades, skip_none, skip_far


def fmt_metrics(m):
    """Format metrics as a compact string."""
    if m is None or m.total_trades == 0:
        return "0 trades"
    return (f"{m.total_trades:3d} trades | WR {m.win_rate:5.1f}% | "
            f"PF {m.profit_factor:5.3f} | P&L ${m.net_pnl:+10,.0f} | "
            f"DD ${m.max_drawdown:8,.0f} | Sharpe {m.sharpe:5.2f}")


def print_comparison_row(label, m_base, m_ceil):
    """Print a side-by-side comparison row."""
    print(f"\n  {label}:")
    print(f"    Baseline : {fmt_metrics(m_base)}")
    print(f"    +Ceiling : {fmt_metrics(m_ceil)}")


# ── Walk-Forward: 5-Fold Sequential ──────────────────────────────────────

def run_5fold_wf(df, ceiling_pts):
    """5-fold sequential walk-forward validation.

    Split data into 5 equal-size sequential folds.
    For each fold i: train on folds != i, test on fold i.
    Report OOS stats per fold + combined.
    """
    print("\n" + "=" * 80)
    print("5-FOLD SEQUENTIAL WALK-FORWARD VALIDATION")
    print("=" * 80)

    n = len(df)
    fold_size = n // 5
    fold_boundaries = []
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 4 else n
        fold_boundaries.append((start, end))

    # Print fold info
    for i, (s, e) in enumerate(fold_boundaries):
        fold_df = df.iloc[s:e]
        print(f"  Fold {i+1}: {fold_df.index[0].strftime('%Y-%m-%d')} to "
              f"{fold_df.index[-1].strftime('%Y-%m-%d')} ({e-s:,} bars)")

    # Per-fold OOS results
    oos_base_trades_all = []
    oos_ceil_trades_all = []
    fold_results = []

    for fold_idx in range(5):
        s, e = fold_boundaries[fold_idx]
        df_oos = df.iloc[s:e].copy()

        fold_label = (f"Fold {fold_idx+1} OOS "
                      f"({df_oos.index[0].strftime('%Y-%m-%d')} to "
                      f"{df_oos.index[-1].strftime('%Y-%m-%d')})")

        # Run baseline
        trades_base = run_baseline(df_oos)
        m_base = compute_metrics(trades_base) if trades_base else None

        # Run ceiling
        trades_ceil, skip_none, skip_far = run_ceiling(df_oos, ceiling_pts)
        m_ceil = compute_metrics(trades_ceil) if trades_ceil else None

        oos_base_trades_all.extend(trades_base)
        oos_ceil_trades_all.extend(trades_ceil)

        fold_results.append({
            "fold": fold_idx + 1,
            "m_base": m_base,
            "m_ceil": m_ceil,
            "skip_none": skip_none,
            "skip_far": skip_far,
        })

        print_comparison_row(fold_label, m_base, m_ceil)
        if skip_none + skip_far > 0:
            print(f"    Skipped: {skip_none} (no resistance above) + "
                  f"{skip_far} (resistance > {ceiling_pts}pt away) = "
                  f"{skip_none + skip_far} total")

    # Combined OOS (all 5 folds together)
    m_base_combined = compute_metrics(oos_base_trades_all) if oos_base_trades_all else None
    m_ceil_combined = compute_metrics(oos_ceil_trades_all) if oos_ceil_trades_all else None

    print("\n" + "-" * 80)
    print_comparison_row("Combined OOS (all 5 folds)", m_base_combined, m_ceil_combined)

    return fold_results, m_base_combined, m_ceil_combined


# ── Walk-Forward: 50/50 Year Split ───────────────────────────────────────

def run_year_split(df, ceiling_pts):
    """50/50 year split: first half = IS (Year 2), second half = OOS (Year 1).

    For 2yr data (Feb 2024 - Feb 2026):
      - IS:  Feb 2024 - Feb 2025 (Year 2 / older data)
      - OOS: Feb 2025 - Feb 2026 (Year 1 / newer data)
    """
    print("\n" + "=" * 80)
    print("50/50 YEAR SPLIT (first half = IS, second half = OOS)")
    print("=" * 80)

    mid = len(df) // 2
    df_is = df.iloc[:mid].copy()
    df_oos = df.iloc[mid:].copy()

    print(f"  IS:  {df_is.index[0].strftime('%Y-%m-%d')} to "
          f"{df_is.index[-1].strftime('%Y-%m-%d')} ({len(df_is):,} bars)")
    print(f"  OOS: {df_oos.index[0].strftime('%Y-%m-%d')} to "
          f"{df_oos.index[-1].strftime('%Y-%m-%d')} ({len(df_oos):,} bars)")

    # --- In-Sample ---
    print("\n  === IN-SAMPLE ===")
    trades_base_is = run_baseline(df_is)
    m_base_is = compute_metrics(trades_base_is) if trades_base_is else None

    trades_ceil_is, skip_none_is, skip_far_is = run_ceiling(df_is, ceiling_pts)
    m_ceil_is = compute_metrics(trades_ceil_is) if trades_ceil_is else None

    print_comparison_row("In-Sample", m_base_is, m_ceil_is)
    print(f"    Skipped (IS): {skip_none_is} (no resistance) + "
          f"{skip_far_is} (too far) = {skip_none_is + skip_far_is}")

    # --- Out-of-Sample ---
    print("\n  === OUT-OF-SAMPLE ===")
    trades_base_oos = run_baseline(df_oos)
    m_base_oos = compute_metrics(trades_base_oos) if trades_base_oos else None

    trades_ceil_oos, skip_none_oos, skip_far_oos = run_ceiling(df_oos, ceiling_pts)
    m_ceil_oos = compute_metrics(trades_ceil_oos) if trades_ceil_oos else None

    print_comparison_row("Out-of-Sample", m_base_oos, m_ceil_oos)
    print(f"    Skipped (OOS): {skip_none_oos} (no resistance) + "
          f"{skip_far_oos} (too far) = {skip_none_oos + skip_far_oos}")

    # --- Full Period ---
    print("\n  === FULL PERIOD ===")
    trades_base_full = run_baseline(df)
    m_base_full = compute_metrics(trades_base_full) if trades_base_full else None

    trades_ceil_full, skip_none_full, skip_far_full = run_ceiling(df, ceiling_pts)
    m_ceil_full = compute_metrics(trades_ceil_full) if trades_ceil_full else None

    print_comparison_row("Full Period", m_base_full, m_ceil_full)
    print(f"    Skipped (Full): {skip_none_full} (no resistance) + "
          f"{skip_far_full} (too far) = {skip_none_full + skip_far_full}")

    # PF ratio
    if (m_ceil_is and m_ceil_oos and
            m_ceil_is.profit_factor > 0 and m_ceil_oos.profit_factor > 0):
        pf_ratio_ceil = m_ceil_oos.profit_factor / m_ceil_is.profit_factor
    else:
        pf_ratio_ceil = 0
    if (m_base_is and m_base_oos and
            m_base_is.profit_factor > 0 and m_base_oos.profit_factor > 0):
        pf_ratio_base = m_base_oos.profit_factor / m_base_is.profit_factor
    else:
        pf_ratio_base = 0

    print(f"\n  PF ratio (OOS/IS): Baseline={pf_ratio_base:.2f} | "
          f"+Ceiling={pf_ratio_ceil:.2f}  (>0.7 = robust)")

    return {
        "m_base_is": m_base_is, "m_ceil_is": m_ceil_is,
        "m_base_oos": m_base_oos, "m_ceil_oos": m_ceil_oos,
        "m_base_full": m_base_full, "m_ceil_full": m_ceil_full,
        "pf_ratio_base": pf_ratio_base, "pf_ratio_ceil": pf_ratio_ceil,
    }


# ── Summary Table ─────────────────────────────────────────────────────────

def print_summary_table(fold_results, m_base_5f, m_ceil_5f, year_split, ceiling_pts):
    """Print a clean summary table of all results."""

    print("\n" + "=" * 80)
    print("FINAL SUMMARY — Resistance Ceiling Filter Walk-Forward Validation")
    print(f"Filter: Skip IB Breakout shorts if nearest resistance > {ceiling_pts}pt above entry")
    print("=" * 80)

    # 5-fold table
    print(f"\n  5-FOLD SEQUENTIAL WALK-FORWARD (each fold = OOS)")
    print(f"  {'Fold':12s} | {'#B':>6s} | {'WR_B':>6s} | {'PF_B':>6s} | {'P&L_B':>10s} | "
          f"{'#C':>6s} | {'WR_C':>6s} | {'PF_C':>6s} | {'P&L_C':>10s} | {'Skip':>5s}")
    print("  " + "-" * 105)

    for r in fold_results:
        mb = r["m_base"]
        mc = r["m_ceil"]
        nb = mb.total_trades if mb else 0
        nc = mc.total_trades if mc else 0
        wrb = f"{mb.win_rate:.1f}%" if mb and nb > 0 else "  n/a"
        wrc = f"{mc.win_rate:.1f}%" if mc and nc > 0 else "  n/a"
        pfb = f"{mb.profit_factor:.3f}" if mb and nb > 0 else "  n/a"
        pfc = f"{mc.profit_factor:.3f}" if mc and nc > 0 else "  n/a"
        plb = f"${mb.net_pnl:+,.0f}" if mb and nb > 0 else "     n/a"
        plc = f"${mc.net_pnl:+,.0f}" if mc and nc > 0 else "     n/a"
        skip = r["skip_none"] + r["skip_far"]
        print(f"  Fold {r['fold']:5d}   | {nb:6d} | {wrb:>6s} | {pfb:>6s} | {plb:>10s} | "
              f"{nc:6d} | {wrc:>6s} | {pfc:>6s} | {plc:>10s} | {skip:5d}")

    # Combined 5-fold
    if m_base_5f and m_ceil_5f and m_base_5f.total_trades > 0 and m_ceil_5f.total_trades > 0:
        nb = m_base_5f.total_trades
        nc = m_ceil_5f.total_trades
        print("  " + "-" * 105)
        print(f"  {'Combined':12s} | {nb:6d} | {m_base_5f.win_rate:5.1f}% | "
              f"{m_base_5f.profit_factor:5.3f} | ${m_base_5f.net_pnl:+10,.0f} | "
              f"{nc:6d} | {m_ceil_5f.win_rate:5.1f}% | "
              f"{m_ceil_5f.profit_factor:5.3f} | ${m_ceil_5f.net_pnl:+10,.0f} |")
    else:
        print("  Combined: insufficient trades")

    # Column legend
    print(f"\n  B = Baseline (v8), C = +Ceiling filter, Skip = trades filtered out by ceiling")

    # Year split table
    ys = year_split
    print(f"\n  50/50 YEAR SPLIT")
    print(f"  {'Period':14s} | {'#B':>6s} | {'WR_B':>6s} | {'PF_B':>6s} | {'P&L_B':>10s} | "
          f"{'#C':>6s} | {'WR_C':>6s} | {'PF_C':>6s} | {'P&L_C':>10s}")
    print("  " + "-" * 95)

    for label, mb_key, mc_key in [("In-Sample", "m_base_is", "m_ceil_is"),
                                    ("Out-of-Sample", "m_base_oos", "m_ceil_oos"),
                                    ("Full Period", "m_base_full", "m_ceil_full")]:
        mb = ys[mb_key]
        mc = ys[mc_key]
        if mb and mc and mb.total_trades > 0 and mc.total_trades > 0:
            print(f"  {label:14s} | {mb.total_trades:6d} | {mb.win_rate:5.1f}% | "
                  f"{mb.profit_factor:5.3f} | ${mb.net_pnl:+10,.0f} | "
                  f"{mc.total_trades:6d} | {mc.win_rate:5.1f}% | "
                  f"{mc.profit_factor:5.3f} | ${mc.net_pnl:+10,.0f}")
        else:
            print(f"  {label:14s} | no trades")

    print(f"\n  PF ratio (OOS/IS): Baseline={ys['pf_ratio_base']:.2f} | "
          f"+Ceiling={ys['pf_ratio_ceil']:.2f}  (>0.7 = robust, <0.5 = overfit)")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Check if ceiling filter improved OOS performance
    if m_ceil_5f and m_base_5f and m_ceil_5f.total_trades > 0 and m_base_5f.total_trades > 0:
        pf_delta_5f = m_ceil_5f.profit_factor - m_base_5f.profit_factor
        pnl_delta_5f = m_ceil_5f.net_pnl - m_base_5f.net_pnl

        print(f"\n  5-Fold OOS: Ceiling PF {m_ceil_5f.profit_factor:.3f} vs "
              f"Baseline PF {m_base_5f.profit_factor:.3f} (delta: {pf_delta_5f:+.3f})")
        print(f"  5-Fold OOS: Ceiling P&L ${m_ceil_5f.net_pnl:+,.0f} vs "
              f"Baseline P&L ${m_base_5f.net_pnl:+,.0f} (delta: ${pnl_delta_5f:+,.0f})")

    oos_m_base = ys.get("m_base_oos")
    oos_m_ceil = ys.get("m_ceil_oos")
    if oos_m_base and oos_m_ceil and oos_m_base.total_trades > 0 and oos_m_ceil.total_trades > 0:
        pf_delta_ys = oos_m_ceil.profit_factor - oos_m_base.profit_factor
        pnl_delta_ys = oos_m_ceil.net_pnl - oos_m_base.net_pnl

        print(f"\n  Year Split OOS: Ceiling PF {oos_m_ceil.profit_factor:.3f} vs "
              f"Baseline PF {oos_m_base.profit_factor:.3f} (delta: {pf_delta_ys:+.3f})")
        print(f"  Year Split OOS: Ceiling P&L ${oos_m_ceil.net_pnl:+,.0f} vs "
              f"Baseline P&L ${oos_m_base.net_pnl:+,.0f} (delta: ${pnl_delta_ys:+,.0f})")

    # Simple pass/fail
    improved_5f = (m_ceil_5f and m_base_5f and
                   m_ceil_5f.profit_factor > m_base_5f.profit_factor)
    improved_ys = (oos_m_ceil and oos_m_base and
                   oos_m_ceil.profit_factor > oos_m_base.profit_factor)

    if improved_5f and improved_ys:
        print("\n  RESULT: PASS — Resistance ceiling improves OOS PF in BOTH validations")
    elif improved_5f or improved_ys:
        print("\n  RESULT: MIXED — Ceiling improves OOS PF in one validation but not the other")
    else:
        print("\n  RESULT: FAIL — Resistance ceiling does NOT improve OOS performance")

    print()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward validation: Resistance Ceiling filter on IB Breakout shorts")
    parser.add_argument("csv_file", help="Path to 2yr Databento ES 5m CSV")
    parser.add_argument("--ceiling", type=float, default=DEFAULT_CEILING_PTS,
                        help=f"Max distance (pts) to nearest resistance (default: {DEFAULT_CEILING_PTS})")
    args = parser.parse_args()

    ceiling_pts = args.ceiling

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file)
    print(f"Loaded {len(df):,} bars: {df.index[0].strftime('%Y-%m-%d')} to "
          f"{df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Resistance ceiling threshold: {ceiling_pts} pts")

    # Run 5-fold walk-forward
    fold_results, m_base_5f, m_ceil_5f = run_5fold_wf(df, ceiling_pts)

    # Run 50/50 year split
    year_split = run_year_split(df, ceiling_pts)

    # Print summary
    print_summary_table(fold_results, m_base_5f, m_ceil_5f, year_split, ceiling_pts)


if __name__ == "__main__":
    main()
