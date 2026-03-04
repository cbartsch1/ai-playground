#!/usr/bin/env python3
"""MS ON-Only Deep Parameter Sweep — Staged approach.

Overnight High/Low is the strongest single MS edge (PF 1.316, +$14,615, 297 trades).
This script does an aggressive staged parameter sweep to find optimal configuration.

Stage 1: zone_pts x stop_buffer (42 combos)
Stage 2: best zone/stop → min_target x min_rr (20 combos)
Stage 3: best from S2 → max_risk x ma_confirm_bars (12 combos)  [corrected: 5*3=15 combos]
Stage 4: best from S3 → max_ms_trades (4 combos)
Stage 5: Walk-forward validation on top 5 overall configs
"""
import sys
import os
import time
import copy
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


# ── Defaults for ON-only MS base ──────────────────────────────────────

def ms_on_base():
    """Base config: MS ON-only levels, SMA 8/24, everything else OFF."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"

    # Turn off all other setups
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False

    # MS ON
    cfg.use_ms = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_vp_levels = True   # needed for VP-derived targets

    # All other level flags OFF
    cfg.ms_use_prev_va = False
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False

    # SMA 8/24 timing
    cfg.ms_ma_type = "sma"

    # Defaults (will be overridden per stage)
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 5.0
    cfg.ms_min_target_pts = 4.0
    cfg.ms_min_rr = 0.5
    cfg.ms_max_risk = 15.0
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8

    return cfg


# ── Evaluation helper ─────────────────────────────────────────────────

def evaluate(trades, label=""):
    """Return dict of metrics for a set of trades."""
    if not trades or len(trades) < 5:
        return None

    m = compute_metrics(trades)
    pnls = [t.pnl_dollar for t in trades]
    _, p_val = stats.ttest_1samp(pnls, 0) if len(trades) >= 5 else (0, 1.0)

    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    l_pnl = sum(t.pnl_dollar for t in longs)
    s_pnl = sum(t.pnl_dollar for t in shorts)

    return {
        "label": label,
        "trades": m.total_trades,
        "win_rate": m.win_rate,
        "pf": m.profit_factor,
        "pnl": m.net_pnl,
        "max_dd": m.max_drawdown,
        "sharpe": m.sharpe,
        "p_val": p_val,
        "trades_per_day": m.trades_per_day,
        "long_pnl": l_pnl,
        "short_pnl": s_pnl,
    }


def print_table_header():
    print(f"  {'Label':<45s}  {'Tr':>5s}  {'WR%':>5s}  {'PF':>6s}  {'P&L':>10s}  "
          f"{'MaxDD':>8s}  {'Sharpe':>6s}  {'p-val':>6s}  {'T/D':>4s}  "
          f"{'Long$':>9s}  {'Short$':>9s}")
    print(f"  {'-'*45}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*10}  "
          f"{'-'*8}  {'-'*6}  {'-'*6}  {'-'*4}  "
          f"{'-'*9}  {'-'*9}")


def print_result(r):
    if r is None:
        return
    sig = "***" if r["p_val"] < 0.01 else "**" if r["p_val"] < 0.05 else "*" if r["p_val"] < 0.10 else ""
    print(f"  {r['label']:<45s}  {r['trades']:>5d}  {r['win_rate']:>5.1f}  {r['pf']:>6.3f}  "
          f"${r['pnl']:>+9,.0f}  ${r['max_dd']:>7,.0f}  {r['sharpe']:>6.2f}  "
          f"{r['p_val']:>.3f}{sig:<3s}  {r['trades_per_day']:>4.1f}  "
          f"${r['long_pnl']:>+8,.0f}  ${r['short_pnl']:>+8,.0f}")


def print_top_n(results, n=10, stage_name=""):
    """Print top N by PF, filtered: >= 100 trades, PF > 1.0."""
    filtered = [r for r in results if r is not None and r["trades"] >= 100 and r["pf"] > 1.0]
    filtered.sort(key=lambda x: x["pf"], reverse=True)
    top = filtered[:n]

    if not top:
        print(f"\n  No configs passed filter (>= 100 trades, PF > 1.0)")
        return top

    print(f"\n  TOP {min(n, len(top))} by PF {stage_name}(>= 100 trades, PF > 1.0):")
    print_table_header()
    for r in top:
        print_result(r)
    return top


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    # Load data
    df = load_tos_csv('data/es_5m_databento_2yr.csv', instrument='ES')
    print(f"Loaded {len(df):,} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Walk-forward split
    split_date = "2025-02-14"
    split_idx = df.index.get_indexer([split_date], method="nearest")[0]
    df_is = df.iloc[:split_idx].copy()   # In-sample (Year 2: Feb 2024 - Feb 2025)
    df_oos = df.iloc[split_idx:].copy()  # Out-of-sample (Year 1: Feb 2025 - Feb 2026)
    print(f"In-sample:  {df_is.index[0].date()} to {df_is.index[-1].date()} ({len(df_is):,} bars)")
    print(f"Out-of-sample: {df_oos.index[0].date()} to {df_oos.index[-1].date()} ({len(df_oos):,} bars)")

    # ── Baseline ──────────────────────────────────────────────────────
    print(f"\n{'='*160}")
    print(f"  BASELINE — ON-only with defaults (zone=3, stop=5, minTgt=4, rr=0.5, maxRisk=15, lag=0, maxTrades=8)")
    print(f"{'='*160}")

    base_cfg = ms_on_base()
    base_trades = run_backtest(df.copy(), base_cfg)
    base_result = evaluate(base_trades, "BASELINE (defaults)")
    print_table_header()
    print_result(base_result)

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: zone_pts x stop_buffer (6 x 7 = 42 combos)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*160}")
    print(f"  STAGE 1 — zone_pts x stop_buffer (42 combos)")
    print(f"{'='*160}")

    zone_vals = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    stop_vals = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]

    s1_results = []
    print_table_header()

    for zone, stop in product(zone_vals, stop_vals):
        cfg = ms_on_base()
        cfg.ms_zone_pts = zone
        cfg.ms_stop_buffer = stop
        label = f"zone={zone:.1f} stop={stop:.1f}"
        trades = run_backtest(df.copy(), cfg)
        r = evaluate(trades, label)
        s1_results.append(r)
        if r:
            print_result(r)
        else:
            print(f"  {label:<45s}  < 5 trades")

    s1_top = print_top_n(s1_results, n=10, stage_name="[Stage 1] ")

    # Pick best zone/stop from Stage 1
    if s1_top:
        best_s1 = s1_top[0]
        # Parse label to extract zone and stop values
        parts = best_s1["label"].split()
        best_zone = float(parts[0].split("=")[1])
        best_stop = float(parts[1].split("=")[1])
    else:
        # Fallback to defaults if nothing passed filters
        best_zone = 3.0
        best_stop = 5.0

    print(f"\n  --> Stage 1 winner: zone={best_zone}, stop={best_stop}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: min_target_pts x min_rr (5 x 4 = 20 combos)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*160}")
    print(f"  STAGE 2 — min_target x min_rr (20 combos) [zone={best_zone}, stop={best_stop}]")
    print(f"{'='*160}")

    target_vals = [3.0, 4.0, 5.0, 6.0, 8.0]
    rr_vals = [0.3, 0.5, 0.8, 1.0]

    s2_results = []
    print_table_header()

    for tgt, rr in product(target_vals, rr_vals):
        cfg = ms_on_base()
        cfg.ms_zone_pts = best_zone
        cfg.ms_stop_buffer = best_stop
        cfg.ms_min_target_pts = tgt
        cfg.ms_min_rr = rr
        label = f"tgt={tgt:.1f} rr={rr:.1f}"
        trades = run_backtest(df.copy(), cfg)
        r = evaluate(trades, label)
        s2_results.append(r)
        if r:
            print_result(r)
        else:
            print(f"  {label:<45s}  < 5 trades")

    s2_top = print_top_n(s2_results, n=10, stage_name="[Stage 2] ")

    if s2_top:
        best_s2 = s2_top[0]
        parts = best_s2["label"].split()
        best_tgt = float(parts[0].split("=")[1])
        best_rr = float(parts[1].split("=")[1])
    else:
        best_tgt = 4.0
        best_rr = 0.5

    print(f"\n  --> Stage 2 winner: tgt={best_tgt}, rr={best_rr}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: max_risk x ma_confirm_bars (5 x 3 = 15 combos)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*160}")
    print(f"  STAGE 3 — max_risk x ma_confirm_bars (15 combos) [zone={best_zone}, stop={best_stop}, tgt={best_tgt}, rr={best_rr}]")
    print(f"{'='*160}")

    risk_vals = [10.0, 15.0, 20.0, 25.0]
    lag_vals = [0, 1, 2]

    # Actually 4*3 = 12, adding max_risk=30 for more coverage as described grid has 5
    # User spec says max_risk [10, 15, 20, 25] = 4 values, lag [0,1,2] = 3 => 12 combos
    # But the description says 12, so let's stay with 4*3=12

    s3_results = []
    print_table_header()

    for risk, lag in product(risk_vals, lag_vals):
        cfg = ms_on_base()
        cfg.ms_zone_pts = best_zone
        cfg.ms_stop_buffer = best_stop
        cfg.ms_min_target_pts = best_tgt
        cfg.ms_min_rr = best_rr
        cfg.ms_max_risk = risk
        cfg.ms_ma_confirm_bars = lag
        label = f"risk={risk:.0f} lag={lag}"
        trades = run_backtest(df.copy(), cfg)
        r = evaluate(trades, label)
        s3_results.append(r)
        if r:
            print_result(r)
        else:
            print(f"  {label:<45s}  < 5 trades")

    s3_top = print_top_n(s3_results, n=10, stage_name="[Stage 3] ")

    if s3_top:
        best_s3 = s3_top[0]
        parts = best_s3["label"].split()
        best_risk = float(parts[0].split("=")[1])
        best_lag = int(parts[1].split("=")[1])
    else:
        best_risk = 15.0
        best_lag = 0

    print(f"\n  --> Stage 3 winner: risk={best_risk}, lag={best_lag}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 4: max_ms_trades (4 combos)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*160}")
    print(f"  STAGE 4 — max_ms_trades (4 combos) [zone={best_zone}, stop={best_stop}, tgt={best_tgt}, rr={best_rr}, risk={best_risk}, lag={best_lag}]")
    print(f"{'='*160}")

    max_trades_vals = [2, 4, 6, 8]

    s4_results = []
    print_table_header()

    for mt in max_trades_vals:
        cfg = ms_on_base()
        cfg.ms_zone_pts = best_zone
        cfg.ms_stop_buffer = best_stop
        cfg.ms_min_target_pts = best_tgt
        cfg.ms_min_rr = best_rr
        cfg.ms_max_risk = best_risk
        cfg.ms_ma_confirm_bars = best_lag
        cfg.max_ms_trades = mt
        label = f"maxTrades={mt}"
        trades = run_backtest(df.copy(), cfg)
        r = evaluate(trades, label)
        s4_results.append(r)
        if r:
            print_result(r)
        else:
            print(f"  {label:<45s}  < 5 trades")

    s4_top = print_top_n(s4_results, n=10, stage_name="[Stage 4] ")

    if s4_top:
        best_s4 = s4_top[0]
        best_max_trades = int(best_s4["label"].split("=")[1])
    else:
        best_max_trades = 8

    print(f"\n  --> Stage 4 winner: maxTrades={best_max_trades}")

    # ══════════════════════════════════════════════════════════════════
    # OPTIMAL CONFIG SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*160}")
    print(f"  OPTIMAL CONFIG FROM STAGED SWEEP")
    print(f"{'='*160}")
    print(f"  zone_pts       = {best_zone}")
    print(f"  stop_buffer    = {best_stop}")
    print(f"  min_target_pts = {best_tgt}")
    print(f"  min_rr         = {best_rr}")
    print(f"  max_risk       = {best_risk}")
    print(f"  ma_confirm_bars= {best_lag}")
    print(f"  max_ms_trades  = {best_max_trades}")

    # Run final optimal on full data
    opt_cfg = ms_on_base()
    opt_cfg.ms_zone_pts = best_zone
    opt_cfg.ms_stop_buffer = best_stop
    opt_cfg.ms_min_target_pts = best_tgt
    opt_cfg.ms_min_rr = best_rr
    opt_cfg.ms_max_risk = best_risk
    opt_cfg.ms_ma_confirm_bars = best_lag
    opt_cfg.max_ms_trades = best_max_trades

    opt_trades = run_backtest(df.copy(), opt_cfg)
    opt_result = evaluate(opt_trades, "OPTIMAL (full 2yr)")
    print(f"\n  Full 2-year result:")
    print_table_header()
    print_result(opt_result)

    # ══════════════════════════════════════════════════════════════════
    # STAGE 5: Walk-Forward Validation on Top 5 Overall
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*160}")
    print(f"  STAGE 5 — WALK-FORWARD VALIDATION")
    print(f"{'='*160}")
    print(f"  In-sample:      {df_is.index[0].date()} to {df_is.index[-1].date()}")
    print(f"  Out-of-sample:  {df_oos.index[0].date()} to {df_oos.index[-1].date()}")

    # Collect top 5 distinct configs from all stages
    # We'll build config objects for the top candidates
    all_results = []

    # Gather the best configs from each stage for walk-forward
    # We need to reconstruct configs — collect (label, cfg) tuples
    wf_configs = []

    # 1) The overall optimal
    wf_configs.append(("OPTIMAL", copy.deepcopy(opt_cfg)))

    # 2) Top configs from S1 (with S2-S4 defaults)
    if len(s1_top) >= 2:
        for i, r in enumerate(s1_top[:2]):
            parts = r["label"].split()
            z = float(parts[0].split("=")[1])
            s = float(parts[1].split("=")[1])
            cfg = ms_on_base()
            cfg.ms_zone_pts = z
            cfg.ms_stop_buffer = s
            wf_configs.append((f"S1-rank{i+1} z={z} s={s}", cfg))

    # 3) Top configs from S2 (with S1 best + S3/S4 defaults)
    if len(s2_top) >= 2:
        for i, r in enumerate(s2_top[:2]):
            parts = r["label"].split()
            tgt = float(parts[0].split("=")[1])
            rr = float(parts[1].split("=")[1])
            cfg = ms_on_base()
            cfg.ms_zone_pts = best_zone
            cfg.ms_stop_buffer = best_stop
            cfg.ms_min_target_pts = tgt
            cfg.ms_min_rr = rr
            wf_configs.append((f"S2-rank{i+1} tgt={tgt} rr={rr}", cfg))

    # 4) Baseline for comparison
    wf_configs.append(("BASELINE (defaults)", ms_on_base()))

    # Deduplicate (keep first 5 unique + baseline)
    seen = set()
    unique_wf = []
    for label, cfg in wf_configs:
        key = (cfg.ms_zone_pts, cfg.ms_stop_buffer, cfg.ms_min_target_pts,
               cfg.ms_min_rr, cfg.ms_max_risk, cfg.ms_ma_confirm_bars, cfg.max_ms_trades)
        if key not in seen:
            seen.add(key)
            unique_wf.append((label, cfg))
    # Limit to 6 (top 5 + baseline)
    unique_wf = unique_wf[:6]

    print(f"\n  Validating {len(unique_wf)} configs:\n")
    print(f"  {'Label':<45s}  {'Period':<4s}  {'Tr':>5s}  {'WR%':>5s}  {'PF':>6s}  "
          f"{'P&L':>10s}  {'MaxDD':>8s}  {'Sharpe':>6s}  {'p-val':>6s}  {'Verdict':>8s}")
    print(f"  {'-'*45}  {'-'*4}  {'-'*5}  {'-'*5}  {'-'*6}  "
          f"{'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}")

    for label, cfg in unique_wf:
        t_is = run_backtest(df_is.copy(), cfg)
        t_oos = run_backtest(df_oos.copy(), cfg)

        r_is = evaluate(t_is, label)
        r_oos = evaluate(t_oos, label)

        if r_is and r_oos and r_is["pf"] > 0:
            ratio = r_oos["pf"] / r_is["pf"]
            if ratio > 0.7:
                verdict = "PASS"
            elif ratio > 0.5:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"

            print(f"  {label:<45s}  {'IS':<4s}  {r_is['trades']:>5d}  {r_is['win_rate']:>5.1f}  "
                  f"{r_is['pf']:>6.3f}  ${r_is['pnl']:>+9,.0f}  ${r_is['max_dd']:>7,.0f}  "
                  f"{r_is['sharpe']:>6.2f}  {r_is['p_val']:>.3f}   ")
            print(f"  {'':<45s}  {'OOS':<4s}  {r_oos['trades']:>5d}  {r_oos['win_rate']:>5.1f}  "
                  f"{r_oos['pf']:>6.3f}  ${r_oos['pnl']:>+9,.0f}  ${r_oos['max_dd']:>7,.0f}  "
                  f"{r_oos['sharpe']:>6.2f}  {r_oos['p_val']:>.3f}   {verdict:<8s}  ratio={ratio:.2f}")
            print()
        else:
            is_str = f"IS: {r_is['trades']}t PF={r_is['pf']:.3f}" if r_is else "IS: no trades"
            oos_str = f"OOS: {r_oos['trades']}t PF={r_oos['pf']:.3f}" if r_oos else "OOS: no trades"
            print(f"  {label:<45s}  {is_str}  {oos_str}  INSUFFICIENT DATA")
            print()

    elapsed = time.time() - t0
    print(f"\n{'='*160}")
    print(f"  SWEEP COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*160}")
