#!/usr/bin/env python3
"""Diagnostic: IS/OOS Regime Split for LVL v13 — Data Leakage Check.

Hypothesis: The walk-forward PF ratio dropped from 0.88 (unfiltered) to 0.49
(regime-filtered) because the regime blocking decision used the FULL 2yr dataset
rather than being trained on IS only.  Accumulation (PF 0.97) and Recovery
(PF 0.82) look slightly negative over 2yr — but they might be negative in one
half and positive in the other, making the 2yr aggregate misleading.

This script:
  1. Runs v13 backtest, tags all LVL fills with HMM regime.
  2. Splits at 2025-02-14 into IS (Year 1) and OOS (Year 2).
  3. Prints per-regime breakdown for IS and OOS separately.
  4. Computes regime-filtered results under three approaches:
       a) "Full Peek"  — block regimes negative over full 2yr (current method)
       b) "IS-Only"    — block regimes negative in IS only (proper walk-forward)
       c) "OOS-Only"   — block regimes negative in OOS only (oracle comparison)
  5. Prints a comparison table of WF PF ratios for each approach.

Usage:
    cd ~/projects/ai-playground/medallion-2.0
    source .venv/bin/activate
    python scripts/regime_profile_lvl_diagnostic.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Project roots (same as regime_profile_lvl.py)
# ---------------------------------------------------------------------------
MEDALLION_ROOT = Path(__file__).parent.parent
AI_PLAYGROUND_ROOT = MEDALLION_ROOT.parent

SPLIT_DATE = "2025-02-14"      # IS / OOS boundary


# ---------------------------------------------------------------------------
# Path management (identical to regime_profile_lvl.py)
# ---------------------------------------------------------------------------
def _swap_path(project_root):
    cleaned = [p for p in sys.path if p not in (str(AI_PLAYGROUND_ROOT),)]
    sys.path[:] = cleaned
    if str(MEDALLION_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDALLION_ROOT))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    mods_to_remove = [k for k in sys.modules if k.startswith("backtester")]
    for k in mods_to_remove:
        del sys.modules[k]


# ---------------------------------------------------------------------------
# Regime loading (reuse from regime_profile_lvl)
# ---------------------------------------------------------------------------
def load_regime_filter():
    _swap_path(MEDALLION_ROOT)
    from models.hmm_regime import RegimeDetector
    from models.regime_api import RegimeFilter
    from data.data_loader import download_ohlcv, compute_hmm_features

    print("=" * 70)
    print("  LOADING MEDALLION 2.0 REGIME MODEL")
    print("=" * 70)

    detector = RegimeDetector.load_latest(n_regimes=7)
    if detector is None:
        print("ERROR: No saved 7-state model found.")
        sys.exit(1)

    spy_hourly = download_ohlcv("SPY", "1h", "730d")
    hmm_features = compute_hmm_features(spy_hourly)
    full_df = spy_hourly.join(hmm_features)
    rf = RegimeFilter(detector, full_df)

    current = detector.get_current_regime(full_df)
    print(f"  Current regime: {current['label']} ({current['confidence']:.1%})")
    return rf


# ---------------------------------------------------------------------------
# v13 config + backtest (reuse from regime_profile_lvl)
# ---------------------------------------------------------------------------
def make_v13_config():
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.config import StrategyConfig

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
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = True
    cfg.rej_trigger = "any"
    cfg.rej_target = "ib_low"
    cfg.rej_zone_pts = 5.0
    cfg.rej_stop_buffer = 8.0
    cfg.rej_require_tema = False
    cfg.max_rej_trades = 8
    cfg.rej_wide_only = True
    cfg.use_level_reject = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = True
    cfg.lvl_ma_filter = "tema"
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 7.0
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_min_rr = 0.5
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3
    return cfg


def run_v13_backtest():
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.data_loader import load_tos_csv
    from backtester.stagger_engine import run_backtest_stagger

    data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento_2yr.csv"
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        sys.exit(1)

    cfg = make_v13_config()

    print(f"\n{'=' * 70}")
    print(f"  RUNNING v13 BACKTEST")
    print(f"{'=' * 70}")

    df = load_tos_csv(str(data_path), instrument="ES")
    print(f"  Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    trades = run_backtest_stagger(df, cfg, n_contracts=3, uniform_skip=2)
    lvl_trades = [t for t in trades if t.setup.startswith("LVL")]
    print(f"  Total trades: {len(trades)} ({len(lvl_trades)} LVL)")
    return lvl_trades


# ---------------------------------------------------------------------------
# Tag trades with regime
# ---------------------------------------------------------------------------
def tag_trades(regime_filter, trades):
    tagged = []
    for t in trades:
        regime = regime_filter.get_regime_at(t.entry_time)
        label = regime.get("label")
        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = "No Data (before model)"
        tagged.append((t, label))
    return tagged


# ---------------------------------------------------------------------------
# Per-regime breakdown (identical to original)
# ---------------------------------------------------------------------------
def compute_breakdown(tagged_trades):
    by_regime = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0, "pnls": []})
    for t, label in tagged_trades:
        d = by_regime[label]
        d["trades"] += 1
        d["pnl"] += t.pnl_dollar
        d["pnls"].append(t.pnl_dollar)
        if t.pnl_dollar > 0:
            d["wins"] += 1

    breakdown = {}
    for label, d in by_regime.items():
        pnls = d["pnls"]
        gw = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p <= 0))
        pf = gw / gl if gl > 0 else float("inf")
        wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
        avg = d["pnl"] / d["trades"] if d["trades"] > 0 else 0
        breakdown[label] = {
            "trades": d["trades"],
            "pnl": d["pnl"],
            "win_rate": wr,
            "avg_trade": avg,
            "profit_factor": pf,
            "gross_win": gw,
            "gross_loss": gl,
        }
    return breakdown


def print_breakdown(breakdown, title):
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    hdr = (f"  {'Regime':<25s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
           f"{'P&L':>12s} {'Avg':>10s}")
    print(hdr)
    print(f"  {'-' * 74}")

    for regime in sorted(breakdown.keys(), key=lambda r: breakdown[r]["pnl"], reverse=True):
        d = breakdown[regime]
        pf_str = f"{d['profit_factor']:.2f}" if d['profit_factor'] < 100 else "inf"
        print(f"  {regime:<25s} {d['trades']:>7d} {d['win_rate']:>6.1f}% "
              f"{pf_str:>7s} ${d['pnl']:>10,.0f} ${d['avg_trade']:>8,.0f}")

    total_trades = sum(d["trades"] for d in breakdown.values())
    total_pnl = sum(d["pnl"] for d in breakdown.values())
    total_gw = sum(d["gross_win"] for d in breakdown.values())
    total_gl = sum(d["gross_loss"] for d in breakdown.values())
    total_pf = total_gw / total_gl if total_gl > 0 else float("inf")
    total_wr = (sum(d["trades"] * d["win_rate"] / 100 for d in breakdown.values())
                / total_trades * 100 if total_trades > 0 else 0)
    total_avg = total_pnl / total_trades if total_trades > 0 else 0

    print(f"  {'-' * 74}")
    pf_str = f"{total_pf:.2f}" if total_pf < 100 else "inf"
    print(f"  {'TOTAL':<25s} {total_trades:>7d} {total_wr:>6.1f}% "
          f"{pf_str:>7s} ${total_pnl:>10,.0f} ${total_avg:>8,.0f}")


# ---------------------------------------------------------------------------
# Identify blocked regimes from a breakdown (negative P&L)
# ---------------------------------------------------------------------------
def get_blocked(breakdown):
    return sorted([r for r, d in breakdown.items()
                   if d["pnl"] < 0 and r != "No Data (before model)"])


# ---------------------------------------------------------------------------
# Compute WF stats for a set of trades given blocked regimes
# ---------------------------------------------------------------------------
def compute_wf_stats(tagged_trades, blocked_regimes):
    """Given tagged trades and a set of blocked regimes, compute IS/OOS/total
    stats after filtering.  Returns dict with all relevant metrics."""
    blocked_set = set(blocked_regimes)

    # Filter
    kept = [(t, lab) for t, lab in tagged_trades if lab not in blocked_set]
    removed = [(t, lab) for t, lab in tagged_trades if lab in blocked_set]

    # Split IS / OOS
    is_kept = [(t, lab) for t, lab in kept if str(t.entry_time) < SPLIT_DATE]
    oos_kept = [(t, lab) for t, lab in kept if str(t.entry_time) >= SPLIT_DATE]

    def _stats(trades_list):
        if not trades_list:
            return {"n": 0, "pnl": 0, "pf": 0, "wr": 0, "avg": 0, "gw": 0, "gl": 0}
        pnls = [t.pnl_dollar for t, _ in trades_list]
        gw = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p <= 0))
        pf = gw / gl if gl > 0 else float("inf")
        wins = sum(1 for p in pnls if p > 0)
        return {
            "n": len(pnls),
            "pnl": sum(pnls),
            "pf": pf,
            "wr": wins / len(pnls) * 100,
            "avg": sum(pnls) / len(pnls),
            "gw": gw,
            "gl": gl,
        }

    is_s = _stats(is_kept)
    oos_s = _stats(oos_kept)
    full_s = _stats(kept)

    pf_ratio = oos_s["pf"] / is_s["pf"] if is_s["pf"] > 0 and is_s["pf"] < float("inf") else 0

    return {
        "blocked": blocked_regimes,
        "removed_count": len(removed),
        "removed_pnl": sum(t.pnl_dollar for t, _ in removed),
        "is": is_s,
        "oos": oos_s,
        "full": full_s,
        "pf_ratio": pf_ratio,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # 1. Load regime model
    rf = load_regime_filter()

    # 2. Run v13 backtest
    lvl_trades = run_v13_backtest()
    if not lvl_trades:
        print("ERROR: No LVL trades.")
        sys.exit(1)

    # 3. Tag all LVL trades with regime
    tagged = tag_trades(rf, lvl_trades)

    # 4. Split into IS / OOS
    tagged_is = [(t, lab) for t, lab in tagged if str(t.entry_time) < SPLIT_DATE]
    tagged_oos = [(t, lab) for t, lab in tagged if str(t.entry_time) >= SPLIT_DATE]

    print(f"\n  Split: {len(tagged_is)} IS trades, {len(tagged_oos)} OOS trades "
          f"(boundary = {SPLIT_DATE})")

    # 5. Compute per-regime breakdown for FULL, IS, OOS
    bd_full = compute_breakdown(tagged)
    bd_is = compute_breakdown(tagged_is)
    bd_oos = compute_breakdown(tagged_oos)

    print_breakdown(bd_full, "FULL 2yr — Per-Regime Breakdown (LVL)")
    print_breakdown(bd_is, "IN-SAMPLE (Year 1) — Per-Regime Breakdown (LVL)")
    print_breakdown(bd_oos, "OUT-OF-SAMPLE (Year 2) — Per-Regime Breakdown (LVL)")

    # 6. Show regime stability side-by-side
    all_regimes = sorted(set(list(bd_full.keys()) + list(bd_is.keys()) + list(bd_oos.keys())))
    print(f"\n{'=' * 90}")
    print(f"  REGIME STABILITY: IS vs OOS P&L comparison")
    print(f"{'=' * 90}")
    print(f"  {'Regime':<25s} {'IS Trades':>9s} {'IS PF':>7s} {'IS P&L':>10s}  |  "
          f"{'OOS Trades':>10s} {'OOS PF':>7s} {'OOS P&L':>10s}  {'Stable?':>8s}")
    print(f"  {'-' * 86}")
    for r in all_regimes:
        d_is = bd_is.get(r, {"trades": 0, "profit_factor": 0, "pnl": 0})
        d_oos = bd_oos.get(r, {"trades": 0, "profit_factor": 0, "pnl": 0})
        pf_is = f"{d_is['profit_factor']:.2f}" if d_is.get('profit_factor', 0) < 100 else "inf"
        pf_oos = f"{d_oos['profit_factor']:.2f}" if d_oos.get('profit_factor', 0) < 100 else "inf"

        # Check stability: both negative, both positive, or sign-flip
        is_neg = d_is["pnl"] < 0
        oos_neg = d_oos["pnl"] < 0
        if d_is["trades"] == 0 or d_oos["trades"] == 0:
            stability = "N/A"
        elif is_neg == oos_neg:
            stability = "YES"
        else:
            stability = "** FLIP"

        print(f"  {r:<25s} {d_is['trades']:>9d} {pf_is:>7s} ${d_is['pnl']:>9,.0f}  |  "
              f"{d_oos['trades']:>10d} {pf_oos:>7s} ${d_oos['pnl']:>9,.0f}  {stability:>8s}")

    # 7. Determine blocked sets under three approaches
    blocked_full = get_blocked(bd_full)
    blocked_is = get_blocked(bd_is)
    blocked_oos = get_blocked(bd_oos)

    print(f"\n{'=' * 70}")
    print(f"  BLOCKED REGIMES UNDER EACH APPROACH")
    print(f"{'=' * 70}")
    print(f"  Full 2yr peek : {blocked_full if blocked_full else '(none)'}")
    print(f"  IS-only       : {blocked_is if blocked_is else '(none)'}")
    print(f"  OOS-only      : {blocked_oos if blocked_oos else '(none)'}")

    # 8. Compute WF stats under each approach + unfiltered baseline
    results = {}

    # Unfiltered baseline
    results["Unfiltered"] = compute_wf_stats(tagged, [])

    # Full peek (current approach)
    results["Full Peek (2yr)"] = compute_wf_stats(tagged, blocked_full)

    # IS-only (proper walk-forward)
    results["IS-Only Filter"] = compute_wf_stats(tagged, blocked_is)

    # OOS-only (oracle)
    results["OOS-Only (oracle)"] = compute_wf_stats(tagged, blocked_oos)

    # 9. Print comparison table
    print(f"\n{'=' * 90}")
    print(f"  WALK-FORWARD COMPARISON TABLE")
    print(f"  IS / OOS boundary: {SPLIT_DATE}")
    print(f"{'=' * 90}")
    print(f"  {'Approach':<22s} {'Blocked':<30s} {'IS PF':>7s} {'OOS PF':>7s} "
          f"{'WF Ratio':>9s} {'OOS P&L':>10s} {'OOS N':>6s}")
    print(f"  {'-' * 86}")

    for label, r in results.items():
        blocked_str = ", ".join(r["blocked"]) if r["blocked"] else "(none)"
        if len(blocked_str) > 28:
            blocked_str = blocked_str[:25] + "..."
        is_pf = f"{r['is']['pf']:.3f}" if r['is']['pf'] < 100 else "inf"
        oos_pf = f"{r['oos']['pf']:.3f}" if r['oos']['pf'] < 100 else "inf"
        ratio_str = f"{r['pf_ratio']:.2f}"
        quality = ""
        if r["pf_ratio"] >= 0.7:
            quality = " ROBUST"
        elif r["pf_ratio"] >= 0.5:
            quality = " OK"
        else:
            quality = " WEAK"
        print(f"  {label:<22s} {blocked_str:<30s} {is_pf:>7s} {oos_pf:>7s} "
              f"{ratio_str:>7s}{quality:<7s} ${r['oos']['pnl']:>9,.0f} {r['oos']['n']:>6d}")

    # 10. Diagnosis
    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSIS")
    print(f"{'=' * 70}")

    base_ratio = results["Unfiltered"]["pf_ratio"]
    full_ratio = results["Full Peek (2yr)"]["pf_ratio"]
    is_ratio = results["IS-Only Filter"]["pf_ratio"]

    if abs(is_ratio - base_ratio) < 0.10 and full_ratio < base_ratio - 0.15:
        print("  CONFIRMED: Full-peek regime filter causes WF degradation.")
        print("  The IS-only filter preserves the walk-forward ratio because it")
        print("  doesn't leak OOS information into the blocking decision.")
        if blocked_full != blocked_is:
            diff = set(blocked_full) - set(blocked_is)
            extra = set(blocked_is) - set(blocked_full)
            if diff:
                print(f"\n  Regimes blocked by full-peek but NOT by IS-only: {sorted(diff)}")
                for r in sorted(diff):
                    is_d = bd_is.get(r, {"pnl": 0, "trades": 0})
                    oos_d = bd_oos.get(r, {"pnl": 0, "trades": 0})
                    print(f"    {r}: IS P&L ${is_d['pnl']:+,.0f} ({is_d['trades']} trades), "
                          f"OOS P&L ${oos_d['pnl']:+,.0f} ({oos_d['trades']} trades)")
                    print(f"    -> Negative in 2yr aggregate but POSITIVE in one half = sign flip")
            if extra:
                print(f"\n  Regimes blocked by IS-only but NOT by full-peek: {sorted(extra)}")
                for r in sorted(extra):
                    is_d = bd_is.get(r, {"pnl": 0, "trades": 0})
                    oos_d = bd_oos.get(r, {"pnl": 0, "trades": 0})
                    print(f"    {r}: IS P&L ${is_d['pnl']:+,.0f} ({is_d['trades']} trades), "
                          f"OOS P&L ${oos_d['pnl']:+,.0f} ({oos_d['trades']} trades)")
    elif is_ratio < base_ratio - 0.15:
        print("  Both full-peek AND IS-only filters degrade WF ratio.")
        print("  The regime filter may not add value for this strategy regardless")
        print("  of how it's trained. Consider dropping regime gating for LVL.")
    else:
        print("  Mixed results — see table above for detailed comparison.")
        print(f"  Unfiltered WF ratio: {base_ratio:.2f}")
        print(f"  Full-peek WF ratio:  {full_ratio:.2f}")
        print(f"  IS-only WF ratio:    {is_ratio:.2f}")

    print(f"\n  Recommendation:")
    if is_ratio >= 0.7:
        print(f"  -> IS-only filter (WF ratio {is_ratio:.2f}) is walk-forward valid.")
        print(f"     Use IS-trained regime blocking in production.")
    elif is_ratio >= 0.5 and is_ratio > full_ratio + 0.10:
        print(f"  -> IS-only filter is better than full-peek ({is_ratio:.2f} vs {full_ratio:.2f})")
        print(f"     but still below 0.70 threshold. Proceed with caution.")
    else:
        print(f"  -> Regime filtering does not help LVL in a walk-forward context.")
        print(f"     Deploy LVL unfiltered (WF ratio {base_ratio:.2f}).")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
