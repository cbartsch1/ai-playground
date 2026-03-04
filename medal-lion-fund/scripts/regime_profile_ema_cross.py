#!/usr/bin/env python3
"""Regime Profiling — EMA Cross (Credit Spreads Both Sides).

Pure data-driven regime profiling WITH per-side breakdown.
Runs the full EMA Cross backtest (both call and put credit spreads),
then post-hoc tags every trade with the HMM regime at entry time.

Key difference from other profilers: per-SIDE analysis.
- Put credits (bullish side): which regimes hurt?
- Call credits (bearish side): which regimes hurt?

Outputs:
  1. Combined per-regime breakdown
  2. Per-side (call/put credit) per-regime breakdown
  3. Monthly P&L timeline with dominant regime per month
  4. JSON results for Medallion registry consumption

Usage:
    cd ~/projects/ai-playground/medallion-2.0
    source .venv/bin/activate
    python scripts/regime_profile_ema_cross.py

    # Also run regime-filtered validation:
    python scripts/regime_profile_ema_cross.py --validate
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Project roots
MEDALLION_ROOT = Path(__file__).parent.parent
AI_PLAYGROUND_ROOT = MEDALLION_ROOT.parent
SPX_OPTIONS_ROOT = AI_PLAYGROUND_ROOT.parent / "spx-options"


def _swap_path(project_root):
    """Set sys.path so only MEDALLION + the given project root are available."""
    cleaned = [p for p in sys.path if p not in (
        str(AI_PLAYGROUND_ROOT), str(SPX_OPTIONS_ROOT),
    )]
    sys.path[:] = cleaned

    if str(MEDALLION_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDALLION_ROOT))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    mods_to_remove = [k for k in sys.modules if k.startswith("backtester")]
    for k in mods_to_remove:
        del sys.modules[k]


def load_regime_filter():
    """Load saved HMM model, download SPY hourly data, create RegimeFilter."""
    _swap_path(MEDALLION_ROOT)

    from models.hmm_regime import RegimeDetector
    from models.regime_api import RegimeFilter
    from data.data_loader import download_ohlcv, compute_hmm_features

    print("=" * 70)
    print("  LOADING MEDALLION 2.0 REGIME MODEL")
    print("=" * 70)

    detector = RegimeDetector.load_latest(n_regimes=7)
    if detector is None:
        print("ERROR: No saved 7-state model found. Run dashboard first to fit.")
        sys.exit(1)

    print(f"  Model: 7-state Gaussian HMM")
    print(f"  Labels: {list(detector.regime_labels.values())}")

    spy_hourly = download_ohlcv("SPY", "1h", "730d")
    hmm_features = compute_hmm_features(spy_hourly)
    full_df = spy_hourly.join(hmm_features)

    rf = RegimeFilter(detector, full_df)

    current = detector.get_current_regime(full_df)
    print(f"\n  Current regime: {current['label']} ({current['confidence']:.1%} confidence)")

    predictions = rf.predictions
    dist = predictions["regime_label"].value_counts(normalize=True).sort_index()
    print(f"\n  Regime Distribution ({len(predictions)} hourly bars):")
    for label, pct in dist.items():
        print(f"    {label:<25s} {pct:>6.1%}")

    valid = predictions.dropna(subset=["regime_label"])
    print(f"\n  Date range: {valid.index[0]} to {valid.index[-1]}")

    return rf


def run_ema_cross_backtest():
    """Run EMA Cross backtest on full 5yr SPY 1m data. Returns all trades."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.ema_cross import EMACross
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    print(f"\n{'=' * 70}")
    print(f"  RUNNING EMA CROSS BACKTEST (no regime filter)")
    print(f"{'=' * 70}")

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    print(f"  Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = EMACross()
    trades = strategy.run_backtest(df, vix_data=vix)

    # Split by side
    short_trades = [t for t in trades if t.direction == "short"]  # call credit
    long_trades = [t for t in trades if t.direction == "long"]    # put credit

    print(f"  Total: {len(trades)} trades")
    print(f"  Call credits (short/bearish): {len(short_trades)} trades")
    print(f"  Put credits (long/bullish):   {len(long_trades)} trades")

    if trades:
        pnls = [t.net_pnl for t in trades]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"  Combined: WR {wr:.1f}% | PF {pf:.2f} | P&L ${sum(pnls):+,.0f}")

    return trades


def tag_trades_with_regime(regime_filter, trades):
    """Post-hoc tag each trade with HMM regime at entry time."""
    tagged = []
    for t in trades:
        regime = regime_filter.get_regime_at(t.entry_time)
        label = regime.get("label")
        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = "No Data (before model)"
        tagged.append((t, label))
    return tagged


def _get_pnl(trade):
    """Get P&L from trade — SPX trades use net_pnl."""
    return getattr(trade, 'net_pnl', 0)


def compute_regime_breakdown(tagged_trades, pnl_fn=_get_pnl):
    """Compute per-regime stats from tagged trades."""
    by_regime = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0, "pnls": []})

    for t, label in tagged_trades:
        pnl = pnl_fn(t)
        d = by_regime[label]
        d["trades"] += 1
        d["pnl"] += pnl
        d["pnls"].append(pnl)
        if pnl > 0:
            d["wins"] += 1

    breakdown = {}
    for label, d in by_regime.items():
        pnls = d["pnls"]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
        avg = d["pnl"] / d["trades"] if d["trades"] > 0 else 0
        breakdown[label] = {
            "trades": d["trades"],
            "pnl": d["pnl"],
            "win_rate": wr,
            "avg_trade": avg,
            "profit_factor": pf,
            "gross_win": gross_win,
            "gross_loss": gross_loss,
        }

    return breakdown


def compute_monthly_timeline(tagged_trades, pnl_fn=_get_pnl):
    """Compute monthly P&L with dominant regime per month."""
    monthly = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "regimes": defaultdict(int)})

    for t, label in tagged_trades:
        pnl = pnl_fn(t)
        month_key = t.exit_time.strftime("%Y-%m")
        monthly[month_key]["pnl"] += pnl
        monthly[month_key]["trades"] += 1
        monthly[month_key]["regimes"][label] += 1

    timeline = []
    for month in sorted(monthly.keys()):
        d = monthly[month]
        dominant = max(d["regimes"], key=d["regimes"].get)
        timeline.append({
            "month": month,
            "pnl": d["pnl"],
            "trades": d["trades"],
            "dominant_regime": dominant,
            "regime_counts": dict(d["regimes"]),
        })

    return timeline


def print_regime_breakdown(breakdown, label="EMA Cross Trades"):
    """Print per-regime breakdown table."""
    print(f"\n{'=' * 70}")
    print(f"  {label} — Per-Regime Breakdown")
    print(f"{'=' * 70}")

    header = (f"  {'Regime':<25s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
              f"{'P&L':>12s} {'Avg Trade':>10s}")
    print(header)
    print(f"  {'-' * 70}")

    for regime in sorted(breakdown.keys(), key=lambda r: breakdown[r]["pnl"], reverse=True):
        d = breakdown[regime]
        pf_str = f"{d['profit_factor']:.2f}" if d['profit_factor'] < 100 else "inf"
        print(f"  {regime:<25s} {d['trades']:>7d} {d['win_rate']:>6.1f}% "
              f"{pf_str:>7s} ${d['pnl']:>10,.0f} ${d['avg_trade']:>8,.0f}")

    total_trades = sum(d["trades"] for d in breakdown.values())
    total_pnl = sum(d["pnl"] for d in breakdown.values())
    total_wins = sum(d["trades"] * d["win_rate"] / 100 for d in breakdown.values())
    total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    total_gross_win = sum(d["gross_win"] for d in breakdown.values())
    total_gross_loss = sum(d["gross_loss"] for d in breakdown.values())
    total_pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else float("inf")
    total_avg = total_pnl / total_trades if total_trades > 0 else 0

    print(f"  {'-' * 70}")
    pf_str = f"{total_pf:.2f}" if total_pf < 100 else "inf"
    print(f"  {'TOTAL':<25s} {total_trades:>7d} {total_wr:>6.1f}% "
          f"{pf_str:>7s} ${total_pnl:>10,.0f} ${total_avg:>8,.0f}")


def print_monthly_timeline(timeline):
    """Print monthly P&L timeline with regime context."""
    print(f"\n{'=' * 70}")
    print(f"  MONTHLY P&L TIMELINE (with dominant regime)")
    print(f"{'=' * 70}")

    print(f"  {'Month':<10s} {'Trades':>7s} {'P&L':>12s} {'Cum P&L':>12s}  {'Dominant Regime':<25s}")
    print(f"  {'-' * 70}")

    cum_pnl = 0
    for m in timeline:
        cum_pnl += m["pnl"]
        bar = "+" * min(int(abs(m["pnl"]) / 2000), 20) if m["pnl"] > 0 else "-" * min(int(abs(m["pnl"]) / 2000), 20)
        print(f"  {m['month']:<10s} {m['trades']:>7d} ${m['pnl']:>10,.0f} ${cum_pnl:>10,.0f}  "
              f"{m['dominant_regime']:<25s} {bar}")


def identify_blocked_regimes(breakdown):
    """Identify regimes to block: any with negative total P&L."""
    blocked = []
    for regime, d in breakdown.items():
        if regime == "No Data (before model)":
            continue
        if d["pnl"] < 0:
            blocked.append(regime)
    return sorted(blocked)


def run_filtered_validation(regime_filter, blocked_call, blocked_put):
    """Re-run EMA Cross backtest, skipping per-side blocked regimes."""
    _swap_path(SPX_OPTIONS_ROOT)
    from backtester.strategies.ema_cross import EMACross
    from backtester.data_loader import load_spy_data, load_vix_data, tag_sessions

    print(f"\n{'=' * 70}")
    print(f"  REGIME-FILTERED VALIDATION (per-side)")
    print(f"  Blocked call credit (short): {blocked_call}")
    print(f"  Blocked put credit (long):   {blocked_put}")
    print(f"{'=' * 70}")

    df = load_spy_data(str(SPX_OPTIONS_ROOT / "data" / "spy_1m_rth.parquet"))
    df = tag_sessions(df)
    vix_path = SPX_OPTIONS_ROOT / "data" / "vix_daily.parquet"
    vix = load_vix_data(str(vix_path)) if vix_path.exists() else None

    strategy = EMACross()
    all_trades = strategy.run_backtest(df, vix_data=vix)

    kept = []
    removed = []
    for t in all_trades:
        regime = regime_filter.get_regime_at(t.entry_time)
        label = regime.get("label", "")
        # Per-side blocking
        if t.direction == "short" and label in blocked_call:
            removed.append(t)
        elif t.direction == "long" and label in blocked_put:
            removed.append(t)
        else:
            kept.append(t)

    print(f"  Baseline: {len(all_trades)} trades")
    print(f"  Removed (blocked regimes): {len(removed)} trades")
    print(f"  Kept (filtered): {len(kept)} trades")

    if kept:
        pnls = [t.net_pnl for t in kept]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        total_pnl = sum(pnls)
        avg_trade = total_pnl / len(pnls)

        print(f"\n  Filtered: {len(kept)} trades | WR {wr:.1f}% | PF {pf:.2f} | "
              f"P&L ${total_pnl:+,.0f} | Avg ${avg_trade:+,.0f}")

    return kept, removed


def save_results(combined_breakdown, call_breakdown, put_breakdown,
                 timeline, blocked_call, blocked_put, total_trades):
    """Save profiling results to JSON for registry consumption."""
    out_path = MEDALLION_ROOT / "data" / "processed" / "regime_profile_ema_cross.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(x) for x in obj]
        return obj

    results = {
        "generated_at": datetime.now().isoformat(),
        "strategy": "ema_cross",
        "instrument": "SPX",
        "direction": "both",
        "total_trades": total_trades,
        "blocked_regimes_call_credit": blocked_call,
        "blocked_regimes_put_credit": blocked_put,
        "combined_regime_breakdown": _clean(combined_breakdown),
        "call_credit_regime_breakdown": _clean(call_breakdown),
        "put_credit_regime_breakdown": _clean(put_breakdown),
        "monthly_timeline": _clean(timeline),
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Regime Profile — EMA Cross (per-side)")
    parser.add_argument("--validate", action="store_true",
                        help="Also run regime-filtered validation after profiling")
    args = parser.parse_args()

    # Step 1: Load regime model
    rf = load_regime_filter()

    # Step 2: Run EMA Cross backtest (no regime filter)
    trades = run_ema_cross_backtest()

    if not trades:
        print("\nERROR: No trades found. Check config.")
        sys.exit(1)

    # Step 3: Tag all trades with HMM regime (post-hoc)
    tagged_all = tag_trades_with_regime(rf, trades)

    # Split by side for per-side analysis
    tagged_call = [(t, label) for t, label in tagged_all if t.direction == "short"]
    tagged_put = [(t, label) for t, label in tagged_all if t.direction == "long"]

    # Step 4: Compute per-regime breakdowns
    combined_breakdown = compute_regime_breakdown(tagged_all)
    call_breakdown = compute_regime_breakdown(tagged_call)
    put_breakdown = compute_regime_breakdown(tagged_put)

    print_regime_breakdown(combined_breakdown, "EMA Cross (All Trades)")
    print_regime_breakdown(call_breakdown, "CALL Credit (Short/Bearish Side)")
    print_regime_breakdown(put_breakdown, "PUT Credit (Long/Bullish Side)")

    # Step 5: Monthly timeline
    timeline = compute_monthly_timeline(tagged_all)
    print_monthly_timeline(timeline)

    # Step 6: Identify blocked regimes PER SIDE
    blocked_call = identify_blocked_regimes(call_breakdown)
    blocked_put = identify_blocked_regimes(put_breakdown)

    print(f"\n{'=' * 70}")
    print(f"  REGIME GATING RECOMMENDATION (per-side, negative P&L = blocked)")
    print(f"{'=' * 70}")

    print(f"\n  CALL CREDITS (short/bearish side):")
    if blocked_call:
        print(f"    Block: {blocked_call}")
        blocked_pnl = sum(call_breakdown[r]["pnl"] for r in blocked_call)
        blocked_cnt = sum(call_breakdown[r]["trades"] for r in blocked_call)
        print(f"    Would remove {blocked_cnt} trades bleeding ${blocked_pnl:+,.0f}")
    else:
        print(f"    No regimes have negative P&L — no blocking needed")

    print(f"\n  PUT CREDITS (long/bullish side):")
    if blocked_put:
        print(f"    Block: {blocked_put}")
        blocked_pnl = sum(put_breakdown[r]["pnl"] for r in blocked_put)
        blocked_cnt = sum(put_breakdown[r]["trades"] for r in blocked_put)
        print(f"    Would remove {blocked_cnt} trades bleeding ${blocked_pnl:+,.0f}")
    else:
        print(f"    No regimes have negative P&L — no blocking needed")

    # Save results
    save_results(combined_breakdown, call_breakdown, put_breakdown,
                 timeline, blocked_call, blocked_put, len(trades))

    # Step 7: Optional — validate per-side regime filtering
    if args.validate and (blocked_call or blocked_put):
        run_filtered_validation(rf, set(blocked_call), set(blocked_put))

    print(f"\n{'=' * 70}")
    print(f"  DONE — Next: register in Medallion strategy registry")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
