#!/usr/bin/env python3
"""Regime Profiling — MS+OS Best (Market Structure + Overnight Sweep).

Pure data-driven regime profiling. No pre-filtering, no hints.
Runs the full MS+OS backtest, then post-hoc tags every trade with
the HMM regime active at entry time. Outputs:

  1. Per-regime breakdown (trade count, P&L, WR, avg trade, PF)
  2. Monthly P&L timeline with dominant regime per month
  3. JSON results for Medallion registry consumption

Usage:
    cd ~/projects/ai-playground/medallion-2.0
    source .venv/bin/activate
    python scripts/regime_profile_ms_os.py

    # Also run regime-filtered validation:
    python scripts/regime_profile_ms_os.py --validate
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


def _swap_path(project_root):
    """Set sys.path so only MEDALLION + the given project root are available.
    Clears cached 'backtester' modules to avoid cross-project import conflicts."""
    cleaned = [p for p in sys.path if p not in (
        str(AI_PLAYGROUND_ROOT),
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


def make_ms_os_config():
    """MS Config B + OS Best — the deployment candidate."""
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.config import StrategyConfig

    cfg = StrategyConfig()
    cfg.direction_filter = "both"

    # All setups OFF except MS and OS
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_fa = False

    # --- MS Config B ---
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 4.0
    cfg.ms_min_target_pts = 8.0
    cfg.ms_min_rr = 0.3
    cfg.ms_max_risk = 25.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False
    cfg.ms_level_directions = {
        "MS_ONH": "both",
        "MS_ONL": "both",
        "MS_pVAH": "short",
    }

    # --- OS Best ---
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = 5.0
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 3.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_require_ma = False
    cfg.max_os_trades = 1
    cfg.os_min_gap = 3.0
    cfg.os_max_gap = 20.0
    cfg.os_entry_window = 1

    return cfg


def run_ms_os_backtest():
    """Run MS+OS backtest on full 2yr ES data. Returns all trades."""
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.data_loader import load_tos_csv
    from backtester.engine import run_backtest

    data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento_2yr.csv"
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        sys.exit(1)

    cfg = make_ms_os_config()

    print(f"\n{'=' * 70}")
    print(f"  RUNNING MS+OS BACKTEST (no regime filter)")
    print(f"{'=' * 70}")

    df = load_tos_csv(str(data_path), instrument="ES")
    print(f"  Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    trades = run_backtest(df, cfg)

    print(f"  Total trades: {len(trades)}")

    if trades:
        pnls = [t.pnl_dollar for t in trades]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"  MS+OS: {len(trades)} trades | WR {wr:.1f}% | PF {pf:.3f} | P&L ${sum(pnls):+,.0f}")

    return trades, cfg


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


def compute_regime_breakdown(tagged_trades):
    """Compute per-regime stats from tagged trades."""
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


def compute_monthly_timeline(tagged_trades):
    """Compute monthly P&L with dominant regime per month."""
    monthly = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "regimes": defaultdict(int)})

    for t, label in tagged_trades:
        month_key = t.exit_time.strftime("%Y-%m")
        monthly[month_key]["pnl"] += t.pnl_dollar
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


def print_regime_breakdown(breakdown, label="MS+OS Trades"):
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
        bar = "+" * min(int(abs(m["pnl"]) / 500), 20) if m["pnl"] > 0 else "-" * min(int(abs(m["pnl"]) / 500), 20)
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


def run_filtered_validation(regime_filter, blocked_regimes):
    """Re-run MS+OS backtest, skipping trades in blocked regimes."""
    _swap_path(AI_PLAYGROUND_ROOT)
    from backtester.data_loader import load_tos_csv
    from backtester.engine import run_backtest

    data_path = AI_PLAYGROUND_ROOT / "data" / "es_5m_databento_2yr.csv"
    cfg = make_ms_os_config()
    df = load_tos_csv(str(data_path), instrument="ES")

    print(f"\n{'=' * 70}")
    print(f"  REGIME-FILTERED VALIDATION")
    print(f"  Blocked: {blocked_regimes}")
    print(f"{'=' * 70}")

    all_trades = run_backtest(df, cfg)

    # Filter out trades in blocked regimes
    kept = []
    removed = []
    for t in all_trades:
        regime = regime_filter.get_regime_at(t.entry_time)
        label = regime.get("label", "")
        if label in blocked_regimes:
            removed.append(t)
        else:
            kept.append(t)

    print(f"  MS+OS baseline: {len(all_trades)} trades")
    print(f"  Removed (blocked regimes): {len(removed)} trades")
    print(f"  Kept (filtered): {len(kept)} trades")

    if kept:
        pnls = [t.pnl_dollar for t in kept]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        total_pnl = sum(pnls)
        avg_trade = total_pnl / len(pnls)

        print(f"\n  Filtered MS+OS: {len(kept)} trades | WR {wr:.1f}% | PF {pf:.3f} | "
              f"P&L ${total_pnl:+,.0f} | Avg ${avg_trade:+,.0f}")

        # Sharpe
        daily_pnl = defaultdict(float)
        for t in kept:
            day = t.exit_time.strftime("%Y-%m-%d")
            daily_pnl[day] += t.pnl_dollar
        daily_arr = np.array(list(daily_pnl.values()))
        if len(daily_arr) > 1 and daily_arr.std() > 0:
            sharpe = (daily_arr.mean() / daily_arr.std()) * np.sqrt(252)
            print(f"  Sharpe (annualized): {sharpe:.2f}")

    if removed:
        r_pnls = [t.pnl_dollar for t in removed]
        print(f"\n  Removed trades P&L: ${sum(r_pnls):+,.0f} "
              f"(avg ${np.mean(r_pnls):+,.0f} per trade)")

    # Walk-forward split
    split_date = "2025-02-14"
    is_kept = [t for t in kept if str(t.entry_time) < split_date]
    oos_kept = [t for t in kept if str(t.entry_time) >= split_date]

    if is_kept and oos_kept:
        is_pnls = [t.pnl_dollar for t in is_kept]
        oos_pnls = [t.pnl_dollar for t in oos_kept]
        is_gw = sum(p for p in is_pnls if p > 0)
        is_gl = abs(sum(p for p in is_pnls if p <= 0))
        oos_gw = sum(p for p in oos_pnls if p > 0)
        oos_gl = abs(sum(p for p in oos_pnls if p <= 0))
        is_pf = is_gw / is_gl if is_gl > 0 else float("inf")
        oos_pf = oos_gw / oos_gl if oos_gl > 0 else float("inf")
        pf_ratio = oos_pf / is_pf if is_pf > 0 else 0

        print(f"\n  Walk-Forward (regime-filtered MS+OS):")
        print(f"    IS:  {len(is_kept)} trades | PF {is_pf:.3f} | P&L ${sum(is_pnls):+,.0f}")
        print(f"    OOS: {len(oos_kept)} trades | PF {oos_pf:.3f} | P&L ${sum(oos_pnls):+,.0f}")
        print(f"    PF ratio: {pf_ratio:.2f}", end="")
        if pf_ratio >= 0.7:
            print("  *** ROBUST")
        elif pf_ratio >= 0.5:
            print("  ** ACCEPTABLE")
        else:
            print("  * WEAK")

    return kept, removed


def save_results(breakdown, timeline, blocked_regimes, total_trades):
    """Save profiling results to JSON for registry consumption."""
    out_path = MEDALLION_ROOT / "data" / "processed" / "regime_profile_ms_os.json"
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
        "strategy": "ms_os_best",
        "instrument": "ES",
        "direction": "both",
        "total_trades": total_trades,
        "blocked_regimes": blocked_regimes,
        "regime_breakdown": _clean(breakdown),
        "monthly_timeline": _clean(timeline),
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Regime Profile — MS+OS Best")
    parser.add_argument("--validate", action="store_true",
                        help="Also run regime-filtered validation after profiling")
    args = parser.parse_args()

    # Step 1: Load regime model
    rf = load_regime_filter()

    # Step 2: Run MS+OS backtest (no regime filter)
    trades, cfg = run_ms_os_backtest()

    if not trades:
        print("\nERROR: No trades found. Check config.")
        sys.exit(1)

    # Step 3: Tag trades with HMM regime (post-hoc, no hints)
    tagged = tag_trades_with_regime(rf, trades)

    # Step 4: Compute per-regime breakdown
    breakdown = compute_regime_breakdown(tagged)
    print_regime_breakdown(breakdown)

    # Step 5: Monthly timeline
    timeline = compute_monthly_timeline(tagged)
    print_monthly_timeline(timeline)

    # Step 6: Identify blocked regimes (negative P&L = blocked)
    blocked = identify_blocked_regimes(breakdown)
    print(f"\n{'=' * 70}")
    print(f"  REGIME GATING RECOMMENDATION (organic — negative P&L = blocked)")
    print(f"{'=' * 70}")
    if blocked:
        print(f"  Block these regimes: {blocked}")
        blocked_pnl = sum(breakdown[r]["pnl"] for r in blocked)
        blocked_trades = sum(breakdown[r]["trades"] for r in blocked)
        print(f"  Would remove {blocked_trades} trades bleeding ${blocked_pnl:+,.0f}")
    else:
        print(f"  No regimes have negative P&L — no blocking needed")

    # Save results
    save_results(breakdown, timeline, blocked, len(trades))

    # Step 7: Optional — validate regime-filtered performance
    if args.validate and blocked:
        run_filtered_validation(rf, set(blocked))

    print(f"\n{'=' * 70}")
    print(f"  DONE — Next: register in Medallion strategy registry")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
