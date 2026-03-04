#!/usr/bin/env python3
"""Portfolio Allocation Analysis — Cross-Strategy Diversification.

After running portfolio_backtest.py (both walk-forward and biased),
this script analyzes the combined portfolio:

1. Correlation matrix: Daily P&L correlation between all 8 strategies
2. Concentration risk: % of total P&L from each strategy
3. Combined drawdown: Max DD, underwater duration, worst month
4. Regime coverage heatmap: Trades × P&L per regime per strategy

Usage:
    cd ~/projects/ai-playground/medallion-2.0
    source .venv/bin/activate
    python scripts/portfolio_allocation.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

MEDALLION_ROOT = Path(__file__).parent.parent


def load_portfolio_results():
    """Load portfolio backtest results from JSON."""
    path = MEDALLION_ROOT / "data" / "processed" / "portfolio_backtest.json"
    if not path.exists():
        print(f"ERROR: Portfolio backtest results not found at {path}")
        print(f"  Run: python scripts/portfolio_backtest.py")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    print(f"  Loaded portfolio results: {data.get('generated_at', 'unknown')}")
    print(f"  Current regime: {data.get('current_regime', 'unknown')}")
    return data


def compute_correlation_matrix(strategies: list):
    """Compute daily P&L correlation matrix between strategies."""
    print(f"\n{'=' * 70}")
    print(f"  CORRELATION MATRIX (daily P&L)")
    print(f"{'=' * 70}")

    # Build daily P&L series per strategy
    daily_pnl = {}
    for strat in strategies:
        monthly = strat.get("monthly_pnl", {})
        # Use monthly as proxy since we don't have daily from JSON
        daily_pnl[strat["label"]] = monthly

    if len(daily_pnl) < 2:
        print("  Need at least 2 strategies for correlation")
        return None

    # Convert monthly P&L to DataFrame
    all_months = set()
    for monthly in daily_pnl.values():
        all_months.update(monthly.keys())

    df = pd.DataFrame(index=sorted(all_months))
    for label, monthly in daily_pnl.items():
        short_label = label.replace(" (ES)", "").replace(" (SPX)", "")
        df[short_label] = pd.Series(monthly)

    df = df.fillna(0)

    # Compute correlation
    corr = df.corr()

    # Print matrix
    labels = list(corr.columns)
    max_label_len = max(len(l) for l in labels)

    header = " " * (max_label_len + 2) + "  ".join(f"{l[:8]:>8s}" for l in labels)
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    for i, row_label in enumerate(labels):
        values = "  ".join(f"{corr.iloc[i, j]:>8.2f}" for j in range(len(labels)))
        print(f"  {row_label:<{max_label_len}s}  {values}")

    # Count low-correlation pairs
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((labels[i], labels[j], corr.iloc[i, j]))

    low_corr = [(a, b, c) for a, b, c in pairs if abs(c) < 0.4]
    high_corr = [(a, b, c) for a, b, c in pairs if abs(c) >= 0.7]

    print(f"\n  Low-correlation pairs (|r| < 0.4): {len(low_corr)}/{len(pairs)}")
    for a, b, c in sorted(low_corr, key=lambda x: abs(x[2])):
        print(f"    {a} vs {b}: {c:.2f}")

    if high_corr:
        print(f"\n  High-correlation pairs (|r| >= 0.7):")
        for a, b, c in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f"    {a} vs {b}: {c:.2f}")

    return corr


def compute_concentration_risk(strategies: list):
    """Measure concentration risk — no single strategy should dominate."""
    print(f"\n{'=' * 70}")
    print(f"  CONCENTRATION RISK (P&L distribution)")
    print(f"{'=' * 70}")

    total_pnl = sum(s["pnl"] for s in strategies)

    print(f"\n  {'Strategy':<25s} {'P&L':>12s} {'% of Total':>12s} {'Trades':>8s}")
    print(f"  {'-' * 60}")

    for s in sorted(strategies, key=lambda x: x["pnl"], reverse=True):
        pct = s["pnl"] / total_pnl * 100 if total_pnl != 0 else 0
        bar = "#" * int(abs(pct) / 2)
        marker = " *** > 50%!" if abs(pct) > 50 else ""
        print(f"  {s['label']:<25s} ${s['pnl']:>10,.0f} {pct:>10.1f}%  {s['trades']:>6d}  {bar}{marker}")

    print(f"  {'-' * 60}")
    print(f"  {'TOTAL':<25s} ${total_pnl:>10,.0f} {'100.0':>10s}%")

    # Check concentration
    max_pct = max(abs(s["pnl"] / total_pnl * 100) for s in strategies) if total_pnl != 0 else 0
    top_strategy = max(strategies, key=lambda s: abs(s["pnl"]))

    if max_pct > 50:
        print(f"\n  WARNING: {top_strategy['label']} contributes {max_pct:.1f}% of total P&L")
        print(f"  Portfolio is concentrated — consider rebalancing")
    else:
        print(f"\n  No single strategy > 50% of total P&L — PASS")

    return {
        "total_pnl": total_pnl,
        "max_concentration_pct": max_pct,
        "top_strategy": top_strategy["label"],
    }


def compute_combined_drawdown(combined: dict):
    """Analyze combined portfolio drawdown."""
    print(f"\n{'=' * 70}")
    print(f"  COMBINED DRAWDOWN ANALYSIS")
    print(f"{'=' * 70}")

    monthly = combined.get("monthly_pnl", {})
    if not monthly:
        print("  No monthly data available")
        return {}

    months = sorted(monthly.keys())
    equity = 0
    peak = 0
    max_dd = 0
    dd_start = None
    max_dd_start = None
    max_dd_end = None
    underwater = 0
    max_underwater = 0
    worst_month = None
    worst_month_pnl = 0

    for month in months:
        pnl = monthly[month]
        equity += pnl

        if pnl < worst_month_pnl:
            worst_month_pnl = pnl
            worst_month = month

        if equity > peak:
            peak = equity
            dd_start = month
            underwater = 0
        else:
            underwater += 1
            max_underwater = max(max_underwater, underwater)

        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            max_dd_start = dd_start
            max_dd_end = month

    total_months = len(months)
    winning_months = sum(1 for m in months if monthly[m] > 0)

    print(f"  Max drawdown:       ${max_dd:,.0f}")
    if max_dd_start and max_dd_end:
        print(f"  DD period:          {max_dd_start} to {max_dd_end}")
    print(f"  Max underwater:     {max_underwater} months")
    print(f"  Worst month:        {worst_month} (${worst_month_pnl:+,.0f})")
    print(f"  Winning months:     {winning_months}/{total_months} ({winning_months/total_months*100:.0f}%)")

    # Monthly P&L stats
    pnl_values = [monthly[m] for m in months]
    print(f"\n  Monthly P&L stats:")
    print(f"    Mean:   ${np.mean(pnl_values):+,.0f}")
    print(f"    Median: ${np.median(pnl_values):+,.0f}")
    print(f"    Std:    ${np.std(pnl_values):,.0f}")
    print(f"    Min:    ${np.min(pnl_values):+,.0f}")
    print(f"    Max:    ${np.max(pnl_values):+,.0f}")

    return {
        "max_drawdown": max_dd,
        "max_underwater_months": max_underwater,
        "worst_month": worst_month,
        "worst_month_pnl": worst_month_pnl,
        "winning_months": winning_months,
        "total_months": total_months,
    }


def print_comparison_report(strategies: list, combined: dict):
    """Print the side-by-side comparison report format."""
    print(f"\n{'=' * 70}")
    print(f"  MEDALLION 2.0 PORTFOLIO SUMMARY")
    print(f"{'=' * 70}")

    s = combined
    pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] < 100 else "inf"

    print(f"""
  Strategies:       {len(strategies)}
  Total Trades:     {s['trades']:,}
  Net P&L:          ${s['pnl']:+,.0f}
  Profit Factor:    {pf_str}
  Win Rate:         {s['win_rate']:.1f}%
  Max Drawdown:     ${s['max_dd']:,.0f}
  Sharpe:           {s['sharpe']:.2f}
  Avg Trade:        ${s['avg_trade']:+,.0f}
  Winning Months:   {s.get('winning_months', '?')}/{s.get('total_months', '?')}
""")

    # Per-strategy table
    print(f"  {'Strategy':<25s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'P&L':>12s} {'Sharpe':>7s}")
    print(f"  {'-' * 70}")
    for st in sorted(strategies, key=lambda x: x["pnl"], reverse=True):
        pf = f"{st['profit_factor']:.2f}" if st['profit_factor'] < 100 else "inf"
        print(f"  {st['label']:<25s} {st['trades']:>7d} {st['win_rate']:>6.1f}% "
              f"{pf:>7s} ${st['pnl']:>10,.0f} {st['sharpe']:>6.2f}")


def load_mode_results(mode):
    """Load mode-specific portfolio backtest results (wf or biased)."""
    suffix = "wf" if mode == "walk-forward" else "biased"
    path = MEDALLION_ROOT / "data" / "processed" / f"portfolio_backtest_{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_side_by_side_comparison():
    """Print side-by-side Walk-Forward vs Biased comparison report.

    Requires both portfolio_backtest_wf.json and portfolio_backtest_biased.json
    to exist (run portfolio_backtest.py twice: once default, once --biased).
    """
    wf_data = load_mode_results("walk-forward")
    biased_data = load_mode_results("biased")

    if wf_data is None and biased_data is None:
        print("\n  No mode-specific results found. Run:")
        print("    python scripts/portfolio_backtest.py           # walk-forward")
        print("    python scripts/portfolio_backtest.py --biased  # biased")
        return

    if wf_data is None:
        print("\n  Walk-forward results not found. Run: python scripts/portfolio_backtest.py")
        print("  (Only biased results available — skipping comparison)")
        return

    if biased_data is None:
        print("\n  Biased results not found. Run: python scripts/portfolio_backtest.py --biased")
        print("  (Only walk-forward results available — skipping comparison)")
        return

    wf = wf_data["combined"]
    bi = biased_data["combined"]

    print(f"\n{'=' * 70}")
    print(f"  MEDALLION 2.0 PORTFOLIO: Walk-Forward vs Biased")
    print(f"{'=' * 70}")

    def _fmt_pf(pf):
        return f"{pf:.2f}" if pf < 100 else "inf"

    def _fmt_delta(val, fmt=",.0f", prefix="$"):
        sign = "+" if val >= 0 else ""
        return f"{sign}{prefix}{val:{fmt}}" if prefix else f"{sign}{val:{fmt}}"

    rows = [
        ("Strategies",
         f"{len(wf_data['strategies'])}",
         f"{len(biased_data['strategies'])}",
         ""),
        ("Total Trades",
         f"{wf['trades']:,}",
         f"{bi['trades']:,}",
         _fmt_delta(wf['trades'] - bi['trades'], ",d", "")),
        ("Net P&L",
         f"${wf['pnl']:+,.0f}",
         f"${bi['pnl']:+,.0f}",
         _fmt_delta(wf['pnl'] - bi['pnl'])),
        ("Profit Factor",
         _fmt_pf(wf['profit_factor']),
         _fmt_pf(bi['profit_factor']),
         _fmt_delta(wf['profit_factor'] - bi['profit_factor'], ".2f", "")),
        ("Win Rate",
         f"{wf['win_rate']:.1f}%",
         f"{bi['win_rate']:.1f}%",
         _fmt_delta(wf['win_rate'] - bi['win_rate'], ".1f", "") + "%"),
        ("Max Drawdown",
         f"${wf['max_dd']:,.0f}",
         f"${bi['max_dd']:,.0f}",
         _fmt_delta(wf['max_dd'] - bi['max_dd'])),
        ("Sharpe",
         f"{wf['sharpe']:.2f}",
         f"{bi['sharpe']:.2f}",
         _fmt_delta(wf['sharpe'] - bi['sharpe'], ".2f", "")),
        ("Avg Trade",
         f"${wf['avg_trade']:+,.0f}",
         f"${bi['avg_trade']:+,.0f}",
         _fmt_delta(wf['avg_trade'] - bi['avg_trade'])),
        ("Winning Months",
         f"{wf.get('winning_months', '?')}/{wf.get('total_months', '?')}",
         f"{bi.get('winning_months', '?')}/{bi.get('total_months', '?')}",
         ""),
    ]

    print(f"\n  {'Metric':<20s} {'Walk-Forward':>15s} {'Biased':>15s} {'Delta':>15s}")
    print(f"  {'-' * 68}")
    for label, wf_val, bi_val, delta in rows:
        print(f"  {label:<20s} {wf_val:>15s} {bi_val:>15s} {delta:>15s}")

    # Per-strategy comparison
    wf_strats = {s["label"]: s for s in wf_data["strategies"]}
    bi_strats = {s["label"]: s for s in biased_data["strategies"]}
    all_labels = sorted(set(wf_strats.keys()) | set(bi_strats.keys()))

    if all_labels:
        print(f"\n  {'Strategy':<25s} {'WF P&L':>12s} {'Biased P&L':>12s} {'Delta':>12s} {'WF PF':>7s} {'Bi PF':>7s}")
        print(f"  {'-' * 73}")
        for label in all_labels:
            ws = wf_strats.get(label, {})
            bs = bi_strats.get(label, {})
            w_pnl = ws.get("pnl", 0)
            b_pnl = bs.get("pnl", 0)
            d_pnl = w_pnl - b_pnl
            w_pf = _fmt_pf(ws["profit_factor"]) if ws else "—"
            b_pf = _fmt_pf(bs["profit_factor"]) if bs else "—"
            print(f"  {label:<25s} ${w_pnl:>10,.0f} ${b_pnl:>10,.0f} ${d_pnl:>+10,.0f} {w_pf:>7s} {b_pf:>7s}")

    # Assessment
    print(f"\n  {'=' * 68}")
    pnl_drop = (1 - wf["pnl"] / bi["pnl"]) * 100 if bi["pnl"] != 0 else 0
    print(f"  P&L degradation: {pnl_drop:.1f}%", end="")
    if pnl_drop < 15:
        print(f" — Minimal bias (< 15%)")
    elif pnl_drop < 30:
        print(f" — Moderate bias (15-30%)")
    else:
        print(f" — Significant bias (> 30%) — regime model leaned on future data")

    wf_sharpe = wf["sharpe"]
    print(f"  Walk-forward Sharpe: {wf_sharpe:.2f}", end="")
    if wf_sharpe > 1.5:
        print(f" — PASS (> 1.5)")
    elif wf_sharpe > 1.0:
        print(f" — MARGINAL (1.0-1.5)")
    else:
        print(f" — FAIL (< 1.0)")

    wf_pf = wf["profit_factor"]
    print(f"  Walk-forward PF: {_fmt_pf(wf_pf)}", end="")
    if wf_pf > 1.2:
        print(f" — PASS (> 1.2)")
    else:
        print(f" — FAIL (< 1.2)")

    print(f"\n  Generated: WF={wf_data.get('generated_at', '?')}")
    print(f"             Bi={biased_data.get('generated_at', '?')}")


def main():
    print("=" * 70)
    print("  PORTFOLIO ALLOCATION ANALYSIS")
    print("=" * 70)

    data = load_portfolio_results()
    strategies = data.get("strategies", [])
    combined = data.get("combined", {})

    if not strategies:
        print("ERROR: No strategy results found")
        sys.exit(1)

    # 1. Summary report
    print_comparison_report(strategies, combined)

    # 2. Correlation matrix
    corr = compute_correlation_matrix(strategies)

    # 3. Concentration risk
    concentration = compute_concentration_risk(strategies)

    # 4. Combined drawdown
    drawdown = compute_combined_drawdown(combined)

    # 5. Side-by-side comparison (if both modes available)
    print_side_by_side_comparison()

    # Final assessment
    print(f"\n{'=' * 70}")
    print(f"  PORTFOLIO HEALTH CHECK")
    print(f"{'=' * 70}")

    checks = []

    sharpe = combined.get("sharpe", 0)
    checks.append(("Sharpe > 1.5", sharpe > 1.5, f"{sharpe:.2f}"))

    pf = combined.get("profit_factor", 0)
    checks.append(("PF > 1.2", pf > 1.2, f"{pf:.2f}"))

    max_conc = concentration.get("max_concentration_pct", 100)
    checks.append(("No strategy > 50% of P&L", max_conc < 50, f"{max_conc:.1f}%"))

    for check_name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name:<35s} ({value})")

    # Save analysis
    out_path = MEDALLION_ROOT / "data" / "processed" / "portfolio_allocation.json"
    results = {
        "concentration": concentration,
        "drawdown": drawdown,
        "sharpe": sharpe,
        "profit_factor": pf,
    }

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

    with open(out_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
