#!/usr/bin/env python3
"""Nexus Long — Inverted VIX Filter per Setup

Thesis (from data):
  - MS_ONL (overnight low bounce): BETTER in high VIX — capitulation = real support
  - OS_GAP_DN (gap-down fade): BETTER in low VIX — calm gaps are noise, fill easily

Inversion: apply OPPOSITE VIX filters to each setup.
  - MS_ONL: require VIX open > threshold (fear environment)
  - OS_GAP_DN: require VIX open <= threshold (calm environment)

Sweep thresholds to find optimal split point.

Usage:
    cd ~/projects/backtesting/es
    python3 scripts/nexus_long_vix_inverted.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


def make_long_config():
    """Long-only MS+OS."""
    cfg = StrategyConfig()
    cfg.direction_filter = "long"
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
    cfg.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short"}

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


def report(label, trades):
    """Print compact metrics for a trade set."""
    if not trades:
        print(f"  {label:<45s}    0t")
        return 0, 0, 1.0

    pnls = [t.pnl_dollar for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(trades) * 100
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p <= 0))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    net = sum(pnls)
    avg = net / len(trades)
    _, p_val = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

    # Max drawdown
    peak = 0
    dd = 0
    eq = 0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)

    print(f"  {label:<45s}  {len(trades):>4d}t  {wr:>5.1f}%  PF {pf:>6.3f}  ${net:>+9,.0f}  avg ${avg:>+6,.0f}  DD ${dd:>6,.0f}  p={p_val:.4f} {sig}")
    return len(trades), pf, p_val


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

print("Loading data...")
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
vix = pd.read_parquet("../spx/data/vix_daily.parquet")
print(f"ES: {len(df):,} bars  |  VIX: {len(vix)} days\n")

# Run long-only backtest
cfg = make_long_config()
trades = run_backtest(df.copy(), cfg)

# Attach VIX
for t in trades:
    vix_row = vix[vix.index.date == t.entry_time.date()]
    if len(vix_row) > 0:
        t.vix_open = vix_row.iloc[0]["open"]
        t.vix_high = vix_row.iloc[0]["high"]
    else:
        t.vix_open = None
        t.vix_high = None

trades = [t for t in trades if t.vix_open is not None]
ms_onl = [t for t in trades if t.setup == "MS_ONL"]
os_gap = [t for t in trades if t.setup == "OS_GAP_DN"]

print(f"Total long trades: {len(trades)}  (MS_ONL={len(ms_onl)}, OS_GAP_DN={len(os_gap)})")

# ═══════════════════════════════════════════════════════════════
#  INVERTED VIX FILTER SWEEP
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  INVERTED VIX FILTER: MS_ONL keeps HIGH VIX, OS_GAP_DN keeps LOW VIX")
print(f"{'=' * 120}")
print(f"  {'Config':<45s}  {'#':>5s}  {'WR':>6s}  {'PF':>9s}  {'P&L':>10s}  {'Avg':>9s}  {'DD':>8s}  {'Sig':>12s}")
print(f"  {'─' * 115}")

# Baseline: no filter
report("BASELINE: all longs, no VIX filter", trades)
print()

# Component baselines
report("  MS_ONL — all", ms_onl)
report("  OS_GAP_DN — all", os_gap)
print()

# Sweep
thresholds = [12, 14, 15, 16, 17, 18, 19, 20]
best_pf = 0
best_thresh = 0
best_pnl = 0

for thresh in thresholds:
    # MS_ONL: keep high VIX (fear bounces)
    ms_filtered = [t for t in ms_onl if t.vix_open > thresh]
    # OS_GAP_DN: keep low VIX (calm gap fades)
    os_filtered = [t for t in os_gap if t.vix_open <= thresh]
    # Combined
    combined = ms_filtered + os_filtered
    combined.sort(key=lambda t: t.entry_time)

    label = f"VIX split @ {thresh} (MS>={thresh}, OS<={thresh})"
    n, pf, p = report(label, combined)

    if combined:
        ms_pnl = sum(t.pnl_dollar for t in ms_filtered)
        os_pnl = sum(t.pnl_dollar for t in os_filtered)
        print(f"    ├─ MS_ONL (VIX>{thresh}): {len(ms_filtered):>3d}t  ${ms_pnl:>+8,.0f}")
        print(f"    └─ OS_GAP_DN (VIX<={thresh}): {len(os_filtered):>3d}t  ${os_pnl:>+8,.0f}")

    net = sum(t.pnl_dollar for t in combined) if combined else 0
    if pf > best_pf and n >= 20:
        best_pf = pf
        best_thresh = thresh
        best_pnl = net

    print()

# ═══════════════════════════════════════════════════════════════
#  BEST CONFIG — DETAILED BREAKDOWN
# ═══════════════════════════════════════════════════════════════
print(f"{'=' * 120}")
print(f"  BEST CONFIG: VIX split @ {best_thresh}")
print(f"{'=' * 120}")

ms_best = [t for t in ms_onl if t.vix_open > best_thresh]
os_best = [t for t in os_gap if t.vix_open <= best_thresh]
combined_best = sorted(ms_best + os_best, key=lambda t: t.entry_time)

if combined_best:
    pnls = [t.pnl_dollar for t in combined_best]
    _, p_val = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)

    # Bootstrap
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(pnls, size=len(pnls), replace=True)) for _ in range(10000)]
    p_profit = np.mean([b > 0 for b in boot_means]) * 100
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # Permutation
    obs_mean = np.mean(pnls)
    perm_count = sum(1 for _ in range(5000)
                     if np.mean(rng.choice([-1, 1], size=len(pnls)) * np.abs(pnls)) >= obs_mean)
    p_perm = perm_count / 5000

    # Walk-forward
    split_date = pd.Timestamp("2025-02-14", tz="US/Eastern")
    t1 = [t for t in combined_best if t.entry_time < split_date]
    t2 = [t for t in combined_best if t.entry_time >= split_date]

    print(f"\n  Trades:         {len(combined_best)}")
    wins = sum(1 for p in pnls if p > 0)
    print(f"  Win Rate:       {wins/len(combined_best)*100:.1f}%")
    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p <= 0))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    print(f"  Profit Factor:  {pf:.3f}")
    print(f"  Net P&L:        ${sum(pnls):>+,.0f}")
    print(f"  Avg Trade:      ${np.mean(pnls):>+,.0f}")
    print(f"  t-test p:       {p_val:.6f}")
    print(f"  Permutation p:  {p_perm:.6f}")
    print(f"  Bootstrap P(profit): {p_profit:.1f}%")
    print(f"  95% CI avg:     ${ci_lo:+,.0f} to ${ci_hi:+,.0f}")

    # Monthly
    monthly = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in combined_best:
        key = t.exit_time.strftime("%Y-%m")
        monthly[key]["count"] += 1
        monthly[key]["pnl"] += t.pnl_dollar
        if t.pnl_dollar > 0:
            monthly[key]["wins"] += 1

    win_months = sum(1 for d in monthly.values() if d["pnl"] > 0)
    print(f"  Monthly:        {win_months}/{len(monthly)} winning ({win_months/len(monthly)*100:.0f}%)")

    print(f"\n  {'Month':<8s}  {'#':>3s}  {'P&L':>9s}  {'Cum':>9s}")
    cum = 0
    for key in sorted(monthly.keys()):
        d = monthly[key]
        cum += d["pnl"]
        bar = "+" * max(0, int(d["pnl"] / 300)) + "-" * max(0, int(-d["pnl"] / 300))
        print(f"  {key:<8s}  {d['count']:>3d}  ${d['pnl']:>+8,.0f}  ${cum:>+8,.0f}  {bar}")

    if t1 and t2 and len(t1) >= 5 and len(t2) >= 5:
        pnls1 = [t.pnl_dollar for t in t1]
        pnls2 = [t.pnl_dollar for t in t2]
        gw1 = sum(p for p in pnls1 if p > 0)
        gl1 = abs(sum(p for p in pnls1 if p <= 0))
        gw2 = sum(p for p in pnls2 if p > 0)
        gl2 = abs(sum(p for p in pnls2 if p <= 0))
        pf1 = gw1 / gl1 if gl1 > 0 else float("inf")
        pf2 = gw2 / gl2 if gl2 > 0 else float("inf")
        _, p2 = stats.ttest_1samp(pnls2, 0)
        ratio = pf2 / pf1 if pf1 > 0 else 0
        verdict = "PASS" if ratio > 0.7 and pf2 > 1.0 else "MARGINAL" if ratio > 0.5 else "FAIL"

        print(f"\n  Walk-Forward (split {split_date.date()}):")
        print(f"    Y1: {len(t1):>3d}t  PF={pf1:.3f}  ${sum(pnls1):>+8,.0f}")
        print(f"    Y2: {len(t2):>3d}t  PF={pf2:.3f}  ${sum(pnls2):>+8,.0f}  p={p2:.4f}")
        print(f"    WF ratio: {ratio:.2f} → {verdict}")

# ═══════════════════════════════════════════════════════════════
#  COMPARISON: MS_ONL-only (no OS) with VIX filter
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 120}")
print(f"  ALTERNATIVE: MS_ONL only (drop OS_GAP_DN entirely)")
print(f"{'=' * 120}")
print(f"  {'Config':<45s}  {'#':>5s}  {'WR':>6s}  {'PF':>9s}  {'P&L':>10s}  {'Avg':>9s}  {'DD':>8s}  {'Sig':>12s}")
print(f"  {'─' * 115}")

report("MS_ONL — no filter", ms_onl)
for thresh in [14, 16, 18, 20]:
    filtered = [t for t in ms_onl if t.vix_open > thresh]
    report(f"MS_ONL — VIX > {thresh}", filtered)

# Also test: MS_ONL + VIX spike exclusion
no_spike = [t for t in ms_onl if t.vix_high / t.vix_open < 1.10]
report("MS_ONL — exclude 10%+ VIX spike days", no_spike)

# Combined: VIX > 16 AND no 10% spike
best_combo = [t for t in ms_onl if t.vix_open > 16 and t.vix_high / t.vix_open < 1.10]
report("MS_ONL — VIX > 16 AND no 10% spike", best_combo)

print()
