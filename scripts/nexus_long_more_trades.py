#!/usr/bin/env python3
"""Nexus Long — Trade Count Expansion

The VIX-inverted filter works (PF 1.584, WF 1.59) but only 75 trades / 2yr.
Need more trades to crack p < 0.05. Test levers that add trades without
diluting the edge.

Levers:
  1. Add IB Low as support bounce level (same Dalton thesis as ONL)
  2. Add developing POC/VAL as intraday support
  3. Widen zone (3.0 → 4.0, 5.0 pts)
  4. Widen OS entry window (1 → 2, 3 bars)
  5. Lower VIX threshold (17 → 15, 14)
  6. Combinations

Usage:
    cd ~/projects/backtesting/es
    python3 scripts/nexus_long_more_trades.py
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
from backtester.metrics import compute_metrics, per_setup_breakdown


def base_long_config():
    """Long-only MS+OS baseline."""
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


def apply_vix_filter(trades, vix_df, ms_thresh=17, os_thresh=17):
    """Apply inverted VIX filter: MS keeps high VIX, OS keeps low VIX."""
    result = []
    for t in trades:
        vix_row = vix_df[vix_df.index.date == t.entry_time.date()]
        if len(vix_row) == 0:
            continue
        vix_open = vix_row.iloc[0]["open"]
        vix_high = vix_row.iloc[0]["high"]

        if t.setup.startswith("MS_"):
            if vix_open > ms_thresh:
                t.vix_open = vix_open
                t.vix_high = vix_high
                result.append(t)
        elif t.setup.startswith("OS_"):
            if vix_open <= os_thresh:
                t.vix_open = vix_open
                t.vix_high = vix_high
                result.append(t)
        else:
            t.vix_open = vix_open
            t.vix_high = vix_high
            result.append(t)
    return result


def report(label, trades, show_setups=False, show_wf=False, df=None):
    """Compact metrics report."""
    if not trades:
        print(f"  {label:<50s}    0t")
        return

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
    peak = eq = 0
    dd = 0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)

    print(f"  {label:<50s}  {len(trades):>4d}t  {wr:>5.1f}%  PF {pf:>6.3f}  ${net:>+9,.0f}  DD ${dd:>5,.0f}  p={p_val:.4f} {sig}")

    if show_setups:
        by_setup = defaultdict(list)
        for t in trades:
            by_setup[t.setup].append(t)
        for setup in sorted(by_setup.keys(), key=lambda s: -sum(t.pnl_dollar for t in by_setup[s])):
            st = by_setup[setup]
            sp = [t.pnl_dollar for t in st]
            sw = sum(1 for p in sp if p > 0)
            sgw = sum(p for p in sp if p > 0)
            sgl = abs(sum(p for p in sp if p <= 0))
            spf = sgw / sgl if sgl > 0 else float("inf")
            swr = sw / len(st) * 100
            print(f"    {setup:<48s}  {len(st):>4d}t  {swr:>5.1f}%  PF {spf:>6.3f}  ${sum(sp):>+9,.0f}")

    if show_wf and df is not None:
        split_date = pd.Timestamp("2025-02-14", tz="US/Eastern")
        t1 = [t for t in trades if t.entry_time < split_date]
        t2 = [t for t in trades if t.entry_time >= split_date]
        if t1 and t2 and len(t1) >= 3 and len(t2) >= 3:
            p1 = [t.pnl_dollar for t in t1]
            p2 = [t.pnl_dollar for t in t2]
            gw1 = sum(p for p in p1 if p > 0)
            gl1 = abs(sum(p for p in p1 if p <= 0))
            gw2 = sum(p for p in p2 if p > 0)
            gl2 = abs(sum(p for p in p2 if p <= 0))
            pf1 = gw1 / gl1 if gl1 > 0 else float("inf")
            pf2 = gw2 / gl2 if gl2 > 0 else float("inf")
            _, p2v = stats.ttest_1samp(p2, 0) if len(p2) >= 5 else (0, 1.0)
            ratio = pf2 / pf1 if pf1 > 0 else 0
            verdict = "PASS" if ratio > 0.7 and pf2 > 1.0 else "MARGINAL" if ratio > 0.5 else "FAIL"
            # Bootstrap P(profit)
            rng = np.random.default_rng(42)
            boot = [np.mean(rng.choice(pnls, size=len(pnls), replace=True)) for _ in range(10000)]
            bp = np.mean([b > 0 for b in boot]) * 100
            # Permutation
            obs = np.mean(pnls)
            pc = sum(1 for _ in range(5000)
                     if np.mean(rng.choice([-1, 1], size=len(pnls)) * np.abs(pnls)) >= obs)
            pp = pc / 5000
            # Monthly
            monthly = defaultdict(float)
            for t in trades:
                monthly[t.exit_time.strftime("%Y-%m")] += t.pnl_dollar
            wm = sum(1 for v in monthly.values() if v > 0)
            print(f"    WF: Y1={len(t1)}t PF {pf1:.3f} ${sum(p1):>+8,.0f}  |  Y2={len(t2)}t PF {pf2:.3f} ${sum(p2):>+8,.0f} p={p2v:.4f}  |  ratio {ratio:.2f} → {verdict}")
            print(f"    Perm p={pp:.4f}  Bootstrap P(profit)={bp:.1f}%  Monthly: {wm}/{len(monthly)} ({wm/len(monthly)*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

print("Loading data...")
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
vix = pd.read_parquet("../spx/data/vix_daily.parquet")
print(f"ES: {len(df):,} bars  |  VIX: {len(vix)} days\n")

header = f"  {'Config':<50s}  {'#':>5s}  {'WR':>6s}  {'PF':>9s}  {'P&L':>10s}  {'DD':>6s}  {'Sig':>12s}"
sep = f"  {'─' * 110}"

# ═══════════════════════════════════════════════════════════════
#  LEVER 1: Add IB Low as support level
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  LEVER 1: Add IB Low bounce (same thesis as ONL bounce)")
print(f"{'=' * 115}")
print(header)
print(sep)

# Baseline
cfg0 = base_long_config()
t0 = run_backtest(df.copy(), cfg0)
t0f = apply_vix_filter(t0, vix)
report("Baseline (ONL+OS, VIX split @17)", t0f, show_setups=True)

# Add IB Low
cfg1 = base_long_config()
cfg1.ms_use_ib_levels = True
cfg1.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short", "MS_IBL": "long"}
t1 = run_backtest(df.copy(), cfg1)
t1f = apply_vix_filter(t1, vix)
report("+ IB Low (VIX split @17)", t1f, show_setups=True)

# Add IB Low + IB High breakout
cfg1b = base_long_config()
cfg1b.ms_use_ib_levels = True
cfg1b.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short", "MS_IBL": "long", "MS_IBH": "long"}
t1b = run_backtest(df.copy(), cfg1b)
t1bf = apply_vix_filter(t1b, vix)
report("+ IB Low + IB High breakout (VIX split @17)", t1bf, show_setups=True)

# ═══════════════════════════════════════════════════════════════
#  LEVER 2: Add developing VA/POC
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  LEVER 2: Add developing POC/VAL as intraday support")
print(f"{'=' * 115}")
print(header)
print(sep)

cfg2 = base_long_config()
cfg2.ms_use_dev_va = True
cfg2.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short", "MS_dVAL": "long"}
t2 = run_backtest(df.copy(), cfg2)
t2f = apply_vix_filter(t2, vix)
report("+ Dev VAL (VIX split @17)", t2f, show_setups=True)

cfg2b = base_long_config()
cfg2b.ms_use_dev_va = True
cfg2b.ms_use_poc = True
cfg2b.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short", "MS_dVAL": "long", "MS_dPOC": "long"}
t2b = run_backtest(df.copy(), cfg2b)
t2bf = apply_vix_filter(t2b, vix)
report("+ Dev VAL + Dev POC (VIX split @17)", t2bf, show_setups=True)

# ═══════════════════════════════════════════════════════════════
#  LEVER 3: Wider zone
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  LEVER 3: Wider proximity zone (more touches at same levels)")
print(f"{'=' * 115}")
print(header)
print(sep)

for zone in [3.0, 4.0, 5.0, 6.0]:
    cfg3 = base_long_config()
    cfg3.ms_zone_pts = zone
    t3 = run_backtest(df.copy(), cfg3)
    t3f = apply_vix_filter(t3, vix)
    report(f"Zone = {zone} pts (VIX split @17)", t3f)

# ═══════════════════════════════════════════════════════════════
#  LEVER 4: Wider OS entry window
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  LEVER 4: OS entry window (more bars to enter on gap-down days)")
print(f"{'=' * 115}")
print(header)
print(sep)

for window in [1, 2, 3, 6]:
    cfg4 = base_long_config()
    cfg4.os_entry_window = window
    t4 = run_backtest(df.copy(), cfg4)
    t4f = apply_vix_filter(t4, vix)
    report(f"OS window = {window} bars (VIX split @17)", t4f, show_setups=True)

# ═══════════════════════════════════════════════════════════════
#  LEVER 5: Looser R:R and target requirements
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  LEVER 5: Looser risk filters")
print(f"{'=' * 115}")
print(header)
print(sep)

for min_tgt in [8.0, 6.0, 5.0, 4.0]:
    cfg5 = base_long_config()
    cfg5.ms_min_target_pts = min_tgt
    t5 = run_backtest(df.copy(), cfg5)
    t5f = apply_vix_filter(t5, vix)
    report(f"min_target = {min_tgt} pts (VIX split @17)", t5f)

# ═══════════════════════════════════════════════════════════════
#  LEVER 6: VIX threshold sensitivity
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  LEVER 6: VIX threshold tuning (fine sweep)")
print(f"{'=' * 115}")
print(header)
print(sep)

for ms_t in [14, 15, 16, 17, 18]:
    for os_t in [15, 16, 17, 18, 19, 20]:
        cfg6 = base_long_config()
        t6 = run_backtest(df.copy(), cfg6)
        t6f = apply_vix_filter(t6, vix, ms_thresh=ms_t, os_thresh=os_t)
        if t6f:
            pnls = [t.pnl_dollar for t in t6f]
            _, p = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)
            if p < 0.15 and len(t6f) >= 30:  # only show promising combos
                report(f"MS VIX>{ms_t}, OS VIX<={os_t}", t6f)

# ═══════════════════════════════════════════════════════════════
#  BEST COMBINATION: stack winning levers
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 115}")
print(f"  COMBINED: Stack best levers together")
print(f"{'=' * 115}")
print(header)
print(sep)

# Combo 1: IB Low + zone 4.0 + VIX split @17
cfgC1 = base_long_config()
cfgC1.ms_use_ib_levels = True
cfgC1.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short", "MS_IBL": "long"}
cfgC1.ms_zone_pts = 4.0
tC1 = run_backtest(df.copy(), cfgC1)
tC1f = apply_vix_filter(tC1, vix)
report("IB Low + zone 4.0 + VIX@17", tC1f, show_setups=True, show_wf=True, df=df)

# Combo 2: IB Low + zone 4.0 + OS window 3 + VIX split @17
cfgC2 = base_long_config()
cfgC2.ms_use_ib_levels = True
cfgC2.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short", "MS_IBL": "long"}
cfgC2.ms_zone_pts = 4.0
cfgC2.os_entry_window = 3
tC2 = run_backtest(df.copy(), cfgC2)
tC2f = apply_vix_filter(tC2, vix)
report("IB Low + zone 4.0 + OS 3bar + VIX@17", tC2f, show_setups=True, show_wf=True, df=df)

# Combo 3: IB Low + Dev VAL + zone 4.0 + VIX split @15
cfgC3 = base_long_config()
cfgC3.ms_use_ib_levels = True
cfgC3.ms_use_dev_va = True
cfgC3.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short", "MS_IBL": "long", "MS_dVAL": "long"}
cfgC3.ms_zone_pts = 4.0
tC3 = run_backtest(df.copy(), cfgC3)
tC3f = apply_vix_filter(tC3, vix, ms_thresh=15, os_thresh=17)
report("IBL + dVAL + zone 4.0 + VIX MS>15/OS<=17", tC3f, show_setups=True, show_wf=True, df=df)

# Combo 4: IB Low + zone 5.0 + min_target 6.0 + VIX split @16
cfgC4 = base_long_config()
cfgC4.ms_use_ib_levels = True
cfgC4.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short", "MS_IBL": "long"}
cfgC4.ms_zone_pts = 5.0
cfgC4.ms_min_target_pts = 6.0
tC4 = run_backtest(df.copy(), cfgC4)
tC4f = apply_vix_filter(tC4, vix, ms_thresh=16, os_thresh=18)
report("IBL + zone 5.0 + tgt 6.0 + VIX 16/18", tC4f, show_setups=True, show_wf=True, df=df)

# Combo 5: Kitchen sink (all levers at moderate settings)
cfgC5 = base_long_config()
cfgC5.ms_use_ib_levels = True
cfgC5.ms_use_dev_va = True
cfgC5.ms_use_poc = True
cfgC5.ms_level_directions = {
    "MS_ONH": "both", "MS_ONL": "both", "MS_pVAH": "short",
    "MS_IBL": "long", "MS_IBH": "long",
    "MS_dVAL": "long", "MS_dPOC": "long",
}
cfgC5.ms_zone_pts = 4.0
cfgC5.ms_min_target_pts = 6.0
cfgC5.os_entry_window = 2
tC5 = run_backtest(df.copy(), cfgC5)
tC5f = apply_vix_filter(tC5, vix, ms_thresh=15, os_thresh=18)
report("KITCHEN SINK (all levels, moderate)", tC5f, show_setups=True, show_wf=True, df=df)

print()
