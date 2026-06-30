#!/usr/bin/env python3
"""Test: Nexus Long-Only — Does Grok's thesis hold up?

Grok built a standalone long-only Nexus (MS+OS) for ES.
We already know ES longs lost money on IB Breakouts (killed Feb 15).
But MS+OS uses different entry logic — worth testing separately.

Three configs compared:
  A) Long-only using existing "both" config (isolate longs from deployed version)
  B) Long-optimized config (ONL + pVAL support bounces, gap-down fades)
  C) Full both-direction baseline (for comparison)

Usage:
    cd ~/projects/backtesting/es
    python3 scripts/nexus_long_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict
from scipy import stats

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


def base_config():
    """Shared config — all setups OFF except MS and OS."""
    cfg = StrategyConfig()
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

    # MS config (same as official)
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

    # OS config (same as official)
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


def config_a():
    """A) Long-only — same levels as deployed, just filter direction."""
    cfg = base_config()
    cfg.direction_filter = "long"
    cfg.ms_level_directions = {
        "MS_ONH": "both",
        "MS_ONL": "both",
        "MS_pVAH": "short",  # won't fire since direction_filter="long"
    }
    return cfg


def config_b():
    """B) Long-optimized — support bounces + gap-down fades only."""
    cfg = base_config()
    cfg.direction_filter = "long"
    cfg.ms_level_directions = {
        "MS_ONL": "long",    # bounce at overnight low
        "MS_pVAL": "long",   # bounce at prev value area low
        "MS_ONH": "long",    # breakout through overnight high
    }
    # Enable pVAL (it's in prev_va)
    cfg.ms_use_prev_va = True
    return cfg


def config_c():
    """C) Baseline — the deployed both-direction version (for comparison)."""
    cfg = base_config()
    cfg.direction_filter = "both"
    cfg.ms_level_directions = {
        "MS_ONH": "both",
        "MS_ONL": "both",
        "MS_pVAH": "short",
    }
    return cfg


def run_and_report(name, cfg, df):
    """Run backtest and print results."""
    trades = run_backtest(df.copy(), cfg)
    if not trades:
        print(f"\n  {name}: 0 TRADES — no signal fired.\n")
        return trades

    m = compute_metrics(trades, cfg.initial_capital)
    pnls = [t.pnl_dollar for t in trades]
    _, p_val = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)

    # Permutation test
    n_perm = 5000
    rng = np.random.default_rng(42)
    obs_mean = np.mean(pnls)
    perm_count = sum(1 for _ in range(n_perm)
                     if np.mean(rng.choice([-1, 1], size=len(pnls)) * np.abs(pnls)) >= obs_mean)
    p_perm = perm_count / n_perm

    # Bootstrap
    n_boot = 10000
    boot_means = [np.mean(rng.choice(pnls, size=len(pnls), replace=True)) for _ in range(n_boot)]
    p_profit = np.mean([b > 0 for b in boot_means]) * 100

    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]

    # Walk-forward split
    split_idx = df.index.get_indexer(["2025-02-14"], method="nearest")[0]
    split_date = df.index[split_idx]
    t1 = [t for t in trades if t.entry_time < split_date]
    t2 = [t for t in trades if t.entry_time >= split_date]

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Trades:       {m.total_trades:>6d}     (L={len(longs)}, S={len(shorts)})")
    print(f"  Win Rate:     {m.win_rate:>6.1f}%")
    print(f"  Profit Factor:{m.profit_factor:>7.3f}")
    print(f"  Net P&L:      ${m.net_pnl:>+10,.0f}")
    print(f"  Avg Trade:    ${m.avg_trade:>+8,.0f}")
    print(f"  Max DD:       ${m.max_drawdown:>8,.0f}  ({m.max_drawdown_pct:.1f}%)")
    print(f"  Sharpe:       {m.sharpe:>7.2f}")
    print(f"  t-test p:     {p_val:>10.6f}  {'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else 'NS'}")
    print(f"  Permutation p:{p_perm:>10.6f}  {'***' if p_perm < 0.01 else '**' if p_perm < 0.05 else '*' if p_perm < 0.10 else 'NS'}")
    print(f"  Bootstrap P(profit): {p_profit:.1f}%")

    # Per-setup breakdown
    breakdown = per_setup_breakdown(trades, cfg.initial_capital)
    print(f"\n  {'Setup':<12s}  {'#':>4s}  {'WR':>6s}  {'PF':>7s}  {'P&L':>10s}")
    for setup, sm in sorted(breakdown.items(), key=lambda x: -x[1].net_pnl):
        print(f"  {setup:<12s}  {sm.total_trades:>4d}  {sm.win_rate:>5.1f}%  {sm.profit_factor:>7.3f}  ${sm.net_pnl:>+9,.0f}")

    # Walk-forward
    if t1 and t2 and len(t1) >= 5 and len(t2) >= 5:
        m1 = compute_metrics(t1, cfg.initial_capital)
        m2 = compute_metrics(t2, cfg.initial_capital)
        _, p2 = stats.ttest_1samp([t.pnl_dollar for t in t2], 0)
        ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
        verdict = "PASS" if ratio > 0.7 and m2.profit_factor > 1.0 else "MARGINAL" if ratio > 0.5 else "FAIL"
        print(f"\n  Walk-Forward (split {split_date.date()}):")
        print(f"    Y1: {m1.total_trades:>3d}t  PF={m1.profit_factor:.3f}  ${m1.net_pnl:>+8,.0f}")
        print(f"    Y2: {m2.total_trades:>3d}t  PF={m2.profit_factor:.3f}  ${m2.net_pnl:>+8,.0f}  p={p2:.4f}")
        print(f"    WF ratio: {ratio:.2f} → {verdict}")
    elif t1 or t2:
        print(f"\n  Walk-forward: insufficient trades in one period (Y1={len(t1)}, Y2={len(t2)})")

    # Monthly
    monthly = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in trades:
        key = t.exit_time.strftime("%Y-%m")
        monthly[key]["count"] += 1
        monthly[key]["pnl"] += t.pnl_dollar
        if t.pnl_dollar > 0:
            monthly[key]["wins"] += 1
    win_months = sum(1 for d in monthly.values() if d["pnl"] > 0)
    print(f"\n  Monthly: {win_months}/{len(monthly)} winning ({win_months/len(monthly)*100:.0f}%)")

    return trades


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

print("Loading ES 5m data (2yr)...")
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
print(f"Loaded {len(df):,} bars  ({df.index[0].date()} to {df.index[-1].date()})\n")

# Run all three configs
print("=" * 70)
print("  NEXUS LONG TEST — Does Grok's thesis hold?")
print("  Testing long-only MS+OS on 2yr ES 5m data")
print("=" * 70)

trades_c = run_and_report("C) BASELINE — Both Directions (deployed)", config_c(), df)
trades_a = run_and_report("A) LONG-ONLY — Same Levels as Deployed", config_a(), df)
trades_b = run_and_report("B) LONG-OPTIMIZED — Support Bounces + Gap Fades", config_b(), df)

# Summary comparison
print(f"\n{'=' * 70}")
print(f"  VERDICT")
print(f"{'=' * 70}")
for label, trades in [("Baseline (both)", trades_c),
                       ("Long-only (A)", trades_a),
                       ("Long-optimized (B)", trades_b)]:
    if not trades:
        print(f"  {label:<25s}  NO TRADES")
        continue
    m = compute_metrics(trades, 100_000)
    pnls = [t.pnl_dollar for t in trades]
    _, p = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "NS"
    print(f"  {label:<25s}  {m.total_trades:>4d}t  PF {m.profit_factor:.3f}  ${m.net_pnl:>+9,.0f}  p={p:.4f} {sig}")

print()
