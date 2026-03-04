#!/usr/bin/env python3
"""Overnight Sweep / Gap Fade — comprehensive test.

Concept (Dalton/auction theory):
  - Overnight session trades above prev close → weak-handed longs = overhead supply
  - Gap up at open → short first candle, stop at opening print
  - Target cascades: prev_close → prev_vah → prev_poc → prev_val

Tests:
  1. Core variants (target modes, stop modes)
  2. Direction analysis (shorts vs longs)
  3. Gap size filters
  4. Entry window (how many bars from open)
  5. ON sweep requirement (with/without)
  6. MA filter (with/without)
  7. Walk-forward validation

Usage:
    python3 scripts/test_overnight_sweep.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


def _os_base():
    """Base config with ONLY overnight sweep enabled."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_ms = False
    cfg.use_fa = False
    cfg.use_os = True
    # Default params
    cfg.os_min_gap = 2.0
    cfg.os_max_gap = 40.0
    cfg.os_stop_mode = "opening_print"
    cfg.os_stop_buffer = 2.0
    cfg.os_max_risk = 20.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 4.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_entry_window = 6  # first 30 min (6 x 5-min bars)
    cfg.os_require_ma = False
    cfg.max_os_trades = 2
    return cfg


def print_result(label, trades, df_len):
    if not trades:
        print(f"  {label:<55s}  NO TRADES")
        return

    m = compute_metrics(trades)
    pnls = [t.pnl_dollar for t in trades]
    _, p_val = stats.ttest_1samp(pnls, 0) if len(trades) >= 5 else (0, 1.0)
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    l_pnl = sum(t.pnl_dollar for t in longs)
    s_pnl = sum(t.pnl_dollar for t in shorts)

    print(f"  {label:<55s}  {m.total_trades:>4d}t  WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+9,.0f}  DD ${m.max_drawdown:>7,.0f}  Sh {m.sharpe:>5.2f}  "
          f"p={p_val:.3f}{sig}  T/D {m.trades_per_day:.2f}  "
          f"L${l_pnl:>+7,.0f}({len(longs)}) S${s_pnl:>+7,.0f}({len(shorts)})")


def print_detail(label, trades, df):
    """Print detailed breakdown including exits and walk-forward."""
    if not trades:
        return

    m = compute_metrics(trades)

    # Exit reasons
    reasons = {}
    for t in trades:
        r = t.exit_reason
        reasons.setdefault(r, {"count": 0, "pnl": 0})
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t.pnl_dollar
    print(f"\n    {label} — Exits:")
    for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        wr = len([t for t in trades if t.exit_reason == r and t.pnl_dollar > 0]) / d["count"] * 100
        print(f"      {r:<12s}  {d['count']:>4d}  ${d['pnl']:>+9,.0f}  WR {wr:.0f}%")

    # Walk-forward
    split_idx = df.index.get_indexer(["2025-02-14"], method="nearest")[0]
    y1 = [t for t in trades if t.entry_time < df.index[split_idx]]
    y2 = [t for t in trades if t.entry_time >= df.index[split_idx]]
    if y1 and y2:
        m1 = compute_metrics(y1)
        m2 = compute_metrics(y2)
        _, p2 = stats.ttest_1samp([t.pnl_dollar for t in y2], 0) if len(y2) >= 5 else (0, 1.0)
        ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
        verdict = "PASS" if ratio > 0.7 else "MARGINAL" if ratio > 0.5 else "FAIL"
        print(f"    Walk-Forward:")
        print(f"      Y1: {m1.total_trades:>3d}t PF={m1.profit_factor:.3f} ${m1.net_pnl:>+8,.0f}")
        print(f"      Y2: {m2.total_trades:>3d}t PF={m2.profit_factor:.3f} ${m2.net_pnl:>+8,.0f}  p={p2:.4f}")
        print(f"      Ratio: {ratio:.2f} {verdict}")


# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
print(f"Loaded {len(df)} bars\n")

# ═══════════════════════════════════════════════════════════════
# Stage 1: Core Variants (target modes × stop modes)
# ═══════════════════════════════════════════════════════════════
print("=" * 150)
print("  STAGE 1: CORE VARIANTS")
print("=" * 150)

configs = []

# 1a. Default: cascade target, opening_print stop
cfg = _os_base()
configs.append(("1a. Default (cascade, opening_print)", cfg))

# 1b. Target: prev_close only
cfg = _os_base()
cfg.os_target_mode = "prev_close"
configs.append(("1b. Target=prev_close", cfg))

# 1c. Target: prev_vah
cfg = _os_base()
cfg.os_target_mode = "prev_vah"
configs.append(("1c. Target=prev_vah", cfg))

# 1d. Target: prev_poc
cfg = _os_base()
cfg.os_target_mode = "prev_poc"
configs.append(("1d. Target=prev_poc", cfg))

# 1e. Stop: ON extreme (wider stop)
cfg = _os_base()
cfg.os_stop_mode = "on_extreme"
configs.append(("1e. Stop=on_extreme", cfg))

# 1f. Stop: fixed 8pt
cfg = _os_base()
cfg.os_stop_mode = "fixed"
cfg.os_fixed_stop = 8.0
configs.append(("1f. Stop=fixed 8pt", cfg))

# 1g. Stop: fixed 12pt
cfg = _os_base()
cfg.os_stop_mode = "fixed"
cfg.os_fixed_stop = 12.0
configs.append(("1g. Stop=fixed 12pt", cfg))

for label, cfg in configs:
    trades = run_backtest(df.copy(), cfg)
    print_result(label, trades, len(df))

# Detail on best
best_cfg = configs[0][1]  # default for now
best_trades = run_backtest(df.copy(), best_cfg)
print_detail("Default", best_trades, df)

# ═══════════════════════════════════════════════════════════════
# Stage 2: Direction Analysis
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print("  STAGE 2: DIRECTION ANALYSIS")
print("=" * 150)

# Short-only (gap up → short)
cfg = _os_base()
cfg.direction_filter = "short"
trades = run_backtest(df.copy(), cfg)
print_result("Short-only (gap up fades)", trades, len(df))
print_detail("Short-only", trades, df)

# Long-only (gap down → long)
cfg = _os_base()
cfg.direction_filter = "long"
trades = run_backtest(df.copy(), cfg)
print_result("Long-only (gap down fades)", trades, len(df))
print_detail("Long-only", trades, df)

# ═══════════════════════════════════════════════════════════════
# Stage 3: Gap Size Filters
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print("  STAGE 3: GAP SIZE FILTERS")
print("=" * 150)

for min_gap, max_gap in [(2, 10), (2, 20), (2, 40), (3, 15), (5, 25), (5, 40), (8, 40), (10, 40)]:
    cfg = _os_base()
    cfg.os_min_gap = float(min_gap)
    cfg.os_max_gap = float(max_gap)
    trades = run_backtest(df.copy(), cfg)
    print_result(f"Gap {min_gap}-{max_gap}pt", trades, len(df))

# ═══════════════════════════════════════════════════════════════
# Stage 4: Entry Window
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print("  STAGE 4: ENTRY WINDOW (bars from RTH open)")
print("=" * 150)

for window in [1, 2, 3, 4, 6, 12, 24]:
    cfg = _os_base()
    cfg.os_entry_window = window
    trades = run_backtest(df.copy(), cfg)
    print_result(f"Window={window} bars ({window*5}min)", trades, len(df))

# ═══════════════════════════════════════════════════════════════
# Stage 5: ON Sweep Requirement
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print("  STAGE 5: ON SWEEP REQUIREMENT")
print("=" * 150)

cfg = _os_base()
cfg.os_require_on_sweep = True
trades = run_backtest(df.copy(), cfg)
print_result("Require ON sweep (default)", trades, len(df))

cfg = _os_base()
cfg.os_require_on_sweep = False
trades = run_backtest(df.copy(), cfg)
print_result("No ON sweep required", trades, len(df))

# ═══════════════════════════════════════════════════════════════
# Stage 6: MA Filter
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print("  STAGE 6: MA FILTER")
print("=" * 150)

cfg = _os_base()
cfg.os_require_ma = False
trades = run_backtest(df.copy(), cfg)
print_result("No MA filter (default)", trades, len(df))

cfg = _os_base()
cfg.os_require_ma = True
cfg.os_ma_type = "sma"
trades = run_backtest(df.copy(), cfg)
print_result("SMA 8/24 filter", trades, len(df))

cfg = _os_base()
cfg.os_require_ma = True
cfg.os_ma_type = "tema"
trades = run_backtest(df.copy(), cfg)
print_result("TEMA filter", trades, len(df))

# ═══════════════════════════════════════════════════════════════
# Stage 7: Risk/Reward Tuning
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print("  STAGE 7: STOP BUFFER & RISK PARAMS")
print("=" * 150)

for buf in [1.0, 2.0, 3.0, 5.0]:
    cfg = _os_base()
    cfg.os_stop_buffer = buf
    trades = run_backtest(df.copy(), cfg)
    print_result(f"Stop buffer={buf}pt", trades, len(df))

for rr in [0.3, 0.5, 0.8, 1.0]:
    cfg = _os_base()
    cfg.os_min_rr = rr
    trades = run_backtest(df.copy(), cfg)
    print_result(f"Min R:R={rr}", trades, len(df))

for tgt in [3.0, 4.0, 6.0, 8.0]:
    cfg = _os_base()
    cfg.os_min_target_pts = tgt
    trades = run_backtest(df.copy(), cfg)
    print_result(f"Min target={tgt}pt", trades, len(df))

# ═══════════════════════════════════════════════════════════════
# Stage 8: Best Combo + Walk-Forward
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 150}")
print("  STAGE 8: PROMISING COMBINATIONS")
print("=" * 150)

combos = []

# Combo A: Short-only, cascade, opening_print, default params
cfg = _os_base()
cfg.direction_filter = "short"
combos.append(("A: Short-only cascade", cfg))

# Combo B: Short-only, prev_close target
cfg = _os_base()
cfg.direction_filter = "short"
cfg.os_target_mode = "prev_close"
combos.append(("B: Short-only prev_close", cfg))

# Combo C: Short-only, wider entry window (12 bars = 1hr)
cfg = _os_base()
cfg.direction_filter = "short"
cfg.os_entry_window = 12
combos.append(("C: Short-only window=12", cfg))

# Combo D: Short-only, no ON sweep requirement
cfg = _os_base()
cfg.direction_filter = "short"
cfg.os_require_on_sweep = False
combos.append(("D: Short-only no-sweep-req", cfg))

# Combo E: Both, tighter gap (5-25), higher min RR
cfg = _os_base()
cfg.os_min_gap = 5.0
cfg.os_max_gap = 25.0
cfg.os_min_rr = 0.8
combos.append(("E: Both gap=5-25 rr=0.8", cfg))

# Combo F: Short-only, SMA filter, opening_print
cfg = _os_base()
cfg.direction_filter = "short"
cfg.os_require_ma = True
cfg.os_ma_type = "sma"
combos.append(("F: Short-only + SMA", cfg))

# Combo G: Short-only, on_extreme stop, cascade
cfg = _os_base()
cfg.direction_filter = "short"
cfg.os_stop_mode = "on_extreme"
combos.append(("G: Short-only on_extreme stop", cfg))

# Combo H: Short-only, ON stop, wider window, gap 3-20
cfg = _os_base()
cfg.direction_filter = "short"
cfg.os_stop_mode = "on_extreme"
cfg.os_entry_window = 12
cfg.os_min_gap = 3.0
cfg.os_max_gap = 20.0
combos.append(("H: Short ON-stop wide-window gap3-20", cfg))

for label, cfg in combos:
    trades = run_backtest(df.copy(), cfg)
    print_result(label, trades, len(df))
    if trades and len(trades) >= 10:
        print_detail(label, trades, df)

print(f"\n{'=' * 150}")
print("  DONE")
print(f"{'=' * 150}")
