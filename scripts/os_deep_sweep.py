#!/usr/bin/env python3
"""Deep sweep for Overnight Sweep — short-only gap-up fades.

Best from initial test:
  G: Short-only, on_extreme stop — 253t, PF 1.194, +$9,998, WF 0.86
  H: Short, ON stop, gap 3-20 — 205t, PF 1.306, +$11,550, WF 0.56

This sweep tries:
  Stage 1: Gap size × stop buffer × entry window
  Stage 2: Target mode × min_target × min_rr  
  Stage 3: Best combo walk-forward validation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


def _base():
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
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
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = 2.0
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 4.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_require_ma = False
    cfg.max_os_trades = 2
    cfg.os_min_gap = 2.0
    cfg.os_max_gap = 40.0
    cfg.os_entry_window = 6
    return cfg


def pr(label, trades):
    if not trades:
        print(f"  {label:<60s}  NO TRADES")
        return None
    m = compute_metrics(trades)
    pnls = [t.pnl_dollar for t in trades]
    _, p = stats.ttest_1samp(pnls, 0) if len(trades) >= 5 else (0, 1.0)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"  {label:<60s}  {m.total_trades:>4d}t  WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+9,.0f}  DD ${m.max_drawdown:>7,.0f}  Sh {m.sharpe:>5.2f}  "
          f"p={p:.3f}{sig}")
    return m


df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
print(f"Loaded {len(df)} bars\n")

# ═══ Stage 1: Gap size × stop buffer × entry window ═══
print("=" * 150)
print("  STAGE 1: GAP × STOP BUFFER × ENTRY WINDOW")
print("=" * 150)

best_pf = 0
best_label = ""
best_params = {}

for min_gap, max_gap in [(2, 20), (2, 40), (3, 15), (3, 20), (3, 30), (5, 25), (5, 40)]:
    for buf in [1.0, 2.0, 3.0, 5.0]:
        for window in [1, 3, 6, 12]:
            cfg = _base()
            cfg.os_min_gap = float(min_gap)
            cfg.os_max_gap = float(max_gap)
            cfg.os_stop_buffer = buf
            cfg.os_entry_window = window
            trades = run_backtest(df.copy(), cfg)
            label = f"gap={min_gap}-{max_gap} buf={buf} win={window}"
            m = pr(label, trades)
            if m and m.profit_factor > best_pf and m.total_trades >= 50:
                best_pf = m.profit_factor
                best_label = label
                best_params = {"min_gap": min_gap, "max_gap": max_gap, "buf": buf, "window": window}

print(f"\n  BEST Stage 1: {best_label} (PF {best_pf:.3f})")

# ═══ Stage 2: Target mode × min_target × min_rr (using best from stage 1) ═══
print(f"\n{'=' * 150}")
print(f"  STAGE 2: TARGET MODE × MIN TARGET × MIN RR (using {best_label})")
print("=" * 150)

best2_pf = 0
best2_label = ""
best2_params = dict(best_params)

for target_mode in ["cascade", "prev_close", "prev_vah", "prev_poc"]:
    for min_tgt in [3.0, 4.0, 6.0, 8.0]:
        for rr in [0.3, 0.5, 0.8]:
            cfg = _base()
            cfg.os_min_gap = float(best_params["min_gap"])
            cfg.os_max_gap = float(best_params["max_gap"])
            cfg.os_stop_buffer = best_params["buf"]
            cfg.os_entry_window = best_params["window"]
            cfg.os_target_mode = target_mode
            cfg.os_min_target_pts = min_tgt
            cfg.os_min_rr = rr
            trades = run_backtest(df.copy(), cfg)
            label = f"tgt={target_mode} min={min_tgt} rr={rr}"
            m = pr(label, trades)
            if m and m.profit_factor > best2_pf and m.total_trades >= 50:
                best2_pf = m.profit_factor
                best2_label = label
                best2_params.update({"target_mode": target_mode, "min_tgt": min_tgt, "rr": rr})

print(f"\n  BEST Stage 2: {best2_label} (PF {best2_pf:.3f})")

# ═══ Stage 3: max_risk × max_trades ═══
print(f"\n{'=' * 150}")
print(f"  STAGE 3: MAX RISK × MAX TRADES")
print("=" * 150)

best3_pf = 0
best3_label = ""

for max_risk in [12.0, 15.0, 20.0, 25.0, 30.0]:
    for max_trades in [1, 2, 3]:
        cfg = _base()
        cfg.os_min_gap = float(best2_params["min_gap"])
        cfg.os_max_gap = float(best2_params["max_gap"])
        cfg.os_stop_buffer = best2_params["buf"]
        cfg.os_entry_window = best2_params["window"]
        cfg.os_target_mode = best2_params.get("target_mode", "cascade")
        cfg.os_min_target_pts = best2_params.get("min_tgt", 4.0)
        cfg.os_min_rr = best2_params.get("rr", 0.5)
        cfg.os_max_risk = max_risk
        cfg.max_os_trades = max_trades
        trades = run_backtest(df.copy(), cfg)
        label = f"max_risk={max_risk} max_trades={max_trades}"
        m = pr(label, trades)
        if m and m.profit_factor > best3_pf and m.total_trades >= 40:
            best3_pf = m.profit_factor
            best3_label = label
            best3_final_risk = max_risk
            best3_final_max = max_trades

print(f"\n  BEST Stage 3: {best3_label} (PF {best3_pf:.3f})")

# ═══ Stage 4: Walk-Forward on top combos ═══
print(f"\n{'=' * 150}")
print(f"  STAGE 4: WALK-FORWARD VALIDATION")
print("=" * 150)

split_idx = df.index.get_indexer(["2025-02-14"], method="nearest")[0]
df_y1 = df.iloc[:split_idx].copy()
df_y2 = df.iloc[split_idx:].copy()
print(f"  Y1: {df_y1.index[0].date()} to {df_y1.index[-1].date()}")
print(f"  Y2: {df_y2.index[0].date()} to {df_y2.index[-1].date()}\n")

# Build top configs to validate
top_configs = []

# Best from deep sweep
cfg = _base()
cfg.os_min_gap = float(best2_params["min_gap"])
cfg.os_max_gap = float(best2_params["max_gap"])
cfg.os_stop_buffer = best2_params["buf"]
cfg.os_entry_window = best2_params["window"]
cfg.os_target_mode = best2_params.get("target_mode", "cascade")
cfg.os_min_target_pts = best2_params.get("min_tgt", 4.0)
cfg.os_min_rr = best2_params.get("rr", 0.5)
try:
    cfg.os_max_risk = best3_final_risk
    cfg.max_os_trades = best3_final_max
except:
    pass
top_configs.append(("Best Deep Sweep", cfg))

# Conservative: gap 3-20, ON stop, buf=3, win=6
cfg = _base()
cfg.os_min_gap = 3.0
cfg.os_max_gap = 20.0
cfg.os_stop_buffer = 3.0
cfg.os_entry_window = 6
top_configs.append(("Conservative (gap3-20, buf3, win6)", cfg))

# Wider: gap 2-40, ON stop, buf=5, win=12
cfg = _base()
cfg.os_min_gap = 2.0
cfg.os_max_gap = 40.0
cfg.os_stop_buffer = 5.0
cfg.os_entry_window = 12
top_configs.append(("Wide (gap2-40, buf5, win12)", cfg))

# Tight: gap 5-25, ON stop, buf=2, win=3
cfg = _base()
cfg.os_min_gap = 5.0
cfg.os_max_gap = 25.0
cfg.os_stop_buffer = 2.0
cfg.os_entry_window = 3
top_configs.append(("Tight (gap5-25, buf2, win3)", cfg))

# With prev_poc target
cfg = _base()
cfg.os_min_gap = float(best2_params["min_gap"])
cfg.os_max_gap = float(best2_params["max_gap"])
cfg.os_stop_buffer = best2_params["buf"]
cfg.os_entry_window = best2_params["window"]
cfg.os_target_mode = "prev_poc"
cfg.os_min_target_pts = 6.0
top_configs.append(("Best + prev_poc target", cfg))

for label, cfg in top_configs:
    t1 = run_backtest(df_y1.copy(), cfg)
    t2 = run_backtest(df_y2.copy(), cfg)
    t_all = run_backtest(df.copy(), cfg)
    if t1 and t2 and t_all:
        m1 = compute_metrics(t1)
        m2 = compute_metrics(t2)
        m_all = compute_metrics(t_all)
        _, p_all = stats.ttest_1samp([t.pnl_dollar for t in t_all], 0) if len(t_all) >= 5 else (0, 1.0)
        _, p2 = stats.ttest_1samp([t.pnl_dollar for t in t2], 0) if len(t2) >= 5 else (0, 1.0)
        ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
        verdict = "PASS" if ratio > 0.7 and m2.profit_factor > 1.0 else "MARGINAL" if ratio > 0.5 else "FAIL"
        print(f"  {label:<45s}")
        print(f"    FULL: {m_all.total_trades:>3d}t PF={m_all.profit_factor:.3f} ${m_all.net_pnl:>+9,.0f} DD=${m_all.max_drawdown:>7,.0f} p={p_all:.4f}")
        print(f"    Y1:   {m1.total_trades:>3d}t PF={m1.profit_factor:.3f} ${m1.net_pnl:>+9,.0f}")
        print(f"    Y2:   {m2.total_trades:>3d}t PF={m2.profit_factor:.3f} ${m2.net_pnl:>+9,.0f} p={p2:.4f}")
        print(f"    WF ratio={ratio:.2f}  {verdict}\n")
    else:
        no = "Y1" if not t1 else "Y2" if not t2 else "FULL"
        print(f"  {label:<45s}  NO TRADES ({no})\n")

print("=" * 150)
print("  DONE")
print("=" * 150)
