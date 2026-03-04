#!/usr/bin/env python3
"""MS Filtered — drill down into which level combinations have edge.

From the full MS scan, clear patterns emerged:
  WINNERS: ONH shorts, ONL longs, pVAH shorts, POC longs (prev + dev)
  LOSERS: pPOC shorts, dVAL longs, pVAL longs, dPOC shorts

This script tests filtered combinations to find the sweet spot.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


def _ms_base():
    """Base MS config — all levels OFF, SMA 8/24."""
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
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 5.0
    cfg.ms_min_target_pts = 4.0
    cfg.ms_min_rr = 0.5
    cfg.ms_max_risk = 15.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    cfg.ms_use_prev_va = False
    cfg.ms_use_on_levels = False
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False
    return cfg


def print_result(label, trades, df_len):
    if not trades:
        print(f"  {label:<50s}  NO TRADES")
        return
    m = compute_metrics(trades)
    pnls = [t.pnl_dollar for t in trades]
    _, p_val = stats.ttest_1samp(pnls, 0) if len(trades) >= 5 else (0, 1.0)
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

    # Direction breakdown
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    l_pnl = sum(t.pnl_dollar for t in longs)
    s_pnl = sum(t.pnl_dollar for t in shorts)

    print(f"  {label:<50s}  {m.total_trades:>5d}t  WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+9,.0f}  DD ${m.max_drawdown:>7,.0f}  Sh {m.sharpe:>5.2f}  "
          f"p={p_val:.3f}{sig}  T/D {m.trades_per_day:.1f}  "
          f"L${l_pnl:>+7,.0f} S${s_pnl:>+7,.0f}")


df = load_tos_csv('data/es_5m_databento_2yr.csv', instrument='ES')
print(f"Loaded {len(df)} bars\n")

# Split for walk-forward
split_idx = df.index.get_indexer(["2025-02-14"], method="nearest")[0]
df_y1 = df.iloc[:split_idx].copy()
df_y2 = df.iloc[split_idx:].copy()

print("="*150)
print("  LEVEL COMBINATION SWEEP — FULL 2-YEAR")
print("="*150)

configs = []

# 1. ON levels only (strongest edge)
cfg = _ms_base()
cfg.ms_use_on_levels = True
configs.append(("ON only (ONH+ONL)", cfg))

# 2. ON + prev VA (VA Fade)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
configs.append(("ON + prev VA", cfg))

# 3. ON + POC (longs+shorts at POC)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_poc = True
configs.append(("ON + POC", cfg))

# 4. ON + dev VA
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_dev_va = True
configs.append(("ON + dev VA", cfg))

# 5. ON + IB
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_ib_levels = True
configs.append(("ON + IB", cfg))

# 6. ON + prev VA + POC (no dev, no IB)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
configs.append(("ON + pVA + POC", cfg))

# 7. ON + prev VA + dev VA (no POC, no IB)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_dev_va = True
configs.append(("ON + pVA + dVA", cfg))

# 8. All except IB (IB was weakest)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_dev_va = True
cfg.ms_use_poc = True
configs.append(("All except IB", cfg))

# 9. Short-only at ALL levels
cfg = _ms_base()
cfg.direction_filter = "short"
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_dev_va = True
cfg.ms_use_poc = True
cfg.ms_use_ib_levels = True
configs.append(("Short-only (all levels)", cfg))

# 10. Long-only at ALL levels
cfg = _ms_base()
cfg.direction_filter = "long"
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_dev_va = True
cfg.ms_use_poc = True
cfg.ms_use_ib_levels = True
configs.append(("Long-only (all levels)", cfg))

# === Parameter variations on "ON + pVA + POC" (best combo candidate) ===

# 11. Wider stop (8 pts)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.ms_stop_buffer = 8.0
configs.append(("ON+pVA+POC stop=8", cfg))

# 12. Wider stop (10 pts)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.ms_stop_buffer = 10.0
configs.append(("ON+pVA+POC stop=10", cfg))

# 13. Tighter zone (2 pts)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.ms_zone_pts = 2.0
configs.append(("ON+pVA+POC zone=2", cfg))

# 14. Wider zone (5 pts)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.ms_zone_pts = 5.0
configs.append(("ON+pVA+POC zone=5", cfg))

# 15. Higher min RR (0.8)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.ms_min_rr = 0.8
configs.append(("ON+pVA+POC rr=0.8", cfg))

# 16. Higher min target (6 pts)
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.ms_min_target_pts = 6.0
configs.append(("ON+pVA+POC minTgt=6", cfg))

# 17. Entry lag = 1 bar
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.ms_ma_confirm_bars = 1
configs.append(("ON+pVA+POC lag=1", cfg))

# 18. Max 4 trades/day
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.max_ms_trades = 4
configs.append(("ON+pVA+POC max=4", cfg))

# 19. TEMA instead of SMA
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.ms_ma_type = "tema"
configs.append(("ON+pVA+POC TEMA", cfg))

# 20. Friday skip + noon blackout
cfg = _ms_base()
cfg.ms_use_on_levels = True
cfg.ms_use_prev_va = True
cfg.ms_use_poc = True
cfg.skip_friday = True
cfg.blackout_start = 1200
cfg.blackout_end = 1300
configs.append(("ON+pVA+POC FriSkip+Noon", cfg))

for label, cfg in configs:
    trades = run_backtest(df.copy(), cfg)
    print_result(label, trades, len(df))

# Walk-forward on best candidates
print(f"\n{'='*150}")
print("  WALK-FORWARD — TOP CANDIDATES")
print(f"{'='*150}")
print(f"  Year 1: {df_y1.index[0].date()} to {df_y1.index[-1].date()}")
print(f"  Year 2: {df_y2.index[0].date()} to {df_y2.index[-1].date()}\n")

# Run top 5 candidates on both years
top_labels = ["ON + pVA + POC", "ON + prev VA", "ON only (ONH+ONL)",
              "ON+pVA+POC stop=8", "ON+pVA+POC zone=5"]
top_configs = [(l, c) for l, c in configs if l in top_labels]

for label, cfg in top_configs:
    t1 = run_backtest(df_y1.copy(), cfg)
    t2 = run_backtest(df_y2.copy(), cfg)
    if t1 and t2:
        m1 = compute_metrics(t1)
        m2 = compute_metrics(t2)
        _, p2 = stats.ttest_1samp([t.pnl_dollar for t in t2], 0) if len(t2) >= 5 else (0, 1.0)
        ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
        verdict = "PASS" if ratio > 0.7 else "MARGINAL" if ratio > 0.5 else "FAIL"
        print(f"  {label:<40s}  IS: {m1.total_trades}t PF={m1.profit_factor:.3f} ${m1.net_pnl:>+8,.0f}  "
              f"OOS: {m2.total_trades}t PF={m2.profit_factor:.3f} ${m2.net_pnl:>+8,.0f}  "
              f"ratio={ratio:.2f} {verdict}  OOS p={p2:.3f}")
