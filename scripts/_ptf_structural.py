#!/usr/bin/env python3
"""PTF structural optimization — wider stops, direction, relaxed entry, combined params."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
from scipy import stats

df = load_tos_csv('data/es_5m_databento_2yr.csv', instrument='ES')
print(f"Loaded {len(df)} bars")

def run_ptf(label, **overrides):
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
    cfg.use_ptf = True
    cfg.ptf_target = "prev_poc"
    cfg.ptf_stop_buffer = 5.0
    cfg.ptf_min_otf = 4
    cfg.ptf_require_reversal = True
    cfg.max_ptf_trades = 2
    cfg.ptf_min_target_pts = 8.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    trades = run_backtest(df.copy(), cfg)
    if not trades:
        print(f"  {label:<55s}  NO TRADES")
        return None
    m = compute_metrics(trades)
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    l_pnl = sum(t.pnl_dollar for t in longs)
    s_pnl = sum(t.pnl_dollar for t in shorts)
    pnls = [t.pnl_dollar for t in trades]
    _, p = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"  {label:<55s}  {m.total_trades:>4d}  WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+8,.0f}  DD ${m.max_drawdown:>6,.0f}  Sh {m.sharpe:>5.2f}  "
          f"L${l_pnl:>+7,.0f} S${s_pnl:>+7,.0f}  p={p:.3f}{sig}")
    return m

print("=" * 140)
print("SECTION 1: BASELINE + KEY VARIATIONS")
print("=" * 140)
run_ptf("Baseline (otf>=4, rev, stop=5, min_tgt=8)")
run_ptf("otf>=5 no-rev (best from diag)", ptf_min_otf=5, ptf_require_reversal=False, max_ptf_trades=4)
run_ptf("otf>=3 no-rev stop=7", ptf_min_otf=3, ptf_require_reversal=False, ptf_stop_buffer=7.0, max_ptf_trades=4)

print("\n" + "=" * 140)
print("SECTION 2: AGGRESSIVE — BIG STOPS, LOW MIN TARGET (increasing risk)")
print("=" * 140)
for otf in [3, 4, 5]:
    for stop in [7, 10, 15, 20]:
        for min_tgt in [4, 6, 8]:
            run_ptf(f"otf>={otf} stop={stop} min_tgt={min_tgt} norev",
                    ptf_min_otf=otf, ptf_stop_buffer=float(stop),
                    ptf_min_target_pts=float(min_tgt),
                    ptf_require_reversal=False, max_ptf_trades=4)

print("\n" + "=" * 140)
print("SECTION 3: DIRECTION FILTER (best aggressive params)")
print("=" * 140)
# Use otf>=5 stop=7 min_tgt=4 norev as base
run_ptf("Both dirs", ptf_min_otf=5, ptf_stop_buffer=7.0, ptf_min_target_pts=4.0,
        ptf_require_reversal=False, max_ptf_trades=4)
run_ptf("Short only", ptf_min_otf=5, ptf_stop_buffer=7.0, ptf_min_target_pts=4.0,
        ptf_require_reversal=False, max_ptf_trades=4, direction_filter="short")
run_ptf("Long only", ptf_min_otf=5, ptf_stop_buffer=7.0, ptf_min_target_pts=4.0,
        ptf_require_reversal=False, max_ptf_trades=4, direction_filter="long")

# Also with stop=10 (bigger risk)
run_ptf("Both stop=10", ptf_min_otf=5, ptf_stop_buffer=10.0, ptf_min_target_pts=4.0,
        ptf_require_reversal=False, max_ptf_trades=4)
run_ptf("Short stop=10", ptf_min_otf=5, ptf_stop_buffer=10.0, ptf_min_target_pts=4.0,
        ptf_require_reversal=False, max_ptf_trades=4, direction_filter="short")
run_ptf("Long stop=10", ptf_min_otf=5, ptf_stop_buffer=10.0, ptf_min_target_pts=4.0,
        ptf_require_reversal=False, max_ptf_trades=4, direction_filter="long")

print("\n" + "=" * 140)
print("SECTION 4: MAX TRADES PER DAY")
print("=" * 140)
for max_t in [1, 2, 3, 4, 6]:
    run_ptf(f"max_trades={max_t} (otf>=5 stop=7 norev min4)",
            ptf_min_otf=5, ptf_stop_buffer=7.0, ptf_min_target_pts=4.0,
            ptf_require_reversal=False, max_ptf_trades=max_t)

print("\n" + "=" * 140)
print("SECTION 5: TARGET TYPE")
print("=" * 140)
for target in ["prev_poc", "single_print_mid"]:
    for stop in [7, 10, 15]:
        run_ptf(f"target={target} stop={stop} (otf>=5 norev min4)",
                ptf_min_otf=5, ptf_stop_buffer=float(stop), ptf_min_target_pts=4.0,
                ptf_require_reversal=False, max_ptf_trades=4, ptf_target=target)

print("\n" + "=" * 140)
print("SECTION 6: WITH REVERSAL REQUIREMENT (using wider stops)")
print("=" * 140)
for otf in [3, 4, 5]:
    for stop in [10, 15, 20]:
        run_ptf(f"otf>={otf} stop={stop} REV min4",
                ptf_min_otf=otf, ptf_stop_buffer=float(stop),
                ptf_min_target_pts=4.0,
                ptf_require_reversal=True, max_ptf_trades=4)
