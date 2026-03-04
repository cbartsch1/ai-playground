#!/usr/bin/env python3
"""VAR structural optimization — direction, time, VA width, VWAP filter."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
from scipy import stats

df = load_tos_csv('data/es_5m_databento_2yr.csv', instrument='ES')
print(f"Loaded {len(df)} bars")

def run_var(label, **overrides):
    cfg = StrategyConfig()
    cfg.direction_filter = "both"
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = True
    cfg.use_ptf = False
    cfg.var_zone_pts = 3.0
    cfg.var_target_pts = 0.0
    cfg.var_stop_buffer = 4.0
    cfg.var_min_ib_periods = 4
    cfg.var_require_rotation = True
    cfg.var_max_otf = 2
    cfg.max_var_trades = 8
    cfg.var_min_rr = 0.8
    for k, v in overrides.items():
        setattr(cfg, k, v)
    trades = run_backtest(df.copy(), cfg)
    if not trades:
        print(f"  {label:<50s}  NO TRADES")
        return None
    m = compute_metrics(trades)
    # Direction breakdown
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    l_pnl = sum(t.pnl_dollar for t in longs)
    s_pnl = sum(t.pnl_dollar for t in shorts)
    pnls = [t.pnl_dollar for t in trades]
    _, p = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"  {label:<50s}  {m.total_trades:>5d}  WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+9,.0f}  DD ${m.max_drawdown:>7,.0f}  Sh {m.sharpe:>5.2f}  "
          f"L${l_pnl:>+8,.0f} S${s_pnl:>+8,.0f}  p={p:.3f}{sig}")
    return m

print("=" * 140)
print("SECTION 1: DIRECTION FILTER")
print("=" * 140)
run_var("Baseline (both)")
run_var("Long only", direction_filter="long")
run_var("Short only", direction_filter="short")

print("\n" + "=" * 140)
print("SECTION 2: LONG-ONLY WITH RISK VARIATIONS (wider stops, bigger targets)")
print("=" * 140)
for stop in [4, 5, 7, 10, 15]:
    for rr in [0.5, 0.8, 1.0]:
        run_var(f"Long stop={stop} rr={rr}", direction_filter="long",
                var_stop_buffer=float(stop), var_min_rr=rr)

print("\n" + "=" * 140)
print("SECTION 3: LONG-ONLY + FIXED TARGETS (vs dynamic POC)")
print("=" * 140)
for tgt in [0, 4, 5, 8, 10]:
    for stop in [5, 7, 10]:
        run_var(f"Long tgt={tgt} stop={stop}", direction_filter="long",
                var_target_pts=float(tgt), var_stop_buffer=float(stop), var_min_rr=0.5)

print("\n" + "=" * 140)
print("SECTION 4: TIME FILTERS (long only, best params from above)")
print("=" * 140)
# Skip 11:00 blackout
run_var("Long base (no time filter)", direction_filter="long",
        var_stop_buffer=7.0, var_min_rr=0.5)
run_var("Long + skip 11am", direction_filter="long",
        var_stop_buffer=7.0, var_min_rr=0.5,
        blackout_start=1100, blackout_end=1200)
run_var("Long + skip Fri", direction_filter="long",
        var_stop_buffer=7.0, var_min_rr=0.5,
        skip_friday=True)
run_var("Long + skip 11am + skip Fri", direction_filter="long",
        var_stop_buffer=7.0, var_min_rr=0.5,
        blackout_start=1100, blackout_end=1200, skip_friday=True)
# Afternoon only (2pm-3:30pm — best hours from diagnostic)
run_var("Long + afternoon trade start 1400", direction_filter="long",
        var_stop_buffer=7.0, var_min_rr=0.5,
        trade_start=1400)
run_var("Long + trade 1300-1530", direction_filter="long",
        var_stop_buffer=7.0, var_min_rr=0.5,
        trade_start=1300, trade_end=1530)

print("\n" + "=" * 140)
print("SECTION 5: PERIOD REQUIREMENTS (long only, more developed VA)")
print("=" * 140)
for periods in [3, 4, 5, 6, 7, 8]:
    run_var(f"Long periods>={periods}", direction_filter="long",
            var_stop_buffer=7.0, var_min_rr=0.5, var_min_ib_periods=periods)

print("\n" + "=" * 140)
print("SECTION 6: OTF THRESHOLD + ROTATION REQUIREMENT")
print("=" * 140)
for otf in [1, 2, 3, 4]:
    run_var(f"Long otf<={otf} (rotation)", direction_filter="long",
            var_stop_buffer=7.0, var_min_rr=0.5, var_max_otf=otf)
run_var("Long no rotation req", direction_filter="long",
        var_stop_buffer=7.0, var_min_rr=0.5, var_require_rotation=False)

print("\n" + "=" * 140)
print("SECTION 7: VA WIDTH REQUIREMENT (wider = cleaner setups)")
print("=" * 140)
# Note: VA width is checked in va_rotation.py as va_width < 4.0
# We can't change the code here, but we can check what happens with zone width
for zone in [1.0, 2.0, 3.0, 5.0, 7.0]:
    run_var(f"Long zone={zone}pt", direction_filter="long",
            var_stop_buffer=7.0, var_min_rr=0.5, var_zone_pts=zone)

print("\n" + "=" * 140)
print("SECTION 8: MAX TRADES PER DAY")
print("=" * 140)
for max_trades in [1, 2, 3, 4, 6, 8, 12]:
    run_var(f"Long max_trades={max_trades}", direction_filter="long",
            var_stop_buffer=7.0, var_min_rr=0.5, max_var_trades=max_trades)
