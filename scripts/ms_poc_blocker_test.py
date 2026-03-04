#!/usr/bin/env python3
"""Test: does prev_day POC between entry and target block short trades?"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
from backtester.session import SessionState, update_session
from backtester.indicators import compute_indicators

def _ms_base():
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
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = True
    cfg.ms_use_dev_va = True
    cfg.ms_use_poc = True
    return cfg

df = load_tos_csv('data/es_5m_databento_2yr.csv', instrument='ES')
print(f"Loaded {len(df)} bars")

# Get trades from different configs
configs_to_test = [
    ("ON only", True, False, False, False, False),
    ("ON + pVA", True, True, False, False, False),
    ("ON + pVA + POC", True, True, False, False, True),
    ("All levels", True, True, True, True, True),
]

for label, on, pva, ib, dva, poc in configs_to_test:
    cfg = _ms_base()
    cfg.ms_use_on_levels = on
    cfg.ms_use_prev_va = pva
    cfg.ms_use_ib_levels = ib
    cfg.ms_use_dev_va = dva
    cfg.ms_use_poc = poc
    # ms_use_vp_levels controls VP levels (pVAH/pVAL/pPOC from volume profile)
    # Turn it off so we only get the structural levels we want
    cfg.ms_use_vp_levels = False

    # Run backtest
    trades = run_backtest(df.copy(), cfg)
    if not trades:
        print(f"{label}: NO TRADES")
        continue

    # Now replay to get prev_vp_poc for each trade's entry time
    df_replay = df.copy()
    compute_indicators(df_replay, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len, atr_avg_len=cfg.atr_avg_len)
    state = SessionState()
    prev_bar = None
    poc_at_time = {}  # timestamp -> prev_vp_poc
    daily_sma5_map = {}
    daily_sma20_map = {}

    # Also compute daily closes for SMA
    daily_closes = []
    current_day_close = None

    for idx, row in df_replay.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx

        if bar.get("new_rth", False) and current_day_close is not None:
            daily_closes.append(current_day_close)

        if bar.get("is_rth", False):
            current_day_close = bar["close"]

        update_session(state, bar, prev_bar, cfg)

        # Store prev_vp_poc for this timestamp
        poc_at_time[idx] = state.prev_vp_poc

        # Compute daily SMAs
        if len(daily_closes) >= 5:
            daily_sma5_map[idx] = sum(daily_closes[-5:]) / 5
        if len(daily_closes) >= 20:
            daily_sma20_map[idx] = sum(daily_closes[-20:]) / 20

        prev_bar = bar

    # Classify each trade
    shorts = [t for t in trades if t.direction == -1]
    longs = [t for t in trades if t.direction == 1]

    # For shorts: check if POC, SMA5, or SMA20 is between entry and target (blocking support)
    shorts_blocked = []  # POC or daily MA between entry and target
    shorts_clear = []    # no blocker

    for t in shorts:
        entry_time = t.entry_time
        entry_price = t.entry_price
        target = t.target

        poc_val = poc_at_time.get(entry_time, float('nan'))
        sma5 = daily_sma5_map.get(entry_time, float('nan'))
        sma20 = daily_sma20_map.get(entry_time, float('nan'))

        blocked = False
        blockers = []

        # POC between entry and target (support blocker)
        if not math.isnan(poc_val) and target < poc_val < entry_price:
            blocked = True
            blockers.append(f"POC={poc_val:.0f}")

        # Daily SMA5 between entry and target
        if not math.isnan(sma5) and target < sma5 < entry_price:
            blocked = True
            blockers.append(f"SMA5={sma5:.0f}")

        # Daily SMA20 between entry and target
        if not math.isnan(sma20) and target < sma20 < entry_price:
            blocked = True
            blockers.append(f"SMA20={sma20:.0f}")

        if blocked:
            shorts_blocked.append(t)
        else:
            shorts_clear.append(t)

    # For longs: check if POC or daily MA is between entry and target (blocking resistance)
    longs_blocked = []
    longs_clear = []

    for t in longs:
        entry_time = t.entry_time
        entry_price = t.entry_price
        target = t.target

        poc_val = poc_at_time.get(entry_time, float('nan'))
        sma5 = daily_sma5_map.get(entry_time, float('nan'))
        sma20 = daily_sma20_map.get(entry_time, float('nan'))

        blocked = False
        if not math.isnan(poc_val) and entry_price < poc_val < target:
            blocked = True
        if not math.isnan(sma5) and entry_price < sma5 < target:
            blocked = True
        if not math.isnan(sma20) and entry_price < sma20 < target:
            blocked = True

        if blocked:
            longs_blocked.append(t)
        else:
            longs_clear.append(t)

    # Print results
    print(f"\n{'='*120}")
    print(f"  {label}")
    print(f"{'='*120}")

    all_m = compute_metrics(trades)
    print(f"  ALL:           {all_m.total_trades:>4d}t  WR {all_m.win_rate:.1f}%  PF {all_m.profit_factor:.3f}  P&L ${all_m.net_pnl:>+9,.0f}")

    if shorts:
        s_m = compute_metrics(shorts)
        print(f"  ALL SHORTS:    {s_m.total_trades:>4d}t  WR {s_m.win_rate:.1f}%  PF {s_m.profit_factor:.3f}  P&L ${s_m.net_pnl:>+9,.0f}")
    if longs:
        l_m = compute_metrics(longs)
        print(f"  ALL LONGS:     {l_m.total_trades:>4d}t  WR {l_m.win_rate:.1f}%  PF {l_m.profit_factor:.3f}  P&L ${l_m.net_pnl:>+9,.0f}")

    print(f"\n  --- SHORTS: BLOCKER ANALYSIS (POC/SMA5/SMA20 between entry & target) ---")
    if shorts_clear:
        sc_m = compute_metrics(shorts_clear)
        print(f"  CLEAR (no blocker):  {sc_m.total_trades:>4d}t  WR {sc_m.win_rate:.1f}%  PF {sc_m.profit_factor:.3f}  P&L ${sc_m.net_pnl:>+9,.0f}")
    else:
        print(f"  CLEAR (no blocker):     0t")
    if shorts_blocked:
        sb_m = compute_metrics(shorts_blocked)
        print(f"  BLOCKED:             {sb_m.total_trades:>4d}t  WR {sb_m.win_rate:.1f}%  PF {sb_m.profit_factor:.3f}  P&L ${sb_m.net_pnl:>+9,.0f}")
    else:
        print(f"  BLOCKED:                0t")

    print(f"\n  --- LONGS: BLOCKER ANALYSIS ---")
    if longs_clear:
        lc_m = compute_metrics(longs_clear)
        print(f"  CLEAR (no blocker):  {lc_m.total_trades:>4d}t  WR {lc_m.win_rate:.1f}%  PF {lc_m.profit_factor:.3f}  P&L ${lc_m.net_pnl:>+9,.0f}")
    else:
        print(f"  CLEAR (no blocker):     0t")
    if longs_blocked:
        lb_m = compute_metrics(longs_blocked)
        print(f"  BLOCKED:             {lb_m.total_trades:>4d}t  WR {lb_m.win_rate:.1f}%  PF {lb_m.profit_factor:.3f}  P&L ${lb_m.net_pnl:>+9,.0f}")
    else:
        print(f"  BLOCKED:                0t")

    # Combined filter: what if we skip ALL blocked trades?
    clear_trades = shorts_clear + longs_clear
    if clear_trades:
        cf_m = compute_metrics(clear_trades)
        pnls = [t.pnl_dollar for t in clear_trades]
        _, p_val = stats.ttest_1samp(pnls, 0) if len(clear_trades) >= 5 else (0, 1.0)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        print(f"\n  FILTERED (clear only):  {cf_m.total_trades:>4d}t  WR {cf_m.win_rate:.1f}%  PF {cf_m.profit_factor:.3f}  P&L ${cf_m.net_pnl:>+9,.0f}  p={p_val:.3f}{sig}")

    blocked_trades = shorts_blocked + longs_blocked
    if blocked_trades:
        bt_m = compute_metrics(blocked_trades)
        print(f"  REMOVED (blocked):     {bt_m.total_trades:>4d}t  WR {bt_m.win_rate:.1f}%  PF {bt_m.profit_factor:.3f}  P&L ${bt_m.net_pnl:>+9,.0f}")

    # Also test: what about using the blocker AS the target instead of skipping?
    # If POC is between entry and target for a short, use POC as target instead
    print(f"\n  --- WHAT IF: Use blocker as target (POC/MA as target instead of skip) ---")
    # This requires re-calculating PnL, which we can estimate
    # If the original target was beyond POC, using POC would give a closer target
    # The trade would need to travel less distance = higher hit rate but smaller wins
    # Just report the distance analysis
    if shorts_blocked:
        avg_orig_dist = sum(t.entry_price - t.target for t in shorts_blocked) / len(shorts_blocked)
        # Estimate POC distance
        poc_dists = []
        for t in shorts_blocked:
            poc_val = poc_at_time.get(t.entry_time, float('nan'))
            if not math.isnan(poc_val) and poc_val < t.entry_price:
                poc_dists.append(t.entry_price - poc_val)
        if poc_dists:
            avg_poc_dist = sum(poc_dists) / len(poc_dists)
            print(f"  Blocked shorts: avg target dist={avg_orig_dist:.1f}pts, avg POC dist={avg_poc_dist:.1f}pts")
            print(f"  Using POC as target would reduce travel by {(1-avg_poc_dist/avg_orig_dist)*100:.0f}%")
        else:
            print(f"  Blocked shorts: avg target dist={avg_orig_dist:.1f}pts (no POC-specific blockers)")
    else:
        print(f"  No blocked shorts to analyze")
