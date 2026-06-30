#!/usr/bin/env python3
"""VIX Spike ES drift compare: validated production config vs deployed config.

Validated (production card, audit/handoff):
  threshold=7%, no ES move filter, 30bps stop, HOLD-ALL-DAY (15:55 flatten),
  9:35 entry start, no upper cutoff (we use 15:55 max as the hold-all-day endpoint).

Deployed (TV port + AR config Apr 8 enhancement):
  threshold=5%, ES move filter=-0.1%, 50bps stop, fixed_target=20pt,
  max_hold=18 bars (90 min), 9:35-15:00 entry window.

Note: TV port self-states it cannot fire in current engine (vix_prev_close
missing in SessionContext). This backtest quantifies what *should* be
running, not what's currently producing $0 of live P&L.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.test_vix_spike_es_v2 import Config, load_data, run_backtest, compute_metrics
from scipy import stats as scipy_stats


def t_test_pnls(pnls):
    if len(pnls) < 5:
        return 1.0
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)
    return t_pval / 2 if t_stat > 0 else 1 - t_pval / 2


def profit_factor(pnls):
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return wins / losses if losses > 0 else float("inf")


def max_dd(pnls):
    eq = np.cumsum(pnls)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0


def report(label, trades):
    pnls = [t.pnl_dollar for t in trades]
    m = compute_metrics(trades) if trades else {"total": 0, "win_rate": 0, "pf": 0, "net_pnl": 0, "sharpe": 0}
    p = t_test_pnls(pnls)
    print(f"\n[{label}]")
    print(f"  trades={m['total']}  PF={m['pf']:.3f}  P&L=${m['net_pnl']:+,.0f}  "
          f"WR={m['win_rate']:.1f}%  Sharpe={m['sharpe']:.2f}  MaxDD=${max_dd(pnls):,.0f}  p={p:.4f}")
    return {"trades": m["total"], "pnl": m["net_pnl"], "pf": m["pf"], "wr": m["win_rate"],
            "sharpe": m["sharpe"], "max_dd": max_dd(pnls), "p": p}


def main():
    print("Loading 2yr ES Databento data...")
    df, vix_lookup, session_opens, prev_session_data = load_data()
    print(f"  {len(df):,} bars")

    print("\n=== VALIDATED (production: 7% / no filter / 30bps / hold-all-day) ===")
    v_cfg = Config(
        signal_mode="daily_spike",
        spike_threshold=0.07,
        entry_mode="red_bar",
        exit_mode="hold_all_day",
        stop_bps=30.0,
        max_hold_bars=78,  # placeholder; hold_all_day ignores max_hold
        es_move_filter=0.0,
        entry_start=935,
        entry_end=1500,
    )
    v_trades = run_backtest(df, vix_lookup, session_opens, prev_session_data, v_cfg)
    v_r = report("validated", v_trades)

    print("\n=== DEPLOYED (TV port: 5% / -0.1% filter / 50bps / fixed-target 20pt / 90min) ===")
    d_cfg = Config(
        signal_mode="daily_spike",
        spike_threshold=0.05,
        entry_mode="red_bar",
        exit_mode="fixed_target",
        target_pts=20.0,
        stop_bps=50.0,
        max_hold_bars=18,
        es_move_filter=-0.001,
        entry_start=935,
        entry_end=1500,
    )
    d_trades = run_backtest(df, vix_lookup, session_opens, prev_session_data, d_cfg)
    d_r = report("deployed", d_trades)

    print("\n=== DELTA (deployed - validated) ===")
    print(f"  Trades: {d_r['trades'] - v_r['trades']:+d}")
    print(f"  P&L:    ${d_r['pnl'] - v_r['pnl']:+,.0f}")
    print(f"  PF:     {d_r['pf'] - v_r['pf']:+.3f}")
    print(f"  Sharpe: {d_r['sharpe'] - v_r['sharpe']:+.2f}")
    print(f"  MaxDD:  ${d_r['max_dd'] - v_r['max_dd']:+,.0f}")
    print("\nNote: TV port cannot fire live (vix_prev_close not in SessionContext).")
    print("Deployed P&L above is what *would* run if engine plumbing were fixed.")


if __name__ == "__main__":
    main()
