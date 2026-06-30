#!/usr/bin/env python3
"""Regression check for test_vix_spike_es_v2.py.

Runs three representative Configs (covering 3 different exit modes) on the
2yr data and prints metrics. Run BEFORE adding fixed_target, save output.
Run AFTER, diff output. They must be identical.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.test_vix_spike_es_v2 import (
    Config, load_data, run_backtest, compute_metrics, t_test
)


def fmt(label, m, p):
    return (f"{label:50s} | trades={m['total']:4d} pf={m['pf']:7.4f} "
            f"net=${m['net_pnl']:>+10,.2f} wr={m['win_rate']:5.2f}% "
            f"sharpe={m['sharpe']:6.3f} dd=${m['max_dd']:>10,.2f} p={p:.6f}")


def main():
    df, vix_lookup, session_opens, prev_session_data = load_data()
    print("\n=== REGRESSION CHECK — pre/post fixed_target add ===\n")

    cfgs = [
        Config(label="baseline-green",
               signal_mode="daily_spike", spike_threshold=0.07,
               entry_mode="red_bar", exit_mode="green_bar",
               stop_bps=30, max_hold_bars=9, es_move_filter=-0.002),
        Config(label="hold-all-day",
               signal_mode="daily_spike", spike_threshold=0.07,
               entry_mode="red_bar", exit_mode="hold_all_day",
               stop_bps=30, max_hold_bars=200, es_move_filter=0.0,
               entry_start=935, entry_end=1555),
        Config(label="atr-target",
               signal_mode="daily_spike", spike_threshold=0.07,
               entry_mode="red_bar", exit_mode="atr_target",
               stop_bps=30, max_hold_bars=9, es_move_filter=-0.002,
               atr_target_mult=3.0),
    ]

    for cfg in cfgs:
        trades = run_backtest(df, vix_lookup, session_opens, prev_session_data, cfg)
        m = compute_metrics(trades)
        p = t_test(trades)
        print(fmt(cfg.label, m, p))

    print("\n=== END ===\n")


if __name__ == "__main__":
    main()
