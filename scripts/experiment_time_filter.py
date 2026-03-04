#!/usr/bin/env python3
"""Experiment: Time Filter — no entries after 14:00 ET.

Hypothesis: 0DTE gamma distortion makes afternoon trading structurally different.
After 14:00 ET, dealer hedging accelerates and creates whipsaws that hurt our setups.
Cutting entries after 14:00 should improve quality without losing much volume.

Tests both v8.1 and v13 with trade_end=1400 vs baseline 1500.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics

DATA = "data/es_5m_databento_2yr.csv"
SPLIT_DATE = pd.Timestamp("2025-02-14", tz="America/New_York")


def v81_config():
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    cfg.use_ib_break = True
    cfg.use_trend_filter = True
    cfg.use_ib_reject = True
    cfg.rej_wide_only = True
    cfg.rej_target = "ib_low"
    cfg.rej_stop_buffer = 8.0
    cfg.max_rej_trades = 8
    return cfg


def v13_config():
    cfg = v81_config()
    cfg.use_level_reject = True
    cfg.lvl_enabled_levels = ("ONH",)
    cfg.lvl_require_tema = False
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = 5.0
    cfg.lvl_stop_buffer = 7.0
    cfg.max_lvl_trades = 4
    cfg.lvl_max_tests = 3
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_rr = 0.5
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_target_skip = 2
    return cfg


def print_metrics(label, m):
    print(f"  {label}: {m.total_trades} trades, PF {m.profit_factor:.2f}, "
          f"WR {m.win_rate:.1f}%, ${m.net_pnl:+,.0f}, Sharpe {m.sharpe:.2f}")


def run_experiment(df, cfg, label):
    df_is = df[df.index < SPLIT_DATE].copy()
    df_oos = df[df.index >= SPLIT_DATE].copy()

    # Baseline (trade_end=1500)
    trades_is = run_backtest(df_is, cfg)
    trades_oos = run_backtest(df_oos, cfg)
    m_is = compute_metrics(trades_is)
    m_oos = compute_metrics(trades_oos)

    # Time-filtered variants
    for cutoff in [1400, 1300, 1200]:
        cfg_t = StrategyConfig(**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()})
        cfg_t.trade_end = cutoff
        # For v13 with own_filters, also set its effective window
        trades_is_t = run_backtest(df_is, cfg_t)
        trades_oos_t = run_backtest(df_oos, cfg_t)
        m_is_t = compute_metrics(trades_is_t)
        m_oos_t = compute_metrics(trades_oos_t)

        print(f"\n  --- trade_end={cutoff} ---")
        print_metrics("IS ", m_is_t)
        print_metrics("OOS", m_oos_t)
        if m_is_t.profit_factor > 0:
            print(f"  PF Ratio: {m_oos_t.profit_factor / m_is_t.profit_factor:.2f}")
        print(f"  Delta vs baseline: IS ${m_is_t.net_pnl - m_is.net_pnl:+,.0f} "
              f"({m_is_t.total_trades - m_is.total_trades:+d} trades), "
              f"OOS ${m_oos_t.net_pnl - m_oos.net_pnl:+,.0f} "
              f"({m_oos_t.total_trades - m_oos.total_trades:+d} trades)")

    # Also check: what happens to JUST the afternoon trades?
    print(f"\n  --- Afternoon-only trades (14:00-15:00) analysis ---")
    all_trades = run_backtest(df.copy(), cfg)
    afternoon = [t for t in all_trades if hasattr(t.entry_time, 'hour') and t.entry_time.hour >= 14]
    morning = [t for t in all_trades if hasattr(t.entry_time, 'hour') and t.entry_time.hour < 14]
    if afternoon:
        m_aft = compute_metrics(afternoon)
        m_morn = compute_metrics(morning)
        print_metrics("Morning (10:35-14:00)", m_morn)
        print_metrics("Afternoon (14:00-15:00)", m_aft)
    else:
        print("  No afternoon trades found (entry_time may not have hour attribute)")

    print(f"\n  BASELINE (trade_end=1500):")
    print_metrics("IS ", m_is)
    print_metrics("OOS", m_oos)


def main():
    print("Loading data...")
    df = load_tos_csv(DATA, instrument="ES")
    print(f"Loaded {len(df)} bars")

    print(f"\n{'='*70}")
    print(f"  v8.1 — Time Filter Test")
    print(f"{'='*70}")
    run_experiment(df.copy(), v81_config(), "v8.1")

    print(f"\n{'='*70}")
    print(f"  v13 — Time Filter Test")
    print(f"{'='*70}")
    run_experiment(df.copy(), v13_config(), "v13")

    print(f"\n{'='*70}")
    print("  Time Filter Experiment Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
