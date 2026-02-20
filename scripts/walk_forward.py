#!/usr/bin/env python3
"""Walk-forward validation for AMT-TEMA v7 filters.

Splits 1-year Databento data into two halves:
  - In-sample:      Feb 16 2025 – Aug 15 2025  (~6 months)
  - Out-of-sample:  Aug 15 2025 – Feb 13 2026  (~6 months)

Runs baseline (v6.1) and v7 (no VA + noon blackout + skip Friday) on each half.
If v7's edge persists out-of-sample, the filters are structural, not overfit.

Usage:
    python scripts/walk_forward.py data/es_5m_databento.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


def make_baseline_config():
    """v6.1 baseline — all defaults."""
    return StrategyConfig()


def make_v7_config():
    """v7 — no VA, noon blackout, skip Friday."""
    return StrategyConfig(
        use_va_fade=False,
        blackout_start=1200,
        blackout_end=1300,
        skip_friday=True,
    )


def run_on_slice(df_slice, cfg, label):
    """Run backtest on a DataFrame slice, return Metrics dataclass."""
    trades = run_backtest(df_slice.copy(), cfg)
    if not trades:
        print(f"  {label}: 0 trades")
        return None
    m = compute_metrics(trades, cfg.initial_capital)
    print(f"  {label}: {m.total_trades} trades | "
          f"WR {m.win_rate:.1f}% | PF {m.profit_factor:.3f} | "
          f"P&L ${m.net_pnl:,.0f} | DD ${m.max_drawdown:,.0f} | "
          f"Sharpe {m.sharpe:.2f}")
    return m


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("csv_file", help="Path to Databento CSV")
    parser.add_argument("--split", default="2025-08-15",
                        help="Split date YYYY-MM-DD (default: 2025-08-15)")
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    df = load_tos_csv(args.csv_file)
    print(f"Loaded {len(df)} bars: {df.index[0]} to {df.index[-1]}\n")

    split_date = args.split
    df_in = df[df.index < split_date]
    df_out = df[df.index >= split_date]
    print(f"In-sample:      {df_in.index[0].date()} to {df_in.index[-1].date()} "
          f"({len(df_in)} bars)")
    print(f"Out-of-sample:  {df_out.index[0].date()} to {df_out.index[-1].date()} "
          f"({len(df_out)} bars)\n")

    # --- In-sample ---
    print("=== IN-SAMPLE ===")
    m_in_base = run_on_slice(df_in, make_baseline_config(), "Baseline")
    m_in_v7 = run_on_slice(df_in, make_v7_config(), "v7     ")

    # --- Out-of-sample ---
    print("\n=== OUT-OF-SAMPLE ===")
    m_out_base = run_on_slice(df_out, make_baseline_config(), "Baseline")
    m_out_v7 = run_on_slice(df_out, make_v7_config(), "v7     ")

    # --- Full year (sanity check) ---
    print("\n=== FULL YEAR ===")
    run_on_slice(df, make_baseline_config(), "Baseline")
    run_on_slice(df, make_v7_config(), "v7     ")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    if m_in_v7 and m_out_v7 and m_in_base and m_out_base:
        in_delta = m_in_v7.net_pnl - m_in_base.net_pnl
        out_delta = m_out_v7.net_pnl - m_out_base.net_pnl
        print(f"\nv7 improvement over baseline:")
        print(f"  In-sample:      ${in_delta:+,.0f}")
        print(f"  Out-of-sample:  ${out_delta:+,.0f}")

        if out_delta > 0:
            print(f"\n  PASS: v7 edge persists out-of-sample (+${out_delta:,.0f})")
            if m_out_v7.profit_factor > 1.2:
                print(f"  STRONG: Out-of-sample PF {m_out_v7.profit_factor:.3f} > 1.2")
            elif m_out_v7.profit_factor > 1.0:
                print(f"  MARGINAL: Out-of-sample PF {m_out_v7.profit_factor:.3f} "
                      f"(profitable but PF < 1.2)")
        else:
            print(f"\n  FAIL: v7 edge does NOT persist out-of-sample (${out_delta:+,.0f})")
            print(f"  The filters may be overfit to the in-sample period.")

        # Compare PF ratios
        pf_ratio = (m_out_v7.profit_factor / m_in_v7.profit_factor
                    if m_in_v7.profit_factor > 0 else 0)
        print(f"\n  PF ratio (out/in): {pf_ratio:.2f} "
              f"(>0.7 = robust, <0.5 = likely overfit)")


if __name__ == "__main__":
    main()
