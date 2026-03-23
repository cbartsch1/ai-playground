#!/usr/bin/env python3
"""Unified strategy runner for autoresearch experiments.

Runs full backtest + walk-forward validation + statistical significance.
Outputs grep-able metrics and appends to results.tsv.

Usage:
    python scripts/run_strategy.py --setup amt_v8
    python scripts/run_strategy.py --setup amt_v8 --desc "experiment: add ATR filter"
"""

import argparse
import importlib
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics
import backtester.setups.ib_breakout as _ib_breakout_module

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_2YR = os.path.join(PROJECT_ROOT, "data", "es_5m_databento_2yr.csv")
RESULTS_TSV = os.path.join(PROJECT_ROOT, "results.tsv")

# --- Walk-forward split ---
WF_SPLIT_DATE = "2025-08-15"

# --- Setup registry ---
SETUP_REGISTRY = {
    "amt_v8": "backtester.setups.amt_breakout",
}


def load_setup(setup_name: str):
    """Import setup module by name."""
    if setup_name not in SETUP_REGISTRY:
        raise ValueError(f"Unknown setup: {setup_name}. Available: {list(SETUP_REGISTRY.keys())}")
    return importlib.import_module(SETUP_REGISTRY[setup_name])


def apply_setup(setup_module) -> StrategyConfig:
    """Get config and monkey-patch ib_breakout signal if setup provides one."""
    cfg = setup_module.get_config()
    if hasattr(setup_module, "check_signal"):
        _ib_breakout_module.check_signal = setup_module.check_signal
    return cfg


def run_full_backtest(df, cfg):
    trades = run_backtest(df.copy(), cfg)
    m = compute_metrics(trades, cfg.initial_capital) if trades else None
    return trades, m


def run_walk_forward(df, cfg, split_date=WF_SPLIT_DATE):
    df_is = df[df.index < split_date]
    df_oos = df[df.index >= split_date]
    trades_is = run_backtest(df_is.copy(), cfg)
    trades_oos = run_backtest(df_oos.copy(), cfg)
    m_is = compute_metrics(trades_is, cfg.initial_capital) if trades_is else None
    m_oos = compute_metrics(trades_oos, cfg.initial_capital) if trades_oos else None
    return m_is, m_oos, len(trades_is), len(trades_oos)


def run_stats(pnls, n_boot=2000, n_perm=2000, seed=42):
    np.random.seed(seed)
    pnls = np.array(pnls)
    n = len(pnls)

    # T-test
    t_stat, t_pval = scipy_stats.ttest_1samp(pnls, 0)

    # Permutation test
    obs_pnl = np.sum(pnls)
    abs_pnls = np.abs(pnls)
    count_better = 0
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if np.dot(signs, abs_pnls) >= obs_pnl:
            count_better += 1
    perm_pval = count_better / n_perm

    # Bootstrap probability of profit
    boot_pnl = np.array([
        np.sum(rng.choice(pnls, size=n, replace=True))
        for _ in range(n_boot)
    ])
    prob_profit = float(np.mean(boot_pnl > 0))

    return t_pval, perm_pval, prob_profit


def quality_gate(m_full, m_is, m_oos, t_pval, baseline_trades):
    """Return status string: PASS or FAIL_<reason>."""
    if m_full is None or m_full.total_trades == 0:
        return "FAIL_NO_TRADES"
    if m_full.profit_factor <= 1.0:
        return "FAIL_PF"
    if t_pval >= 0.05:
        return "FAIL_SIG"
    if m_oos is None or m_oos.profit_factor <= 1.0:
        return "FAIL_OOS"
    if m_is and m_oos and m_is.profit_factor > 0:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor
        if pf_ratio < 0.65:
            return "FAIL_WF"
    if baseline_trades > 0 and m_full.total_trades < baseline_trades * 0.70:
        return "FAIL_TRADES"
    return "PASS"


def print_metrics(m_full, m_is, m_oos, t_pval, perm_pval, prob_profit):
    print("=" * 60)
    print("  TRAPDOOR — AMT-TEMA v8 Results")
    print("=" * 60)
    if m_full:
        print(f"  Net P&L:             ${m_full.net_pnl:,.2f}")
        print(f"  Win Rate:            {m_full.win_rate:.1f}%")
        print(f"  Profit Factor:       {m_full.profit_factor:.3f}")
        print(f"  Sharpe:              {m_full.sharpe:.2f}")
        print(f"  Total Trades:        {m_full.total_trades}")
        print(f"  Max Drawdown:        ${m_full.max_drawdown:,.2f}")
        print(f"  Avg Trade:           ${m_full.avg_trade:,.2f}")
    print()
    print("--- Walk-Forward Validation ---")
    if m_is:
        print(f"  IS  Trades: {m_is.total_trades:4d}  PF: {m_is.profit_factor:.3f}  "
              f"P&L: ${m_is.net_pnl:,.0f}")
    if m_oos:
        print(f"  OOS PF:              {m_oos.profit_factor:.3f}")
        print(f"  OOS Net P&L:         ${m_oos.net_pnl:,.0f}")
        print(f"  OOS Win Rate:        {m_oos.win_rate:.1f}%")
    if m_is and m_oos and m_is.profit_factor > 0:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor
        print(f"  WF PF ratio:         {pf_ratio:.3f}")
    print()
    print("--- Statistical Significance ---")
    print(f"  t-test p-value:      {t_pval:.6f}")
    print(f"  Permutation p-value: {perm_pval:.6f}")
    print(f"  Bootstrap P(profit): {prob_profit:.2%}")
    print("=" * 60)


def update_results_tsv(commit, m_full, m_oos, m_is, t_pval, status, description):
    header = "commit\tpf\toos_pf\tpf_ratio\tpnl\tsharpe\ttrades\tp_value\tstatus\tdescription\n"
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write(header)

    pf = m_full.profit_factor if m_full else 0.0
    oos_pf = m_oos.profit_factor if m_oos else 0.0
    pf_ratio = (m_oos.profit_factor / m_is.profit_factor
                if m_is and m_oos and m_is.profit_factor > 0 else 0.0)
    pnl = m_full.net_pnl if m_full else 0.0
    sharpe = m_full.sharpe if m_full else 0.0
    trades = m_full.total_trades if m_full else 0

    row = (f"{commit}\t{pf:.3f}\t{oos_pf:.3f}\t{pf_ratio:.3f}\t"
           f"{pnl:.2f}\t{sharpe:.2f}\t{trades}\t{t_pval:.6f}\t{status}\t{description}\n")

    with open(RESULTS_TSV, "a") as f:
        f.write(row)


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="AMT-TEMA Autoresearch Runner")
    parser.add_argument("--setup", required=True, help="Strategy setup name (e.g., amt_v8)")
    parser.add_argument("--data", default=DATA_2YR, help="CSV data file path")
    parser.add_argument("--split", default=WF_SPLIT_DATE, help="Walk-forward split date YYYY-MM-DD")
    parser.add_argument("--desc", default="", help="Experiment description for results.tsv")
    parser.add_argument("--baseline-trades", type=int, default=177,
                        help="Baseline trade count (for 70%% floor check, default: 177)")
    parser.add_argument("--no-tsv", action="store_true", help="Skip results.tsv update")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load setup and apply config + monkey-patch
    print(f"Setup: {args.setup}")
    setup_module = load_setup(args.setup)
    cfg = apply_setup(setup_module)

    # Load data
    print(f"Loading: {args.data}")
    df = load_tos_csv(args.data, instrument=cfg.instrument)
    print(f"Bars: {len(df)} | {df.index[0].date()} to {df.index[-1].date()}")

    # Full backtest
    print("Running full backtest...")
    trades_full, m_full = run_full_backtest(df, cfg)
    print(f"Trades: {len(trades_full)}")

    # Walk-forward
    print(f"Walk-forward split: {args.split}")
    m_is, m_oos, n_is, n_oos = run_walk_forward(df, cfg, args.split)
    print(f"IS trades: {n_is} | OOS trades: {n_oos}")

    # Stats
    print("Running significance tests...")
    pnls = [t.pnl_dollar for t in trades_full]
    if pnls:
        t_pval, perm_pval, prob_profit = run_stats(pnls, seed=args.seed)
    else:
        t_pval, perm_pval, prob_profit = 1.0, 1.0, 0.0

    # Print
    print()
    print_metrics(m_full, m_is, m_oos, t_pval, perm_pval, prob_profit)

    # Quality gate
    status = quality_gate(m_full, m_is, m_oos, t_pval, args.baseline_trades)
    print(f"\n  Quality Gate: {status}")

    # Save
    if not args.no_tsv:
        commit = get_git_commit()
        desc = args.desc or f"run @ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        update_results_tsv(commit, m_full, m_oos, m_is, t_pval, status, desc)
        print(f"  Saved to: {RESULTS_TSV}")


if __name__ == "__main__":
    main()
