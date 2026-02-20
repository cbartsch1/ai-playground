#!/usr/bin/env python3
"""Statistical significance testing for AMT-TEMA v7 strategy.

Tests whether the strategy's returns are statistically different from zero
(i.e., not explainable by random chance).

Methods:
  1. One-sample t-test on trade P&L (H0: mean trade P&L = 0)
  2. Bootstrap confidence intervals on key metrics (P&L, PF, Sharpe, WR)
  3. Monte Carlo permutation test (shuffle entry/exit pairing)

Usage:
    python scripts/stat_significance.py output/combined/trade_log.csv
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000
INITIAL_CAPITAL = 100_000.0
COMMISSION_RT = 5.0  # $2.50 per side x 2


def load_trades(filepath):
    """Load trade log CSV."""
    df = pd.read_csv(filepath)
    return df


def compute_sharpe(pnls, capital=INITIAL_CAPITAL):
    """Annualized Sharpe from trade P&Ls."""
    if len(pnls) < 2:
        return 0.0
    returns = np.array(pnls) / capital
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(returns) / std * np.sqrt(252)


def compute_pf(pnls):
    """Profit factor from P&L array."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


def compute_max_dd(pnls, capital=INITIAL_CAPITAL):
    """Max drawdown from P&L array."""
    equity = capital
    peak = equity
    max_dd = 0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def t_test(pnls):
    """One-sample t-test: H0 = mean trade P&L is zero."""
    t_stat, p_value = stats.ttest_1samp(pnls, 0)
    n = len(pnls)
    mean = np.mean(pnls)
    se = stats.sem(pnls)
    ci_95 = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return t_stat, p_value, ci_95


def bootstrap_ci(pnls, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap confidence intervals for key metrics."""
    pnls = np.array(pnls)
    n = len(pnls)

    boot_pnl = np.zeros(n_boot)
    boot_pf = np.zeros(n_boot)
    boot_wr = np.zeros(n_boot)
    boot_sharpe = np.zeros(n_boot)
    boot_dd = np.zeros(n_boot)
    boot_avg = np.zeros(n_boot)

    for i in range(n_boot):
        sample = np.random.choice(pnls, size=n, replace=True)
        boot_pnl[i] = np.sum(sample)
        boot_pf[i] = compute_pf(sample)
        boot_wr[i] = np.sum(sample > 0) / n * 100
        boot_sharpe[i] = compute_sharpe(sample)
        boot_dd[i] = compute_max_dd(sample)
        boot_avg[i] = np.mean(sample)

    alpha = (1 - ci) / 2
    lo = alpha * 100
    hi = (1 - alpha) * 100

    results = {
        "Net P&L": (np.percentile(boot_pnl, lo), np.percentile(boot_pnl, hi)),
        "Avg Trade": (np.percentile(boot_avg, lo), np.percentile(boot_avg, hi)),
        "Profit Factor": (np.percentile(boot_pf, lo), np.percentile(boot_pf, hi)),
        "Win Rate %": (np.percentile(boot_wr, lo), np.percentile(boot_wr, hi)),
        "Sharpe": (np.percentile(boot_sharpe, lo), np.percentile(boot_sharpe, hi)),
        "Max Drawdown": (np.percentile(boot_dd, lo), np.percentile(boot_dd, hi)),
    }

    # Probability of loss (P&L < 0)
    prob_loss = np.mean(boot_pnl < 0)

    return results, prob_loss


def permutation_test(pnls, n_perm=N_PERMUTATIONS):
    """Monte Carlo permutation test.

    Randomly flip the sign of each trade's P&L to simulate a strategy
    with no edge (50/50 random direction). Count how often the random
    strategy produces net P&L >= observed.
    """
    observed_pnl = np.sum(pnls)
    abs_pnls = np.abs(pnls)
    n = len(abs_pnls)

    count_better = 0
    for _ in range(n_perm):
        # Random signs: +1 or -1
        signs = np.random.choice([-1, 1], size=n)
        random_pnl = np.sum(signs * abs_pnls)
        if random_pnl >= observed_pnl:
            count_better += 1

    p_value = count_better / n_perm
    return p_value, observed_pnl


def main():
    parser = argparse.ArgumentParser(description="Statistical significance for AMT-TEMA")
    parser.add_argument("trade_log", help="Path to trade_log.csv")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP,
                        help=f"Number of bootstrap samples (default: {N_BOOTSTRAP})")
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS,
                        help=f"Number of permutations (default: {N_PERMUTATIONS})")
    args = parser.parse_args()

    df = load_trades(args.trade_log)
    pnls = df["pnl_dollar"].values
    n = len(pnls)

    print(f"Trade log: {args.trade_log}")
    print(f"Trades: {n}")
    print(f"Net P&L: ${np.sum(pnls):,.2f}")
    print(f"Avg Trade: ${np.mean(pnls):,.2f}")
    print(f"Win Rate: {np.sum(pnls > 0) / n * 100:.1f}%")
    print(f"Profit Factor: {compute_pf(pnls):.3f}")
    print(f"Sharpe: {compute_sharpe(pnls):.2f}")
    print(f"Max Drawdown: ${compute_max_dd(pnls):,.2f}")

    # --- 1. T-Test ---
    print("\n" + "=" * 60)
    print("1. ONE-SAMPLE T-TEST (H0: mean trade P&L = $0)")
    print("=" * 60)
    t_stat, p_value, ci_95 = t_test(pnls)
    print(f"  t-statistic:  {t_stat:.4f}")
    print(f"  p-value:      {p_value:.6f}")
    print(f"  95% CI:       ${ci_95[0]:,.2f} to ${ci_95[1]:,.2f}")
    if p_value < 0.01:
        print(f"  Result:       HIGHLY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        print(f"  Result:       SIGNIFICANT (p < 0.05)")
    elif p_value < 0.10:
        print(f"  Result:       MARGINALLY SIGNIFICANT (p < 0.10)")
    else:
        print(f"  Result:       NOT SIGNIFICANT (p >= 0.10)")

    # --- 2. Bootstrap CI ---
    print("\n" + "=" * 60)
    print(f"2. BOOTSTRAP CONFIDENCE INTERVALS ({args.bootstrap:,} samples)")
    print("=" * 60)
    boot_results, prob_loss = bootstrap_ci(pnls, n_boot=args.bootstrap)
    for metric, (lo, hi) in boot_results.items():
        if "%" in metric:
            print(f"  {metric:20s}  [{lo:8.1f}%  , {hi:8.1f}%  ]")
        elif "Factor" in metric or "Sharpe" in metric:
            print(f"  {metric:20s}  [{lo:8.3f}   , {hi:8.3f}   ]")
        else:
            print(f"  {metric:20s}  [${lo:>10,.2f} , ${hi:>10,.2f} ]")

    print(f"\n  Probability of net loss (bootstrap): {prob_loss:.2%}")
    print(f"  Probability of net profit:           {1 - prob_loss:.2%}")

    # --- 3. Permutation Test ---
    print("\n" + "=" * 60)
    print(f"3. MONTE CARLO PERMUTATION TEST ({args.permutations:,} permutations)")
    print("=" * 60)
    perm_p, obs_pnl = permutation_test(pnls, n_perm=args.permutations)
    print(f"  Observed P&L: ${obs_pnl:,.2f}")
    print(f"  p-value:      {perm_p:.6f}")
    if perm_p < 0.01:
        print(f"  Result:       HIGHLY SIGNIFICANT (p < 0.01)")
    elif perm_p < 0.05:
        print(f"  Result:       SIGNIFICANT (p < 0.05)")
    elif perm_p < 0.10:
        print(f"  Result:       MARGINALLY SIGNIFICANT (p < 0.10)")
    else:
        print(f"  Result:       NOT SIGNIFICANT (p >= 0.10)")
    print(f"  Interpretation: {perm_p:.2%} chance that random trading "
          f"produces P&L >= ${obs_pnl:,.2f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  t-test p-value:        {p_value:.6f}")
    print(f"  Permutation p-value:   {perm_p:.6f}")
    print(f"  Bootstrap P(profit):   {1 - prob_loss:.2%}")
    print(f"  95% CI on avg trade:   ${ci_95[0]:,.2f} to ${ci_95[1]:,.2f}")

    both_sig = p_value < 0.05 and perm_p < 0.05
    if both_sig:
        print(f"\n  CONCLUSION: The strategy's edge is STATISTICALLY SIGNIFICANT")
        print(f"  at the 95% confidence level by both parametric and non-parametric tests.")
    elif p_value < 0.05 or perm_p < 0.05:
        print(f"\n  CONCLUSION: MIXED — significant by one test but not the other.")
        print(f"  More data recommended.")
    else:
        print(f"\n  CONCLUSION: The strategy's edge is NOT statistically significant.")
        print(f"  The observed profits could be explained by random chance.")


if __name__ == "__main__":
    main()
