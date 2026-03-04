#!/usr/bin/env python3
"""
Medallion 2.0 — HMM Optimizer

3-phase optimization:
  Phase 1: n_regimes sweep (2-8) — BIC/AIC + abbreviated walk-forward
  Phase 2: Confirmation sensitivity — leave-one-out impact on filter value
  Phase 3: Combined validation — full walk-forward at best n_regimes

Usage:
    python scripts/optimizer.py                    # all phases
    python scripts/optimizer.py --phase 1          # n_regimes only
    python scripts/optimizer.py --phase 2          # confirmations only
    python scripts/optimizer.py --n-regimes 5      # force specific n for phase 2-3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import time

from config.settings import (
    HMM_FEATURES,
    DEFAULT_N_REGIMES,
    CONFIRMATIONS,
    MACRO_CONFIRMATIONS,
    OPTIMIZER_N_RANGE,
    OPTIMIZER_MIN_OOS_P,
    OPTIMIZER_WF_FOLDS,
    BULLISH_REGIMES,
    BEARISH_REGIMES,
)
from models.hmm_regime import RegimeDetector
from data.data_loader import load_data
from backtester.regime_quality import RegimeQualityAnalyzer
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


# ============================================================
# Phase 1: n_regimes sweep
# ============================================================

def phase1_n_regimes_sweep(
    ohlcv: pd.DataFrame,
    hmm_features: pd.DataFrame,
    n_range: range = OPTIMIZER_N_RANGE,
    wf_folds: int = OPTIMIZER_WF_FOLDS,
    min_oos_p: float = OPTIMIZER_MIN_OOS_P,
) -> pd.DataFrame:
    """
    Sweep n_regimes from 2-8. For each n:
    - Fit HMM, compute BIC/AIC
    - Run quality analysis (summary score)
    - Run abbreviated walk-forward (wf_folds folds)
    - Report OOS p-value and separation

    Returns DataFrame with comparison table.
    """
    print("=" * 70)
    print("PHASE 1: n_regimes Sweep")
    print("=" * 70)

    features = hmm_features.dropna().copy()
    X = features[HMM_FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)
    n_samples = len(X_scaled)
    n_features = X_scaled.shape[1]

    # Prepare monthly blocks for walk-forward
    features_wf = features.copy()
    if features_wf.index.tz is not None:
        features_wf["year_month"] = features_wf.index.tz_localize(None).to_period("M")
    else:
        features_wf["year_month"] = features_wf.index.to_period("M")
    months = features_wf["year_month"].unique()

    results = []

    for n in n_range:
        print(f"\n--- Testing n_regimes = {n} ---")
        t0 = time.time()

        row = {"n_regimes": n}

        # Fit full model (multiple restarts)
        best_score = -np.inf
        best_model = None
        for seed in range(10):
            try:
                model = GaussianHMM(
                    n_components=n,
                    covariance_type="full",
                    n_iter=200,
                    random_state=42 + seed,
                    verbose=False,
                )
                model.fit(X_scaled)
                score = model.score(X_scaled)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            print(f"  FAILED to fit n={n}")
            row["error"] = "fit_failed"
            results.append(row)
            continue

        # BIC / AIC
        log_likelihood = best_model.score(X_scaled) * n_samples
        n_params = (
            n * n_features                              # means
            + n * n_features * (n_features + 1) / 2     # covariances (full)
            + n * (n - 1)                               # transitions
        )
        row["bic"] = -2 * log_likelihood + n_params * np.log(n_samples)
        row["aic"] = -2 * log_likelihood + 2 * n_params
        row["log_likelihood"] = log_likelihood
        row["n_params"] = int(n_params)

        # Quality score (full sample)
        detector = RegimeDetector(n_regimes=n, n_restarts=5, n_iter=100)
        detector.fit(features, feature_cols=HMM_FEATURES)
        preds = detector.predict(features)

        analyzer = RegimeQualityAnalyzer()
        quality = analyzer.analyze(ohlcv, preds)
        row["quality_score"] = quality.summary_score
        row["filter_sharpe"] = quality.filter_value.get("filtered_sharpe", 0)

        # Regime separation (full sample)
        if quality.regime_separation is not None:
            bull_row = quality.regime_separation[quality.regime_separation["signal"] == "bullish"]
            if not bull_row.empty and "p_value_vs_bearish" in bull_row.columns:
                row["separation_p"] = bull_row["p_value_vs_bearish"].iloc[0]

        # Abbreviated walk-forward (last wf_folds months as OOS)
        oos_p, oos_sep, oos_stability = _abbreviated_walk_forward(
            features_wf, ohlcv, months, n, wf_folds
        )
        row["oos_p_value"] = oos_p
        row["oos_separation"] = oos_sep
        row["oos_stability"] = oos_stability

        elapsed = time.time() - t0
        print(f"  BIC={row['bic']:.0f}  AIC={row['aic']:.0f}  Quality={row['quality_score']:.0f}  OOS p={oos_p:.4f}  ({elapsed:.1f}s)")

        results.append(row)

    df = pd.DataFrame(results)

    # Determine best
    if "error" in df.columns:
        valid = df[df["error"].isna()].copy() if "error" in df.columns else df.copy()
    else:
        valid = df.copy()

    if len(valid) > 0:
        # Candidates: OOS p-value < threshold
        candidates = valid[valid["oos_p_value"] < min_oos_p]
        if len(candidates) > 0:
            best_idx = candidates["bic"].idxmin()
        else:
            # No candidate passes OOS threshold — pick lowest BIC anyway but warn
            best_idx = valid["bic"].idxmin()
            print(f"\n  WARNING: No model passed OOS p < {min_oos_p}. Picking best BIC anyway.")

        best_n = int(valid.loc[best_idx, "n_regimes"])
        df["verdict"] = ""
        df.loc[best_idx, "verdict"] = "BEST"

        current_idx = df[df["n_regimes"] == DEFAULT_N_REGIMES].index
        if len(current_idx) > 0 and current_idx[0] != best_idx:
            df.loc[current_idx[0], "verdict"] = "CURRENT"
    else:
        best_n = DEFAULT_N_REGIMES

    # Print table
    _print_phase1_table(df)

    return df


def _abbreviated_walk_forward(
    features_wf: pd.DataFrame,
    ohlcv: pd.DataFrame,
    months,
    n_regimes: int,
    n_folds: int,
) -> tuple[float, float, float]:
    """
    Run abbreviated walk-forward: last n_folds months as OOS.
    Returns (oos_p_value, oos_mean_separation, transition_stability).
    """
    if len(months) < n_folds + 3:
        return 1.0, 0.0, 0.0

    # Use last n_folds months as test, rest as expanding train
    start_idx = max(3, len(months) - n_folds)
    bullish_returns = []
    bearish_returns = []
    transition_matrices = []

    for i in range(start_idx, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_mask = features_wf["year_month"].isin(train_months)
        test_mask = features_wf["year_month"] == test_month

        train_data = features_wf[train_mask][HMM_FEATURES]
        test_data = features_wf[test_mask][HMM_FEATURES]

        if len(test_data) < 10:
            continue

        detector = RegimeDetector(n_regimes=n_regimes, n_restarts=3, n_iter=100)
        try:
            detector.fit(train_data, feature_cols=HMM_FEATURES)
        except Exception:
            continue

        test_preds = detector.predict(test_data)
        transition_matrices.append(detector.model.transmat_.copy())

        # OOS returns by signal
        test_prices = ohlcv.loc[test_data.index, "Close"]
        returns = test_prices.pct_change()

        for regime_id in range(n_regimes):
            mask = test_preds["regime"] == regime_id
            if mask.sum() == 0:
                continue
            signal = test_preds.loc[mask, "signal"].iloc[0]
            r_mean = returns[mask].mean()
            if np.isnan(r_mean):
                continue
            count = mask.sum()
            if signal == "bullish":
                bullish_returns.extend([r_mean * 100] * count)
            elif signal == "bearish":
                bearish_returns.extend([r_mean * 100] * count)

    # OOS p-value
    bullish_returns = [x for x in bullish_returns if not np.isnan(x)]
    bearish_returns = [x for x in bearish_returns if not np.isnan(x)]

    if bullish_returns and bearish_returns:
        _, p_val = stats.ttest_ind(bullish_returns, bearish_returns, equal_var=False)
        separation = np.mean(bullish_returns) - np.mean(bearish_returns)
    elif bullish_returns:
        p_val = 0.5
        separation = np.mean(bullish_returns)
    else:
        p_val = 1.0
        separation = 0.0

    # Transition stability
    stability = 0.0
    if len(transition_matrices) >= 2:
        diffs = []
        for j in range(1, len(transition_matrices)):
            diff = np.linalg.norm(transition_matrices[j] - transition_matrices[j - 1], "fro")
            diffs.append(diff)
        stability = float(np.mean(diffs))

    return float(p_val), float(separation), stability


def _print_phase1_table(df: pd.DataFrame):
    """Pretty-print the n_regimes comparison table."""
    print("\n" + "=" * 100)
    print("n_REGIMES COMPARISON")
    print("=" * 100)

    cols = ["n_regimes", "bic", "aic", "quality_score", "oos_p_value", "oos_separation", "oos_stability", "verdict"]
    available = [c for c in cols if c in df.columns]

    header = f"{'n':>3} | {'BIC':>10} | {'AIC':>10} | {'Quality':>7} | {'OOS p':>8} | {'OOS Sep':>8} | {'Stability':>9} | {'Verdict':>8}"
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        if "error" in df.columns and pd.notna(row.get("error")):
            print(f"{int(row['n_regimes']):>3} | {'FAILED':>10} |")
            continue
        line = (
            f"{int(row['n_regimes']):>3} | "
            f"{row.get('bic', 0):>10.0f} | "
            f"{row.get('aic', 0):>10.0f} | "
            f"{row.get('quality_score', 0):>7.0f} | "
            f"{row.get('oos_p_value', 1):>8.4f} | "
            f"{row.get('oos_separation', 0):>8.4f} | "
            f"{row.get('oos_stability', 0):>9.4f} | "
            f"{row.get('verdict', ''):>8}"
        )
        print(line)

    # Summary
    best_rows = df[df.get("verdict", pd.Series(dtype=str)) == "BEST"]
    if len(best_rows) > 0:
        best = best_rows.iloc[0]
        print(f"\nRECOMMENDATION: {int(best['n_regimes'])} states (BIC={best['bic']:.0f}, OOS p={best['oos_p_value']:.4f})")
        if int(best['n_regimes']) != DEFAULT_N_REGIMES:
            print(f"  Current default is {DEFAULT_N_REGIMES} states — consider updating.")
        else:
            print(f"  Current default ({DEFAULT_N_REGIMES}) is already optimal.")


# ============================================================
# Phase 2: Confirmation sensitivity (leave-one-out)
# ============================================================

def phase2_confirmation_sensitivity(
    ohlcv: pd.DataFrame,
    hmm_features: pd.DataFrame,
    confirmations: pd.DataFrame,
    n_regimes: int = DEFAULT_N_REGIMES,
) -> pd.DataFrame:
    """
    For each of 13 confirmations: remove it, recompute filter value.
    Also test min_confirmations from 6-12.

    Measures impact on filter_value Sharpe — not curve-fitting thresholds.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Confirmation Sensitivity (Leave-One-Out)")
    print("=" * 70)

    # Fit model once
    features = hmm_features.dropna()
    detector = RegimeDetector(n_regimes=n_regimes, n_restarts=5, n_iter=100)
    detector.fit(features, feature_cols=HMM_FEATURES)
    preds = detector.predict(features)

    # Baseline quality
    analyzer = RegimeQualityAnalyzer()
    baseline = analyzer.analyze(ohlcv, preds)
    baseline_sharpe = baseline.filter_value.get("filtered_sharpe", 0)
    baseline_score = baseline.summary_score

    print(f"\nBaseline: Quality={baseline_score:.0f}, Filter Sharpe={baseline_sharpe:.3f}")

    # All confirmation pass columns
    tech_confs = list(CONFIRMATIONS.keys())  # 8 technical
    macro_confs = list(MACRO_CONFIRMATIONS.keys())  # 5 macro
    all_conf_names = tech_confs + macro_confs

    # Map confirmation name → pass column name
    pass_col_map = {}
    for name in tech_confs:
        pass_col_map[name] = f"{name}_pass"
    # Macro pass columns use different naming
    pass_col_map["vix_term_structure"] = "vix_term_structure_pass"
    pass_col_map["credit_spread"] = "credit_spread_pass"
    pass_col_map["yield_curve"] = "yield_curve_pass"
    pass_col_map["market_breadth"] = "breadth_pass"
    pass_col_map["dollar_strength"] = "dollar_pass"

    # Leave-one-out analysis
    results = []
    for conf_name in all_conf_names:
        pass_col = pass_col_map.get(conf_name)
        if pass_col is None or pass_col not in confirmations.columns:
            continue

        # Check how often this confirmation passes
        col_data = confirmations[pass_col].dropna()
        if len(col_data) == 0:
            continue
        pass_rate = col_data.mean() * 100

        # Simulate removing this confirmation:
        # If it always passes or always fails, removing it changes nothing
        # We measure the impact on which bars pass the overall gate

        # Count confirmations WITHOUT this one
        all_pass_cols = [c for c in confirmations.columns if c.endswith("_pass")]
        other_pass_cols = [c for c in all_pass_cols if c != pass_col]

        if not other_pass_cols:
            continue

        total_without = confirmations[other_pass_cols].sum(axis=1)
        total_with = confirmations[all_pass_cols].sum(axis=1)

        # How many bars change gate status at various min thresholds?
        # Use current min_confirmations proportionally
        n_total = len(all_pass_cols)
        n_without = len(other_pass_cols)

        # Test at proportional threshold: e.g. 10/13 → 9/12
        min_with = round(10 / 13 * n_total)
        min_without = round(10 / 13 * n_without)

        bars_pass_with = (total_with >= min_with).sum()
        bars_pass_without = (total_without >= min_without).sum()

        results.append({
            "confirmation": conf_name,
            "type": "technical" if conf_name in tech_confs else "macro",
            "pass_rate": pass_rate,
            "bars_pass_with": bars_pass_with,
            "bars_pass_without": bars_pass_without,
            "delta_bars": bars_pass_without - bars_pass_with,
            "delta_pct": (bars_pass_without - bars_pass_with) / max(bars_pass_with, 1) * 100,
        })

    loo_df = pd.DataFrame(results)

    # Min confirmations sweep
    print("\n--- Min Confirmations Sweep ---")
    min_conf_results = []
    all_pass_cols = [c for c in confirmations.columns if c.endswith("_pass")]
    n_total = len(all_pass_cols)
    total_confs = confirmations[all_pass_cols].sum(axis=1)

    for min_c in range(max(1, n_total - 7), n_total + 1):
        gate_pass = total_confs >= min_c
        bars_in = gate_pass.sum()
        pct_in = bars_in / len(gate_pass) * 100

        # Compute filter Sharpe at this threshold
        prices = ohlcv.loc[confirmations.index, "Close"].reindex(confirmations.index)
        returns = prices.pct_change().dropna()
        aligned_signal = preds["signal"].reindex(returns.index).shift(1)
        aligned_gate = gate_pass.reindex(returns.index).shift(1).fillna(False).astype(bool)

        bullish_mask = (aligned_signal == "bullish") & aligned_gate
        filtered_returns = returns.where(bullish_mask, 0)
        if filtered_returns.std() > 0:
            sharpe = filtered_returns.mean() / filtered_returns.std() * np.sqrt(252 * 6.5)
        else:
            sharpe = 0.0

        min_conf_results.append({
            "min_confirmations": min_c,
            "bars_passing": bars_in,
            "pct_time_in": pct_in,
            "filter_sharpe": sharpe,
        })

    min_conf_df = pd.DataFrame(min_conf_results)

    # Print results
    _print_phase2_results(loo_df, min_conf_df, baseline_sharpe)

    return loo_df, min_conf_df


def _print_phase2_results(loo_df: pd.DataFrame, min_conf_df: pd.DataFrame, baseline_sharpe: float):
    """Print Phase 2 results."""
    print("\n" + "=" * 80)
    print("LEAVE-ONE-OUT: Confirmation Impact")
    print("=" * 80)

    header = f"{'Confirmation':<25} | {'Type':<10} | {'Pass%':>6} | {'Delta Bars':>10} | {'Delta%':>7}"
    print(header)
    print("-" * len(header))

    for _, row in loo_df.sort_values("delta_pct", key=abs, ascending=False).iterrows():
        print(
            f"{row['confirmation']:<25} | "
            f"{row['type']:<10} | "
            f"{row['pass_rate']:>5.1f}% | "
            f"{row['delta_bars']:>+10d} | "
            f"{row['delta_pct']:>+6.1f}%"
        )

    # Identify low-impact confirmations
    low_impact = loo_df[loo_df["delta_pct"].abs() < 1.0]
    if len(low_impact) > 0:
        print(f"\nLow-impact confirmations (|delta| < 1%): {', '.join(low_impact['confirmation'].tolist())}")
        print("  These can be removed without significantly changing the gate.")

    high_impact = loo_df[loo_df["delta_pct"].abs() >= 5.0]
    if len(high_impact) > 0:
        print(f"High-impact confirmations (|delta| >= 5%): {', '.join(high_impact['confirmation'].tolist())}")
        print("  These are the most influential gatekeepers.")

    print("\n" + "=" * 60)
    print("MIN CONFIRMATIONS SWEEP")
    print("=" * 60)

    header2 = f"{'Min':>4} | {'Bars In':>8} | {'% Time':>7} | {'Filter Sharpe':>13}"
    print(header2)
    print("-" * len(header2))
    for _, row in min_conf_df.iterrows():
        marker = " <-- BEST" if row["filter_sharpe"] == min_conf_df["filter_sharpe"].max() else ""
        print(
            f"{int(row['min_confirmations']):>4} | "
            f"{int(row['bars_passing']):>8} | "
            f"{row['pct_time_in']:>6.1f}% | "
            f"{row['filter_sharpe']:>13.3f}{marker}"
        )

    best_min = min_conf_df.loc[min_conf_df["filter_sharpe"].idxmax()]
    print(f"\nOptimal min_confirmations: {int(best_min['min_confirmations'])} (Sharpe={best_min['filter_sharpe']:.3f})")


# ============================================================
# Phase 3: Combined validation
# ============================================================

def phase3_combined_validation(
    ohlcv: pd.DataFrame,
    hmm_features: pd.DataFrame,
    best_n: int,
) -> dict:
    """
    Run full walk-forward validation at the recommended n_regimes.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 3: Full Walk-Forward Validation (n_regimes={best_n})")
    print("=" * 70)

    from scripts.walk_forward import run_walk_forward, print_results

    # Re-use the walk_forward module with the best n
    results = run_walk_forward(
        n_regimes=best_n,
        min_train_months=6,
    )
    print_results(results)

    # Final comparison: best vs current default
    if best_n != DEFAULT_N_REGIMES:
        print(f"\n{'=' * 60}")
        print(f"COMPARISON: {best_n} states (recommended) vs {DEFAULT_N_REGIMES} states (current)")
        print(f"{'=' * 60}")
        print("Run walk-forward for current default too...")
        current_results = run_walk_forward(
            n_regimes=DEFAULT_N_REGIMES,
            min_train_months=6,
        )

        oos_best = results.get("oos_regime_test", {})
        oos_current = current_results.get("oos_regime_test", {})

        print(f"\n  {best_n} states: OOS p={oos_best.get('p_value', 1):.4f}, "
              f"bull={oos_best.get('bullish_mean', 0):.4f}%, "
              f"bear={oos_best.get('bearish_mean', 0):.4f}%")
        print(f"  {DEFAULT_N_REGIMES} states: OOS p={oos_current.get('p_value', 1):.4f}, "
              f"bull={oos_current.get('bullish_mean', 0):.4f}%, "
              f"bear={oos_current.get('bearish_mean', 0):.4f}%")

        if oos_best.get("p_value", 1) < oos_current.get("p_value", 1):
            print(f"\n  VERDICT: {best_n} states has better OOS separation. Consider switching.")
        else:
            print(f"\n  VERDICT: {DEFAULT_N_REGIMES} states holds up. Keep current default.")
    else:
        print(f"\nCurrent default ({DEFAULT_N_REGIMES}) is already optimal by BIC.")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Medallion 2.0 — HMM Optimizer")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run specific phase (default: all)")
    parser.add_argument("--n-regimes", type=int, help="Force specific n_regimes for phases 2-3")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--period", default="730d")
    args = parser.parse_args()

    print("=" * 70)
    print("MEDALLION 2.0 — HMM OPTIMIZER")
    print("=" * 70)

    # Load data
    print(f"\nLoading {args.ticker} {args.interval} data ({args.period})...")
    ohlcv, hmm_features, confirmations = load_data(
        ticker=args.ticker,
        interval=args.interval,
        period=args.period,
        cache=True,
        include_macro=False,
    )
    print(f"Loaded {len(ohlcv):,} bars, {len(hmm_features.dropna()):,} valid feature bars")

    best_n = args.n_regimes or DEFAULT_N_REGIMES
    phase1_results = None

    # Phase 1
    if args.phase is None or args.phase == 1:
        phase1_results = phase1_n_regimes_sweep(ohlcv, hmm_features)

        # Extract best n from phase 1
        if "verdict" in phase1_results.columns:
            best_rows = phase1_results[phase1_results["verdict"] == "BEST"]
            if len(best_rows) > 0:
                best_n = int(best_rows.iloc[0]["n_regimes"])

        if args.n_regimes:
            best_n = args.n_regimes
            print(f"\n(Overriding best with --n-regimes {best_n})")

    # Phase 2
    if args.phase is None or args.phase == 2:
        phase2_confirmation_sensitivity(ohlcv, hmm_features, confirmations, n_regimes=best_n)

    # Phase 3
    if args.phase is None or args.phase == 3:
        phase3_combined_validation(ohlcv, hmm_features, best_n)

    # Final summary
    if args.phase is None:
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Recommended n_regimes: {best_n}")
        if phase1_results is not None and "verdict" in phase1_results.columns:
            best_row = phase1_results[phase1_results["verdict"] == "BEST"]
            if len(best_row) > 0:
                r = best_row.iloc[0]
                print(f"  BIC: {r['bic']:.0f}")
                print(f"  Quality Score: {r['quality_score']:.0f}")
                print(f"  OOS p-value: {r['oos_p_value']:.4f}")
        if best_n != DEFAULT_N_REGIMES:
            print(f"\nTo apply: update DEFAULT_N_REGIMES in config/settings.py to {best_n}")
            print(f"Or use the 'Apply Best' button in the dashboard Model Selection tab.")


if __name__ == "__main__":
    main()
