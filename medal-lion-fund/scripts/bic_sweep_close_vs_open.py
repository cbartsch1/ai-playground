#!/usr/bin/env python3
"""
BIC sweep for optimal n_states with the expanded 5-feature set.

Tests n_states = 5, 6, 7, 8 with 10 random restarts, full covariance.
For each: BIC, log-likelihood, quality score, state population, regime separation.
Then runs walk-forward validation for the top 2 candidates by BIC.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import time

from config.settings import HMM_FEATURES, BULLISH_REGIMES, BEARISH_REGIMES
from models.hmm_regime import RegimeDetector
from data.data_loader import load_data
from backtester.regime_quality import RegimeQualityAnalyzer


def compute_bic(model, X_scaled, n_states, n_features):
    """Compute BIC for a fitted HMM."""
    n_samples = len(X_scaled)
    log_likelihood = model.score(X_scaled) * n_samples
    n_params = (
        n_states * n_features                              # means
        + n_states * n_features * (n_features + 1) / 2     # covariances (full)
        + n_states * (n_states - 1)                        # transitions
    )
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic, log_likelihood, int(n_params)


def count_alive_states(model, X_scaled, n_states, threshold_pct=1.0):
    """Count states with >= threshold_pct of total bars."""
    states = model.predict(X_scaled)
    alive = 0
    state_pcts = {}
    for s in range(n_states):
        pct = (states == s).sum() / len(states) * 100
        state_pcts[s] = pct
        if pct >= threshold_pct:
            alive += 1
    return alive, state_pcts


def compute_regime_separation(detector, features, ohlcv):
    """Get mean return for bullish vs bearish regimes."""
    preds = detector.predict(features)
    returns = ohlcv.loc[features.index, "Close"].pct_change()

    aligned_signal = preds["signal"].reindex(returns.index)
    bull_mask = aligned_signal == "bullish"
    bear_mask = aligned_signal == "bearish"

    bull_ret = returns[bull_mask].mean() * 100 if bull_mask.sum() > 0 else 0.0
    bear_ret = returns[bear_mask].mean() * 100 if bear_mask.sum() > 0 else 0.0

    return bull_ret, bear_ret, bull_ret - bear_ret


def walk_forward_validation(ohlcv, hmm_features, n_regimes, min_train_months=6):
    """
    Run expanding-window walk-forward validation.
    Returns dict with OOS p-value, separation, quality, and per-fold results.
    """
    features = hmm_features.dropna().copy()

    if features.index.tz is not None:
        features["year_month"] = features.index.tz_localize(None).to_period("M")
    else:
        features["year_month"] = features.index.to_period("M")
    months = features["year_month"].unique()

    if len(months) < min_train_months + 1:
        return {"oos_p_value": 1.0, "oos_separation": 0.0, "error": "insufficient data"}

    bullish_returns = []
    bearish_returns = []
    fold_results = []

    for i in range(min_train_months, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_mask = features["year_month"].isin(train_months)
        test_mask = features["year_month"] == test_month

        train_data = features[train_mask][HMM_FEATURES]
        test_data = features[test_mask][HMM_FEATURES]

        if len(test_data) < 10:
            continue

        detector = RegimeDetector(n_regimes=n_regimes, n_restarts=3, n_iter=100)
        try:
            detector.fit(train_data, feature_cols=HMM_FEATURES)
        except Exception:
            continue

        test_preds = detector.predict(test_data)

        # OOS returns by signal
        test_prices = ohlcv.loc[test_data.index, "Close"]
        returns = test_prices.pct_change()

        fold_bull = []
        fold_bear = []
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
                fold_bull.extend([r_mean * 100] * count)
            elif signal == "bearish":
                bearish_returns.extend([r_mean * 100] * count)
                fold_bear.extend([r_mean * 100] * count)

        fold_results.append({
            "fold": i - min_train_months + 1,
            "test_month": str(test_month),
            "bull_mean": np.mean(fold_bull) if fold_bull else None,
            "bear_mean": np.mean(fold_bear) if fold_bear else None,
            "confidence": test_preds["confidence"].mean(),
        })

    # Aggregate
    bullish_returns = [x for x in bullish_returns if not np.isnan(x)]
    bearish_returns = [x for x in bearish_returns if not np.isnan(x)]

    if bullish_returns and bearish_returns:
        t_stat, p_val = stats.ttest_ind(bullish_returns, bearish_returns, equal_var=False)
        separation = np.mean(bullish_returns) - np.mean(bearish_returns)
    elif bullish_returns:
        p_val = 0.5
        separation = np.mean(bullish_returns)
        t_stat = 0.0
    else:
        p_val = 1.0
        separation = 0.0
        t_stat = 0.0

    avg_conf = np.mean([f["confidence"] for f in fold_results]) if fold_results else 0.0

    return {
        "oos_p_value": float(p_val),
        "oos_separation": float(separation),
        "oos_t_stat": float(t_stat),
        "n_folds": len(fold_results),
        "bull_mean": float(np.mean(bullish_returns)) if bullish_returns else 0.0,
        "bear_mean": float(np.mean(bearish_returns)) if bearish_returns else 0.0,
        "avg_confidence": float(avg_conf),
        "folds": fold_results,
    }


def main():
    print("=" * 70)
    print("BIC SWEEP — 5-Feature HMM (+ close_vs_open)")
    print("=" * 70)

    # Load data
    print(f"\nLoading SPY 1h data...")
    ohlcv, hmm_features, _ = load_data(
        ticker="SPY", interval="1h", period="730d", cache=True
    )
    features = hmm_features.dropna().copy()
    print(f"Loaded {len(ohlcv):,} bars, {len(features):,} valid feature bars")
    print(f"Features: {HMM_FEATURES}")
    print(f"Date range: {features.index[0]} to {features.index[-1]}")

    # Verify close_vs_open is present
    assert "close_vs_open" in features.columns, "close_vs_open not in features!"
    print(f"\nclose_vs_open stats: min={features['close_vs_open'].min():.6f}, "
          f"max={features['close_vs_open'].max():.6f}, "
          f"mean={features['close_vs_open'].mean():.6f}, "
          f"NaN={features['close_vs_open'].isna().sum()}")

    X = features[HMM_FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)
    n_features = X_scaled.shape[1]

    # ============================================================
    # BIC Sweep: n_states = 5, 6, 7, 8
    # ============================================================
    print("\n" + "=" * 70)
    print("BIC SWEEP")
    print("=" * 70)

    sweep_results = []

    for n in [5, 6, 7, 8]:
        print(f"\n--- n_states = {n} ---")
        t0 = time.time()

        # Fit with 10 random restarts
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
            sweep_results.append({"n_states": n, "error": "fit_failed"})
            continue

        # BIC / log-likelihood
        bic, log_ll, n_params = compute_bic(best_model, X_scaled, n, n_features)

        # State population
        alive, state_pcts = count_alive_states(best_model, X_scaled, n)
        collapsed = n - alive

        # Quality score via RegimeDetector
        detector = RegimeDetector(n_regimes=n, n_restarts=10, n_iter=200)
        detector.fit(features, feature_cols=HMM_FEATURES)
        preds = detector.predict(features)

        analyzer = RegimeQualityAnalyzer()
        quality = analyzer.analyze(ohlcv, preds)
        quality_score = quality.summary_score

        # Regime separation
        bull_ret, bear_ret, separation = compute_regime_separation(detector, features, ohlcv)

        elapsed = time.time() - t0

        row = {
            "n_states": n,
            "bic": bic,
            "log_likelihood": log_ll,
            "n_params": n_params,
            "quality_score": quality_score,
            "states_alive": alive,
            "collapsed": collapsed,
            "bull_return": bull_ret,
            "bear_return": bear_ret,
            "separation": separation,
            "elapsed": elapsed,
        }
        sweep_results.append(row)

        # Print state details
        print(f"  BIC={bic:.0f}  LogL={log_ll:.0f}  Quality={quality_score:.0f}  "
              f"Alive={alive}/{n}  Sep={separation:.4f}%  ({elapsed:.1f}s)")

        # Print regime labels and populations
        for regime_id in range(n):
            label = detector.regime_labels.get(regime_id, f"Regime {regime_id}")
            pct = state_pcts.get(regime_id, 0)
            tag = " [COLLAPSED]" if pct < 1.0 else ""
            print(f"    {label}: {pct:.1f}%{tag}")

    # ============================================================
    # Rank by BIC
    # ============================================================
    df = pd.DataFrame([r for r in sweep_results if "error" not in r])
    df = df.sort_values("bic")

    print("\n" + "=" * 70)
    print("BIC SWEEP RESULTS (sorted by BIC)")
    print("=" * 70)
    header = f"{'n':>3} | {'BIC':>12} | {'LogL':>12} | {'Quality':>7} | {'Alive':>7} | {'Separation':>10} | {'Bull%':>7} | {'Bear%':>7}"
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        n = int(row["n_states"])
        alive_str = f"{int(row['states_alive'])}/{n}"
        print(f"{n:>3} | {row['bic']:>12.0f} | {row['log_likelihood']:>12.0f} | "
              f"{row['quality_score']:>7.0f} | {alive_str:>7} | {row['separation']:>10.4f} | "
              f"{row['bull_return']:>7.4f} | {row['bear_return']:>7.4f}")

    # ============================================================
    # Walk-forward for top 2 candidates
    # ============================================================
    top2 = df.head(2)
    wf_results = {}

    for _, row in top2.iterrows():
        n = int(row["n_states"])
        print(f"\n{'=' * 70}")
        print(f"WALK-FORWARD VALIDATION: n_states = {n}")
        print(f"{'=' * 70}")
        t0 = time.time()
        wf = walk_forward_validation(ohlcv, hmm_features, n_regimes=n)
        elapsed = time.time() - t0
        wf_results[n] = wf

        print(f"  Folds: {wf['n_folds']}")
        print(f"  OOS p-value: {wf['oos_p_value']:.6f}")
        print(f"  OOS separation: {wf['oos_separation']:.4f}%")
        print(f"  OOS bull mean: {wf['bull_mean']:.4f}%")
        print(f"  OOS bear mean: {wf['bear_mean']:.4f}%")
        print(f"  OOS t-stat: {wf['oos_t_stat']:.3f}")
        print(f"  Avg confidence: {wf['avg_confidence']:.1%}")
        print(f"  Elapsed: {elapsed:.1f}s")

    # ============================================================
    # Final Report
    # ============================================================
    print("\n\n")
    print("=" * 60)
    print("=== REGIME MODEL OPTIMIZATION RESULTS ===")
    print("=" * 60)
    print(f"Feature added: close_vs_open")
    print(f"Previous features: returns, range, volume_vol, close_in_range")
    print(f"New features: returns, range, volume_vol, close_in_range, close_vs_open")

    print(f"\nBIC SWEEP:")
    for _, row in df.iterrows():
        n = int(row["n_states"])
        alive = int(row["states_alive"])
        print(f"  n={n}: BIC={row['bic']:.0f}, states_alive={alive}/{n}, "
              f"separation={row['separation']:.4f}%, quality={row['quality_score']:.0f}")

    # Determine winner
    # Criteria: must have all states alive (no collapsed), then lowest BIC
    all_alive = df[df["states_alive"] == df["n_states"]]
    if len(all_alive) > 0:
        best_n = int(all_alive.iloc[0]["n_states"])
        reason = f"lowest BIC with all states alive"
    else:
        best_n = int(df.iloc[0]["n_states"])
        reason = f"lowest BIC (some states collapsed in all candidates)"

    # Also check: if the best BIC has collapsed states but runner-up has all alive, prefer runner-up
    best_row = df.iloc[0]
    if int(best_row["states_alive"]) < int(best_row["n_states"]) and len(all_alive) > 0:
        best_n = int(all_alive.iloc[0]["n_states"])
        reason = f"all states populated (lowest BIC candidate n={int(best_row['n_states'])} has collapsed states)"

    print(f"\nOPTIMAL: n={best_n} ({reason})")

    # Print walk-forward results
    for n, wf in wf_results.items():
        label = "WINNER" if n == best_n else "runner-up"
        print(f"\nWALK-FORWARD (n={n}) [{label}]:")
        print(f"  OOS p-value: {wf['oos_p_value']:.6f}")
        print(f"  OOS separation: {wf['oos_separation']:.4f}%")
        print(f"  OOS bull mean: {wf['bull_mean']:.4f}%")
        print(f"  OOS bear mean: {wf['bear_mean']:.4f}%")
        print(f"  Avg confidence: {wf['avg_confidence']:.1%}")

    # Return best_n for Part 3
    return best_n, df, wf_results


if __name__ == "__main__":
    best_n, sweep_df, wf_results = main()

    # Write result to a temp file for consumption by Part 3
    result_path = Path(__file__).parent.parent / "data" / "processed" / "bic_sweep_result.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        f.write(str(best_n))
    print(f"\nBest n_states={best_n} written to {result_path}")
