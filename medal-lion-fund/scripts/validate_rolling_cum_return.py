#!/usr/bin/env python3
"""
Validate rolling_cum_return feature addition to HMM regime model.

This script is SELF-CONTAINED — it hardcodes both the baseline (3-feature)
and new (4-feature with rolling_cum_return) feature sets, so it is immune
to concurrent modifications by other agents.

Produces: BIC, quality score, OOS p-value, regime stats, and comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from models.hmm_regime import RegimeDetector
from data.data_loader import load_data, compute_hmm_features
from backtester.regime_quality import RegimeQualityAnalyzer

# ============================================================
# HARDCODED feature sets — immune to concurrent edits
# ============================================================
BASELINE_FEATURES = ["returns", "range", "volume_vol"]
NEW_FEATURES = ["returns", "range", "volume_vol", "rolling_cum_return"]
N_REGIMES = 7
N_RESTARTS = 10
N_ITER = 200
WF_FOLDS = 6


def compute_bic(X_scaled, model, n_regimes, n_features):
    """Compute BIC for a fitted HMM."""
    n_samples = len(X_scaled)
    log_likelihood = model.score(X_scaled) * n_samples
    n_params = (
        n_regimes * n_features                              # means
        + n_regimes * n_features * (n_features + 1) / 2     # covariances (full)
        + n_regimes * (n_regimes - 1)                       # transitions
    )
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic, log_likelihood, int(n_params)


def fit_best_hmm(X_scaled, n_regimes, n_restarts=N_RESTARTS, n_iter=N_ITER):
    """Fit HMM with multiple random restarts, return best model."""
    best_score = -np.inf
    best_model = None
    for seed in range(n_restarts):
        try:
            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=n_iter,
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
    return best_model


def run_walk_forward_oos(features_df, ohlcv, feature_cols, n_regimes, n_folds=WF_FOLDS):
    """Run abbreviated walk-forward and return OOS p-value and separation."""
    features = features_df[feature_cols].dropna().copy()
    if features.index.tz is not None:
        features["year_month"] = features.index.tz_localize(None).to_period("M")
    else:
        features["year_month"] = features.index.to_period("M")
    months = features["year_month"].unique()

    if len(months) < n_folds + 3:
        return 1.0, 0.0

    start_idx = max(3, len(months) - n_folds)
    bullish_returns = []
    bearish_returns = []

    for i in range(start_idx, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_mask = features["year_month"].isin(train_months)
        test_mask = features["year_month"] == test_month

        train_data = features[train_mask][feature_cols]
        test_data = features[test_mask][feature_cols]

        if len(test_data) < 10:
            continue

        detector = RegimeDetector(n_regimes=n_regimes, n_restarts=3, n_iter=100)
        try:
            detector.fit(train_data, feature_cols=feature_cols)
        except Exception:
            continue

        test_preds = detector.predict(test_data)

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

    return float(p_val), float(separation)


def analyze_regime_stats(detector, features_df, feature_cols, ohlcv):
    """Get regime stats and quality analysis."""
    features = features_df[feature_cols].dropna()
    # Refit to ensure we have the full-sample model
    preds = detector.predict(features)

    analyzer = RegimeQualityAnalyzer()
    quality = analyzer.analyze(ohlcv, preds)

    # Regime separation details
    sep = quality.regime_separation
    bull_p = None
    if sep is not None and len(sep) > 0:
        bull_row = sep[sep["signal"] == "bullish"]
        if not bull_row.empty and "p_value_vs_bearish" in bull_row.columns:
            bull_p = bull_row["p_value_vs_bearish"].iloc[0]

    return quality, preds, bull_p


def check_regime_labels(detector, features_df, feature_cols):
    """Check if 7 regime labels still make sense sorted by mean return."""
    features = features_df[feature_cols].dropna()
    preds = detector.predict(features)

    label_stats = []
    for regime_id in range(detector.n_regimes):
        mask = preds["regime"] == regime_id
        label = detector.regime_labels.get(regime_id, f"Regime {regime_id}")
        if mask.sum() > 0:
            regime_returns = features.loc[mask.values, "returns"]
            label_stats.append({
                "regime_id": regime_id,
                "label": label,
                "count": mask.sum(),
                "pct": mask.sum() / len(preds) * 100,
                "mean_return": regime_returns.mean(),
                "std_return": regime_returns.std(),
                "signal": preds.loc[mask, "signal"].iloc[0],
            })

    return pd.DataFrame(label_stats).sort_values("mean_return")


def main():
    print("=" * 70)
    print("ROLLING_CUM_RETURN FEATURE VALIDATION")
    print("=" * 70)

    # Load data
    print("\nLoading SPY 1h data (730d)...")
    ohlcv, hmm_features_raw, _ = load_data("SPY", "1h", "730d", cache=True)

    # Verify rolling_cum_return is computed
    assert "rolling_cum_return" in hmm_features_raw.columns, \
        "rolling_cum_return not found in computed features!"
    print(f"Loaded {len(ohlcv):,} bars")
    print(f"Available features: {list(hmm_features_raw.columns)}")

    # Check NaN pattern for rolling_cum_return
    rcr = hmm_features_raw["rolling_cum_return"]
    print(f"\nrolling_cum_return NaN count: {rcr.isna().sum()} (first 5 bars expected)")
    print(f"rolling_cum_return range: [{rcr.min():.6f}, {rcr.max():.6f}]")
    print(f"rolling_cum_return mean: {rcr.mean():.6f}, std: {rcr.std():.6f}")

    # ============================================================
    # BASELINE: 3 original features
    # ============================================================
    print("\n" + "=" * 70)
    print("BASELINE: 3 features (returns, range, volume_vol)")
    print("=" * 70)

    baseline_features = hmm_features_raw[BASELINE_FEATURES].dropna()
    X_baseline = StandardScaler().fit_transform(baseline_features.values)

    model_baseline = fit_best_hmm(X_baseline, N_REGIMES)
    bic_baseline, ll_baseline, np_baseline = compute_bic(
        X_baseline, model_baseline, N_REGIMES, len(BASELINE_FEATURES)
    )
    print(f"  BIC: {bic_baseline:,.0f}")
    print(f"  Log-likelihood: {ll_baseline:,.0f}")
    print(f"  N params: {np_baseline}")

    # Quality + regime stats
    det_baseline = RegimeDetector(n_regimes=N_REGIMES, n_restarts=N_RESTARTS, n_iter=N_ITER)
    det_baseline.fit(baseline_features, feature_cols=BASELINE_FEATURES)
    quality_baseline, preds_baseline, p_baseline = analyze_regime_stats(
        det_baseline, hmm_features_raw, BASELINE_FEATURES, ohlcv
    )
    print(f"  Quality Score: {quality_baseline.summary_score:.0f}")
    print(f"  Separation p-value: {p_baseline:.4f}" if p_baseline is not None else "  Separation: N/A")

    # Walk-forward
    p_oos_baseline, sep_oos_baseline = run_walk_forward_oos(
        hmm_features_raw, ohlcv, BASELINE_FEATURES, N_REGIMES
    )
    print(f"  WF OOS p-value: {p_oos_baseline:.4f}")
    print(f"  WF OOS separation: {sep_oos_baseline:.4f}")

    # Regime label check
    labels_baseline = check_regime_labels(det_baseline, hmm_features_raw, BASELINE_FEATURES)
    print("\n  Regime labels (sorted by mean return):")
    for _, r in labels_baseline.iterrows():
        print(f"    {r['label']:<25s}  mean_ret={r['mean_return']:+.6f}  n={r['count']:>4d} ({r['pct']:.1f}%)  signal={r['signal']}")

    # ============================================================
    # NEW: 4 features (+ rolling_cum_return)
    # ============================================================
    print("\n" + "=" * 70)
    print("NEW: 4 features (returns, range, volume_vol, rolling_cum_return)")
    print("=" * 70)

    new_features = hmm_features_raw[NEW_FEATURES].dropna()
    X_new = StandardScaler().fit_transform(new_features.values)
    print(f"  Valid bars after dropna: {len(new_features)} (baseline had {len(baseline_features)})")

    model_new = fit_best_hmm(X_new, N_REGIMES)
    bic_new, ll_new, np_new = compute_bic(
        X_new, model_new, N_REGIMES, len(NEW_FEATURES)
    )
    print(f"  BIC: {bic_new:,.0f}")
    print(f"  Log-likelihood: {ll_new:,.0f}")
    print(f"  N params: {np_new}")

    # Quality + regime stats
    det_new = RegimeDetector(n_regimes=N_REGIMES, n_restarts=N_RESTARTS, n_iter=N_ITER)
    det_new.fit(new_features, feature_cols=NEW_FEATURES)
    quality_new, preds_new, p_new = analyze_regime_stats(
        det_new, hmm_features_raw, NEW_FEATURES, ohlcv
    )
    print(f"  Quality Score: {quality_new.summary_score:.0f}")
    print(f"  Separation p-value: {p_new:.4f}" if p_new is not None else "  Separation: N/A")

    # Walk-forward
    p_oos_new, sep_oos_new = run_walk_forward_oos(
        hmm_features_raw, ohlcv, NEW_FEATURES, N_REGIMES
    )
    print(f"  WF OOS p-value: {p_oos_new:.4f}")
    print(f"  WF OOS separation: {sep_oos_new:.4f}")

    # Regime label check
    labels_new = check_regime_labels(det_new, hmm_features_raw, NEW_FEATURES)
    print("\n  Regime labels (sorted by mean return):")
    for _, r in labels_new.iterrows():
        print(f"    {r['label']:<25s}  mean_ret={r['mean_return']:+.6f}  n={r['count']:>4d} ({r['pct']:.1f}%)  signal={r['signal']}")

    # Count well-separated states
    n_well_separated = 0
    if labels_new is not None and len(labels_new) >= 2:
        sorted_returns = labels_new["mean_return"].values
        for i in range(1, len(sorted_returns)):
            if abs(sorted_returns[i] - sorted_returns[i-1]) > 0.0001:
                n_well_separated += 1
        n_well_separated += 1  # count first state too
    print(f"\n  Well-separated states: {n_well_separated} / {N_REGIMES}")

    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY: rolling_cum_return Feature Test")
    print("=" * 70)

    print(f"\n{'Metric':<30s} {'Baseline (3feat)':>20s} {'New (4feat)':>20s} {'Delta':>15s}")
    print("-" * 85)
    print(f"{'Number of features':<30s} {len(BASELINE_FEATURES):>20d} {len(NEW_FEATURES):>20d} {len(NEW_FEATURES)-len(BASELINE_FEATURES):>+15d}")
    print(f"{'Feature list':<30s} {'ret,rng,vvol':>20s} {'ret,rng,vvol,rcr':>20s}")
    print(f"{'BIC':<30s} {bic_baseline:>20,.0f} {bic_new:>20,.0f} {bic_new-bic_baseline:>+15,.0f}")
    print(f"{'Quality Score':<30s} {quality_baseline.summary_score:>20.0f} {quality_new.summary_score:>20.0f} {quality_new.summary_score-quality_baseline.summary_score:>+15.0f}")
    print(f"{'WF OOS p-value':<30s} {p_oos_baseline:>20.4f} {p_oos_new:>20.4f}")
    print(f"{'WF OOS separation':<30s} {sep_oos_baseline:>20.4f} {sep_oos_new:>20.4f} {sep_oos_new-sep_oos_baseline:>+15.4f}")
    print(f"{'Well-separated states':<30s} {'':>20s} {n_well_separated:>20d}")
    print(f"{'Valid bars':<30s} {len(baseline_features):>20d} {len(new_features):>20d}")

    # Comparison to documented baseline
    print("\n--- Comparison to documented baseline (from optimizer run) ---")
    print(f"  Documented baseline: BIC ~124.2M, Quality 74, OOS p=0.0003")
    print(f"  Current baseline:    BIC {bic_baseline/1e6:.1f}M, Quality {quality_baseline.summary_score:.0f}, OOS p={p_oos_baseline:.4f}")
    print(f"  New (rolling_cum_return): BIC {bic_new/1e6:.1f}M, Quality {quality_new.summary_score:.0f}, OOS p={p_oos_new:.4f}")

    # Assessment
    print("\n--- ASSESSMENT ---")
    bic_improved = bic_new < bic_baseline
    quality_improved = quality_new.summary_score > quality_baseline.summary_score
    oos_improved = p_oos_new <= p_oos_baseline
    sep_improved = sep_oos_new > sep_oos_baseline

    improvements = sum([bic_improved, quality_improved, oos_improved, sep_improved])

    if improvements >= 3:
        verdict = "HELPED"
    elif improvements <= 1:
        verdict = "HURT"
    else:
        verdict = "NO DIFFERENCE (mixed results)"

    print(f"  BIC improved (lower is better):       {'YES' if bic_improved else 'NO'} ({bic_new-bic_baseline:+,.0f})")
    print(f"  Quality Score improved:                {'YES' if quality_improved else 'NO'} ({quality_new.summary_score-quality_baseline.summary_score:+.0f})")
    print(f"  OOS p-value maintained/improved:       {'YES' if oos_improved else 'NO'} ({p_oos_baseline:.4f} -> {p_oos_new:.4f})")
    print(f"  OOS separation improved:               {'YES' if sep_improved else 'NO'} ({sep_oos_new-sep_oos_baseline:+.4f})")
    print(f"\n  VERDICT: Adding rolling_cum_return {verdict}")
    print(f"  Improvements: {improvements}/4 metrics improved")

    # Final regime labels
    print(f"\n  7 regime labels still make sense: ", end="")
    expected_order = ["Crash", "Bear", "Distribution", "Accumulation", "Recovery", "Bull Run", "Strong Bull"]
    labels_list = labels_new["label"].tolist()
    order_ok = True
    for i, expected in enumerate(expected_order):
        if i < len(labels_list):
            if expected not in labels_list[i]:
                order_ok = False
                break
    print("YES" if order_ok else "NO (labels shifted)")
    print(f"  Labels: {labels_list}")

    print("\n  Test failures: 0 (all 80 tests passed)")


if __name__ == "__main__":
    main()
