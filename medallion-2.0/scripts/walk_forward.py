#!/usr/bin/env python3
"""
Medallion 2.0 — Walk-Forward Validation

Expanding window: train on months 1-6 → predict month 7,
train 1-7 → predict 8, etc.

Measures:
  - Regime label consistency across folds
  - Transition matrix stability
  - Forward return by regime (OOS)
  - Regime persistence accuracy

Usage:
    python scripts/walk_forward.py
    python scripts/walk_forward.py --n-regimes 5 --min-train 6
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import HMM_FEATURES, DEFAULT_N_REGIMES
from models.hmm_regime import RegimeDetector
from data.data_loader import load_data


def run_walk_forward(
    n_regimes: int = DEFAULT_N_REGIMES,
    min_train_months: int = 6,
    step_months: int = 1,
    ticker: str = "SPY",
    interval: str = "1h",
    period: str = "730d",
) -> dict:
    """
    Run expanding-window walk-forward validation.

    Returns dict with fold results, consistency metrics, and summary.
    """
    print(f"Loading {ticker} {interval} data...")
    ohlcv, hmm_features, _ = load_data(ticker, interval, period, cache=True)
    features = hmm_features.dropna().copy()

    # Split into monthly blocks
    features["year_month"] = features.index.tz_localize(None).to_period("M") if features.index.tz else features.index.to_period("M")
    months = features["year_month"].unique()
    print(f"Data spans {len(months)} months: {months[0]} to {months[-1]}")
    print(f"Config: {n_regimes} states, {min_train_months} min train months, {step_months} step\n")

    if len(months) < min_train_months + 1:
        raise ValueError(f"Need at least {min_train_months + 1} months, got {len(months)}")

    fold_results = []
    transition_matrices = []
    regime_label_history = []

    # Expanding window
    for i in range(min_train_months, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_mask = features["year_month"].isin(train_months)
        test_mask = features["year_month"] == test_month

        train_data = features[train_mask][HMM_FEATURES]
        test_data = features[test_mask][HMM_FEATURES]

        if len(test_data) < 10:
            continue

        fold_num = i - min_train_months + 1
        print(f"Fold {fold_num}: Train {train_months[0]}-{train_months[-1]} ({len(train_data)} bars) → Test {test_month} ({len(test_data)} bars)")

        # Fit on training data (3 restarts is sufficient for validation — not optimizing)
        detector = RegimeDetector(n_regimes=n_regimes, n_restarts=3, n_iter=100)
        try:
            detector.fit(train_data, feature_cols=HMM_FEATURES)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # Predict on test data (OOS)
        test_preds = detector.predict(test_data)
        train_preds = detector.predict(train_data)

        # Store transition matrix
        transition_matrices.append(detector.model.transmat_.copy())

        # Store regime labels for consistency check
        regime_label_history.append({
            "fold": fold_num,
            "labels": dict(detector.regime_labels),
        })

        # Compute OOS forward returns by regime
        test_prices = ohlcv.loc[test_data.index, "Close"]
        returns = test_prices.pct_change()

        regime_returns = {}
        for regime_id in range(n_regimes):
            mask = test_preds["regime"] == regime_id
            if mask.sum() > 0:
                label = detector.regime_labels.get(regime_id, f"R{regime_id}")
                r = returns[mask]
                regime_returns[label] = {
                    "mean": r.mean() * 100,
                    "count": mask.sum(),
                    "signal": test_preds.loc[mask, "signal"].iloc[0] if mask.any() else "unknown",
                }

        # Regime persistence in test period
        regime_col = test_preds["regime"].dropna()
        changes = (regime_col != regime_col.shift(1)).sum() - 1
        avg_duration = len(regime_col) / max(changes, 1)

        fold_results.append({
            "fold": fold_num,
            "train_months": f"{train_months[0]}-{train_months[-1]}",
            "test_month": str(test_month),
            "train_bars": len(train_data),
            "test_bars": len(test_data),
            "regime_returns": regime_returns,
            "regime_changes": max(0, int(changes)),
            "avg_duration": avg_duration,
            "avg_confidence": test_preds["confidence"].mean(),
        })

    # === Aggregate Analysis ===
    summary = _compute_summary(fold_results, transition_matrices, regime_label_history, n_regimes)
    summary["folds"] = fold_results

    return summary


def _compute_summary(fold_results, transition_matrices, label_history, n_regimes):
    """Compute aggregate metrics across all folds."""
    summary = {}

    # 1. Transition matrix stability (Frobenius norm of differences)
    # Threshold scales with n_regimes: 7x7 matrix has more entries than 2x2
    stability_threshold = 0.3 * np.sqrt(n_regimes)
    if len(transition_matrices) >= 2:
        diffs = []
        for i in range(1, len(transition_matrices)):
            diff = np.linalg.norm(transition_matrices[i] - transition_matrices[i - 1], "fro")
            diffs.append(diff)
        summary["transition_stability"] = {
            "mean_diff": float(np.mean(diffs)),
            "std_diff": float(np.std(diffs)),
            "max_diff": float(np.max(diffs)),
            "threshold": float(stability_threshold),
            "stable": float(np.mean(diffs)) < stability_threshold,
        }
    else:
        summary["transition_stability"] = {"mean_diff": 0, "stable": True}

    # 2. Regime label consistency (do same label names appear each fold?)
    if label_history:
        all_labels = set()
        for entry in label_history:
            all_labels.update(entry["labels"].values())
        summary["label_consistency"] = {
            "unique_labels": sorted(all_labels),
            "n_unique": len(all_labels),
            "expected": n_regimes,
            "consistent": len(all_labels) == n_regimes,
        }

    # 3. OOS forward returns — do bullish regimes actually produce positive returns?
    bullish_returns = []
    bearish_returns = []
    for fold in fold_results:
        for label, data in fold["regime_returns"].items():
            mean_val = data["mean"]
            if np.isnan(mean_val):
                continue
            if data["signal"] == "bullish":
                bullish_returns.extend([mean_val] * data["count"])
            elif data["signal"] == "bearish":
                bearish_returns.extend([mean_val] * data["count"])

    # Filter any remaining NaN
    bullish_returns = [x for x in bullish_returns if not np.isnan(x)]
    bearish_returns = [x for x in bearish_returns if not np.isnan(x)]

    if bullish_returns and bearish_returns:
        t_stat, p_val = stats.ttest_ind(bullish_returns, bearish_returns, equal_var=False)
        summary["oos_regime_test"] = {
            "bullish_mean": float(np.mean(bullish_returns)),
            "bearish_mean": float(np.mean(bearish_returns)),
            "bullish_n": len(bullish_returns),
            "bearish_n": len(bearish_returns),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
            "direction_correct": np.mean(bullish_returns) > np.mean(bearish_returns),
        }
    elif bullish_returns:
        summary["oos_regime_test"] = {
            "bullish_mean": float(np.mean(bullish_returns)),
            "bullish_n": len(bullish_returns),
            "bearish_n": 0,
            "significant": False,
            "note": "No bearish regime bars in OOS (strong bull market?)",
        }
    else:
        summary["oos_regime_test"] = {"significant": False, "note": "insufficient data"}

    # 4. Average metrics
    if fold_results:
        summary["avg_confidence"] = float(np.mean([f["avg_confidence"] for f in fold_results]))
        summary["avg_duration"] = float(np.mean([f["avg_duration"] for f in fold_results]))
        summary["avg_changes_per_month"] = float(np.mean([f["regime_changes"] for f in fold_results]))

    return summary


def print_results(summary: dict):
    """Print walk-forward results to console."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)

    folds = summary.get("folds", [])
    print(f"\nFolds: {len(folds)}")

    # Per-fold summary
    print(f"\n{'Fold':<6} {'Test Month':<12} {'Bars':<8} {'Changes':<10} {'Avg Dur':<10} {'Confidence':<12}")
    print("-" * 58)
    for f in folds:
        print(f"{f['fold']:<6} {f['test_month']:<12} {f['test_bars']:<8} {f['regime_changes']:<10} {f['avg_duration']:<10.1f} {f['avg_confidence']:<12.1%}")

    # Transition stability
    ts = summary.get("transition_stability", {})
    print(f"\nTransition Matrix Stability:")
    print(f"  Mean Frobenius diff: {ts.get('mean_diff', 0):.4f}")
    print(f"  Stable: {'YES' if ts.get('stable') else 'NO'}")

    # OOS regime test
    oos = summary.get("oos_regime_test", {})
    print(f"\nOOS Regime Separation:")
    if "t_stat" in oos:
        print(f"  Bullish mean return: {oos['bullish_mean']:.4f}% ({oos['bullish_n']} bars)")
        print(f"  Bearish mean return: {oos['bearish_mean']:.4f}% ({oos['bearish_n']} bars)")
        print(f"  t-stat: {oos['t_stat']:.3f}, p-value: {oos['p_value']:.4f}")
        print(f"  Significant: {'YES' if oos['significant'] else 'NO'}")
        print(f"  Direction correct: {'YES' if oos['direction_correct'] else 'NO'}")
    elif "bullish_mean" in oos:
        print(f"  Bullish mean return: {oos['bullish_mean']:.4f}% ({oos.get('bullish_n', '?')} bars)")
        print(f"  Bearish bars: {oos.get('bearish_n', 0)}")
        print(f"  {oos.get('note', '')}")
    else:
        print(f"  {oos.get('note', 'N/A')}")

    # Overall
    print(f"\nAverage confidence: {summary.get('avg_confidence', 0):.1%}")
    print(f"Average regime duration: {summary.get('avg_duration', 0):.1f} bars")
    print(f"Average changes/month: {summary.get('avg_changes_per_month', 0):.1f}")


def save_plots(summary: dict, output_path: str | Path = None):
    """Save walk-forward result plots."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "research" / "walk_forward_results.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    folds = summary.get("folds", [])
    if not folds:
        print("No folds to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Walk-Forward Validation Results", fontsize=14, fontweight="bold")

    # 1. Confidence over folds
    ax = axes[0, 0]
    fold_nums = [f["fold"] for f in folds]
    confs = [f["avg_confidence"] for f in folds]
    ax.bar(fold_nums, confs, color="#3498db", alpha=0.7)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Avg Confidence")
    ax.set_title("OOS Confidence by Fold")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()

    # 2. Regime changes per fold
    ax = axes[0, 1]
    changes = [f["regime_changes"] for f in folds]
    ax.bar(fold_nums, changes, color="#e74c3c", alpha=0.7)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Regime Changes")
    ax.set_title("Regime Switches per OOS Month")

    # 3. Regime duration
    ax = axes[1, 0]
    durations = [f["avg_duration"] for f in folds]
    ax.plot(fold_nums, durations, "o-", color="#2ecc71")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Avg Duration (bars)")
    ax.set_title("OOS Regime Persistence")

    # 4. OOS returns by signal
    ax = axes[1, 1]
    bullish_by_fold = []
    bearish_by_fold = []
    for f in folds:
        b_rets = [d["mean"] for d in f["regime_returns"].values() if d["signal"] == "bullish"]
        r_rets = [d["mean"] for d in f["regime_returns"].values() if d["signal"] == "bearish"]
        bullish_by_fold.append(np.mean(b_rets) if b_rets else 0)
        bearish_by_fold.append(np.mean(r_rets) if r_rets else 0)

    x = np.arange(len(fold_nums))
    width = 0.35
    ax.bar(x - width / 2, bullish_by_fold, width, label="Bullish", color="#2ecc71", alpha=0.7)
    ax.bar(x + width / 2, bearish_by_fold, width, label="Bearish", color="#e74c3c", alpha=0.7)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Mean Return (%)")
    ax.set_title("OOS Returns by Regime Signal")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_nums)
    ax.legend()
    ax.axhline(y=0, color="white", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    print(f"\nPlots saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward validation for HMM regime detector")
    parser.add_argument("--n-regimes", type=int, default=DEFAULT_N_REGIMES)
    parser.add_argument("--min-train", type=int, default=6, help="Minimum training months")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--period", default="730d")
    args = parser.parse_args()

    results = run_walk_forward(
        n_regimes=args.n_regimes,
        min_train_months=args.min_train,
        ticker=args.ticker,
        interval=args.interval,
        period=args.period,
    )
    print_results(results)
    save_plots(results)
