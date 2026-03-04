#!/usr/bin/env python3
"""Walk-Forward Regime Timeline — Bias-Free HMM Labels for 5 Years.

Expanding window, monthly steps:
  - Train on months 1 through M-1, predict month M
  - Store regime_label + confidence + signal for every bar in month M
  - Advance to month M+1, repeat

Minimum 6 months training before first prediction.

This produces regime labels where each bar's label was computed by a model
that ONLY saw prior data — no look-ahead bias.

Output:
    data/processed/walk_forward_regimes.parquet
    One row per hourly bar, columns:
      - regime_label, regime_signal, regime_confidence
      - regime_id, prob_0..prob_6
      - fold (which training fold produced this bar's label)

Usage:
    cd ~/projects/ai-playground/medallion-2.0
    source .venv/bin/activate
    python scripts/walk_forward_regimes.py

    # Use 5-year resampled data (recommended):
    python scripts/walk_forward_regimes.py --use-5yr

    # Custom min training months:
    python scripts/walk_forward_regimes.py --use-5yr --min-train 12
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

MEDALLION_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(MEDALLION_ROOT))

from config.settings import HMM_FEATURES, DEFAULT_N_REGIMES
from models.hmm_regime import RegimeDetector
from data.data_loader import compute_hmm_features


def load_hourly_data(use_5yr: bool = True):
    """Load hourly data — 5yr resampled or 2yr yfinance."""
    if use_5yr:
        path = MEDALLION_ROOT / "data" / "processed" / "spy_1h_5yr.parquet"
        if not path.exists():
            print(f"ERROR: 5yr hourly data not found at {path}")
            print(f"  Run: python data/resample_spy.py")
            sys.exit(1)
        print(f"  Loading 5yr resampled data: {path.name}")
        ohlcv = pd.read_parquet(path)
    else:
        from data.data_loader import download_ohlcv
        print(f"  Downloading SPY 1h from yfinance (730d)...")
        ohlcv = download_ohlcv("SPY", "1h", "730d")

    # Compute HMM features
    hmm_features = compute_hmm_features(ohlcv)
    features = hmm_features.dropna().copy()

    print(f"  {len(features):,} bars with valid features")
    print(f"  Range: {features.index[0]} to {features.index[-1]}")

    return ohlcv, features


def build_walk_forward_timeline(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    n_regimes: int = DEFAULT_N_REGIMES,
    min_train_months: int = 6,
    n_restarts: int = 5,
    n_iter: int = 150,
):
    """Build walk-forward regime timeline with expanding window.

    Returns DataFrame with regime labels for every bar, where each bar's
    label was computed by a model trained ONLY on prior data.
    """
    # Split into monthly blocks
    if features.index.tz is not None:
        month_index = features.index.tz_localize(None).to_period("M")
    else:
        month_index = features.index.to_period("M")

    features = features.copy()
    features["year_month"] = month_index
    months = features["year_month"].unique()

    print(f"\n  Data spans {len(months)} months: {months[0]} to {months[-1]}")
    print(f"  Config: {n_regimes} states, {min_train_months} min train months")
    print(f"  Will produce {len(months) - min_train_months} folds of OOS predictions")

    if len(months) < min_train_months + 1:
        raise ValueError(f"Need at least {min_train_months + 1} months, got {len(months)}")

    # Collect OOS predictions from each fold
    all_predictions = []
    fold_info = []

    for i in range(min_train_months, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_mask = features["year_month"].isin(train_months)
        test_mask = features["year_month"] == test_month

        train_data = features[train_mask][HMM_FEATURES]
        test_data = features[test_mask][HMM_FEATURES]

        if len(test_data) < 5:
            continue

        fold_num = i - min_train_months + 1
        print(f"  Fold {fold_num:>3d}: Train {train_months[0]}-{train_months[-1]} "
              f"({len(train_data):,} bars) → Test {test_month} ({len(test_data)} bars)",
              end="")

        # Fit HMM on training data only
        detector = RegimeDetector(
            n_regimes=n_regimes,
            n_restarts=n_restarts,
            n_iter=n_iter,
        )
        try:
            detector.fit(train_data, feature_cols=HMM_FEATURES)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # Predict on OOS (test) data
        test_preds = detector.predict(test_data)

        # Store predictions with fold number
        test_preds["fold"] = fold_num
        all_predictions.append(test_preds)

        # Quick stats
        avg_conf = test_preds["confidence"].mean()
        changes = (test_preds["regime"].dropna().diff() != 0).sum() - 1
        print(f"  | conf={avg_conf:.1%} | changes={max(0, int(changes))}")

        fold_info.append({
            "fold": fold_num,
            "train_start": str(train_months[0]),
            "train_end": str(train_months[-1]),
            "test_month": str(test_month),
            "train_bars": len(train_data),
            "test_bars": len(test_data),
            "avg_confidence": float(avg_conf),
            "regime_changes": max(0, int(changes)),
        })

    if not all_predictions:
        raise RuntimeError("No folds produced valid predictions")

    # Concatenate all OOS predictions into one timeline
    timeline = pd.concat(all_predictions)

    # Remove duplicates (shouldn't happen with monthly non-overlapping folds)
    timeline = timeline[~timeline.index.duplicated(keep="last")]

    # Sort by timestamp
    timeline = timeline.sort_index()

    print(f"\n  Walk-forward timeline: {len(timeline):,} bars")
    print(f"  Date range: {timeline.index[0]} to {timeline.index[-1]}")
    print(f"  Folds completed: {len(fold_info)}")

    # Regime distribution
    dist = timeline["regime_label"].value_counts(normalize=True).sort_index()
    print(f"\n  Regime distribution (walk-forward):")
    for label, pct in dist.items():
        print(f"    {label:<25s} {pct:>6.1%}")

    return timeline, fold_info


def save_timeline(timeline: pd.DataFrame, fold_info: list):
    """Save walk-forward regime timeline to parquet."""
    out_path = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Select columns to save
    save_cols = ["regime", "regime_label", "confidence", "signal", "fold"]
    # Include probability columns
    prob_cols = [c for c in timeline.columns if c.startswith("prob_")]
    save_cols.extend(prob_cols)

    available = [c for c in save_cols if c in timeline.columns]
    timeline[available].to_parquet(out_path)

    print(f"\n  Saved timeline to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")

    # Also save fold info as JSON
    import json
    info_path = out_path.with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump({
            "n_folds": len(fold_info),
            "n_bars": len(timeline),
            "date_range": [str(timeline.index[0]), str(timeline.index[-1])],
            "folds": fold_info,
        }, f, indent=2)
    print(f"  Saved fold info to {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Regime Timeline")
    parser.add_argument("--use-5yr", action="store_true", default=True,
                        help="Use 5yr resampled SPY data (default)")
    parser.add_argument("--use-2yr", action="store_true",
                        help="Use 2yr yfinance SPY data instead")
    parser.add_argument("--min-train", type=int, default=6,
                        help="Minimum training months (default: 6)")
    parser.add_argument("--n-regimes", type=int, default=DEFAULT_N_REGIMES,
                        help=f"Number of HMM states (default: {DEFAULT_N_REGIMES})")
    parser.add_argument("--n-restarts", type=int, default=5,
                        help="Random restarts per fold (default: 5)")
    args = parser.parse_args()

    use_5yr = not args.use_2yr

    print("=" * 70)
    print("  WALK-FORWARD REGIME TIMELINE (bias-free)")
    print(f"  Data: {'5yr resampled' if use_5yr else '2yr yfinance'}")
    print(f"  States: {args.n_regimes}, Min train: {args.min_train} months")
    print("=" * 70)

    ohlcv, features = load_hourly_data(use_5yr=use_5yr)

    timeline, fold_info = build_walk_forward_timeline(
        ohlcv, features,
        n_regimes=args.n_regimes,
        min_train_months=args.min_train,
        n_restarts=args.n_restarts,
    )

    save_timeline(timeline, fold_info)

    # Summary
    avg_conf = timeline["confidence"].mean()
    total_changes = (timeline["regime"].dropna().diff() != 0).sum()
    n_months = len(fold_info)
    changes_per_month = total_changes / n_months if n_months > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD TIMELINE COMPLETE")
    print(f"  {len(timeline):,} bars | {n_months} folds | avg conf {avg_conf:.1%}")
    print(f"  ~{changes_per_month:.1f} regime changes/month")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
