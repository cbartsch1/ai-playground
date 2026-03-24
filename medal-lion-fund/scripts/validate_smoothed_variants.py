#!/usr/bin/env python3
"""
Test smoothing variants to find the best directional accuracy.

Variants:
1. Both smoothed with window=5 (current)
2. Both smoothed with window=3
3. Only close_in_range smoothed (window=5), close_vs_open raw
4. Only close_vs_open smoothed (window=5), close_in_range raw
5. Both smoothed with window=3, close_vs_open raw

For each variant, fit 7-state HMM and measure directional accuracy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from models.hmm_regime import RegimeDetector
from data.data_loader import load_data, compute_hmm_features
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def compute_variant_features(ohlcv, cir_window, cvo_window):
    """
    Compute HMM features with specified smoothing windows.
    window=0 means raw (no smoothing).
    """
    df = ohlcv
    features = pd.DataFrame(index=df.index)

    # Log returns (unchanged)
    features["returns"] = np.log(df["Close"] / df["Close"].shift(1))

    # Range (unchanged)
    features["range"] = (df["High"] - df["Low"]) / df["Close"]

    # Volume volatility (unchanged)
    log_vol = np.log(df["Volume"].replace(0, np.nan))
    features["volume_vol"] = log_vol.rolling(20).std()

    # close_in_range
    bar_range = df["High"] - df["Low"]
    raw_cir = (df["Close"] - df["Low"]) / bar_range
    raw_cir = raw_cir.where(bar_range != 0, 0.5)
    if cir_window > 1:
        features["close_in_range"] = raw_cir.rolling(cir_window).mean()
    else:
        features["close_in_range"] = raw_cir

    # close_vs_open
    raw_cvo = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)
    if cvo_window > 1:
        features["close_vs_open"] = raw_cvo.rolling(cvo_window).mean()
    else:
        features["close_vs_open"] = raw_cvo

    return features


def get_daily_returns(ohlcv):
    close = ohlcv["Close"].copy()
    if close.index.tz is not None:
        dates = close.index.tz_localize(None).date
    else:
        dates = close.index.date
    daily = close.groupby(dates).last()
    daily_returns = daily.pct_change().dropna()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    return daily_returns


def measure_directional_accuracy(detector, features, ohlcv, feature_cols):
    """Measure bearish and bullish day classification accuracy."""
    preds = detector.predict(features)
    valid = preds.dropna(subset=["regime_label"])

    if valid.index.tz is not None:
        dates = valid.index.tz_localize(None).date
    else:
        dates = valid.index.date
    valid = valid.copy()
    valid["date"] = dates

    # Majority signal per day
    day_signals = valid.groupby("date")["signal"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else None
    )
    day_signals.index = pd.to_datetime(day_signals.index)

    # Daily returns
    daily_rets = get_daily_returns(ohlcv)
    common = day_signals.index.intersection(daily_rets.index)

    bearish_days = daily_rets.loc[common][daily_rets.loc[common] < -0.005]
    bullish_days = daily_rets.loc[common][daily_rets.loc[common] > 0.005]

    bear_correct = day_signals.loc[bearish_days.index].isin(["bearish"]).sum()
    bear_total = len(bearish_days)
    bear_pct = bear_correct / bear_total * 100 if bear_total > 0 else 0

    bull_correct = day_signals.loc[bullish_days.index].isin(["bullish"]).sum()
    bull_total = len(bullish_days)
    bull_pct = bull_correct / bull_total * 100 if bull_total > 0 else 0

    return bear_pct, bear_total, bull_pct, bull_total


def compute_oos_separation(detector, features, ohlcv, feature_cols, n_regimes=7, min_train=6):
    """Quick walk-forward OOS p-value."""
    feats = features.dropna().copy()
    if feats.index.tz is not None:
        feats["year_month"] = feats.index.tz_localize(None).to_period("M")
    else:
        feats["year_month"] = feats.index.to_period("M")
    months = feats["year_month"].unique()

    bullish_rets = []
    bearish_rets = []

    for i in range(min_train, len(months)):
        train_months = months[:i]
        test_month = months[i]

        train_mask = feats["year_month"].isin(train_months)
        test_mask = feats["year_month"] == test_month

        train_data = feats[train_mask][feature_cols]
        test_data = feats[test_mask][feature_cols]

        if len(test_data) < 10:
            continue

        det = RegimeDetector(n_regimes=n_regimes, n_restarts=3, n_iter=100)
        try:
            det.fit(train_data, feature_cols=feature_cols)
        except Exception:
            continue

        test_preds = det.predict(test_data)
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
                bullish_rets.extend([r_mean * 100] * count)
            elif signal == "bearish":
                bearish_rets.extend([r_mean * 100] * count)

    bullish_rets = [x for x in bullish_rets if not np.isnan(x)]
    bearish_rets = [x for x in bearish_rets if not np.isnan(x)]

    if bullish_rets and bearish_rets:
        _, p_val = stats.ttest_ind(bullish_rets, bearish_rets, equal_var=False)
        separation = np.mean(bullish_rets) - np.mean(bearish_rets)
        return p_val, separation
    return 1.0, 0.0


def main():
    print("=" * 70)
    print("SMOOTHING VARIANT COMPARISON")
    print("=" * 70)

    # Load data
    print("\nLoading SPY 1h data...")
    ohlcv, _, _ = load_data("SPY", "1h", "730d", cache=True, include_macro=False)
    print(f"Loaded {len(ohlcv):,} bars")

    feature_cols = ["returns", "range", "volume_vol", "close_in_range", "close_vs_open"]
    old_feature_cols = ["returns", "range", "volume_vol"]

    # Variants to test
    variants = [
        ("OLD: 3 feat, 7 states", None, None, old_feature_cols, 7),
        ("5-bar smooth both", 5, 5, feature_cols, 7),
        ("3-bar smooth both", 3, 3, feature_cols, 7),
        ("5-bar CIR only, raw CVO", 5, 0, feature_cols, 7),
        ("5-bar CVO only, raw CIR", 0, 5, feature_cols, 7),
        ("3-bar CIR only, raw CVO", 3, 0, feature_cols, 7),
        ("3-bar CVO only, raw CIR", 0, 3, feature_cols, 7),
        ("Raw both (no smoothing)", 0, 0, feature_cols, 7),
        # Also test n=5 and n=6 with best smoothing
        ("5-bar smooth both, n=5", 5, 5, feature_cols, 5),
        ("5-bar smooth both, n=6", 5, 5, feature_cols, 6),
        ("3-bar smooth both, n=5", 3, 3, feature_cols, 5),
        ("3-bar smooth both, n=6", 3, 3, feature_cols, 6),
    ]

    results = []

    for name, cir_w, cvo_w, feat_cols, n_states in variants:
        print(f"\n  Testing: {name}...", end=" ", flush=True)

        if cir_w is None:
            # OLD model: only 3 features
            from data.data_loader import compute_hmm_features as orig_features
            hmm_feats = orig_features(ohlcv)
            # Use only 3 features for old model
            feats_clean = hmm_feats[old_feature_cols].dropna()
            use_cols = old_feature_cols
        else:
            hmm_feats = compute_variant_features(ohlcv, cir_w if cir_w else 0, cvo_w if cvo_w else 0)
            feats_clean = hmm_feats[feat_cols].dropna()
            use_cols = feat_cols

        detector = RegimeDetector(n_regimes=n_states, n_restarts=10, n_iter=200)
        detector.fit(feats_clean, feature_cols=use_cols)

        bear_pct, bear_n, bull_pct, bull_n = measure_directional_accuracy(
            detector, feats_clean, ohlcv, use_cols
        )

        # Quick OOS check
        p_val, separation = compute_oos_separation(
            detector, hmm_feats, ohlcv, use_cols, n_regimes=n_states
        )

        print(f"Bear={bear_pct:.1f}% Bull={bull_pct:.1f}% OOS_p={p_val:.4f}")

        results.append({
            "variant": name,
            "bear_pct": bear_pct,
            "bear_n": bear_n,
            "bull_pct": bull_pct,
            "bull_n": bull_n,
            "oos_p": p_val,
            "oos_sep": separation,
            "n_states": n_states,
        })

    # Print comparison table
    print("\n\n" + "=" * 100)
    print("VARIANT COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Variant':<35} | {'Bear%':>6} | {'Bull%':>6} | {'OOS p':>8} | {'OOS Sep':>8} | {'n':>2}")
    print("-" * 80)

    for r in results:
        marker = ""
        # Best if bear > old AND bull >= old
        print(f"{r['variant']:<35} | {r['bear_pct']:>5.1f}% | {r['bull_pct']:>5.1f}% | "
              f"{r['oos_p']:>8.4f} | {r['oos_sep']:>8.4f} | {r['n_states']:>2}")

    # Find best variant (highest bear_pct where bull_pct >= old bull_pct)
    old_result = results[0]
    candidates = [r for r in results[1:] if r["bull_pct"] >= old_result["bull_pct"] and r["oos_p"] < 0.05]
    if candidates:
        best = max(candidates, key=lambda x: x["bear_pct"])
        print(f"\nBEST (bull >= old, OOS significant): {best['variant']}")
        print(f"  Bear: {best['bear_pct']:.1f}% (vs {old_result['bear_pct']:.1f}% old)")
        print(f"  Bull: {best['bull_pct']:.1f}% (vs {old_result['bull_pct']:.1f}% old)")
    else:
        # Relax: find best bear where bull didn't drop more than 5%
        candidates2 = [r for r in results[1:]
                       if r["bull_pct"] >= old_result["bull_pct"] - 5
                       and r["oos_p"] < 0.05]
        if candidates2:
            best = max(candidates2, key=lambda x: x["bear_pct"])
            print(f"\nBEST (bull within 5% of old, OOS significant): {best['variant']}")
            print(f"  Bear: {best['bear_pct']:.1f}% (vs {old_result['bear_pct']:.1f}% old)")
            print(f"  Bull: {best['bull_pct']:.1f}% (vs {old_result['bull_pct']:.1f}% old)")
        else:
            print("\nNo variant beats old model on BOTH metrics. Best tradeoffs:")
            for r in sorted(results[1:], key=lambda x: x["bear_pct"], reverse=True)[:3]:
                delta_bear = r["bear_pct"] - old_result["bear_pct"]
                delta_bull = r["bull_pct"] - old_result["bull_pct"]
                print(f"  {r['variant']}: Bear {delta_bear:+.1f}%, Bull {delta_bull:+.1f}%, OOS p={r['oos_p']:.4f}")


if __name__ == "__main__":
    main()
