#!/usr/bin/env python3
"""Regime Transition Latency — Measure Detection Speed + False Alarms.

Analyzes the walk-forward regime labels to answer:
1. Transition detection speed: When regime changes, how fast does confidence build?
2. False alarm rate: Regime assignments lasting < 3 hours (flickers)
3. Regime stability at 5m: What % of 5m trades enter during first/last hour of a regime?
4. Consistency check: Walk-forward labels vs biased labels — how similar?

Usage:
    cd ~/projects/ai-playground/medallion-2.0
    source .venv/bin/activate
    python scripts/regime_transition_latency.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from collections import defaultdict

MEDALLION_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(MEDALLION_ROOT))


def load_walk_forward_labels():
    """Load the walk-forward regime timeline."""
    wf_path = MEDALLION_ROOT / "data" / "processed" / "walk_forward_regimes.parquet"
    if not wf_path.exists():
        print(f"ERROR: Walk-forward regimes not found at {wf_path}")
        print(f"  Run: python scripts/walk_forward_regimes.py --use-5yr")
        sys.exit(1)

    wf = pd.read_parquet(wf_path)
    print(f"  Walk-forward: {len(wf):,} bars, {wf.index[0]} to {wf.index[-1]}")
    return wf


def load_biased_labels():
    """Load the biased (full-sample) regime labels for comparison."""
    from models.hmm_regime import RegimeDetector
    from models.regime_api import RegimeFilter
    from data.data_loader import download_ohlcv, compute_hmm_features

    detector = RegimeDetector.load_latest(n_regimes=7)
    if detector is None:
        print("  WARNING: No saved 7-state model found, skipping biased comparison")
        return None

    spy_hourly = download_ohlcv("SPY", "1h", "730d")
    hmm_features = compute_hmm_features(spy_hourly)
    full_df = spy_hourly.join(hmm_features)

    rf = RegimeFilter(detector, full_df)
    biased = rf.predictions.copy()
    print(f"  Biased: {len(biased):,} bars, {biased.index[0]} to {biased.index[-1]}")
    return biased


def analyze_transition_speed(wf: pd.DataFrame):
    """Measure how quickly confidence builds after regime transitions.

    For each transition: track confidence for the next N bars.
    """
    print(f"\n{'=' * 70}")
    print(f"  TRANSITION DETECTION SPEED")
    print(f"{'=' * 70}")

    regime_col = wf["regime_label"].dropna()
    conf_col = wf["confidence"].dropna()

    # Find transition points
    transitions = []
    prev_label = None
    for i, (idx, label) in enumerate(regime_col.items()):
        if prev_label is not None and label != prev_label:
            transitions.append({
                "idx": idx,
                "from": prev_label,
                "to": label,
                "pos": i,
            })
        prev_label = label

    print(f"  Total transitions: {len(transitions)}")

    if not transitions:
        return {}

    # Track confidence ramp-up after each transition
    lookback = 12  # 12 hours after transition
    ramp_profiles = []

    for trans in transitions:
        pos = trans["pos"]
        # Get confidence values for the next N bars
        end_pos = min(pos + lookback, len(conf_col))
        profile = conf_col.iloc[pos:end_pos].values
        if len(profile) > 0:
            ramp_profiles.append(profile)

    # Compute average confidence at each hour after transition
    avg_confidence_by_hour = []
    for h in range(lookback):
        values = [p[h] for p in ramp_profiles if len(p) > h]
        if values:
            avg_confidence_by_hour.append(np.mean(values))
        else:
            avg_confidence_by_hour.append(np.nan)

    print(f"\n  Average confidence after regime transition:")
    for h, conf in enumerate(avg_confidence_by_hour):
        if not np.isnan(conf):
            bar = "#" * int(conf * 40)
            marker = " ***" if conf >= 0.7 else ""
            print(f"    +{h:>2d}h: {conf:.1%}  {bar}{marker}")

    # Time to reach 70% confidence
    above_70 = [h for h, c in enumerate(avg_confidence_by_hour) if c >= 0.7]
    if above_70:
        hours_to_70 = above_70[0]
        print(f"\n  Average time to 70% confidence: {hours_to_70} hours")
    else:
        hours_to_70 = None
        print(f"\n  Average confidence never reaches 70% within {lookback}h")

    return {
        "total_transitions": len(transitions),
        "avg_confidence_by_hour": [float(c) if not np.isnan(c) else None for c in avg_confidence_by_hour],
        "hours_to_70pct": hours_to_70,
    }


def analyze_false_alarms(wf: pd.DataFrame):
    """Identify regime assignments lasting < 3 hours (flickers).

    A flicker is a regime that appears briefly and disappears — likely noise.
    """
    print(f"\n{'=' * 70}")
    print(f"  FALSE ALARM RATE (flickers < 3 hours)")
    print(f"{'=' * 70}")

    regime_col = wf["regime_label"].dropna()

    # Find contiguous regime segments
    segments = []
    start_idx = regime_col.index[0]
    current_label = regime_col.iloc[0]
    count = 1

    for i in range(1, len(regime_col)):
        label = regime_col.iloc[i]
        if label == current_label:
            count += 1
        else:
            segments.append({
                "label": current_label,
                "start": start_idx,
                "end": regime_col.index[i - 1],
                "bars": count,
            })
            start_idx = regime_col.index[i]
            current_label = label
            count = 1

    # Add last segment
    segments.append({
        "label": current_label,
        "start": start_idx,
        "end": regime_col.index[-1],
        "bars": count,
    })

    total_segments = len(segments)
    flickers = [s for s in segments if s["bars"] < 3]
    flicker_rate = len(flickers) / total_segments * 100 if total_segments > 0 else 0

    print(f"  Total regime segments: {total_segments}")
    print(f"  Segments < 3 hours (flickers): {len(flickers)} ({flicker_rate:.1f}%)")

    # Breakdown by regime
    flicker_by_regime = defaultdict(int)
    total_by_regime = defaultdict(int)
    for s in segments:
        total_by_regime[s["label"]] += 1
    for s in flickers:
        flicker_by_regime[s["label"]] += 1

    print(f"\n  Flicker rate by regime:")
    for label in sorted(total_by_regime.keys()):
        total = total_by_regime[label]
        n_flicker = flicker_by_regime.get(label, 0)
        rate = n_flicker / total * 100 if total > 0 else 0
        print(f"    {label:<25s} {n_flicker:>3d}/{total:>3d} ({rate:.0f}%)")

    # Duration distribution
    durations = [s["bars"] for s in segments]
    print(f"\n  Regime segment duration (hours):")
    print(f"    Mean:   {np.mean(durations):.1f}")
    print(f"    Median: {np.median(durations):.1f}")
    print(f"    Min:    {np.min(durations)}")
    print(f"    Max:    {np.max(durations)}")
    print(f"    Std:    {np.std(durations):.1f}")

    return {
        "total_segments": total_segments,
        "flickers": len(flickers),
        "flicker_rate_pct": flicker_rate,
        "mean_duration_hours": float(np.mean(durations)),
        "median_duration_hours": float(np.median(durations)),
    }


def analyze_consistency(wf: pd.DataFrame, biased: pd.DataFrame):
    """Compare walk-forward labels vs biased (full-sample) labels.

    If they're very similar, the bias was small.
    If very different, the full-sample model was leaning on future data.
    """
    print(f"\n{'=' * 70}")
    print(f"  CONSISTENCY CHECK: Walk-Forward vs Biased Labels")
    print(f"{'=' * 70}")

    # Align to common index
    common_idx = wf.index.intersection(biased.index)
    if len(common_idx) == 0:
        # Try timezone alignment
        if wf.index.tz is not None and biased.index.tz is None:
            biased.index = biased.index.tz_localize(wf.index.tz)
        elif wf.index.tz is None and biased.index.tz is not None:
            wf_aligned = wf.copy()
            wf_aligned.index = wf_aligned.index.tz_localize(biased.index.tz)
            common_idx = wf_aligned.index.intersection(biased.index)
            if len(common_idx) > 0:
                wf = wf_aligned

    common_idx = wf.index.intersection(biased.index)
    print(f"  Overlapping bars: {len(common_idx):,}")

    if len(common_idx) < 100:
        print(f"  Too few overlapping bars for meaningful comparison")
        return {}

    wf_labels = wf.loc[common_idx, "regime_label"]
    biased_labels = biased.loc[common_idx, "regime_label"]

    # Agreement rate
    agreement = (wf_labels == biased_labels).mean() * 100
    print(f"  Label agreement: {agreement:.1f}%")

    # Signal agreement (bullish/bearish/neutral — coarser)
    from config.settings import BULLISH_REGIMES, BEARISH_REGIMES

    def to_signal(label):
        if label in BULLISH_REGIMES:
            return "bullish"
        elif label in BEARISH_REGIMES:
            return "bearish"
        return "neutral"

    wf_signals = wf_labels.map(to_signal)
    biased_signals = biased_labels.map(to_signal)
    signal_agreement = (wf_signals == biased_signals).mean() * 100
    print(f"  Signal agreement (bull/bear/neutral): {signal_agreement:.1f}%")

    # Per-regime agreement
    print(f"\n  Per-regime agreement:")
    all_labels = sorted(set(wf_labels.dropna()) | set(biased_labels.dropna()))
    for label in all_labels:
        wf_has = (wf_labels == label)
        biased_has = (biased_labels == label)
        both = (wf_has & biased_has).sum()
        either = (wf_has | biased_has).sum()
        jaccard = both / either * 100 if either > 0 else 0
        print(f"    {label:<25s} Jaccard: {jaccard:.0f}%  (WF: {wf_has.sum()}, Biased: {biased_has.sum()}, Both: {both})")

    return {
        "overlapping_bars": len(common_idx),
        "label_agreement_pct": agreement,
        "signal_agreement_pct": signal_agreement,
    }


def main():
    print("=" * 70)
    print("  REGIME TRANSITION LATENCY ANALYSIS")
    print("=" * 70)

    # Load walk-forward labels
    wf = load_walk_forward_labels()

    # 1. Transition detection speed
    speed_results = analyze_transition_speed(wf)

    # 2. False alarm rate
    flicker_results = analyze_false_alarms(wf)

    # 3. Consistency check vs biased labels
    biased = load_biased_labels()
    consistency_results = {}
    if biased is not None:
        consistency_results = analyze_consistency(wf, biased)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")

    hours_to_70 = speed_results.get("hours_to_70pct")
    flicker_rate = flicker_results.get("flicker_rate_pct", 0)
    signal_agree = consistency_results.get("signal_agreement_pct", 0)

    print(f"  Transition detection:  {'< 3h' if hours_to_70 and hours_to_70 < 3 else '> 3h' if hours_to_70 else 'N/A'} "
          f"(target: < 3h) {'PASS' if hours_to_70 and hours_to_70 < 3 else 'FAIL'}")
    print(f"  False alarm rate:      {flicker_rate:.1f}% "
          f"(target: < 15%) {'PASS' if flicker_rate < 15 else 'FAIL'}")
    if signal_agree:
        print(f"  Signal agreement:      {signal_agree:.1f}% "
              f"(higher = less bias in original)")

    # Save results
    import json
    out_path = MEDALLION_ROOT / "data" / "processed" / "regime_transition_latency.json"
    with open(out_path, "w") as f:
        json.dump({
            "transition_speed": speed_results,
            "false_alarms": flicker_results,
            "consistency": consistency_results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
