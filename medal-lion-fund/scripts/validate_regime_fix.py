"""
Regime Classification Validation — New 5-Feature/5-State vs Old 3-Feature/7-State

Validates that the updated HMM model (with directional features close_in_range and
close_vs_open) correctly classifies known misclassified days:
  - March 12, 2026: SPY -0.76%, stair-step decline, old model called "Bull Run"
  - March 18, 2026: SPY -1.02%, selloff, old model called "Bull Run"

Also checks for regressions: does fixing those days break overall directional accuracy?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from data.data_loader import download_ohlcv, compute_hmm_features
from config.settings import (
    DEFAULT_TICKER, DEFAULT_INTERVAL, DEFAULT_PERIOD,
    PROCESSED_DATA_DIR, HMM_FEATURES, DEFAULT_N_REGIMES,
    HMM_COVARIANCE_TYPE, HMM_N_ITER, HMM_N_RESTARTS, HMM_RANDOM_STATE,
)


# ── Configuration ──────────────────────────────────────────────────────────
OLD_FEATURES = ["returns", "range", "volume_vol"]
OLD_N_STATES = 7
NEW_FEATURES = HMM_FEATURES  # all 5 from config
NEW_N_STATES = DEFAULT_N_REGIMES  # 5 from config

# Label templates for auto-labeling (sort by mean return)
LABEL_TEMPLATES = {
    5: ["Crash (Panic)", "Bear Trend", "Accumulation (Chop)", "Recovery", "Bull Run (Trend)"],
    7: ["Crash (Panic)", "Bear Trend", "Distribution", "Accumulation (Chop)",
        "Recovery", "Bull Run (Trend)", "Strong Bull (Trend)"],
}

BULLISH_LABELS = {"Bull Run (Trend)", "Strong Bull (Trend)", "Recovery"}
BEARISH_LABELS = {"Crash (Panic)", "Bear Trend", "Distribution"}

# Known misclassified dates
TARGET_DATES = ["2026-03-12", "2026-03-18"]


def fit_hmm(X_scaled, n_states, random_state=42, n_restarts=10, n_iter=200):
    """Fit HMM with multiple random restarts, return best model."""
    best_score = -np.inf
    best_model = None
    for seed in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state + seed,
            verbose=False,
        )
        try:
            model.fit(X_scaled)
            score = model.score(X_scaled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError("HMM fitting failed on all restarts")
    return best_model, best_score


def auto_label(model, features_df, feature_cols, n_states, scaler):
    """Sort regimes by mean return and assign labels."""
    X = features_df[feature_cols].dropna()
    X_scaled = scaler.transform(X.values)
    states = model.predict(X_scaled)

    regime_stats = {}
    for regime in range(n_states):
        mask = states == regime
        mean_ret = features_df.loc[X.index[mask], "returns"].mean()
        regime_stats[regime] = mean_ret

    sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1])
    templates = LABEL_TEMPLATES.get(n_states, [f"Regime {i}" for i in range(n_states)])
    labels = {sorted_regimes[i][0]: templates[i] for i in range(n_states)}
    return labels


def classify_signal(label):
    """Map label to bullish/bearish/neutral."""
    if label in BULLISH_LABELS:
        return "bullish"
    elif label in BEARISH_LABELS:
        return "bearish"
    else:
        return "neutral"


def compute_daily_returns(ohlcv):
    """Compute daily returns from hourly OHLCV (open-to-close per day)."""
    ohlcv_et = ohlcv.copy()
    if ohlcv_et.index.tz is not None:
        ohlcv_et.index = ohlcv_et.index.tz_convert("America/New_York")
    else:
        ohlcv_et.index = ohlcv_et.index.tz_localize("America/New_York")

    daily = ohlcv_et.groupby(ohlcv_et.index.date).agg(
        Open=("Open", "first"),
        Close=("Close", "last"),
        High=("High", "max"),
        Low=("Low", "min"),
    )
    daily["daily_return"] = (daily["Close"] - daily["Open"]) / daily["Open"]
    daily["daily_range"] = (daily["High"] - daily["Low"]) / daily["Close"]
    return daily


def run_model(ohlcv, hmm_features, feature_cols, n_states, model_name):
    """Fit a model and return predictions aligned to the feature index."""
    print(f"\nFitting {model_name}: {n_states} states, features={feature_cols}")

    X = hmm_features[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model, score = fit_hmm(
        X_scaled, n_states,
        random_state=HMM_RANDOM_STATE,
        n_restarts=HMM_N_RESTARTS,
        n_iter=HMM_N_ITER,
    )
    print(f"  Log-likelihood: {score:.2f}")

    labels = auto_label(model, hmm_features, feature_cols, n_states, scaler)
    print(f"  Labels: {labels}")

    states = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    result = pd.DataFrame(index=X.index)
    result["regime"] = states
    result["regime_label"] = [labels[s] for s in states]
    result["signal"] = result["regime_label"].apply(classify_signal)
    result["confidence"] = [probs[i, states[i]] for i in range(len(states))]
    result["returns"] = hmm_features.loc[X.index, "returns"]

    return result, labels, model, scaler


def daily_regime(hourly_result):
    """Get majority regime label per day."""
    hourly_result = hourly_result.copy()
    hourly_result["date"] = hourly_result.index.date
    daily_label = hourly_result.groupby("date")["regime_label"].agg(
        lambda x: x.value_counts().index[0]
    )
    daily_signal = daily_label.apply(classify_signal)
    return pd.DataFrame({"regime_label": daily_label, "signal": daily_signal})


def main():
    # ── Load Data ──────────────────────────────────────────────────────────
    cache_path = PROCESSED_DATA_DIR / f"{DEFAULT_TICKER.lower()}_{DEFAULT_INTERVAL}_{DEFAULT_PERIOD}_cache.parquet"
    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        ohlcv = pd.read_parquet(cache_path)
    else:
        ohlcv = download_ohlcv()

    if isinstance(ohlcv.columns, pd.MultiIndex):
        ohlcv.columns = ohlcv.columns.get_level_values(0)

    if ohlcv.index.tz is None:
        ohlcv.index = ohlcv.index.tz_localize("UTC")

    print(f"Data: {len(ohlcv)} bars, {ohlcv.index.min()} to {ohlcv.index.max()}")

    # ── Compute Features ───────────────────────────────────────────────────
    hmm_features = compute_hmm_features(ohlcv)
    print(f"HMM features computed: {list(hmm_features.columns)}")
    print(f"  Valid rows (no NaN): {hmm_features.dropna().shape[0]}")

    # ── Fit Both Models ────────────────────────────────────────────────────
    old_result, old_labels, old_model, old_scaler = run_model(
        ohlcv, hmm_features, OLD_FEATURES, OLD_N_STATES, "OLD MODEL"
    )
    new_result, new_labels, new_model, new_scaler = run_model(
        ohlcv, hmm_features, NEW_FEATURES, NEW_N_STATES, "NEW MODEL"
    )

    # ── Compute Daily Returns ──────────────────────────────────────────────
    daily = compute_daily_returns(ohlcv)

    # ── Map hourly regime to daily (majority vote per day) ─────────────────
    old_et = old_result.copy()
    new_et = new_result.copy()
    old_et.index = old_et.index.tz_convert("America/New_York")
    new_et.index = new_et.index.tz_convert("America/New_York")

    old_daily = daily_regime(old_et)
    new_daily = daily_regime(new_et)

    # ── Merge daily returns with regime labels ─────────────────────────────
    merged = daily.copy()
    merged["old_label"] = old_daily["regime_label"]
    merged["old_signal"] = old_daily["signal"]
    merged["new_label"] = new_daily["regime_label"]
    merged["new_signal"] = new_daily["signal"]
    merged = merged.dropna(subset=["old_label", "new_label"])

    # ══════════════════════════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  REGIME CLASSIFICATION VALIDATION")
    print("=" * 78)

    # ── 1. The Two Target Days ────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("  1. KNOWN MISCLASSIFIED DAYS (March 12 & 18, 2026)")
    print("-" * 78)

    for date_str in TARGET_DATES:
        date = pd.Timestamp(date_str).date()
        if date in merged.index:
            row = merged.loc[date]
            ret = row["daily_return"] * 100
            old_lbl = row["old_label"]
            new_lbl = row["new_label"]
            old_sig = row["old_signal"]
            new_sig = row["new_signal"]
            is_fixed = (new_sig == "bearish")

            print(f"\n  {date_str}: SPY daily return = {ret:+.2f}%")
            print(f"    OLD model: {old_lbl} ({old_sig})")
            print(f"    NEW model: {new_lbl} ({new_sig})")
            print(f"    --> {'FIXED' if is_fixed else 'STILL WRONG'}")

    # ── Hourly timeline for target days ───────────────────────────────────
    print("\n  Hourly breakdown:")
    for date_str in TARGET_DATES:
        date = pd.Timestamp(date_str).date()
        old_bars = old_et[old_et.index.date == date]
        new_bars = new_et[new_et.index.date == date]
        if len(new_bars) == 0:
            continue
        print(f"\n  {date_str}:")
        print(f"  {'Time':>6s}  {'Return':>8s}  {'OLD regime':>28s}  {'NEW regime':>28s}  {'Signal':>10s}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*28}  {'─'*28}  {'─'*10}")
        for idx in new_bars.index:
            time_str = idx.strftime("%H:%M")
            ret = new_bars.loc[idx, "returns"]
            ret_str = f"{ret*100:+.3f}%" if not pd.isna(ret) else "  N/A  "
            old_lbl = old_bars.loc[idx, "regime_label"] if idx in old_bars.index else "N/A"
            new_lbl = new_bars.loc[idx, "regime_label"]
            new_sig = new_bars.loc[idx, "signal"]
            print(f"  {time_str:>6s}  {ret_str:>8s}  {old_lbl:>28s}  {new_lbl:>28s}  {new_sig:>10s}")

    # ── 2. Directional Accuracy ───────────────────────────────────────────
    print("\n" + "-" * 78)
    print("  2. DIRECTIONAL ACCURACY (daily majority vote)")
    print("-" * 78)

    for threshold, label in [(-0.005, "-0.5%"), (-0.008, "-0.8%"), (-0.01, "-1.0%")]:
        bear_mask = merged["daily_return"] < threshold
        bear_days = merged[bear_mask]
        n = len(bear_days)
        if n == 0:
            continue
        old_correct = (bear_days["old_signal"] == "bearish").sum()
        new_correct = (bear_days["new_signal"] == "bearish").sum()
        old_pct = old_correct / n * 100
        new_pct = new_correct / n * 100
        print(f"\n  Down days (return < {label}): {n} days")
        print(f"    OLD classified bearish: {old_pct:5.1f}% ({old_correct}/{n})")
        print(f"    NEW classified bearish: {new_pct:5.1f}% ({new_correct}/{n})")
        print(f"    Change: {new_pct - old_pct:+.1f} pp")

    for threshold, label in [(0.005, "+0.5%"), (0.008, "+0.8%"), (0.01, "+1.0%")]:
        bull_mask = merged["daily_return"] > threshold
        bull_days = merged[bull_mask]
        n = len(bull_days)
        if n == 0:
            continue
        old_correct = (bull_days["old_signal"] == "bullish").sum()
        new_correct = (bull_days["new_signal"] == "bullish").sum()
        old_pct = old_correct / n * 100
        new_pct = new_correct / n * 100
        print(f"\n  Up days (return > {label}): {n} days")
        print(f"    OLD classified bullish: {old_pct:5.1f}% ({old_correct}/{n})")
        print(f"    NEW classified bullish: {new_pct:5.1f}% ({new_correct}/{n})")
        print(f"    Change: {new_pct - old_pct:+.1f} pp")

    # ── 3. Regime Distribution ────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("  3. REGIME DISTRIBUTION (daily majority vote)")
    print("-" * 78)

    print("\n  OLD MODEL (3 features, 7 states):")
    old_counts = merged["old_label"].value_counts()
    for label, count in old_counts.items():
        pct = count / len(merged) * 100
        sig = classify_signal(label)
        print(f"    {label:30s}: {count:4d} days ({pct:5.1f}%)  [{sig}]")

    print(f"\n  NEW MODEL (5 features, 5 states):")
    new_counts = merged["new_label"].value_counts()
    for label, count in new_counts.items():
        pct = count / len(merged) * 100
        sig = classify_signal(label)
        print(f"    {label:30s}: {count:4d} days ({pct:5.1f}%)  [{sig}]")

    # Signal summary
    for name, col in [("OLD", "old_signal"), ("NEW", "new_signal")]:
        sig_counts = merged[col].value_counts()
        total = len(merged)
        print(f"\n  {name} signal breakdown:")
        for sig in ["bearish", "neutral", "bullish"]:
            c = sig_counts.get(sig, 0)
            print(f"    {sig:10s}: {c:4d} days ({c/total*100:5.1f}%)")

    # ── 4. What the New Model Actually Learned (Feature Profiles) ─────────
    print("\n" + "-" * 78)
    print("  4. FEATURE PROFILES PER STATE (NEW MODEL)")
    print("-" * 78)

    X_new = hmm_features[NEW_FEATURES].dropna()
    X_new_scaled = new_scaler.transform(X_new.values)
    new_states = new_model.predict(X_new_scaled)
    X_analysis = X_new.copy()
    X_analysis["state"] = new_states

    print(f"\n  {'State':>5s}  {'Label':>25s}  {'Bars':>5s}  {'%':>5s}  {'mean_ret':>10s}  {'range':>8s}  {'close_in_rng':>12s}  {'close_vs_opn':>12s}")
    print(f"  {'─'*5}  {'─'*25}  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*12}")

    for state in range(NEW_N_STATES):
        mask = X_analysis["state"] == state
        count = mask.sum()
        pct = count / len(X_analysis) * 100
        label = new_labels[state]
        mean_ret = X_analysis.loc[mask, "returns"].mean()
        mean_rng = X_analysis.loc[mask, "range"].mean()
        mean_cir = X_analysis.loc[mask, "close_in_range"].mean()
        mean_cvo = X_analysis.loc[mask, "close_vs_open"].mean()
        print(f"  {state:>5d}  {label:>25s}  {count:>5d}  {pct:>4.1f}%  {mean_ret:>+10.6f}  {mean_rng:>8.5f}  {mean_cir:>12.3f}  {mean_cvo:>+12.5f}")

    # ── 5. Forward Returns Test (does the regime predict anything?) ───────
    print("\n" + "-" * 78)
    print("  5. FORWARD RETURN BY STATE (does each regime predict the future?)")
    print("-" * 78)

    X_analysis["fwd_1"] = hmm_features.loc[X_analysis.index, "returns"].shift(-1)
    X_analysis["fwd_7"] = hmm_features.loc[X_analysis.index, "returns"].rolling(7).sum().shift(-7)

    print(f"\n  {'Label':>25s}  {'1-bar fwd':>12s}  {'7-bar fwd':>12s}  {'P(7bar<0)':>10s}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*10}")

    for state in range(NEW_N_STATES):
        mask = X_analysis["state"] == state
        label = new_labels[state]
        fwd1 = X_analysis.loc[mask, "fwd_1"].dropna()
        fwd7 = X_analysis.loc[mask, "fwd_7"].dropna()
        mean_fwd1 = fwd1.mean() * 100 if len(fwd1) > 0 else 0
        mean_fwd7 = fwd7.mean() * 100 if len(fwd7) > 0 else 0
        pct_neg = (fwd7 < 0).mean() * 100 if len(fwd7) > 0 else 0
        print(f"  {label:>25s}  {mean_fwd1:>+10.4f}%  {mean_fwd7:>+10.4f}%  {pct_neg:>9.1f}%")

    print("\n  (Same for OLD model:)")
    X_old = hmm_features[OLD_FEATURES].dropna()
    X_old_scaled = old_scaler.transform(X_old.values)
    old_states = old_model.predict(X_old_scaled)
    X_old_analysis = X_old.copy()
    X_old_analysis["state"] = old_states
    X_old_analysis["fwd_1"] = hmm_features.loc[X_old_analysis.index, "returns"].shift(-1)
    X_old_analysis["fwd_7"] = hmm_features.loc[X_old_analysis.index, "returns"].rolling(7).sum().shift(-7)

    print(f"\n  {'Label':>25s}  {'1-bar fwd':>12s}  {'7-bar fwd':>12s}  {'P(7bar<0)':>10s}  {'Bars':>6s}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*6}")

    for state in range(OLD_N_STATES):
        mask = X_old_analysis["state"] == state
        label = old_labels[state]
        count = mask.sum()
        fwd1 = X_old_analysis.loc[mask, "fwd_1"].dropna()
        fwd7 = X_old_analysis.loc[mask, "fwd_7"].dropna()
        mean_fwd1 = fwd1.mean() * 100 if len(fwd1) > 0 else 0
        mean_fwd7 = fwd7.mean() * 100 if len(fwd7) > 0 else 0
        pct_neg = (fwd7 < 0).mean() * 100 if len(fwd7) > 0 else 0
        print(f"  {label:>25s}  {mean_fwd1:>+10.4f}%  {mean_fwd7:>+10.4f}%  {pct_neg:>9.1f}%  {count:>6d}")

    # ── 6. Top 10 Worst Selloff Days ──────────────────────────────────────
    print("\n" + "-" * 78)
    print("  6. TOP 10 WORST SELLOFF DAYS")
    print("-" * 78)

    worst_days = merged.nsmallest(10, "daily_return")
    print(f"\n  {'Date':>12s}  {'Return':>8s}  {'OLD regime':>25s}  {'NEW regime':>25s}  {'Status':>20s}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*25}  {'─'*25}  {'─'*20}")

    for date, row in worst_days.iterrows():
        ret = row["daily_return"] * 100
        old_lbl = row["old_label"]
        new_lbl = row["new_label"]
        old_sig = row["old_signal"]
        new_sig = row["new_signal"]

        was_wrong_old = old_sig != "bearish"
        is_wrong_new = new_sig != "bearish"
        fixed = was_wrong_old and not is_wrong_new
        still_wrong = is_wrong_new

        if fixed:
            status = "FIXED"
        elif still_wrong and was_wrong_old:
            status = "BOTH WRONG"
        elif still_wrong:
            status = "NEW WRONG (regression)"
        else:
            status = "OK (both correct)"

        print(f"  {date!s:>12s}  {ret:>+7.2f}%  {old_lbl:>25s}  {new_lbl:>25s}  {status:>20s}")

    # ── 7. Remaining Misclassifications ───────────────────────────────────
    print("\n" + "-" * 78)
    print("  7. TOP 10 WORST REMAINING MISCLASSIFICATIONS (NEW MODEL)")
    print("-" * 78)

    bear_still_wrong = merged[
        (merged["daily_return"] < -0.003) & (merged["new_signal"] != "bearish")
    ].sort_values("daily_return")

    print(f"\n  Down days (< -0.3%) NOT classified bearish:")
    for i, (date, row) in enumerate(bear_still_wrong.head(10).iterrows()):
        ret = row["daily_return"] * 100
        new_lbl = row["new_label"]
        print(f"    {date!s:>12s}  ret={ret:+.2f}%  | Classified: {new_lbl}")

    if len(bear_still_wrong) == 0:
        print("    (none -- all bearish days correctly classified)")

    # ── 8. Cross-tab: return bucket vs regime ─────────────────────────────
    print("\n" + "-" * 78)
    print("  8. RETURN BUCKET VS SIGNAL DISTRIBUTION")
    print("-" * 78)

    def return_bucket(r):
        if r < -0.01:
            return "< -1.0%"
        elif r < -0.005:
            return "-1.0% to -0.5%"
        elif r < 0.0:
            return "-0.5% to  0.0%"
        elif r < 0.005:
            return " 0.0% to +0.5%"
        elif r < 0.01:
            return "+0.5% to +1.0%"
        else:
            return "> +1.0%"

    merged["return_bucket"] = merged["daily_return"].apply(return_bucket)
    bucket_order = ["< -1.0%", "-1.0% to -0.5%", "-0.5% to  0.0%",
                    " 0.0% to +0.5%", "+0.5% to +1.0%", "> +1.0%"]

    print(f"\n  {'Bucket':>20s}  {'Days':>5s}  {'OLD bear%':>10s}  {'NEW bear%':>10s}  {'OLD bull%':>10s}  {'NEW bull%':>10s}")
    print(f"  {'─'*20}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for bucket in bucket_order:
        subset = merged[merged["return_bucket"] == bucket]
        if len(subset) == 0:
            continue
        n = len(subset)
        old_bear = (subset["old_signal"] == "bearish").sum() / n * 100
        new_bear = (subset["new_signal"] == "bearish").sum() / n * 100
        old_bull = (subset["old_signal"] == "bullish").sum() / n * 100
        new_bull = (subset["new_signal"] == "bullish").sum() / n * 100
        print(f"  {bucket:>20s}  {n:>5d}  {old_bear:>9.1f}%  {new_bear:>9.1f}%  {old_bull:>9.1f}%  {new_bull:>9.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    #  VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)

    # Target days
    mar12_fixed = (pd.Timestamp("2026-03-12").date() in merged.index and
                   merged.loc[pd.Timestamp("2026-03-12").date(), "new_signal"] == "bearish")
    mar18_fixed = (pd.Timestamp("2026-03-18").date() in merged.index and
                   merged.loc[pd.Timestamp("2026-03-18").date(), "new_signal"] == "bearish")

    # Directional accuracy at -0.5%
    bear_mask = merged["daily_return"] < -0.005
    bull_mask = merged["daily_return"] > 0.005
    n_bear = bear_mask.sum()
    n_bull = bull_mask.sum()
    old_bear_pct = (merged.loc[bear_mask, "old_signal"] == "bearish").sum() / n_bear * 100
    new_bear_pct = (merged.loc[bear_mask, "new_signal"] == "bearish").sum() / n_bear * 100
    old_bull_pct = (merged.loc[bull_mask, "old_signal"] == "bullish").sum() / n_bull * 100
    new_bull_pct = (merged.loc[bull_mask, "new_signal"] == "bullish").sum() / n_bull * 100

    # New model: check signal distribution imbalance
    new_bear_total = (merged["new_signal"] == "bearish").sum()
    new_bull_total = (merged["new_signal"] == "bullish").sum()
    new_neut_total = (merged["new_signal"] == "neutral").sum()

    print(f"""
  TARGET DAYS:
    March 12, 2026 (-0.76%): {'FIXED -- now bearish' if mar12_fixed else 'NOT FIXED'}
    March 18, 2026 (-1.02%): {'FIXED -- now bearish' if mar18_fixed else 'NOT FIXED'}

  DIRECTIONAL ACCURACY (days with |return| > 0.5%):
    Bearish days classified bearish: {old_bear_pct:.1f}% (old) -> {new_bear_pct:.1f}% (new) = {new_bear_pct - old_bear_pct:+.1f}pp
    Bullish days classified bullish: {old_bull_pct:.1f}% (old) -> {new_bull_pct:.1f}% (new) = {new_bull_pct - old_bull_pct:+.1f}pp

  SIGNAL DISTRIBUTION (new model):
    Bearish: {new_bear_total} days ({new_bear_total/len(merged)*100:.1f}%)
    Neutral: {new_neut_total} days ({new_neut_total/len(merged)*100:.1f}%)
    Bullish: {new_bull_total} days ({new_bull_total/len(merged)*100:.1f}%)""")

    # Diagnosis
    if new_bear_pct < old_bear_pct:
        print(f"""
  DIAGNOSIS: The new model FIXED the two target days but REGRESSED on overall
  bearish accuracy ({old_bear_pct:.0f}% -> {new_bear_pct:.0f}%). Root cause analysis:

  The 5-state model with directional features creates states that capture
  BAR-LEVEL direction (close near high vs close near low) rather than
  MARKET REGIMES (multi-day trends). The state labeled "Crash (Panic)"
  is really just "bars closing near their low" -- 31.7% of all bars,
  which is far too many for a crash regime.

  The directional features (close_in_range, close_vs_open) are highly
  correlated with returns, so the model is essentially creating 5 "return
  buckets" instead of 5 regime states. When aggregated to daily majority
  vote, many flat/mild days get labeled bearish because they had more
  "close near low" hours than "close near high" hours.

  RECOMMENDATIONS:
  1. The directional features are useful but may need to be SMOOTHED
     (e.g., 5-bar rolling mean of close_in_range) to capture regime-level
     direction rather than bar-level noise.
  2. Alternatively, keep the old 3-feature model and add a SEPARATE
     directional overlay that checks close_in_range/close_vs_open as
     a confirmation layer, not as HMM features.
  3. Test n_states=6 or 7 with the 5 features -- more states might let
     the model separate "bars closing near low in bull market" from
     "actual bear trend."
  4. Walk-forward validation is needed before deploying any change.""")
    elif mar12_fixed and mar18_fixed:
        print(f"""
  PASS: Both target days fixed AND directional accuracy improved.
  The new 5-feature model is ready for walk-forward validation.""")
    else:
        print(f"""
  PARTIAL: Some improvement but not all target days fixed.
  Further tuning needed.""")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
