"""
Medallion 2.0 — Hidden Markov Model Regime Detector

7-state Gaussian HMM for market regime detection.
Trains on 3 features: returns, range (high-low/close), volume volatility.
Auto-labels regimes by return + volatility characteristics.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from typing import Optional
import pickle
import hashlib
import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class RegimeDetector:
    """
    HMM-based market regime detection.

    Fits a Gaussian HMM to observable market features and infers
    hidden regime states. Supports 2-7 states with automatic labeling.
    """

    def __init__(
        self,
        n_regimes: int = 7,
        covariance_type: str = "full",
        n_iter: int = 200,
        n_restarts: int = 10,
        random_state: int = 42,
        features: Optional[list[str]] = None,
    ):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.feature_names = features
        self.scaler = StandardScaler()
        self.model: Optional[GaussianHMM] = None
        self.regime_labels: dict[int, str] = {}
        self.regime_order: list[int] = []  # regime IDs sorted by mean return
        self.is_fitted = False

    def fit(self, features: pd.DataFrame, feature_cols: Optional[list[str]] = None) -> "RegimeDetector":
        """
        Fit the HMM to feature data with multiple random restarts.

        Args:
            features: DataFrame with feature columns
            feature_cols: Which columns to use (default: all numeric)
        """
        if feature_cols:
            X = features[feature_cols].copy()
            self.feature_names = feature_cols
        elif self.feature_names:
            X = features[self.feature_names].copy()
        else:
            X = features.select_dtypes(include=[np.number]).copy()
            self.feature_names = list(X.columns)

        # Drop NaN rows
        X = X.dropna()
        self._fit_index = X.index

        # Scale features
        X_scaled = self.scaler.fit_transform(X.values)

        # Fit HMM (multiple restarts to avoid local optima)
        best_score = -np.inf
        best_model = None

        for seed in range(self.n_restarts):
            model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state + seed,
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

        self.model = best_model
        self.is_fitted = True

        # Auto-label regimes based on mean returns + volatility
        self._auto_label_regimes(features.loc[self._fit_index])

        return self

    def _auto_label_regimes(self, features: pd.DataFrame):
        """
        Assign human-readable labels based on regime characteristics.
        Sorts regimes by mean return and volatility to assign labels.
        """
        if not self.is_fitted:
            return

        X_scaled = self.scaler.transform(features[self.feature_names].values)
        states = self.model.predict(X_scaled)

        # Compute return and vol stats per regime
        regime_stats = {}
        ret_col = "returns" if "returns" in features.columns else (
            "log_returns" if "log_returns" in features.columns else None
        )

        for regime in range(self.n_regimes):
            mask = states == regime
            if ret_col and ret_col in features.columns:
                mean_ret = features.loc[features.index[mask], ret_col].mean()
                mean_vol = features.loc[features.index[mask], ret_col].std()
            else:
                mean_ret = self.model.means_[regime, 0]
                if self.covariance_type == "full":
                    mean_vol = np.sqrt(self.model.covars_[regime][0, 0])
                else:
                    mean_vol = np.sqrt(self.model.covars_[regime, 0])

            regime_stats[regime] = {
                "mean_return": mean_ret,
                "mean_vol": mean_vol,
                "count": mask.sum(),
                "pct": mask.sum() / len(states) * 100,
            }

        # Sort by return
        sorted_by_return = sorted(regime_stats.items(), key=lambda x: x[1]["mean_return"])
        self.regime_order = [r[0] for r in sorted_by_return]

        if self.n_regimes == 2:
            self.regime_labels = {
                sorted_by_return[0][0]: "Bear / Risk-Off",
                sorted_by_return[1][0]: "Bull / Risk-On",
            }
        elif self.n_regimes == 3:
            self.regime_labels = {
                sorted_by_return[0][0]: "Bear Trend",
                sorted_by_return[1][0]: "Accumulation (Chop)",
                sorted_by_return[2][0]: "Bull Run (Trend)",
            }
        elif self.n_regimes == 4:
            sorted_by_vol = sorted(regime_stats.items(), key=lambda x: x[1]["mean_vol"], reverse=True)
            highest_vol = sorted_by_vol[0][0]

            labels = {}
            for i, (regime, stats) in enumerate(sorted_by_return):
                if regime == highest_vol and stats["mean_return"] < 0:
                    labels[regime] = "Crash (Panic)"
                elif i == 0:
                    labels[regime] = "Bear Trend"
                elif i == len(sorted_by_return) - 1:
                    labels[regime] = "Bull Run (Trend)"
                else:
                    labels[regime] = "Accumulation (Chop)"
            self.regime_labels = labels
        elif self.n_regimes >= 5:
            # 5-7 state labeling: sort by return, assign from worst to best
            label_templates = {
                5: ["Crash (Panic)", "Bear Trend", "Accumulation (Chop)", "Recovery", "Bull Run (Trend)"],
                6: ["Crash (Panic)", "Bear Trend", "Distribution", "Accumulation (Chop)", "Recovery", "Bull Run (Trend)"],
                7: ["Crash (Panic)", "Bear Trend", "Distribution", "Accumulation (Chop)", "Recovery", "Bull Run (Trend)", "Strong Bull (Trend)"],
            }
            templates = label_templates.get(self.n_regimes, [f"Regime {i}" for i in range(self.n_regimes)])
            self.regime_labels = {
                sorted_by_return[i][0]: templates[i]
                for i in range(self.n_regimes)
            }
        else:
            self.regime_labels = {i: f"Regime {i}" for i in range(self.n_regimes)}

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime states and probabilities.

        Returns DataFrame with columns:
            - regime: int (most likely regime)
            - regime_label: str (human-readable label)
            - prob_0, prob_1, ...: per-regime probabilities
            - confidence: probability of predicted regime
            - signal: 'bullish', 'bearish', or 'neutral'
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = features[self.feature_names].copy()
        valid_mask = ~X.isna().any(axis=1)
        X_valid = X[valid_mask]

        X_scaled = self.scaler.transform(X_valid.values)

        states = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)

        result = pd.DataFrame(index=features.index)
        result["regime"] = np.nan
        result.loc[valid_mask, "regime"] = states
        result["regime"] = result["regime"].astype("Int64")

        result["regime_label"] = result["regime"].map(self.regime_labels)

        for i in range(self.n_regimes):
            col = f"prob_{i}"
            result[col] = np.nan
            result.loc[valid_mask, col] = probs[:, i]

        # Confidence = probability of predicted regime
        result["confidence"] = np.nan
        for idx in X_valid.index:
            regime = int(result.loc[idx, "regime"])
            result.loc[idx, "confidence"] = result.loc[idx, f"prob_{regime}"]

        # Signal classification
        from config.settings import BULLISH_REGIMES, BEARISH_REGIMES
        result["signal"] = result["regime_label"].apply(
            lambda x: "bullish" if x in BULLISH_REGIMES
            else ("bearish" if x in BEARISH_REGIMES else "neutral")
            if pd.notna(x) else None
        )

        return result

    def forecast_transitions(self, current_probs: np.ndarray, horizons: list[int] = None) -> dict:
        """
        Forecast regime transition probabilities using matrix exponentiation.

        P(state at t+n) = current_probs @ T^n

        Args:
            current_probs: Current posterior probability vector (from predict)
            horizons: List of forecast horizons in bars (default from config)

        Returns:
            dict with:
                - p_change: {horizon: probability of leaving current regime}
                - most_likely_next: label of most likely regime at longest horizon
                - p_bearish: probability of bearish regime at longest horizon
                - alert_level: 'stable', 'watch', 'warning', 'critical'
                - distributions: {horizon: {label: probability}}
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        if horizons is None:
            from config.settings import TRANSITION_FORECAST_HORIZONS
            horizons = TRANSITION_FORECAST_HORIZONS

        T = self.model.transmat_
        current_probs = np.asarray(current_probs, dtype=float)
        current_regime = int(np.argmax(current_probs))

        p_change = {}
        distributions = {}

        for h in sorted(horizons):
            T_h = np.linalg.matrix_power(T, h)
            future_probs = current_probs @ T_h

            # Probability of being in a DIFFERENT regime than current
            p_change[h] = float(1.0 - future_probs[current_regime])

            # Distribution at this horizon
            dist = {}
            for i in range(self.n_regimes):
                label = self.regime_labels.get(i, f"Regime {i}")
                dist[label] = float(future_probs[i])
            distributions[h] = dist

        # Most likely regime at longest horizon
        max_h = max(horizons)
        T_max = np.linalg.matrix_power(T, max_h)
        final_probs = current_probs @ T_max
        most_likely_idx = int(np.argmax(final_probs))
        most_likely_next = self.regime_labels.get(most_likely_idx, f"Regime {most_likely_idx}")

        # Probability of bearish regime at longest horizon
        from config.settings import BEARISH_REGIMES
        p_bearish = 0.0
        for i in range(self.n_regimes):
            label = self.regime_labels.get(i, f"Regime {i}")
            if label in BEARISH_REGIMES:
                p_bearish += float(final_probs[i])

        # Alert level based on P(change) at 6-hour horizon (or closest)
        ref_horizon = min(horizons, key=lambda h: abs(h - 6))
        p_ref = p_change[ref_horizon]

        if p_ref < 0.20:
            alert_level = "stable"
        elif p_ref < 0.40:
            alert_level = "watch"
        elif p_ref < 0.60:
            alert_level = "warning"
        else:
            alert_level = "critical"

        return {
            "p_change": p_change,
            "most_likely_next": most_likely_next,
            "p_bearish": p_bearish,
            "alert_level": alert_level,
            "distributions": distributions,
            "current_regime": self.regime_labels.get(current_regime, f"Regime {current_regime}"),
        }

    def get_transition_matrix(self) -> pd.DataFrame:
        """Return the transition probability matrix as a labeled DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        labels = [self.regime_labels.get(i, f"Regime {i}") for i in range(self.n_regimes)]
        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels,
        )

    def get_regime_stats(self, features: pd.DataFrame, ohlcv: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get summary statistics for each regime."""
        result = self.predict(features)
        merged = features.join(result[["regime", "regime_label"]])

        ret_col = "returns" if "returns" in features.columns else (
            "log_returns" if "log_returns" in features.columns else None
        )

        stats = []
        for regime in range(self.n_regimes):
            mask = merged["regime"] == regime
            regime_data = merged[mask]

            stat = {
                "regime": regime,
                "label": self.regime_labels.get(regime, f"Regime {regime}"),
                "days": mask.sum(),
                "pct_time": mask.sum() / len(merged) * 100 if len(merged) > 0 else 0,
            }

            if ret_col and ret_col in merged.columns:
                rets = regime_data[ret_col]
                stat["mean_return"] = rets.mean()
                stat["annualized_return"] = rets.mean() * 252
                stat["annualized_vol"] = rets.std() * np.sqrt(252)
                stat["sharpe"] = stat["annualized_return"] / stat["annualized_vol"] if stat["annualized_vol"] > 0 else 0
                stat["max_drawdown"] = (rets.cumsum() - rets.cumsum().cummax()).min()

            # Expected duration from transition matrix
            if self.is_fitted:
                self_prob = self.model.transmat_[regime, regime]
                stat["expected_duration"] = 1 / (1 - self_prob) if self_prob < 1 else np.inf
                stat["self_transition"] = self_prob

            stats.append(stat)

        return pd.DataFrame(stats)

    def model_selection(self, features: pd.DataFrame, max_regimes: int = 8) -> pd.DataFrame:
        """Test different numbers of regimes and return BIC/AIC scores."""
        cols = self.feature_names if self.feature_names else features.select_dtypes(include=[np.number]).columns
        X = features[cols].dropna()
        X_scaled = StandardScaler().fit_transform(X.values)

        results = []
        for n in range(2, max_regimes + 1):
            try:
                model = GaussianHMM(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                )
                model.fit(X_scaled)
                log_likelihood = model.score(X_scaled) * len(X_scaled)

                n_features = X_scaled.shape[1]
                n_params = (
                    n * n_features  # means
                    + n * n_features * (n_features + 1) / 2  # covariances
                    + n * (n - 1)  # transitions
                )
                bic = -2 * log_likelihood + n_params * np.log(len(X_scaled))
                aic = -2 * log_likelihood + 2 * n_params

                results.append({
                    "n_regimes": n,
                    "log_likelihood": log_likelihood,
                    "bic": bic,
                    "aic": aic,
                    "n_params": int(n_params),
                })
            except Exception as e:
                results.append({"n_regimes": n, "error": str(e)})

        return pd.DataFrame(results)

    def get_current_regime(self, features: pd.DataFrame) -> dict:
        """Get the current (most recent) regime state."""
        result = self.predict(features)
        last = result.dropna(subset=["regime"]).iloc[-1]

        # Count consecutive days in current regime
        regimes = result["regime"].dropna()
        current = regimes.iloc[-1]
        streak = 0
        for r in reversed(regimes.values):
            if r == current:
                streak += 1
            else:
                break

        # Count regime switches in last 90 rows
        recent = regimes.tail(90)
        switches = (recent != recent.shift(1)).sum() - 1  # first diff is always a "change"

        return {
            "regime": int(current),
            "label": self.regime_labels.get(int(current), f"Regime {int(current)}"),
            "confidence": float(last["confidence"]),
            "signal": last["signal"],
            "streak": streak,
            "switches_90": max(0, int(switches)),
            "probs": {
                self.regime_labels.get(i, f"Regime {i}"): float(last[f"prob_{i}"])
                for i in range(self.n_regimes)
            },
        }

    def _config_hash(self) -> str:
        """Short hash of model config for filename."""
        cfg = json.dumps({
            "n_regimes": self.n_regimes,
            "covariance_type": self.covariance_type,
            "features": self.feature_names,
        }, sort_keys=True)
        return hashlib.md5(cfg.encode()).hexdigest()[:8]

    def save(self, path: str | Path):
        """Save fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "regime_labels": self.regime_labels,
                "regime_order": self.regime_order,
                "n_regimes": self.n_regimes,
                "covariance_type": self.covariance_type,
                "saved_at": datetime.now().isoformat(),
                "config_hash": self._config_hash(),
            }, f)

    def save_latest(self, directory: str | Path | None = None):
        """Save model with timestamp + config hash to trained/ directory."""
        from config.settings import MODELS_DIR
        directory = Path(directory) if directory else MODELS_DIR
        directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_hash = self._config_hash()
        filename = f"regime_{self.n_regimes}s_{timestamp}_{cfg_hash}.pkl"
        path = directory / filename
        self.save(path)

        # Also save as "latest" symlink-style (just overwrite a known name)
        latest_path = directory / f"regime_{self.n_regimes}s_latest.pkl"
        self.save(latest_path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RegimeDetector":
        """Load a fitted model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        detector = cls(
            n_regimes=data["n_regimes"],
            covariance_type=data["covariance_type"],
        )
        detector.model = data["model"]
        detector.scaler = data["scaler"]
        detector.feature_names = data["feature_names"]
        detector.regime_labels = data["regime_labels"]
        detector.regime_order = data.get("regime_order", [])
        detector.is_fitted = True
        return detector

    @classmethod
    def load_latest(cls, n_regimes: int = 7, directory: str | Path | None = None) -> Optional["RegimeDetector"]:
        """Load the most recent model for given n_regimes. Returns None if not found."""
        from config.settings import MODELS_DIR
        directory = Path(directory) if directory else MODELS_DIR

        latest_path = directory / f"regime_{n_regimes}s_latest.pkl"
        if latest_path.exists():
            return cls.load(latest_path)

        # Fallback: find newest timestamped file
        pattern = f"regime_{n_regimes}s_*.pkl"
        files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            return cls.load(files[0])

        return None
