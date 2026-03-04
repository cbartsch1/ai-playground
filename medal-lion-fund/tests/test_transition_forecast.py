"""Tests for transition forecasting — matrix power, probabilities, alerts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from models.hmm_regime import RegimeDetector


@pytest.fixture
def fitted_detector():
    """Create a fitted 3-state RegimeDetector with synthetic data."""
    np.random.seed(42)
    n = 500
    # Generate features with regime-like structure
    returns = np.concatenate([
        np.random.normal(-0.01, 0.02, n // 3),
        np.random.normal(0.0, 0.01, n // 3),
        np.random.normal(0.01, 0.015, n - 2 * (n // 3)),
    ])
    range_vals = np.abs(returns) + np.random.uniform(0, 0.005, n)
    vol = np.random.uniform(0.1, 0.5, n)

    features = pd.DataFrame({
        "returns": returns,
        "range": range_vals,
        "volume_vol": vol,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

    detector = RegimeDetector(n_regimes=3, n_restarts=3, n_iter=50)
    detector.fit(features, feature_cols=["returns", "range", "volume_vol"])
    return detector, features


class TestForecastTransitions:
    def test_forecast_returns_dict(self, fitted_detector):
        detector, features = fitted_detector
        result = detector.predict(features)
        # Get current posterior probabilities
        last_row = result.dropna(subset=["confidence"]).iloc[-1]
        current_probs = np.array([last_row[f"prob_{i}"] for i in range(3)])

        forecast = detector.forecast_transitions(current_probs)
        assert isinstance(forecast, dict)
        assert "p_change" in forecast
        assert "most_likely_next" in forecast
        assert "alert_level" in forecast

    def test_p_change_in_valid_range(self, fitted_detector):
        detector, features = fitted_detector
        result = detector.predict(features)
        last_row = result.dropna(subset=["confidence"]).iloc[-1]
        current_probs = np.array([last_row[f"prob_{i}"] for i in range(3)])

        forecast = detector.forecast_transitions(current_probs)
        for horizon, p in forecast["p_change"].items():
            assert 0.0 <= p <= 1.0, f"P(change) at h={horizon} = {p} out of range"

    def test_p_change_increases_with_horizon(self, fitted_detector):
        detector, features = fitted_detector
        result = detector.predict(features)
        last_row = result.dropna(subset=["confidence"]).iloc[-1]
        current_probs = np.array([last_row[f"prob_{i}"] for i in range(3)])

        forecast = detector.forecast_transitions(current_probs)
        horizons = sorted(forecast["p_change"].keys())
        # P(change) should generally increase (or stay same) with longer horizons
        for i in range(len(horizons) - 1):
            h1, h2 = horizons[i], horizons[i + 1]
            # Allow small tolerance for numerical precision
            assert forecast["p_change"][h2] >= forecast["p_change"][h1] - 0.01, \
                f"P(change) decreased: h={h1}:{forecast['p_change'][h1]:.3f} > h={h2}:{forecast['p_change'][h2]:.3f}"

    def test_alert_levels_valid(self, fitted_detector):
        detector, features = fitted_detector
        result = detector.predict(features)
        last_row = result.dropna(subset=["confidence"]).iloc[-1]
        current_probs = np.array([last_row[f"prob_{i}"] for i in range(3)])

        forecast = detector.forecast_transitions(current_probs)
        valid_levels = {"stable", "watch", "warning", "critical"}
        assert forecast["alert_level"] in valid_levels

    def test_matrix_power_matches_manual(self, fitted_detector):
        """Verify T^n computation matches manual matrix multiplication."""
        detector, _ = fitted_detector
        T = detector.model.transmat_
        # Manual T^2
        T2_manual = T @ T
        T2_power = np.linalg.matrix_power(T, 2)
        np.testing.assert_array_almost_equal(T2_manual, T2_power, decimal=10)

    def test_forecast_probabilities_sum_to_one(self, fitted_detector):
        """Forecasted probability vectors should sum to 1.0."""
        detector, features = fitted_detector
        result = detector.predict(features)
        last_row = result.dropna(subset=["confidence"]).iloc[-1]
        current_probs = np.array([last_row[f"prob_{i}"] for i in range(3)])

        forecast = detector.forecast_transitions(current_probs)
        for horizon_dist in forecast.get("distributions", {}).values():
            total = sum(horizon_dist.values())
            assert abs(total - 1.0) < 1e-6, f"Distribution sums to {total}"

    def test_most_likely_next_is_valid_label(self, fitted_detector):
        detector, features = fitted_detector
        result = detector.predict(features)
        last_row = result.dropna(subset=["confidence"]).iloc[-1]
        current_probs = np.array([last_row[f"prob_{i}"] for i in range(3)])

        forecast = detector.forecast_transitions(current_probs)
        valid_labels = set(detector.regime_labels.values())
        assert forecast["most_likely_next"] in valid_labels

    def test_pure_state_has_low_initial_p_change(self, fitted_detector):
        """A pure state (100% in one regime) should have low P(change) at h=1."""
        detector, _ = fitted_detector
        pure_probs = np.array([1.0, 0.0, 0.0])
        forecast = detector.forecast_transitions(pure_probs)
        # At h=1, P(change) = 1 - T[0,0] (self-transition probability)
        expected = 1 - detector.model.transmat_[0, 0]
        assert abs(forecast["p_change"][1] - expected) < 0.01
