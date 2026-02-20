"""Tests for RegimeFilter — alignment, position sizing, should_trade, alerts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from models.hmm_regime import RegimeDetector
from models.regime_api import RegimeFilter


@pytest.fixture
def fitted_detector():
    """Create and fit a detector on synthetic data."""
    np.random.seed(42)
    n = 500
    returns = np.concatenate([
        np.random.normal(0.001, 0.005, n // 2),
        np.random.normal(-0.002, 0.015, n // 2),
    ])
    df = pd.DataFrame({
        "returns": returns,
        "range": np.abs(returns) * 2 + np.random.uniform(0, 0.01, n),
        "volume_vol": pd.Series(returns).rolling(20).std().fillna(0.005).values,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

    det = RegimeDetector(n_regimes=3, n_restarts=2, n_iter=50)
    det.fit(df, feature_cols=["returns", "range", "volume_vol"])
    return det, df


@pytest.fixture
def regime_filter(fitted_detector):
    """Create a RegimeFilter from fitted detector."""
    det, features = fitted_detector
    return RegimeFilter(det, features)


class TestAlignment:
    def test_align_to_5min(self, regime_filter, fitted_detector):
        _, features = fitted_detector
        # Create 5-min target index spanning same period
        target = pd.DataFrame(
            index=pd.date_range(features.index[0], features.index[-1], freq="5min"),
            data={"price": 100.0},
        )
        aligned = regime_filter.align_to_timeframe(target)
        assert len(aligned) == len(target)
        assert "regime_label" in aligned.columns
        # Forward fill should produce no NaN in the middle
        mid = len(aligned) // 2
        assert pd.notna(aligned["regime_label"].iloc[mid])

    def test_add_regime_columns(self, regime_filter, fitted_detector):
        _, features = fitted_detector
        target = pd.DataFrame(
            index=features.index,
            data={"price": np.random.randn(len(features))},
        )
        result = regime_filter.add_regime_columns(target)
        assert "regime_label" in result.columns
        assert "regime_signal" in result.columns
        assert "regime_confidence" in result.columns


class TestGetRegimeAt:
    def test_valid_timestamp(self, regime_filter, fitted_detector):
        _, features = fitted_detector
        ts = features.index[100]
        regime = regime_filter.get_regime_at(ts)
        assert "label" in regime
        assert "signal" in regime
        assert "confidence" in regime
        assert regime["confidence"] >= 0

    def test_between_bars(self, regime_filter, fitted_detector):
        _, features = fitted_detector
        # Query between two hourly bars
        ts = features.index[50] + pd.Timedelta(minutes=30)
        regime = regime_filter.get_regime_at(ts)
        assert regime["label"] is not None

    def test_before_data_returns_none(self, regime_filter):
        regime = regime_filter.get_regime_at(pd.Timestamp("2020-01-01"))
        assert regime["label"] is None


class TestPositionSizing:
    def test_multiplier_range(self, regime_filter, fitted_detector):
        _, features = fitted_detector
        ts = features.index[100]
        mult = regime_filter.position_size_multiplier(ts)
        assert 0.0 <= mult <= 1.0

    def test_high_confidence_full_size(self, regime_filter, fitted_detector):
        """If any bar has >90% confidence, it should get 1.0 multiplier."""
        _, features = fitted_detector
        preds = regime_filter.predictions
        high_conf = preds[preds["confidence"] > 0.9]
        if len(high_conf) > 0:
            ts = high_conf.index[0]
            assert regime_filter.position_size_multiplier(ts) == 1.0


class TestShouldTrade:
    def test_returns_boolean(self, regime_filter, fitted_detector):
        _, features = fitted_detector
        ts = features.index[100]
        result = regime_filter.should_trade(ts, direction="short")
        assert isinstance(result, bool)


class TestRegimeChangeAlert:
    def test_no_change_returns_none(self, regime_filter):
        result = regime_filter.check_regime_change("Bull Run (Trend)", "Bull Run (Trend)")
        assert result is None

    def test_change_returns_alert(self, regime_filter):
        result = regime_filter.check_regime_change("Bull Run (Trend)", "Bear Trend")
        assert result is not None
        assert result["severity"] == "critical"
        assert "Bull Run" in result["message"]
        assert "Bear Trend" in result["message"]

    def test_info_severity_within_group(self, regime_filter):
        result = regime_filter.check_regime_change("Bull Run (Trend)", "Recovery")
        assert result is not None
        assert result["severity"] == "info"

    def test_warning_severity_neutral_change(self, regime_filter):
        result = regime_filter.check_regime_change("Accumulation (Chop)", "Bear Trend")
        assert result is not None
        assert result["severity"] == "warning"

    def test_none_inputs(self, regime_filter):
        assert regime_filter.check_regime_change(None, "Bull Run (Trend)") is None
        assert regime_filter.check_regime_change("Bull Run (Trend)", None) is None
