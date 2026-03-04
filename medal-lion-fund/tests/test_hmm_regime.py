"""Tests for HMM RegimeDetector — fit/predict, auto-labeling, save/load."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil

from models.hmm_regime import RegimeDetector


@pytest.fixture
def sample_features():
    """Generate synthetic market data with clear regime structure."""
    np.random.seed(42)
    n = 500

    # Regime 1: low vol, positive returns (bull)
    r1 = np.random.normal(0.001, 0.005, n // 2)
    # Regime 2: high vol, negative returns (bear)
    r2 = np.random.normal(-0.002, 0.015, n // 2)

    returns = np.concatenate([r1, r2])
    range_vals = np.abs(returns) * 2 + np.random.uniform(0, 0.01, n)
    vol_vol = pd.Series(returns).rolling(20).std().fillna(0.005).values

    df = pd.DataFrame({
        "returns": returns,
        "range": range_vals,
        "volume_vol": vol_vol,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

    return df


@pytest.fixture
def tmp_model_dir():
    """Temp directory for model save/load tests."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


class TestRegimeDetectorFit:
    def test_fit_2_states(self, sample_features):
        det = RegimeDetector(n_regimes=2, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        assert det.is_fitted
        assert det.model is not None
        assert len(det.regime_labels) == 2

    def test_fit_7_states(self, sample_features):
        det = RegimeDetector(n_regimes=7, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        assert det.is_fitted
        assert len(det.regime_labels) == 7

    def test_fit_auto_labels_sorted_by_return(self, sample_features):
        det = RegimeDetector(n_regimes=3, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        labels = list(det.regime_labels.values())
        assert "Bear Trend" in labels
        assert "Bull Run (Trend)" in labels

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_fit_all_state_counts(self, sample_features, n):
        det = RegimeDetector(n_regimes=n, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        assert det.is_fitted
        assert len(det.regime_labels) == n


class TestRegimeDetectorPredict:
    def test_predict_returns_correct_columns(self, sample_features):
        det = RegimeDetector(n_regimes=3, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        result = det.predict(sample_features)
        assert "regime" in result.columns
        assert "regime_label" in result.columns
        assert "confidence" in result.columns
        assert "signal" in result.columns

    def test_predict_confidence_range(self, sample_features):
        det = RegimeDetector(n_regimes=3, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        result = det.predict(sample_features)
        valid_conf = result["confidence"].dropna()
        assert (valid_conf >= 0).all()
        assert (valid_conf <= 1).all()

    def test_predict_signal_values(self, sample_features):
        det = RegimeDetector(n_regimes=7, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        result = det.predict(sample_features)
        valid_signals = result["signal"].dropna().unique()
        for s in valid_signals:
            assert s in ("bullish", "bearish", "neutral")


class TestSaveLoad:
    def test_save_load_roundtrip(self, sample_features, tmp_model_dir):
        det = RegimeDetector(n_regimes=3, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])

        path = tmp_model_dir / "test_model.pkl"
        det.save(path)

        loaded = RegimeDetector.load(path)
        assert loaded.is_fitted
        assert loaded.n_regimes == 3
        assert loaded.regime_labels == det.regime_labels

    def test_save_latest_creates_files(self, sample_features, tmp_model_dir):
        det = RegimeDetector(n_regimes=3, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])

        path = det.save_latest(directory=tmp_model_dir)
        assert path.exists()

        latest = tmp_model_dir / "regime_3s_latest.pkl"
        assert latest.exists()

    def test_load_latest_roundtrip(self, sample_features, tmp_model_dir):
        det = RegimeDetector(n_regimes=4, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        det.save_latest(directory=tmp_model_dir)

        loaded = RegimeDetector.load_latest(n_regimes=4, directory=tmp_model_dir)
        assert loaded is not None
        assert loaded.n_regimes == 4

    def test_load_latest_returns_none_if_missing(self, tmp_model_dir):
        result = RegimeDetector.load_latest(n_regimes=7, directory=tmp_model_dir)
        assert result is None

    def test_predict_consistency_after_load(self, sample_features, tmp_model_dir):
        det = RegimeDetector(n_regimes=3, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        before = det.predict(sample_features)

        det.save_latest(directory=tmp_model_dir)
        loaded = RegimeDetector.load_latest(n_regimes=3, directory=tmp_model_dir)
        after = loaded.predict(sample_features)

        pd.testing.assert_series_equal(before["regime"], after["regime"])


class TestModelSelection:
    def test_model_selection_returns_bic_aic(self, sample_features):
        det = RegimeDetector(n_regimes=2, n_restarts=2, n_iter=50)
        det.fit(sample_features, feature_cols=["returns", "range", "volume_vol"])
        ms = det.model_selection(sample_features, max_regimes=4)
        assert "bic" in ms.columns
        assert "aic" in ms.columns
        assert len(ms) == 3  # 2, 3, 4
