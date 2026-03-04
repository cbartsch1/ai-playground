"""Tests for data loader — feature computation, confirmations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from data.data_loader import compute_hmm_features, compute_confirmations


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    close = 500 + np.cumsum(np.random.normal(0, 1, n))
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    open_ = close + np.random.normal(0, 0.3, n)
    volume = np.random.randint(100000, 1000000, n).astype(float)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


class TestHMMFeatures:
    def test_returns_computed(self, sample_ohlcv):
        features = compute_hmm_features(sample_ohlcv)
        assert "returns" in features.columns
        # First row should be NaN (no previous close)
        assert pd.isna(features["returns"].iloc[0])
        # Should be log returns
        expected = np.log(sample_ohlcv["Close"].iloc[1] / sample_ohlcv["Close"].iloc[0])
        assert abs(features["returns"].iloc[1] - expected) < 1e-10

    def test_range_computed(self, sample_ohlcv):
        features = compute_hmm_features(sample_ohlcv)
        assert "range" in features.columns
        # Range = (High - Low) / Close
        expected = (sample_ohlcv["High"].iloc[0] - sample_ohlcv["Low"].iloc[0]) / sample_ohlcv["Close"].iloc[0]
        assert abs(features["range"].iloc[0] - expected) < 1e-10
        # Should always be positive
        assert (features["range"].dropna() >= 0).all()

    def test_volume_vol_computed(self, sample_ohlcv):
        features = compute_hmm_features(sample_ohlcv)
        assert "volume_vol" in features.columns
        # First 19 rows should be NaN (rolling window = 20)
        assert features["volume_vol"].iloc[:19].isna().all()
        # After warmup, should have values
        assert features["volume_vol"].iloc[20:].notna().all()

    def test_output_index_matches(self, sample_ohlcv):
        features = compute_hmm_features(sample_ohlcv)
        pd.testing.assert_index_equal(features.index, sample_ohlcv.index)


class TestConfirmations:
    def test_all_8_pass_columns(self, sample_ohlcv):
        confs = compute_confirmations(sample_ohlcv)
        pass_cols = [c for c in confs.columns if c.endswith("_pass")]
        assert len(pass_cols) == 8

    def test_confirmations_met_range(self, sample_ohlcv):
        confs = compute_confirmations(sample_ohlcv)
        valid = confs["confirmations_met"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 8).all()

    def test_rsi_range(self, sample_ohlcv):
        confs = compute_confirmations(sample_ohlcv)
        valid_rsi = confs["rsi"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_adx_non_negative(self, sample_ohlcv):
        confs = compute_confirmations(sample_ohlcv)
        valid_adx = confs["adx"].dropna()
        assert (valid_adx >= 0).all()

    def test_confirmation_pass_is_boolean(self, sample_ohlcv):
        confs = compute_confirmations(sample_ohlcv)
        for col in [c for c in confs.columns if c.endswith("_pass")]:
            valid = confs[col].dropna()
            assert valid.isin([True, False]).all(), f"{col} has non-boolean values"

    def test_output_index_matches(self, sample_ohlcv):
        confs = compute_confirmations(sample_ohlcv)
        pd.testing.assert_index_equal(confs.index, sample_ohlcv.index)
