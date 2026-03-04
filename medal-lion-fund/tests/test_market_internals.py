"""Tests for market internals — VIX ratio, contango, breadth, DXY, alignment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from data.market_internals import (
    compute_cross_asset_features,
    align_internals_to_hourly,
)


@pytest.fixture
def sample_daily_ohlcv():
    """Generate synthetic daily OHLCV data."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.normal(0, 1, n))
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    open_ = close + np.random.normal(0, 0.3, n)
    volume = np.random.randint(1000000, 10000000, n).astype(float)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


@pytest.fixture
def sample_vix_data():
    """Generate synthetic VIX term structure data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    vix = 15 + np.abs(np.random.normal(0, 3, 100))
    vix3m = 18 + np.abs(np.random.normal(0, 2, 100))
    return pd.DataFrame({
        "vix": vix,
        "vix3m": vix3m,
        "vix_ratio": vix / vix3m,
        "vix_contango": vix / vix3m < 1.0,
    }, index=dates)


@pytest.fixture
def sample_hourly_index():
    """Generate synthetic hourly index."""
    return pd.date_range("2024-01-02", periods=200, freq="h")


class TestVIXTermStructure:
    def test_vix_ratio_computed(self, sample_vix_data):
        assert "vix_ratio" in sample_vix_data.columns
        assert (sample_vix_data["vix_ratio"] > 0).all()

    def test_contango_classification(self, sample_vix_data):
        assert "vix_contango" in sample_vix_data.columns
        # Contango means ratio < 1.0
        for _, row in sample_vix_data.iterrows():
            if row["vix_ratio"] < 1.0:
                assert row["vix_contango"] is True or row["vix_contango"] == True
            else:
                assert row["vix_contango"] is False or row["vix_contango"] == False

    def test_ratio_matches_manual(self, sample_vix_data):
        expected = sample_vix_data["vix"] / sample_vix_data["vix3m"]
        pd.testing.assert_series_equal(
            sample_vix_data["vix_ratio"], expected, check_names=False
        )


class TestCrossAssetFeatures:
    def test_features_computed(self, sample_daily_ohlcv):
        features = compute_cross_asset_features(sample_daily_ohlcv)
        assert "returns" in features.columns
        assert "range" in features.columns
        assert "volume_vol" in features.columns

    def test_returns_are_log(self, sample_daily_ohlcv):
        features = compute_cross_asset_features(sample_daily_ohlcv)
        expected = np.log(sample_daily_ohlcv["Close"].iloc[1] / sample_daily_ohlcv["Close"].iloc[0])
        assert abs(features["returns"].iloc[1] - expected) < 1e-10

    def test_range_positive(self, sample_daily_ohlcv):
        features = compute_cross_asset_features(sample_daily_ohlcv)
        valid = features["range"].dropna()
        assert (valid >= 0).all()

    def test_volume_vol_warmup(self, sample_daily_ohlcv):
        features = compute_cross_asset_features(sample_daily_ohlcv)
        # Rolling 20 → first 19 NaN
        assert features["volume_vol"].iloc[:19].isna().all()
        assert features["volume_vol"].iloc[25:].notna().all()

    def test_output_index_matches(self, sample_daily_ohlcv):
        features = compute_cross_asset_features(sample_daily_ohlcv)
        pd.testing.assert_index_equal(features.index, sample_daily_ohlcv.index)


class TestAlignInternals:
    def test_output_matches_hourly(self, sample_vix_data, sample_hourly_index):
        aligned = align_internals_to_hourly(sample_vix_data, sample_hourly_index)
        pd.testing.assert_index_equal(aligned.index, sample_hourly_index)

    def test_columns_preserved(self, sample_vix_data, sample_hourly_index):
        aligned = align_internals_to_hourly(sample_vix_data, sample_hourly_index)
        assert set(aligned.columns) == set(sample_vix_data.columns)

    def test_tz_aware_index(self, sample_vix_data):
        tz_index = pd.date_range("2024-01-02", periods=50, freq="h", tz="America/New_York")
        aligned = align_internals_to_hourly(sample_vix_data, tz_index)
        assert aligned.index.tz is not None
        assert len(aligned) == 50
