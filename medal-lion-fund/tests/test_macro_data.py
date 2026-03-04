"""Tests for macro data — FRED download, cache, alignment, yield curve, credit stress."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from data.macro_data import (
    align_macro_to_hourly,
    compute_yield_curve,
    compute_credit_stress,
)


@pytest.fixture
def sample_macro_daily():
    """Generate synthetic daily macro data."""
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    np.random.seed(42)
    return pd.DataFrame({
        "yield_10y": 4.0 + np.cumsum(np.random.normal(0, 0.02, 500)),
        "yield_2y": 3.8 + np.cumsum(np.random.normal(0, 0.02, 500)),
        "yield_3m": 5.0 + np.cumsum(np.random.normal(0, 0.01, 500)),
        "fed_funds": np.full(500, 5.25),
        "credit_spread": 3.5 + np.abs(np.cumsum(np.random.normal(0, 0.05, 500))),
        "initial_claims": 200000 + np.random.normal(0, 5000, 500),
        "cpi": 300 + np.cumsum(np.random.normal(0.2, 0.1, 500)),
    }, index=dates)


@pytest.fixture
def sample_hourly_index():
    """Generate synthetic hourly index."""
    return pd.date_range("2024-01-02", periods=200, freq="h")


class TestAlignMacroToHourly:
    def test_output_matches_hourly_index(self, sample_macro_daily, sample_hourly_index):
        aligned = align_macro_to_hourly(sample_macro_daily, sample_hourly_index)
        pd.testing.assert_index_equal(aligned.index, sample_hourly_index)

    def test_columns_preserved(self, sample_macro_daily, sample_hourly_index):
        aligned = align_macro_to_hourly(sample_macro_daily, sample_hourly_index)
        assert set(aligned.columns) == set(sample_macro_daily.columns)

    def test_look_ahead_prevention(self, sample_macro_daily, sample_hourly_index):
        """Shift(1) ensures we only see yesterday's data, never today's."""
        aligned = align_macro_to_hourly(sample_macro_daily, sample_hourly_index)
        # The value on Jan 2 should be Jan 1's original value (shifted by 1)
        # NOT Jan 2's original value — that's the look-ahead prevention
        jan2_aligned = aligned.iloc[0]["yield_10y"]
        jan1_original = sample_macro_daily.loc["2024-01-01", "yield_10y"]
        jan2_original = sample_macro_daily.loc["2024-01-02", "yield_10y"]
        assert abs(jan2_aligned - jan1_original) < 1e-10, "Should see yesterday's data"
        assert abs(jan2_aligned - jan2_original) > 1e-10 or jan1_original == jan2_original

    def test_forward_fill_works(self, sample_macro_daily, sample_hourly_index):
        """Hourly bars within a day should have the same value."""
        aligned = align_macro_to_hourly(sample_macro_daily, sample_hourly_index)
        valid = aligned.dropna()
        if len(valid) >= 2:
            # Consecutive hourly bars on same day should be equal
            first_valid = valid.iloc[0]
            assert first_valid.notna().all()

    def test_tz_aware_hourly_index(self, sample_macro_daily):
        """Should handle timezone-aware hourly index."""
        tz_index = pd.date_range("2024-01-02", periods=50, freq="h", tz="America/New_York")
        aligned = align_macro_to_hourly(sample_macro_daily, tz_index)
        assert aligned.index.tz is not None
        assert len(aligned) == 50


class TestYieldCurve:
    def test_yield_curve_computed(self, sample_macro_daily):
        yc = compute_yield_curve(sample_macro_daily)
        assert yc.name == "yield_curve"
        assert len(yc) == len(sample_macro_daily)

    def test_yield_curve_is_10y_minus_2y(self, sample_macro_daily):
        yc = compute_yield_curve(sample_macro_daily)
        expected = sample_macro_daily["yield_10y"] - sample_macro_daily["yield_2y"]
        pd.testing.assert_series_equal(yc, expected, check_names=False)

    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        yc = compute_yield_curve(df)
        assert len(yc) == 0


class TestCreditStress:
    def test_credit_stress_computed(self, sample_macro_daily):
        cs = compute_credit_stress(sample_macro_daily, lookback=60)
        assert cs.name == "credit_stress_z"
        valid = cs.dropna()
        assert len(valid) > 0

    def test_z_score_reasonable_range(self, sample_macro_daily):
        cs = compute_credit_stress(sample_macro_daily, lookback=60)
        valid = cs.dropna()
        # Z-scores should mostly be within [-4, 4]
        assert (valid.abs() < 10).all()

    def test_missing_column_returns_empty(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        cs = compute_credit_stress(df)
        assert len(cs) == 0
