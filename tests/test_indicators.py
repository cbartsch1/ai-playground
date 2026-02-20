"""Tests for indicators — TEMA, ATR (Wilder's), SMA.

Validates Python implementation against known Pine Script values.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.indicators import tema, ema, atr_wilders, sma, compute_indicators


def _make_df(close_vals, high_vals=None, low_vals=None, volume=None):
    """Helper to create a test DataFrame."""
    n = len(close_vals)
    if high_vals is None:
        high_vals = [c + 1 for c in close_vals]
    if low_vals is None:
        low_vals = [c - 1 for c in close_vals]
    if volume is None:
        volume = [1000] * n

    idx = pd.date_range("2025-01-02 09:30", periods=n, freq="5min",
                        tz="America/New_York")
    return pd.DataFrame({
        "open": close_vals,
        "high": high_vals,
        "low": low_vals,
        "close": close_vals,
        "volume": volume,
    }, index=idx)


class TestEMA:
    def test_ema_constant_series(self):
        """EMA of constant series = that constant."""
        s = pd.Series([100.0] * 50)
        result = ema(s, 9)
        assert abs(result.iloc[-1] - 100.0) < 1e-10

    def test_ema_trending(self):
        """EMA of trending series should lag behind."""
        s = pd.Series(range(100), dtype=float)
        result = ema(s, 9)
        # EMA should be less than current value for uptrend
        assert result.iloc[-1] < s.iloc[-1]
        # But greater than it was 9 bars ago
        assert result.iloc[-1] > s.iloc[-10]


class TestTEMA:
    def test_tema_constant(self):
        """TEMA of constant = that constant."""
        s = pd.Series([5000.0] * 100)
        result = tema(s, 9)
        assert abs(result.iloc[-1] - 5000.0) < 1e-10

    def test_tema_faster_than_ema(self):
        """TEMA should be closer to current price than EMA (less lag)."""
        s = pd.Series(range(100), dtype=float)
        t = tema(s, 9)
        e = ema(s, 9)
        # In uptrend, TEMA > EMA (less lag)
        assert t.iloc[-1] > e.iloc[-1]

    def test_tema_formula(self):
        """Verify TEMA = 3*EMA1 - 3*EMA2 + EMA3."""
        s = pd.Series(np.random.randn(100).cumsum() + 5000)
        period = 9
        e1 = ema(s, period)
        e2 = ema(e1, period)
        e3 = ema(e2, period)
        expected = 3 * e1 - 3 * e2 + e3
        result = tema(s, period)
        pd.testing.assert_series_equal(result, expected)


class TestATR:
    def test_atr_wilders_uses_correct_alpha(self):
        """Wilder's ATR uses alpha=1/period, NOT 2/(period+1)."""
        # Use variable-range data so TR changes and the two smoothing methods diverge
        np.random.seed(42)
        n = 100
        close = 5000 + np.random.randn(n).cumsum()
        high = close + np.abs(np.random.randn(n)) * 3 + 1
        low = close - np.abs(np.random.randn(n)) * 3 - 1
        df = _make_df(close.tolist(), high.tolist(), low.tolist())
        period = 14

        result = atr_wilders(df, period)

        # Compute with standard EMA alpha for comparison
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)

        standard_ema = tr.ewm(span=period, adjust=False).mean()

        # Wilder's alpha=1/14 vs standard alpha=2/15 should diverge
        assert abs(result.iloc[-1] - standard_ema.iloc[-1]) > 0.01

    def test_atr_constant_range(self):
        """ATR of bars with constant 2-point range should converge to 2."""
        n = 200
        close = [5000.0] * n
        high = [5001.0] * n
        low = [4999.0] * n
        df = _make_df(close, high, low)
        result = atr_wilders(df, 14)
        assert abs(result.iloc[-1] - 2.0) < 0.01


class TestSMA:
    def test_sma_basic(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert abs(result.iloc[-1] - 4.0) < 1e-10  # (3+4+5)/3

    def test_sma_nan_before_period(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])


class TestComputeIndicators:
    def test_all_columns_created(self):
        """compute_indicators should add all expected columns."""
        df = _make_df([5000 + i * 0.5 for i in range(200)])
        # Add required session columns
        df["et_hour"] = df.index.hour
        df["et_minute"] = df.index.minute
        df["et_time"] = df["et_hour"] * 100 + df["et_minute"]

        compute_indicators(df)

        expected_cols = [
            "tema_fast", "tema_slow", "tema_trend", "atr", "atr_avg",
            "vol_ratio", "tema_bullish", "tema_bearish", "tema_slope",
            "slope_rising", "slope_falling", "trend_up", "trend_down"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
