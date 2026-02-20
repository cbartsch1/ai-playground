"""Tests for RegimeQualityAnalyzer — forward returns, stability, separation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from models.hmm_regime import RegimeDetector
from backtester.regime_quality import RegimeQualityAnalyzer


@pytest.fixture
def analysis_data():
    """Generate synthetic OHLCV + regime predictions."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    close = 500 + np.cumsum(np.random.normal(0, 1, n))
    ohlcv = pd.DataFrame({
        "Open": close + np.random.normal(0, 0.3, n),
        "High": close + np.abs(np.random.normal(0, 0.5, n)),
        "Low": close - np.abs(np.random.normal(0, 0.5, n)),
        "Close": close,
        "Volume": np.random.randint(100000, 1000000, n).astype(float),
    }, index=dates)

    features = pd.DataFrame({
        "returns": np.log(ohlcv["Close"] / ohlcv["Close"].shift(1)),
        "range": (ohlcv["High"] - ohlcv["Low"]) / ohlcv["Close"],
        "volume_vol": np.log(ohlcv["Volume"]).rolling(20).std(),
    }, index=dates)

    det = RegimeDetector(n_regimes=3, n_restarts=2, n_iter=50)
    det.fit(features.dropna(), feature_cols=["returns", "range", "volume_vol"])
    preds = det.predict(features)

    return ohlcv, preds


class TestForwardReturns:
    def test_has_all_horizons(self, analysis_data):
        ohlcv, preds = analysis_data
        analyzer = RegimeQualityAnalyzer()
        result = analyzer.analyze(ohlcv, preds)
        fwd = result.forward_returns
        assert "fwd_1h_mean" in fwd.columns
        assert "fwd_4h_mean" in fwd.columns
        assert "fwd_1d_mean" in fwd.columns
        assert "fwd_1w_mean" in fwd.columns

    def test_one_row_per_regime(self, analysis_data):
        ohlcv, preds = analysis_data
        analyzer = RegimeQualityAnalyzer()
        result = analyzer.analyze(ohlcv, preds)
        n_regimes = preds["regime"].dropna().nunique()
        assert len(result.forward_returns) == n_regimes


class TestStabilityMetrics:
    def test_false_alarm_rate_range(self, analysis_data):
        ohlcv, preds = analysis_data
        analyzer = RegimeQualityAnalyzer(min_regime_bars=2)
        result = analyzer.analyze(ohlcv, preds)
        stab = result.stability_metrics
        assert (stab["false_alarm_rate"] >= 0).all()
        assert (stab["false_alarm_rate"] <= 100).all()

    def test_avg_duration_positive(self, analysis_data):
        ohlcv, preds = analysis_data
        analyzer = RegimeQualityAnalyzer()
        result = analyzer.analyze(ohlcv, preds)
        stab = result.stability_metrics
        assert (stab["avg_duration"] > 0).all()

    def test_pct_time_sums_to_100(self, analysis_data):
        ohlcv, preds = analysis_data
        analyzer = RegimeQualityAnalyzer()
        result = analyzer.analyze(ohlcv, preds)
        stab = result.stability_metrics
        assert abs(stab["pct_time"].sum() - 100) < 1.0  # allow small rounding


class TestFilterValue:
    def test_filter_value_keys(self, analysis_data):
        ohlcv, preds = analysis_data
        analyzer = RegimeQualityAnalyzer()
        result = analyzer.analyze(ohlcv, preds)
        fv = result.filter_value
        assert "buy_hold_return" in fv
        assert "filtered_return" in fv
        assert "filtered_sharpe" in fv
        assert "time_in_market" in fv

    def test_time_in_market_range(self, analysis_data):
        ohlcv, preds = analysis_data
        analyzer = RegimeQualityAnalyzer()
        result = analyzer.analyze(ohlcv, preds)
        assert 0 <= result.filter_value["time_in_market"] <= 100


class TestRegimeSeparation:
    def test_has_signal_column(self, analysis_data):
        ohlcv, preds = analysis_data
        analyzer = RegimeQualityAnalyzer()
        result = analyzer.analyze(ohlcv, preds)
        sep = result.regime_separation
        assert "signal" in sep.columns
        assert "mean_return" in sep.columns


class TestSummaryScore:
    def test_score_range(self, analysis_data):
        ohlcv, preds = analysis_data
        analyzer = RegimeQualityAnalyzer()
        result = analyzer.analyze(ohlcv, preds)
        assert 0 <= result.summary_score <= 100
