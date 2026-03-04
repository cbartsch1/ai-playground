"""
Medallion 2.0 — Regime Quality Analyzer

Replaces the naive backtester with metrics that measure whether
the regime detector is USEFUL as a trading filter:
  - Forward returns by regime (do regimes predict future returns?)
  - Regime stability (noise vs real regime changes)
  - Filter value score (regime-filtered vs buy-and-hold risk-adjusted)
  - Regime separation (are Bull/Bear statistically different?)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, field


@dataclass
class RegimeQualityResult:
    """Container for all regime quality metrics."""
    forward_returns: pd.DataFrame = None          # avg return by regime at various horizons
    stability_metrics: pd.DataFrame = None        # duration, false alarm rate per regime
    filter_value: dict = field(default_factory=dict)  # filtered vs buy-and-hold
    regime_separation: pd.DataFrame = None        # t-test between Bull/Bear returns
    summary_score: float = 0.0                    # overall quality 0-100


class RegimeQualityAnalyzer:
    """
    Analyze how useful the HMM regime detector is as a trading filter.
    Does NOT simulate trades — measures regime predictive power directly.
    """

    def __init__(self, min_regime_bars: int = 2):
        self.min_regime_bars = min_regime_bars  # regimes shorter than this = noise

    def analyze(
        self,
        ohlcv: pd.DataFrame,
        regime_predictions: pd.DataFrame,
    ) -> RegimeQualityResult:
        """
        Run all quality analyses.

        Args:
            ohlcv: OHLCV price data with DatetimeIndex
            regime_predictions: Output from RegimeDetector.predict()

        Returns:
            RegimeQualityResult with all metrics
        """
        idx = ohlcv.index.intersection(regime_predictions.index)
        prices = ohlcv.loc[idx, "Close"]
        regimes = regime_predictions.loc[idx]

        result = RegimeQualityResult()
        result.forward_returns = self._forward_returns(prices, regimes)
        result.stability_metrics = self._stability_metrics(regimes)
        result.filter_value = self._filter_value(prices, regimes)
        result.regime_separation = self._regime_separation(prices, regimes)
        result.summary_score = self._compute_summary_score(result)
        return result

    def _forward_returns(self, prices: pd.Series, regimes: pd.DataFrame) -> pd.DataFrame:
        """
        Compute average forward returns by regime at multiple horizons.
        Horizons: +1h, +4h, +1d (6.5 bars), +1w (32.5 bars)
        """
        horizons = {"1h": 1, "4h": 4, "1d": 7, "1w": 33}
        log_returns = np.log(prices / prices.shift(1))

        records = []
        for regime_id in sorted(regimes["regime"].dropna().unique()):
            regime_id = int(regime_id)
            mask = regimes["regime"] == regime_id
            label = regimes.loc[mask, "regime_label"].iloc[0] if mask.any() else f"Regime {regime_id}"

            row = {"regime": regime_id, "label": label, "count": mask.sum()}
            for name, bars in horizons.items():
                # Forward return from each bar in this regime
                fwd = prices.shift(-bars) / prices - 1
                regime_fwd = fwd[mask].dropna()
                row[f"fwd_{name}_mean"] = regime_fwd.mean() * 100 if len(regime_fwd) > 0 else np.nan
                row[f"fwd_{name}_std"] = regime_fwd.std() * 100 if len(regime_fwd) > 0 else np.nan
                row[f"fwd_{name}_n"] = len(regime_fwd)
            records.append(row)

        return pd.DataFrame(records)

    def _stability_metrics(self, regimes: pd.DataFrame) -> pd.DataFrame:
        """
        Measure regime stability: average duration, false alarm rate.
        A regime lasting < min_regime_bars is a false alarm.
        """
        regime_col = regimes["regime"].dropna()
        label_col = regimes["regime_label"].reindex(regime_col.index)
        if len(regime_col) == 0:
            return pd.DataFrame()

        # Identify regime runs
        shifts = regime_col != regime_col.shift(1)
        run_ids = shifts.cumsum()

        records = []
        for regime_id in sorted(regime_col.unique()):
            regime_id = int(regime_id)
            mask = regime_col == regime_id
            label = label_col[mask].iloc[0] if mask.any() else f"Regime {regime_id}"

            # Get durations of each run of this regime
            regime_runs = run_ids[mask]
            run_lengths = regime_runs.groupby(regime_runs).count()

            total_runs = len(run_lengths)
            false_alarms = int((run_lengths < self.min_regime_bars).sum())

            records.append({
                "regime": regime_id,
                "label": label,
                "total_bars": int(mask.sum()),
                "pct_time": mask.sum() / len(regime_col) * 100,
                "num_episodes": total_runs,
                "avg_duration": float(run_lengths.mean()) if total_runs > 0 else 0,
                "median_duration": float(run_lengths.median()) if total_runs > 0 else 0,
                "max_duration": int(run_lengths.max()) if total_runs > 0 else 0,
                "min_duration": int(run_lengths.min()) if total_runs > 0 else 0,
                "false_alarms": false_alarms,
                "false_alarm_rate": false_alarms / total_runs * 100 if total_runs > 0 else 0,
            })

        return pd.DataFrame(records)

    def _filter_value(self, prices: pd.Series, regimes: pd.DataFrame) -> dict:
        """
        Compare holding only during bullish regimes vs buy-and-hold.
        Shows how much value the regime filter adds as a risk reducer.
        """
        from config.settings import BULLISH_REGIMES, BEARISH_REGIMES

        returns = prices.pct_change().dropna()
        aligned_signals = regimes["signal"].reindex(returns.index).shift(1)  # trade on NEXT bar

        # Buy-and-hold
        bh_total = (1 + returns).prod() - 1
        bh_vol = returns.std() * np.sqrt(252 * 6.5)  # annualized (hourly)
        bh_sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 6.5)) if returns.std() > 0 else 0

        # Filtered: only hold during bullish regimes
        bullish_mask = aligned_signals == "bullish"
        filtered_returns = returns.where(bullish_mask, 0)
        filtered_total = (1 + filtered_returns).prod() - 1
        filtered_vol = filtered_returns[filtered_returns != 0].std() * np.sqrt(252 * 6.5) if (filtered_returns != 0).any() else 0
        filtered_sharpe = (filtered_returns.mean() / filtered_returns.std() * np.sqrt(252 * 6.5)) if filtered_returns.std() > 0 else 0

        # Time in market
        time_in_market = bullish_mask.sum() / len(bullish_mask) * 100

        # Max drawdown comparison
        bh_equity = (1 + returns).cumprod()
        bh_dd = (bh_equity / bh_equity.cummax() - 1).min() * 100

        filt_equity = (1 + filtered_returns).cumprod()
        filt_dd = (filt_equity / filt_equity.cummax() - 1).min() * 100

        return {
            "buy_hold_return": bh_total * 100,
            "buy_hold_vol": bh_vol * 100,
            "buy_hold_sharpe": bh_sharpe,
            "buy_hold_max_dd": bh_dd,
            "filtered_return": filtered_total * 100,
            "filtered_vol": filtered_vol * 100,
            "filtered_sharpe": filtered_sharpe,
            "filtered_max_dd": filt_dd,
            "time_in_market": time_in_market,
            "return_per_exposure": (filtered_total * 100) / time_in_market * 100 if time_in_market > 0 else 0,
        }

    def _regime_separation(self, prices: pd.Series, regimes: pd.DataFrame) -> pd.DataFrame:
        """
        Test if different regime signals have statistically different returns.
        Uses Welch's t-test between bullish and bearish regime bars.
        """
        returns = prices.pct_change().dropna()
        aligned = regimes["signal"].reindex(returns.index)

        records = []
        signal_groups = {}
        for signal in ["bullish", "bearish", "neutral"]:
            mask = aligned == signal
            if mask.sum() > 0:
                signal_groups[signal] = returns[mask]
                records.append({
                    "signal": signal,
                    "bars": mask.sum(),
                    "mean_return": returns[mask].mean() * 100,
                    "std_return": returns[mask].std() * 100,
                    "median_return": returns[mask].median() * 100,
                })

        # T-test: bullish vs bearish
        if "bullish" in signal_groups and "bearish" in signal_groups:
            t_stat, p_val = stats.ttest_ind(
                signal_groups["bullish"],
                signal_groups["bearish"],
                equal_var=False,
            )
            for r in records:
                if r["signal"] == "bullish":
                    r["t_stat_vs_bearish"] = t_stat
                    r["p_value_vs_bearish"] = p_val
                    r["significant"] = p_val < 0.05

        return pd.DataFrame(records)

    def _compute_summary_score(self, result: RegimeQualityResult) -> float:
        """
        Compute a 0-100 quality score based on all metrics.
        Weighted: separation (40%), stability (30%), filter value (30%).
        """
        score = 0.0

        # Separation score (0-40): are regimes meaningfully different?
        if result.regime_separation is not None and len(result.regime_separation) > 0:
            bull_row = result.regime_separation[result.regime_separation["signal"] == "bullish"]
            if not bull_row.empty and "p_value_vs_bearish" in bull_row.columns:
                p = bull_row["p_value_vs_bearish"].iloc[0]
                if pd.notna(p):
                    if p < 0.01:
                        score += 40
                    elif p < 0.05:
                        score += 30
                    elif p < 0.10:
                        score += 20
                    else:
                        score += 5

        # Stability score (0-30): low false alarm rate = good
        if result.stability_metrics is not None and len(result.stability_metrics) > 0:
            avg_fa_rate = result.stability_metrics["false_alarm_rate"].mean()
            stability = max(0, 30 * (1 - avg_fa_rate / 100))
            score += stability

        # Filter value score (0-30): does filtering improve Sharpe?
        fv = result.filter_value
        if fv:
            bh_sharpe = fv.get("buy_hold_sharpe", 0)
            filt_sharpe = fv.get("filtered_sharpe", 0)
            if filt_sharpe > bh_sharpe:
                improvement = min(filt_sharpe - bh_sharpe, 1.0)
                score += 30 * improvement
            elif filt_sharpe > 0:
                score += 10

        return round(score, 1)
