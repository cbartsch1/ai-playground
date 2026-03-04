"""
Medallion 2.0 — Regime Integration API

Bridge between hourly HMM regime detection and trading strategies
running on different timeframes (5m ES, 1m SPY, etc.).

Usage:
    from models.regime_api import RegimeFilter
    rf = RegimeFilter(detector, hmm_features)
    rf.should_trade(timestamp, direction="short")
    rf.position_size_multiplier(timestamp)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime


class RegimeFilter:
    """
    Clean bridge between HMM regime detector and trading strategies.
    Handles timeframe alignment, confidence sizing, and trade gating.
    """

    def __init__(self, detector, hmm_features: pd.DataFrame):
        """
        Args:
            detector: Fitted RegimeDetector instance
            hmm_features: DataFrame used for prediction (with HMM feature columns)
        """
        from config.settings import CONFIDENCE_TIERS, BULLISH_REGIMES, BEARISH_REGIMES

        self.detector = detector
        self.predictions = detector.predict(hmm_features)
        self.confidence_tiers = CONFIDENCE_TIERS
        self.bullish_regimes = BULLISH_REGIMES
        self.bearish_regimes = BEARISH_REGIMES

    def align_to_timeframe(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill hourly regime data to match any target timeframe.
        Handles timezone alignment automatically.

        Args:
            target_df: DataFrame with the target timeframe index (e.g., 5m ES bars)

        Returns:
            DataFrame with regime columns aligned to target index
        """
        regime_cols = ["regime", "regime_label", "confidence", "signal"]
        available = [c for c in regime_cols if c in self.predictions.columns]
        regime_data = self.predictions[available].copy()

        # Localize if needed for alignment
        if regime_data.index.tz is not None and target_df.index.tz is None:
            regime_data.index = regime_data.index.tz_localize(None)
        elif regime_data.index.tz is None and target_df.index.tz is not None:
            regime_data.index = regime_data.index.tz_localize(target_df.index.tz)

        # Reindex to target with forward-fill (hourly regime applies until next hourly bar)
        aligned = regime_data.reindex(target_df.index, method="ffill")
        return aligned

    def add_regime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime_label, regime_signal, regime_confidence columns to any DataFrame.

        Args:
            df: Trading DataFrame with DatetimeIndex

        Returns:
            Same DataFrame with 3 new columns added
        """
        aligned = self.align_to_timeframe(df)
        df = df.copy()
        df["regime_label"] = aligned["regime_label"]
        df["regime_signal"] = aligned["signal"]
        df["regime_confidence"] = aligned["confidence"]
        return df

    def get_regime_at(self, timestamp: pd.Timestamp | datetime) -> dict:
        """
        Get regime state at a specific timestamp.

        Args:
            timestamp: Point in time to query

        Returns:
            Dict with label, signal, confidence
        """
        ts = pd.Timestamp(timestamp)

        # Localize if needed
        if self.predictions.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(self.predictions.index.tz)
        elif self.predictions.index.tz is None and ts.tz is not None:
            ts = ts.tz_localize(None)

        # Find nearest preceding bar
        mask = self.predictions.index <= ts
        if not mask.any():
            return {"label": None, "signal": None, "confidence": 0.0}

        row = self.predictions.loc[mask].iloc[-1]
        return {
            "label": row.get("regime_label"),
            "signal": row.get("signal"),
            "confidence": float(row.get("confidence", 0)) if pd.notna(row.get("confidence")) else 0.0,
        }

    def should_trade(self, timestamp: pd.Timestamp | datetime, direction: str = "short") -> bool:
        """
        Check if trading is allowed at this timestamp given regime + confidence.

        For shorts: allowed during bearish regimes with sufficient confidence
        For longs: allowed during bullish regimes with sufficient confidence

        Args:
            timestamp: When to check
            direction: "short" or "long"

        Returns:
            True if regime supports trading in this direction
        """
        regime = self.get_regime_at(timestamp)
        confidence = regime["confidence"]
        signal = regime["signal"]

        # Minimum confidence threshold (lowest non-zero tier)
        min_conf = min(k for k, v in self.confidence_tiers.items() if v > 0)

        if confidence < min_conf:
            return False

        if direction == "short":
            return signal == "bearish"
        elif direction == "long":
            return signal == "bullish"
        else:
            return signal in ("bullish", "bearish")

    def position_size_multiplier(self, timestamp: pd.Timestamp | datetime) -> float:
        """
        Get position size multiplier (0.0 to 1.0) based on regime confidence.

        Tiers (default):
            90%+ → 1.0x (full size)
            70-90% → 0.75x
            50-70% → 0.5x
            <50% → 0.0 (skip)

        Args:
            timestamp: When to check

        Returns:
            Multiplier from 0.0 to 1.0
        """
        regime = self.get_regime_at(timestamp)
        confidence = regime["confidence"]

        # Walk tiers from highest to lowest
        for threshold in sorted(self.confidence_tiers.keys(), reverse=True):
            if confidence >= threshold:
                return self.confidence_tiers[threshold]

        return 0.0

    def check_regime_change(
        self, previous_regime: Optional[str], current_regime: Optional[str]
    ) -> Optional[dict]:
        """
        Detect actionable regime changes.

        Returns alert dict if regime changed, None otherwise.
        Classifies severity: critical (bull→crash), warning (bull→bear), info (within same signal).
        """
        if previous_regime is None or current_regime is None:
            return None
        if previous_regime == current_regime:
            return None

        prev_signal = self._classify_signal(previous_regime)
        curr_signal = self._classify_signal(current_regime)

        # Determine severity
        if prev_signal == "bullish" and curr_signal == "bearish":
            severity = "critical"
        elif prev_signal == "bearish" and curr_signal == "bullish":
            severity = "critical"
        elif prev_signal != curr_signal:
            severity = "warning"
        else:
            severity = "info"

        return {
            "previous": previous_regime,
            "current": current_regime,
            "previous_signal": prev_signal,
            "current_signal": curr_signal,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "message": f"Regime changed: {previous_regime} → {current_regime} ({prev_signal} → {curr_signal})",
        }

    def transition_risk(self, timestamp: pd.Timestamp | datetime = None) -> dict:
        """
        Get transition risk at a timestamp (or latest).

        Returns:
            dict with p_change at each horizon, alert_level, current_regime, most_likely_next
        """
        if timestamp is None:
            last = self.predictions.dropna(subset=["confidence"]).iloc[-1]
        else:
            ts = pd.Timestamp(timestamp)
            if self.predictions.index.tz is not None and ts.tz is None:
                ts = ts.tz_localize(self.predictions.index.tz)
            elif self.predictions.index.tz is None and ts.tz is not None:
                ts = ts.tz_localize(None)
            mask = self.predictions.index <= ts
            if not mask.any():
                return {"alert_level": "stable", "p_change": {}, "current_regime": None}
            last = self.predictions.loc[mask].dropna(subset=["confidence"]).iloc[-1]

        current_probs = np.array([
            last[f"prob_{i}"] for i in range(self.detector.n_regimes)
        ])
        return self.detector.forecast_transitions(current_probs)

    def should_reduce_size(self, timestamp: pd.Timestamp | datetime = None) -> bool:
        """
        Check if position size should be reduced due to high transition risk.

        Returns True if alert_level is 'warning' or 'critical'.
        """
        risk = self.transition_risk(timestamp)
        return risk.get("alert_level") in ("warning", "critical")

    @staticmethod
    def is_macro_event_day(date: pd.Timestamp | datetime | str, event_type: str = None) -> dict:
        """
        Check if a date has a macro event (FOMC, CPI, NFP).

        Args:
            date: Date to check
            event_type: Specific type or None for all

        Returns:
            dict with {event_type: True/False} for each event type
        """
        from config.settings import MACRO_EVENTS_2025_2026

        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        result = {}
        for etype, dates in MACRO_EVENTS_2025_2026.items():
            if event_type and etype != event_type:
                continue
            result[etype] = date_str in dates
        return result

    @staticmethod
    def cross_asset_alignment(cross_asset_regimes: dict[str, dict]) -> dict:
        """
        Analyze regime alignment across multiple assets.

        Args:
            cross_asset_regimes: {ticker: regime_dict} from get_current_regime()

        Returns:
            dict with alignment status, bullish/bearish counts, risk_off flag
        """
        from config.settings import BULLISH_REGIMES, BEARISH_REGIMES

        signals = {}
        for ticker, regime in cross_asset_regimes.items():
            label = regime.get("label", "")
            if label in BULLISH_REGIMES:
                signals[ticker] = "bullish"
            elif label in BEARISH_REGIMES:
                signals[ticker] = "bearish"
            else:
                signals[ticker] = "neutral"

        bullish = sum(1 for s in signals.values() if s == "bullish")
        bearish = sum(1 for s in signals.values() if s == "bearish")
        total = len(signals)

        # Cross-asset risk-off: 3+ assets bearish
        risk_off = bearish >= 3

        if bullish >= total - 1:
            alignment = "ALIGNED_BULLISH"
        elif bearish >= total - 1:
            alignment = "ALIGNED_BEARISH"
        else:
            alignment = "DIVERGING"

        return {
            "alignment": alignment,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": total - bullish - bearish,
            "total_assets": total,
            "risk_off": risk_off,
            "signals": signals,
        }

    def _classify_signal(self, label: str) -> str:
        """Classify a regime label as bullish/bearish/neutral."""
        if label in self.bullish_regimes:
            return "bullish"
        elif label in self.bearish_regimes:
            return "bearish"
        return "neutral"
