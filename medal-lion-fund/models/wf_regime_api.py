"""
Medallion 2.0 — Walk-Forward Regime Filter (Bias-Free)

Drop-in replacement for RegimeFilter that reads from pre-computed
walk-forward regime labels instead of a live HMM model.

Every bar's regime label was computed by a model that ONLY saw prior data,
eliminating look-ahead bias from backtesting.

Usage:
    from models.wf_regime_api import WalkForwardRegimeFilter
    rf = WalkForwardRegimeFilter("data/processed/walk_forward_regimes.parquet")
    rf.get_regime_at(timestamp)        # Same API as RegimeFilter
    rf.should_trade(timestamp, "short")
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from pathlib import Path


class WalkForwardRegimeFilter:
    """
    Bias-free regime filter using pre-computed walk-forward labels.

    Implements the same public API as RegimeFilter so it can be used as
    a drop-in replacement in backtesting engines.
    """

    def __init__(self, parquet_path: str | Path):
        """
        Load walk-forward regime timeline from parquet.

        Args:
            parquet_path: Path to walk_forward_regimes.parquet
        """
        from config.settings import CONFIDENCE_TIERS, BULLISH_REGIMES, BEARISH_REGIMES

        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Walk-forward regimes not found: {self.parquet_path}")

        self.predictions = pd.read_parquet(self.parquet_path)
        self.confidence_tiers = CONFIDENCE_TIERS
        self.bullish_regimes = BULLISH_REGIMES
        self.bearish_regimes = BEARISH_REGIMES

        # Ensure signal column exists
        if "signal" not in self.predictions.columns and "regime_label" in self.predictions.columns:
            self.predictions["signal"] = self.predictions["regime_label"].apply(
                lambda x: "bullish" if x in BULLISH_REGIMES
                else ("bearish" if x in BEARISH_REGIMES else "neutral")
                if pd.notna(x) else None
            )

        n_bars = len(self.predictions)
        date_range = f"{self.predictions.index[0]} to {self.predictions.index[-1]}"
        print(f"  WalkForwardRegimeFilter: {n_bars:,} bars, {date_range}")

    def get_regime_at(self, timestamp: pd.Timestamp | datetime) -> dict:
        """Get regime state at a specific timestamp. Same API as RegimeFilter."""
        ts = pd.Timestamp(timestamp)

        # Handle timezone alignment
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
        """Check if trading is allowed. Same API as RegimeFilter."""
        regime = self.get_regime_at(timestamp)
        confidence = regime["confidence"]
        signal = regime["signal"]

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
        """Position size multiplier based on confidence. Same API as RegimeFilter."""
        regime = self.get_regime_at(timestamp)
        confidence = regime["confidence"]

        for threshold in sorted(self.confidence_tiers.keys(), reverse=True):
            if confidence >= threshold:
                return self.confidence_tiers[threshold]

        return 0.0

    def align_to_timeframe(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill regime data to match target timeframe."""
        regime_cols = ["regime", "regime_label", "confidence", "signal"]
        available = [c for c in regime_cols if c in self.predictions.columns]
        regime_data = self.predictions[available].copy()

        if regime_data.index.tz is not None and target_df.index.tz is None:
            regime_data.index = regime_data.index.tz_localize(None)
        elif regime_data.index.tz is None and target_df.index.tz is not None:
            regime_data.index = regime_data.index.tz_localize(target_df.index.tz)

        aligned = regime_data.reindex(target_df.index, method="ffill")
        return aligned

    def add_regime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime columns to a DataFrame. Same API as RegimeFilter."""
        aligned = self.align_to_timeframe(df)
        df = df.copy()
        df["regime_label"] = aligned["regime_label"]
        df["regime_signal"] = aligned["signal"]
        df["regime_confidence"] = aligned["confidence"]
        return df
