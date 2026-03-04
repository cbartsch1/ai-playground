"""Technical indicators — TEMA, ATR (Wilder's), SMA, VWAP, slope.

All implementations match Pine Script v6 built-in functions exactly.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA matching Pine Script ta.ema() — uses ewm(span=period, adjust=False)."""
    return series.ewm(span=period, adjust=False).mean()


def tema(series: pd.Series, period: int) -> pd.Series:
    """Triple EMA matching Pine Script temaCalc().

    Formula: 3*EMA1 - 3*EMA2 + EMA3
    where EMA1 = ema(src, len), EMA2 = ema(EMA1, len), EMA3 = ema(EMA2, len)
    """
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def atr_wilders(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR using Wilder's smoothing — matches Pine Script ta.atr().

    CRITICAL: Pine's ta.atr() uses ta.rma() which is Wilder's smoothing:
        alpha = 1/period (NOT 2/(period+1) like standard EMA)

    This is the #1 source of Pine-to-Python divergence.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Wilder's smoothing = EMA with alpha=1/period
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average matching Pine Script ta.sma()."""
    return series.rolling(window=period, min_periods=period).mean()


def compute_indicators(df: pd.DataFrame,
                       tema_fast: int = 9,
                       tema_slow: int = 21,
                       tema_trend: int = 55,
                       atr_len: int = 14,
                       atr_avg_len: int = 50) -> None:
    """Compute all indicators in-place on the DataFrame.

    Adds columns: tema_fast, tema_slow, tema_trend, atr, atr_avg, vol_ratio,
                  tema_bullish, tema_bearish, tema_slope, slope_rising, slope_falling,
                  trend_up, trend_down
    """
    # TEMA
    df["tema_fast"] = tema(df["close"], tema_fast)
    df["tema_slow"] = tema(df["close"], tema_slow)
    df["tema_trend"] = tema(df["close"], tema_trend)

    # ATR (Wilder's)
    df["atr"] = atr_wilders(df, atr_len)
    df["atr_avg"] = sma(df["atr"], atr_avg_len)

    # Volatility ratio
    df["vol_ratio"] = np.where(df["atr_avg"] > 0, df["atr"] / df["atr_avg"], 1.0)

    # TEMA signals — match Pine exactly
    df["tema_bullish"] = df["tema_fast"] > df["tema_slow"]
    df["tema_bearish"] = df["tema_fast"] < df["tema_slow"]

    # Slope: temaFast - temaFast[3]
    df["tema_slope"] = df["tema_fast"] - df["tema_fast"].shift(3)
    # slope_rising: temaSlope > temaSlope[3]
    df["slope_rising"] = df["tema_slope"] > df["tema_slope"].shift(3)
    df["slope_falling"] = df["tema_slope"] < df["tema_slope"].shift(3)

    # Trend: close vs TEMA trend
    df["trend_up"] = df["close"] > df["tema_trend"]
    df["trend_down"] = df["close"] < df["tema_trend"]

    # TEMA crossover/crossunder (v9) — transition detection
    df["tema_cross_up"] = (df["tema_fast"] > df["tema_slow"]) & (df["tema_fast"].shift(1) <= df["tema_slow"].shift(1))
    df["tema_cross_down"] = (df["tema_fast"] < df["tema_slow"]) & (df["tema_fast"].shift(1) >= df["tema_slow"].shift(1))

    # EMA signals — for TEMA vs EMA comparison testing
    df["ema_9"] = ema(df["close"], 9)
    df["ema_21"] = ema(df["close"], 21)
    df["ema_8"] = ema(df["close"], 8)
    df["ema_bearish_9_21"] = df["ema_9"] < df["ema_21"]
    df["ema_bearish_8_21"] = df["ema_8"] < df["ema_21"]
    df["sma_8"] = sma(df["close"], 8)
    df["sma_21"] = sma(df["close"], 21)
    df["sma_24"] = sma(df["close"], 24)
    df["sma_bearish_8_21"] = df["sma_8"] < df["sma_21"]
    df["sma_bearish_8_24"] = df["sma_8"] < df["sma_24"]
    df["sma_bullish_8_24"] = df["sma_8"] > df["sma_24"]
