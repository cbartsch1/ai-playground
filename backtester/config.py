"""Strategy configuration — 1:1 match with Pine Script v6.1 inputs."""

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    """All parameters match amt-tema-strategy.pine v6.1 inputs exactly."""

    # --- Instrument ---
    instrument: str = "ES"
    tick_size: float = 0.25        # ES tick = 0.25 pts
    point_value: float = 50.0      # $50 per point for ES
    commission: float = 2.50       # per contract per side
    slippage_ticks: int = 1        # 1 tick slippage per fill
    initial_capital: float = 100_000.0

    # --- TEMA ---
    tema_fast: int = 9
    tema_slow: int = 21
    tema_trend: int = 55

    # --- Session (ET times as HHMM integers) ---
    rth_start: int = 930
    rth_end: int = 1600
    ib_end_time: int = 1030
    trade_start: int = 1035
    trade_end: int = 1500
    flatten_time: int = 1555

    # --- Day Type ---
    ib_avg_len: int = 20
    ib_narrow_ratio: float = 0.8
    ib_wide_ratio: float = 1.2
    use_day_type: bool = True

    # --- Value Area ---
    va_stdev_mult: float = 1.0

    # --- ATR & Risk ---
    atr_len: int = 14
    cooldown_bars: int = 2

    # --- Volatility ---
    use_vol_filter: bool = True
    atr_avg_len: int = 50
    vol_low_ratio: float = 0.5
    vol_high_ratio: float = 2.0

    # --- Setup 1: IB Breakout ---
    use_ib_break: bool = True
    use_trend_filter: bool = True
    min_ib_range: float = 8.0
    max_ib_range: float = 80.0
    ib_stop_type: str = "IB Mid"       # "IB Mid", "IB Edge", "ATR"
    ib_max_stop_pts: float = 20.0
    ib_min_target: float = 10.0
    max_ib_trades: int = 2

    # --- Setup 2: VA Fade ---
    use_va_fade: bool = True
    va_buffer: float = 4.0             # ticks
    va_stop_mult: float = 0.5          # ATR multiplier
    va_min_rr: float = 0.5
    max_va_trades: int = 1

    # --- Setup 3: 80% Rule ---
    use_eighty: bool = False            # Default OFF
    eighty_conf_bars: int = 6
    eighty_stop_buf: float = 0.5        # ATR multiplier
    max_eighty_trades: int = 1

    # --- Time Filters ---
    blackout_start: int = 0             # Skip entries during this window (HHMM), 0=disabled
    blackout_end: int = 0               # End of blackout window (HHMM), 0=disabled
    skip_friday: bool = False           # Skip all entries on Fridays

    # --- Direction Filter ---
    direction_filter: str = "both"      # "both", "long", "short"

    # --- Percentage-Based Stops ---
    pct_stop_mode: bool = False         # If True, ib_max_stop_pts scales as % of price
    pct_stop_bps: float = 30.0          # Basis points for max stop (30 bps ≈ 20pt at ES 6700)

    # --- Engine ---
    pessimistic_fills: bool = True      # When both stop and target could hit, stop wins

    @property
    def slippage_pts(self) -> float:
        return self.slippage_ticks * self.tick_size

    @property
    def va_buffer_pts(self) -> float:
        return self.va_buffer * self.tick_size


# Presets
ES_DEFAULTS = StrategyConfig()

SPX_DEFAULTS = StrategyConfig(
    instrument="SPX",
    tick_size=0.01,
    point_value=100.0,       # SPX options — notional, not used for P&L here
    commission=0.0,           # Signal-only, user trades options manually
    slippage_ticks=0,
    va_buffer=100.0,          # 4 ES ticks = 1pt = 100 SPX ticks
)
