"""Strategy configuration — 1:1 match with Pine Script inputs."""

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    """All parameters match amt-tema-strategy.pine inputs exactly."""

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
    tp_atr_mult: float = 0.0             # v9: cap TP at ATR * mult (0=disabled, uses v8 logic)
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

    # --- Setup 5: IB Rejection ---
    use_ib_reject: bool = False
    rej_trigger: str = "wick"           # "bearish_close", "wick", "failed_break", "any"
    rej_zone_pts: float = 5.0           # Points from IB high to define rejection zone
    rej_stop_buffer: float = 3.0        # Points above IB high for stop
    rej_target: str = "vwap"            # "vwap", "ib_mid", "ib_low", "prev_poc", "fixed"
    rej_target_pts: float = 25.0        # Fixed points target (when rej_target="fixed")
    rej_require_tema: bool = True       # Require TEMA bearish state (not crossover)
    max_rej_trades: int = 5             # Max rejection trades per day
    rej_wide_only: bool = False         # Only take rejections on wide IB days (ratio >= 1.2)

    # --- Setup 6: Level Rejection ---
    use_level_reject: bool = False
    lvl_trigger: str = "any"          # "any" | "wick" | "bearish_close" | "failed_break"
    lvl_zone_pts: float = 5.0         # proximity zone around each resistance level
    lvl_stop_buffer: float = 8.0      # stop = level + buffer, capped by pct stop
    lvl_require_tema: bool = False    # require bearish MA filter for entry
    lvl_ma_filter: str = "tema"      # "tema" | "ema_9_21" | "ema_8_21" | "sma_8_21"
    max_lvl_trades: int = 4           # shared across ALL levels (prevents chop)
    lvl_ibh_wide_only: bool = True    # IBH rejection only on wide IB days (proven edge)
    lvl_max_tests: int = 3            # skip level after N tests (defenders exhausted)
    lvl_broken_bars: int = 2          # consecutive closes above level to mark broken (1=old behavior)
    lvl_own_filters: bool = False     # LVL uses own Friday/blackout settings (not inherited from v8)

    # Confluence scoring: levels within this distance count as confluent
    lvl_confluence_zone: float = 5.0  # points — levels within this range cluster
    lvl_min_confluence: int = 1       # minimum confluence score to take trade (1=any single level)

    # Risk-reward filter: skip trade if target too close relative to stop
    lvl_min_rr: float = 0.0           # min reward:risk ratio (0=disabled, 1.5 recommended)
    lvl_min_target_pts: float = 0.0   # min distance to target in pts (0=disabled, 5.0 recommended)
    lvl_target_skip: int = 0          # 0=nearest support, 1=second, 2=third (for staggered exits)

    # Level selection: which resistance levels to trade (empty = all)
    lvl_enabled_levels: tuple = ()     # e.g., ("ONH", "PDH") — empty tuple = all levels

    # Bar quality metrics at rejection
    lvl_use_bar_metrics: bool = False  # require bar quality filter for entry
    lvl_min_wick_ratio: float = 0.3    # min upper wick as % of bar range (0.3 = 30%)

    # --- Setup 6B: Level Rejection LONG (support bounce) ---
    use_level_reject_long: bool = False
    lvl_long_trigger: str = "any"       # "any" | "bullish_close" | "wick" | "failed_break"
    lvl_long_zone_pts: float = 5.0
    lvl_long_stop_buffer: float = 8.0
    lvl_long_require_tema: bool = False  # require TEMA bullish for long entry
    lvl_long_ma_filter: str = "tema"
    max_lvl_long_trades: int = 4
    lvl_long_max_tests: int = 3
    lvl_long_broken_bars: int = 2
    lvl_long_own_filters: bool = False
    lvl_long_min_rr: float = 0.0
    lvl_long_min_target_pts: float = 0.0
    lvl_long_enabled_levels: tuple = ()  # e.g., ("ONL", "PDL") — empty = all support levels

    # --- Setup 4: TEMA Cross Short (v9) ---
    use_tema_cross: bool = False          # Default OFF (enable with --v9)
    tx_day_type_filter: str = "narrow"    # "narrow", "narrow+normal", "all"
    max_tx_trades: int = 2
    tx_stop_bps: float = 30.0            # Stop in basis points
    tx_tp_atr_mult: float = 2.0          # TP as ATR multiple

    # --- Trailing Stop (v9) ---
    use_trail_stop: bool = False          # Default OFF (enable with --v9)
    trail_trigger_bps: float = 15.0       # Profit in bps before trail activates
    trail_dist_bps: float = 20.0          # Trail distance from best price in bps

    # --- TEMA Exit (v9) ---
    use_tema_exit: bool = False           # Default OFF (enable with --v9)

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
