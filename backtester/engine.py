"""Bar-by-bar backtest engine.

Processes bars sequentially (NOT vectorized) because position state affects signals.
Matches Pine Script strategy execution model.
"""

from typing import List

import pandas as pd

from .config import StrategyConfig
from .indicators import compute_indicators
from .session import SessionState, update_session
from .position import Position, Trade
from .setups import ib_breakout, ib_rejection, level_rejection, va_fade, eighty_rule, tema_cross


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig, regime_filter=None,
                 regime_blocked=None) -> List[Trade]:
    """Run the AMT-TEMA strategy bar-by-bar.

    Args:
        df: DataFrame from data_loader with session tags
        cfg: Strategy configuration
        regime_filter: Optional RegimeFilter instance for regime gating
        regime_blocked: Set of regime labels to block (default: Strong Bull, Bull Run)

    Returns:
        List of completed Trade objects
    """
    # Compute indicators on full DataFrame first
    compute_indicators(
        df,
        tema_fast=cfg.tema_fast,
        tema_slow=cfg.tema_slow,
        tema_trend=cfg.tema_trend,
        atr_len=cfg.atr_len,
        atr_avg_len=cfg.atr_avg_len,
    )

    state = SessionState()
    pos = Position()
    trades: List[Trade] = []
    prev_bar = None
    prev_position_size = 0  # Track for cooldown

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx  # Preserve datetime index

        # ── Update session state ──
        update_session(state, bar, prev_bar, cfg)

        # ── Level state tracking (must run every bar, even in position) ──
        level_rejection.update_level_state(bar, state, cfg)
        level_rejection.update_support_level_state(bar, state, cfg)

        # ── Cooldown tracking ──
        # Match Pine: if position_size==0 and position_size[1]!=0 → just exited
        current_position_size = 0 if pos.is_flat else pos.direction
        if current_position_size == 0 and prev_position_size != 0:
            state.bars_since_exit = 0
        elif current_position_size == 0:
            state.bars_since_exit += 1

        # ── Check exits on current bar (stop/target/trail fill) ──
        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            _finalize_trade(trade, cfg)
            trades.append(trade)

        # ── TEMA Exit (v9) — close shorts when TEMA turns bullish ──
        if not pos.is_flat and cfg.use_tema_exit:
            if pos.direction == -1 and bar.get("tema_cross_up", False):
                trade = pos.close_at_market(bar, "tema_exit")
                if trade is not None:
                    _finalize_trade(trade, cfg)
                    trades.append(trade)
            elif pos.direction == 1 and bar.get("tema_cross_down", False):
                trade = pos.close_at_market(bar, "tema_exit")
                if trade is not None:
                    _finalize_trade(trade, cfg)
                    trades.append(trade)

        # ── Session flatten ──
        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                _finalize_trade(trade, cfg)
                trades.append(trade)

        # ── Check entry signals (only if flat) ──
        if pos.is_flat:
            signal = None

            # Time filters: blackout window and Friday skip
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            # Regime filter: block entries during specified regimes
            regime_ok = True
            if regime_filter is not None:
                blocked = regime_blocked or {"Strong Bull (Trend)", "Bull Run (Trend)"}
                regime_info = regime_filter.get_regime_at(idx)
                if regime_info.get("label") in blocked:
                    regime_ok = False

            time_ok = not (in_blackout or (cfg.skip_friday and is_friday))

            if regime_ok and time_ok:
                # Priority: IB Breakout > IB Rejection > Level Rejection Short > Level Rejection Long > VA Fade > 80% Rule > TEMA Cross
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = ib_rejection.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = level_rejection.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = level_rejection.check_signal_long(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = va_fade.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = eighty_rule.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = tema_cross.check_signal(bar, prev_bar, state, cfg)

            # Level Rejection gets its own shot if time filters blocked above
            if regime_ok and not time_ok and signal is None:
                if cfg.lvl_own_filters:
                    signal = level_rejection.check_signal(bar, prev_bar, state, cfg)
                if signal is None and cfg.lvl_long_own_filters:
                    signal = level_rejection.check_signal_long(bar, prev_bar, state, cfg)

            # Direction filter
            if signal is not None and cfg.direction_filter != "both":
                if cfg.direction_filter == "short" and signal["direction"] == 1:
                    signal = None
                elif cfg.direction_filter == "long" and signal["direction"] == -1:
                    signal = None

            if signal is not None:
                # Compute trail params for this entry (v9)
                trail_trigger = 0.0
                trail_dist = 0.0
                if cfg.use_trail_stop:
                    trail_trigger = bar["close"] * cfg.trail_trigger_bps / 10000.0
                    trail_dist = bar["close"] * cfg.trail_dist_bps / 10000.0

                pos.enter(
                    direction=signal["direction"],
                    price=bar["close"],
                    stop=signal["stop"],
                    target=signal["target"],
                    setup=signal["setup"],
                    time=idx,
                    slippage=cfg.slippage_pts,
                    trail_trigger_pts=trail_trigger,
                    trail_dist_pts=trail_dist,
                )

        # ── Track for next iteration ──
        prev_position_size = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    # Close any remaining position at end of data
    if not pos.is_flat and prev_bar is not None:
        trade = pos.flatten(prev_bar)
        if trade is not None:
            _finalize_trade(trade, cfg)
            trades.append(trade)

    return trades


def _finalize_trade(trade: Trade, cfg: StrategyConfig) -> None:
    """Compute dollar P&L including commission and slippage."""
    # Points P&L is already computed in Trade
    # Dollar P&L = points * point_value - round-trip commission
    trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
