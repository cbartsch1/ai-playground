"""Stagger-aware backtest engine for multi-contract Level Rejection.

Runs a SINGLE backtest where:
  - IB Breakout and IB Rejection trades use 1 contract (baseline)
  - Level Rejection trades enter up to N contracts with staggered targets
  - All share one SessionState and one trade-per-day counter
  - No new entries while any position is active

This replaces the old approach of running N independent backtests (which
triple-counted baseline trades and allowed up to N * max_lvl_trades entries
per day instead of the correct max_lvl_trades).
"""

from typing import List

import pandas as pd

from .config import StrategyConfig
from .indicators import compute_indicators
from .session import SessionState, update_session
from .position import Position, Trade
from .setups import ib_breakout, ib_rejection, level_rejection


def run_backtest_stagger(df: pd.DataFrame, cfg: StrategyConfig,
                         n_contracts: int = 3,
                         uniform_skip: int = -1) -> List[Trade]:
    """Run the AMT-TEMA strategy with multi-contract stagger for LVL trades.

    Args:
        df: DataFrame from data_loader with session tags
        cfg: Strategy configuration (must have use_level_reject=True)
        n_contracts: Number of stagger contracts for LVL entries (default 3)
        uniform_skip: If >= 0, all contracts target the same skip level (e.g., 2 = all
                      target 3rd support). Default -1 = normal stagger (0,1,2).

    Returns:
        List of completed Trade objects. LVL trades have contract=1,2,3.
        Baseline trades (IB/REJ) have contract=1 (default).
    """
    compute_indicators(
        df,
        tema_fast=cfg.tema_fast,
        tema_slow=cfg.tema_slow,
        tema_trend=cfg.tema_trend,
        atr_len=cfg.atr_len,
        atr_avg_len=cfg.atr_avg_len,
    )

    state = SessionState()

    # Position slots: 1 for baseline (IB/REJ), N for LVL stagger
    baseline_pos = Position()
    lvl_positions = [Position() for _ in range(n_contracts)]

    trades: List[Trade] = []
    prev_bar = None
    was_any_active = False

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx

        # ── Update session state ──
        update_session(state, bar, prev_bar, cfg)

        # ── Level state tracking (must run every bar, even in position) ──
        level_rejection.update_level_state(bar, state, cfg)

        # ── Cooldown tracking ──
        # Mirror original engine: check position state BEFORE exits
        any_active_now = (not baseline_pos.is_flat or
                          any(not p.is_flat for p in lvl_positions))
        if not any_active_now and was_any_active:
            state.bars_since_exit = 0
        elif not any_active_now:
            state.bars_since_exit += 1

        # ── Check exits — baseline ──
        trade = baseline_pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            _finalize_trade(trade, cfg)
            trades.append(trade)

        # ── Check exits — LVL contracts ──
        for pos in lvl_positions:
            trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
            if trade is not None:
                _finalize_trade(trade, cfg)
                trades.append(trade)

        # ── Session flatten ──
        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time:
            if not baseline_pos.is_flat:
                trade = baseline_pos.flatten(bar)
                if trade is not None:
                    _finalize_trade(trade, cfg)
                    trades.append(trade)
            for pos in lvl_positions:
                if not pos.is_flat:
                    trade = pos.flatten(bar)
                    if trade is not None:
                        _finalize_trade(trade, cfg)
                        trades.append(trade)

        # ── Check entry signals (only if ALL positions are flat) ──
        all_flat = (baseline_pos.is_flat and
                    all(p.is_flat for p in lvl_positions))

        if all_flat:
            signal = None

            # Time filters: blackout window and Friday skip
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4
            time_ok = not (in_blackout or (cfg.skip_friday and is_friday))

            if time_ok:
                # Priority: IB Breakout > IB Rejection
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)
                if signal is None:
                    signal = ib_rejection.check_signal(bar, prev_bar, state, cfg)

            # Enter baseline position (1 contract)
            if signal is not None:
                # Direction filter
                if cfg.direction_filter != "both":
                    if cfg.direction_filter == "short" and signal["direction"] == 1:
                        signal = None
                    elif cfg.direction_filter == "long" and signal["direction"] == -1:
                        signal = None

                if signal is not None:
                    baseline_pos.enter(
                        direction=signal["direction"],
                        price=bar["close"],
                        stop=signal["stop"],
                        target=signal["target"],
                        setup=signal["setup"],
                        time=idx,
                        slippage=cfg.slippage_pts,
                    )
            else:
                # Check LVL stagger (lower priority than IB/REJ)
                # LVL may bypass blackout/Friday if lvl_own_filters is True
                lvl_ok = time_ok or cfg.lvl_own_filters

                if lvl_ok:
                    lvl_signals = level_rejection.check_signal_multi(
                        bar, prev_bar, state, cfg, n_contracts=n_contracts,
                        uniform_skip=uniform_skip
                    )

                    for sig in lvl_signals:
                        c_idx = sig["contract"] - 1  # 0-indexed
                        if c_idx < n_contracts:
                            lvl_positions[c_idx].enter(
                                direction=sig["direction"],
                                price=bar["close"],
                                stop=sig["stop"],
                                target=sig["target"],
                                setup=sig["setup"],
                                time=idx,
                                slippage=cfg.slippage_pts,
                                contract=sig["contract"],
                            )

        # ── Track end-of-bar state for next iteration ──
        was_any_active = (not baseline_pos.is_flat or
                          any(not p.is_flat for p in lvl_positions))
        prev_bar = bar

    # Close any remaining positions at end of data
    if prev_bar is not None:
        if not baseline_pos.is_flat:
            trade = baseline_pos.flatten(prev_bar)
            if trade is not None:
                _finalize_trade(trade, cfg)
                trades.append(trade)
        for pos in lvl_positions:
            if not pos.is_flat:
                trade = pos.flatten(prev_bar)
                if trade is not None:
                    _finalize_trade(trade, cfg)
                    trades.append(trade)

    return trades


def _finalize_trade(trade: Trade, cfg: StrategyConfig) -> None:
    """Compute dollar P&L including commission and slippage."""
    trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
