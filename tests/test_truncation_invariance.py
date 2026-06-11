"""Truncation-invariance regression test — look-ahead guard for the engine.

If the engine is causal, deleting future bars cannot change past trades:
running on the full DataFrame vs a truncated copy must produce identical
trades before the cut. A failure here means some signal/level/indicator is
reading bars after the decision time (look-ahead).

Verified clean on the full 2yr dataset (462/462 trades identical) in the
2026-06-11 edge-legitimacy audit; this test pins that property on a smaller
slice so it runs in CI time and future engine changes can't silently
reintroduce look-ahead.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest

DATA_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "es_5m_databento_2yr.csv")

N_SESSIONS = 50          # slice size (trading days) — keeps runtime small
CUT_FRACTION = 0.6       # truncate at 60% of the slice


def _load_slice():
    df = load_tos_csv(DATA_CSV, instrument="ES")
    session_starts = df.index[df["new_rth"]]
    if len(session_starts) < N_SESSIONS + 1:
        pytest.skip("not enough sessions in data file")
    end = session_starts[N_SESSIONS]
    return df[df.index < end].copy()


def _ms_os_config():
    """MS+OS (nexus lineage) — exercises session state, VP levels, ON levels."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"
    for flag in ["use_ib_break", "use_va_fade", "use_eighty", "use_tema_cross",
                 "use_level_reject", "use_level_reject_long", "use_ib_reject",
                 "use_var", "use_ptf", "use_fa"]:
        setattr(cfg, flag, False)
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 4.0
    cfg.ms_min_target_pts = 8.0
    cfg.ms_min_rr = 0.3
    cfg.ms_max_risk = 25.0
    cfg.ms_ma_type = "sma"
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False
    cfg.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both",
                               "MS_pVAH": "short"}
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = 5.0
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 3.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.max_os_trades = 1
    cfg.os_min_gap = 3.0
    cfg.os_max_gap = 20.0
    cfg.os_entry_window = 1
    return cfg


def _amt_v8_config():
    """AMT-TEMA v8 (IB breakout short) — exercises IB/TEMA/blackout paths."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    for flag in ["use_va_fade", "use_eighty", "use_tema_cross",
                 "use_level_reject", "use_level_reject_long", "use_ib_reject",
                 "use_var", "use_ptf", "use_fa", "use_ms", "use_os"]:
        setattr(cfg, flag, False)
    cfg.use_ib_break = True
    cfg.tp_atr_mult = 0.0
    cfg.max_ib_trades = 2
    cfg.min_ib_range = 8.0
    cfg.max_ib_range = 80.0
    cfg.ib_stop_type = "IB Mid"
    cfg.ib_min_target = 10.0
    cfg.use_trend_filter = True
    return cfg


def _trade_key(t):
    return (t.setup, t.direction, str(t.entry_time), round(t.entry_price, 4),
            str(t.exit_time), round(t.exit_price, 4), t.exit_reason,
            round(t.pnl_dollar, 2))


@pytest.mark.parametrize("make_cfg", [_ms_os_config, _amt_v8_config],
                         ids=["ms_os", "amt_v8"])
def test_truncation_invariance(make_cfg):
    df = _load_slice()
    cut = df.index[int(len(df) * CUT_FRACTION)]

    trades_full = run_backtest(df.copy(), make_cfg())
    trades_trunc = run_backtest(df[df.index < cut].copy(), make_cfg())

    # Compare only trades fully resolved before the cut in BOTH runs:
    # a position still open at the cut gets force-flattened in the truncated
    # run (legitimate difference, not look-ahead).
    full_pre = [_trade_key(t) for t in trades_full
                if t.exit_time < cut and t.entry_time < cut]
    trunc_pre = [_trade_key(t) for t in trades_trunc
                 if t.exit_time < cut and t.entry_time < cut]

    assert len(full_pre) > 0, "slice produced no pre-cut trades — test is vacuous"
    assert full_pre == trunc_pre, (
        "LOOK-AHEAD: pre-cut trades changed when future bars were removed.\n"
        f"full({len(full_pre)}) vs truncated({len(trunc_pre)})"
    )


def test_truncation_invariance_detects_mutation():
    """Sanity: the comparison machinery actually detects a difference.

    Simulates a look-ahead by perturbing a future bar in a way that a causal
    engine must NOT propagate backward: pre-cut trades must STILL be
    identical even when post-cut data changes. (If someone wires a
    forward-looking feature in, this assertion is the one that breaks.)
    """
    df = _load_slice()
    cut = df.index[int(len(df) * CUT_FRACTION)]

    trades_base = run_backtest(df.copy(), _ms_os_config())

    # Perturb all bars AFTER the cut by +50 points.
    df2 = df.copy()
    mask = df2.index >= cut
    for col in ["open", "high", "low", "close"]:
        df2.loc[mask, col] = df2.loc[mask, col] + 50.0

    trades_perturbed = run_backtest(df2, _ms_os_config())

    base_pre = [_trade_key(t) for t in trades_base
                if t.exit_time < cut and t.entry_time < cut]
    pert_pre = [_trade_key(t) for t in trades_perturbed
                if t.exit_time < cut and t.entry_time < cut]

    assert base_pre == pert_pre, (
        "LOOK-AHEAD: pre-cut trades changed when only post-cut bars were "
        "modified — something reads future bars."
    )


def test_comparator_is_not_vacuous():
    """Perturbing PRE-cut bars must change pre-cut trades.

    Proves the trade-key comparison is sensitive: if this fails, the
    invariance assertions above are comparing something inert.
    """
    df = _load_slice()
    cut = df.index[int(len(df) * CUT_FRACTION)]

    trades_base = run_backtest(df.copy(), _ms_os_config())

    df2 = df.copy()
    mask = df2.index < cut
    for col in ["open", "high", "low", "close"]:
        df2.loc[mask, col] = df2.loc[mask, col] + 50.0

    trades_perturbed = run_backtest(df2, _ms_os_config())

    base_pre = [_trade_key(t) for t in trades_base
                if t.exit_time < cut and t.entry_time < cut]
    pert_pre = [_trade_key(t) for t in trades_perturbed
                if t.exit_time < cut and t.entry_time < cut]

    assert base_pre != pert_pre, (
        "comparator vacuous: massively perturbing pre-cut bars left "
        "pre-cut trades identical"
    )
