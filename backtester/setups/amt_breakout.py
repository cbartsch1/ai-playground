"""Trapdoor — AMT-TEMA v8 IB Breakout Short-Only.

THIS IS THE FILE MODIFIED DURING AUTORESEARCH EXPERIMENTS.
Entry/exit logic, filters, and parameters all live here.

v8 baseline:
- Short-only (IB low breakout)
- 30bps percentage-based stop (scales with ES price)
- Friday skip, noon blackout 12:00-13:00
- VA Fade OFF
- One-shot crossunder (not continuous re-arm)
- Max 2 IB trades per day
- TEMA fast/slow bearish required (tema_fast < tema_slow)
- TEMA trend filter: close < TEMA 55

2-year results (baseline):
  177 trades | 45.2% WR | PF 1.511 | +$30,084 | DD $5,055 | Sharpe 2.65
  t-test p=0.028 | permutation p=0.013 | bootstrap P(profit)=98.75%
"""

import math
from typing import Optional

from backtester.config import StrategyConfig


def get_config() -> StrategyConfig:
    """Return v8 baseline configuration for Trapdoor strategy."""
    cfg = StrategyConfig()

    # v8 core settings
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False

    # IB breakout settings
    cfg.use_ib_break = True
    cfg.tp_atr_mult = 0.0   # v8 mode: use IB range as TP, not ATR cap
    cfg.max_ib_trades = 2
    cfg.min_ib_range = 8.0
    cfg.max_ib_range = 80.0
    cfg.ib_stop_type = "IB Mid"
    cfg.ib_min_target = 10.0
    cfg.use_trend_filter = True

    # All other setups off
    cfg.use_ib_reject = False
    cfg.use_level_reject = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_ms = False
    cfg.use_os = False
    cfg.use_fa = False
    cfg.use_var = False
    cfg.use_ptf = False

    return cfg


def check_signal(bar: dict, prev_bar, session, cfg) -> Optional[dict]:
    """IB Breakout Short signal — v8 one-shot crossunder.

    Entry:  close crosses BELOW IB low (one-shot, prev close >= IB low)
    TEMA:   tema_fast < tema_slow (bearish state)
    Trend:  close < TEMA 55 (trend_down)
    Stop:   IB mid, capped at 30bps from entry
    Target: IB range from entry, min 10pts
    """
    if not cfg.use_ib_break:
        return None

    if not session.ib_done:
        return None

    if not bar["is_trading_window"]:
        return None

    # Afternoon filter: no new entries at 14:00+ ET (afternoon chop)
    et_time = bar.get("et_time", 0)
    if et_time >= 1400:
        return None

    if session.bars_since_exit < cfg.cooldown_bars:
        return None

    # IB range validation
    ib_range = session.ib_range
    if ib_range < cfg.min_ib_range or ib_range > cfg.max_ib_range:
        return None

    if prev_bar is None:
        return None

    ib_low = session.ib_low
    ib_high = session.ib_high

    # v8 mode: one-shot crossunder
    cross_down = bar["close"] < ib_low and prev_bar["close"] >= ib_low
    if not cross_down:
        return None

    # TEMA bearish state (fast < slow)
    if not bar["tema_bearish"]:
        return None

    # TEMA trend filter: close < TEMA 55
    if cfg.use_trend_filter and not bar["trend_down"]:
        return None

    # Max IB trades per day
    if session.ib_trades_s >= cfg.max_ib_trades:
        return None

    # AMT context: skip when day opened BELOW previous value area
    # Below-VA opens = discount territory = potential support = higher rejection risk
    if session.open_below_va:
        return None

    # Stop: IB mid capped by pct stop
    max_stop = (bar["close"] * cfg.pct_stop_bps / 10000.0
                if cfg.pct_stop_mode else cfg.ib_max_stop_pts)

    if cfg.ib_stop_type == "IB Mid":
        raw_sl = session.ib_mid
    elif cfg.ib_stop_type == "IB Edge":
        raw_sl = ib_high
    else:  # ATR
        raw_sl = bar["close"] + bar["atr"] * 1.5

    stop = min(raw_sl, bar["close"] + max_stop)

    # Target: IB range from entry, minimum ib_min_target
    tp_pts = max(ib_range, cfg.ib_min_target)
    target = bar["close"] - tp_pts

    session.ib_trades_s += 1
    return {"direction": -1, "stop": stop, "target": target, "setup": "IB"}
