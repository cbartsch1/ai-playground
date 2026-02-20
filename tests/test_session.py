"""Tests for session logic — IB tracking, day type, VA levels."""

import math
import pandas as pd
import pytest

from backtester.config import StrategyConfig
from backtester.session import SessionState, update_session


def _make_bar(et_time, high, low, close, volume=1000, open_=None, is_new_rth=False):
    """Create a minimal bar dict for testing."""
    if open_ is None:
        open_ = close
    et_hour = et_time // 100
    et_minute = et_time % 100

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "hlc3": (high + low + close) / 3,
        "et_time": et_time,
        "et_hour": et_hour,
        "et_minute": et_minute,
        "is_rth": 930 <= et_time < 1600,
        "is_ib_period": 930 <= et_time < 1030,
        "is_trading_window": 1035 <= et_time < 1500,
        "new_rth": is_new_rth,
        "vol_ratio": 1.0,
    }


class TestIBTracking:
    def test_ib_builds_during_ib_period(self):
        cfg = StrategyConfig()
        state = SessionState()

        # First bar: new RTH
        bar1 = _make_bar(930, 5010, 4990, 5000, is_new_rth=True)
        update_session(state, bar1, None, cfg)
        assert state.ib_high == 5010
        assert state.ib_low == 4990
        assert not state.ib_done

        # Second IB bar: extends range
        bar2 = _make_bar(935, 5020, 4985, 5005)
        update_session(state, bar2, bar1, cfg)
        assert state.ib_high == 5020
        assert state.ib_low == 4985
        assert not state.ib_done

    def test_ib_done_after_ib_period(self):
        cfg = StrategyConfig()
        state = SessionState()

        bar1 = _make_bar(930, 5010, 4990, 5000, is_new_rth=True)
        update_session(state, bar1, None, cfg)

        # Last IB bar
        bar2 = _make_bar(1025, 5015, 4995, 5005)
        update_session(state, bar2, bar1, cfg)
        assert not state.ib_done

        # First post-IB bar
        bar3 = _make_bar(1030, 5012, 4998, 5008)
        # This is RTH but not IB period, so should NOT set ib_done
        # Because is_ib_period check is et_time >= 930 and et_time < 1030
        # 1030 is NOT in IB period
        update_session(state, bar3, bar2, cfg)
        assert state.ib_done

    def test_ib_resets_on_new_rth(self):
        cfg = StrategyConfig()
        state = SessionState()
        state.ib_high = 5050
        state.ib_low = 4950
        state.ib_done = True

        bar = _make_bar(930, 5010, 4990, 5000, is_new_rth=True)
        update_session(state, bar, None, cfg)
        assert state.ib_high == 5010
        assert state.ib_low == 4990
        assert not state.ib_done


class TestDayType:
    def test_narrow_ib(self):
        cfg = StrategyConfig(ib_narrow_ratio=0.8, ib_wide_ratio=1.2)
        state = SessionState()
        state.ib_range_avg = 20.0  # Average IB is 20 pts
        state.ib_high = 5010
        state.ib_low = 5000  # IB range = 10 pts = 0.5x avg → narrow

        bar = _make_bar(1035, 5012, 4998, 5005)
        update_session(state, bar, None, cfg)

        assert state.is_narrow_ib
        assert not state.is_wide_ib

    def test_wide_ib(self):
        cfg = StrategyConfig(ib_narrow_ratio=0.8, ib_wide_ratio=1.2)
        state = SessionState()
        state.ib_range_avg = 20.0
        state.ib_high = 5030
        state.ib_low = 5000  # IB range = 30 pts = 1.5x avg → wide

        bar = _make_bar(1035, 5025, 5005, 5015)
        update_session(state, bar, None, cfg)

        assert state.is_wide_ib
        assert not state.is_narrow_ib


class TestVALevels:
    def test_va_stored_on_new_rth(self):
        cfg = StrategyConfig()
        state = SessionState()

        # Simulate accumulated VWAP data
        state.rth_vwap_sum = 5000 * 1000 * 78  # ~78 RTH bars
        state.rth_vol_sum = 1000 * 78
        state.rth_sq_dev = 100.0  # Some deviation
        state.rth_bars = 78

        # New RTH should store previous session
        bar = _make_bar(930, 5010, 4990, 5000, is_new_rth=True)
        update_session(state, bar, None, cfg)

        assert not math.isnan(state.prev_poc)
        assert abs(state.prev_poc - 5000.0) < 0.01
        assert state.prev_vah > state.prev_poc
        assert state.prev_val < state.prev_poc

    def test_va_position_tracking(self):
        cfg = StrategyConfig()
        state = SessionState()
        state.prev_vah = 5020.0
        state.prev_val = 4980.0
        state.prev_poc = 5000.0

        bar_above = _make_bar(1035, 5030, 5022, 5025)
        update_session(state, bar_above, None, cfg)
        assert state.above_va
        assert not state.below_va

        bar_below = _make_bar(1040, 4978, 4970, 4975)
        update_session(state, bar_below, bar_above, cfg)
        assert state.below_va
        assert not state.above_va

        bar_inside = _make_bar(1045, 5010, 4990, 5000)
        update_session(state, bar_inside, bar_below, cfg)
        assert state.inside_va


class TestTradeCounters:
    def test_counters_reset_on_new_rth(self):
        cfg = StrategyConfig()
        state = SessionState()
        state.ib_trades_l = 2
        state.ib_trades_s = 1
        state.va_trades_l = 1

        bar = _make_bar(930, 5010, 4990, 5000, is_new_rth=True)
        update_session(state, bar, None, cfg)

        assert state.ib_trades_l == 0
        assert state.ib_trades_s == 0
        assert state.va_trades_l == 0
