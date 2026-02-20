"""Tests for setup signal generators — IB Breakout, VA Fade, 80% Rule."""

import math
import pytest

from backtester.config import StrategyConfig
from backtester.session import SessionState
from backtester.setups import ib_breakout, va_fade, eighty_rule


def _make_bar(close, high=None, low=None, atr=5.0, **kwargs):
    """Create a minimal bar dict for setup testing."""
    if high is None:
        high = close + 2
    if low is None:
        low = close - 2
    bar = {
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000,
        "hlc3": (high + low + close) / 3,
        "atr": atr,
        "vol_ratio": 1.0,
        "is_rth": True,
        "is_ib_period": False,
        "is_trading_window": True,
        "new_rth": False,
        "tema_bullish": True,
        "tema_bearish": False,
        "trend_up": True,
        "trend_down": False,
        "slope_rising": True,
        "slope_falling": False,
    }
    bar.update(kwargs)
    return bar


class TestIBBreakout:
    def test_long_signal_on_crossover(self):
        cfg = StrategyConfig()
        state = SessionState()
        state.ib_high = 5020
        state.ib_low = 4990
        state.ib_range = 30
        state.ib_mid = 5005
        state.ib_done = True
        state.bars_since_exit = 100

        prev_bar = _make_bar(5018)  # Below IB high
        bar = _make_bar(5022)       # Above IB high → crossover

        signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)
        assert signal is not None
        assert signal["direction"] == 1
        assert signal["setup"] == "IB"

    def test_short_signal_on_crossunder(self):
        cfg = StrategyConfig()
        state = SessionState()
        state.ib_high = 5020
        state.ib_low = 4990
        state.ib_range = 30
        state.ib_mid = 5005
        state.ib_done = True
        state.bars_since_exit = 100

        prev_bar = _make_bar(4992, tema_bearish=True, tema_bullish=False,
                            trend_down=True, trend_up=False)
        bar = _make_bar(4988, tema_bearish=True, tema_bullish=False,
                       trend_down=True, trend_up=False)

        signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)
        assert signal is not None
        assert signal["direction"] == -1

    def test_no_signal_during_ib(self):
        cfg = StrategyConfig()
        state = SessionState()
        state.ib_done = False

        signal = ib_breakout.check_signal(_make_bar(5022), _make_bar(5018), state, cfg)
        assert signal is None

    def test_no_signal_outside_trading_window(self):
        cfg = StrategyConfig()
        state = SessionState()
        state.ib_high = 5020
        state.ib_low = 4990
        state.ib_done = True
        state.bars_since_exit = 100

        bar = _make_bar(5022, is_trading_window=False)
        prev_bar = _make_bar(5018)

        signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)
        assert signal is None

    def test_max_trades_blocks(self):
        cfg = StrategyConfig(max_ib_trades=2)
        state = SessionState()
        state.ib_high = 5020
        state.ib_low = 4990
        state.ib_done = True
        state.bars_since_exit = 100
        state.ib_trades_l = 2  # Already at max

        prev_bar = _make_bar(5018)
        bar = _make_bar(5022)

        signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)
        assert signal is None

    def test_stop_capped_at_max_pts(self):
        cfg = StrategyConfig(ib_max_stop_pts=20.0, ib_stop_type="IB Mid")
        state = SessionState()
        state.ib_high = 5060
        state.ib_low = 5000  # IB Mid = 5030, but 5022-5030 = 8pts, so IB Mid wins
        state.ib_mid = 5030
        state.ib_range = 60
        state.ib_done = True
        state.bars_since_exit = 100

        prev_bar = _make_bar(5058)
        bar = _make_bar(5062)

        signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)
        assert signal is not None
        # max(5030, 5062-20) = max(5030, 5042) = 5042
        assert signal["stop"] == 5042.0

    def test_ib_range_validation(self):
        cfg = StrategyConfig(min_ib_range=8.0, max_ib_range=80.0)
        state = SessionState()
        state.ib_high = 5003
        state.ib_low = 5000  # Range = 3 pts < min 8
        state.ib_range = 3
        state.ib_done = True
        state.bars_since_exit = 100

        signal = ib_breakout.check_signal(_make_bar(5005), _make_bar(5002), state, cfg)
        assert signal is None


class TestVAFade:
    def test_long_at_val(self):
        cfg = StrategyConfig()
        state = SessionState()
        state.prev_vah = 5050.0
        state.prev_val = 4950.0
        state.prev_poc = 5000.0
        state.bars_since_exit = 100
        state.vol_ok = True

        # Touch VAL: low touches 4951 (within 4-tick=1pt buffer), close above VAL
        bar = _make_bar(4952, high=4955, low=4950.5, atr=10.0)

        signal = va_fade.check_signal(bar, None, state, cfg)
        assert signal is not None
        assert signal["direction"] == 1
        assert signal["target"] == 5000.0  # POC

    def test_no_signal_without_vol_ok(self):
        cfg = StrategyConfig()
        state = SessionState()
        state.prev_vah = 5050.0
        state.prev_val = 4950.0
        state.prev_poc = 5000.0
        state.bars_since_exit = 100
        state.vol_ok = False  # Blocked

        bar = _make_bar(4952, high=4955, low=4950.5, atr=10.0)
        signal = va_fade.check_signal(bar, None, state, cfg)
        assert signal is None

    def test_no_signal_on_narrow_ib(self):
        cfg = StrategyConfig(use_day_type=True)
        state = SessionState()
        state.prev_vah = 5050.0
        state.prev_val = 4950.0
        state.prev_poc = 5000.0
        state.bars_since_exit = 100
        state.vol_ok = True
        state.is_narrow_ib = True  # Trending day, skip fades

        bar = _make_bar(4952, high=4955, low=4950.5, atr=10.0)
        signal = va_fade.check_signal(bar, None, state, cfg)
        assert signal is None


class TestEightyRule:
    def test_long_signal_after_confirmation(self):
        cfg = StrategyConfig(use_eighty=True)
        state = SessionState()
        state.prev_vah = 5050.0
        state.prev_val = 4950.0
        state.open_below_va = True
        state.eighty_confirmed = True
        state.bars_since_exit = 100
        state.vol_ok = True

        bar = _make_bar(4980, atr=10.0)
        signal = eighty_rule.check_signal(bar, None, state, cfg)
        assert signal is not None
        assert signal["direction"] == 1
        assert signal["target"] == 5050.0  # Opposite VA edge

    def test_no_signal_when_disabled(self):
        cfg = StrategyConfig(use_eighty=False)
        state = SessionState()
        state.eighty_confirmed = True
        state.open_below_va = True

        signal = eighty_rule.check_signal(_make_bar(4980), None, state, cfg)
        assert signal is None

    def test_no_signal_without_confirmation(self):
        cfg = StrategyConfig(use_eighty=True)
        state = SessionState()
        state.prev_vah = 5050.0
        state.prev_val = 4950.0
        state.open_below_va = True
        state.eighty_confirmed = False
        state.bars_since_exit = 100
        state.vol_ok = True

        signal = eighty_rule.check_signal(_make_bar(4980), None, state, cfg)
        assert signal is None
