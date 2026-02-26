"""Tests for setup signal generators — IB Breakout, VA Fade, 80% Rule, Level Rejection."""

import math
import pytest

from backtester.config import StrategyConfig
from backtester.session import SessionState
from backtester.setups import ib_breakout, va_fade, eighty_rule, level_rejection


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


def _make_lvl_state(**overrides):
    """Create a SessionState pre-configured for level rejection testing."""
    state = SessionState()
    state.ib_high = 5020
    state.ib_low = 4980
    state.ib_range = 40
    state.ib_mid = 5000
    state.ib_done = True
    state.bars_since_exit = 100
    state.prev_day_high = 5050
    state.prev_day_low = 4920
    state.on_high = 5035
    state.on_low = 4940
    state.prev_vah = 5030
    state.prev_val = 4960
    state.prev_poc = 4990
    state.is_wide_ib = True
    # VWAP: simulate accumulated data → VWAP ~4998
    state.rth_vwap_sum = 4998 * 1000 * 20
    state.rth_vol_sum = 1000 * 20
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


class TestLevelRejection:
    def test_short_at_prev_day_high(self):
        """Entry near PDH, stop above, target at next support below."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="any",
                             lvl_zone_pts=5.0, lvl_stop_buffer=8.0)
        state = _make_lvl_state()
        # Price near PDH (5050), high reaches into 5050-5 = 5045 zone
        bar = _make_bar(5048, high=5051, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is not None
        assert signal["direction"] == -1
        assert signal["setup"] == "LVL_PDH"
        assert signal["stop"] == 5050 + 8.0  # PDH + buffer
        assert signal["target"] < 5048  # Some support below entry

    def test_short_at_overnight_high(self):
        """Entry near ONH (below PDH)."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="any",
                             lvl_zone_pts=5.0, lvl_stop_buffer=8.0)
        # PDH far above so doesn't trigger, ONH = 5035
        state = _make_lvl_state(prev_day_high=5100)
        bar = _make_bar(5033, high=5036, low=5030)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is not None
        assert signal["setup"] == "LVL_ONH"

    def test_no_signal_when_disabled(self):
        """use_level_reject=False → None."""
        cfg = StrategyConfig(use_level_reject=False)
        state = _make_lvl_state()
        bar = _make_bar(5048, high=5051, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is None

    def test_max_trades_shared(self):
        """Shared counter blocks after max_lvl_trades across ALL levels."""
        cfg = StrategyConfig(use_level_reject=True, max_lvl_trades=2)
        state = _make_lvl_state(lvl_trades_s=2)
        bar = _make_bar(5048, high=5051, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is None

    def test_level_to_level_targeting(self):
        """Entry at PDH, target = next support below entry."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="any",
                             lvl_zone_pts=5.0, lvl_stop_buffer=8.0)
        state = _make_lvl_state()
        bar = _make_bar(5048, high=5051, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is not None
        # Target should be the highest support below 5048
        # Support levels: ONH=5035, VAH=5030, IBH=5020, VWAP~4998, IB_MID=5000, PREV_POC=4990
        # Highest support below 5048: ONH (5035) — wait, ONH is resistance not support
        # Support: VWAP~4998, PREV_POC=4990, IB_MID=5000, IB_LOW=4980, VAL=4960, ONL=4940, PDL=4920
        # Sorted desc: IB_MID=5000, VWAP~4998, PREV_POC=4990, IB_LOW=4980, VAL=4960, ONL=4940, PDL=4920
        # First below 5048 → IB_MID=5000
        assert signal["target"] == 5000  # IB Mid

    def test_first_match_wins(self):
        """PDH wins over ONH when both in zone (highest level priority)."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="any",
                             lvl_zone_pts=10.0, lvl_stop_buffer=8.0)
        # PDH=5050, ONH=5048 — both within 10pt zone of bar high 5051
        state = _make_lvl_state(on_high=5048)
        bar = _make_bar(5048, high=5051, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is not None
        assert signal["setup"] == "LVL_PDH"  # Highest wins

    def test_ibh_wide_day_filter(self):
        """IBH only fires on wide days when lvl_ibh_wide_only=True."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="any",
                             lvl_zone_pts=5.0, lvl_stop_buffer=8.0,
                             lvl_ibh_wide_only=True)
        # PDH and ONH far away, only IBH=5020 in zone. NOT wide day.
        state = _make_lvl_state(prev_day_high=5200, on_high=5200,
                                prev_vah=5200, is_wide_ib=False)
        bar = _make_bar(5018, high=5021, low=5015)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is None  # IBH blocked on non-wide day

    def test_ibh_fires_on_wide_day(self):
        """IBH fires when wide day."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="any",
                             lvl_zone_pts=5.0, lvl_stop_buffer=8.0,
                             lvl_ibh_wide_only=True)
        # Only IBH=5020 in zone, wide day
        state = _make_lvl_state(prev_day_high=5200, on_high=5200,
                                prev_vah=5200, is_wide_ib=True)
        bar = _make_bar(5018, high=5021, low=5015)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is not None
        assert signal["setup"] == "LVL_IBH"

    def test_broken_level_skipped(self):
        """After level is broken (close above), no more entries there."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="any",
                             lvl_zone_pts=5.0, lvl_stop_buffer=8.0)
        state = _make_lvl_state()
        # Mark PDH as broken
        state.lvl_broken["PDH"] = True
        bar = _make_bar(5048, high=5051, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        # Should skip PDH and try next — ONH at 5035
        # Bar high=5051 >= 5035-5=5030 → ONH triggers
        assert signal is not None
        assert signal["setup"] == "LVL_ONH"

    def test_max_tests_exhaustion(self):
        """After 3 tests, level is skipped (defenders exhausted)."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="any",
                             lvl_zone_pts=5.0, lvl_max_tests=3)
        state = _make_lvl_state()
        # PDH tested 3 times already
        state.lvl_test_count["PDH"] = 3
        bar = _make_bar(5048, high=5051, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        # Should skip PDH, try ONH
        assert signal is not None
        assert signal["setup"] == "LVL_ONH"

    def test_failed_break_trigger(self):
        """High above level + close below = valid failed_break trigger."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="failed_break",
                             lvl_zone_pts=5.0, lvl_stop_buffer=8.0)
        state = _make_lvl_state()
        # High pierces above PDH (5050), close back below
        bar = _make_bar(5048, high=5053, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is not None
        assert signal["setup"] == "LVL_PDH"

    def test_failed_break_no_signal_when_close_above(self):
        """If close is above level, failed_break trigger does NOT fire."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="failed_break",
                             lvl_zone_pts=5.0, lvl_stop_buffer=8.0)
        state = _make_lvl_state()
        # Close ABOVE PDH → not a failed break
        bar = _make_bar(5052, high=5055, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        # PDH broken (close above), so try next levels
        # ONH=5035: high=5055 > 5035, close=5052 > 5035 → not failed break
        # IBH=5020: same logic
        # All levels have close above → no failed break anywhere
        assert signal is None

    def test_level_state_tracks_every_bar(self):
        """State updates even when not generating signals (e.g., in position)."""
        cfg = StrategyConfig(use_level_reject=True, lvl_zone_pts=5.0)
        state = _make_lvl_state()

        # Bar that reaches PDH zone
        bar = _make_bar(5048, high=5051, low=5045, is_rth=True, is_ib_period=False)
        level_rejection.update_level_state(bar, state, cfg)
        assert state.lvl_test_count.get("PDH", 0) >= 1

    def test_level_clustering(self):
        """When PDH and ONH within 5pts, only one signal fires (highest)."""
        cfg = StrategyConfig(use_level_reject=True, lvl_trigger="any",
                             lvl_zone_pts=5.0, lvl_stop_buffer=8.0)
        # PDH=5050, ONH=5048 — within 5pts of each other
        state = _make_lvl_state(on_high=5048)
        bar = _make_bar(5048, high=5051, low=5045)

        signal = level_rejection.check_signal(bar, None, state, cfg)
        assert signal is not None
        # Only one signal returned (first match = highest = PDH)
        assert signal["setup"] == "LVL_PDH"
