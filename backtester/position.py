"""Position tracking and fill logic.

Tracks a single position (no pyramiding), checks stop/target fills each bar.
Matches Pine Script strategy behavior: entry at close, stop/limit checked on next bar's OHLC.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    """Completed trade record."""
    setup: str           # "IB", "VA", "80%"
    direction: int       # +1 long, -1 short
    entry_time: object   # datetime index
    entry_price: float
    exit_time: object
    exit_price: float
    exit_reason: str     # "stop", "target", "flatten"
    pnl_pts: float       # points P&L (before commission)
    pnl_dollar: float    # dollar P&L (after commission & slippage)
    stop: float
    target: float


@dataclass
class Position:
    """Active position state."""
    direction: int = 0      # 0=flat, +1=long, -1=short
    entry_price: float = 0.0
    entry_time: object = None
    stop: float = 0.0
    target: float = 0.0
    setup: str = ""

    @property
    def is_flat(self) -> bool:
        return self.direction == 0

    def enter(self, direction: int, price: float, stop: float, target: float,
              setup: str, time: object, slippage: float = 0.0) -> None:
        """Open a new position with slippage applied to entry."""
        self.direction = direction
        self.entry_price = price + slippage * direction  # slippage works against us
        self.stop = stop
        self.target = target
        self.setup = setup
        self.entry_time = time

    def check_exit(self, bar: dict, pessimistic: bool = True) -> Optional[Trade]:
        """Check if stop or target is hit on this bar.

        Pine Script checks stop/limit against the bar's OHLC range.
        For longs: stop hit if low <= stop, target hit if high >= target
        For shorts: stop hit if high >= stop, target hit if low <= target

        When both could trigger (ambiguous), pessimistic=True means stop wins.
        """
        if self.is_flat:
            return None

        stop_hit = False
        target_hit = False

        if self.direction == 1:  # Long
            stop_hit = bar["low"] <= self.stop
            target_hit = bar["high"] >= self.target
        else:  # Short
            stop_hit = bar["high"] >= self.stop
            target_hit = bar["low"] <= self.target

        if not stop_hit and not target_hit:
            return None

        # Determine which fills
        if stop_hit and target_hit:
            # Ambiguous — both could hit within this bar
            if pessimistic:
                return self._close("stop", self.stop, bar)
            else:
                return self._close("target", self.target, bar)
        elif stop_hit:
            return self._close("stop", self.stop, bar)
        else:
            return self._close("target", self.target, bar)

    def flatten(self, bar: dict) -> Optional[Trade]:
        """Force close at bar's close price (session end)."""
        if self.is_flat:
            return None
        return self._close("flatten", bar["close"], bar)

    def _close(self, reason: str, exit_price: float, bar: dict) -> Trade:
        """Create trade record and reset position."""
        pnl_pts = (exit_price - self.entry_price) * self.direction
        trade = Trade(
            setup=self.setup,
            direction=self.direction,
            entry_time=self.entry_time,
            entry_price=self.entry_price,
            exit_time=bar["_time"],
            exit_price=exit_price,
            exit_reason=reason,
            pnl_pts=pnl_pts,
            pnl_dollar=0.0,  # Computed by engine with commission
            stop=self.stop,
            target=self.target,
        )
        # Reset
        self.direction = 0
        self.entry_price = 0.0
        self.stop = 0.0
        self.target = 0.0
        self.setup = ""
        self.entry_time = None
        return trade
