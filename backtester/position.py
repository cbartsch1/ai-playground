"""Position tracking and fill logic.

Tracks a single position (no pyramiding), checks stop/target fills each bar.
Matches Pine Script strategy behavior: entry at close, stop/limit checked on next bar's OHLC.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    """Completed trade record."""
    setup: str           # "IB", "VA", "80%", "TX"
    direction: int       # +1 long, -1 short
    entry_time: object   # datetime index
    entry_price: float
    exit_time: object
    exit_price: float
    exit_reason: str     # "stop", "target", "trail", "tema_exit", "flatten"
    pnl_pts: float       # points P&L (before commission)
    pnl_dollar: float    # dollar P&L (after commission & slippage)
    stop: float
    target: float
    contract: int = 1    # Contract number in stagger (1=C1, 2=C2, 3=C3)


@dataclass
class Position:
    """Active position state."""
    direction: int = 0      # 0=flat, +1=long, -1=short
    entry_price: float = 0.0
    entry_time: object = None
    stop: float = 0.0
    target: float = 0.0
    setup: str = ""

    # Stagger contract tracking
    contract: int = 1               # Contract number (1=C1, 2=C2, 3=C3)

    # Trailing stop state (v9)
    trail_trigger_pts: float = 0.0   # Points of profit to activate trail
    trail_dist_pts: float = 0.0      # Trail distance from best price in points
    trail_active: bool = False
    best_price: float = 0.0          # Best price reached (lowest for short, highest for long)

    @property
    def is_flat(self) -> bool:
        return self.direction == 0

    def enter(self, direction: int, price: float, stop: float, target: float,
              setup: str, time: object, slippage: float = 0.0,
              trail_trigger_pts: float = 0.0, trail_dist_pts: float = 0.0,
              contract: int = 1) -> None:
        """Open a new position with slippage applied to entry."""
        self.direction = direction
        self.entry_price = price + slippage * direction  # slippage works against us
        self.stop = stop
        self.target = target
        self.setup = setup
        self.entry_time = time
        self.contract = contract
        self.trail_trigger_pts = trail_trigger_pts
        self.trail_dist_pts = trail_dist_pts
        self.trail_active = False
        self.best_price = self.entry_price

    def check_exit(self, bar: dict, pessimistic: bool = True) -> Optional[Trade]:
        """Check if stop, target, or trail stop is hit on this bar.

        Pine Script checks stop/limit against the bar's OHLC range.
        For longs: stop hit if low <= stop, target hit if high >= target
        For shorts: stop hit if high >= stop, target hit if low <= target

        When both could trigger (ambiguous), pessimistic=True means stop wins.
        Trail stop is between fixed stop and target in priority.
        """
        if self.is_flat:
            return None

        # Update trailing stop state
        trail_stop_level = None
        if self.trail_trigger_pts > 0:
            if self.direction == 1:  # Long
                self.best_price = max(self.best_price, bar["high"])
                profit = self.best_price - self.entry_price
                if profit >= self.trail_trigger_pts:
                    self.trail_active = True
                if self.trail_active:
                    trail_stop_level = self.best_price - self.trail_dist_pts
            else:  # Short
                self.best_price = min(self.best_price, bar["low"])
                profit = self.entry_price - self.best_price
                if profit >= self.trail_trigger_pts:
                    self.trail_active = True
                if self.trail_active:
                    trail_stop_level = self.best_price + self.trail_dist_pts

        # Check fixed stop and target
        stop_hit = False
        target_hit = False
        trail_hit = False

        if self.direction == 1:  # Long
            stop_hit = bar["low"] <= self.stop
            target_hit = bar["high"] >= self.target
            if trail_stop_level is not None:
                trail_hit = bar["low"] <= trail_stop_level
        else:  # Short
            stop_hit = bar["high"] >= self.stop
            target_hit = bar["low"] <= self.target
            if trail_stop_level is not None:
                trail_hit = bar["high"] >= trail_stop_level

        if not stop_hit and not target_hit and not trail_hit:
            return None

        # Priority: stop (worst) > trail > target (best) when pessimistic
        if pessimistic:
            if stop_hit:
                return self._close("stop", self.stop, bar)
            if trail_hit:
                return self._close("trail", trail_stop_level, bar)
            return self._close("target", self.target, bar)
        else:
            if target_hit:
                return self._close("target", self.target, bar)
            if trail_hit:
                return self._close("trail", trail_stop_level, bar)
            return self._close("stop", self.stop, bar)

    def close_at_market(self, bar: dict, reason: str = "flatten") -> Optional[Trade]:
        """Close at bar's close price with a custom reason."""
        if self.is_flat:
            return None
        return self._close(reason, bar["close"], bar)

    def flatten(self, bar: dict) -> Optional[Trade]:
        """Force close at bar's close price (session end)."""
        return self.close_at_market(bar, "flatten")

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
            contract=self.contract,
        )
        # Reset
        self.direction = 0
        self.entry_price = 0.0
        self.stop = 0.0
        self.target = 0.0
        self.setup = ""
        self.entry_time = None
        self.contract = 1
        self.trail_trigger_pts = 0.0
        self.trail_dist_pts = 0.0
        self.trail_active = False
        self.best_price = 0.0
        return trade
