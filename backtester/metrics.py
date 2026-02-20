"""Performance metrics — PF, WR, DD, Sharpe, per-setup breakdown."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .position import Trade


@dataclass
class Metrics:
    """Summary statistics for a set of trades."""
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    calmar: float = 0.0
    trades_per_day: float = 0.0
    longest_win_streak: int = 0
    longest_lose_streak: int = 0


def compute_metrics(trades: List[Trade], initial_capital: float = 100_000.0) -> Metrics:
    """Compute performance metrics from a list of trades."""
    m = Metrics()

    if not trades:
        return m

    m.total_trades = len(trades)

    pnls = [t.pnl_dollar for t in trades]
    m.net_pnl = sum(pnls)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    m.winners = len(wins)
    m.losers = len(losses)
    m.win_rate = m.winners / m.total_trades * 100 if m.total_trades > 0 else 0

    m.gross_profit = sum(wins) if wins else 0
    m.gross_loss = abs(sum(losses)) if losses else 0

    m.profit_factor = m.gross_profit / m.gross_loss if m.gross_loss > 0 else float("inf")

    m.avg_win = np.mean(wins) if wins else 0
    m.avg_loss = np.mean(losses) if losses else 0
    m.avg_trade = np.mean(pnls) if pnls else 0

    # Max drawdown
    equity = initial_capital
    peak = equity
    max_dd = 0
    max_dd_pct = 0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
    m.max_drawdown = max_dd
    m.max_drawdown_pct = max_dd_pct

    # Sharpe (annualized, using trade returns)
    if len(pnls) > 1:
        returns = np.array(pnls) / initial_capital
        m.sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252) if np.std(returns, ddof=1) > 0 else 0
    else:
        m.sharpe = 0

    # Calmar = annualized return / max drawdown
    if m.max_drawdown > 0:
        # Estimate trading days from first to last trade
        first = trades[0].entry_time
        last = trades[-1].exit_time
        if hasattr(first, 'date') and hasattr(last, 'date'):
            days = max((last - first).days, 1)
        else:
            days = 90  # fallback
        annual_return = m.net_pnl * (365.0 / days)
        m.calmar = annual_return / m.max_drawdown
    else:
        m.calmar = float("inf") if m.net_pnl > 0 else 0

    # Trades per day
    if trades:
        first = trades[0].entry_time
        last = trades[-1].exit_time
        if hasattr(first, 'date') and hasattr(last, 'date'):
            days = max((last - first).days, 1)
            m.trades_per_day = m.total_trades / days
        else:
            m.trades_per_day = 0

    # Win/loss streaks
    streak = 0
    best_streak = 0
    worst_streak = 0
    for pnl in pnls:
        if pnl > 0:
            if streak > 0:
                streak += 1
            else:
                streak = 1
            best_streak = max(best_streak, streak)
        else:
            if streak < 0:
                streak -= 1
            else:
                streak = -1
            worst_streak = max(worst_streak, abs(streak))
    m.longest_win_streak = best_streak
    m.longest_lose_streak = worst_streak

    return m


def per_setup_breakdown(trades: List[Trade], initial_capital: float = 100_000.0) -> Dict[str, Metrics]:
    """Break down metrics by setup type (IB, VA, 80%)."""
    by_setup: Dict[str, List[Trade]] = {}
    for t in trades:
        by_setup.setdefault(t.setup, []).append(t)

    return {setup: compute_metrics(tlist, initial_capital) for setup, tlist in sorted(by_setup.items())}
