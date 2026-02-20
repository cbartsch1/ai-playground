"""
Medallion 2.0 — Regime-Filtered Backtester

Entry: Bullish regime + 7/8 confirmations → go long
Exit:  Regime flips to bearish/crash → close position
Cooldown: 48 hours after any exit before re-entry allowed
Leverage: Configurable (default 2.5x)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from config.settings import (
    INITIAL_CAPITAL,
    LEVERAGE,
    COOLDOWN_HOURS,
    MIN_CONFIRMATIONS,
    MIN_REGIME_CONFIDENCE,
    BULLISH_REGIMES,
    BEARISH_REGIMES,
)


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    regime_at_entry: str = ""
    regime_at_exit: str = ""
    confirmations_at_entry: int = 0
    hold_bars: int = 0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = None
    total_return: float = 0.0
    total_pnl: float = 0.0
    alpha_vs_bh: float = 0.0
    buy_hold_return: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    num_trades: int = 0
    avg_hold_bars: float = 0.0
    profit_factor: float = 0.0
    initial_capital: float = 0.0
    final_value: float = 0.0
    leverage: float = 0.0


class RegimeBacktester:
    """
    Backtester that combines HMM regime detection with confirmation voting.

    Entry: bullish regime + min_confirmations met
    Exit: regime flips to bearish or crash
    Cooldown: wait cooldown_hours after any exit
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        leverage: float = LEVERAGE,
        cooldown_hours: int = COOLDOWN_HOURS,
        min_confirmations: int = MIN_CONFIRMATIONS,
        min_confidence: float = MIN_REGIME_CONFIDENCE,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.cooldown_hours = cooldown_hours
        self.min_confirmations = min_confirmations
        self.min_confidence = min_confidence

    def run(
        self,
        ohlcv: pd.DataFrame,
        regime_predictions: pd.DataFrame,
        confirmations: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            ohlcv: OHLCV price data
            regime_predictions: Output from RegimeDetector.predict()
            confirmations: Output from compute_confirmations()

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        # Align all DataFrames
        idx = ohlcv.index.intersection(regime_predictions.index).intersection(confirmations.index)
        prices = ohlcv.loc[idx, "Close"]
        regimes = regime_predictions.loc[idx]
        confs = confirmations.loc[idx]

        capital = self.initial_capital
        position = 0.0  # number of shares/units held
        entry_price = 0.0
        entry_date = None
        entry_regime = ""
        entry_confs = 0
        cooldown_until = None

        trades = []
        equity = []
        bar_count_in_trade = 0

        for i, dt in enumerate(idx):
            price = float(prices.iloc[i])
            regime_label = regimes.loc[dt, "regime_label"] if pd.notna(regimes.loc[dt, "regime_label"]) else ""
            signal = regimes.loc[dt, "signal"] if pd.notna(regimes.loc[dt, "signal"]) else ""
            confidence = float(regimes.loc[dt, "confidence"]) if pd.notna(regimes.loc[dt, "confidence"]) else 0.0
            confs_met = int(confs.loc[dt, "confirmations_met"]) if pd.notna(confs.loc[dt, "confirmations_met"]) else 0

            # Current equity
            if position > 0:
                unrealized = (price - entry_price) * position
                current_equity = capital + unrealized
                bar_count_in_trade += 1
            else:
                current_equity = capital

            equity.append(current_equity)

            # === EXIT LOGIC ===
            if position > 0 and signal == "bearish":
                # Regime flipped to bear/crash — close immediately
                exit_pnl = (price - entry_price) * position
                capital += exit_pnl
                pnl_pct = (price - entry_price) / entry_price

                trade = Trade(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=dt,
                    exit_price=price,
                    pnl=exit_pnl,
                    pnl_pct=pnl_pct,
                    regime_at_entry=entry_regime,
                    regime_at_exit=regime_label,
                    confirmations_at_entry=entry_confs,
                    hold_bars=bar_count_in_trade,
                )
                trades.append(trade)

                position = 0.0
                bar_count_in_trade = 0
                cooldown_until = dt + pd.Timedelta(hours=self.cooldown_hours)
                continue

            # === ENTRY LOGIC ===
            if position == 0:
                # Check cooldown
                if cooldown_until is not None and dt < cooldown_until:
                    continue

                # Check regime is bullish with sufficient confidence
                if signal != "bullish" or confidence < self.min_confidence:
                    continue

                # Check confirmations
                if confs_met < self.min_confirmations:
                    continue

                # ENTER LONG
                position_size = capital * self.leverage
                position = position_size / price
                entry_price = price
                entry_date = dt
                entry_regime = regime_label
                entry_confs = confs_met
                bar_count_in_trade = 0

        # Close any open position at end
        if position > 0:
            price = float(prices.iloc[-1])
            exit_pnl = (price - entry_price) * position
            capital += exit_pnl
            pnl_pct = (price - entry_price) / entry_price

            trade = Trade(
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=idx[-1],
                exit_price=price,
                pnl=exit_pnl,
                pnl_pct=pnl_pct,
                regime_at_entry=entry_regime,
                regime_at_exit="End of Data",
                confirmations_at_entry=entry_confs,
                hold_bars=bar_count_in_trade,
            )
            trades.append(trade)
            position = 0.0

        # Build result
        equity_series = pd.Series(equity, index=idx, name="equity")
        buy_hold_return = (float(prices.iloc[-1]) / float(prices.iloc[0]) - 1) * 100

        result = BacktestResult()
        result.trades = trades
        result.equity_curve = equity_series
        result.initial_capital = self.initial_capital
        result.final_value = capital
        result.leverage = self.leverage
        result.total_pnl = capital - self.initial_capital
        result.total_return = (capital / self.initial_capital - 1) * 100
        result.buy_hold_return = buy_hold_return
        result.alpha_vs_bh = result.total_return - buy_hold_return
        result.num_trades = len(trades)

        if trades:
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]
            result.win_rate = len(winners) / len(trades) * 100
            result.avg_hold_bars = np.mean([t.hold_bars for t in trades])

            gross_profit = sum(t.pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        if not equity_series.empty:
            peak = equity_series.cummax()
            drawdown = (equity_series - peak) / peak
            result.max_drawdown = drawdown.min() * 100

        # Sharpe ratio (annualized)
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            if returns.std() > 0:
                # Assume hourly bars, ~6.5 trading hours/day, 252 days/year
                periods_per_year = 6.5 * 252
                result.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods_per_year)

        return result

    def get_trade_log(self, result: BacktestResult) -> pd.DataFrame:
        """Convert trades to a DataFrame for display."""
        if not result.trades:
            return pd.DataFrame()

        records = []
        for t in result.trades:
            records.append({
                "Entry Date": t.entry_date,
                "Exit Date": t.exit_date,
                "Entry Price": round(t.entry_price, 2),
                "Exit Price": round(t.exit_price, 2) if t.exit_price else None,
                "P&L": round(t.pnl, 2),
                "P&L %": f"{t.pnl_pct * 100:.2f}%",
                "Regime (Entry)": t.regime_at_entry,
                "Regime (Exit)": t.regime_at_exit,
                "Confirmations": f"{t.confirmations_at_entry}/8",
                "Hold Bars": t.hold_bars,
                "Result": "Win" if t.pnl > 0 else "Loss",
            })

        return pd.DataFrame(records)
