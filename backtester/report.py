"""Report generation — equity curve, trade log, summary text."""

import os
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from .config import StrategyConfig
from .position import Trade
from .metrics import compute_metrics, per_setup_breakdown, Metrics


def generate_report(trades: List[Trade], cfg: StrategyConfig,
                    output_dir: str = "output") -> str:
    """Generate full backtest report: trade log CSV, equity curve PNG, summary text.

    Returns the summary text.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Trade log CSV
    trade_log_path = os.path.join(output_dir, "trade_log.csv")
    _write_trade_log(trades, trade_log_path)

    # Metrics
    m = compute_metrics(trades, cfg.initial_capital)
    breakdown = per_setup_breakdown(trades, cfg.initial_capital)

    # Summary text
    summary = _format_summary(m, breakdown, cfg)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    # Equity curve
    equity_path = os.path.join(output_dir, "equity_curve.png")
    _plot_equity_curve(trades, cfg.initial_capital, equity_path)

    # Drawdown chart
    dd_path = os.path.join(output_dir, "drawdown.png")
    _plot_drawdown(trades, cfg.initial_capital, dd_path)

    print(summary)
    print(f"\nFiles saved to {output_dir}/")
    print(f"  trade_log.csv  — {len(trades)} trades")
    print(f"  equity_curve.png")
    print(f"  drawdown.png")
    print(f"  summary.txt")

    return summary


def _write_trade_log(trades: List[Trade], path: str) -> None:
    """Write trade log as CSV."""
    rows = []
    for t in trades:
        rows.append({
            "setup": t.setup,
            "direction": "LONG" if t.direction == 1 else "SHORT",
            "entry_time": t.entry_time,
            "entry_price": round(t.entry_price, 2),
            "exit_time": t.exit_time,
            "exit_price": round(t.exit_price, 2),
            "exit_reason": t.exit_reason,
            "stop": round(t.stop, 2),
            "target": round(t.target, 2),
            "pnl_pts": round(t.pnl_pts, 2),
            "pnl_dollar": round(t.pnl_dollar, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _format_summary(m: Metrics, breakdown: dict, cfg: StrategyConfig) -> str:
    """Format summary text."""
    lines = [
        "=" * 60,
        f"  AMT-TEMA Backtest Report — {cfg.instrument}",
        "=" * 60,
        "",
        f"  Net P&L:          ${m.net_pnl:,.2f}",
        f"  Total Trades:     {m.total_trades}",
        f"  Win Rate:         {m.win_rate:.1f}%",
        f"  Profit Factor:    {m.profit_factor:.3f}",
        f"  Max Drawdown:     ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.1f}%)",
        f"  Avg Win:          ${m.avg_win:,.2f}",
        f"  Avg Loss:         ${m.avg_loss:,.2f}",
        f"  Avg Trade:        ${m.avg_trade:,.2f}",
        f"  Sharpe:           {m.sharpe:.2f}",
        f"  Calmar:           {m.calmar:.2f}",
        f"  Trades/Day:       {m.trades_per_day:.2f}",
        f"  Win Streak:       {m.longest_win_streak}",
        f"  Lose Streak:      {m.longest_lose_streak}",
        "",
    ]

    if breakdown:
        lines.append("-" * 60)
        lines.append("  Per-Setup Breakdown")
        lines.append("-" * 60)
        for setup, sm in breakdown.items():
            lines.append(f"  {setup:6s}  |  {sm.total_trades:3d} trades  |  "
                        f"WR {sm.win_rate:5.1f}%  |  PF {sm.profit_factor:5.3f}  |  "
                        f"P&L ${sm.net_pnl:+,.2f}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def _plot_equity_curve(trades: List[Trade], initial_capital: float, path: str) -> None:
    """Plot equity curve and save to PNG."""
    if not trades:
        return

    times = [trades[0].entry_time]
    equity = [initial_capital]

    for t in trades:
        times.append(t.exit_time)
        equity.append(equity[-1] + t.pnl_dollar)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, equity, linewidth=1.5, color="#2196F3")
    ax.axhline(y=initial_capital, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(times, initial_capital, equity, alpha=0.1, color="#2196F3")

    ax.set_title("AMT-TEMA Equity Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_drawdown(trades: List[Trade], initial_capital: float, path: str) -> None:
    """Plot drawdown chart and save to PNG."""
    if not trades:
        return

    times = [trades[0].entry_time]
    equity_val = initial_capital
    peak = equity_val
    dd_values = [0]

    for t in trades:
        equity_val += t.pnl_dollar
        if equity_val > peak:
            peak = equity_val
        dd_values.append(peak - equity_val)
        times.append(t.exit_time)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(times, 0, dd_values, color="#F44336", alpha=0.4)
    ax.plot(times, dd_values, linewidth=1, color="#F44336")

    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown ($)")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
