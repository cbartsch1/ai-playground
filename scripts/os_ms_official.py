#!/usr/bin/env python3
"""Official Backtest: MS+OS Best — Market Structure + Overnight Sweep Combined.

Strategy:
  MS Config B: ON bilateral (ONH/ONL) + pVAH short-only, SMA 8/24 timing
  OS Best:     Short-only gap-up fade, ON extreme stop+5pt buffer, first RTH bar

This is the deployment candidate:
  437 trades, PF 1.411, +$30,402, p=0.004, Sharpe 2.17
  Walk-forward ratio 0.98 (PASS), Y2 p=0.037

Usage:
    python3 scripts/os_ms_official.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


OUTPUT_DIR = "output/os_ms_official"


def make_config():
    """MS Config B + OS Best — the deployment candidate."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"

    # All setups OFF except MS and OS
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_fa = False

    # --- MS Config B ---
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 4.0
    cfg.ms_min_target_pts = 8.0
    cfg.ms_min_rr = 0.3
    cfg.ms_max_risk = 25.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False
    cfg.ms_level_directions = {
        "MS_ONH": "both",
        "MS_ONL": "both",
        "MS_pVAH": "short",
    }

    # --- OS Best ---
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = 5.0
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 3.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_require_ma = False
    cfg.max_os_trades = 1
    cfg.os_min_gap = 3.0
    cfg.os_max_gap = 20.0
    cfg.os_entry_window = 1

    return cfg


# ═══════════════════════════════════════════════════════════════
#  CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════

SETUP_COLORS = {
    "MS_ONH":   "#E53935",  # red
    "MS_ONL":   "#43A047",  # green
    "MS_pVAH":  "#FB8C00",  # orange
    "OS_GAP_UP": "#7B1FA2", # purple
    "OS_GAP_DN": "#00897B", # teal
}

SETUP_MARKERS = {
    "MS_ONH":   "v",  # down triangle
    "MS_ONL":   "^",  # up triangle
    "MS_pVAH":  "v",  # down triangle
    "OS_GAP_UP": "D", # diamond
    "OS_GAP_DN": "D", # diamond
}


def plot_equity_curve(trades, initial_capital, path):
    """Equity curve with per-setup coloring."""
    if not trades:
        return

    fig, ax = plt.subplots(figsize=(16, 7))

    times = [trades[0].entry_time]
    equity = [initial_capital]
    for t in trades:
        times.append(t.exit_time)
        equity.append(equity[-1] + t.pnl_dollar)

    ax.plot(times, equity, linewidth=1.5, color="#1565C0", zorder=2)
    ax.fill_between(times, initial_capital, equity, alpha=0.08, color="#1565C0")
    ax.axhline(y=initial_capital, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # Mark winners/losers
    for t in trades:
        eq_before = initial_capital + sum(tt.pnl_dollar for tt in trades[:trades.index(t)])
        color = SETUP_COLORS.get(t.setup, "#888")
        marker = "^" if t.pnl_dollar > 0 else "v"
        alpha = 0.6 if t.pnl_dollar > 0 else 0.3
        ax.scatter(t.exit_time, eq_before + t.pnl_dollar, color=color,
                   marker=marker, s=15, alpha=alpha, zorder=3)

    ax.set_title("MS+OS Best — Equity Curve (2yr)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_drawdown(trades, initial_capital, path):
    """Drawdown chart."""
    if not trades:
        return

    fig, ax = plt.subplots(figsize=(16, 4))

    times = [trades[0].entry_time]
    eq = initial_capital
    peak = eq
    dd = [0]
    for t in trades:
        eq += t.pnl_dollar
        peak = max(peak, eq)
        dd.append(peak - eq)
        times.append(t.exit_time)

    ax.fill_between(times, 0, dd, color="#E53935", alpha=0.35)
    ax.plot(times, dd, linewidth=0.8, color="#E53935")
    ax.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown ($)")
    ax.grid(True, alpha=0.2)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_monthly_pnl(trades, path):
    """Monthly P&L bar chart."""
    if not trades:
        return

    monthly = defaultdict(float)
    for t in trades:
        key = t.exit_time.strftime("%Y-%m")
        monthly[key] += t.pnl_dollar

    months = sorted(monthly.keys())
    pnls = [monthly[m] for m in months]
    colors = ["#43A047" if p > 0 else "#E53935" for p in pnls]

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(range(len(months)), pnls, color=colors, alpha=0.8, edgecolor="none")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_title("Monthly P&L", fontsize=12, fontweight="bold")
    ax.set_ylabel("P&L ($)")
    ax.grid(True, alpha=0.2, axis="y")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Win/loss count
    win_months = sum(1 for p in pnls if p > 0)
    total_months = len(pnls)
    ax.text(0.02, 0.95, f"{win_months}/{total_months} winning months ({win_months/total_months*100:.0f}%)",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pnl_distribution(trades, path):
    """P&L distribution histogram."""
    if not trades:
        return

    pnls = [t.pnl_dollar for t in trades]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pnls, bins=40, color="#1565C0", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(x=np.mean(pnls), color="#E53935", linestyle="-", linewidth=1.5, label=f"Mean: ${np.mean(pnls):+,.0f}")
    ax.axvline(x=np.median(pnls), color="#43A047", linestyle="-", linewidth=1.5, label=f"Median: ${np.median(pnls):+,.0f}")

    ax.set_title("Trade P&L Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("P&L ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_setup_breakdown(trades, path):
    """Per-setup performance bar chart."""
    if not trades:
        return

    by_setup = defaultdict(list)
    for t in trades:
        by_setup[t.setup].append(t)

    setups = sorted(by_setup.keys(), key=lambda s: -sum(t.pnl_dollar for t in by_setup[s]))
    pnls = [sum(t.pnl_dollar for t in by_setup[s]) for s in setups]
    counts = [len(by_setup[s]) for s in setups]
    colors = [SETUP_COLORS.get(s, "#888") for s in setups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # P&L by setup
    ax1.barh(range(len(setups)), pnls, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(setups)))
    ax1.set_yticklabels(setups, fontsize=10)
    ax1.set_xlabel("P&L ($)")
    ax1.set_title("P&L by Setup", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="x")
    ax1.axvline(x=0, color="gray", linewidth=0.5)
    for i, (p, c) in enumerate(zip(pnls, counts)):
        ax1.text(max(p, 0) + 200, i, f"${p:+,.0f} ({c}t)", va="center", fontsize=9)

    # Win rate by setup
    wrs = [sum(1 for t in by_setup[s] if t.pnl_dollar > 0) / len(by_setup[s]) * 100 for s in setups]
    ax2.barh(range(len(setups)), wrs, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(setups)))
    ax2.set_yticklabels(setups, fontsize=10)
    ax2.set_xlabel("Win Rate (%)")
    ax2.set_title("Win Rate by Setup", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="x")
    ax2.axvline(x=50, color="gray", linestyle="--", linewidth=0.5)
    for i, w in enumerate(wrs):
        ax2.text(w + 1, i, f"{w:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_trades_on_price(trades, df, path, window_days=30, start_date=None):
    """Plot price chart with trade entries/exits marked.

    Shows a window_days slice of the price data with trade markers.
    """
    if not trades:
        return

    # Get RTH data only for cleaner charts
    rth = df[df["is_rth"]].copy()

    if start_date:
        rth = rth[rth.index >= start_date]
    rth = rth.iloc[:window_days * 78]  # ~78 RTH bars per day (6.5hr * 12 bars/hr)

    if len(rth) == 0:
        return

    chart_start = rth.index[0]
    chart_end = rth.index[-1]

    # Filter trades in this window
    window_trades = [t for t in trades
                     if t.entry_time >= chart_start and t.entry_time <= chart_end]

    if not window_trades:
        # Find a window with trades
        for t in trades:
            if hasattr(t.entry_time, 'date'):
                start_date = t.entry_time - pd.Timedelta(days=2)
                rth = df[df["is_rth"]].copy()
                rth = rth[(rth.index >= start_date)]
                rth = rth.iloc[:window_days * 78]
                chart_start = rth.index[0]
                chart_end = rth.index[-1]
                window_trades = [t for t in trades
                                 if t.entry_time >= chart_start and t.entry_time <= chart_end]
                if window_trades:
                    break

    fig, ax = plt.subplots(figsize=(18, 8))

    # Plot price as a line (close prices)
    ax.plot(rth.index, rth["close"], linewidth=0.6, color="#555", alpha=0.7, zorder=1)

    # Shade high-low range
    ax.fill_between(rth.index, rth["low"], rth["high"], alpha=0.05, color="#333")

    # Plot trades
    for t in window_trades:
        color = SETUP_COLORS.get(t.setup, "#888")
        entry_marker = "^" if t.direction == 1 else "v"
        win = t.pnl_dollar > 0

        # Entry arrow
        ax.scatter(t.entry_time, t.entry_price, color=color, marker=entry_marker,
                   s=80, zorder=5, edgecolors="black", linewidths=0.5)

        # Exit marker
        exit_marker = "o"
        exit_color = "#43A047" if win else "#E53935"
        ax.scatter(t.exit_time, t.exit_price, color=exit_color, marker=exit_marker,
                   s=40, zorder=5, edgecolors="black", linewidths=0.5)

        # Connect entry to exit
        line_color = "#43A047" if win else "#E53935"
        ax.plot([t.entry_time, t.exit_time], [t.entry_price, t.exit_price],
                color=line_color, linewidth=1.0, alpha=0.5, linestyle="--", zorder=3)

        # Stop and target lines
        ax.plot([t.entry_time, t.exit_time], [t.stop, t.stop],
                color="#E53935", linewidth=0.5, alpha=0.3, linestyle=":", zorder=2)
        ax.plot([t.entry_time, t.exit_time], [t.target, t.target],
                color="#43A047", linewidth=0.5, alpha=0.3, linestyle=":", zorder=2)

        # Label
        label = f"{t.setup}\n${t.pnl_dollar:+,.0f}"
        y_offset = 8 if t.direction == -1 else -8
        ax.annotate(label, (t.entry_time, t.entry_price),
                    textcoords="offset points", xytext=(5, y_offset),
                    fontsize=6, color=color, alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for setup, color in SETUP_COLORS.items():
        if any(t.setup == setup for t in window_trades):
            legend_elements.append(Line2D([0], [0], marker="D", color="w", markerfacecolor=color,
                                          markersize=8, label=setup))
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.8)

    date_range = f"{chart_start.strftime('%Y-%m-%d')} to {chart_end.strftime('%Y-%m-%d')}"
    ax.set_title(f"MS+OS Best — Trade Chart ({date_range}, {len(window_trades)} trades)", fontsize=12, fontweight="bold")
    ax.set_ylabel("ES Price")
    ax.grid(True, alpha=0.15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_walk_forward(trades, df, split_idx, initial_capital, path):
    """Walk-forward equity curve split Y1/Y2."""
    if not trades:
        return

    split_date = df.index[split_idx]
    t1 = [t for t in trades if t.entry_time < split_date]
    t2 = [t for t in trades if t.entry_time >= split_date]

    fig, ax = plt.subplots(figsize=(16, 7))

    # Y1
    times1 = [t1[0].entry_time] if t1 else []
    eq1 = [initial_capital]
    for t in t1:
        times1.append(t.exit_time)
        eq1.append(eq1[-1] + t.pnl_dollar)
    if times1:
        ax.plot(times1, eq1, linewidth=1.5, color="#1565C0", label=f"Y1 (in-sample): {len(t1)}t", zorder=2)
        ax.fill_between(times1, initial_capital, eq1, alpha=0.06, color="#1565C0")

    # Y2
    if t2:
        y2_start = eq1[-1] if eq1 else initial_capital
        times2 = [t2[0].entry_time]
        eq2 = [y2_start]
        for t in t2:
            times2.append(t.exit_time)
            eq2.append(eq2[-1] + t.pnl_dollar)
        ax.plot(times2, eq2, linewidth=1.5, color="#E65100", label=f"Y2 (out-of-sample): {len(t2)}t", zorder=2)
        ax.fill_between(times2, y2_start, eq2, alpha=0.06, color="#E65100")

    ax.axhline(y=initial_capital, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvline(x=split_date, color="red", linestyle="-", alpha=0.5, linewidth=1.5, label="Walk-Forward Split")

    ax.set_title("Walk-Forward Validation — Y1 In-Sample vs Y2 Out-of-Sample", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
cfg = make_config()
trades = run_backtest(df.copy(), cfg)
m = compute_metrics(trades, cfg.initial_capital)

print(f"Loaded {len(df)} bars")
print(f"Generated {len(trades)} trades\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  Official Results
# ═══════════════════════════════════════════════════════════════
pnls = [t.pnl_dollar for t in trades]
_, p_val = stats.ttest_1samp(pnls, 0)

# Bootstrap confidence interval
n_boot = 10000
rng = np.random.default_rng(42)
boot_means = [np.mean(rng.choice(pnls, size=len(pnls), replace=True)) for _ in range(n_boot)]
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
p_profit = np.mean([b > 0 for b in boot_means]) * 100

# Permutation test
n_perm = 5000
obs_mean = np.mean(pnls)
perm_count = 0
for _ in range(n_perm):
    shuffled = rng.choice([-1, 1], size=len(pnls)) * np.abs(pnls)
    if np.mean(shuffled) >= obs_mean:
        perm_count += 1
p_perm = perm_count / n_perm

longs = [t for t in trades if t.direction == 1]
shorts = [t for t in trades if t.direction == -1]
l_pnl = sum(t.pnl_dollar for t in longs)
s_pnl = sum(t.pnl_dollar for t in shorts)

trading_days = len(set(df[df["is_rth"]].index.date))

print("=" * 80)
print("  OFFICIAL BACKTEST: MS+OS Best")
print("  Market Structure (ON + pVAH short) + Overnight Sweep (gap-up fade)")
print("=" * 80)
print(f"""
  Data:             ES 5-min, 2 years ({df.index[0].date()} to {df.index[-1].date()})
  Bars:             {len(df):,}
  Trading Days:     {trading_days}

  {'─' * 60}
  PERFORMANCE
  {'─' * 60}
  Net P&L:          ${m.net_pnl:>+12,.2f}
  Total Trades:     {m.total_trades:>8d}
  Win Rate:         {m.win_rate:>8.1f}%
  Profit Factor:    {m.profit_factor:>8.3f}
  Avg Trade:        ${m.avg_trade:>+8,.2f}
  Avg Win:          ${m.avg_win:>+8,.2f}
  Avg Loss:         ${m.avg_loss:>+8,.2f}
  Win Streak:       {m.longest_win_streak:>8d}
  Lose Streak:      {m.longest_lose_streak:>8d}

  {'─' * 60}
  RISK
  {'─' * 60}
  Max Drawdown:     ${m.max_drawdown:>8,.2f} ({m.max_drawdown_pct:.1f}%)
  Sharpe:           {m.sharpe:>8.2f}
  Calmar:           {m.calmar:>8.2f}
  Trades/Day:       {m.trades_per_day:>8.2f}

  {'─' * 60}
  DIRECTION
  {'─' * 60}
  Long Trades:      {len(longs):>4d}  ${l_pnl:>+9,.0f}
  Short Trades:     {len(shorts):>4d}  ${s_pnl:>+9,.0f}

  {'─' * 60}
  STATISTICAL SIGNIFICANCE
  {'─' * 60}
  t-test p-value:     {p_val:.6f}  {'***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''}
  Permutation p:      {p_perm:.6f}  {'***' if p_perm < 0.01 else '**' if p_perm < 0.05 else '*' if p_perm < 0.10 else ''}
  Bootstrap P(profit): {p_profit:.1f}%
  95% CI avg trade:   ${ci_lo:+,.2f} to ${ci_hi:+,.2f}
""")

# Per-setup breakdown
print(f"  {'─' * 60}")
print(f"  PER-SETUP BREAKDOWN")
print(f"  {'─' * 60}")
breakdown = per_setup_breakdown(trades, cfg.initial_capital)
print(f"  {'Setup':<15s}  {'#':>4s}  {'WR':>6s}  {'PF':>7s}  {'P&L':>10s}  {'Avg':>8s}  {'DD':>8s}")
for setup, sm in sorted(breakdown.items(), key=lambda x: -x[1].net_pnl):
    avg = sm.net_pnl / sm.total_trades if sm.total_trades else 0
    print(f"  {setup:<15s}  {sm.total_trades:>4d}  {sm.win_rate:>5.1f}%  {sm.profit_factor:>7.3f}  "
          f"${sm.net_pnl:>+9,.0f}  ${avg:>+7,.0f}  ${sm.max_drawdown:>7,.0f}")

# Exit reasons
print(f"\n  {'─' * 60}")
print(f"  EXIT REASONS")
print(f"  {'─' * 60}")
reasons = {}
for t in trades:
    r = t.exit_reason
    reasons.setdefault(r, {"count": 0, "pnl": 0, "wins": 0})
    reasons[r]["count"] += 1
    reasons[r]["pnl"] += t.pnl_dollar
    if t.pnl_dollar > 0:
        reasons[r]["wins"] += 1
print(f"  {'Exit':<12s}  {'#':>4s}  {'P&L':>10s}  {'WR':>6s}")
for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
    wr = d["wins"] / d["count"] * 100
    print(f"  {r:<12s}  {d['count']:>4d}  ${d['pnl']:>+9,.0f}  {wr:>5.1f}%")

# Monthly breakdown
print(f"\n  {'─' * 60}")
print(f"  MONTHLY BREAKDOWN")
print(f"  {'─' * 60}")
monthly = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
for t in trades:
    key = t.exit_time.strftime("%Y-%m")
    monthly[key]["count"] += 1
    monthly[key]["pnl"] += t.pnl_dollar
    if t.pnl_dollar > 0:
        monthly[key]["wins"] += 1

print(f"  {'Month':<8s}  {'#':>4s}  {'WR':>6s}  {'P&L':>10s}  {'Cum P&L':>10s}")
cum_pnl = 0
for key in sorted(monthly.keys()):
    d = monthly[key]
    wr = d["wins"] / d["count"] * 100 if d["count"] else 0
    cum_pnl += d["pnl"]
    bar = "+" * max(0, int(d["pnl"] / 500)) + "-" * max(0, int(-d["pnl"] / 500))
    print(f"  {key:<8s}  {d['count']:>4d}  {wr:>5.1f}%  ${d['pnl']:>+9,.0f}  ${cum_pnl:>+9,.0f}  {bar}")

win_months = sum(1 for d in monthly.values() if d["pnl"] > 0)
print(f"\n  Winning months: {win_months}/{len(monthly)} ({win_months/len(monthly)*100:.0f}%)")

# Walk-forward
print(f"\n  {'─' * 60}")
print(f"  WALK-FORWARD VALIDATION")
print(f"  {'─' * 60}")
split_idx = df.index.get_indexer(["2025-02-14"], method="nearest")[0]
split_date = df.index[split_idx]

t1 = [t for t in trades if t.entry_time < split_date]
t2 = [t for t in trades if t.entry_time >= split_date]

if t1 and t2:
    m1 = compute_metrics(t1, cfg.initial_capital)
    m2 = compute_metrics(t2, cfg.initial_capital)
    _, p1 = stats.ttest_1samp([t.pnl_dollar for t in t1], 0) if len(t1) >= 5 else (0, 1.0)
    _, p2 = stats.ttest_1samp([t.pnl_dollar for t in t2], 0) if len(t2) >= 5 else (0, 1.0)
    ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
    verdict = "PASS" if ratio > 0.7 and m2.profit_factor > 1.0 else "MARGINAL" if ratio > 0.5 else "FAIL"

    print(f"""
  Split:  {split_date.date()}
  Y1 (in-sample):     {m1.total_trades:>4d}t  PF={m1.profit_factor:.3f}  ${m1.net_pnl:>+9,.0f}  Sh={m1.sharpe:.2f}  p={p1:.4f}
  Y2 (out-of-sample):  {m2.total_trades:>4d}t  PF={m2.profit_factor:.3f}  ${m2.net_pnl:>+9,.0f}  Sh={m2.sharpe:.2f}  p={p2:.4f}
  WF ratio:            {ratio:.2f}  → {verdict}
""")

# ═══════════════════════════════════════════════════════════════
#  Generate Charts
# ═══════════════════════════════════════════════════════════════
print(f"{'─' * 60}")
print(f"  GENERATING CHARTS → {OUTPUT_DIR}/")
print(f"{'─' * 60}")

plot_equity_curve(trades, cfg.initial_capital, f"{OUTPUT_DIR}/equity_curve.png")
plot_drawdown(trades, cfg.initial_capital, f"{OUTPUT_DIR}/drawdown.png")
plot_monthly_pnl(trades, f"{OUTPUT_DIR}/monthly_pnl.png")
plot_pnl_distribution(trades, f"{OUTPUT_DIR}/pnl_distribution.png")
plot_setup_breakdown(trades, f"{OUTPUT_DIR}/setup_breakdown.png")
plot_walk_forward(trades, df, split_idx, cfg.initial_capital, f"{OUTPUT_DIR}/walk_forward.png")

# Trade charts — pick interesting windows
# 1. Best month
best_month = max(monthly.keys(), key=lambda k: monthly[k]["pnl"])
best_month_start = pd.Timestamp(best_month + "-01", tz="US/Eastern")
plot_trades_on_price(trades, df, f"{OUTPUT_DIR}/trades_best_month.png",
                     window_days=22, start_date=best_month_start)

# 2. Most recent month
last_month = max(monthly.keys())
last_month_start = pd.Timestamp(last_month + "-01", tz="US/Eastern")
plot_trades_on_price(trades, df, f"{OUTPUT_DIR}/trades_recent.png",
                     window_days=22, start_date=last_month_start)

# 3. A 5-day zoom showing individual trades
# Find a week with multiple trades
from collections import Counter
week_counts = Counter()
for t in trades:
    week_key = t.entry_time.isocalendar()[:2]
    week_counts[week_key] += 1
best_week = max(week_counts, key=week_counts.get)
# Find the Monday of that week
for t in trades:
    wk = t.entry_time.isocalendar()[:2]
    if wk == best_week:
        week_start = t.entry_time - pd.Timedelta(days=t.entry_time.weekday())
        break
plot_trades_on_price(trades, df, f"{OUTPUT_DIR}/trades_5day_zoom.png",
                     window_days=5, start_date=week_start)

# Write trade log CSV
trade_rows = []
for t in trades:
    trade_rows.append({
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
trade_df = pd.DataFrame(trade_rows)
trade_df.to_csv(f"{OUTPUT_DIR}/trade_log.csv", index=False)
print(f"  Saved: {OUTPUT_DIR}/trade_log.csv ({len(trades)} trades)")

# Write summary text
with open(f"{OUTPUT_DIR}/summary.txt", "w") as f:
    f.write(f"MS+OS Best Official Backtest\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Data: ES 5-min, {df.index[0].date()} to {df.index[-1].date()}\n\n")
    f.write(f"Trades: {m.total_trades}\n")
    f.write(f"Win Rate: {m.win_rate:.1f}%\n")
    f.write(f"Profit Factor: {m.profit_factor:.3f}\n")
    f.write(f"Net P&L: ${m.net_pnl:+,.2f}\n")
    f.write(f"Max Drawdown: ${m.max_drawdown:,.2f}\n")
    f.write(f"Sharpe: {m.sharpe:.2f}\n")
    f.write(f"t-test p: {p_val:.6f}\n")
    f.write(f"Permutation p: {p_perm:.6f}\n")
    f.write(f"Bootstrap P(profit): {p_profit:.1f}%\n")
    f.write(f"WF ratio: {ratio:.2f} ({verdict})\n")
print(f"  Saved: {OUTPUT_DIR}/summary.txt")

print(f"\n{'=' * 80}")
print(f"  DONE — All output in {OUTPUT_DIR}/")
print(f"{'=' * 80}")
