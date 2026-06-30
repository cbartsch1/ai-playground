#!/usr/bin/env python3
"""Full Validation: Nexus Long — Dev VAL + Dev POC + VIX Inverted Filter

Config under test:
  - MS levels: ONL (both), ONH (both), pVAH (short-blocked), dVAL (long), dPOC (long)
  - OS: gap-down fades, first bar, cascade targets
  - VIX inversion: MS fires when VIX open > 17, OS fires when VIX open <= 17
  - direction_filter = "long"

Validation pipeline:
  1. Full 2yr backtest — headline numbers
  2. Per-setup breakdown — which setups carry the edge
  3. Walk-forward (Y1 IS / Y2 OOS) — does it hold out of sample
  4. Statistical battery — t-test, permutation, bootstrap
  5. Monthly consistency — winning months, streaks
  6. Drawdown analysis — max DD, recovery, worst trades
  7. Year-by-year stability — each year standalone
  8. VIX regime check — edge real across VIX ranges, not just one bucket
  9. Day-of-week — any day-specific pattern (Friday drag?)
  10. Exit reason analysis — where does the P&L come from

Usage:
    cd ~/projects/backtesting/es
    python3 scripts/nexus_long_validation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


def make_config():
    """Nexus Long — Dev VAL + Dev POC + VIX inverted."""
    cfg = StrategyConfig()
    cfg.direction_filter = "long"

    # All OFF except MS + OS
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

    # MS config
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
    cfg.ms_use_dev_va = True       # ← KEY: developing VAL
    cfg.ms_use_poc = True          # ← KEY: developing POC
    cfg.ms_level_directions = {
        "MS_ONH": "both",
        "MS_ONL": "both",
        "MS_pVAH": "short",   # blocked by direction_filter="long"
        "MS_dVAL": "long",
        "MS_dPOC": "long",
    }

    # OS config
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


def apply_vix_filter(trades, vix_df, ms_thresh=17, os_thresh=17):
    """Inverted VIX: MS keeps high VIX, OS keeps low VIX."""
    result = []
    for t in trades:
        vix_row = vix_df[vix_df.index.date == t.entry_time.date()]
        if len(vix_row) == 0:
            continue
        t.vix_open = vix_row.iloc[0]["open"]
        t.vix_high = vix_row.iloc[0]["high"]
        t.vix_close = vix_row.iloc[0]["close"]

        if t.setup.startswith("MS_"):
            if t.vix_open > ms_thresh:
                result.append(t)
        elif t.setup.startswith("OS_"):
            if t.vix_open <= os_thresh:
                result.append(t)
        else:
            result.append(t)
    return result


def pf(trades):
    """Compute profit factor."""
    gw = sum(t.pnl_dollar for t in trades if t.pnl_dollar > 0)
    gl = abs(sum(t.pnl_dollar for t in trades if t.pnl_dollar <= 0))
    return gw / gl if gl > 0 else float("inf")


# ═══════════════════════════════════════════════════════════════
#  LOAD DATA & RUN
# ═══════════════════════════════════════════════════════════════

print("Loading data...")
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
vix = pd.read_parquet("../spx/data/vix_daily.parquet")
print(f"ES: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")
print(f"VIX: {len(vix)} days\n")

cfg = make_config()
all_trades = run_backtest(df.copy(), cfg)
trades = apply_vix_filter(all_trades, vix)
trades.sort(key=lambda t: t.entry_time)

unfiltered_count = len(all_trades)
filtered_count = len(trades)

# ═══════════════════════════════════════════════════════════════
#  1. HEADLINE NUMBERS
# ═══════════════════════════════════════════════════════════════
m = compute_metrics(trades, cfg.initial_capital)
pnls = [t.pnl_dollar for t in trades]
_, p_ttest = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)

trading_days = len(set(df[df["is_rth"]].index.date))

print("=" * 80)
print("  NEXUS LONG — FULL VALIDATION")
print("  Dev VAL + Dev POC + ONL/ONH + OS gap-down")
print("  VIX Inverted Filter: MS when VIX > 17, OS when VIX <= 17")
print("=" * 80)
print(f"""
  Data:             ES 5-min, 2 years ({df.index[0].date()} to {df.index[-1].date()})
  Bars:             {len(df):,}
  Trading Days:     {trading_days}
  Raw Trades:       {unfiltered_count} (before VIX filter)
  Filtered Trades:  {filtered_count} (after VIX filter)

  {'─' * 60}
  PERFORMANCE
  {'─' * 60}
  Net P&L:          ${m.net_pnl:>+12,.2f}
  Total Trades:     {m.total_trades:>8d}  ({m.total_trades/2:.0f}/yr, {m.total_trades/trading_days:.2f}/day)
  Win Rate:         {m.win_rate:>8.1f}%
  Profit Factor:    {m.profit_factor:>8.3f}
  Avg Trade:        ${m.avg_trade:>+8,.2f}
  Avg Win:          ${m.avg_win:>+8,.2f}
  Avg Loss:         ${m.avg_loss:>+8,.2f}
  Best Trade:       ${max(pnls):>+8,.2f}
  Worst Trade:      ${min(pnls):>+8,.2f}
  Win Streak:       {m.longest_win_streak:>8d}
  Lose Streak:      {m.longest_lose_streak:>8d}

  {'─' * 60}
  RISK
  {'─' * 60}
  Max Drawdown:     ${m.max_drawdown:>8,.2f} ({m.max_drawdown_pct:.1f}%)
  Sharpe:           {m.sharpe:>8.2f}
  Calmar:           {m.calmar:>8.2f}
""")

# ═══════════════════════════════════════════════════════════════
#  2. PER-SETUP BREAKDOWN
# ═══════════════════════════════════════════════════════════════
print(f"  {'─' * 60}")
print(f"  PER-SETUP BREAKDOWN")
print(f"  {'─' * 60}")
breakdown = per_setup_breakdown(trades, cfg.initial_capital)
print(f"  {'Setup':<15s}  {'#':>4s}  {'WR':>6s}  {'PF':>7s}  {'P&L':>10s}  {'Avg':>8s}  {'DD':>8s}")
for setup, sm in sorted(breakdown.items(), key=lambda x: -x[1].net_pnl):
    avg = sm.net_pnl / sm.total_trades if sm.total_trades else 0
    print(f"  {setup:<15s}  {sm.total_trades:>4d}  {sm.win_rate:>5.1f}%  {sm.profit_factor:>7.3f}  "
          f"${sm.net_pnl:>+9,.0f}  ${avg:>+7,.0f}  ${sm.max_drawdown:>7,.0f}")

# ═══════════════════════════════════════════════════════════════
#  3. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'─' * 60}")
print(f"  WALK-FORWARD VALIDATION")
print(f"  {'─' * 60}")

split_date = pd.Timestamp("2025-02-14", tz="US/Eastern")
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
  Split:            {split_date.date()}
  Y1 (in-sample):   {m1.total_trades:>4d}t  WR={m1.win_rate:.1f}%  PF={m1.profit_factor:.3f}  ${m1.net_pnl:>+9,.0f}  Sh={m1.sharpe:.2f}  p={p1:.4f}
  Y2 (out-of-sample):{m2.total_trades:>4d}t  WR={m2.win_rate:.1f}%  PF={m2.profit_factor:.3f}  ${m2.net_pnl:>+9,.0f}  Sh={m2.sharpe:.2f}  p={p2:.4f}
  WF ratio:          {ratio:.2f} → {verdict}
""")

    # Per-setup Y1 vs Y2
    print(f"  Per-setup walk-forward:")
    setups = sorted(set(t.setup for t in trades))
    for setup in setups:
        s1 = [t for t in t1 if t.setup == setup]
        s2 = [t for t in t2 if t.setup == setup]
        pf1 = pf(s1) if s1 else 0
        pf2 = pf(s2) if s2 else 0
        p1s = sum(t.pnl_dollar for t in s1)
        p2s = sum(t.pnl_dollar for t in s2)
        r = pf2 / pf1 if pf1 > 0 else 0
        v = "PASS" if r > 0.7 and pf2 > 1.0 else "MARGINAL" if r > 0.5 else "FAIL"
        print(f"    {setup:<12s}  Y1: {len(s1):>3d}t PF {pf1:.3f} ${p1s:>+8,.0f}  |  Y2: {len(s2):>3d}t PF {pf2:.3f} ${p2s:>+8,.0f}  |  {r:.2f} → {v}")

# ═══════════════════════════════════════════════════════════════
#  4. STATISTICAL BATTERY
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'─' * 60}")
print(f"  STATISTICAL SIGNIFICANCE")
print(f"  {'─' * 60}")

rng = np.random.default_rng(42)

# t-test
print(f"  t-test p-value:        {p_ttest:.6f}  {'***' if p_ttest < 0.01 else '**' if p_ttest < 0.05 else '*' if p_ttest < 0.10 else ''}")

# Permutation test
n_perm = 10000
obs_mean = np.mean(pnls)
perm_count = sum(1 for _ in range(n_perm)
                 if np.mean(rng.choice([-1, 1], size=len(pnls)) * np.abs(pnls)) >= obs_mean)
p_perm = perm_count / n_perm
print(f"  Permutation p:         {p_perm:.6f}  {'***' if p_perm < 0.01 else '**' if p_perm < 0.05 else '*' if p_perm < 0.10 else ''}")

# Bootstrap
n_boot = 10000
boot_means = [np.mean(rng.choice(pnls, size=len(pnls), replace=True)) for _ in range(n_boot)]
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
p_profit = np.mean([b > 0 for b in boot_means]) * 100
print(f"  Bootstrap P(profit):   {p_profit:.1f}%")
print(f"  95% CI avg trade:      ${ci_lo:+,.2f} to ${ci_hi:+,.2f}")

# Wilcoxon signed-rank (non-parametric, no normality assumption)
_, p_wilcox = stats.wilcoxon(pnls, alternative="greater") if len(pnls) >= 10 else (0, 1.0)
print(f"  Wilcoxon signed-rank p:{p_wilcox:.6f}  {'***' if p_wilcox < 0.01 else '**' if p_wilcox < 0.05 else '*' if p_wilcox < 0.10 else ''}")

# ═══════════════════════════════════════════════════════════════
#  5. MONTHLY CONSISTENCY
# ═══════════════════════════════════════════════════════════════
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
    bar = "+" * max(0, int(d["pnl"] / 400)) + "-" * max(0, int(-d["pnl"] / 400))
    print(f"  {key:<8s}  {d['count']:>4d}  {wr:>5.1f}%  ${d['pnl']:>+9,.0f}  ${cum_pnl:>+9,.0f}  {bar}")

win_months = sum(1 for d in monthly.values() if d["pnl"] > 0)
lose_months = sum(1 for d in monthly.values() if d["pnl"] <= 0)
print(f"\n  Winning months: {win_months}/{len(monthly)} ({win_months/len(monthly)*100:.0f}%)")

# Consecutive losing months
max_lose_streak = 0
current_streak = 0
for key in sorted(monthly.keys()):
    if monthly[key]["pnl"] <= 0:
        current_streak += 1
        max_lose_streak = max(max_lose_streak, current_streak)
    else:
        current_streak = 0
print(f"  Max consecutive losing months: {max_lose_streak}")

# ═══════════════════════════════════════════════════════════════
#  6. DRAWDOWN ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'─' * 60}")
print(f"  DRAWDOWN ANALYSIS")
print(f"  {'─' * 60}")

eq = 0
peak = 0
dd_current = 0
dd_start = None
max_dd = 0
max_dd_start = None
max_dd_end = None
worst_5 = sorted(pnls)[:5]
dd_trades = 0  # trades to recover from max DD

for i, t in enumerate(trades):
    eq += t.pnl_dollar
    if eq > peak:
        peak = eq
        dd_start = None
        dd_trades = 0
    dd_current = peak - eq
    if dd_current > max_dd:
        max_dd = dd_current
        max_dd_end = t.exit_time

# Recovery: trades from max DD to new peak
in_dd = False
dd_count = 0
for t in trades:
    eq_check = sum(tt.pnl_dollar for tt in trades[:trades.index(t)+1])
    peak_check = max(sum(tt.pnl_dollar for tt in trades[:j+1]) for j in range(trades.index(t)+1))
    if peak_check - eq_check >= max_dd * 0.9:
        in_dd = True
        dd_count = 0
    if in_dd:
        dd_count += 1
        if eq_check >= peak_check:
            in_dd = False

print(f"  Max Drawdown:       ${max_dd:>8,.0f}  ({max_dd/cfg.initial_capital*100:.1f}%)")
print(f"  Worst 5 trades:     {['${:+,.0f}'.format(p) for p in worst_5]}")
print(f"  Max lose streak:    {m.longest_lose_streak}")

# Consecutive losers distribution
streaks = []
current = 0
for p in pnls:
    if p <= 0:
        current += 1
    else:
        if current > 0:
            streaks.append(current)
        current = 0
if current > 0:
    streaks.append(current)
if streaks:
    print(f"  Lose streak dist:   {dict(sorted(defaultdict(int, {s: streaks.count(s) for s in set(streaks)}).items()))}")

# ═══════════════════════════════════════════════════════════════
#  7. YEAR-BY-YEAR STABILITY
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'─' * 60}")
print(f"  YEAR-BY-YEAR PERFORMANCE")
print(f"  {'─' * 60}")

for year in sorted(set(t.entry_time.year for t in trades)):
    yt = [t for t in trades if t.entry_time.year == year]
    yp = [t.pnl_dollar for t in yt]
    yw = sum(1 for p in yp if p > 0)
    ywr = yw / len(yt) * 100
    ygw = sum(p for p in yp if p > 0)
    ygl = abs(sum(p for p in yp if p <= 0))
    ypf = ygw / ygl if ygl > 0 else float("inf")
    ynet = sum(yp)
    _, yp_val = stats.ttest_1samp(yp, 0) if len(yp) >= 5 else (0, 1.0)
    sig = "***" if yp_val < 0.01 else "**" if yp_val < 0.05 else "*" if yp_val < 0.10 else ""
    print(f"  {year}:  {len(yt):>4d}t  WR {ywr:>5.1f}%  PF {ypf:>6.3f}  ${ynet:>+9,.0f}  p={yp_val:.4f} {sig}")

# ═══════════════════════════════════════════════════════════════
#  8. VIX REGIME CHECK
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'─' * 60}")
print(f"  VIX REGIME CHECK (edge shouldn't concentrate in one VIX bucket)")
print(f"  {'─' * 60}")

# Note: VIX filter already applied, so this shows the distribution WITHIN the filtered set
vix_buckets = {
    "VIX 12-16": (12, 16),
    "VIX 16-20": (16, 20),
    "VIX 20-25": (20, 25),
    "VIX 25-30": (25, 30),
    "VIX 30+":   (30, 100),
}

for label, (lo, hi) in vix_buckets.items():
    bucket = [t for t in trades if lo <= t.vix_open < hi]
    if not bucket:
        print(f"  {label:<12s}    0 trades")
        continue
    bp = [t.pnl_dollar for t in bucket]
    bw = sum(1 for p in bp if p > 0)
    bgw = sum(p for p in bp if p > 0)
    bgl = abs(sum(p for p in bp if p <= 0))
    bpf = bgw / bgl if bgl > 0 else float("inf")
    print(f"  {label:<12s}  {len(bucket):>4d}t  WR {bw/len(bucket)*100:>5.1f}%  PF {bpf:>6.3f}  ${sum(bp):>+9,.0f}")

# ═══════════════════════════════════════════════════════════════
#  9. DAY-OF-WEEK ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'─' * 60}")
print(f"  DAY-OF-WEEK ANALYSIS")
print(f"  {'─' * 60}")

days = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
for dow in range(5):
    dt = [t for t in trades if t.entry_time.weekday() == dow]
    if not dt:
        print(f"  {days[dow]:<4s}    0 trades")
        continue
    dp = [t.pnl_dollar for t in dt]
    dw = sum(1 for p in dp if p > 0)
    dgw = sum(p for p in dp if p > 0)
    dgl = abs(sum(p for p in dp if p <= 0))
    dpf = dgw / dgl if dgl > 0 else float("inf")
    print(f"  {days[dow]:<4s}  {len(dt):>4d}t  WR {dw/len(dt)*100:>5.1f}%  PF {dpf:>6.3f}  ${sum(dp):>+9,.0f}")

# ═══════════════════════════════════════════════════════════════
#  10. EXIT REASON ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'─' * 60}")
print(f"  EXIT REASONS")
print(f"  {'─' * 60}")

reasons = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
for t in trades:
    r = t.exit_reason
    reasons[r]["count"] += 1
    reasons[r]["pnl"] += t.pnl_dollar
    if t.pnl_dollar > 0:
        reasons[r]["wins"] += 1

print(f"  {'Exit':<12s}  {'#':>4s}  {'P&L':>10s}  {'WR':>6s}  {'Avg':>8s}")
for r in sorted(reasons.keys(), key=lambda x: -reasons[x]["count"]):
    d = reasons[r]
    wr = d["wins"] / d["count"] * 100
    avg = d["pnl"] / d["count"]
    print(f"  {r:<12s}  {d['count']:>4d}  ${d['pnl']:>+9,.0f}  {wr:>5.1f}%  ${avg:>+7,.0f}")

# ═══════════════════════════════════════════════════════════════
#  11. HOLD TIME ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'─' * 60}")
print(f"  HOLD TIME")
print(f"  {'─' * 60}")

hold_mins = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]
win_holds = [h for h, t in zip(hold_mins, trades) if t.pnl_dollar > 0]
lose_holds = [h for h, t in zip(hold_mins, trades) if t.pnl_dollar <= 0]

print(f"  Avg hold (all):    {np.mean(hold_mins):>6.0f} min")
print(f"  Avg hold (wins):   {np.mean(win_holds):>6.0f} min" if win_holds else "  Avg hold (wins):   N/A")
print(f"  Avg hold (losses): {np.mean(lose_holds):>6.0f} min" if lose_holds else "  Avg hold (losses): N/A")
print(f"  Median hold:       {np.median(hold_mins):>6.0f} min")
print(f"  Max hold:          {max(hold_mins):>6.0f} min")

# ═══════════════════════════════════════════════════════════════
#  12. ENTRY TIME DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
print(f"\n  {'─' * 60}")
print(f"  ENTRY TIME DISTRIBUTION")
print(f"  {'─' * 60}")

hour_buckets = defaultdict(lambda: {"count": 0, "pnl": 0})
for t in trades:
    h = t.entry_time.hour
    hour_buckets[h]["count"] += 1
    hour_buckets[h]["pnl"] += t.pnl_dollar

for h in sorted(hour_buckets.keys()):
    d = hour_buckets[h]
    bar = "+" * max(0, int(d["pnl"] / 500)) + "-" * max(0, int(-d["pnl"] / 500))
    print(f"  {h:>2d}:00  {d['count']:>4d}t  ${d['pnl']:>+9,.0f}  {bar}")

# ═══════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"  FINAL VERDICT")
print(f"{'=' * 80}")

checks = []
checks.append(("Trades >= 100", len(trades) >= 100, len(trades)))
checks.append(("PF > 1.2", m.profit_factor > 1.2, f"{m.profit_factor:.3f}"))
checks.append(("t-test p < 0.05", p_ttest < 0.05, f"{p_ttest:.4f}"))
checks.append(("Permutation p < 0.05", p_perm < 0.05, f"{p_perm:.4f}"))
checks.append(("Bootstrap P(profit) > 90%", p_profit > 90, f"{p_profit:.1f}%"))
if t1 and t2:
    checks.append(("WF ratio > 0.7", ratio > 0.7, f"{ratio:.2f}"))
    checks.append(("OOS PF > 1.0", m2.profit_factor > 1.0, f"{m2.profit_factor:.3f}"))
    checks.append(("OOS p < 0.10", p2 < 0.10, f"{p2:.4f}"))
checks.append(("Both years profitable", all(
    sum(t.pnl_dollar for t in trades if t.entry_time.year == y) > 0
    for y in set(t.entry_time.year for t in trades)
), ""))
checks.append(("Win months > 60%", win_months/len(monthly) > 0.60, f"{win_months/len(monthly)*100:.0f}%"))
checks.append(("Max DD < 5%", m.max_drawdown_pct < 5, f"{m.max_drawdown_pct:.1f}%"))

passed = sum(1 for _, ok, _ in checks if ok)
total = len(checks)

for label, ok, val in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label:<35s} ({val})")

print(f"\n  Score: {passed}/{total}")
if passed >= total - 1:
    print(f"  VERDICT: DEPLOY CANDIDATE — run through AR optimization next")
elif passed >= total - 3:
    print(f"  VERDICT: PROMISING — needs more data or parameter tuning")
else:
    print(f"  VERDICT: NOT READY — fundamental issues remain")

print()
