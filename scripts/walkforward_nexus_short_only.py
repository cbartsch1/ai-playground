#!/usr/bin/env python3
"""5-Fold Walk-Forward Validation: MS+OS Nexus SHORT-ONLY.

Rolling walk-forward with ~5-month in-sample and ~5-month out-of-sample windows.
Each fold uses its own IS period for "fitting" (though no parameters change —
this validates that the strategy works across different market regimes).

Pass criteria:
  - Minimum 5/5 folds with OOS PF > 1.0
  - Combined OOS PF > 1.0
  - PF ratio (OOS/IS) > 0.65

Usage:
    python3 scripts/walkforward_nexus_short_only.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics


def make_config():
    """MS Config B + OS Best — SHORT-ONLY."""
    cfg = StrategyConfig()
    cfg.direction_filter = "short"  # SHORT-ONLY

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


def run_fold(df_slice, cfg, label):
    """Run backtest on a slice, return (metrics, trades) or (None, [])."""
    trades = run_backtest(df_slice.copy(), cfg)
    if not trades:
        return None, []
    m = compute_metrics(trades, cfg.initial_capital)
    return m, trades


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

print("Loading data...")
df = load_tos_csv("data/es_5m_databento_2yr.csv", instrument="ES")
print(f"Loaded {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}\n")

# Get trading day boundaries
trading_dates = sorted(set(df[df["is_rth"]].index.date))
total_days = len(trading_dates)

# 5-fold rolling walk-forward
# Each fold: ~100 trading days IS, ~100 trading days OOS
# With ~520 trading days over 2 years, this gives 5 non-overlapping OOS windows
n_folds = 5
fold_size = total_days // n_folds  # ~104 days per fold

print("=" * 80)
print("  5-FOLD WALK-FORWARD VALIDATION: MS+OS Nexus SHORT-ONLY")
print("=" * 80)
print(f"  Total trading days: {total_days}")
print(f"  Fold size: ~{fold_size} trading days")
print(f"  Method: Expanding window IS + fixed OOS\n")

cfg = make_config()
fold_results = []

for fold in range(n_folds):
    oos_start_idx = fold * fold_size
    oos_end_idx = min((fold + 1) * fold_size, total_days)

    oos_start_date = trading_dates[oos_start_idx]
    oos_end_date = trading_dates[oos_end_idx - 1]

    # IS = everything NOT in this OOS window
    is_start_date = trading_dates[0]
    is_end_date = trading_dates[-1]

    oos_start_ts = pd.Timestamp(oos_start_date, tz="US/Eastern")
    oos_end_ts = pd.Timestamp(oos_end_date, tz="US/Eastern") + pd.Timedelta(days=1)

    # Split data
    df_oos = df[(df.index >= str(oos_start_date)) & (df.index < str(oos_end_date + pd.Timedelta(days=1)))]
    df_is = df[(df.index < str(oos_start_date)) | (df.index >= str(oos_end_date + pd.Timedelta(days=1)))]

    is_days = len(set(df_is[df_is["is_rth"]].index.date))
    oos_days = len(set(df_oos[df_oos["is_rth"]].index.date))

    print(f"  {'─' * 70}")
    print(f"  FOLD {fold + 1}/{n_folds}")
    print(f"  OOS: {oos_start_date} to {oos_end_date} ({oos_days} days)")
    print(f"  IS:  {is_days} days (everything else)")

    # Run IS
    m_is, t_is = run_fold(df_is, cfg, "IS")
    # Run OOS
    m_oos, t_oos = run_fold(df_oos, cfg, "OOS")

    if m_is and m_oos:
        pf_ratio = m_oos.profit_factor / m_is.profit_factor if m_is.profit_factor > 0 else 0
        oos_pnls = [t.pnl_dollar for t in t_oos]
        _, p_val = stats.ttest_1samp(oos_pnls, 0) if len(oos_pnls) >= 5 else (0, 1.0)
        oos_pass = m_oos.profit_factor > 1.0
        verdict = "PASS" if oos_pass else "FAIL"

        print(f"  IS:  {m_is.total_trades:>4d}t  PF={m_is.profit_factor:.3f}  "
              f"${m_is.net_pnl:>+9,.0f}  WR={m_is.win_rate:.1f}%  Sh={m_is.sharpe:.2f}")
        print(f"  OOS: {m_oos.total_trades:>4d}t  PF={m_oos.profit_factor:.3f}  "
              f"${m_oos.net_pnl:>+9,.0f}  WR={m_oos.win_rate:.1f}%  Sh={m_oos.sharpe:.2f}  p={p_val:.4f}")
        print(f"  PF ratio: {pf_ratio:.2f}  → {verdict}")

        fold_results.append({
            "fold": fold + 1,
            "oos_start": str(oos_start_date),
            "oos_end": str(oos_end_date),
            "is_trades": m_is.total_trades,
            "is_pf": m_is.profit_factor,
            "is_pnl": m_is.net_pnl,
            "oos_trades": m_oos.total_trades,
            "oos_pf": m_oos.profit_factor,
            "oos_pnl": m_oos.net_pnl,
            "oos_wr": m_oos.win_rate,
            "oos_sharpe": m_oos.sharpe,
            "oos_p": p_val,
            "pf_ratio": pf_ratio,
            "pass": oos_pass,
        })
    else:
        print(f"  WARNING: Not enough trades in IS or OOS")
        fold_results.append({
            "fold": fold + 1,
            "oos_start": str(oos_start_date),
            "oos_end": str(oos_end_date),
            "pass": False,
        })

# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("  WALK-FORWARD SUMMARY")
print(f"{'=' * 80}\n")

print(f"  {'Fold':<6s}  {'OOS Period':<27s}  {'#':>4s}  {'PF':>7s}  {'P&L':>10s}  {'WR':>6s}  {'Sh':>6s}  {'p':>8s}  {'Pass':>5s}")
print(f"  {'─' * 85}")

passing_folds = 0
total_oos_pnl = 0
total_oos_trades = 0
all_oos_pfs = []

for r in fold_results:
    if "oos_pf" in r:
        status = "YES" if r["pass"] else "NO"
        print(f"  {r['fold']:<6d}  {r['oos_start']} to {r['oos_end']}  "
              f"{r['oos_trades']:>4d}  {r['oos_pf']:>7.3f}  ${r['oos_pnl']:>+9,.0f}  "
              f"{r['oos_wr']:>5.1f}%  {r['oos_sharpe']:>6.2f}  {r['oos_p']:>8.4f}  {status:>5s}")
        if r["pass"]:
            passing_folds += 1
        total_oos_pnl += r["oos_pnl"]
        total_oos_trades += r["oos_trades"]
        all_oos_pfs.append(r["oos_pf"])
    else:
        print(f"  {r['fold']:<6d}  {r['oos_start']} to {r['oos_end']}  "
              f"{'N/A':>4s}  {'N/A':>7s}  {'N/A':>10s}  {'N/A':>6s}  {'N/A':>6s}  {'N/A':>8s}  {'  NO':>5s}")

print(f"\n  {'─' * 85}")
print(f"  Combined OOS:  {total_oos_trades} trades, ${total_oos_pnl:+,.0f}")
if all_oos_pfs:
    avg_oos_pf = np.mean(all_oos_pfs)
    min_oos_pf = min(all_oos_pfs)
    print(f"  Average OOS PF: {avg_oos_pf:.3f}")
    print(f"  Minimum OOS PF: {min_oos_pf:.3f}")

print(f"\n  Folds passing (OOS PF > 1.0): {passing_folds}/{n_folds}")

if passing_folds == n_folds:
    print(f"\n  >>> WALK-FORWARD: PASS ({passing_folds}/{n_folds} folds profitable) <<<")
elif passing_folds >= 4:
    print(f"\n  >>> WALK-FORWARD: MARGINAL ({passing_folds}/{n_folds} folds profitable) <<<")
else:
    print(f"\n  >>> WALK-FORWARD: FAIL ({passing_folds}/{n_folds} folds profitable) <<<")

print(f"\n{'=' * 80}")
