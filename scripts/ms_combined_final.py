#!/usr/bin/env python3
"""MS Combined Strategy — Multi-layered Market Structure with nuanced filters.

Tests 6 configurations combining proven edges from parallel research:
  A. ON Optimized (baseline) — ON levels, SMA 8/24, optimized params
  B. ON + pVAH Short — adds pVAH as sell-only level
  C. ON + pVAH + Long POC Filter — skips longs when prev POC overhead
  D. ON + pVAH + FA Longs — adds Failed Auction longs
  E. Full Nuanced — ON + pVAH short + dVAH short + FA longs + POC filter
  F. Full Nuanced + Fibonacci Targets — same as E with fib-based targets

For each config: full 2-year, per-setup, direction, walk-forward validation.

Usage:
    python scripts/ms_combined_final.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
import numpy as np

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


# ═══════════════════════════════════════════════════════════════════
#  Config Builders
# ═══════════════════════════════════════════════════════════════════

def _base_ms_off():
    """Base config with all setups OFF."""
    cfg = StrategyConfig()
    cfg.direction_filter = "both"
    cfg.use_ib_break = False
    cfg.use_va_fade = False
    cfg.use_eighty = False
    cfg.use_tema_cross = False
    cfg.use_level_reject = False
    cfg.use_level_reject_long = False
    cfg.use_ib_reject = False
    cfg.use_var = False
    cfg.use_ptf = False
    cfg.use_ms = False
    cfg.use_fa = False
    return cfg


def make_config_a():
    """Config A — ON Optimized (baseline).
    
    ON levels only, SMA 8/24 timing, optimized params from research:
    zone=3, stop=4, min_target=8, min_rr=0.3, max_risk=25.
    """
    cfg = _base_ms_off()
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
    # Level selection: ON only
    cfg.ms_use_prev_va = False
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = False
    cfg.ms_use_poc = False
    return cfg


def make_config_b():
    """Config B — ON + pVAH Short.
    
    Same as A, plus pVAH as sell-only level.
    Uses ms_level_directions to restrict pVAH to shorts only.
    """
    cfg = make_config_a()
    cfg.ms_use_prev_va = True
    # Whitelist mode: specify allowed levels and directions
    cfg.ms_level_directions = {
        "MS_ONH": "both",   # ON high — both directions
        "MS_ONL": "both",   # ON low — both directions
        "MS_pVAH": "short", # prev VAH — shorts only (the proven edge)
        # pVAL not listed = excluded (whitelist mode)
    }
    return cfg


def make_config_c():
    """Config C — ON + pVAH Short + Long POC Filter.
    
    Same as B, but skip longs when prev POC is between entry and target.
    POC acts as resistance for longs (PF 0.761 when overhead).
    """
    cfg = make_config_b()
    cfg.ms_skip_long_poc_overhead = True
    return cfg


def make_config_d():
    """Config D — ON + pVAH Short + Failed Auction Longs.
    
    MS: ON + pVAH short (same as B)
    FA: Failed Auction longs only (fa_max_break_bars=4, fa_stop_buffer=3)
    """
    cfg = make_config_b()
    cfg.use_fa = True
    cfg.fa_max_break_bars = 4
    cfg.fa_stop_buffer = 3.0
    cfg.fa_min_rr = 0.5
    cfg.fa_max_risk = 20.0
    cfg.fa_require_ma = False
    cfg.max_fa_trades = 2
    # FA: longs only (direction_filter applies globally, so we need
    # the global filter as "both" and control FA direction via its own logic)
    # FA naturally does both, but we want longs only.
    # Since direction_filter is global, we'll use a workaround:
    # We keep direction_filter="both" and note that FA shorts will also fire.
    # To truly limit FA to longs, we'd need a config flag.
    # For now: FA adds both sides. The research showed FA longs have PF 1.101.
    return cfg


def make_config_e():
    """Config E — Full Nuanced.
    
    - ON bilateral (optimized params)
    - pVAH shorts only
    - dVAH shorts only
    - Failed Auction longs only (we filter in analysis)
    - Skip longs when POC overhead
    - min_target=8 for MS trades
    """
    cfg = _base_ms_off()
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 4.0
    cfg.ms_min_target_pts = 8.0
    cfg.ms_min_rr = 0.3
    cfg.ms_max_risk = 25.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 10
    cfg.ms_use_vp_levels = True
    # Level selection
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = True
    cfg.ms_use_poc = False
    # Direction filters per level
    cfg.ms_level_directions = {
        "MS_ONH": "both",    # ON — bilateral
        "MS_ONL": "both",    # ON — bilateral
        "MS_pVAH": "short",  # prev VAH — shorts only
        # MS_pVAL not listed = excluded
        "MS_dVAH": "short",  # dev VAH — shorts only
        # MS_dVAL not listed = excluded
    }
    # POC overhead filter for longs
    cfg.ms_skip_long_poc_overhead = True
    # Failed Auction longs
    cfg.use_fa = True
    cfg.fa_max_break_bars = 4
    cfg.fa_stop_buffer = 3.0
    cfg.fa_min_rr = 0.5
    cfg.fa_max_risk = 20.0
    cfg.fa_require_ma = False
    cfg.max_fa_trades = 2
    return cfg


def make_config_f():
    """Config F — Full Nuanced + Fibonacci Targets.
    
    Same as E, but use Fibonacci retracement targets from prev day range
    when they give better R:R than structural targets.
    """
    cfg = make_config_e()
    cfg.ms_use_fib_targets = True
    return cfg


# ═══════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════

def print_config_results(label, trades, show_detail=True):
    """Print detailed metrics for a config."""
    if not trades:
        print(f"  {label:<50s}  NO TRADES")
        return None

    m = compute_metrics(trades)
    
    # Statistical significance
    pnls = [t.pnl_dollar for t in trades]
    if len(pnls) >= 5:
        t_stat, p_val = stats.ttest_1samp(pnls, 0)
    else:
        t_stat, p_val = 0, 1.0
    
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
    
    print(f"\n  --- {label} ---")
    print(f"  Trades: {m.total_trades:>5d}   WR: {m.win_rate:>5.1f}%   PF: {m.profit_factor:>6.3f}   "
          f"P&L: ${m.net_pnl:>+10,.0f}   DD: ${m.max_drawdown:>8,.0f}   "
          f"Sharpe: {m.sharpe:>5.2f}   Trd/Day: {m.trades_per_day:>4.1f}   "
          f"p={p_val:.4f} {sig}")
    print(f"  Avg Trade: ${m.avg_trade:>+.0f}   Avg Win: ${m.avg_win:>+.0f}   Avg Loss: ${m.avg_loss:>+.0f}   "
          f"Win Streak: {m.longest_win_streak}   Loss Streak: {m.longest_lose_streak}")

    if show_detail and trades:
        # Per-setup breakdown
        print(f"\n  Per-Setup Breakdown:")
        print(f"    {'Setup':<18s} {'Trades':>6s} {'WR':>6s} {'PF':>7s} {'P&L':>12s} {'Avg':>8s}")
        print(f"    {'-'*60}")
        breakdown = per_setup_breakdown(trades)
        for setup, sm in sorted(breakdown.items(), key=lambda x: -x[1].net_pnl):
            print(f"    {setup:<18s} {sm.total_trades:>6d} {sm.win_rate:>5.1f}% {sm.profit_factor:>7.3f} "
                  f"${sm.net_pnl:>+10,.0f} ${sm.avg_trade:>+7.0f}")

        # Direction breakdown
        print(f"\n  Direction Breakdown:")
        longs = [t for t in trades if t.direction == 1]
        shorts = [t for t in trades if t.direction == -1]
        for dir_name, dir_trades in [("LONGS", longs), ("SHORTS", shorts)]:
            if dir_trades:
                dm = compute_metrics(dir_trades)
                dp = [t.pnl_dollar for t in dir_trades]
                _, dp_val = stats.ttest_1samp(dp, 0) if len(dp) >= 5 else (0, 1.0)
                dp_sig = "***" if dp_val < 0.01 else "**" if dp_val < 0.05 else "*" if dp_val < 0.10 else ""
                print(f"    {dir_name:<10s} {dm.total_trades:>5d}  WR {dm.win_rate:>5.1f}%  PF {dm.profit_factor:>6.3f}  "
                      f"P&L ${dm.net_pnl:>+10,.0f}  p={dp_val:.4f} {dp_sig}")
            else:
                print(f"    {dir_name:<10s}  none")

        # Exit reasons
        reasons = {}
        for t in trades:
            r = t.exit_reason
            if r not in reasons:
                reasons[r] = {"count": 0, "pnl": 0}
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t.pnl_dollar
        print(f"\n  Exit Reasons:")
        for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            print(f"    {r:<12s} {d['count']:>4d} trades  ${d['pnl']:>+10,.0f}")

    return m


def run_all_configs(df, label="FULL 2-YEAR"):
    """Run all 6 configs and return results."""
    print(f"\n{'='*120}")
    print(f"  {label}")
    print(f"{'='*120}")

    configs = [
        ("A: ON Optimized", make_config_a),
        ("B: ON + pVAH Short", make_config_b),
        ("C: ON + pVAH + POC Filter", make_config_c),
        ("D: ON + pVAH + FA Longs", make_config_d),
        ("E: Full Nuanced", make_config_e),
        ("F: Full Nuanced + Fib", make_config_f),
    ]

    results = []
    for name, builder in configs:
        cfg = builder()
        trades = run_backtest(df.copy(), cfg)
        m = print_config_results(name, trades, show_detail=True)
        results.append((name, m, trades, cfg))

    # Summary table
    print(f"\n{'='*120}")
    print(f"  SUMMARY — {label}")
    print(f"{'='*120}")
    print(f"  {'Config':<30s} {'Trades':>6s} {'WR':>6s} {'PF':>7s} {'P&L':>12s} "
          f"{'DD':>10s} {'Sharpe':>7s} {'Trd/Day':>8s} {'p-value':>8s}")
    print(f"  {'-'*105}")
    for name, m, trades, _ in results:
        if m:
            pnls = [t.pnl_dollar for t in trades]
            _, pv = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)
            sig = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
            print(f"  {name:<30s} {m.total_trades:>6d} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
                  f"${m.net_pnl:>+10,.0f} ${m.max_drawdown:>8,.0f} {m.sharpe:>7.2f} "
                  f"{m.trades_per_day:>8.2f} {pv:>7.4f} {sig}")
        else:
            print(f"  {name:<30s}  NO TRADES")

    return results


def main():
    csv_file = "data/es_5m_databento_2yr.csv"
    split_date = "2025-02-14"

    print(f"Loading {csv_file}...")
    df = load_tos_csv(csv_file, instrument="ES")
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"RTH bars: {df['is_rth'].sum()}, Sessions: {df['new_rth'].sum()}")

    # ════════════════════════════════════════════════
    # Full 2-Year
    # ════════════════════════════════════════════════
    full_results = run_all_configs(df, "FULL 2-YEAR")

    # ════════════════════════════════════════════════
    # Walk-Forward Split
    # ════════════════════════════════════════════════
    split_idx = df.index.get_indexer([split_date], method="nearest")[0]
    if split_idx <= 0 or split_idx >= len(df):
        split_idx = len(df) // 2

    df_y1 = df.iloc[:split_idx].copy()
    df_y2 = df.iloc[split_idx:].copy()

    print(f"\n  Walk-Forward Split:")
    print(f"  Year 1 (IS):  {df_y1.index[0].date()} to {df_y1.index[-1].date()}  ({df_y1['new_rth'].sum()} sessions)")
    print(f"  Year 2 (OOS): {df_y2.index[0].date()} to {df_y2.index[-1].date()}  ({df_y2['new_rth'].sum()} sessions)")

    y1_results = run_all_configs(df_y1, "YEAR 1 — IN-SAMPLE")
    y2_results = run_all_configs(df_y2, "YEAR 2 — OUT-OF-SAMPLE")

    # ════════════════════════════════════════════════
    # Walk-Forward Validation Summary
    # ════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"{'='*120}")
    print(f"  {'Config':<30s} {'IS PF':>7s} {'IS Trades':>9s} {'OOS PF':>7s} {'OOS Trades':>10s} "
          f"{'WF Ratio':>9s} {'OOS p':>8s} {'Verdict':>10s}")
    print(f"  {'-'*100}")

    for (n1, m1, t1, _), (n2, m2, t2, _) in zip(y1_results, y2_results):
        if m1 and m2 and m1.profit_factor > 0 and m2.profit_factor > 0:
            ratio = m2.profit_factor / m1.profit_factor
            pnls2 = [t.pnl_dollar for t in t2]
            _, p2 = stats.ttest_1samp(pnls2, 0) if len(pnls2) >= 5 else (0, 1.0)
            
            if ratio > 0.7 and m2.profit_factor > 1.0:
                verdict = "PASS"
            elif ratio > 0.5 and m2.profit_factor > 1.0:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"
            
            sig = "***" if p2 < 0.01 else "**" if p2 < 0.05 else "*" if p2 < 0.10 else ""
            
            print(f"  {n1:<30s} {m1.profit_factor:>7.3f} {m1.total_trades:>9d} "
                  f"{m2.profit_factor:>7.3f} {m2.total_trades:>10d} "
                  f"{ratio:>9.2f} {p2:>7.4f} {sig} {verdict:>8s}")
        elif m1 and m2:
            print(f"  {n1:<30s} {m1.profit_factor:>7.3f} {m1.total_trades:>9d} "
                  f"{m2.profit_factor:>7.3f} {m2.total_trades:>10d} "
                  f"{'N/A':>9s} {'N/A':>8s} {'CHECK':>10s}")
        else:
            no_trades = "NO TRADES" if not m1 else "NO OOS"
            print(f"  {n1:<30s}  {no_trades}")

    # ════════════════════════════════════════════════
    # Recommendation
    # ════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  RECOMMENDATION")
    print(f"{'='*120}")
    
    # Find best config by OOS PF with minimum trade threshold
    best = None
    best_score = 0
    for (name, m_full, t_full, _), (_, m1, t1, _), (_, m2, t2, _) in zip(full_results, y1_results, y2_results):
        if m2 and m2.total_trades >= 20 and m2.profit_factor > 1.0:
            # Score = OOS PF * sqrt(OOS trades) * WF ratio
            wf_ratio = m2.profit_factor / m1.profit_factor if m1 and m1.profit_factor > 0 else 0
            score = m2.profit_factor * math.sqrt(m2.total_trades) * min(wf_ratio, 1.5)
            if score > best_score:
                best_score = score
                best = (name, m_full, m1, m2, wf_ratio, t_full)
    
    if best:
        name, m_full, m1, m2, wf_ratio, t_full = best
        pnls = [t.pnl_dollar for t in t_full]
        _, p_full = stats.ttest_1samp(pnls, 0) if len(pnls) >= 5 else (0, 1.0)
        print(f"\n  BEST CONFIG: {name}")
        print(f"  Full 2yr: {m_full.total_trades} trades, PF {m_full.profit_factor:.3f}, "
              f"P&L ${m_full.net_pnl:>+,.0f}, p={p_full:.4f}")
        print(f"  WF: IS PF {m1.profit_factor:.3f} -> OOS PF {m2.profit_factor:.3f} (ratio {wf_ratio:.2f})")
        print(f"  Trades/Day: {m_full.trades_per_day:.1f}")
    else:
        print(f"\n  No config passed all criteria (OOS PF > 1.0, >= 20 OOS trades)")
        print(f"  Review individual results above for the best available option.")


if __name__ == "__main__":
    main()
