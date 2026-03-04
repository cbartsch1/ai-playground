#!/usr/bin/env python3
"""MS Direction Sweep — per-level direction-filtered combinations.

Critical insight from prior analysis: each level has a DIFFERENT directional edge.
  WINNERS: ONH shorts (+$7,485), ONL longs (+$6,380), pVAH shorts (+$1,935),
           pPOC longs (+$2,328), dPOC longs (+$4,032)
  LOSERS:  pPOC shorts (-$9,555), dVAL longs (-$9,040), dPOC shorts (-$3,680),
           pVAL longs (-$2,052)

The old approach used GLOBAL direction filters. This script uses the new
ms_level_directions config for PER-LEVEL direction filtering — a whitelist
dict that specifies which direction each level is allowed to trade.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, per_setup_breakdown


def _ms_base():
    """Base MS config — all levels OFF, SMA 8/24, per-level directions empty."""
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
    cfg.use_ms = True
    cfg.ms_zone_pts = 3.0
    cfg.ms_stop_buffer = 5.0
    cfg.ms_min_target_pts = 4.0
    cfg.ms_min_rr = 0.5
    cfg.ms_max_risk = 15.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    # Enable all level groups (direction filtering is via ms_level_directions)
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False   # IB not in any test combo
    cfg.ms_use_dev_va = True
    cfg.ms_use_poc = True
    cfg.ms_level_directions = {}   # empty = allow all (backward compatible)
    return cfg


# ── Direction maps for each combo ────────────────────────────────────────

# Winners only: ONH short, ONL long, pVAH short, pPOC long, dPOC long
WINNERS_ONLY = {
    "MS_ONH": "short",
    "MS_ONL": "long",
    "MS_pVAH": "short",
    "MS_pPOC": "long",    # long only (shorts lose -$9,555)
    "MS_dPOC": "long",    # long only (shorts lose -$3,680)
}

# ON Bilateral + pVAH short
ON_BILATERAL_PVAH = {
    "MS_ONH": "both",
    "MS_ONL": "both",
    "MS_pVAH": "short",
}

# ON Bilateral + Winners (ON both + pVAH short + pPOC long + dPOC long)
ON_BILATERAL_WINNERS = {
    "MS_ONH": "both",
    "MS_ONL": "both",
    "MS_pVAH": "short",
    "MS_pPOC": "long",
    "MS_dPOC": "long",
}

# ON + pVA shorts only (ON both + pVAH short, no pVAL)
ON_PVA_SHORTS = {
    "MS_ONH": "both",
    "MS_ONL": "both",
    "MS_pVAH": "short",
    # pVAL NOT included (loser)
}

# All Winners + dVAH shorts (test adding dVAH)
WINNERS_PLUS_DVAH = {
    "MS_ONH": "short",
    "MS_ONL": "long",
    "MS_pVAH": "short",
    "MS_pPOC": "long",
    "MS_dPOC": "long",
    "MS_dVAH": "short",   # test: does adding dVAH shorts help?
}

# ON Bilateral only (baseline comparison)
ON_BILATERAL = {
    "MS_ONH": "both",
    "MS_ONL": "both",
}


def print_result(label, trades, df_len):
    """Print single-line result with direction breakdown and p-value."""
    if not trades:
        print(f"  {label:<55s}  NO TRADES")
        return None, None
    m = compute_metrics(trades)
    pnls = [t.pnl_dollar for t in trades]
    _, p_val = stats.ttest_1samp(pnls, 0) if len(trades) >= 5 else (0, 1.0)
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

    # Direction breakdown
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    l_pnl = sum(t.pnl_dollar for t in longs)
    s_pnl = sum(t.pnl_dollar for t in shorts)

    print(f"  {label:<55s}  {m.total_trades:>5d}t  WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
          f"P&L ${m.net_pnl:>+9,.0f}  DD ${m.max_drawdown:>7,.0f}  Sh {m.sharpe:>5.2f}  "
          f"p={p_val:.3f}{sig}  T/D {m.trades_per_day:.1f}  "
          f"L${l_pnl:>+8,.0f} S${s_pnl:>+8,.0f}")
    return m, p_val


def print_setup_breakdown(trades):
    """Print per-setup metrics breakdown."""
    if not trades:
        return
    breakdown = per_setup_breakdown(trades)
    for setup, m in sorted(breakdown.items()):
        sub_trades = [t for t in trades if t.setup == setup]
        longs = [t for t in sub_trades if t.direction == 1]
        shorts = [t for t in sub_trades if t.direction == -1]
        l_pnl = sum(t.pnl_dollar for t in longs)
        s_pnl = sum(t.pnl_dollar for t in shorts)
        print(f"      {setup:<20s}  {m.total_trades:>4d}t  WR {m.win_rate:>5.1f}%  PF {m.profit_factor:>6.3f}  "
              f"P&L ${m.net_pnl:>+8,.0f}  L({len(longs)}) ${l_pnl:>+7,.0f}  S({len(shorts)}) ${s_pnl:>+7,.0f}")


# ── Load data ────────────────────────────────────────────────────────────

df = load_tos_csv('data/es_5m_databento_2yr.csv', instrument='ES')
print(f"Loaded {len(df)} bars  ({df.index[0].date()} to {df.index[-1].date()})\n")

# Split for walk-forward (Year 1 = in-sample, Year 2 = out-of-sample)
split_idx = df.index.get_indexer(["2025-02-14"], method="nearest")[0]
df_y1 = df.iloc[:split_idx].copy()
df_y2 = df.iloc[split_idx:].copy()

# ── SECTION 1: Direction-filtered combos (full 2-year) ───────────────────

print("=" * 170)
print("  DIRECTION-FILTERED LEVEL COMBINATIONS — FULL 2-YEAR")
print("=" * 170)

configs = []

# 1. Winners Only
cfg = _ms_base()
cfg.ms_level_directions = WINNERS_ONLY.copy()
configs.append(("1. Winners Only (ONH-s ONL-l pVAH-s pPOC-l dPOC-l)", cfg))

# 2. ON Bilateral + pVAH short
cfg = _ms_base()
cfg.ms_level_directions = ON_BILATERAL_PVAH.copy()
configs.append(("2. ON Bilateral + pVAH short", cfg))

# 3. ON Bilateral + Winners
cfg = _ms_base()
cfg.ms_level_directions = ON_BILATERAL_WINNERS.copy()
configs.append(("3. ON Bilateral + Winners (pVAH-s pPOC-l dPOC-l)", cfg))

# 4. ON + pVA shorts only
cfg = _ms_base()
cfg.ms_level_directions = ON_PVA_SHORTS.copy()
configs.append(("4. ON + pVA shorts only", cfg))

# 5. All Winners + dVAH shorts
cfg = _ms_base()
cfg.ms_level_directions = WINNERS_PLUS_DVAH.copy()
configs.append(("5. All Winners + dVAH shorts", cfg))

# 6. All Winners stop=8
cfg = _ms_base()
cfg.ms_level_directions = WINNERS_ONLY.copy()
cfg.ms_stop_buffer = 8.0
configs.append(("6. All Winners stop=8", cfg))

# 7. All Winners stop=10
cfg = _ms_base()
cfg.ms_level_directions = WINNERS_ONLY.copy()
cfg.ms_stop_buffer = 10.0
configs.append(("7. All Winners stop=10", cfg))

# 8. All Winners zone=2
cfg = _ms_base()
cfg.ms_level_directions = WINNERS_ONLY.copy()
cfg.ms_zone_pts = 2.0
configs.append(("8. All Winners zone=2", cfg))

# 9. All Winners lag=1
cfg = _ms_base()
cfg.ms_level_directions = WINNERS_ONLY.copy()
cfg.ms_ma_confirm_bars = 1
configs.append(("9. All Winners lag=1", cfg))

# 10. All Winners max=4
cfg = _ms_base()
cfg.ms_level_directions = WINNERS_ONLY.copy()
cfg.max_ms_trades = 4
configs.append(("10. All Winners max=4", cfg))

# 11. All Winners rr=0.8
cfg = _ms_base()
cfg.ms_level_directions = WINNERS_ONLY.copy()
cfg.ms_min_rr = 0.8
configs.append(("11. All Winners rr=0.8", cfg))

# 12. All Winners minTgt=6
cfg = _ms_base()
cfg.ms_level_directions = WINNERS_ONLY.copy()
cfg.ms_min_target_pts = 6.0
configs.append(("12. All Winners minTgt=6", cfg))

# 13. ON Bilateral only (baseline comparison)
cfg = _ms_base()
cfg.ms_level_directions = ON_BILATERAL.copy()
configs.append(("13. ON Bilateral only (baseline)", cfg))

# Run all configs
full_results = []
for label, cfg in configs:
    trades = run_backtest(df.copy(), cfg)
    m, pv = print_result(label, trades, len(df))
    full_results.append((label, trades, m, pv, cfg))

# ── SECTION 2: Per-setup breakdown for top candidates ────────────────────

print(f"\n{'=' * 170}")
print("  PER-SETUP BREAKDOWN — TOP CANDIDATES")
print(f"{'=' * 170}")

# Show breakdown for the first 5 configs (the main directional combos)
for label, trades, m, pv, cfg in full_results[:5]:
    if trades:
        print(f"\n  >> {label}")
        print_setup_breakdown(trades)

# ── SECTION 3: Direction breakdown for top candidates ────────────────────

print(f"\n{'=' * 170}")
print("  DIRECTION BREAKDOWN — LONG vs SHORT")
print(f"{'=' * 170}")

for label, trades, m, pv, cfg in full_results[:5]:
    if not trades:
        continue
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    print(f"\n  >> {label}")
    if longs:
        ml = compute_metrics(longs)
        _, pl = stats.ttest_1samp([t.pnl_dollar for t in longs], 0) if len(longs) >= 5 else (0, 1.0)
        print(f"      LONG:   {ml.total_trades:>4d}t  WR {ml.win_rate:>5.1f}%  PF {ml.profit_factor:>6.3f}  P&L ${ml.net_pnl:>+8,.0f}  p={pl:.3f}")
    if shorts:
        ms = compute_metrics(shorts)
        _, ps = stats.ttest_1samp([t.pnl_dollar for t in shorts], 0) if len(shorts) >= 5 else (0, 1.0)
        print(f"      SHORT:  {ms.total_trades:>4d}t  WR {ms.win_rate:>5.1f}%  PF {ms.profit_factor:>6.3f}  P&L ${ms.net_pnl:>+8,.0f}  p={ps:.3f}")

# ── SECTION 4: Walk-forward validation ───────────────────────────────────

print(f"\n{'=' * 170}")
print("  WALK-FORWARD VALIDATION — TOP CANDIDATES")
print(f"{'=' * 170}")
print(f"  Year 1 (in-sample):   {df_y1.index[0].date()} to {df_y1.index[-1].date()}")
print(f"  Year 2 (out-of-sample): {df_y2.index[0].date()} to {df_y2.index[-1].date()}\n")

# Select top candidates based on results (pick combos with best PF and significance)
# Run all 13 in walk-forward to let the data speak
wf_results = []
for label, trades, m, pv, cfg in full_results:
    if not trades or m is None:
        continue
    t1 = run_backtest(df_y1.copy(), cfg)
    t2 = run_backtest(df_y2.copy(), cfg)
    if t1 and t2:
        m1 = compute_metrics(t1)
        m2 = compute_metrics(t2)
        _, p1 = stats.ttest_1samp([t.pnl_dollar for t in t1], 0) if len(t1) >= 5 else (0, 1.0)
        _, p2 = stats.ttest_1samp([t.pnl_dollar for t in t2], 0) if len(t2) >= 5 else (0, 1.0)
        ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
        verdict = "PASS" if ratio > 0.7 else "MARGINAL" if ratio > 0.5 else "FAIL"
        print(f"  {label:<55s}")
        print(f"      IS:  {m1.total_trades:>4d}t  WR {m1.win_rate:>5.1f}%  PF {m1.profit_factor:>6.3f}  "
              f"P&L ${m1.net_pnl:>+9,.0f}  DD ${m1.max_drawdown:>7,.0f}  p={p1:.3f}")
        print(f"      OOS: {m2.total_trades:>4d}t  WR {m2.win_rate:>5.1f}%  PF {m2.profit_factor:>6.3f}  "
              f"P&L ${m2.net_pnl:>+9,.0f}  DD ${m2.max_drawdown:>7,.0f}  p={p2:.3f}")
        print(f"      PF ratio (OOS/IS) = {ratio:.2f}  >>> {verdict}")
        wf_results.append((label, m1, m2, ratio, verdict, p2))
    elif t1:
        print(f"  {label:<55s}  IS: {len(t1)}t  OOS: NO TRADES")
    else:
        print(f"  {label:<55s}  IS: NO TRADES")

# ── SECTION 5: Summary ranking ──────────────────────────────────────────

print(f"\n{'=' * 170}")
print("  FINAL RANKING — SORTED BY OOS PROFIT FACTOR (walk-forward survivors)")
print(f"{'=' * 170}")
print(f"  {'Rank':<5s}  {'Config':<55s}  {'IS PF':>7s}  {'OOS PF':>7s}  {'OOS P&L':>10s}  {'OOS p':>7s}  {'Ratio':>6s}  {'Verdict':<8s}")
print(f"  {'─'*5}  {'─'*55}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*7}  {'─'*6}  {'─'*8}")

# Sort by OOS PF descending
wf_sorted = sorted(wf_results, key=lambda x: x[2].profit_factor, reverse=True)
for i, (label, m1, m2, ratio, verdict, p2) in enumerate(wf_sorted, 1):
    print(f"  {i:<5d}  {label:<55s}  {m1.profit_factor:>7.3f}  {m2.profit_factor:>7.3f}  "
          f"${m2.net_pnl:>+9,.0f}  {p2:>7.3f}  {ratio:>6.2f}  {verdict:<8s}")

print(f"\n  Done. {len(configs)} configurations tested.")
