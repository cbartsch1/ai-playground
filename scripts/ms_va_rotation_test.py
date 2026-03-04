#!/usr/bin/env python3
"""Test VA Acceptance + Rotation concept with REAL VP-derived levels (68%).

Dalton's 80% Rule applied with actual volume-profile-derived Value Area:
- When price opens OUTSIDE prev day's VA and "accepts" back inside,
  expect rotation to the opposite VA edge.
- LONG: opened below VAL, crossed above VAL → target VAH
- SHORT: opened above VAH, crossed below VAH → target VAL

Tests 6 variants:
1. Basic (enter on acceptance cross)
2. + SMA 8/24 filter
3. + 3-bar acceptance (stay inside VA for 3+ bars)
4. + Deep acceptance (close 3+ pts inside VA)
5. Short-only
6. Long-only
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.indicators import compute_indicators
from backtester.session import SessionState, update_session
from backtester.position import Position, Trade
from backtester.metrics import compute_metrics

# ─── Load data ───
df = load_tos_csv('data/es_5m_databento_2yr.csv', instrument='ES')
print(f"Loaded {len(df)} bars")

cfg = StrategyConfig()
compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                   tema_trend=cfg.tema_trend, atr_len=cfg.atr_len, atr_avg_len=cfg.atr_avg_len)


def run_va_rotation(df, variant="basic"):
    """Run VA rotation backtest with specified variant.

    Variants:
        "basic"           — enter on first acceptance cross, no MA filter
        "sma"             — require SMA 8 > 24 for longs, < 24 for shorts
        "3bar"            — require 3+ consecutive bars inside VA before entry
        "deep"            — require close 3+ pts inside VA (not just barely crossing)
        "short_only"      — only take short rotations (entered from above)
        "long_only"       — only take long rotations (entered from below)
        "sma_deep"        — SMA filter + deep acceptance combined
        "3bar_sma"        — 3-bar acceptance + SMA filter combined
    """
    state = SessionState()
    pos = Position()
    trades = []
    prev_bar = None

    # VA rotation state (reset each session)
    opened_above_va = False
    opened_below_va = False
    accepted_into_va = False
    acceptance_direction = 0   # 1 = from below (long), -1 = from above (short)
    bars_inside_va = 0
    rotation_traded = False
    open_bar_count = 0         # count bars in RTH to skip the very first bar

    STOP_BUFFER = 5.0          # points beyond VA edge for stop
    DEEP_THRESHOLD = 3.0       # points inside VA for "deep" acceptance
    ACCEPTANCE_BARS = 3        # bars needed for "3bar" variant
    MIN_TARGET_PTS = 4.0       # minimum distance to target (skip tiny VA ranges)
    MAX_RISK = 25.0            # max stop distance

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)

        # ─── New RTH session: reset rotation state ───
        if bar.get("new_rth", False):
            opened_above_va = False
            opened_below_va = False
            accepted_into_va = False
            acceptance_direction = 0
            bars_inside_va = 0
            rotation_traded = False
            open_bar_count = 0

        if not bar.get("is_rth", False):
            prev_bar = bar
            continue

        open_bar_count += 1

        close = bar["close"]
        prev_vah = state.prev_vp_vah
        prev_val = state.prev_vp_val

        if math.isnan(prev_vah) or math.isnan(prev_val):
            prev_bar = bar
            continue

        va_range = prev_vah - prev_val
        if va_range < 4:
            prev_bar = bar
            continue

        # ─── Determine opening position (first 2 bars of RTH) ───
        if open_bar_count <= 2 and not opened_above_va and not opened_below_va:
            if close > prev_vah:
                opened_above_va = True
            elif close < prev_val:
                opened_below_va = True
            # If inside VA at open, no rotation setup today
            prev_bar = bar
            continue

        # ─── Detect acceptance into VA ───
        if not accepted_into_va and not rotation_traded:
            if opened_above_va and close < prev_vah and close > prev_val:
                # Came from above, now inside VA → SHORT setup
                acceptance_direction = -1
                accepted_into_va = True
                bars_inside_va = 1
            elif opened_below_va and close > prev_val and close < prev_vah:
                # Came from below, now inside VA → LONG setup
                acceptance_direction = 1
                accepted_into_va = True
                bars_inside_va = 1
        elif accepted_into_va and not rotation_traded:
            # Count consecutive bars inside VA
            if close > prev_val and close < prev_vah:
                bars_inside_va += 1
            else:
                # Price left VA → reset acceptance
                accepted_into_va = False
                bars_inside_va = 0
                acceptance_direction = 0

        # ─── Check entry conditions ───
        if (accepted_into_va and not rotation_traded and pos.is_flat
                and bar.get("is_rth", False) and state.ib_done):

            sma8 = bar.get("sma_8", float("nan"))
            sma24 = bar.get("sma_24", float("nan"))

            # Direction filter
            if variant == "short_only" and acceptance_direction == 1:
                prev_bar = bar
                continue
            if variant == "long_only" and acceptance_direction == -1:
                prev_bar = bar
                continue

            # SMA filter
            sma_ok = True
            if variant in ("sma", "sma_deep", "3bar_sma"):
                if math.isnan(sma8) or math.isnan(sma24):
                    sma_ok = False
                elif acceptance_direction == 1 and sma8 <= sma24:
                    sma_ok = False
                elif acceptance_direction == -1 and sma8 >= sma24:
                    sma_ok = False

            # 3-bar acceptance filter
            bars_ok = True
            if variant in ("3bar", "3bar_sma"):
                if bars_inside_va < ACCEPTANCE_BARS:
                    bars_ok = False

            # Deep acceptance filter
            deep_ok = True
            if variant in ("deep", "sma_deep"):
                if acceptance_direction == 1:
                    # Long: close must be DEEP_THRESHOLD pts above prev_val
                    if close - prev_val < DEEP_THRESHOLD:
                        deep_ok = False
                elif acceptance_direction == -1:
                    # Short: close must be DEEP_THRESHOLD pts below prev_vah
                    if prev_vah - close < DEEP_THRESHOLD:
                        deep_ok = False

            # All filters pass?
            if sma_ok and bars_ok and deep_ok:
                if acceptance_direction == -1:
                    # SHORT rotation to prev_val
                    stop = prev_vah + STOP_BUFFER
                    target = prev_val
                    distance = close - target
                    risk = stop - close
                    if distance >= MIN_TARGET_PTS and risk > 0 and risk <= MAX_RISK:
                        pos.enter(direction=-1, price=close, stop=stop, target=target,
                                  setup="VA_ROT_S", time=idx, slippage=cfg.slippage_pts)
                        rotation_traded = True

                elif acceptance_direction == 1:
                    # LONG rotation to prev_vah
                    stop = prev_val - STOP_BUFFER
                    target = prev_vah
                    distance = target - close
                    risk = close - stop
                    if distance >= MIN_TARGET_PTS and risk > 0 and risk <= MAX_RISK:
                        pos.enter(direction=1, price=close, stop=stop, target=target,
                                  setup="VA_ROT_L", time=idx, slippage=cfg.slippage_pts)
                        rotation_traded = True

        # ─── Check exits ───
        if not pos.is_flat:
            et_time = bar.get("et_time", 0)
            if et_time >= 1555:
                trade = pos.flatten(bar)
                if trade:
                    trade.pnl_dollar = (trade.pnl_pts * 50) - 5
                    trades.append(trade)
            else:
                trade = pos.check_exit(bar, pessimistic=True)
                if trade:
                    trade.pnl_dollar = (trade.pnl_pts * 50) - 5
                    trades.append(trade)

        prev_bar = bar

    # Close any open position
    if not pos.is_flat and prev_bar:
        trade = pos.flatten(prev_bar)
        if trade:
            trade.pnl_dollar = (trade.pnl_pts * 50) - 5
            trades.append(trade)

    return trades


def print_results(trades, label, df):
    """Print formatted results for a variant."""
    if not trades:
        print(f"  {label:<24s}  NO TRADES")
        return

    m = compute_metrics(trades)
    pnls = [t.pnl_dollar for t in trades]
    _, p_val = stats.ttest_1samp(pnls, 0) if len(trades) >= 5 else (0, 1.0)

    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]

    print(f"\n  --- {label} ---")
    print(f"  ALL:     {m.total_trades:>4d}t  WR {m.win_rate:.1f}%  PF {m.profit_factor:.3f}  "
          f"P&L ${m.net_pnl:>+9,.0f}  DD ${m.max_drawdown:>7,.0f}  Sh {m.sharpe:.2f}  "
          f"Avg ${m.avg_trade:>+6,.0f}  p={p_val:.4f}")
    print(f"  T/Day: {m.trades_per_day:.2f}  "
          f"AvgWin ${m.avg_win:>+6,.0f}  AvgLoss ${m.avg_loss:>+6,.0f}  "
          f"WinStrk {m.longest_win_streak}  LoseStrk {m.longest_lose_streak}")

    if longs:
        lm = compute_metrics(longs)
        l_pnls = [t.pnl_dollar for t in longs]
        _, lp = stats.ttest_1samp(l_pnls, 0) if len(longs) >= 5 else (0, 1.0)
        print(f"  LONGS:   {lm.total_trades:>4d}t  WR {lm.win_rate:.1f}%  PF {lm.profit_factor:.3f}  "
              f"P&L ${lm.net_pnl:>+9,.0f}  DD ${lm.max_drawdown:>7,.0f}  p={lp:.4f}")
    if shorts:
        sm = compute_metrics(shorts)
        s_pnls = [t.pnl_dollar for t in shorts]
        _, sp = stats.ttest_1samp(s_pnls, 0) if len(shorts) >= 5 else (0, 1.0)
        print(f"  SHORTS:  {sm.total_trades:>4d}t  WR {sm.win_rate:.1f}%  PF {sm.profit_factor:.3f}  "
              f"P&L ${sm.net_pnl:>+9,.0f}  DD ${sm.max_drawdown:>7,.0f}  p={sp:.4f}")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t.exit_reason
        reasons.setdefault(r, {"count": 0, "pnl": 0})
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t.pnl_dollar
    print(f"  Exits:")
    for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        wr_r = len([t for t in trades if t.exit_reason == r and t.pnl_dollar > 0]) / d["count"] * 100
        print(f"    {r:<12s}  {d['count']:>4d}  ${d['pnl']:>+9,.0f}  WR {wr_r:.0f}%")

    # Walk-forward split
    split_idx = df.index.get_indexer(["2025-02-14"], method="nearest")[0]
    y1_trades = [t for t in trades if t.entry_time < df.index[split_idx]]
    y2_trades = [t for t in trades if t.entry_time >= df.index[split_idx]]
    if y1_trades and y2_trades:
        m1 = compute_metrics(y1_trades)
        m2 = compute_metrics(y2_trades)
        _, p2 = stats.ttest_1samp([t.pnl_dollar for t in y2_trades], 0) if len(y2_trades) >= 5 else (0, 1.0)
        ratio = m2.profit_factor / m1.profit_factor if m1.profit_factor > 0 else 0
        verdict = "PASS" if ratio > 0.7 else "MARGINAL" if ratio > 0.5 else "FAIL"
        print(f"  Walk-Forward:")
        print(f"    Y1: {m1.total_trades:>3d}t PF={m1.profit_factor:.3f} WR={m1.win_rate:.1f}% ${m1.net_pnl:>+8,.0f}")
        print(f"    Y2: {m2.total_trades:>3d}t PF={m2.profit_factor:.3f} WR={m2.win_rate:.1f}% ${m2.net_pnl:>+8,.0f}  p={p2:.4f}")
        print(f"    Ratio: {ratio:.2f} {verdict}")


# ─── Run all variants ───
print(f"\n{'='*110}")
print(f"  VA ACCEPTANCE + ROTATION — REAL VP LEVELS (68% Value Area)")
print(f"  Concept: Price opens outside prev VA, re-enters (accepts), rotates to opposite edge")
print(f"  Stop: VA edge crossed + 5pt buffer  |  Target: opposite VA edge")
print(f"{'='*110}")

variants = [
    ("basic",     "1. Basic (immediate entry)"),
    ("sma",       "2. + SMA 8/24 filter"),
    ("3bar",      "3. + 3-bar acceptance"),
    ("deep",      "4. + Deep acceptance (3pt)"),
    ("3bar_sma",  "5. + 3-bar + SMA filter"),
    ("sma_deep",  "6. + SMA + Deep combined"),
    ("short_only","7. Short-only"),
    ("long_only", "8. Long-only"),
]

all_results = {}
for variant_key, label in variants:
    trades = run_va_rotation(df, variant=variant_key)
    all_results[variant_key] = trades
    print_results(trades, label, df)

# ─── Summary comparison table ───
print(f"\n{'='*110}")
print(f"  SUMMARY COMPARISON")
print(f"{'='*110}")
print(f"  {'Variant':<28s} {'Trades':>6s} {'WR':>6s} {'PF':>7s} {'P&L':>10s} {'DD':>8s} {'Sharpe':>7s} {'p-val':>7s} {'Avg$':>7s}")
print(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*7} {'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

for variant_key, label in variants:
    trades = all_results[variant_key]
    if not trades:
        print(f"  {label:<28s}     NO TRADES")
        continue
    m = compute_metrics(trades)
    pnls = [t.pnl_dollar for t in trades]
    _, p_val = stats.ttest_1samp(pnls, 0) if len(trades) >= 5 else (0, 1.0)
    marker = " ***" if p_val < 0.05 and m.profit_factor > 1.2 else " **" if p_val < 0.10 else ""
    print(f"  {label:<28s} {m.total_trades:>6d} {m.win_rate:>5.1f}% {m.profit_factor:>7.3f} "
          f"${m.net_pnl:>+9,.0f} ${m.max_drawdown:>7,.0f} {m.sharpe:>7.2f} {p_val:>7.4f} "
          f"${m.avg_trade:>+6,.0f}{marker}")

print(f"\n  *** = p<0.05 & PF>1.2  |  ** = p<0.10")
print(f"{'='*110}")
