#!/usr/bin/env python3
"""Walk-forward validation: v8 + ONH Rejection (wide IB + TEMA + zone=8).

5-fold sequential walk-forward + 50/50 year split.
Compares v8 baseline vs v8 + ONH wide combo.
"""

import sys, os, math
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.metrics import compute_metrics, per_setup_breakdown
from backtester.indicators import compute_indicators
from backtester.session import SessionState, update_session
from backtester.position import Position
from backtester.setups import ib_breakout


def make_v8():
    cfg = StrategyConfig()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = 30.0
    cfg.skip_friday = True
    cfg.blackout_start = 1200
    cfg.blackout_end = 1300
    cfg.use_va_fade = False
    return cfg


def run_v8_plus_onh(df, tema_required=True, zone=8.0, max_onh=6,
                     wide_only=True, trigger="any", target="ib_low",
                     stop_buffer=8.0):
    """Run v8 IB Breakout + ONH Rejection with configurable params."""
    cfg = make_v8()
    compute_indicators(df, tema_fast=cfg.tema_fast, tema_slow=cfg.tema_slow,
                       tema_trend=cfg.tema_trend, atr_len=cfg.atr_len,
                       atr_avg_len=cfg.atr_avg_len)

    state = SessionState()
    pos = Position()
    trades = []
    prev_bar = None
    prev_ps = 0
    onh_trades_today = 0
    last_date = None

    for idx, row in df.iterrows():
        bar = row.to_dict()
        bar["_time"] = idx
        update_session(state, bar, prev_bar, cfg)

        cur_date = idx.date()
        if cur_date != last_date:
            onh_trades_today = 0
            last_date = cur_date

        cur_ps = 0 if pos.is_flat else pos.direction
        if cur_ps == 0 and prev_ps != 0:
            state.bars_since_exit = 0
        elif cur_ps == 0:
            state.bars_since_exit += 1

        trade = pos.check_exit(bar, pessimistic=cfg.pessimistic_fills)
        if trade is not None:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

        et_time = bar.get("et_time", 0)
        if et_time >= cfg.flatten_time and not pos.is_flat:
            trade = pos.flatten(bar)
            if trade is not None:
                trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
                trades.append(trade)

        if pos.is_flat:
            signal = None
            in_blackout = (cfg.blackout_start > 0 and cfg.blackout_end > 0
                           and cfg.blackout_start <= et_time < cfg.blackout_end)
            is_friday = bar.get("weekday", -1) == 4

            if not (in_blackout or (cfg.skip_friday and is_friday)):
                signal = ib_breakout.check_signal(bar, prev_bar, state, cfg)

                # ONH Rejection — only when no IB signal
                if signal is None and state.ib_done and bar["is_trading_window"]:
                    if (state.bars_since_exit >= cfg.cooldown_bars
                            and onh_trades_today < max_onh):
                        onh = state.on_high
                        if not math.isnan(onh) and state.on_frozen:
                            # Wide IB filter
                            if wide_only and not state.is_wide_ib:
                                pass  # skip
                            elif bar["high"] >= onh - zone:
                                # TEMA filter
                                tema_ok = (not tema_required
                                           or bar.get("tema_bearish", False))
                                if tema_ok:
                                    # Trigger
                                    triggered = False
                                    if trigger == "any":
                                        triggered = True
                                    elif trigger == "failed_break":
                                        triggered = (bar["high"] > onh
                                                     and bar["close"] <= onh)
                                    elif trigger == "bearish_close":
                                        triggered = bar["close"] < bar["open"]
                                    elif trigger == "wick":
                                        mid = (bar["high"] + bar["low"]) / 2
                                        triggered = (bar["close"] < bar["open"]
                                                     and bar["close"] < mid)

                                    if triggered:
                                        stop = onh + stop_buffer
                                        if cfg.pct_stop_mode:
                                            max_s = (bar["close"]
                                                     * cfg.pct_stop_bps / 10000.0)
                                            stop = min(stop, bar["close"] + max_s)
                                        if stop <= bar["close"]:
                                            stop = bar["close"] + 2.0

                                        tgt = None
                                        if target == "ib_low":
                                            if (not math.isnan(state.ib_low)
                                                    and state.ib_low < bar["close"]):
                                                tgt = state.ib_low
                                        elif target.startswith("fixed_"):
                                            pts = float(target.split("_")[1])
                                            tgt = bar["close"] - pts

                                        if tgt is not None and tgt < bar["close"]:
                                            signal = {
                                                "direction": -1,
                                                "stop": stop,
                                                "target": tgt,
                                                "setup": "ONH_REJ",
                                            }
                                            onh_trades_today += 1

            if (signal is not None and cfg.direction_filter == "short"
                    and signal["direction"] == 1):
                signal = None

            if signal is not None:
                pos.enter(direction=signal["direction"], price=bar["close"],
                          stop=signal["stop"], target=signal["target"],
                          setup=signal["setup"], time=idx,
                          slippage=cfg.slippage_pts)

        prev_ps = 0 if pos.is_flat else pos.direction
        prev_bar = bar

    if not pos.is_flat and prev_bar:
        trade = pos.flatten(prev_bar)
        if trade:
            trade.pnl_dollar = (trade.pnl_pts * cfg.point_value) - (cfg.commission * 2)
            trades.append(trade)

    return trades


def run_v8_baseline(df):
    """Run plain v8 via the standard engine."""
    from backtester.engine import run_backtest
    cfg = make_v8()
    return run_backtest(df, cfg)


def hdr(title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/es_5m_databento_2yr.csv"
    print(f"Loading {csv}...")
    df = load_tos_csv(csv, instrument="ES")
    print(f"Loaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    n = len(df)
    sessions = df["new_rth"].sum()
    print(f"Sessions: {sessions}")

    # ────────────────────────────────────────────────
    # 5-Fold Walk-Forward
    # ────────────────────────────────────────────────
    hdr("5-FOLD WALK-FORWARD VALIDATION")

    n_folds = 5
    fold_size = n // n_folds

    print(f"\n  {'Fold':<8s}  {'OOS Range':<30s}  {'Variant':<12s}  "
          f"{'Trades':>6s}  {'WR':>6s}  {'PF':>7s}  {'P&L':>10s}  {'DD':>8s}  {'Sharpe':>7s}")
    print(f"  {'-'*105}")

    oos_trades_v8 = []
    oos_trades_onh = []

    for fold in range(n_folds):
        oos_start = fold * fold_size
        oos_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

        df_oos = df.iloc[oos_start:oos_end].copy()
        oos_range = f"{df_oos.index[0].date()} → {df_oos.index[-1].date()}"

        # v8 baseline OOS
        t_v8 = run_v8_baseline(df_oos)
        m_v8 = compute_metrics(t_v8)
        oos_trades_v8.extend(t_v8)

        # v8 + ONH wide OOS
        t_onh = run_v8_plus_onh(df_oos)
        m_onh = compute_metrics(t_onh)
        oos_trades_onh.extend(t_onh)

        print(f"  Fold {fold+1:<3d}  {oos_range:<30s}  {'v8':<12s}  "
              f"{m_v8.total_trades:>6d}  {m_v8.win_rate:>5.1f}%  {m_v8.profit_factor:>7.3f}  "
              f"${m_v8.net_pnl:>+9,.0f}  ${m_v8.max_drawdown:>7,.0f}  {m_v8.sharpe:>7.2f}")
        print(f"  {'':8s}  {'':30s}  {'v8+ONH':<12s}  "
              f"{m_onh.total_trades:>6d}  {m_onh.win_rate:>5.1f}%  {m_onh.profit_factor:>7.3f}  "
              f"${m_onh.net_pnl:>+9,.0f}  ${m_onh.max_drawdown:>7,.0f}  {m_onh.sharpe:>7.2f}")

    # Combined OOS
    print(f"\n  --- Combined OOS (all 5 folds) ---")
    m_v8_all = compute_metrics(oos_trades_v8)
    m_onh_all = compute_metrics(oos_trades_onh)
    print(f"  {'v8 baseline':<20s}  {m_v8_all.total_trades:>4d} trades  "
          f"WR {m_v8_all.win_rate:>5.1f}%  PF {m_v8_all.profit_factor:>6.3f}  "
          f"P&L ${m_v8_all.net_pnl:>+10,.0f}  DD ${m_v8_all.max_drawdown:>8,.0f}  "
          f"Sharpe {m_v8_all.sharpe:>5.2f}")
    print(f"  {'v8+ONH wide':<20s}  {m_onh_all.total_trades:>4d} trades  "
          f"WR {m_onh_all.win_rate:>5.1f}%  PF {m_onh_all.profit_factor:>6.3f}  "
          f"P&L ${m_onh_all.net_pnl:>+10,.0f}  DD ${m_onh_all.max_drawdown:>8,.0f}  "
          f"Sharpe {m_onh_all.sharpe:>5.2f}")

    # Profitable folds
    v8_profitable = sum(1 for fold in range(n_folds)
                        for _ in [1]  # placeholder
                        if True)  # count below
    # Recount properly
    v8_wins = 0
    onh_wins = 0
    for fold in range(n_folds):
        s = fold * fold_size
        e = (fold + 1) * fold_size if fold < n_folds - 1 else n
        df_f = df.iloc[s:e].copy()
        m1 = compute_metrics(run_v8_baseline(df_f))
        m2 = compute_metrics(run_v8_plus_onh(df_f))
        if m1.net_pnl > 0:
            v8_wins += 1
        if m2.net_pnl > 0:
            onh_wins += 1

    print(f"\n  Profitable folds: v8 = {v8_wins}/5, v8+ONH = {onh_wins}/5")

    # PF ratio (OOS / full-sample)
    t_full_v8 = run_v8_baseline(df.copy())
    t_full_onh = run_v8_plus_onh(df.copy())
    m_full_v8 = compute_metrics(t_full_v8)
    m_full_onh = compute_metrics(t_full_onh)

    pf_ratio_v8 = m_v8_all.profit_factor / m_full_v8.profit_factor if m_full_v8.profit_factor > 0 else 0
    pf_ratio_onh = m_onh_all.profit_factor / m_full_onh.profit_factor if m_full_onh.profit_factor > 0 else 0

    print(f"\n  PF Ratio (OOS/Full): v8 = {pf_ratio_v8:.2f}, v8+ONH = {pf_ratio_onh:.2f}")
    print(f"  (>0.7 = robust, >0.5 = acceptable)")

    # ────────────────────────────────────────────────
    # 50/50 Year Split
    # ────────────────────────────────────────────────
    hdr("50/50 YEAR SPLIT (Y2=IS, Y1=OOS)")

    mid = n // 2
    df_y2 = df.iloc[:mid].copy()
    df_y1 = df.iloc[mid:].copy()

    print(f"\n  IS  (Year 2): {df_y2.index[0].date()} → {df_y2.index[-1].date()}")
    print(f"  OOS (Year 1): {df_y1.index[0].date()} → {df_y1.index[-1].date()}")

    # v8
    t_v8_is = run_v8_baseline(df_y2)
    t_v8_oos = run_v8_baseline(df_y1)
    m_v8_is = compute_metrics(t_v8_is)
    m_v8_oos = compute_metrics(t_v8_oos)

    # ONH
    t_onh_is = run_v8_plus_onh(df_y2)
    t_onh_oos = run_v8_plus_onh(df_y1)
    m_onh_is = compute_metrics(t_onh_is)
    m_onh_oos = compute_metrics(t_onh_oos)

    print(f"\n  {'Variant':<20s}  {'Split':<5s}  {'Trades':>6s}  {'WR':>6s}  "
          f"{'PF':>7s}  {'P&L':>10s}  {'DD':>8s}  {'Sharpe':>7s}")
    print(f"  {'-'*80}")
    print(f"  {'v8':<20s}  {'IS':<5s}  {m_v8_is.total_trades:>6d}  {m_v8_is.win_rate:>5.1f}%  "
          f"{m_v8_is.profit_factor:>7.3f}  ${m_v8_is.net_pnl:>+9,.0f}  "
          f"${m_v8_is.max_drawdown:>7,.0f}  {m_v8_is.sharpe:>7.2f}")
    print(f"  {'v8':<20s}  {'OOS':<5s}  {m_v8_oos.total_trades:>6d}  {m_v8_oos.win_rate:>5.1f}%  "
          f"{m_v8_oos.profit_factor:>7.3f}  ${m_v8_oos.net_pnl:>+9,.0f}  "
          f"${m_v8_oos.max_drawdown:>7,.0f}  {m_v8_oos.sharpe:>7.2f}")
    print(f"  {'v8+ONH wide':<20s}  {'IS':<5s}  {m_onh_is.total_trades:>6d}  {m_onh_is.win_rate:>5.1f}%  "
          f"{m_onh_is.profit_factor:>7.3f}  ${m_onh_is.net_pnl:>+9,.0f}  "
          f"${m_onh_is.max_drawdown:>7,.0f}  {m_onh_is.sharpe:>7.2f}")
    print(f"  {'v8+ONH wide':<20s}  {'OOS':<5s}  {m_onh_oos.total_trades:>6d}  {m_onh_oos.win_rate:>5.1f}%  "
          f"{m_onh_oos.profit_factor:>7.3f}  ${m_onh_oos.net_pnl:>+9,.0f}  "
          f"${m_onh_oos.max_drawdown:>7,.0f}  {m_onh_oos.sharpe:>7.2f}")

    # PF ratio
    pf_r_v8 = m_v8_oos.profit_factor / m_v8_is.profit_factor if m_v8_is.profit_factor > 0 else 0
    pf_r_onh = m_onh_oos.profit_factor / m_onh_is.profit_factor if m_onh_is.profit_factor > 0 else 0
    print(f"\n  PF Ratio (OOS/IS): v8 = {pf_r_v8:.2f}, v8+ONH = {pf_r_onh:.2f}")

    # ────────────────────────────────────────────────
    # Per-setup breakdown (full 2yr)
    # ────────────────────────────────────────────────
    hdr("FULL 2-YEAR BREAKDOWN")

    print(f"\n  --- v8+ONH Full 2yr ---")
    pm_full = compute_metrics(t_full_onh)
    print(f"  ALL: {pm_full.total_trades} trades  PF {pm_full.profit_factor:.3f}  "
          f"P&L ${pm_full.net_pnl:+,.0f}  DD ${pm_full.max_drawdown:,.0f}  "
          f"Sharpe {pm_full.sharpe:.2f}")

    bd = per_setup_breakdown(t_full_onh)
    for setup, sm in sorted(bd.items()):
        avg = sm.net_pnl / sm.total_trades if sm.total_trades > 0 else 0
        print(f"    {setup:<10s}  {sm.total_trades:>4d} trades  "
              f"WR {sm.win_rate:>5.1f}%  PF {sm.profit_factor:>6.3f}  "
              f"P&L ${sm.net_pnl:>+10,.0f}  Avg ${avg:>+7,.0f}")

    # ────────────────────────────────────────────────
    # Statistical significance
    # ────────────────────────────────────────────────
    hdr("STATISTICAL SIGNIFICANCE")

    onh_only = [t for t in t_full_onh if t.setup == "ONH_REJ"]
    if onh_only:
        pnls = np.array([t.pnl_dollar for t in onh_only])
        from scipy import stats

        # t-test
        t_stat, p_ttest = stats.ttest_1samp(pnls, 0)
        print(f"\n  ONH trades only ({len(onh_only)} trades):")
        print(f"    Mean P&L:  ${np.mean(pnls):+,.2f}")
        print(f"    Std P&L:   ${np.std(pnls, ddof=1):,.2f}")
        print(f"    t-stat:    {t_stat:.3f}")
        print(f"    p-value:   {p_ttest:.4f} {'***' if p_ttest < 0.01 else '**' if p_ttest < 0.05 else '*' if p_ttest < 0.1 else ''}")

        # Permutation test
        n_perms = 10000
        observed_mean = np.mean(pnls)
        count_ge = 0
        for _ in range(n_perms):
            shuffled = pnls * np.random.choice([-1, 1], size=len(pnls))
            if np.mean(shuffled) >= observed_mean:
                count_ge += 1
        p_perm = count_ge / n_perms
        print(f"    Perm p:    {p_perm:.4f} {'***' if p_perm < 0.01 else '**' if p_perm < 0.05 else '*' if p_perm < 0.1 else ''}")

        # Bootstrap P(profit)
        n_boot = 10000
        profits = 0
        for _ in range(n_boot):
            sample = np.random.choice(pnls, size=len(pnls), replace=True)
            if np.sum(sample) > 0:
                profits += 1
        print(f"    Bootstrap P(profit): {profits/n_boot*100:.1f}%")

    # Combined system
    all_pnls = np.array([t.pnl_dollar for t in t_full_onh])
    t_stat, p_ttest = stats.ttest_1samp(all_pnls, 0)
    print(f"\n  Combined system ({len(t_full_onh)} trades):")
    print(f"    Mean P&L:  ${np.mean(all_pnls):+,.2f}")
    print(f"    t-stat:    {t_stat:.3f}")
    print(f"    p-value:   {p_ttest:.4f} {'***' if p_ttest < 0.01 else '**' if p_ttest < 0.05 else '*' if p_ttest < 0.1 else ''}")

    # Monthly breakdown
    hdr("MONTHLY BREAKDOWN (v8+ONH)")
    monthly = defaultdict(list)
    for t in t_full_onh:
        if hasattr(t.entry_time, 'strftime'):
            key = t.entry_time.strftime("%Y-%m")
            monthly[key].append(t.pnl_dollar)

    print(f"\n  {'Month':<10s}  {'Trades':>6s}  {'P&L':>10s}  {'WR':>6s}")
    print(f"  {'-'*40}")
    winners = 0
    for month in sorted(monthly.keys()):
        pnls = monthly[month]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0
        total = sum(pnls)
        if total > 0:
            winners += 1
        print(f"  {month:<10s}  {len(pnls):>6d}  ${total:>+9,.0f}  {wr:>5.1f}%")
    print(f"\n  Winning months: {winners}/{len(monthly)} ({winners/len(monthly)*100:.0f}%)")


if __name__ == "__main__":
    main()
