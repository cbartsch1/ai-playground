#!/usr/bin/env python3
"""VIX Spike ES — 5yr drift compare (validated vs deployed).

Loads the canonical 5yr ES parquet (built per 2026-04-26-es-5yr-build.md) and
runs both the validated config (7%/no-filter/30bps/hold-all-day/9:35-15:55) and
the deployed config (5%/-0.1%/50bps/fixed_target 20pt/9:35-15:00) using the
v2 simulator with the newly-added fixed_target exit mode.

Drift quantification — completes the deferred item from
2026-04-25-portfolio-drift-backtest-RESULTS.md.

Output: prints metrics + walk-forward + comparison + writes nothing (caller
collects stdout for the report).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.indicators import compute_indicators
from scripts.test_vix_spike_es_v2 import (
    Config, run_backtest, compute_metrics, t_test, permutation_test,
    bootstrap_profit_prob,
)

ES_5YR_PARQUET = os.path.expanduser(
    "~/projects/backtesting/es/data/es_5m_5yr_from_1m.parquet"
)
VIX_PARQUET = os.path.expanduser("~/projects/backtesting/spx/data/vix_daily.parquet")

# 5yr WF split — clean 2yr era starts 2024-02-15, so IS=3yr+ gap-fill, OOS=2yr clean.
WF_SPLIT = "2024-02-15"


def load_data_5yr():
    """Load canonical 5yr ES parquet, attach indicators + session tags, plus VIX.

    Mirrors `scripts.test_vix_spike_es_v2.load_data` post-conditions:
      - df indexed by ET-tz timestamp with: open/high/low/close/volume/et_time/
        is_rth/session_date/weekday/hlc3 + indicator columns from compute_indicators
      - vix_lookup: dict[date] -> {open,high,low,close,prev_close,prev_high,prev_low}
      - session_opens: dict[session_date] -> first RTH open
      - prev_session_data: dict[session_date] -> {prev_close, prev_low}
    """
    print(f"Loading ES 5yr parquet: {ES_5YR_PARQUET}")
    df = pd.read_parquet(ES_5YR_PARQUET)

    # Index is `timestamp` column or already the index — handle both.
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()

    # Drop the source/is_rth columns we'll recompute below for parity.
    if "source" in df.columns:
        df = df.drop(columns=["source"])
    if "is_rth" in df.columns:
        df = df.drop(columns=["is_rth"])

    # Session tags — mirror tag_sessions() from data_loader.py
    et_hour = df.index.hour
    et_minute = df.index.minute
    df["et_time"] = et_hour * 100 + et_minute
    df["weekday"] = df.index.weekday
    df["is_rth"] = (df["et_time"] >= 930) & (df["et_time"] < 1600)
    df["new_rth"] = df["is_rth"] & ~df["is_rth"].shift(1, fill_value=False)
    rth_dates = pd.Series(pd.NaT, index=df.index, dtype="object")
    rth_dates[df["new_rth"]] = df.index[df["new_rth"]].date
    df["session_date"] = rth_dates.ffill()
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0

    print(f"  {len(df):,} bars: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  RTH bars: {df['is_rth'].sum():,} | Sessions: {df['new_rth'].sum()}")

    # Indicators (TEMA, ATR, vol_ratio, etc.) — required by run_backtest's
    # vol_ratio/tema_trend filters even when those filters are disabled.
    compute_indicators(df)

    print(f"Loading VIX data: {VIX_PARQUET}")
    vix = pd.read_parquet(VIX_PARQUET)
    vix_lookup = {}
    prev_close = prev_high = prev_low = None
    for idx, row in vix.iterrows():
        d = idx.date() if hasattr(idx, "date") else idx
        vix_lookup[d] = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row.get("low", 0)),
            "close": float(row["close"]),
            "prev_close": prev_close,
            "prev_high": prev_high,
            "prev_low": prev_low,
        }
        prev_close = float(row["close"])
        prev_high = float(row["high"])
        prev_low = float(row.get("low", 0))

    es_dates = set(df[df["is_rth"]]["session_date"].dropna().unique())
    vix_dates = set(vix_lookup.keys())
    overlap = es_dates & vix_dates
    print(f"  VIX/ES overlap: {len(overlap)} sessions")

    rth = df[df["is_rth"]]
    session_opens = rth.groupby("session_date")["open"].first().to_dict()
    session_closes = rth.groupby("session_date")["close"].last().to_dict()
    session_lows = rth.groupby("session_date")["low"].min().to_dict()

    sorted_dates = sorted(session_closes.keys())
    prev_session_data = {}
    for i, d in enumerate(sorted_dates):
        if i > 0:
            prev_d = sorted_dates[i - 1]
            prev_session_data[d] = {
                "prev_close": session_closes[prev_d],
                "prev_low": session_lows.get(prev_d, 0),
            }

    return df, vix_lookup, session_opens, prev_session_data


def run_with_wf(df, vix_lookup, session_opens, prev_session_data, cfg, label):
    print(f"\n{'─' * 78}")
    print(f"  {label}")
    print(f"  signal={cfg.signal_mode} thresh={cfg.spike_threshold} "
          f"es_filter={cfg.es_move_filter} stop={cfg.stop_bps}bps "
          f"exit={cfg.exit_mode} target={cfg.target_pts}pts "
          f"max_hold={cfg.max_hold_bars}b entry=[{cfg.entry_start},{cfg.entry_end})")
    print(f"{'─' * 78}")

    trades = run_backtest(df, vix_lookup, session_opens, prev_session_data, cfg)
    m = compute_metrics(trades)
    p = t_test(trades)

    df_is = df[df.index < WF_SPLIT]
    df_oos = df[df.index >= WF_SPLIT]
    trades_is = run_backtest(df_is, vix_lookup, session_opens, prev_session_data, cfg)
    trades_oos = run_backtest(df_oos, vix_lookup, session_opens, prev_session_data, cfg)
    m_is = compute_metrics(trades_is)
    m_oos = compute_metrics(trades_oos)
    p_oos = t_test(trades_oos) if trades_oos else 1.0
    wf = m_oos["pf"] / m_is["pf"] if (m_is.get("pf", 0) and m_is["pf"] > 0
                                       and m_oos.get("total", 0) > 0) else 0

    perm = permutation_test(trades) if len(trades) >= 5 else 1.0
    boot = bootstrap_profit_prob(trades) if len(trades) >= 5 else 0.0

    print(f"  FULL: trades={m['total']:>4} pf={m['pf']:>7.4f} "
          f"net=${m['net_pnl']:>+12,.0f} wr={m['win_rate']:>5.2f}% "
          f"sharpe={m['sharpe']:>6.3f} dd=${m['max_dd']:>10,.0f} p={p:.6f}")
    print(f"  IS  : trades={m_is['total']:>4} pf={m_is['pf']:>7.4f} "
          f"net=${m_is['net_pnl']:>+12,.0f} wr={m_is['win_rate']:>5.2f}% "
          f"sharpe={m_is['sharpe']:>6.3f}")
    print(f"  OOS : trades={m_oos['total']:>4} pf={m_oos['pf']:>7.4f} "
          f"net=${m_oos['net_pnl']:>+12,.0f} wr={m_oos['win_rate']:>5.2f}% "
          f"sharpe={m_oos['sharpe']:>6.3f} p={p_oos:.6f}")
    print(f"  WF ratio: {wf:.3f}  perm-p: {perm:.4f}  bootstrap P(profit): {boot:.2%}")

    # Exit-reason breakdown
    reasons = {}
    for t in trades:
        r = t.exit_reason
        d = reasons.setdefault(r, {"count": 0, "pnl": 0.0, "wins": 0})
        d["count"] += 1
        d["pnl"] += t.pnl_dollar
        if t.pnl_dollar > 0:
            d["wins"] += 1
    print(f"  EXITS:")
    for r, d in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        wr = d["wins"] / d["count"] * 100 if d["count"] else 0
        print(f"    {r:<16} {d['count']:>4} trades  WR {wr:>5.1f}%  ${d['pnl']:>+10,.0f}")

    return {"cfg": cfg, "m": m, "p": p, "m_is": m_is, "m_oos": m_oos,
            "p_oos": p_oos, "wf": wf, "perm": perm, "boot": boot,
            "trades": trades}


def main():
    import argparse
    from scripts.test_vix_spike_es_v2 import LOOKAHEAD_MSG
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-lookahead-daily-mode", action="store_true",
                    help="Explicit opt-in to the NON-CAUSAL daily-VIX-HIGH "
                         "spike detection (diagnostics only).")
    args = ap.parse_args()
    if not args.allow_lookahead_daily_mode:
        raise SystemExit("VOIDED (2026-06-11 audit): "
                         + LOOKAHEAD_MSG.format(mode="daily_spike"))

    df, vix_lookup, session_opens, prev_session_data = load_data_5yr()

    print("\n" + "=" * 78)
    print("  VIX SPIKE ES — 5YR DRIFT COMPARE  (2026-04-26)")
    print("=" * 78)
    print(f"  Data: ES 5m 5yr canonical parquet (2021-01-03 → 2026-02-23)")
    print(f"  VIX:  spx/data/vix_daily.parquet")
    print(f"  WF split: {WF_SPLIT} (IS=2021-01..2024-02 = 3yr; "
          f"OOS=2024-02..2026-02 = 2yr clean databento)")

    # ── Validated: 7% / no filter / 30bps / hold-all-day / 9:35-15:55 ──
    validated_cfg = Config(
        label="Validated 7% / no-filter / 30bps / hold-all-day / 9:35-15:55",
        signal_mode="daily_spike", spike_threshold=0.07,
        allow_lookahead_daily_mode=True,
        entry_mode="red_bar", exit_mode="hold_all_day",
        stop_bps=30.0, max_hold_bars=200,
        es_move_filter=0.0,
        entry_start=935, entry_end=1555,
    )

    # ── Deployed: 5% / -0.1% / 50bps / fixed_target 20pt / 90min / 9:35-15:00 ──
    deployed_cfg = Config(
        label="Deployed 5% / ES-0.1% / 50bps / fixed_target 20pt / 90m / 9:35-15:00",
        signal_mode="daily_spike", spike_threshold=0.05,
        allow_lookahead_daily_mode=True,
        entry_mode="red_bar", exit_mode="fixed_target",
        stop_bps=50.0, max_hold_bars=18,
        es_move_filter=-0.001,
        target_pts=20.0,
        entry_start=935, entry_end=1500,
    )

    val = run_with_wf(df, vix_lookup, session_opens, prev_session_data,
                       validated_cfg, "VALIDATED config — 5yr")
    dep = run_with_wf(df, vix_lookup, session_opens, prev_session_data,
                       deployed_cfg, "DEPLOYED config — 5yr")

    # ── Side-by-side ──
    print(f"\n{'=' * 78}")
    print("  HEAD-TO-HEAD (5yr full period)")
    print(f"{'=' * 78}")
    print(f"  {'metric':<14} {'VALIDATED':>16} {'DEPLOYED':>16} {'DELTA':>16}")
    print("  " + "-" * 70)

    def row(name, vk, dk, is_dollar=False, is_int=False):
        v, d = vk, dk
        delta = d - v
        if is_dollar:
            print(f"  {name:<14} {('$' + format(v, '+,.0f')):>16} "
                  f"{('$' + format(d, '+,.0f')):>16} {('$' + format(delta, '+,.0f')):>16}")
        elif is_int:
            print(f"  {name:<14} {v:>16d} {d:>16d} {delta:>+16d}")
        else:
            print(f"  {name:<14} {v:>16.4f} {d:>16.4f} {delta:>+16.4f}")

    row("trades", val["m"]["total"], dep["m"]["total"], is_int=True)
    row("PF", val["m"]["pf"], dep["m"]["pf"])
    row("Net P&L", val["m"]["net_pnl"], dep["m"]["net_pnl"], is_dollar=True)
    row("Win Rate %", val["m"]["win_rate"], dep["m"]["win_rate"])
    row("Sharpe", val["m"]["sharpe"], dep["m"]["sharpe"])
    row("Max DD", val["m"]["max_dd"], dep["m"]["max_dd"], is_dollar=True)
    row("OOS PF", val["m_oos"]["pf"], dep["m_oos"]["pf"])
    row("OOS Net", val["m_oos"]["net_pnl"], dep["m_oos"]["net_pnl"], is_dollar=True)
    row("WF ratio", val["wf"], dep["wf"])

    print(f"\n  per-year normalize:")
    yrs = 5.14  # 2021-01-03 → 2026-02-23
    print(f"    Validated $/yr (5yr): ${val['m']['net_pnl'] / yrs:>+,.0f}")
    print(f"    Deployed  $/yr (5yr): ${dep['m']['net_pnl'] / yrs:>+,.0f}")

    print(f"\n{'=' * 78}\n")


if __name__ == "__main__":
    main()
