"""Custom simulators for trend_cont_es, vector_es, vix_spike_es (causal).

Each mirrors the deployed TV impl semantics and the corresponding es test
harness's cost model. All return trade tuples (setup, date_iso, hhmm, pnl).

lag=True re-runs with entry filled one 5m bar later (next bar close, same
session) — the honest live-fill bracket for the top-3 sensitivity check.
"""

import numpy as np
import pandas as pd

ES_POINT_VALUE = 50.0
TICK = 0.25


# ── shared session tables ────────────────────────────────────────────

def session_tables(df):
    """Per-session ON low / prev RTH close / RTH open, from the engine-ready df."""
    rth = df[df["is_rth"]]
    closes = rth.groupby("session_date")["close"].last()
    opens = rth.groupby("session_date")["open"].first()
    sessions = list(closes.index)
    prev_close = {}
    for i in range(1, len(sessions)):
        prev_close[sessions[i]] = float(closes.iloc[i - 1])
    # ON low for session d = min low of globex bars carrying the PREV
    # session_date (18:00 prev day -> 9:25 of d)
    glx = df[df["is_globex"]]
    on_low_by_prev = glx.groupby("session_date")["low"].min()
    on_low = {}
    for i in range(1, len(sessions)):
        v = on_low_by_prev.get(sessions[i - 1])
        if v is not None and not np.isnan(v):
            on_low[sessions[i]] = float(v)
    rth_open = {d: float(v) for d, v in opens.items()}
    return sessions, prev_close, on_low, rth_open


# ── trend_cont_es ────────────────────────────────────────────────────

def _aggregate_30m(df_5m):
    rth = df_5m[df_5m["is_rth"]].copy()
    rth["m30_bucket"] = rth.index.floor("30min")
    grouped = rth.groupby([rth["session_date"], rth["m30_bucket"]])
    bars = grouped.agg(open=("open", "first"), high=("high", "max"),
                       low=("low", "min"), close=("close", "last"),
                       volume=("volume", "sum"))
    bars = bars.droplevel(0).sort_index()
    bars["session_date"] = grouped["session_date"].first().droplevel(0)
    return bars


def trend_cont_sim(df, spec, lag=False):
    """30m breakdown short. Mirrors scripts/test_trend_cont_es.run_trend_cont
    (same +30min merge shift, same fills/costs: $4.62 RT + 0.5pt slippage)
    plus the deployed vol-surge and Friday-skip filters."""
    COMMISSION_RT = 4.62
    SLIPPAGE_PTS = 0.50
    FLATTEN = 1555

    d30 = _aggregate_30m(df)
    c = d30["close"].values
    l = d30["low"].values
    s30 = d30["session_date"].values
    n30 = len(d30)
    lower_close = np.zeros(n30, dtype=bool)
    for i in range(1, n30):
        if s30[i] == s30[i - 1] and c[i] < l[i - 1]:
            lower_close[i] = True

    # Vol surge: 30m volume > rolling mean of prior 20 RTH 30m vols * mult
    surge_ok = np.ones(n30, dtype=bool)
    if spec["vol_surge"]:
        avg = pd.Series(d30["volume"].values).shift(1).rolling(
            20, min_periods=1).mean().values
        surge_ok = d30["volume"].values > avg * spec["vol_surge"]

    sig30 = lower_close & surge_ok
    # +30min shift: a 30m bar's signal becomes usable only at/after its close
    # (the audited-clean alignment from test_trend_cont_es)
    sig_df = pd.DataFrame({"datetime": d30.index + pd.Timedelta(minutes=30),
                           "sig": sig30})

    rth = df[df["is_rth"]].copy()
    left = rth.reset_index()
    left = left.rename(columns={left.columns[0]: "datetime"})
    merged = pd.merge_asof(left, sig_df, on="datetime",
                           direction="backward").set_index("datetime")
    sig = merged["sig"].fillna(False).values

    n = len(merged)
    et = merged["et_time"].values
    sess = merged["session_date"].values
    cl = merged["close"].values
    op = merged["open"].values
    hi = merged["high"].values
    lo = merged["low"].values
    wd = merged["weekday"].values
    times = merged.index

    trades = []
    in_pos = False
    entry_price = stop_price = target_price = 0.0
    entry_idx = 0
    trade_count = {}
    pending_entry = -1  # for lag mode

    def book(i, exit_price, reason):
        pnl_pts = entry_price - exit_price
        pnl = (pnl_pts * ES_POINT_VALUE - COMMISSION_RT) - SLIPPAGE_PTS * ES_POINT_VALUE
        d = times[entry_idx]
        trades.append(("TC", str(d.date()), int(d.hour * 100 + d.minute),
                       round(pnl, 2)))

    for i in range(n):
        s = sess[i]
        t = et[i]

        if in_pos:
            if s != sess[entry_idx]:
                book(i, cl[i - 1] if i > 0 else cl[i], "session_end")
                in_pos = False
            elif hi[i] >= stop_price:
                book(i, stop_price, "stop")
                in_pos = False
                continue
            elif t >= FLATTEN:
                book(i, cl[i], "flatten")
                in_pos = False
                continue
            elif spec["exit_mode"] == "fixed_target" and lo[i] <= target_price:
                book(i, target_price, "target")
                in_pos = False
                continue
            if in_pos and (i - entry_idx) >= spec["max_hold"]:
                book(i, cl[i], "max_hold")
                in_pos = False
                continue
            if in_pos and spec["exit_mode"] == "green_bar" and cl[i] > op[i]:
                book(i, cl[i], "green_bar")
                in_pos = False
                continue
            if in_pos:
                continue

        # pending lag entry fills at this bar's close (same session only)
        if pending_entry >= 0:
            if sess[i] == sess[pending_entry] and t < FLATTEN:
                entry_price = cl[i]
                entry_idx = i
                stop_price = entry_price * (1.0 + spec["stop_bps"] / 10000.0)
                target_price = (entry_price - spec["target_pts"]
                                if spec["exit_mode"] == "fixed_target" else 0.0)
                in_pos = True
            pending_entry = -1
            continue

        if t < 935 or t >= spec["entry_end"]:
            continue
        if trade_count.get(s, 0) >= 1:
            continue
        if not sig[i]:
            continue
        if spec["skip_friday"] and wd[i] == 4:
            continue

        trade_count[s] = trade_count.get(s, 0) + 1
        if lag:
            pending_entry = i
            continue
        entry_price = cl[i]
        entry_idx = i
        stop_price = entry_price * (1.0 + spec["stop_bps"] / 10000.0)
        target_price = (entry_price - spec["target_pts"]
                        if spec["exit_mode"] == "fixed_target" else 0.0)
        in_pos = True

    if in_pos:
        book(n - 1, cl[-1], "data_end")
    return trades


# ── vector_es ────────────────────────────────────────────────────────

def vector_sim(df, spec, tables=None, lag=False):
    """EMA bearish cross short with ON-gap filter (deployed TV vector_es).

    Entry: EMA fast crosses below slow on RTH closes, prev_close - ONL >=
    gap threshold, window 9:35-15:00, max N/day. Stop: bps. Target: ONL or
    fixed pts. EOD flatten 15:55. Costs: $5.00 RT + 1 tick slippage/side.
    """
    COMMISSION_RT = 5.0
    FLATTEN = 1555

    if tables is None:
        tables = session_tables(df)
    sessions, prev_close, on_low, rth_open = tables

    rth = df[df["is_rth"]]
    cl = rth["close"].values
    op = rth["open"].values
    hi = rth["high"].values
    lo = rth["low"].values
    et = rth["et_time"].values
    sess = rth["session_date"].values
    times = rth.index
    n = len(rth)

    fast = pd.Series(cl).ewm(span=spec["ema_fast"], adjust=False).mean().values
    slow = pd.Series(cl).ewm(span=spec["ema_slow"], adjust=False).mean().values
    cross = np.zeros(n, dtype=bool)
    cross[1:] = (fast[:-1] >= slow[:-1]) & (fast[1:] < slow[1:])

    trades = []
    in_pos = False
    entry_price = stop_price = target_price = 0.0
    entry_idx = 0
    count = {}
    pending = -1

    def book(exit_price):
        pnl_pts = entry_price - exit_price
        pnl = pnl_pts * ES_POINT_VALUE - COMMISSION_RT
        d = times[entry_idx]
        trades.append(("VEC", str(d.date()), int(d.hour * 100 + d.minute),
                       round(pnl, 2)))

    def enter(i):
        nonlocal entry_price, stop_price, target_price, entry_idx, in_pos
        entry_price = cl[i] - TICK
        entry_idx = i
        stop_price = entry_price * (1.0 + spec["stop_bps"] / 10000.0)
        if spec["target_mode"] == "onl":
            target_price = on_low[sess[i]]
        else:
            target_price = entry_price - spec["target_pts"]
        in_pos = True

    for i in range(n):
        s = sess[i]
        t = et[i]

        if in_pos:
            if s != sess[entry_idx]:
                book(cl[i - 1] + TICK)
                in_pos = False
            elif hi[i] >= stop_price:
                book(stop_price + TICK)
                in_pos = False
                continue
            elif lo[i] <= target_price:
                book(target_price + TICK)
                in_pos = False
                continue
            elif t >= FLATTEN:
                book(cl[i] + TICK)
                in_pos = False
                continue
            else:
                continue

        if pending >= 0:
            if sess[i] == sess[pending] and t < FLATTEN:
                enter(i)
            pending = -1
            continue

        if t < 935 or t >= 1500:
            continue
        if count.get(s, 0) >= spec["max_per_day"]:
            continue
        if not cross[i]:
            continue
        pc = prev_close.get(s)
        onl = on_low.get(s)
        if pc is None or onl is None:
            continue
        if pc - onl < spec["gap_threshold"]:
            continue
        # NOTE: deployed impl has no "ONL below entry" guard — if price is
        # already at/below ONL the trade exits immediately at target (a
        # scratch/small loss). Mirrored faithfully.

        count[s] = count.get(s, 0) + 1
        if lag:
            pending = i
            continue
        enter(i)

    if in_pos:
        book(cl[-1] + TICK)
    return trades


# ── vix_spike_es (causal intraday, hourly granularity) ──────────────

def vix_spike_sim(df, spec, vix_hourly, vix_daily, tables=None, lag=False):
    """Causal intraday VIX-spike short.

    Spike detection on HOURLY VIX bars: first bar whose high >= ref *
    (1 + threshold); usable from the bar's close minute (granularity
    caveat: up to ~60 min later than live's 60s polling). ref =
    prior-day VIX close (deployed live semantics) or day open.
    Entry: first red 5m ES bar at/after the spike-known time, in window,
    with ES-move-from-open filter. 1 trade/day. Costs mirror
    test_vix_spike_es: $2.50/side + 1 tick slippage/side.
    """
    COMMISSION_RT = 5.0
    FLATTEN = 1555

    if tables is None:
        tables = session_tables(df)
    sessions, prev_close_es, on_low, rth_open = tables

    # spike-known minute per session
    known_min = {}
    for d in sessions:
        obs = vix_hourly.get(d)
        if not obs:
            continue
        if spec["ref"] == "prev_close":
            v = vix_daily.get(d)
            ref = v["prev_close"] if v else None
        else:
            ref = obs["open"]
        if not ref or ref <= 0:
            continue
        for b in obs["bars"]:
            if b["high"] >= ref * (1.0 + spec["threshold"]):
                known_min[d] = b["close_min"]
                break

    rth = df[df["is_rth"]]
    cl = rth["close"].values
    op = rth["open"].values
    hi = rth["high"].values
    lo = rth["low"].values
    et = rth["et_time"].values
    sess = rth["session_date"].values
    times = rth.index
    n = len(rth)
    minute = (et // 100) * 60 + (et % 100)

    trades = []
    in_pos = False
    entry_price = stop_price = target_price = 0.0
    entry_idx = 0
    traded = set()
    pending = -1

    def book(exit_price):
        pnl_pts = entry_price - exit_price
        pnl = pnl_pts * ES_POINT_VALUE - COMMISSION_RT
        d = times[entry_idx]
        trades.append(("VSPK", str(d.date()), int(d.hour * 100 + d.minute),
                       round(pnl, 2)))

    def enter(i):
        nonlocal entry_price, stop_price, target_price, entry_idx, in_pos
        entry_price = cl[i] - TICK
        entry_idx = i
        stop_price = entry_price * (1.0 + spec["stop_bps"] / 10000.0)
        if spec["exit_mode"] == "fixed_target":
            target_price = entry_price - spec["target_pts"]
        else:
            target_price = -1e9   # unreachable
        in_pos = True

    for i in range(n):
        s = sess[i]
        t = et[i]

        if in_pos:
            held = i - entry_idx
            if s != sess[entry_idx]:
                book(cl[i - 1] + TICK)
                in_pos = False
            elif hi[i] >= stop_price:
                book(stop_price + TICK)
                in_pos = False
                continue
            elif spec["exit_mode"] == "fixed_target" and lo[i] <= target_price:
                book(target_price + TICK)
                in_pos = False
                continue
            elif t >= FLATTEN:
                book(cl[i] + TICK)
                in_pos = False
                continue
            elif spec["exit_mode"] == "green_bar" and cl[i] > op[i]:
                book(cl[i] + TICK)
                in_pos = False
                continue
            elif held >= spec["max_hold"]:
                book(cl[i] + TICK)
                in_pos = False
                continue
            else:
                continue

        if pending >= 0:
            if sess[i] == sess[pending] and t < FLATTEN:
                enter(i)
            pending = -1
            continue

        if s in traded:
            continue
        km = known_min.get(s)
        if km is None or minute[i] < km:
            continue
        if t < 935 or t >= 1500:
            continue
        if cl[i] >= op[i]:
            continue
        if spec["es_move_filter"] < 0:
            ro = rth_open.get(s)
            if ro and (cl[i] - ro) / ro > spec["es_move_filter"]:
                continue

        traded.add(s)
        if lag:
            pending = i
            continue
        enter(i)

    if in_pos:
        book(cl[-1] + TICK)
    return trades
