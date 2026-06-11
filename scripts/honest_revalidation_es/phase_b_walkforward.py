#!/usr/bin/env python3
"""Phase B — vix_spike causal-intraday expanding-window walk-forward.

Pre-registered: lab plans/2026-06-10-edge-legitimacy-audit/
phase-b-es-preregistration.md (lab commit 0b48087, BEFORE this ran).

Pipeline:
  1. Fidelity gate: fast known-min sim must exactly reproduce
     sims.vix_spike_sim for all 128 specs, lag=False and lag=True, plus a
     truncation-equivalence check on the Phase A top-3 at fold-1 train end.
  2. Real walk-forward (6 expanding folds, per-fold TRAIN-only selection,
     scaled floors, deterministic tiebreak), no-lag and +1-bar-lag modes.
  3. Survival criteria + pooled sign-flip p (seed 777).
  4. Permutation null on the walk-forward itself: 100 year-matched
     random-day replicates (seeds 1000+r; sign-flip seeds 778+r), no-lag.

Usage: python scripts/honest_revalidation_es/phase_b_walkforward.py
       [--workers N] [--skip-gate]
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import date
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.honest_revalidation_es.lockbox import Lockbox
from scripts.honest_revalidation_es import grids as G
from scripts.honest_revalidation_es import sims as S
from scripts.honest_revalidation_es.scan_lib import metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")
ES_POINT_VALUE = 50.0
TICK = 0.25
COMMISSION_RT = 5.0
FLATTEN = 1555

# ── pre-registered constants ─────────────────────────────────────────

DATA_START = date(2021, 1, 4)
FOLDS = [  # (oos_start, oos_end) — train = DATA_START .. day before oos_start
    (date(2022, 1, 1), date(2022, 7, 31)),
    (date(2022, 8, 1), date(2023, 2, 28)),
    (date(2023, 3, 1), date(2023, 9, 30)),
    (date(2023, 10, 1), date(2024, 4, 30)),
    (date(2024, 5, 1), date(2024, 11, 30)),
    (date(2024, 12, 1), date(2025, 6, 30)),
]
BASE_FLOOR, BASE_YEARS = 60, 4.4
MIN_FLOOR = 12
N_NULL = 100
N_SIGNFLIP = 10_000
SEED_SIGNFLIP = 777
SEED_NULL_BASE = 1000
SEED_NULL_SIGNFLIP_BASE = 778

CRIT_FOLDS_PASS = 4          # >= 4/6 folds OOS PF > 1.0
CRIT_POOLED_PF = 1.5
CRIT_P = 0.05
CRIT_POOLED_N = 40


def scaled_floor(oos_start):
    train_years = (oos_start - DATA_START).days / 365.25
    return max(MIN_FLOOR, math.ceil(BASE_FLOOR * train_years / BASE_YEARS))


# ── spike-day machinery (identical math to sims.vix_spike_sim) ───────

COMBOS = [(thr, ref) for thr in (0.03, 0.05, 0.07, 0.10)
          for ref in ("prev_close", "day_open")]


def compute_known_min(sessions, vix_hourly, vix_daily, threshold, ref_mode):
    """date -> ET minute at which the spike becomes knowable (first hourly
    bar whose high >= ref*(1+threshold)); exactly sims.vix_spike_sim."""
    known = {}
    for d in sessions:
        obs = vix_hourly.get(d)
        if not obs:
            continue
        if ref_mode == "prev_close":
            v = vix_daily.get(d)
            ref = v["prev_close"] if v else None
        else:
            ref = obs["open"]
        if not ref or ref <= 0:
            continue
        for b in obs["bars"]:
            if b["high"] >= ref * (1.0 + threshold):
                known[d] = b["close_min"]
                break
    return known


# ── fast per-session sim (fidelity-gated vs sims.vix_spike_sim) ──────

def build_session_arrays(df):
    """Per-session contiguous numpy views of the RTH bars."""
    rth = df[df["is_rth"]]
    cl = rth["close"].values.astype(float)
    op = rth["open"].values.astype(float)
    hi = rth["high"].values.astype(float)
    lo = rth["low"].values.astype(float)
    et = rth["et_time"].values.astype(int)
    sess = rth["session_date"].values
    minute = (et // 100) * 60 + (et % 100)
    bounds = {}
    start = 0
    for i in range(1, len(sess)):
        if sess[i] != sess[i - 1]:
            bounds[sess[start]] = (start, i)
            start = i
    bounds[sess[start]] = (start, len(sess))
    return {"cl": cl, "op": op, "hi": hi, "lo": lo, "et": et,
            "minute": minute, "bounds": bounds,
            "sessions": [s for s in bounds]}


def sim_session(A, spec, s, km, rth_open_s, lag):
    """One session's trade (or None). Same per-bar logic as
    sims.vix_spike_sim restricted to one session."""
    cl, op, hi, lo, et, minute = (A["cl"], A["op"], A["hi"], A["lo"],
                                  A["et"], A["minute"])
    i0, i1 = A["bounds"][s]
    in_pos = False
    entry_price = stop_price = target_price = 0.0
    entry_i = -1
    pending = -1
    mv = spec["es_move_filter"]
    fixed_target = spec["exit_mode"] == "fixed_target"
    green_bar = spec["exit_mode"] == "green_bar"
    max_hold = spec["max_hold"]
    pnl = None
    entry_hhmm = entry_date = None

    def book(exit_price):
        nonlocal pnl
        pnl = round((entry_price - exit_price) * ES_POINT_VALUE
                    - COMMISSION_RT, 2)

    def enter(i):
        nonlocal entry_price, stop_price, target_price, entry_i, in_pos
        nonlocal entry_hhmm
        entry_price = cl[i] - TICK
        entry_i = i
        stop_price = entry_price * (1.0 + spec["stop_bps"] / 10000.0)
        target_price = (entry_price - spec["target_pts"] if fixed_target
                        else -1e9)
        in_pos = True
        entry_hhmm = int(et[i])

    for i in range(i0, i1):
        t = et[i]
        if in_pos:
            held = i - entry_i
            if hi[i] >= stop_price:
                book(stop_price + TICK)
                in_pos = False
                break
            elif fixed_target and lo[i] <= target_price:
                book(target_price + TICK)
                in_pos = False
                break
            elif t >= FLATTEN:
                book(cl[i] + TICK)
                in_pos = False
                break
            elif green_bar and cl[i] > op[i]:
                book(cl[i] + TICK)
                in_pos = False
                break
            elif held >= max_hold:
                book(cl[i] + TICK)
                in_pos = False
                break
            else:
                continue
        if pending >= 0:
            if t < FLATTEN:
                enter(i)
            pending = -1
            continue
        if entry_i >= 0:        # already traded this session
            break
        if minute[i] < km:
            continue
        if t < 935 or t >= 1500:
            continue
        if cl[i] >= op[i]:
            continue
        if mv < 0 and rth_open_s and (cl[i] - rth_open_s) / rth_open_s > mv:
            continue
        if lag:
            pending = i
            continue
        enter(i)

    if in_pos:                  # survived to session end
        book(cl[i1 - 1] + TICK)
    if pnl is None:
        return None
    return ("VSPK", str(s), entry_hhmm, pnl)


def fast_sim(A, known_min, spec, rth_open, lag=False):
    trades = []
    for s in sorted(known_min):
        if s not in A["bounds"]:
            continue
        tr = sim_session(A, spec, s, known_min[s], rth_open.get(s), lag)
        if tr is not None:
            trades.append(tr)
    return trades


# ── walk-forward (selection + criteria) ──────────────────────────────

def _spec_key(spec):
    return json.dumps({k: v for k, v in sorted(spec.items())})


def trades_in(trades, start, end):
    s, e = start.isoformat(), end.isoformat()
    return [t for t in trades if s <= t[1] <= e]


def signflip_p(pnls, seed):
    """One-sided sign-flip permutation p for mean PnL > 0."""
    if len(pnls) == 0:
        return 1.0
    x = np.asarray(pnls)
    obs = x.mean()
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(N_SIGNFLIP, len(x)),
                         dtype=np.int8) * 2 - 1
    means = (signs * x).mean(axis=1)
    return float((1 + int((means >= obs).sum())) / (N_SIGNFLIP + 1))


def pooled_pf(pnls):
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    if gl == 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def walk_forward(trades_by_spec, signflip_seed):
    """Run the registered fold selection + criteria over precomputed
    full-DEV trade lists keyed by spec key. Returns the full record."""
    folds_out = []
    pooled = []
    for (oos_start, oos_end) in FOLDS:
        train_end = date.fromordinal(oos_start.toordinal() - 1)
        floor_f = scaled_floor(oos_start)
        candidates = []
        for key, trades in trades_by_spec.items():
            m = metrics(trades_in(trades, DATA_START, train_end))
            if m["n"] >= floor_f:
                candidates.append((m["pf"], m["n"], m["net"], key))
        fold = {"oos_start": oos_start.isoformat(),
                "oos_end": oos_end.isoformat(), "floor": floor_f,
                "eligible": len(candidates)}
        if not candidates:
            fold.update({"chosen": None, "oos_n": 0, "oos_pf": 0.0,
                         "oos_net": 0.0, "pass_pf_gt1": False})
            folds_out.append(fold)
            continue
        candidates.sort(key=lambda c: (-c[0], -c[1], -c[2], c[3]))
        pf_t, n_t, net_t, key = candidates[0]
        oos = trades_in(trades_by_spec[key], oos_start, oos_end)
        om = metrics(oos)
        pooled.extend(t[3] for t in oos)
        fold.update({"chosen": key, "train_pf": pf_t, "train_n": n_t,
                     "train_net": net_t, "oos_n": om["n"],
                     "oos_pf": om["pf"], "oos_net": om["net"],
                     "pass_pf_gt1": om["n"] > 0 and om["pf"] > 1.0})
        folds_out.append(fold)

    n_pass = sum(1 for f in folds_out if f["pass_pf_gt1"])
    ppf = pooled_pf(pooled)
    p = signflip_p(pooled, signflip_seed)
    crit = {"folds_gt1": n_pass,
            "c1_folds": n_pass >= CRIT_FOLDS_PASS,
            "c2_pooled_pf": ppf >= CRIT_POOLED_PF,
            "c3_p": p < CRIT_P,
            "c4_n": len(pooled) >= CRIT_POOLED_N}
    if all(crit[k] for k in ("c1_folds", "c2_pooled_pf", "c3_p", "c4_n")):
        verdict = "SURVIVES-PHASE-B"
    elif crit["c1_folds"] and crit["c2_pooled_pf"] and crit["c3_p"]:
        verdict = "NOT-DECIDABLE-AT-N"
    else:
        verdict = "DIES"
    return {"folds": folds_out, "pooled_n": len(pooled),
            "pooled_pf": round(ppf, 4) if ppf != float("inf") else 99.0,
            "pooled_net": round(sum(pooled), 2), "pooled_p": p,
            "criteria": crit, "verdict": verdict}


# ── shared data (built once, reused by workers via fork) ─────────────

_SHARED = {}


def load_shared():
    box = Lockbox()
    df = box.load_es_5m()
    A = build_session_arrays(df)
    tables = S.session_tables(df)
    sessions, prev_close, on_low, rth_open = tables
    vix_hourly = box.load_vix_hourly()
    vix_daily = box.load_vix_daily()
    real_km = {c: compute_known_min(sessions, vix_hourly, vix_daily, *c)
               for c in COMBOS}
    grid = G.vix_spike_grid()
    _SHARED.update(df=df, A=A, tables=tables, rth_open=rth_open,
                   sessions=sessions, vix_hourly=vix_hourly,
                   vix_daily=vix_daily, real_km=real_km, grid=grid)
    return _SHARED


def run_grid(km_by_combo, lag=False):
    """Full-DEV trades for all 128 specs given a known-min map per combo."""
    A, rth_open = _SHARED["A"], _SHARED["rth_open"]
    out = {}
    for spec in _SHARED["grid"]:
        km = km_by_combo[(spec["threshold"], spec["ref"])]
        out[_spec_key(spec)] = fast_sim(A, km, spec, rth_open, lag=lag)
    return out


# ── fidelity gate ────────────────────────────────────────────────────

def fidelity_gate():
    print("[gate] fast sim vs sims.vix_spike_sim, 128 specs x 2 modes...")
    df, tables = _SHARED["df"], _SHARED["tables"]
    vh, vd = _SHARED["vix_hourly"], _SHARED["vix_daily"]
    A, rth_open = _SHARED["A"], _SHARED["rth_open"]
    mismatches = 0
    pf_at_floor = []
    for k, spec in enumerate(_SHARED["grid"]):
        km = _SHARED["real_km"][(spec["threshold"], spec["ref"])]
        for lag in (False, True):
            ref = S.vix_spike_sim(df, spec, vh, vd, tables=tables, lag=lag)
            fast = fast_sim(A, km, spec, rth_open, lag=lag)
            if ref != fast:
                mismatches += 1
                print(f"  MISMATCH spec={spec} lag={lag} "
                      f"ref_n={len(ref)} fast_n={len(fast)}")
        m = metrics(fast_sim(A, km, spec, rth_open, lag=False))
        if m["n"] >= 60:
            pf_at_floor.append(m["pf"])
        if (k + 1) % 32 == 0:
            print(f"  [gate] {k+1}/128")
    gmax = max(pf_at_floor)
    print(f"[gate] mismatches={mismatches} grid_max_at_floor={gmax:.4f} "
          f"(Phase A: 1.5898)")

    # truncation-equivalence: Phase A top-3 at fold-1 train end
    train_end = date.fromordinal(FOLDS[0][0].toordinal() - 1)
    import pandas as pd
    df_tr = df[[(not pd.isna(d)) and d <= train_end
                for d in df["session_date"]]]
    tables_tr = S.session_tables(df_tr)
    top3 = [
        {"threshold": 0.10, "ref": "day_open", "es_move_filter": 0.0,
         "stop_bps": 50.0, "exit_mode": "fixed_target", "target_pts": 10.0,
         "max_hold": 18},
        {"threshold": 0.10, "ref": "day_open", "es_move_filter": -0.001,
         "stop_bps": 50.0, "exit_mode": "fixed_target", "target_pts": 10.0,
         "max_hold": 18},
        {"threshold": 0.10, "ref": "day_open", "es_move_filter": -0.001,
         "stop_bps": 30.0, "exit_mode": "hold_all_day", "target_pts": 0.0,
         "max_hold": 200},
    ]
    trunc_ok = True
    for spec in top3:
        ref_tr = S.vix_spike_sim(df_tr, spec, vh, vd, tables=tables_tr)
        km = _SHARED["real_km"][(spec["threshold"], spec["ref"])]
        full = fast_sim(A, km, spec, rth_open)
        filt = trades_in(full, DATA_START, train_end)
        if ref_tr != filt:
            trunc_ok = False
            print(f"  TRUNC MISMATCH {spec}: truncated_n={len(ref_tr)} "
                  f"filtered_n={len(filt)}")
    ok = (mismatches == 0 and abs(gmax - 1.5898) < 5e-4 and trunc_ok)
    print(f"[gate] {'PASS' if ok else 'FAIL'} (trunc_ok={trunc_ok})")
    return ok


# ── permutation null ─────────────────────────────────────────────────

def build_null_km(replicate):
    """Year-matched random-day null known-min maps for all 8 combos."""
    rng = np.random.default_rng(SEED_NULL_BASE + replicate)
    sessions = _SHARED["sessions"]
    by_year = {}
    for s in sessions:
        by_year.setdefault(s.year, []).append(s)
    perms = {}
    for y in sorted(by_year):
        days = by_year[y]
        perms[y] = [days[i] for i in rng.permutation(len(days))]
    null_km = {}
    for combo in COMBOS:
        real = _SHARED["real_km"][combo]
        km = {}
        per_year = {}
        for d in sorted(real):
            per_year.setdefault(d.year, []).append(real[d])
        for y, minutes in per_year.items():
            for j, m in enumerate(minutes):
                km[perms[y][j]] = m
        null_km[combo] = km
    return null_km


def _null_task(replicate):
    km = build_null_km(replicate)
    trades_by_spec = run_grid(km, lag=False)
    wf = walk_forward(trades_by_spec, SEED_NULL_SIGNFLIP_BASE + replicate)
    return {"replicate": replicate, "pooled_n": wf["pooled_n"],
            "pooled_pf": wf["pooled_pf"], "pooled_p": wf["pooled_p"],
            "criteria": wf["criteria"], "verdict": wf["verdict"],
            "folds_gt1": wf["criteria"]["folds_gt1"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    load_shared()
    print(f"[load] shared data ready ({time.time()-t0:.0f}s)")

    if not args.skip_gate:
        if not fidelity_gate():
            print("[ABORT] fidelity gate FAILED — nothing below is run.")
            sys.exit(1)

    print("[wf] real walk-forward, no-lag...")
    real_trades = run_grid(_SHARED["real_km"], lag=False)
    wf_nolag = walk_forward(real_trades, SEED_SIGNFLIP)
    print("[wf] real walk-forward, +1-bar lag...")
    lag_trades = run_grid(_SHARED["real_km"], lag=True)
    wf_lag = walk_forward(lag_trades, SEED_SIGNFLIP)

    for name, wf in (("no-lag", wf_nolag), ("lag", wf_lag)):
        print(f"\n== {name} ==")
        for f in wf["folds"]:
            print(f"  {f['oos_start']}..{f['oos_end']} floor={f['floor']} "
                  f"train_pf={f.get('train_pf')} train_n={f.get('train_n')} "
                  f"oos n={f['oos_n']} pf={f['oos_pf']} net={f['oos_net']} "
                  f"| {f['chosen']}")
        print(f"  pooled n={wf['pooled_n']} pf={wf['pooled_pf']} "
              f"net={wf['pooled_net']} p={wf['pooled_p']:.4f} "
              f"verdict={wf['verdict']} crit={wf['criteria']}")

    print(f"\n[null] {N_NULL} replicates on {args.workers} workers...")
    t1 = time.time()
    with Pool(args.workers, initializer=load_shared) as pool:
        nulls = list(pool.imap_unordered(_null_task, range(N_NULL),
                                         chunksize=2))
    nulls.sort(key=lambda r: r["replicate"])
    print(f"[null] done in {time.time()-t1:.0f}s")

    obs_pf = wf_nolag["pooled_pf"]
    npfs = [r["pooled_pf"] for r in nulls]
    null_summary = {
        "n_replicates": N_NULL,
        "n_pass_all_criteria": sum(1 for r in nulls
                                   if r["verdict"] == "SURVIVES-PHASE-B"),
        "n_pooled_pf_ge_observed": sum(1 for p in npfs if p >= obs_pf),
        "n_pass_c1": sum(1 for r in nulls if r["criteria"]["c1_folds"]),
        "n_pass_c2": sum(1 for r in nulls if r["criteria"]["c2_pooled_pf"]),
        "n_pass_c3": sum(1 for r in nulls if r["criteria"]["c3_p"]),
        "n_pass_c4": sum(1 for r in nulls if r["criteria"]["c4_n"]),
        "n_not_decidable": sum(1 for r in nulls
                               if r["verdict"] == "NOT-DECIDABLE-AT-N"),
        "pooled_pf_min": min(npfs), "pooled_pf_max": max(npfs),
        "pooled_pf_median": float(np.median(npfs)),
        "pooled_pf_deciles": [round(float(q), 4) for q in
                              np.percentile(npfs, range(10, 100, 10))],
    }
    print(f"[null] P(pass all)={null_summary['n_pass_all_criteria']}/100  "
          f"P(pooled PF >= {obs_pf})="
          f"{null_summary['n_pooled_pf_ge_observed']}/100")
    print(f"[null] {null_summary}")

    out = {"preregistration": "lab 0b48087 phase-b-es-preregistration.md",
           "wf_nolag": wf_nolag, "wf_lag": wf_lag,
           "null_summary": null_summary, "null_replicates": nulls}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase_b_vix_spike.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\n[done] {path}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
