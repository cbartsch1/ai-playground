#!/usr/bin/env python3
"""Ringer ES kill-scan runner. All data through the Lockbox (DEV only).

Usage:
    python scripts/honest_revalidation_es/run_scan.py [--concepts a,b,...]
                                                      [--workers N]

Writes results/<concept>.json with every config's metrics, the deployed row,
and +1-bar-lag for the top-3 of the custom-harness concepts.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.honest_revalidation_es.lockbox import Lockbox
from scripts.honest_revalidation_es import grids as G
from scripts.honest_revalidation_es.scan_lib import (
    ENGINE_BASES, metrics, apply_bedrock_filter, apply_bedrock_filter_leaky,
)
from scripts.honest_revalidation_es import sims as S

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")

_DF = None


def _init_worker():
    global _DF
    _DF = Lockbox().load_es_5m()


def _run_engine_task(task):
    concept, spec = task
    from backtester.engine import run_backtest
    cfg = ENGINE_BASES[concept](spec)
    trades = run_backtest(_DF.copy(), cfg)
    compact = [(t.setup, str(t.entry_time.date()),
                int(t.entry_time.hour * 100 + t.entry_time.minute),
                round(t.pnl_dollar, 2)) for t in trades]
    return concept, spec, compact


def _spec_key(spec):
    return json.dumps({k: list(v) if isinstance(v, tuple) else v
                       for k, v in sorted(spec.items())})


def scan_engine_concepts(concepts, workers):
    """Run engine-based concepts (nexus_short, bedrock, amt_v8, lvl_v13)."""
    tasks = []
    if "nexus_short" in concepts:
        tasks += [("nexus_short", s) for s in G.nexus_short_grid()]
    if "bedrock" in concepts:
        tasks += [("bedrock", s) for s in G.bedrock_engine_grid()]
    if "amt_v8" in concepts:
        tasks += [("amt_v8", s) for s in G.amt_v8_grid()]
    if "lvl_v13" in concepts:
        tasks += [("lvl_v13", s) for s in G.lvl_v13_grid()]
    if not tasks:
        return {}

    print(f"[engine] {len(tasks)} engine runs on {workers} workers...")
    t0 = time.time()
    raw = {}
    with Pool(workers, initializer=_init_worker) as pool:
        for i, (concept, spec, trades) in enumerate(
                pool.imap_unordered(_run_engine_task, tasks, chunksize=2)):
            raw.setdefault(concept, []).append((spec, trades))
            if (i + 1) % 50 == 0:
                print(f"  [engine] {i+1}/{len(tasks)} "
                      f"({time.time()-t0:.0f}s)")
    print(f"[engine] done in {time.time()-t0:.0f}s")

    results = {}
    box = Lockbox()
    vix_daily = box.load_vix_daily()

    for concept, runs in raw.items():
        rows = []
        if concept == "bedrock":
            for spec, trades in runs:
                is_deployed_engine = all(
                    spec[k] == G.BEDROCK_DEPLOYED[k] for k in spec)
                for vix in G.BEDROCK_VIX_VARIANTS:
                    for cutoff in G.BEDROCK_CUTOFFS:
                        full = dict(spec, vix=vix, cutoff=cutoff)
                        filt = apply_bedrock_filter(trades, vix_daily, vix,
                                                    cutoff)
                        rows.append({"spec": _spec_key(full),
                                     **metrics(filt)})
                        # leak diagnostic on the deployed config only
                        if (is_deployed_engine
                                and vix == G.BEDROCK_DEPLOYED["vix"]
                                and cutoff == G.BEDROCK_DEPLOYED["cutoff"]):
                            leaky = apply_bedrock_filter_leaky(
                                trades, vix_daily, vix, cutoff)
                            rows.append({"spec": "DIAG_LEAKY_SAMEDAY_VIX:"
                                         + _spec_key(full),
                                         **metrics(leaky)})
            deployed_key = _spec_key(G.BEDROCK_DEPLOYED)
        else:
            for spec, trades in runs:
                rows.append({"spec": _spec_key(spec), **metrics(trades)})
            deployed_key = _spec_key({
                "nexus_short": G.NEXUS_SHORT_DEPLOYED,
                "amt_v8": G.AMT_V8_DEPLOYED,
                "lvl_v13": G.LVL_V13_DEPLOYED}[concept])
        results[concept] = {"rows": rows, "deployed_key": deployed_key}
    return results


def scan_custom_concepts(concepts):
    """trend_cont / vector / vix_spike via custom sims (single process)."""
    box = Lockbox()
    df = box.load_es_5m()
    tables = S.session_tables(df)
    results = {}

    if "trend_cont" in concepts:
        print("[trend_cont] scanning...")
        rows = []
        grid = G.trend_cont_grid()
        for spec in grid:
            rows.append({"spec": _spec_key(spec),
                         **metrics(S.trend_cont_sim(df, spec))})
        results["trend_cont"] = {
            "rows": rows, "deployed_key": _spec_key(G.TREND_CONT_DEPLOYED),
            "lag_fn": lambda spec: metrics(S.trend_cont_sim(df, spec,
                                                            lag=True))}

    if "vector" in concepts:
        print("[vector] scanning...")
        rows = []
        for spec in G.vector_grid():
            rows.append({"spec": _spec_key(spec),
                         **metrics(S.vector_sim(df, spec, tables=tables))})
        results["vector"] = {
            "rows": rows, "deployed_key": _spec_key(G.VECTOR_DEPLOYED),
            "lag_fn": lambda spec: metrics(S.vector_sim(df, spec,
                                                        tables=tables,
                                                        lag=True))}

    if "vix_spike" in concepts:
        print("[vix_spike] scanning (causal hourly-intraday variant)...")
        vix_hourly = box.load_vix_hourly()
        vix_daily = box.load_vix_daily()
        rows = []
        for spec in G.vix_spike_grid():
            rows.append({"spec": _spec_key(spec),
                         **metrics(S.vix_spike_sim(df, spec, vix_hourly,
                                                   vix_daily,
                                                   tables=tables))})
        results["vix_spike"] = {
            "rows": rows,
            "deployed_key": _spec_key(G.VIX_SPIKE_DEPLOYED_ADJ),
            "lag_fn": lambda spec: metrics(S.vix_spike_sim(
                df, spec, vix_hourly, vix_daily, tables=tables, lag=True))}

    return results


def finalize(concept, payload):
    """Add floor stats, top-10, lag rows; write JSON."""
    floor = G.FLOORS[concept]
    rows = [r for r in payload["rows"]
            if not r["spec"].startswith("DIAG_")]
    diag = [r for r in payload["rows"] if r["spec"].startswith("DIAG_")]
    at_floor = [r for r in rows if r["n"] >= floor]
    any_n = [r for r in rows if r["n"] > 0]
    top10 = sorted(at_floor, key=lambda r: -r["pf"])[:10]
    above_kill_below_floor = sorted(
        [r for r in rows if r["n"] < floor and r["n"] >= 10
         and r["pf"] >= G.KILL_PF], key=lambda r: -r["pf"])[:10]

    deployed = next((r for r in rows
                     if r["spec"] == payload["deployed_key"]), None)

    lag_rows = []
    if "lag_fn" in payload:
        for r in sorted(at_floor, key=lambda r: -r["pf"])[:3]:
            spec = {k: tuple(v) if isinstance(v, list) else v
                    for k, v in json.loads(r["spec"]).items()}
            lag_rows.append({"spec": r["spec"], "causal_pf": r["pf"],
                             "causal_n": r["n"], **{f"lag_{k}": v for k, v
                             in payload["lag_fn"](spec).items()
                             if k in ("n", "pf", "net")}})

    pfs = sorted(r["pf"] for r in any_n)
    out = {
        "concept": concept,
        "floor": floor,
        "configs_total": len(rows),
        "configs_meeting_floor": len(at_floor),
        "grid_max_pf_any_n": max((r["pf"] for r in any_n), default=0),
        "grid_max_pf_at_floor": max((r["pf"] for r in at_floor), default=0),
        "median_pf": pfs[len(pfs) // 2] if pfs else 0,
        "share_pf_gt_1": (sum(1 for p in pfs if p > 1.0) / len(pfs)
                          if pfs else 0),
        "n_pf_ge_kill_at_floor": sum(1 for r in at_floor
                                     if r["pf"] >= G.KILL_PF),
        "verdict": ("DEAD" if not any(r["pf"] >= G.KILL_PF
                                      for r in at_floor) else "SURVIVOR"),
        "deployed": deployed,
        "top10_at_floor": top10,
        "above_kill_below_floor": above_kill_below_floor,
        "lag_top3": lag_rows,
        "diagnostics": diag,
        "all_rows": rows,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{concept}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"[{concept}] verdict={out['verdict']} configs={len(rows)} "
          f"at_floor={len(at_floor)} max_pf_floor="
          f"{out['grid_max_pf_at_floor']:.2f} -> {path}")
    return out


ENGINE_CONCEPTS = ["nexus_short", "bedrock", "amt_v8", "lvl_v13"]
CUSTOM_CONCEPTS = ["trend_cont", "vector", "vix_spike"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=",".join(ENGINE_CONCEPTS
                                                   + CUSTOM_CONCEPTS))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    concepts = args.concepts.split(",")

    t0 = time.time()
    custom = scan_custom_concepts([c for c in concepts
                                   if c in CUSTOM_CONCEPTS])
    for c, payload in custom.items():
        finalize(c, payload)

    engine = scan_engine_concepts([c for c in concepts
                                   if c in ENGINE_CONCEPTS], args.workers)
    for c, payload in engine.items():
        finalize(c, payload)

    print(f"TOTAL scan time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
