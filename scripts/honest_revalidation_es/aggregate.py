#!/usr/bin/env python3
"""Render markdown tables from results/<concept>.json for the results doc."""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")

ORDER = ["nexus_short", "bedrock", "trend_cont", "vector", "amt_v8",
         "lvl_v13", "vix_spike"]


def fmt_spec(spec_str):
    s = json.loads(spec_str.replace("DIAG_LEAKY_SAMEDAY_VIX:", ""))
    return ",".join(f"{k}={v}" for k, v in sorted(s.items()))


def fmt_py(py):
    return " / ".join(f"{y}:{d['pf']:.2f}(n{d['n']})"
                      for y, d in sorted(py.items()))


def row_line(r):
    return (f"| {fmt_spec(r['spec'])} | {r['n']} | {r['pf']:.2f} | "
            f"{r['wr']:.1f}% | ${r['net']:,.0f} | ${r['max_dd']:,.0f} | "
            f"{fmt_py(r['per_year'])} |")


def main():
    for c in ORDER:
        p = os.path.join(RESULTS_DIR, f"{c}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        print(f"\n## {c} — {r['verdict']}\n")
        print(f"- Grid: {r['configs_total']} configs (pre-registered); "
              f"floor n >= {r['floor']}")
        print(f"- Causal PF distribution (DEV): max {r['grid_max_pf_any_n']:.2f} "
              f"(any n), max at floor {r['grid_max_pf_at_floor']:.2f}, "
              f"median {r['median_pf']:.2f}, share PF>1.0: "
              f"{100*r['share_pf_gt_1']:.1f}%, configs PF>=1.3 at floor: "
              f"{r['n_pf_ge_kill_at_floor']}")
        print(f"- Configs meeting floor: {r['configs_meeting_floor']}"
              f"/{r['configs_total']}")
        d = r.get("deployed")
        if d:
            print(f"- Deployed config: n={d['n']}, PF {d['pf']:.2f}, "
                  f"net ${d['net']:,.0f}; per-year {fmt_py(d['per_year'])}")
        print("\nTop-10 floor-meeting configs by causal PF "
              "(selection-biased by construction):\n")
        print("| Config | n | PF | WR | Net | MaxDD | per-year PF |")
        print("|---|---|---|---|---|---|---|")
        for t in r["top10_at_floor"]:
            print(row_line(t))
        if d:
            print(row_line(d).replace(" |", " **(deployed)** |", 1))
        if r.get("above_kill_below_floor"):
            print("\nConfigs PF >= 1.3 BELOW the floor:\n")
            print("| Config | n | PF | Net | per-year PF |")
            print("|---|---|---|---|---|")
            for t in r["above_kill_below_floor"]:
                print(f"| {fmt_spec(t['spec'])} | {t['n']} | {t['pf']:.2f} "
                      f"| ${t['net']:,.0f} | {fmt_py(t['per_year'])} |")
        if r.get("lag_top3"):
            print("\nLag sensitivity (top-3, entry one 5m bar later):\n")
            print("| Config | causal PF (n) | +1-bar-lag PF (n) | lag net |")
            print("|---|---|---|---|")
            for l in r["lag_top3"]:
                print(f"| {fmt_spec(l['spec'])} | {l['causal_pf']:.2f} "
                      f"({l['causal_n']}) | {l['lag_pf']:.2f} ({l['lag_n']}) "
                      f"| ${l['lag_net']:,.0f} |")
        if r.get("diagnostics"):
            print("\nDiagnostics:\n")
            for t in r["diagnostics"]:
                print(f"- {t['spec'].split(':')[0]}: n={t['n']}, "
                      f"PF {t['pf']:.2f}, net ${t['net']:,.0f}")


if __name__ == "__main__":
    main()
