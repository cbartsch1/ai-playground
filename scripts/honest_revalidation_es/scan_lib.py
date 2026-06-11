"""Shared pieces for the ringer-es kill scans: metrics + engine base configs.

Trade tuple format used everywhere: (setup, date_iso, entry_hhmm, pnl_dollar).

Engine base configs are LOCAL COPIES of the official scripts' configs
(os_ms_official.make_config, run_nexus_long_5yr.make_config,
amt_breakout.get_config @ ed86b73, ib_comparison.make_sentinel_config) —
NOT imported, because (1) some of those scripts execute a full backtest at
import time, and (2) the working tree carries uncommitted WIP in
amt_breakout.py (use_ib_break=False) that must not infect the scan.
"""

from backtester.config import StrategyConfig

ALL_SETUP_FLAGS = ["use_ib_break", "use_va_fade", "use_eighty",
                   "use_tema_cross", "use_level_reject",
                   "use_level_reject_long", "use_ib_reject", "use_var",
                   "use_ptf", "use_fa", "use_ms", "use_os"]


def _blank_config():
    cfg = StrategyConfig()
    for f in ALL_SETUP_FLAGS:
        setattr(cfg, f, False)
    return cfg


# ── engine base configs ──────────────────────────────────────────────

def nexus_short_base(spec):
    """MS+OS short-only (os_ms_official lineage / TV nexus.py)."""
    cfg = _blank_config()
    cfg.direction_filter = "short"
    cfg.use_ms = True
    cfg.ms_zone_pts = spec["ms_zone_pts"]
    cfg.ms_stop_buffer = spec["ms_stop_buffer"]
    cfg.ms_min_target_pts = spec["ms_min_target_pts"]
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
    cfg.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both",
                               "MS_pVAH": "short"}
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = spec["os_stop_buffer"]
    cfg.os_max_risk = 25.0
    cfg.os_target_mode = "cascade"
    cfg.os_min_target_pts = 3.0
    cfg.os_min_rr = 0.5
    cfg.os_require_on_sweep = True
    cfg.os_require_ma = False
    cfg.max_os_trades = 1
    cfg.os_min_gap = spec["os_gap"][0]
    cfg.os_max_gap = spec["os_gap"][1]
    cfg.os_entry_window = 1
    return cfg


def bedrock_base(spec):
    """MS+OS long (nexus_long / TV bedrock.py). VIX filter applied POST-HOC
    by the parent (causal prior-day close) — engine axes only here."""
    cfg = _blank_config()
    cfg.direction_filter = "long"
    cfg.use_ms = True
    cfg.ms_zone_pts = spec["ms_zone_pts"]
    cfg.ms_stop_buffer = spec["ms_stop_buffer"]
    cfg.ms_min_target_pts = spec["ms_min_target_pts"]
    cfg.ms_min_rr = 0.3
    cfg.ms_max_risk = 25.0
    cfg.ms_ma_type = "sma"
    cfg.ms_ma_confirm_bars = 0
    cfg.max_ms_trades = 8
    cfg.ms_use_vp_levels = True
    cfg.ms_use_prev_va = True
    cfg.ms_use_on_levels = True
    cfg.ms_use_ib_levels = False
    cfg.ms_use_dev_va = True
    cfg.ms_use_poc = True
    cfg.ms_level_directions = {"MS_ONH": "both", "MS_ONL": "both",
                               "MS_pVAH": "short", "MS_dVAL": "long",
                               "MS_dPOC": "long"}
    cfg.use_os = True
    cfg.os_stop_mode = "on_extreme"
    cfg.os_stop_buffer = spec["os_stop_buffer"]
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


def amt_v8_base(spec):
    """AMT-TEMA v8 IB breakout short (restored baseline ed86b73)."""
    cfg = _blank_config()
    cfg.direction_filter = "short"
    cfg.pct_stop_mode = True
    cfg.pct_stop_bps = spec["pct_stop_bps"]
    cfg.skip_friday = spec["skip_friday"]
    if spec["noon_blackout"]:
        cfg.blackout_start = 1200
        cfg.blackout_end = 1300
    cfg.use_ib_break = True
    cfg.tp_atr_mult = 0.0
    cfg.max_ib_trades = 2
    cfg.min_ib_range = spec["min_ib_range"]
    cfg.max_ib_range = spec["max_ib_range"]
    cfg.ib_stop_type = spec["ib_stop_type"]
    cfg.ib_min_target = 10.0
    cfg.use_trend_filter = True
    return cfg


def lvl_v13_base(spec):
    """Sentinel's ES parent: ONH level rejection short (LVL v13)."""
    cfg = _blank_config()
    cfg.direction_filter = "short"
    cfg.use_level_reject = True
    cfg.lvl_trigger = "any"
    cfg.lvl_zone_pts = spec["lvl_zone_pts"]
    cfg.lvl_stop_buffer = spec["lvl_stop_buffer"]
    cfg.lvl_require_tema = spec["lvl_require_tema"]
    cfg.lvl_ma_filter = "tema"
    cfg.max_lvl_trades = 4
    cfg.lvl_ibh_wide_only = True
    cfg.lvl_max_tests = 3
    cfg.lvl_broken_bars = 2
    cfg.lvl_own_filters = True
    cfg.lvl_min_rr = spec["lvl_min_rr"]
    cfg.lvl_min_target_pts = 5.0
    cfg.lvl_target_skip = spec["lvl_target_skip"]
    cfg.lvl_enabled_levels = tuple(spec["levels"])
    cfg.pct_stop_mode = False
    return cfg


ENGINE_BASES = {
    "nexus_short": nexus_short_base,
    "bedrock": bedrock_base,
    "amt_v8": amt_v8_base,
    "lvl_v13": lvl_v13_base,
}


# ── bedrock post-hoc causal VIX filter ───────────────────────────────

def apply_bedrock_filter(trades, vix_daily, vix_variant, cutoff):
    """Causal version of run_nexus_long_5yr.apply_vix_filter.

    Uses PRIOR-day VIX close (vix_daily[date]['prev_close']) — the original
    used the SAME-day close (look-ahead). vix_variant: None or
    (ms_min, os_max). cutoff: entry hhmm cutoff.
    """
    import datetime
    out = []
    for setup, date_iso, hhmm, pnl in trades:
        if hhmm >= cutoff:
            continue
        if vix_variant is None:
            out.append((setup, date_iso, hhmm, pnl))
            continue
        d = datetime.date.fromisoformat(date_iso)
        v = vix_daily.get(d)
        ref = v["prev_close"] if v else None
        if ref is None:
            continue
        ms_min, os_max = vix_variant
        if setup.startswith("MS_") and ref > ms_min:
            out.append((setup, date_iso, hhmm, pnl))
        elif setup.startswith("OS_") and ref <= os_max:
            out.append((setup, date_iso, hhmm, pnl))
    return out


def apply_bedrock_filter_leaky(trades, vix_daily, vix_variant, cutoff):
    """SAME-day close variant — diagnostics ONLY (quantifies the original
    apply_vix_filter leak on the deployed config). Never used for verdicts."""
    import datetime
    out = []
    for setup, date_iso, hhmm, pnl in trades:
        if hhmm >= cutoff:
            continue
        if vix_variant is None:
            out.append((setup, date_iso, hhmm, pnl))
            continue
        d = datetime.date.fromisoformat(date_iso)
        v = vix_daily.get(d)
        ref = v["close"] if v else None
        if ref is None:
            continue
        ms_min, os_max = vix_variant
        if setup.startswith("MS_") and ref > ms_min:
            out.append((setup, date_iso, hhmm, pnl))
        elif setup.startswith("OS_") and ref <= os_max:
            out.append((setup, date_iso, hhmm, pnl))
    return out


# ── metrics ──────────────────────────────────────────────────────────

def metrics(trades):
    """Compact metrics from trade tuples (setup, date_iso, hhmm, pnl)."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "pf": 0.0, "wr": 0.0, "net": 0.0, "max_dd": 0.0,
                "per_year": {}}
    pnls = [t[3] for t in trades]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = (gp / gl) if gl > 0 else float("inf")
    wr = 100.0 * sum(1 for p in pnls if p > 0) / n
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    per_year = {}
    for setup, date_iso, hhmm, pnl in trades:
        y = date_iso[:4]
        d = per_year.setdefault(y, {"n": 0, "gp": 0.0, "gl": 0.0})
        d["n"] += 1
        if pnl > 0:
            d["gp"] += pnl
        else:
            d["gl"] += abs(pnl)
    py = {}
    for y, d in sorted(per_year.items()):
        ypf = (d["gp"] / d["gl"]) if d["gl"] > 0 else float("inf")
        py[y] = {"n": d["n"], "pf": round(ypf, 2) if ypf != float("inf") else 99.0}
    return {"n": n, "pf": round(pf, 4) if pf != float("inf") else 99.0,
            "wr": round(wr, 2), "net": round(sum(pnls), 2),
            "max_dd": round(dd, 2), "per_year": py}
