"""Pre-registered grids — MUST match ringer-es-protocol.md (lab dadd288).

Every grid contains the deployed/deployed-adjacent config; DEPLOYED maps the
exact deployed point for the deployed-config row in results.
"""

from itertools import product

FLOORS = {"nexus_short": 150, "bedrock": 150, "trend_cont": 150,
          "vector": 100, "amt_v8": 100, "lvl_v13": 100, "vix_spike": 60}

KILL_PF = 1.3


def nexus_short_grid():
    out = []
    for zone, buf, tgt, osb, gap in product(
            [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [6.0, 8.0, 10.0],
            [4.0, 5.0, 6.0], [(3.0, 20.0), (2.0, 30.0)]):
        out.append({"ms_zone_pts": zone, "ms_stop_buffer": buf,
                    "ms_min_target_pts": tgt, "os_stop_buffer": osb,
                    "os_gap": gap})
    return out


NEXUS_SHORT_DEPLOYED = {"ms_zone_pts": 3.0, "ms_stop_buffer": 4.0,
                        "ms_min_target_pts": 8.0, "os_stop_buffer": 5.0,
                        "os_gap": (3.0, 20.0)}


def bedrock_engine_grid():
    """Engine axes only (36); VIX/cutoff variants are applied post-hoc."""
    out = []
    for zone, buf, tgt, osb in product(
            [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [8.0, 10.0], [4.0, 5.0]):
        out.append({"ms_zone_pts": zone, "ms_stop_buffer": buf,
                    "ms_min_target_pts": tgt, "os_stop_buffer": osb})
    return out


BEDROCK_VIX_VARIANTS = [None, (18.0, 17.0), (16.0, 19.0), (20.0, 15.0)]
BEDROCK_CUTOFFS = [1300, 1500]
# Deployed (AR-adopted): target 10, os_buf 4, VIX (18,17), cutoff 1475≈1500
BEDROCK_DEPLOYED = {"ms_zone_pts": 3.0, "ms_stop_buffer": 4.0,
                    "ms_min_target_pts": 10.0, "os_stop_buffer": 4.0,
                    "vix": (18.0, 17.0), "cutoff": 1500}


def trend_cont_grid():
    out = []
    for end, stop, hold, surge, fri in product(
            [1100, 1300, 1500], [20.0, 30.0, 40.0], [6, 12],
            [0.0, 1.2], [True, False]):
        for exit_spec in [
                {"exit_mode": "fixed_target", "target_pts": 15.0},
                {"exit_mode": "fixed_target", "target_pts": 25.0},
                {"exit_mode": "fixed_target", "target_pts": 35.0},
                {"exit_mode": "green_bar", "target_pts": 0.0}]:
            out.append({"entry_end": end, "stop_bps": stop,
                        "max_hold": hold, "vol_surge": surge,
                        "skip_friday": fri, **exit_spec})
    return out


TREND_CONT_DEPLOYED = {"entry_end": 1300, "stop_bps": 30.0, "max_hold": 6,
                       "vol_surge": 1.2, "skip_friday": True,
                       "exit_mode": "fixed_target", "target_pts": 25.0}


def vector_grid():
    out = []
    for ema, gap, stop, tgt, mpd in product(
            [(8, 24), (9, 21), (12, 26)], [5.0, 10.0, 15.0, 20.0, 25.0],
            [20.0, 30.0, 40.0],
            [("onl", 0.0), ("fixed", 10.0), ("fixed", 20.0)], [1, 2]):
        out.append({"ema_fast": ema[0], "ema_slow": ema[1],
                    "gap_threshold": gap, "stop_bps": stop,
                    "target_mode": tgt[0], "target_pts": tgt[1],
                    "max_per_day": mpd})
    return out


VECTOR_DEPLOYED = {"ema_fast": 9, "ema_slow": 21, "gap_threshold": 15.0,
                   "stop_bps": 20.0, "target_mode": "onl", "target_pts": 0.0,
                   "max_per_day": 2}


def amt_v8_grid():
    out = []
    for mn, mx, stop, st, fri, blk in product(
            [3.0, 8.0, 15.0], [60.0, 80.0, 120.0], [20.0, 30.0, 40.0],
            ["IB Mid", "IB Edge"], [True, False], [True, False]):
        out.append({"min_ib_range": mn, "max_ib_range": mx,
                    "pct_stop_bps": stop, "ib_stop_type": st,
                    "skip_friday": fri, "noon_blackout": blk})
    return out


AMT_V8_DEPLOYED = {"min_ib_range": 8.0, "max_ib_range": 80.0,
                   "pct_stop_bps": 30.0, "ib_stop_type": "IB Mid",
                   "skip_friday": True, "noon_blackout": True}


def lvl_v13_grid():
    out = []
    for zone, buf, skip, rr, tema, lv in product(
            [3.0, 5.0, 7.0], [5.0, 7.0, 9.0], [0, 1, 2], [0.0, 0.5],
            [True, False], [("ONH",), ("ONH", "PDH")]):
        out.append({"lvl_zone_pts": zone, "lvl_stop_buffer": buf,
                    "lvl_target_skip": skip, "lvl_min_rr": rr,
                    "lvl_require_tema": tema, "levels": lv})
    return out


LVL_V13_DEPLOYED = {"lvl_zone_pts": 5.0, "lvl_stop_buffer": 7.0,
                    "lvl_target_skip": 2, "lvl_min_rr": 0.5,
                    "lvl_require_tema": True, "levels": ("ONH",)}


def vix_spike_grid():
    out = []
    for thr, ref, mv, stop, exit_spec in product(
            [0.03, 0.05, 0.07, 0.10], ["prev_close", "day_open"],
            [0.0, -0.001], [30.0, 50.0],
            [{"exit_mode": "fixed_target", "target_pts": 10.0, "max_hold": 18},
             {"exit_mode": "fixed_target", "target_pts": 20.0, "max_hold": 18},
             {"exit_mode": "green_bar", "target_pts": 0.0, "max_hold": 18},
             {"exit_mode": "hold_all_day", "target_pts": 0.0, "max_hold": 200}]):
        out.append({"threshold": thr, "ref": ref, "es_move_filter": mv,
                    "stop_bps": stop, **exit_spec})
    return out


VIX_SPIKE_DEPLOYED_ADJ = {"threshold": 0.05, "ref": "prev_close",
                          "es_move_filter": -0.001, "stop_bps": 50.0,
                          "exit_mode": "fixed_target", "target_pts": 20.0,
                          "max_hold": 18}
