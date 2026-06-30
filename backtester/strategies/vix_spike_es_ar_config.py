"""VIX Spike ES — AutoResearch v3 parameter config.

Parameters below are modified by AutoResearch during optimization.
The cfg_attr pattern matches lines like: cfg.param_name = value

DO NOT rename the 'cfg' variable or change the assignment format.

Enhanced Apr 8, 2026: threshold 7%→5%, stop 30→50bps, fixed target 20pt,
ES move filter -0.1%, max hold 90min (18 bars).
"""

from types import SimpleNamespace

cfg = SimpleNamespace()

# ── Tunable mechanical parameters ──
cfg.spike_threshold = 0.05
cfg.stop_bps = 50
cfg.target_pts = 20.0
cfg.max_hold_bars = 18
cfg.es_move_filter = -0.001
cfg.entry_start = 935
cfg.entry_end = 1500
