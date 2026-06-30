"""Nexus Long — AutoResearch v3 parameter config.

Parameters below are modified by AutoResearch during optimization.
The cfg_attr pattern matches lines like: cfg.param_name = value

DO NOT rename the 'cfg' variable or change the assignment format.

AR-OPTIMIZED (Apr 3, 2026): 71 experiments, 3 improvements, OOS PF 1.608→1.931
  Verification: GREEN (0 flags), sensitivity stable, no overfitting
Baseline: 255t, PF 1.589, +$27,962, p=0.002, OOS PF 1.608, WF 1.04 PASS
Optimized: 182t, PF 1.623, +$22,602, p=0.007, OOS PF 1.931, WF 1.61 PASS
"""
from types import SimpleNamespace

cfg = SimpleNamespace()

# ── MS — structural level params ──
cfg.ms_zone_pts = 3.0
cfg.ms_stop_buffer = 4
cfg.ms_min_target_pts = 10
cfg.ms_min_rr = 0.3

# ── OS — gap-down fade params ──
cfg.os_stop_buffer = 4

# ── VIX filter — inverted per setup ──
cfg.vix_ms_thresh = 18
cfg.vix_os_thresh = 17

# ── Entry time cutoff (ET, HHMM) ──
cfg.entry_cutoff = 1475
