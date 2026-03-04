#!/usr/bin/env python3
"""Debug: dump first N REJ trades to see what's happening."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics

cfg = StrategyConfig()
cfg.direction_filter = "short"
cfg.pct_stop_mode = True
cfg.pct_stop_bps = 30.0
cfg.skip_friday = True
cfg.blackout_start = 1200
cfg.blackout_end = 1300
cfg.use_va_fade = False

# Best combo from sweep
cfg.use_ib_reject = True
cfg.rej_trigger = "any"
cfg.rej_target = "ib_mid"
cfg.rej_zone_pts = 5.0
cfg.rej_stop_buffer = 3.0
cfg.rej_require_tema = False
cfg.max_rej_trades = 5

print("Loading data...")
df = load_tos_csv(sys.argv[1], instrument="ES")
print(f"Loaded {len(df)} bars")

trades = run_backtest(df, cfg)
rej = [t for t in trades if t.setup == "REJ"]
ib = [t for t in trades if t.setup == "IB"]

print(f"\nTotal: {len(trades)} trades ({len(rej)} REJ, {len(ib)} IB)")

# Exit reason breakdown
reasons = {}
for t in rej:
    reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
print(f"\nREJ exit reasons:")
for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
    pnl = sum(t.pnl_dollar for t in rej if t.exit_reason == reason)
    avg = pnl / count if count > 0 else 0
    avg_pts = sum(t.pnl_pts for t in rej if t.exit_reason == reason) / count
    print(f"  {reason:<10} {count:>5} trades  ${pnl:>+12,.0f}  avg ${avg:>+8,.0f}  avg {avg_pts:>+6.1f}pts")

# Print first 30 REJ trades
print(f"\nFirst 30 REJ trades:")
print(f"{'Entry Time':<22} {'Entry$':>8} {'Stop':>8} {'Target':>8} {'Exit$':>8} {'Reason':<8} {'PnL pts':>8} {'PnL $':>10}")
print("-" * 95)
for t in rej[:30]:
    print(f"{str(t.entry_time):<22} {t.entry_price:>8.2f} {t.stop:>8.2f} {t.target:>8.2f} "
          f"{t.exit_price:>8.2f} {t.exit_reason:<8} {t.pnl_pts:>+8.2f} ${t.pnl_dollar:>+9,.0f}")

# Distribution of PnL
import numpy as np
pnls = [t.pnl_dollar for t in rej]
pts = [t.pnl_pts for t in rej]
print(f"\nREJ P&L distribution:")
print(f"  Mean:   ${np.mean(pnls):>+8,.0f} ({np.mean(pts):>+6.1f} pts)")
print(f"  Median: ${np.median(pnls):>+8,.0f} ({np.median(pts):>+6.1f} pts)")
print(f"  Std:    ${np.std(pnls):>+8,.0f} ({np.std(pts):>+6.1f} pts)")
print(f"  Min:    ${np.min(pnls):>+8,.0f} ({np.min(pts):>+6.1f} pts)")
print(f"  Max:    ${np.max(pnls):>+8,.0f} ({np.max(pts):>+6.1f} pts)")

# Biggest winners
print(f"\nTop 10 winners:")
winners = sorted(rej, key=lambda t: t.pnl_dollar, reverse=True)[:10]
for t in winners:
    print(f"  {str(t.entry_time):<22} entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
          f"target={t.target:.2f} stop={t.stop:.2f} reason={t.exit_reason} "
          f"pnl={t.pnl_pts:+.1f}pts ${t.pnl_dollar:+,.0f}")
