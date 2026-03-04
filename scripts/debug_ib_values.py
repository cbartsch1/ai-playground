#!/usr/bin/env python3
"""Debug: find days where IB low is suspiciously wrong."""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.session import SessionState, update_session

cfg = StrategyConfig()

print("Loading data...")
df = load_tos_csv(sys.argv[1], instrument="ES")
print(f"Loaded {len(df)} bars")

state = SessionState()
prev_bar = None
prev_ib_done = False

for idx, row in df.iterrows():
    bar = row.to_dict()
    bar["_time"] = idx
    update_session(state, bar, prev_bar, cfg)

    # Print IB values when IB just completed
    if state.ib_done and not prev_ib_done:
        # Check if IB low seems wrong (more than 50% below current price)
        if not math.isnan(state.ib_low) and state.ib_low < bar["close"] * 0.5:
            print(f"  *** BAD IB LOW: {idx} ib_high={state.ib_high:.2f} ib_low={state.ib_low:.2f} "
                  f"ib_mid={state.ib_mid:.2f} ib_range={state.ib_range:.2f} close={bar['close']:.2f}")
        elif state.ib_range > 200:
            print(f"  *** HUGE IB RANGE: {idx} ib_high={state.ib_high:.2f} ib_low={state.ib_low:.2f} "
                  f"ib_mid={state.ib_mid:.2f} ib_range={state.ib_range:.2f}")

    prev_ib_done = state.ib_done
    prev_bar = bar

# Also check for bars with very low prices
print("\nBars with suspiciously low prices (low < 1000):")
low_bars = df[df["low"] < 1000]
if len(low_bars) > 0:
    print(f"  Found {len(low_bars)} bars")
    for idx, row in low_bars.head(20).iterrows():
        print(f"  {idx} O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} "
              f"V={row['volume']:.0f} is_rth={row['is_rth']} is_ib={row['is_ib_period']}")
else:
    print("  None found")
