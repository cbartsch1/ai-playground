#!/usr/bin/env python3
"""Generate realistic sample ES 5m data for backtester testing.

Creates ~2 weeks of data with RTH+ETH, realistic price action,
IB ranges, trend days, and rotational days.
"""

import sys
import os
import random
import math

import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)


def generate_es_5m_data(days=15, start_price=5950.0):
    """Generate realistic ES 5m OHLCV data."""
    rows = []
    price = start_price

    # Start on a Monday
    base_date = pd.Timestamp("2025-11-03", tz="America/New_York")

    for day_idx in range(days):
        current_date = base_date + pd.Timedelta(days=day_idx)

        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # Decide day character
        day_type = random.choice(["trend_up", "trend_down", "rotational", "rotational", "rotational"])

        # ETH: 18:00 previous day to 09:25 (simplified: just 6:00-9:25 for brevity)
        # Generate ETH bars
        for hour in range(6, 9):
            for minute in range(0, 60, 5):
                et_time = hour * 100 + minute
                noise = np.random.randn() * 1.5
                price += noise
                price = round(price * 4) / 4  # Snap to tick
                bar_high = round((price + abs(np.random.randn()) * 1.0) * 4) / 4
                bar_low = round((price - abs(np.random.randn()) * 1.0) * 4) / 4
                bar_open = round((price + np.random.randn() * 0.5) * 4) / 4
                vol = random.randint(500, 3000)
                dt = current_date.replace(hour=hour, minute=minute, second=0)
                rows.append({
                    "Date/Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "Open": bar_open, "High": max(bar_high, bar_open, price),
                    "Low": min(bar_low, bar_open, price), "Close": price,
                    "Volume": vol
                })

        # 9:25-9:30 bars
        for minute in [25]:
            noise = np.random.randn() * 1.0
            price += noise
            price = round(price * 4) / 4
            dt = current_date.replace(hour=9, minute=minute, second=0)
            rows.append({
                "Date/Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Open": price, "High": price + 0.5, "Low": price - 0.5,
                "Close": price, "Volume": random.randint(2000, 8000)
            })

        # RTH: 9:30-16:00
        rth_open = price
        ib_drift = 0

        # IB period (9:30-10:30): set IB range
        if day_type == "trend_up":
            ib_target_range = random.uniform(10, 25)  # Narrowish IB for trend
            ib_drift = ib_target_range * 0.3
        elif day_type == "trend_down":
            ib_target_range = random.uniform(10, 25)
            ib_drift = -ib_target_range * 0.3
        else:
            ib_target_range = random.uniform(12, 35)  # Wider IB for rotational
            ib_drift = 0

        ib_bars = 12  # 60 min / 5 min
        ib_high = price
        ib_low = price

        for bar_idx in range(ib_bars):
            minute = 30 + bar_idx * 5
            hour = 9 + minute // 60
            minute = minute % 60

            # IB: oscillate within range with slight drift
            noise = np.random.randn() * (ib_target_range / 8)
            drift = ib_drift / ib_bars
            price += noise + drift
            price = round(price * 4) / 4

            spread = abs(np.random.randn()) * 2 + 1
            bar_high = round((price + spread) * 4) / 4
            bar_low = round((price - spread) * 4) / 4
            bar_open = round((price + np.random.randn() * 1) * 4) / 4

            ib_high = max(ib_high, bar_high)
            ib_low = min(ib_low, bar_low)

            dt = current_date.replace(hour=hour, minute=minute, second=0)
            vol = random.randint(5000, 25000)
            rows.append({
                "Date/Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Open": bar_open, "High": max(bar_high, bar_open, price),
                "Low": min(bar_low, bar_open, price), "Close": price,
                "Volume": vol
            })

        # Post-IB: 10:30-16:00
        post_ib_bars = 66  # 330 min / 5 min

        for bar_idx in range(post_ib_bars):
            total_min = 630 + bar_idx * 5  # 630 = 10:30
            hour = total_min // 60
            minute = total_min % 60

            if hour >= 16:
                break

            if day_type == "trend_up":
                # Breakout above IB high, then trend
                if bar_idx < 3:
                    target = ib_high + random.uniform(2, 8)
                    price += (target - price) * 0.5 + np.random.randn() * 1.5
                else:
                    price += np.random.randn() * 1.5 + 0.8  # Upward drift
            elif day_type == "trend_down":
                if bar_idx < 3:
                    target = ib_low - random.uniform(2, 8)
                    price += (target - price) * 0.5 + np.random.randn() * 1.5
                else:
                    price += np.random.randn() * 1.5 - 0.8  # Downward drift
            else:
                # Rotational: oscillate around IB mid
                ib_mid = (ib_high + ib_low) / 2
                mean_revert = (ib_mid - price) * 0.03
                price += np.random.randn() * 2.0 + mean_revert

            price = round(price * 4) / 4
            spread = abs(np.random.randn()) * 1.5 + 0.75
            bar_high = round((price + spread) * 4) / 4
            bar_low = round((price - spread) * 4) / 4
            bar_open = round((price + np.random.randn() * 0.75) * 4) / 4

            dt = current_date.replace(hour=hour, minute=minute, second=0)
            vol = random.randint(3000, 20000)
            rows.append({
                "Date/Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Open": bar_open, "High": max(bar_high, bar_open, price),
                "Low": min(bar_low, bar_open, price), "Close": price,
                "Volume": vol
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "es_5m_sample.csv")
    df = generate_es_5m_data(days=20)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} bars → {output}")
