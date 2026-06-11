"""Holdout-lockbox enforcement tests for the ES honest re-validation.

The holdout (2025-07-01 .. end of data) must be physically unreachable from
locked code paths.
"""

import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.honest_revalidation_es.lockbox import (
    Lockbox, HoldoutViolation, HOLDOUT_START, ES_5YR_PARQUET,
)

needs_data = pytest.mark.skipif(
    not ES_5YR_PARQUET.exists(), reason="5yr parquet not on disk")


def test_guard_date_raises_in_holdout():
    box = Lockbox()
    with pytest.raises(HoldoutViolation):
        box.guard_date(date(2025, 7, 1))
    with pytest.raises(HoldoutViolation):
        box.guard_date(pd.Timestamp("2025-12-01 10:00",
                                    tz="America/New_York"))
    box.guard_date(date(2025, 6, 30))  # last DEV day passes


def test_truncate_bars_tz_aware():
    idx = pd.date_range("2025-06-28", "2025-07-03", freq="1h",
                        tz="America/New_York")
    df = pd.DataFrame({"close": range(len(idx))}, index=idx)
    out = Lockbox().truncate_bars(df)
    assert out.index.max() < pd.Timestamp("2025-07-01",
                                          tz="America/New_York")
    assert len(out) > 0


def test_truncate_session_dict():
    obs = {date(2025, 6, 30): 1, date(2025, 7, 1): 2, date(2025, 12, 1): 3}
    out = Lockbox().truncate_session_dict(obs)
    assert set(out) == {date(2025, 6, 30)}


def test_unlock_passes_through():
    obs = {date(2025, 7, 1): 2}
    box = Lockbox(unlock_holdout=True)
    assert box.truncate_session_dict(obs) == obs
    box.guard_date(date(2025, 12, 1))  # no raise


@needs_data
def test_load_es_5m_is_truncated_and_engine_ready():
    df = Lockbox().load_es_5m()
    assert df.index.max() < pd.Timestamp("2025-07-01",
                                         tz="America/New_York")
    # engine-ready post-conditions
    for col in ["open", "high", "low", "close", "volume", "et_time",
                "is_rth", "is_ib_period", "is_trading_window", "new_rth",
                "session_date", "hlc3", "weekday", "is_globex"]:
        assert col in df.columns, f"missing column {col}"
    # ~1130 DEV sessions x 78 RTH bars/day ≈ 88k
    assert df["is_rth"].sum() > 80_000


def test_load_vix_hourly_truncated():
    obs = Lockbox().load_vix_hourly()
    assert max(obs) < HOLDOUT_START
    # bars are knowable-time stamped (close_min = start + 60)
    some_day = next(iter(obs.values()))
    assert all(b["close_min"] >= 60 for b in some_day["bars"])


def test_load_vix_daily_truncated_with_prev_threaded():
    lookup = Lockbox().load_vix_daily()
    assert max(lookup) < HOLDOUT_START
    days = sorted(lookup)
    # prev_close of day N equals close of day N-1
    for a, b in zip(days[100:103], days[101:104]):
        assert lookup[b]["prev_close"] == lookup[a]["close"]
