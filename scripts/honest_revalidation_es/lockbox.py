"""Holdout lockbox — physical holdout discipline for the ES honest re-validation.

DEV window:     data start (2021-01) .. 2025-06-30  (all scanning / optimization)
HOLDOUT window: 2025-07-01 .. end of data           (touched ONCE, Phase C only)

Mirrors spx/scripts/honest_revalidation/lockbox.py. Every data path used by
the ringer-es scans goes through this wrapper: ES 5m bars, daily VIX, and
hourly intraday VIX are hard-truncated at 2025-06-30 23:59 ET, and any
explicit request for later data raises HoldoutViolation — unless the lockbox
was constructed with unlock_holdout=True, which prints a loud HOLDOUT UNLOCK
warning (Phase C only).

Pre-registered protocol:
~/projects/lab/plans/2026-06-10-edge-legitimacy-audit/ringer-es-protocol.md
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ES_5YR_PARQUET = REPO / "data" / "es_5m_5yr_from_1m.parquet"
SPX_DATA = REPO.parent / "spx" / "data"
VIX_DAILY_PARQUET = SPX_DATA / "vix_daily.parquet"
VIX_HOURLY_CSV = SPX_DATA / "VIX_60min_2018_2025.csv"

DEV_END = date(2025, 6, 30)              # last session inside DEV
HOLDOUT_START = date(2025, 7, 1)

# tz-aware cutoff for intraday bar indexes (ts < this is DEV)
_CUTOFF_ET = pd.Timestamp("2025-07-01 00:00", tz="America/New_York")
# naive cutoff for daily indexes (vix_daily.parquet has naive midnights)
_CUTOFF_NAIVE = pd.Timestamp("2025-07-01 00:00")

_UNLOCK_WARNING = """
################################################################################
#  HOLDOUT UNLOCK — this process can now see data after 2025-06-30.            #
#  This is permitted ONLY for Phase C (one look per surviving config).         #
#  If you are scanning or walk-forwarding, STOP: the holdout is being burned.  #
################################################################################
"""


class HoldoutViolation(RuntimeError):
    """Raised when locked code asks for data on/after 2025-07-01."""


class Lockbox:
    def __init__(self, unlock_holdout: bool = False):
        self.unlocked = bool(unlock_holdout)
        if self.unlocked:
            print(_UNLOCK_WARNING, file=sys.stderr)
            print(_UNLOCK_WARNING)

    # -- guards ---------------------------------------------------------

    def guard_date(self, d):
        """Raise HoldoutViolation if d (date/Timestamp) is in the holdout."""
        ts = pd.Timestamp(d)
        if ts.tz is not None:
            ts = ts.tz_convert("America/New_York").tz_localize(None)
        if ts >= _CUTOFF_NAIVE and not self.unlocked:
            raise HoldoutViolation(
                f"Requested {d} — on/after holdout start {HOLDOUT_START}. "
                f"Phase A/B must not see holdout data.")

    # -- truncation -----------------------------------------------------

    def truncate_bars(self, df):
        """Truncate a tz-aware intraday-indexed frame at the DEV cutoff."""
        if self.unlocked:
            return df
        cutoff = _CUTOFF_ET if df.index.tz is not None else _CUTOFF_NAIVE
        return df[df.index < cutoff]

    def truncate_daily(self, df):
        """Truncate a daily-indexed frame (naive or tz-aware) at the cutoff."""
        return self.truncate_bars(df)

    def truncate_session_dict(self, obs):
        """Truncate a {date: ...} mapping (e.g. intraday-VIX observations)."""
        if self.unlocked:
            return obs
        return {d: v for d, v in obs.items() if d < HOLDOUT_START}

    # -- loaders (the only way the scanner gets market data) -------------

    def load_es_5m(self):
        """DEV-truncated engine-ready ES 5m frame from the canonical 5yr parquet.

        Post-conditions match data_loader.load_tos_csv (session tags, hlc3,
        weekday) EXCEPT clean_bad_ticks is NOT applied: the parquet was
        scrubbed at build time (build_es_5m_5yr_from_1m.py drops rows with
        OHLC < $500 or OHLC inconsistency) and clean_bad_ticks' ffill/bfill
        future-fills corrupted ticks (known caveat, deliberately avoided).
        """
        from backtester.data_loader import tag_sessions
        df = pd.read_parquet(ES_5YR_PARQUET)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df = df.sort_index()
        for col in ("source", "is_rth"):
            if col in df.columns:
                df = df.drop(columns=[col])
        df["volume"] = df["volume"].fillna(0).astype(int)
        df["et_hour"] = df.index.hour
        df["et_minute"] = df.index.minute
        df["et_time"] = df["et_hour"] * 100 + df["et_minute"]
        tag_sessions(df)
        return self.truncate_bars(df)

    def load_vix_daily(self):
        """DEV-truncated daily VIX lookup: date -> {open,high,low,close,prev_*}.

        prev_close/prev_high/prev_low are threaded BEFORE truncation so the
        first DEV day still has its true prior-day values (prior-day data is
        causal regardless of window).
        """
        vix = pd.read_parquet(VIX_DAILY_PARQUET)
        lookup = {}
        prev_close = prev_high = prev_low = None
        for idx, row in vix.iterrows():
            d = idx.date() if hasattr(idx, "date") else idx
            lookup[d] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row.get("low", 0)),
                "close": float(row["close"]),
                "prev_close": prev_close,
                "prev_high": prev_high,
                "prev_low": prev_low,
            }
            prev_close = float(row["close"])
            prev_high = float(row["high"])
            prev_low = float(row.get("low", 0))
        return self.truncate_session_dict(lookup)

    def load_vix_hourly(self):
        """DEV-truncated hourly intraday VIX: date -> {open, bars, granularity}.

        bars = [{close_min, high, close}, ...] where close_min is the ET
        minute-of-day at which the bar's values become KNOWABLE (bar start
        + 60). Mirrors spx honest_revalidation/bush_sim_grid.load_vix_intraday_raw
        (hourly part).
        """
        hourly = pd.read_csv(VIX_HOURLY_CSV, parse_dates=["time"])
        et = pd.to_datetime(hourly["time"], utc=True).dt.tz_convert(
            "America/New_York")
        hourly["d"] = et.dt.date
        hourly["minute"] = et.dt.hour * 60 + et.dt.minute
        obs = {}
        for d, g in hourly.groupby("d"):
            g = g.sort_values("minute")
            bars = [{"close_min": int(m) + 60, "high": float(h),
                     "close": float(c)}
                    for m, h, c in zip(g["minute"], g["high"], g["close"])]
            obs[d] = {"open": float(g.iloc[0]["open"]), "bars": bars,
                      "granularity": "60m"}
        return self.truncate_session_dict(obs)
