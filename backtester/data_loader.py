"""Parse thinkorSwim chart export CSV into a pandas DataFrame with session tags."""

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")


def load_tos_csv(filepath: str, instrument: str = "ES") -> pd.DataFrame:
    """Load a thinkorSwim chart export CSV.

    thinkorSwim export format (right-click chart > Export chart data):
        Date/Time, Open, High, Low, Close, Volume

    Returns a DataFrame indexed by ET datetime with columns:
        open, high, low, close, volume, et_hour, et_minute, et_time,
        is_rth, is_ib_period, is_trading_window, new_rth, session_date
    """
    df = pd.read_csv(filepath)

    # Normalize column names (thinkorSwim sometimes varies case/spacing)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename date/time column variants
    for col in df.columns:
        if "date" in col and "time" in col:
            df = df.rename(columns={col: "datetime"})
            break

    # Parse datetime — thinkorSwim exports in ET by default
    df["datetime"] = pd.to_datetime(df["datetime"])

    # If naive, localize to ET; if aware, convert to ET
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(ET, ambiguous="NaT", nonexistent="NaT")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(ET)

    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    # Standardize OHLCV column names
    rename_map = {}
    for target in ["open", "high", "low", "close", "volume"]:
        for col in df.columns:
            if target in col.lower():
                rename_map[col] = target
                break
    df = df.rename(columns=rename_map)

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0).astype(int)

    # Session tags
    df["et_hour"] = df.index.hour
    df["et_minute"] = df.index.minute
    df["et_time"] = df["et_hour"] * 100 + df["et_minute"]

    tag_sessions(df)

    return df


def tag_sessions(df: pd.DataFrame,
                 rth_start: int = 930,
                 rth_end: int = 1600,
                 ib_end: int = 1030,
                 trade_start: int = 1035,
                 trade_end: int = 1500) -> None:
    """Add session boolean columns in-place."""
    et = df["et_time"]

    df["is_rth"] = (et >= rth_start) & (et < rth_end)
    df["is_ib_period"] = (et >= rth_start) & (et < ib_end)
    df["is_trading_window"] = (et >= trade_start) & (et < trade_end)

    # new_rth: first RTH bar after a non-RTH bar
    df["new_rth"] = df["is_rth"] & ~df["is_rth"].shift(1, fill_value=False)

    # session_date: the trading date for each bar.
    # Bars during RTH get that day's date. ETH bars get the most recent RTH date.
    rth_dates = pd.Series(pd.NaT, index=df.index, dtype="object")
    rth_dates[df["new_rth"]] = df.index[df["new_rth"]].date
    df["session_date"] = rth_dates.ffill()

    # hlc3 for VWAP calculations
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0

    # Weekday (0=Monday, 4=Friday) for day-of-week filters
    df["weekday"] = df.index.weekday
