"""Clean calendar-spread ticks out of the two canonical ES source CSVs.

Applies Lucy's scrub policy (drop rows where any OHLC < OUTRIGHT_MIN_PRICE,
enforce OHLC consistency) in place, preserving original column names and
column order so downstream consumers see no schema change.

Backups (`.raw`) must already exist — this script refuses to run otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HOME = Path.home()
# $500 floor: real ES outright low in the 16yr dataset is $1002.75 (2010), so a
# $500 buffer is safely below any legitimate bar. Catches both low-rate-era
# spreads (-$15 to +$15) and high-rate-era spreads (up to ~$200 in 2024-2026).
OUTRIGHT_MIN_PRICE = 500.0

TARGETS = [
    {
        "path": HOME / "projects/backtesting/spx/data/ES_1min_2010_2022_databento.csv",
        "ohlc_cols": ("open", "high", "low", "close"),
        "label": "databento 1m 2010-2022",
    },
    {
        "path": HOME / "projects/backtesting/es/data/es_5m_databento_2yr.csv",
        "ohlc_cols": ("Open", "High", "Low", "Close"),
        "label": "databento 5m 2yr baseline",
    },
]


def scrub(df: pd.DataFrame, ohlc_cols: tuple[str, str, str, str]) -> tuple[pd.DataFrame, int, int]:
    """Drop rows where any OHLC < OUTRIGHT_MIN_PRICE or fails consistency.

    Returns (clean_df, dropped_outright_count, dropped_consistency_count).
    """
    o_col, h_col, l_col, c_col = ohlc_cols
    n0 = len(df)
    ohlc = df[[o_col, h_col, l_col, c_col]]
    outright = (ohlc >= OUTRIGHT_MIN_PRICE).all(axis=1)
    n_after_outright = outright.sum()
    dropped_spread = n0 - n_after_outright

    consistent = (
        (df[l_col] <= df[h_col])
        & (df[o_col].between(df[l_col], df[h_col]))
        & (df[c_col].between(df[l_col], df[h_col]))
    )
    n_after_all = (outright & consistent).sum()
    dropped_inconsistent = n_after_outright - n_after_all

    clean = df[outright & consistent].reset_index(drop=True)
    return clean, int(dropped_spread), int(dropped_inconsistent)


def process_target(target: dict) -> None:
    path: Path = target["path"]
    raw: Path = path.with_suffix(path.suffix + ".raw")

    if not raw.exists():
        print(f"[SKIP] {target['label']} — no .raw backup at {raw}")
        sys.exit(1)

    print(f"\n=== {target['label']} ===")
    print(f"  source:  {path}")
    print(f"  backup:  {raw}")

    df = pd.read_csv(raw)
    print(f"  loaded:  {len(df):,} rows, columns={list(df.columns)}")

    clean, dropped_spread, dropped_inconsistent = scrub(df, target["ohlc_cols"])
    print(f"  dropped: {dropped_spread:,} spread ticks + {dropped_inconsistent:,} OHLC-inconsistent")
    print(f"  kept:    {len(clean):,} rows ({len(clean)/len(df)*100:.3f}%)")

    clean.to_csv(path, index=False)
    print(f"  wrote:   {path}")


def main() -> int:
    for t in TARGETS:
        process_target(t)
    print("\nDone. Re-run verification to confirm zero-corrupt state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
