# Data Directory — thinkorSwim Chart Exports

## How to Export ES 5-Minute Data from thinkorSwim

1. Open a chart of `/ES` (E-mini S&P 500 futures) with **5-minute** aggregation
2. Set the time range: **1 year** or **2 years** for extended backtesting
3. Make sure the chart shows both RTH and ETH data (the backtester handles session filtering)
4. Right-click on the chart → **Export chart data...**
5. Save as CSV to this directory (e.g., `es_5m.csv`)

### Expected CSV Format

thinkorSwim exports with these columns:

```
Date/Time, Open, High, Low, Close, Volume
2025-01-02 09:30:00, 5950.25, 5952.50, 5949.00, 5951.75, 12345
```

- Date/Time is in Eastern Time (ET)
- OHLCV are standard candle data
- Volume may be 0 for some bars (handled by the loader)

## For SPX

Same process but on the `SPX` (S&P 500 Index) chart:
- SPX only has RTH data (9:30-16:00 ET)
- Volume is synthetic (not real exchange volume)
- Save as `spx_5m.csv`

## Files

- `es_5m.csv` — ES futures 5-minute data (export from thinkorSwim)
- `spx_5m.csv` — SPX index 5-minute data (export from thinkorSwim)
