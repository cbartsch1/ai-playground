# AMT-TEMA v8 -- Limitations

## Critical Limitations

### 1. Year 2 Performance Is Weak (PF 1.129)

The most concerning finding is the year-over-year degradation:

| Period | PF | P&L | Sharpe |
|--------|------|------|--------|
| Year 1 (Feb 25-26) | 1.820 | +$26,672 | 3.95 |
| Year 2 (Feb 24-25) | 1.129 | +$3,412 | 0.78 |

Year 2's PF 1.129 is barely above breakeven. A Sharpe of 0.78 is below the commonly accepted threshold of 1.0 for tradeable strategies. If the trend continues:
- Year 3 at PF ~0.9: the strategy becomes a loser
- Year 3 at PF ~1.1: the strategy treads water

The combined 2-year number (PF 1.511) is heavily weighted by Year 1's strong performance. An investor evaluating this strategy based on Year 2 alone would likely pass.

### 2. Only 2 Years of Data

The SPX options strategies have 5 years of data (2021-2026). AMT-TEMA has only 2 years (2024-2026). This limits:
- Statistical power (177 trades vs 400+ for longer-running strategies)
- Regime coverage (no 2022 bear market data, no 2021 recovery data)
- Confidence in tail behavior (fewer extreme events observed)

Extending the dataset further back would require additional Databento downloads and careful handling of ES contract rollovers and price level changes.

### 3. VWAP-Based Value Area Is Imprecise

The strategy uses VWAP as a proxy for POC, with standard deviation bands for VAH/VAL. This is an approximation of the real volume profile:

- **Real POC**: Exact price level with highest traded volume (from TPO or volume profile)
- **VWAP POC**: Volume-weighted average price -- close to POC but not identical
- **Real VA**: 70% volume zone computed from TPO distribution
- **VWAP VA**: VWAP +/- 1 stdev -- approximation based on price dispersion, not volume distribution

Impact on the strategy:
- VA Fade setup (currently OFF) failed partly due to imprecise VA levels
- IB Breakout does not directly use VA levels, so the impact on v8 is minimal
- The 80% Rule setup would benefit from precise VA levels but is also currently OFF

**Pending improvement**: `volume.profile_session()` in Pine Script would provide real TPO-based VA levels. This is a future enhancement that could revive the VA Fade and 80% Rule setups.

### 4. Natural Frequency Is Low (~0.25 Trades/Day)

177 trades over 2 years = 88.5 trades/year = ~0.25 trades per day. This means:
- Many days have zero trades (75% of trading days)
- The strategy cannot be evaluated week-by-week
- Monthly evaluation is marginal (avg 7 trades/month)
- A losing streak of 5 trades takes ~20 trading days (~1 month) to develop

All attempts to increase frequency degraded performance:
- Adding more setups (VA Fade, 80% Rule) added losing trades
- Loosening filters (Fridays, noon hour) added losing trades
- Widening IB range constraints added negligible trades

177 is the number. This is not a deficiency -- it is the natural frequency of IB breakout trend days on ES.

### 5. Short-Only Limits Opportunity Set

The strategy only takes short entries. In a persistent bull market:
- IB breakdowns to the downside become less frequent
- Trend days tend to be upside breakouts (which the strategy ignores)
- The strategy may go weeks without a trade during strong rallies

Over 2 years, longs lost -$5,589, making SHORT_ONLY the correct decision. But this means the strategy captures only half of the potential trend days.

### 6. Single Instrument Dependency

The strategy trades only ES. There is no diversification across:
- Other futures (NQ, RTY, YM) -- though the user explicitly does NOT trade NQ
- Other asset classes
- Other timeframes

If ES market structure changes (exchange fee changes, market maker behavior shifts, tick size changes), the strategy has no fallback.

## Failed Approaches (What Was Tried and Rejected)

### VA Fade Setup

- 1-year result: -$2,825 (PF 0.835, 33.9% WR)
- VWAP proxy VA levels were too imprecise for the tight stop/target placement required
- The setup concept is sound (mean reversion at VA edges) but the implementation data is inadequate
- Kept in code (default OFF) pending volume profile improvement

### Long Trades

- 2-year result: -$5,589 (PF 0.912)
- Consistently unprofitable across both years
- Structural disadvantage: IB upside breakouts on ES are slower and less reliable than downside
- Removed in v8 via SHORT_ONLY default

### Fixed 20pt Stop

- At ES 5000: 20pt = 0.40% (appropriate)
- At ES 6800: 20pt = 0.29% (too tight -- gets stopped out on normal retracements)
- Replaced with 30bps percentage-based stop in v8

### TEMA Crossover Setup

- Isolated test: -$3,313 loss (PF 0.788, 65 trades)
- TEMA crossover as a standalone entry has no edge on ES 5m
- The crossover is used only as a CONFIRMATION filter for IB breakout, not as an entry signal

### Friday Trading

- 1-year result: 30% WR, -$13,415
- Systematic loser across all months -- not noise
- Friday liquidity dynamics destroy IB breakout reliability
- Permanently filtered out in v7

### Noon Hour Trading (12:00-13:00 ET)

- 1-year result: 33.3% WR, -$10,320
- Lunch hour volume collapse creates false breakouts
- IB crossover events during noon are whipsaws
- Permanently blacked out in v7

### v5.2's PF 2.5

- Based on 3.5 months of TradingView data (~49 trades)
- Full 1-year backtest revealed PF 1.031 (the 2.5 was a data window artifact)
- Lesson: short backtesting windows on TradingView are dangerously misleading

### Wider Stops (22pt, 25pt)

- Both made results WORSE than 20pt
- Wider stops let losing trades accumulate more loss without rescuing more winners
- The percentage-based 30bps solved the scaling problem without widening the stop

### Persistent Breakout State

- Attempted to replace `ta.crossover` with an armed flag + extended check window
- Result: 103 trades, PF 0.987, +$732, DD $12.5K
- Added garbage trades on normal days while still missing the critical Feb 12 trade
- Reverted to `ta.crossover`/`ta.crossunder` (one-shot)

## What Could Kill This Strategy

1. **Year 3 PF < 1.0**: If the degradation from Year 1 (PF 1.82) to Year 2 (PF 1.13) continues, Year 3 could be a loser. Monitor monthly and pull the plug if PF drops below 1.0 for 3 consecutive months.

2. **ES Market Structure Change**: Any change to ES tick size, margin requirements, or trading hours that alters IB dynamics could invalidate the strategy. CME rule changes are low-frequency but high-impact events.

3. **Flatten Failure**: The most dangerous single-event risk. If the 15:55 ET flatten alert fails and an overnight gap hits a short position, the loss could exceed the entire 2-year P&L.

4. **Persistent Low Volatility**: If ATR contracts and IB ranges shrink below 8 points consistently, the strategy simply stops generating signals. No direct loss, but no income either.

5. **TEMA Lag in Sudden Moves**: `ta.crossunder` requires TEMA 9 < TEMA 21 on the exact bar of the IB breakout. If a sudden move breaks the IB low before TEMA has time to turn bearish, the trade is missed. This was identified as a suspected blocker for certain trend days (e.g., Feb 12, 2026 -- a 180pt move that the strategy missed).

6. **Automation Chain Failure**: TradingView -> PickMyTrade -> Tradovate has three potential failure points. Any link going down means missed entries or (worse) unfilled flatten orders.

## Pending Validation Checklist

- [x] 2-year Databento backtest -- DONE (177 trades, PF 1.511, p=0.028)
- [x] Walk-forward validation -- DONE (PF ratio 0.77, PASS)
- [x] TradingView verification -- DONE (directionally consistent)
- [x] Demo automation -- DONE (live on Tradovate DEMO2361007 since Feb 15)
- [ ] 1-2 weeks of demo monitoring -- verify entries match TradingView signals
- [ ] Verify SL/TP bracket orders placed correctly on Tradovate
- [ ] Verify 15:55 ET flatten fires reliably every day
- [ ] Go live with 1 contract on funded account
- [ ] 3-month live performance comparison vs backtest
- [ ] Upgrade VA levels to volume.profile_session()
- [ ] Re-evaluate VA Fade and 80% Rule with improved VA data
- [ ] Path 2 exploration: Python bot + IBKR API (permanent solution)
