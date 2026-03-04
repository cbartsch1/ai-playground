# AMT-TEMA v8 -- Exit Analysis

## Exit Architecture

The strategy uses TradingView's built-in `strategy.exit()` function with both stop and limit orders, plus an end-of-day flatten. This creates a bracket order structure that maps directly to the PickMyTrade automation.

### Exit Types

1. **IB Extension Target** (limit order) -- take profit
2. **Percentage-Based Stop** (stop order) -- cut loss at 30bps
3. **End-of-Session Flatten** (market order at 15:55 ET) -- close any remaining position

## IB Extension Target -- The Profit Engine

### Logic

Target = Entry Price - max(IB Range, ibMinTarget)

When price breaks below the IB low, the AMT expectation is that the breakout will extend by at least one IB range:
- If IB range is 12 points: target is 12 points below entry
- If IB range is 8 points (minimum): target defaults to ibMinTarget (10 points)

This "measured move" concept is well-established in Market Profile analysis: the IB range represents the initial price exploration, and the breakout leg tends to match or exceed that exploration distance.

### Performance Context

From the 1-year A/B testing, flatten exits were identified as the profit engine: +$23,310 from 63 trades (66.7% WR) out of the v6.1 baseline's 305 trades. This suggests that many winning trades were NOT reaching the IB extension target -- they were profitable at end-of-day but hadn't traveled far enough for the limit to fill.

The flatten exit acts as a "natural take profit" for trades that are correct in direction but insufficient in magnitude to reach the IB extension. This is important: without the flatten, these trades would simply be held and exited at the stop or target on the next day (which is not possible since the strategy flattens daily).

## Percentage-Based Stop -- v8 Innovation

### The Problem with Fixed Stops

v7 used a fixed 20-point maximum stop (ibMaxStopPts = 20). Over the 2-year period:
- At ES 5000 (Year 2): 20pt = 0.40% of price
- At ES 6800 (Year 1): 20pt = 0.29% of price

The same stop was proportionally tighter during Year 1 (higher ES price), which may have contributed to more stop-outs during volatile moves that would have otherwise been winners.

### The Solution: 30bps Stop

v8 replaced the fixed stop with a percentage-based stop:

```
ibMaxStopDyn = ibPctStop ? close * ibStopBps / 10000.0 : ibMaxStopPts
```

At 30bps (ibStopBps = 30):
- ES 5000: 15.0 points
- ES 6000: 18.0 points
- ES 6500: 19.5 points
- ES 6800: 20.4 points

This scales the stop proportionally with price, maintaining consistent risk exposure regardless of the ES price level.

### Stop Placement

The actual stop is placed at:
```
ibSL_S = math.min(ibRawSL_S, close + ibMaxStopDyn)
```

Where `ibRawSL_S` is determined by the stop type setting:
- **IB Mid** (default, optimal): Midpoint of the IB range -- the "point of no return"
- **IB Edge**: The IB high -- wider stop, allows more room
- **ATR-based**: 1.5x ATR above entry -- volatility-adaptive

The stop is capped at the percentage-based maximum. So if IB Mid is 25 points above entry but the 30bps stop is only 20 points, the stop is placed at entry + 20 points (the tighter of the two).

### Stop Size Optimization History

| Stop Width | Result | Notes |
|------------|--------|-------|
| 15pt fixed | Worse | Too tight, stopped out on normal retracements |
| 20pt fixed | v7 default | Good for ES 6800 but too wide at ES 5000 |
| 22pt fixed | Worse | Adding 2pt doesn't help, adds loss on wrong trades |
| 25pt fixed | Worse | Same direction -- wider stops lose more on wrong trades |
| **30bps pct** | **v8 (best)** | Scales correctly, 15pt at 5000, 20.4pt at 6800 |

## End-of-Session Flatten

```pine
if etTime >= flattenTime and strategy.position_size != 0
    strategy.close_all("RTH End", alert_message=pmtCloseMsg())
```

The flatten at 15:55 ET serves multiple purposes:

1. **Avoids overnight risk**: ES futures trade nearly 24 hours. Holding through ETH (extended trading hours) introduces gap risk.
2. **Captures intraday edge**: The AMT framework is a day-trading framework. IB breakout dynamics are intraday phenomena that do not persist overnight.
3. **Acts as a natural profit-taker**: Winning trades that haven't reached the IB extension target are closed at current P&L.
4. **Commission efficiency**: One close at flatten vs. a potential whipsaw stop-and-reverse in the final 5 minutes.

### Flatten Performance

From the 1-year analysis, flatten exits accounted for +$23,310 from 63 trades (66.7% WR). These are trades where:
- The direction was correct (price moved below IB after short entry)
- The magnitude was insufficient to reach the IB extension target
- The trade was profitable at 15:55 but would not have hit the limit

Without the flatten, these trades would hit either the stop (loss) or the target (bigger win) -- but the high win rate (66.7%) suggests most of them are moderately profitable trades that the flatten correctly captures.

## Exit Interaction: Stop vs Target vs Flatten

The three exits create a natural distribution:

| Scenario | Exit Type | Expected Outcome |
|----------|-----------|-----------------|
| Strong trend day | Target hit | Large win (IB extension reached) |
| Moderate trend day | Flatten at 15:55 | Moderate win (correct direction, not far enough) |
| Failed breakout | Stop hit | Loss at 30bps (breakout reversed) |
| Late entry, right direction | Flatten at 15:55 | Small win or breakeven |
| Chop day (shouldn't happen) | Stop hit | Loss (but filters should prevent entry) |

## Alternative Exit Approaches Considered and Rejected

### Trailing Stop

Not used. The IB breakout setup has a defined target (IB extension) and a defined invalidation point (IB mid / 30bps). A trailing stop would add complexity without improving the clear bracket structure.

### Time-Based Exit (e.g., 2-Hour Max Hold)

Not used for IB Breakout. The trend day thesis says price should continue all day after IB breakout. Cutting the trade after 2 hours would miss late-day extensions. The flatten at 15:55 is the correct time-based exit.

### Multiple Targets / Scaling Out

Multi-contract scaling was tested and confirmed to be pure linear. Scaling out at partial targets (e.g., 1 contract at 50% of IB extension, 1 at 100%) would NOT improve risk-adjusted returns -- it would simply be equivalent to running half the contracts with one target and half with another.

### Breakeven Stop

Moving the stop to breakeven after price travels N points in the right direction was not implemented. This approach often gets stopped out at breakeven on normal retracements, converting would-be winners into scratches. The fixed stop at IB mid / 30bps is designed to give the trade room to work.

## What This Means for Live Trading (Automation)

The bracket order structure (entry + stop + target) maps perfectly to PickMyTrade's execution model:

1. **Entry**: Market order on `strategy.entry()` -- fires the webhook
2. **Stop + Target**: Bracket orders on `strategy.exit()` -- sent simultaneously with entry
3. **Flatten**: `strategy.close_all()` at 15:55 ET -- separate close webhook

PickMyTrade receives the full bracket (SL and TP as absolute prices) in the JSON payload. Tradovate places the bracket order on receipt. This means:
- Stop and target are working orders on the exchange, not client-side
- If TradingView or PickMyTrade goes down, the bracket remains active
- The flatten at 15:55 requires TradingView to fire the alert -- this is a single point of failure

### Critical Point: Flatten Reliability

If the flatten alert fails to fire (TradingView outage, webhook failure, PMT downtime), the position will remain open overnight. This is the most dangerous failure mode:
- ES can gap significantly overnight
- A position held through a gap against could lose 50+ points
- Mitigation: Set a secondary alert at 15:55, monitor manually on critical days
