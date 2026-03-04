# AMT-TEMA v8 -- Strategy Thesis

## Core Idea

Trade ES (E-mini S&P 500) on the 5-minute chart using Auction Market Theory (AMT) for structural context and Triple Exponential Moving Average (TEMA) for momentum confirmation. Enter short after the Initial Balance (IB) range is broken to the downside with TEMA bearish confirmation. Exit at IB extension target, percentage-based stop, or end-of-session flatten.

## Why Auction Market Theory?

Auction Market Theory, developed by J. Peter Steidlmayer and refined through Market Profile analysis, models the market as a continuous auction between buyers and sellers. Key concepts:

### Initial Balance (IB) -- The Day's Foundation

The Initial Balance is the price range established in the first hour of regular trading (9:30-10:30 ET). It represents the initial balance between overnight positions and fresh morning orders. The IB sets the tone for the rest of the day:

- **Narrow IB**: Suggests indecision. The day is likely to break out and trend (trend day).
- **Wide IB**: Suggests early conviction. The day may rotate within the established range (rotational day).
- **IB breakout**: When price moves decisively beyond the IB range, it signals a directional commitment. The breakout direction tends to persist, especially on narrow IB days.

### Value Area (VA) -- Where the Market Agrees on Value

The Value Area is the price range where approximately 70% of the day's volume trades. It represents the zone of accepted value:

- **POC** (Point of Control): The price with the highest volume -- the market's fair value estimate.
- **VAH** (Value Area High): Upper boundary of accepted value.
- **VAL** (Value Area Low): Lower boundary -- prices below here are perceived as undervalued.

The strategy uses VWAP as a proxy for POC, with standard deviation bands for VAH/VAL. This is an approximation; true volume profile data would be more accurate (pending improvement to `volume.profile_session()`).

### Day Type Classification

The strategy classifies each day based on how today's IB width compares to a 20-day rolling EMA average:

| IB Ratio | Classification | Implication |
|----------|----------------|-------------|
| < 0.8x avg | NARROW | Trend day potential -- favor IB breakout |
| 0.8x -- 1.2x avg | NORMAL | Either direction possible |
| > 1.2x avg | WIDE | Rotational -- fades more likely |

## Why TEMA?

Triple Exponential Moving Average is a low-lag trend indicator that applies the EMA formula three times and combines the results to reduce lag while maintaining smoothness:

```
TEMA(n) = 3 * EMA(n) - 3 * EMA(EMA(n)) + EMA(EMA(EMA(n)))
```

Three TEMA periods are used:

| TEMA | Period | Role |
|------|--------|------|
| Fast | 9 | Short-term momentum |
| Slow | 21 | Medium-term trend |
| Trend | 55 | Regime filter |

**TEMA 9 < TEMA 21** = bearish (short-term momentum below medium-term trend)
**Close < TEMA 55** = downtrend (price below long-term trend filter)

Both conditions must be true for a short entry. This prevents entering short during pullbacks in an uptrend.

### Why TEMA Over EMA?

TEMA's triple-exponential construction reduces lag by approximately 3x compared to a simple EMA of the same period. This matters for IB breakout entries where timing is critical: a regular EMA(21) would confirm the breakout too late (after the move has already extended), while TEMA(21) confirms closer to the breakout point.

## Setup 1: IB Breakout + TEMA (THE Setup)

This is the active, profitable setup. The other setups are either OFF or exploratory.

### Entry Conditions (ALL must be true)

1. **IB is complete** (past 10:30 ET, ibDone = true)
2. **IB range is valid** (8 -- 80 points, minIBRange to maxIBRange)
3. **Price breaks below IB low** (`ta.crossunder(close, ibLow)`)
4. **TEMA bearish** (TEMA 9 < TEMA 21)
5. **Trend down** (Close < TEMA 55) -- optional via `useTrendFilter`
6. **In trading window** (tradeStart 10:35 -- tradeEnd 15:00 ET)
7. **Entry allowed** (not Friday if skipFriday, not in noon blackout 12:00-13:00)
8. **Past cooldown** (2+ bars since last exit)
9. **No existing position** (`strategy.position_size == 0`)
10. **Within daily trade limit** (< maxIBTrades per direction per day, default 2)
11. **Short allowed** (dirFilter != "Long Only", default "Short Only")

### Stop Loss

**Percentage-based stop (v8)**: 30 basis points from entry price.
- At ES 5000: 15.0 points
- At ES 6800: 20.4 points
- Scales automatically with price, solving the v7 problem where 20pt fixed stop was 0.4% at ES 5000 but only 0.29% at ES 6800.

The stop is placed at IB Mid (midpoint of IB range), capped at the dynamic percentage-based stop. The IB Mid represents the "point of no return" -- if price retraces to the middle of the IB, the breakout thesis is invalidated.

### Target

IB extension: entry price minus IB range (or ibMinTarget of 10 points, whichever is larger).

The IB extension target is based on the AMT concept that when price breaks out of the IB, it tends to travel at least one IB range beyond the breakout level. This is the "measured move" in classical technical analysis, grounded in the auction framework.

## Setup 2: Value Area Fade (DEFAULT OFF)

Mean reversion at previous day's Value Area edges toward POC.

- Long at VAL (undervalued boundary), target POC
- Short at VAH (overvalued boundary), target POC
- Requires TEMA slope confirmation (rising for longs, falling for shorts)
- Minimum R:R ratio of 0.5

**Status**: OFF by default. Lost -$2,825 over 1 year of full backtesting (PF 0.835, 33.9% WR). The VWAP-based VA proxy is too imprecise for the tight stop/target placement this setup requires. Could be revived with real volume profile data.

In isolated testing on 3.5 months of TradingView data, VA Fade showed +$8,993 (PF 3.507, 58.33% WR, 24 trades). This demonstrates that the SETUP works in principle, but the data window was too short and the VA proxy too imprecise for production use.

## Setup 3: 80% Rule (DEFAULT OFF)

When price opens outside the previous day's Value Area and re-enters, there is approximately an 80% probability it will traverse to the opposite VA edge (classic Market Profile rule).

- Tracks re-entry with N-bar confirmation (default 6 bars inside VA)
- Requires TEMA direction confirmation
- Target: opposite VA edge

**Status**: OFF by default. Added ~13 trades in testing but lost -$3,475 net. Same VA proxy precision issue as Setup 2.

## Why SHORT ONLY?

Over 2 years of Databento data (140,966 bars):
- Short IB Breakout: +$30,084 (177 trades, 45.2% WR, PF 1.511)
- Long IB Breakout: -$5,589 (dead weight across both years)

Longs consistently underperformed across all months and market conditions. The structural reasons:
1. **ES selloffs are faster than rallies**: More IB breakout distance covered per unit time
2. **Fear spikes volatility**: Short entries benefit from expanding ATR
3. **Rallies grind slowly**: Long entries face tighter ranges and more noise
4. **TEMA lag penalizes longs more**: By the time TEMA confirms bullish, the initial breakout move is often exhausted

## Filters

### Skip Fridays

Friday trades showed 30% win rate across all months, losing -$13,415 in the 1-year dataset. This is systematic, not noise:
- Friday is typically the lowest-volume day (liquidity dries up into the weekend)
- Position squaring and options expiration (0DTE) distort normal IB dynamics
- The Friday filter alone improved PF from 1.031 to 1.190

### Noon Blackout (12:00-13:00 ET)

The lunch hour showed 33.3% WR and -$10,320 in losses:
- Volume drops significantly during lunch
- IB breakouts that occur during lunch are more likely to be false breakouts
- Price chops around the IB edge without following through
- The blackout filter improved PF from 1.031 to 1.143

### Volatility Filter (on VA Fade only)

ATR relative to 50-bar average must be between 0.5x and 2.0x. This prevents VA Fade entries during dead markets (no edge) or chaotic markets (stops get blown). Intentionally NOT applied to IB Breakout because high volatility IS the trend day signal.

## Position Sizing

1 contract of ES futures. This is the natural starting position:
- Multi-contract scaling is confirmed pure linear (2x contracts = exactly 2x P&L, 2x DD)
- Position sizing is a capital decision, not a strategy parameter
- The recommendation is to start with 1 contract and scale after live validation
