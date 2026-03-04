# Strategy Review Package — March 1, 2026

5 new strategies developed autonomously. 4 Pine Script (SPY, ready for TradingView) + 1 Python backtester (SPX 0DTE iron condors).


## STRATEGY 1: VWAP Mean Reversion
**File**: `spy-vwap-mean-reversion.pine` (378 lines)
**Type**: Mean reversion at VWAP standard deviation bands
**Direction**: Both long and short

**How it works:**
- Computes session VWAP with 1x and 2x standard deviation bands
- LONG when price touches lower band + RSI oversold + bounces back
- SHORT when price touches upper band + RSI overbought + fails
- Band 2 (extreme) = immediate entry, Band 1 = needs confirmation candle
- Target: VWAP (full mean reversion)
- Stop: 2.5x stdev beyond entry band, capped at 1.5% of price
- ADX < 40 filter (skip strong trends — mean reversion doesn't work there)

**Expected frequency**: 0.5-2 signals/day (~10-25 trades/month)
**Best conditions**: Range-bound, rotational days
**Worst conditions**: Strong trend days (ADX filter blocks)

**Paste into TradingView on SPY 5m chart. Set date range to 5 years.**


## STRATEGY 2: Opening Range Breakout (ORB-15)
**File**: `spy-orb-breakout.pine` (465 lines)
**Type**: Breakout of first 15-minute range with volume confirmation
**Direction**: Both long and short

**How it works:**
- Tracks high/low of first 3 bars (15 min) after RTH open
- LONG when price breaks above OR high with volume > 1.3x average
- SHORT when price breaks below OR low with same volume gate
- Optional retest entry (default ON) — second chance after pullback to OR level
- Stop: OR midpoint (aggressive, per your preference) or opposite side (conservative)
- Target: 1x OR range extension beyond breakout level
- Cutoff: no new entries after 11:30 ET (edge decays after lunch)
- Gap filter: skip if overnight gap > 1%
- OR width filter: skip if too narrow (< 0.1%) or too wide (> 1.5%)

**Expected frequency**: 0.5-1.5 signals/day
**Best conditions**: Directional days with clear opening conviction
**Worst conditions**: Gap days, narrow/choppy opens

**Paste into TradingView on SPY 5m chart. Set date range to 5 years.**


## STRATEGY 3: Previous Day Level Fade
**File**: `spy-level-fade.pine` (500 lines)
**Type**: Auction theory — fade at PDH, PDL, PDC
**Direction**: Both long and short

**How it works:**
- Tracks previous day's high (PDH), low (PDL), and close (PDC)
- PDH SHORT: price reaches PDH zone + bearish candle + RSI > 55 → fade short to VWAP
- PDL LONG: price reaches PDL zone + bullish candle + RSI < 45 → fade long to VWAP
- PDC FADE: after a gap, price returns to previous close → fade the gap fill
- One trade per level per day (PDH/PDL/PDC flags prevent re-trading broken levels)
- Priority: PDH > PDL > PDC
- Stop: percentage-based buffer beyond level (0.3% default)
- Target: session VWAP (developing value = natural magnet)
- Volume confirmation required (bar volume > 0.8x average)

**Expected frequency**: ~1 trade every 2-3 days (~65-120 trades/year)
**Best conditions**: Range-bound days where price oscillates between prior levels
**Worst conditions**: Strong trend days where levels get blown through

**Paste into TradingView on SPY 5m chart. Set date range to 5 years.**


## STRATEGY 4: EMA Pullback in Trend
**File**: `spy-ema-pullback.pine` (500 lines)
**Type**: Momentum continuation — buy the dip / sell the rally
**Direction**: Both long and short

**How it works:**
- 3 EMAs: 9 (fast), 21 (medium), 50 (slow) establish trend
- LONG: EMA stack bullish (9>21>50) + price pulls back to EMA 21 zone + bounces
- SHORT: EMA stack bearish (9<21<50) + price rallies to EMA 21 zone + fails
- ADX > 20 confirms trending (skip choppy markets)
- DI+/DI- directional confirmation
- Pullback quality scoring (0-3): volume decreasing, orderly 3+ bar pullback, RSI sweet spot
- Stop: below EMA 50 + buffer (trend broken)
- Target: 2:1 R:R ratio (fixed, simple)
- Trailing stop activates after 1R in favor (ATR-based, 1.5x ATR)
- Noon blackout 12-13 ET default ON (lunch pullbacks are traps)
- Max 4 trades per day

**Expected frequency**: ~1 signal/day on average
**Best conditions**: Strong trending days with orderly pullbacks
**Worst conditions**: Choppy, range-bound, sideways markets

**Paste into TradingView on SPY 5m chart. Set date range to 5 years.**


## STRATEGY 5: 0DTE Break-Even Iron Condor (MEIC)
**Strategy**: `spx-options/backtester/strategies/breakeven_condor.py` (1,020 lines)
**Runner**: `spx-options/scripts/run_breakeven_condor.py` (428 lines)
**Type**: Premium selling — delta-neutral iron condors with break-even stops
**Direction**: Neutral (collect theta)

**How it works:**
- Sell 0DTE SPX iron condor (put credit spread + call credit spread)
- Short strikes at ~12 delta, wing width 30 points
- Equal premium on both sides (adjusts call wing width to match)
- Stop loss on each side = total premium collected (break-even stop)
- If one side stops: ~break even. Only lose on double stops (~8% historically)
- Multiple entries per day (~1 per hour, up to 7 per day)
- Entry requires "stabilization" — 3 consecutive 5m bars with closes within 3 SPX pts
- Close shorts at $0.05 (don't wait for expiration)
- Flatten all positions at 15:55 ET
- Risk limits: max 2% account risk per day, max 50% buying power usage

**Backtest results (5 years, $25K capital):**
- 5,829 trades, 91.9% WR, PF 14.39
- CAVEAT: These results are UNREALISTICALLY GOOD
- Root cause: Black-Scholes with daily VIX underestimates 0DTE option volatility
- Real 0DTE options have massive gamma — intraday moves create much larger
  P&L swings than the model captures
- John's real stats: 40% WR, 5.65% premium capture rate, ~8% double stop rate
- Our model: 91.9% WR, 80.66% capture rate, 0.05% double stop rate
- The INFRASTRUCTURE is correct (logic, tracking, stops, premium equalization)
- The PRICING needs real options data or better IV calibration

**To run:**
```
cd ~/projects/spx-options
source .venv/bin/activate
python scripts/run_breakeven_condor.py
python scripts/run_breakeven_condor.py --capital 50000 --target-delta 10
python scripts/run_breakeven_condor.py --start 2024-01-01 --end 2025-01-01
```

**Next steps for iron condor:**
- Get real SPX options intraday data (Databento OPRA or similar) for accurate pricing
- Or calibrate IV model: 0DTE IV should be 1.5-2x VIX, not 1.15x
- Or use realized volatility from 1-min bars to estimate intraday IV
- The framework is ready — just needs better pricing inputs


## HOW TO TEST (all 4 Pine Scripts)

1. Open TradingView → SPY chart → 5-minute timeframe
2. Open Pine Editor → paste script → Save → Add to chart
3. Strategy Tester → set date range (5 years available on SPY)
4. Check: total trades, PF, win rate, equity curve, max DD
5. If results look good → optimize inputs using TV's built-in optimizer
6. Paper trade winners before live

All scripts have:
- Dashboard in top-right corner showing real-time status
- All inputs with minval/maxval/step for TradingView optimizer
- Friday filter (default ON per trading rules)
- Flatten at 15:55 ET
- Alert conditions for future webhook automation
- No request.security() calls (no lookahead concerns)
- Percentage-based stops (scales with SPY price over 5 years)


## STRATEGY COMPARISON

| # | Strategy | Approach | Direction | Est. Freq | Best For |
|---|----------|----------|-----------|-----------|----------|
| 1 | VWAP MR | Mean reversion | Both | 10-25/mo | Range days |
| 2 | ORB-15 | Breakout | Both | 10-30/mo | Trend days |
| 3 | PDLevel Fade | Level rejection | Both | 5-10/mo | Range days |
| 4 | EMA Pullback | Trend continuation | Both | ~20/mo | Trend days |
| 5 | Iron Condor | Premium selling | Neutral | ~120/mo | All days |

Natural diversification: strategies 1+3 (mean reversion) complement 2+4 (momentum/breakout). Strategy 5 is market-neutral and works alongside all others.


## FILES CREATED

```
~/projects/ai-playground/
  spy-vwap-mean-reversion.pine   (378 lines)
  spy-orb-breakout.pine          (465 lines)
  spy-level-fade.pine            (500 lines)
  spy-ema-pullback.pine          (500 lines)
  strategy-review-package.md     (this file)

~/projects/spx-options/
  backtester/strategies/breakeven_condor.py  (1,020 lines)
  scripts/run_breakeven_condor.py            (428 lines)
  backtester/strategies/__init__.py          (updated — registered)
```

Total: 3,291 lines of new code across 6 files.
