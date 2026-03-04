# AMT-TEMA v8 -- Pine Script & Automation Notes

## Pine Script Implementation

### File: `amt-tema-strategy.pine`

Pine Script v6 strategy, 571 lines, fully implemented and live on TradingView.

### Strategy Declaration

```pine
strategy(
    title="AMT-TEMA v8",
    shorttitle="AMT-TEMA-v8",
    overlay=true,
    pyramiding=0,
    initial_capital=100000,
    currency=currency.USD,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    commission_type=strategy.commission.cash_per_contract,
    commission_value=2.50,
    slippage=1,
    margin_long=5,
    margin_short=5
)
```

Key settings:
- `pyramiding=0`: One position at a time, no stacking
- `commission_value=2.50`: $2.50 per contract per side
- `slippage=1`: 1 tick slippage on entry/exit
- `margin_long=5, margin_short=5`: 5% margin (day trading margin for futures)

### Input Groups

All inputs include `minval/maxval/step` for TradingView's strategy optimizer.

| Group | Key Inputs |
|-------|-----------|
| TEMA | temaFastLen (9), temaSlowLen (21), temaTrendLen (55) |
| Session | tradeStart (1035), tradeEnd (1500), flattenTime (1555), skipFriday (true), blackoutStart/End (1200/1300), dirFilter ("Short Only") |
| Day Type | ibAvgLen (20), ibNarrowRatio (0.8), ibWideRatio (1.2), useDayType (true) |
| Value Area | vaStdevMult (1.0) |
| Risk | atrLen (14), cooldownBars (2) |
| Volatility | useVolFilter (true), atrAvgLen (50), volLowRatio (0.5), volHighRatio (2.0) |
| IB Breakout | useIBBreak (true), useTrendFilter (true), minIBRange (8), maxIBRange (80), ibStopType ("IB Mid"), ibMaxStopPts (20), ibMinTarget (10), ibPctStop (true), ibStopBps (30), maxIBTrades (2) |
| VA Fade | useVAFade (false), vaBuffer (4), vaStopMult (0.5), vaMinRR (0.5), maxVATrades (1) |
| 80% Rule | useEighty (false), eightyConfBars (6), eightyStopBuf (0.5), maxEightyTrades (1) |
| Automation | pmtToken, pmtAccount |

### Core Components

#### 1. TEMA Engine

Triple Exponential Moving Average computed via the standard formula:

```pine
temaCalc(src, len) =>
    e1 = ta.ema(src, len)
    e2 = ta.ema(e1, len)
    e3 = ta.ema(e2, len)
    3 * e1 - 3 * e2 + e3
```

Three TEMA lines: Fast (9), Slow (21), Trend (55).

Conditions derived:
- `temaBullish = temaFast > temaSlow`
- `temaBearish = temaFast < temaSlow`
- `trendUp = close > temaTrend`
- `trendDown = close < temaTrend`
- `temaSlope = temaFast - temaFast[3]` (momentum acceleration)

#### 2. Session & Time Detection

All time logic uses Eastern Time via `hour(time, "America/New_York")` and `minute(time, "America/New_York")`, combined into `etTime = etHour * 100 + etMinute`.

Key time gates:
- RTH: 9:30 -- 16:00 ET
- IB period: 9:30 -- 10:30 ET
- Trading window: 10:35 -- 15:00 ET (default)
- Blackout: 12:00 -- 13:00 ET
- Flatten: 15:55 ET
- Friday detection: `dayofweek(time, "America/New_York") == dayofweek.friday`

#### 3. Initial Balance Tracker

Uses `var` variables that persist across bars:
- `ibHigh`, `ibLow`: Updated during IB period (9:30-10:30), frozen after
- `ibDone`: Set to true when IB period ends
- `ibRange = ibHigh - ibLow`
- `ibMid = (ibHigh + ibLow) / 2`

#### 4. VWAP-Based Value Area

Running VWAP computed from RTH bars:
- `rthVwapSum += hlc3 * volume`
- `rthVolSum += volume`
- Session VWAP = rthVwapSum / rthVolSum
- Standard deviation accumulated via running `rthSqDev`
- At new RTH: previous session's VWAP becomes `prevPOC`, VWAP +/- stdev becomes `prevVAH`/`prevVAL`

#### 5. Day Type Classification

`ibRatio = ibRange / ibRangeAvg` (20-day EMA of IB ranges)
- Narrow: ibRatio < 0.8
- Wide: ibRatio > 1.2
- Normal: in between

#### 6. Volatility Filter

`volRatio = atrVal / atrAvg` (ATR(14) / SMA(ATR, 50))
- `volOK = volRatio >= 0.5 and volRatio <= 2.0`
- Applied to VA Fade only (intentionally removed from IB Breakout)

### Signal Logic (IB Breakout Short)

```pine
ibShortSignal = allowShorts
    and useIBBreak
    and ibCrossDown           // ta.crossunder(close, ibLow)
    and temaBearish           // TEMA 9 < TEMA 21
    and ibTrendOK_S           // close < TEMA 55 (if useTrendFilter)
    and ibValid               // 8 <= ibRange <= 80
    and isTradingWindow       // 10:35 - 15:00 ET
    and entryAllowed          // not Friday, not in blackout
    and pastCooldown          // 2+ bars since last exit
    and strategy.position_size == 0
    and ibTradesS < maxIBTrades  // < 2 short trades today
```

### Stop/Target Computation

```pine
// Percentage-based stop (v8)
ibMaxStopDyn = ibPctStop ? close * ibStopBps / 10000.0 : ibMaxStopPts

// Stop at IB Mid, capped at percentage-based maximum
ibRawSL_S = ibStopType == "IB Mid" ? ibMid : ...
ibSL_S = math.min(ibRawSL_S, close + ibMaxStopDyn)

// Target at IB extension (or minimum target floor)
ibTP_S = close - math.max(ibRange, ibMinTarget)
```

### Execution

```pine
if ibShortSignal
    strategy.entry("IB_S", strategy.short, alert_message=pmtMsg("sell", ibSL_S, ibTP_S))
    strategy.exit("IB_SX", "IB_S", stop=ibSL_S, limit=ibTP_S, alert_message=pmtCloseMsg())
```

### Session Flatten

```pine
if etTime >= flattenTime and strategy.position_size != 0
    strategy.close_all("RTH End", alert_message=pmtCloseMsg())
```

## Automation -- PickMyTrade Integration

### Architecture

```
TradingView Alert (Pine Script strategy)
    --> Webhook POST to PickMyTrade API
        --> PickMyTrade routes to Tradovate
            --> Bracket order placed (Entry + SL + TP)
```

### PickMyTrade Configuration

| Setting | Value |
|---------|-------|
| Token | `k170b15680ae5ebcff062d` |
| Webhook URL | `https://api.pickmytrade.trade/v2/add-trade-data-latest?t=11215` |
| Symbol Mapping | ES1! -> ESH6 (March 2026 contract) |
| Order Type | Market (MKT) |
| Quantity | 1 |
| Lot Size | 50 |
| Override Mapping | No |
| Account | DEMO2361007 ($50K sim on Tradovate) |

### JSON Payload Format

The `pmtMsg()` function builds the full JSON payload:

```json
{
    "symbol": "ES1!",
    "data": "sell",
    "quantity": "1",
    "risk_percentage": 0,
    "price": "6800.50",
    "tp": 6788.50,
    "percentage_tp": 0,
    "dollar_tp": 0,
    "sl": 6820.50,
    "dollar_sl": 0,
    "percentage_sl": 0,
    "trail": 0,
    "trail_stop": 0,
    "trail_trigger": 0,
    "trail_freq": 0,
    "update_tp": false,
    "update_sl": false,
    "breakeven": 0,
    "breakeven_offset": 0,
    "token": "k170b15680ae5ebcff062d",
    "pyramid": false,
    "same_direction_ignore": false,
    "reverse_order_close": false,
    "multiple_accounts": [{
        "token": "k170b15680ae5ebcff062d",
        "account_id": "DEMO2361007",
        "risk_percentage": 0,
        "quantity_multiplier": 1
    }]
}
```

**CRITICAL**: ALL fields must be present. PickMyTrade rejects the payload as "Invalid Alert Data Json" if any field is missing. Simplified JSON does NOT work.

### Close/Flatten Payload

The `pmtCloseMsg()` function builds the close payload:
- `"data": "close"`
- All SL/TP fields set to 0
- Same account/token structure

### TradingView Alert Configuration

| Setting | Value |
|---------|-------|
| Condition | AMT-TEMA-v8 |
| Timeframe | 5m |
| Trigger | Once Per Bar Close |
| Webhook | Enabled, URL = PMT webhook URL |
| Message | `{{strategy.order.alert_message}}` |
| Expiry | March 20, 2026 (RENEW BEFORE THIS DATE) |

The `{{strategy.order.alert_message}}` placeholder is replaced by TradingView with the JSON from `alert_message` parameter in `strategy.entry()` or `strategy.close_all()`.

### Anti-Repainting Settings

The Pine Script does NOT use `calc_on_every_tick` (defaults to false). Entries fire on bar close only. This means:
- Signals are confirmed at the END of each 5-minute bar
- No mid-bar signal generation that could repaint
- TradingView alert fires at bar close, not tick-by-tick

## Contract Rollover

ES contracts roll quarterly:
- ESH6 = March 2026
- ESM6 = June 2026
- ESU6 = September 2026
- ESZ6 = December 2026

Before each rollover:
1. Update PickMyTrade symbol mapping: ESH6 -> ESM6
2. No Pine Script change needed (uses ES1! continuous contract)
3. IB range history resets with new contract (minor impact)

**Next rollover**: March 2026 -- update ESH6 -> ESM6 in PMT settings before expiry.

## Visual Dashboard

The Pine Script includes a 2-column, 18-row dashboard table in the top-right corner:

| Row | Label | Content |
|-----|-------|---------|
| 0 | AMT-TEMA v8 | Header (orange background) |
| 1 | TEMA | BULL / BEAR |
| 2 | Trend | UP / DOWN |
| 3 | Slope | Numeric value |
| 4 | IB Range | Points + valid indicator |
| 5 | Day Type | NARROW / NORMAL / WIDE + ratio |
| 6 | IB Status | ABOVE / INSIDE / BELOW |
| 7 | vs VA | ABOVE / INSIDE / BELOW |
| 8 | POC (VWAP) | Numeric value |
| 9 | Vol Regime | Ratio + OK/blocked |
| 10 | Session | IB / ACTIVE / RTH / CLOSED |
| 11 | Filters | FRI-OFF / BLACKOUT / CLEAR |
| 12 | Direction | Short Only / Both / Long Only |
| 13 | Max Stop | BPS value + dynamic points |
| 14 | Cooldown | Bars since exit / required |
| 15 | 80% Rule | OFF / INACTIVE / COUNTING / CONFIRMED |
| 16 | Trades Today | IB: / VA: / 80: counts |
| 17 | Capital | Current equity |

## Chart Visuals

| Element | Description |
|---------|-------------|
| TEMA Fast (9) | Lime line, width 1 |
| TEMA Slow (21) | Orange line, width 1 |
| TEMA Trend (55) | Yellow line, width 2 |
| IB High/Low | Blue step lines, width 2 |
| IB Mid | Blue step line (faded), width 1 |
| Prev VAH | Red step line (faded), width 1 |
| Prev VAL | Green step line (faded), width 1 |
| Prev POC | White step line (faded), width 1 |
| VWAP | Purple line, width 2 |
| IB Long signal | Green triangle up below bar |
| IB Short signal | Red triangle down above bar |
| VA Long signal | Green circle below bar |
| VA Short signal | Orange circle above bar |
| 80% Long signal | Aqua diamond below bar |
| 80% Short signal | Fuchsia diamond above bar |
| Background | Gray (non-RTH), Blue (IB), Lime/Orange (narrow/wide day type) |

## Other Platform Conversions

### thinkorSwim Study

File: `amt-tema-strategy-tos.ts` -- v6 conversion (Feb 13, 2026)
- Study overlay, not strategy (thinkorSwim limitations)
- Uses `RegularTradingStart()` for RTH/IB detection
- Dashboard: vertical bubbles in expansion area
- **Known issue**: Entry/exit signal marker colors always grey (thinkorSwim overrides plot colors)

### SPX Fork

File: `amt-tema-strategy-spx.pine` -- same logic as ES v8, adapted for SPX options signals
- `commission_value=0, slippage=0` (options context)
- `vaBuffer=100` ticks (wider for SPX price levels)
- Title/dashboard updated to "AMT-TEMA SPX v8"

## Monthly Cost

| Item | Cost |
|------|------|
| TradingView Premium | Already subscribed |
| PickMyTrade | $50/mo |
| Tradovate | $0 (commissions only) |
| Commissions (~88 trades/yr x $5 RT) | ~$36/mo |
| **Total** | **~$86/mo** |

At v8's average net P&L of ~$15,042/yr ($1,254/mo), the monthly cost of $86 is 6.9% of average monthly income. This is acceptable but represents a fixed cost that must be covered even during drawdowns.
