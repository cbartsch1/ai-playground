# Plan: Convert AMT-TEMA v6 to thinkorSwim

## Status: PHASE 1-4 COMPLETE (Feb 13, 2026)

## Output
- **`/Users/chuck_mf_norris/projects/ai-playground/amt-tema-strategy-tos.ts`** — thinkScript study
- Built as study overlay (not strategy) for live chart signals + dashboard
- Targets /ES 5m on thinkorSwim (same as Pine Script original)

## What was built
- Full 1:1 conversion of all 3 setups (IB Breakout, VA Fade, 80% Rule)
- TEMA engine, IB tracker, VWAP-based VA, day type classification, volatility filter
- Position tracking via `rec posDir` with stop/target levels and cooldown
- Timezone-safe time handling via `RegularTradingStart()` — works regardless of chart TZ
- Dashboard: 11 AddLabel() calls matching Pine Script table
- Alerts: distinct sounds per signal type
- Visuals: TEMA lines, IB/VA/POC levels, entry arrows, stop/target dots, exit squares

## What's NOT yet done (Phase 5: SPX Optimization)
- SPX-specific parameter recalibration (if user wants SPX version later)
- Volume handling for SPX synthetic volume
- Gap adjustment for RTH-only SPX data

## Source
- `/Users/chuck_mf_norris/projects/ai-playground/amt-tema-strategy.pine` (AMT-TEMA v6)
- Setups: IB Breakout + VA Fade (80% Rule included but default OFF)

## Key Differences: Pine Script vs thinkScript

### Language
| Feature | Pine Script | thinkScript |
|---------|-------------|-------------|
| Syntax | C-like, `//@version=6` | Java-like, `declare` statements |
| Variables | `var float x = na` | `def x = ...` (no persistent state by default) |
| Persistent state | `var` keyword | `rec` (recursive) keyword |
| Strategy orders | `strategy.entry()`, `strategy.exit()` | `AddOrder()` |
| Plots | `plot()`, `plotshape()` | `plot`, `AddLabel()`, `AddCloud()` |
| Tables/Dashboard | `table.new()` + `table.cell()` | `AddLabel()` (no tables — use labels instead) |
| Time functions | `hour(time, "America/New_York")` | `GetTime()`, `RegularTradingStart()` |
| Bar state | `barstate.islast` | `IsNaN(close[-1])` or `BarNumber()` |

### SPX vs ES Differences
| Aspect | ES (E-mini S&P 500) | SPX (S&P 500 Index) |
|--------|---------------------|---------------------|
| Type | Futures contract | Cash index |
| Trading hours | Nearly 24h (ETH+RTH) | RTH only (9:30-16:00 ET) |
| Tick size | 0.25 pts ($12.50) | 0.01 pts |
| Commission | $2.50/contract | N/A (index — trade via options or SPY) |
| Volume | Real futures volume | Synthetic/calculated |
| Contract rolls | Quarterly (ESH, ESM...) | No rolls — continuous |
| Overnight gaps | Minimal (trades overnight) | Daily gaps (RTH only) |
| Point value | $50/pt | N/A (or $100/pt for SPX options) |

### Strategy Implications for SPX
1. **No ETH data** — RTH-only simplifies session logic but means IB is always the first hour
2. **No commission modeling** — thinkorSwim backtester handles this differently
3. **Volume profile** — SPX volume is synthetic; may need to use SPY volume as proxy or skip volume-based filters
4. **VWAP** — thinkorSwim has built-in VWAP; SPX VWAP may be less meaningful without real volume
5. **Gaps** — SPX gaps overnight, so "open outside VA" for 80% Rule may trigger more often
6. **ATR levels** — SPX and ES track very closely in points but ATR-based stops need recalibration
7. **No slippage concern** — if using as signal overlay (not auto-trading), slippage is moot

## Implementation Plan

### Phase 1: Core Translation (thinkScript skeleton)
1. **Header & declarations** — `declare lower` or `declare once_per_bar`, strategy setup with `AddOrder()`
2. **TEMA Engine** — translate `temaCalc()` function (thinkScript uses `def` and `fold` or inline)
3. **Session & Time** — `GetTime()`, `SecondsFromTime()`, `SecondsTillTime()` for ET-based windows
4. **IB Tracker** — `rec` variables for ibHigh/ibLow/ibDone, reset on new session via `GetDay() != GetDay()[1]`
5. **Value Area** — Translate VWAP-based VA (or use thinkorSwim's built-in `VolumeProfile` study)
6. **ATR & Volatility** — `ATR()` is built-in, straightforward translation

### Phase 2: Setups
1. **Setup 1: IB Breakout** — `AddOrder(OrderType.BUY_TO_OPEN, ...)` with stop/target
2. **Setup 2: VA Fade** — Same pattern with VA touch detection
3. **Setup 3: 80% Rule** — Keep in code, default disabled (same as Pine version)
4. **Day Type Classification** — Translate IB width ratio logic

### Phase 3: Risk & Execution
1. **Stop/target orders** — thinkorSwim uses `AddOrder()` for each leg
2. **Cooldown** — `rec barsSinceExit` counter
3. **Session flatten** — `AddOrder()` at flatten time
4. **Max trades per day** — `rec` counters, reset on new day

### Phase 4: Visuals
1. **TEMA lines** — `plot temaFast`, `plot temaSlow`, `plot temaTrend`
2. **IB levels** — `plot` with `PaintingStrategy.HORIZONTAL`
3. **VA levels** — `plot` with step-line style
4. **Entry signals** — `AddVerticalLine()` or `plot` with arrows
5. **Dashboard** — `AddLabel()` calls (thinkorSwim doesn't support tables)

### Phase 5: SPX Optimization
1. **Recalibrate all parameters** — IB range, ATR multipliers, VA buffer in SPX points
2. **Adjust for daily gaps** — SPX gaps may affect IB/VA calculations
3. **Volume handling** — Decide: use SPX synthetic volume, proxy from SPY, or remove volume-dependent features
4. **Backtest and compare** — Run on SPX 5m, compare metrics to ES baseline
5. **Parameter sweep** — Same 22-test approach used for Pine Script optimization

## Challenges & Risks
- **thinkScript backtester limitations** — Less flexible than TradingView's; no pyramiding control, limited order types
- **No optimizer** — thinkorSwim doesn't have TradingView's parameter optimizer; manual sweeps only
- **Volume data quality** — SPX volume is not real market volume
- **Execution gap** — thinkorSwim strategies are harder to forward-test vs TradingView alerts

## Output File
- `/Users/chuck_mf_norris/projects/ai-playground/amt-tema-strategy-tos.ts` (thinkScript)

## Dependencies
- Need AMT-TEMA Pine Script v6 as reference (already at `amt-tema-strategy.pine`)
- thinkorSwim platform access for testing
- SPX 5m chart data in thinkorSwim
