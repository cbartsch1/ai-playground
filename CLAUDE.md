# AI Playground Project

## Overview
Multi-AI workflow project for ES/NQ futures scalping strategies in Pine Script v6.
**Machine**: M3 Ultra, 96 GB RAM. See `cheatsheet.md` for full team setup.

## The AI Team (7 members)
- `claude` — Claude Opus 4.6 (cloud) — Architecture, debugging, agentic tasks
- `claude-local` — GLM4.7-Flash 30B (local/free) — General tasks, boilerplate
- `claude-coder` — Qwen3-Coder 30B (local/free) — Python, pandas, trading logic
- `gemini` — Gemini 2.5 (cloud) — Complex tasks, second opinion
- `gemini-local` — GLM4.7-Flash via LiteLLM (local/free) — General tasks
- `grok` — Grok 4 (cloud) — Heavy reasoning, complex refactors
- GLM Chat — Desktop app (local/free) — Quick chat without terminal

**Rule of thumb**: Local models for boilerplate/simple edits. Cloud models for strategy architecture, debugging, and reviews. Claude Opus found 4/4 real bugs in strategy review test; local models found 0/4.

## Project Structure
- Root directory for experimentation and testing multi-AI workflows
- `cheatsheet.md` — Full AI team reference, commands, services, troubleshooting
- `research/` — Shared research archive (all team members read before working)
  - `auction-theory-tema-research.md` — AMT + TEMA concepts, Pine Script implementation notes, strategy setups

### Pine Script Strategies

**Original strategies:**
- `proven-scalper.pine` — Simple HTF EMA bias + EMA crossover + RSI + VWAP on 1m chart (has lookahead_on bias)
- `ichimoku-rsi-v2.pine` — Ichimoku cloud + TK cross + Chikou + RSI with Kijun trailing stop on 5m chart
- `futures-scalper.pine` through `futures-scalper-v8.pine` — Earlier iterations
- `ichimoku-rsi-v1.pine` — First Ichimoku version
- `claude-strategy.pine`, `gemini-strategy.pine`, `combined-strategy.pine` — AI-generated experiments

**Optimization-ready versions (archived):**
- `proven-scalper-opt.pine` — Proven Scalper with minval/maxval/step on all inputs. **Fixed: lookahead_off**.
- `ichimoku-rsi-v2-opt.pine` — Ichimoku RSI v2 with full parameter sweep ranges.
- `hybrid-scalper-ichimoku.pine` — HYBRID v7 (abandoned — structure breaks have no edge on ES 5m)

**Current flagship:**
- `amt-tema-strategy-v9.pine` — **AMT-TEMA v9** — Exit overhaul + TEMA Cross entry
  - **Setup 1**: IB Breakout + TEMA (continuous re-arm, scaled TP capped at ATR mult)
  - **Setup 2**: Value Area Fade + TEMA Slope (default OFF)
  - **Setup 3**: 80% Rule + TEMA (default OFF)
  - **Setup 4**: TEMA Cross Short (momentum short on TEMA crossdown, day-type gated, default ON)
  - ATR trailing stop (trigger + distance in bps), TEMA exit (close on TEMA cross)
  - PMT JSON with trail fields for PickMyTrade
  - All inputs optimizable, lookahead_off, Pine Script v6, designed for ES 5m chart
  - **NEEDS BACKTESTING** — not yet validated on 2yr data
- `amt-tema-strategy.pine` — **AMT-TEMA v8** — Proven baseline (short-only, 30bps pct stop)
  - 183 trades, PF 1.430, p=0.047 (t-test) — restored baseline (commit ed86b73, Apr 1 2026: two unreverted AR experiment filters removed)
  - The 177t / PF 1.511 / p=0.028 figures below in Version History are the original Feb 2026 measurement; current committed code reproduces 183t / PF 1.430 / p=0.047
  - **Automation**: PickMyTrade webhook integration (Automation input group: token + account ID)
  - `pmtMsg()` / `pmtCloseMsg()` build full PMT JSON with SL/TP as absolute prices
  - `alert_message` on all `strategy.entry()` and `strategy.close_all()` calls
- `amt-tema-test-ib.pine` — IB Breakout isolated test file
- `amt-tema-test-va.pine` — VA Fade isolated test file
- `amt-tema-test-tx.pine` — TEMA Crossover isolated test file (proved unprofitable)

**SPX Signal Adapter:**
- `amt-tema-strategy-spx.pine` — **AMT-TEMA SPX v8** — Fork of ES strategy for SPX options signals
  - Same logic as ES v8 (short-only, 30bps pct stop, Friday filter, noon blackout)
  - `commission_value=0, slippage=0` (options, no per-contract commission)
  - `vaBuffer=100` ticks (wider buffer for SPX price levels)
  - Title: "AMT-TEMA SPX v8", dashboard: "AMT-TEMA SPX v8"

### Automated Execution — Path 1: TradingView → PickMyTrade → Tradovate (Feb 15, 2026)
- **Status**: LIVE ON DEMO — webhook chain fully tested and verified
- **Flow**: Pine Script `strategy.entry()` fires TradingView alert → webhook sends JSON to PickMyTrade → PMT routes bracket order (entry + SL + TP) to Tradovate
- **PickMyTrade**: app.pickmytrade.trade, 8-day demo trial
  - Token: `k170b15680ae5ebcff062d`
  - Webhook URL: `https://api.pickmytrade.trade/v2/add-trade-data-latest?t=11215`
  - Settings: ES1! → ESH6 (March 2026), MKT, Qty 1, Lot Size 50, no override mapping
  - Tradovate Account: DEMO2379660 ($50K sim)
- **TradingView Alert**: Condition AMT-TEMA-v8, 5m, Once Per Bar Close, webhook enabled
  - Message: `{{strategy.order.alert_message}}`
  - Expiration: March 15, 2026 (renew before then)
- **PMT JSON format**: Must include ALL fields (tp, sl, trail, breakeven, percentage_sl, etc.) or PMT rejects as "Invalid Alert Data Json". Simplified JSON does NOT work.
- **Test results** (Feb 15): Sell order placed with bracket (Entry ID + LMT ID + STP ID), close/flatten verified
- **Contract rollover**: Manual — update ESH6 → ESM6 in PMT Settings before March expiry
- **Monthly cost**: TradingView Premium (already have) + PickMyTrade $50 + Tradovate $0 + commissions ~$36 = ~$86/mo
- **Path 2 (future)**: Python bot + IBKR API — 90% of logic already in backtester, cheapest long-term

### thinkorSwim Conversion
- `amt-tema-strategy-tos.ts` — **AMT-TEMA v6 thinkScript study** (Feb 13, 2026)
  - 1:1 conversion of Pine Script v6 → thinkScript for thinkorSwim
  - Study overlay (not strategy) — signals, levels, dashboard on chart
  - Timezone-safe: uses `RegularTradingStart()` for RTH/IB detection
  - Position tracking via `rec posDir` with stop/target levels
  - **WORKING**: TEMA lines, IB/VA levels, stop/target dots, dashboard (vertical bubbles in expansion area), alerts
  - **BROKEN**: Entry/exit signal markers — arrow plot colors always grey (thinkorSwim overrides). AddChartBubble too small. See `thinkscript-issues.md` for full details and untried approaches.
  - Dashboard: vertical bubbles in chart expansion area (upper-right)
  - All same inputs/defaults as Pine Script version
  - See `plans/thinkorswim-spx-conversion.md` for conversion reference
  - See `thinkscript-issues.md` for open bugs and thinkScript gotchas

## Version History

### Backtest Results (Oct 26 2025 – Feb 12 2026, ES 5m)

| Version | P&L | Trades | Win Rate | PF | Drawdown |
|---------|------|--------|----------|------|----------|
| v4 | -$1,557 | 309 | 41.1% | 0.937 | $4,750 |
| v5 | -$1,380 | 191 | 34.0% | 0.916 | $3,787 |
| v5.1 | -$2,610 | 207 | 37.7% | 0.844 | $4,390 |
| v6 | -$1,762 | 45 | 28.9% | 0.574 | $2,142 |

**Key finding**: v4 had the best win rate (41.1%) and profit factor (0.937). Its only problem was commission drag from 309 trades (~$1,545). v6's pullback mechanism destroyed win rate.

### HYBRID v4 → v5 (Feb 12, 2026)
- Structure Lookback: 5 → 8, Cooldown: 3 → 6, Candle Strength: 0.6 → 0.7
- Added EMA Alignment gate, ATR filter, Smart Kijun trail
- TP: 16 → 20 ticks, tighter sessions (11:30/15:30)
- Result: Trades dropped to 191 but WR fell to 34% — filters too restrictive

### HYBRID v6 (Feb 12, 2026)
- Replaced direct breakout with Breakout+Pullback entry (arm on break, wait for pullback to Kijun)
- Result: Only 45 trades but 28.9% WR, PF 0.574 — pullback too restrictive, price falls through Kijun

### HYBRID v7 (Feb 12, 2026)
- Return to v4's direct breakout + v5's trade-reduction filters
- Result on ES: -$3,968, 246 trades, 37% WR, PF 0.803 — WORSE than v4
- Result on RTY: -$1,350, 223 trades, 39.9% WR, PF 0.819
- **Conclusion: Structure break entry does not have enough edge on ES 5m. Approach abandoned.**

### NEW DIRECTION: Auction Market Theory + TEMA (Feb 12, 2026)
**Decision:** Pivot from structure break entries to Auction Market Theory (AMT) foundation.
ES is most profitably traded using auction theory. TEMA (Triple EMA) for trend/momentum confirmation.

**Research archive:** `research/auction-theory-tema-research.md` (full details, all team members read this)

**Core concepts being implemented:**
1. **Value Area** (VAH/VAL/POC from volume profile) — where 70% of volume trades
2. **Initial Balance** (IB) — first hour's range (9:30-10:30 ET)
3. **TEMA crossover** — TEMA(9)/TEMA(21) for signals, TEMA(55) as trend filter
4. **Day type classification** — balance vs. trend days

**Planned setups (prioritized):**
1. IB Breakout + TEMA Confirmation (trend-day setup) — IMPLEMENTED v1
2. Value Area Fade + TEMA Slope (mean-reversion setup) — IMPLEMENTED v1
3. 80% Rule + TEMA Direction (high-probability setup) — TODO v2
4. TEMA Trend + Auction Context (adaptive setup) — TODO v2

### AMT-TEMA Results (Oct 26 2025 – Feb 12 2026, ES 5m, 1 contract)

| Version | P&L | Trades | Win Rate | PF | Max DD |
|---------|------|--------|----------|------|--------|
| v1 (3 setups) | +$3,850 | 55 | 34.55% | 1.279 | $4,630 |
| v5 (IB+VA only) | +$11,313 | 45 | 53.33% | 2.358 | $2,238 |
| v5.1 (optimized) | +$12,315 | 47 | 55.32% | 2.427 | $2,198 |
| **v5.2 (IB max=2)** | **+$13,365** | **~49** | **57.5%** | **~2.5** | **$2,198** |
| v6 (80% Rule ON) | +$9,890 | 62 | 53.2% | 1.685 | $3,790 |
| **v6 (80% Rule OFF)** | **= v5.2** | **~49** | **57.5%** | **~2.5** | **$2,198** |

**v6 finding**: 80% Rule added ~13 trades but lost ~-$3,475 net (-$267/trade avg). VWAP-based VA proxy too imprecise for tight stop/target placement. Setup kept in code (default OFF) pending real volume profile VA improvement.

**Key discovery**: Isolated testing revealed TEMA Crossover (Setup 3) was losing -$3,313 while VA Fade made +$8,993. Removing it flipped the strategy from marginal to strong. ADX-based setups (4 approaches tested) all degraded performance.

### v6.1 — Fix Missing Trend Day Trades (Feb 13, 2026 — IN PROGRESS)

**Problem**: Feb 11-13, ES moved 180+ pts but v6 took only 1 trade (-$580). IB Range=60.25pt on Feb 12 was blocked by maxIBRange=25 AND Day Type filter AND volatility filter.

**Changes applied to code (file is current with these):**
1. `maxIBRange` 25 → 80 (maxval 120) — capture wide IB trend days
2. `ibMaxStopPts` = 20pt (new input) — caps IB Mid stop risk on wide IB
3. `ibMinTarget` = 10pt (new input) — minimum target floor for narrow IB
4. `ibDayTypeOK = true` — removed Day Type gate on IB Breakout (wide IB breakouts are strongest AMT pattern)
5. Removed `volOK` from IB Breakout signals — high vol IS the trend, don't filter it (VA still has volOK)

**Failed approach (reverted):**
- Persistent breakout state (replace `ta.crossover`/`ta.crossunder` with armed flag + extended check) → 103 trades, PF 0.987, +$732, DD $12.5K. Added garbage trades on normal days AND still missed Feb 12. Reverted to crossover/crossunder.

**Status: NOT YET BACKTESTED on full 3M data.** Last test was 30-day window.

**Remaining suspected blockers (if Feb 12 still missed after full backtest):**
- `temaBearish` (TEMA 9 < TEMA 21) — may not be true on the exact crossunder bar during sudden moves
- `ibTrendOK_S` (close < TEMA 55) — TEMA 55 lags behind sudden reversals
- KEY INSIGHT: `ta.crossunder` is one-shot. If ANY filter is false on that exact bar, trade is missed forever.
- Possible fix: add small window (6 bars/30min) after crossunder for TEMA confirmation
- Other idea: add dashboard debug row showing which filter blocked last IB signal attempt

### v7 — Data-Driven Optimization from 1-Year Backtest (Feb 15, 2026)

**Problem**: v5.2's +$13,365 / PF 2.5 was based on TradingView's 3.5-month window (overfit). Full 1-year Python backtest on Databento data showed only +$3,714, PF 1.031, 305 trades, $14.8K DD.

**Python Backtester**: Built exact Pine Script replica, validated TEMA/ATR indicators match 100%, ran on 70,331 bars of real ES 5-min data (Feb 2025 - Feb 2026, Databento CME GLBX.MDP3). 31 unit tests passing.

**Analysis findings (305 trades over 12 months):**
- VA Fade: -$2,825 (PF 0.835, 33.9% WR) — VWAP proxy too imprecise over full year
- Friday trades: -$13,415 (30% WR) — systematic across all months, not noise
- Noon hour (12:00-13:00): -$10,320 (33.3% WR) — lunch hour whipsaws
- IB SHORT (+$8,369) outperforms IB LONG (-$1,830) by $10K
- Wider stops (22pt, 25pt) made things WORSE — 20pt cap is optimal
- Flatten exits are the profit engine: +$23,310 from 63 trades (66.7% WR)

**A/B testing results (each change tested independently on same 1-year dataset):**

| Variant | Trades | WR | PF | P&L | Max DD | Sharpe |
|---------|--------|------|------|------|--------|--------|
| Baseline (v6.1) | 305 | 40.3% | 1.031 | +$3,714 | $14,854 | 0.20 |
| No VA | 251 | 43.0% | 1.119 | +$12,351 | $11,595 | 0.75 |
| Noon Blackout | 266 | 41.4% | 1.143 | +$14,406 | $11,007 | 0.86 |
| No Friday | 245 | 42.9% | 1.190 | +$17,129 | $8,511 | 1.12 |
| **Combined (v7)** | **174** | **47.1%** | **1.443** | **+$28,936** | **$6,145** | **2.41** |

**v7 changes applied to Pine Script:**
1. VA Fade default OFF (code retained, just `useVAFade=false`)
2. Friday filter ON (`skipFriday=true`, new input)
3. Noon blackout 12:00-13:00 (`blackoutStart=1200, blackoutEnd=1300`, new inputs)
4. Dashboard row "Filters" shows FRI-OFF / BLACKOUT / CLEAR status
5. `entryAllowed` gate added to all 3 setup signal conditions

**Monthly consistency (v7)**: 9 winning months, 4 losing (May -$4.3K, Sep -$4K, Dec -$0.6K, Feb -$0.4K).

**Walk-forward validation (Feb 15, 2026) — PASS:**

| Period | Config | Trades | WR | PF | P&L | Max DD | Sharpe |
|--------|--------|--------|------|------|------|--------|--------|
| In-sample (Feb-Aug 25) | Baseline | 166 | 39.8% | 1.059 | +$4,038 | $14,854 | 0.37 |
| In-sample (Feb-Aug 25) | v7 | 102 | 49.0% | 1.597 | +$22,909 | $5,168 | 3.03 |
| Out-of-sample (Aug 25-Feb 26) | Baseline | 138 | 41.3% | 0.998 | -$101 | $8,148 | -0.01 |
| Out-of-sample (Aug 25-Feb 26) | v7 | 72 | 44.4% | 1.224 | +$6,028 | $6,145 | 1.37 |

PF ratio (out/in) = 0.77 (>0.7 = robust). v7 filters are structural market patterns, not overfit.

**HOWEVER**: v7 failed 2-year validation. Year 2 (Feb 2024-2025) lost -$10,491 (p=0.319). Combined 2yr: +$18,445, PF 1.142, p=0.305 — NOT SIGNIFICANT. Longs dragged performance, fixed 20pt stop didn't scale with ES price (5000→6800).

### v8 — Short-Only + Percentage Stop (Feb 15, 2026)

**Problem**: v7's edge didn't survive 2-year validation. Root causes:
1. Longs lost -$5,589 over 2 years (PF 0.912) — dead weight across both years
2. Fixed 20pt max stop = 0.4% at ES 5000 but only 0.29% at ES 6800 — doesn't scale

**v8 changes:**
1. Short-only mode (default "Short Only", input toggle to re-enable longs)
2. 30bps percentage-based stop (scales with price: 15pt at ES 5000, 20.4pt at ES 6800)
3. `dirFilter` and `ibPctStop`/`ibStopBps` inputs added to Pine Script
4. Dashboard rows: "Direction" (Short Only/Both/Long Only) and "Max Stop" (shows bps + dynamic pts)
5. `allowLongs`/`allowShorts` gates on all 3 setup signal conditions

**2-year results (Feb 2024 - Feb 2026, 140,966 bars):**

| Period | Trades | WR | PF | P&L | Max DD | Sharpe |
|--------|--------|------|------|------|--------|--------|
| Year 2 (Feb 24-25) | 87 | 41.4% | 1.129 | +$3,412 | $3,534 | 0.78 |
| Year 1 (Feb 25-26) | 90 | 48.9% | 1.820 | +$26,672 | $5,055 | 3.95 |
| **Combined** | **177** | **45.2%** | **1.511** | **+$30,084** | **$5,055** | **2.65** |

**Statistical significance (2 years):**
- t-test p-value: **0.028** (significant)
- Permutation p-value: **0.013** (significant)
- Bootstrap P(profit): **98.75%**
- 95% CI avg trade: **$18.67 to $321.27** (entirely positive)
- Monthly winners: **16/25 (64%)**

**What was tested and rejected for more trades:**
- Cooldown 2→0: no effect (crossunder is one-shot)
- MinIB range 8→3: +4 trades, negligible
- MaxIBTrades 2→4: +5 trades, dilutes PF
- TEMA trend OFF: +8 trades, nearly neutral
- VA Fade shorts: +24 trades, LOSERS (29% WR, PF 0.54)
- Drop noon blackout: +25 trades, kills significance
- Allow Fridays: +50 trades, LOSERS (36% WR, PF 0.81)

**Conclusion**: 177 trades / 2yr (~0.25/day) is the natural frequency. More trades require a different setup, not parameter tweaks.

### v9 — Exit Overhaul + TEMA Cross Entry (Feb 24, 2026)

**Problem**: v8 took only 1 late trade when 3 were available on a 100pt selloff. TP was set at 67pt (full IB range) — unreachable. No trailing stop meant a 45pt winner was held until flatten. No re-entries after the first trade. TEMA crossovers lined up as perfect short entries but weren't being used.

**File**: `amt-tema-strategy-v9.pine` (separate from v8 — v8 preserved as proven baseline)

**v9 changes:**
1. Scaled TP: `ibTP = math.max(ibMinTarget, math.min(ibRange, atrVal * tpAtrMult))` — caps at ATR x 3.0 (wide IB 67pt → 45pt, narrow IB unchanged)
2. Continuous re-arm: removed arm window mechanism, signal fires every bar below IB. Cooldown (2 bars) + maxIBTrades (3, was 2) prevent rapid-fire.
3. ATR trailing stop: `trail_points` + `trail_offset` on all exits. Trigger 15bps, distance 20bps. At ES 6000: 9pt trigger, 12pt trail.
4. TEMA exit: close shorts when TEMA fast crosses above slow (thesis invalidated). Uses `ta.crossover` not state.
5. TEMA Cross setup (Setup 4): short-only on `ta.crossunder(temaFast, temaSlow)`, day-type gated ("Narrow Only" default). Requires `trendDown`. 30bps stop, 2x ATR TP.
6. PMT JSON: `pmtMsg()` now takes `trailTrig` + `trailDist` params. Sets `trail:1, trail_trigger:X, trail_stop:Y, trail_freq:1`.
7. Dashboard: 21 rows (was 18). New rows: Trail (trigger/dist bps), TEMA Exit (ON/OFF), TX Cross (ON/OFF + day type).

**Status: NEEDS 2-YEAR BACKTESTING** — not yet validated. Plan: run Python backtest, A/B test by regime, optimize TEMA Cross day-type filter.

### Isolated Setup Performance
| Setup | P&L | Trades | WR | PF |
|-------|------|--------|------|------|
| VA Fade | +$8,993 | 24 | 58.33% | 3.507 |
| IB Breakout | +$2,245 | 21 | 47.62% | 1.466 |
| TEMA Crossover | -$3,313 | 65 | 52.31% | 0.788 |

### v5.1 Optimization Findings (22-test manual sweep)
- **TEMA Fast = 9 is optimal** (5 and 13 both worse). Slow (16-28) and Trend (39-69) are insensitive — strategy is robust, not curve-fitted.
- **Stop = IB Mid is optimal** (IB Edge and ATR both worse)
- **Winners applied to v5.1**: Cooldown 4→2 (+$1,380), VA Buffer 2→4 (+$965), VA R:R 1.0→0.5 (+$840), Trade End 1530→1500 (+$282), Min IB 5→8 (+$205)
- Combined v5.1 improvement: +$1,002 over v5 (parameters interact, expected)
- **TradingView 5m bar limit**: ~20,000 bars = ~3.5 months with ETH on ES1!

## Guidelines
- All AI tools share this context
- Coordinate work to avoid conflicts
- When modifying Pine Script strategies, always use `lookahead=barmerge.lookahead_off` in `request.security()` calls
- All strategy inputs should include `minval/maxval/step` for optimization
- Target instruments: ES (E-mini S&P 500), NQ (E-mini Nasdaq) futures
- Each AI has strengths:
  - **Claude Code**: Deep analysis, long-form writing, specialized agents
  - **Gemini CLI**: Fast web research, current info, quick iterations
  - **Grok CLI**: Real-time X/Twitter data, different perspective

## Current Tasks — AMT-TEMA
- [x] Design and code AMT+TEMA v1 — **DONE** (+$3,850, PF 1.279)
- [x] Implement TEMA engine, IB tracker, VA levels, IB Breakout + VA Fade setups
- [x] Isolate and test each setup independently — VA Fade star, TEMA Crossover loser
- [x] Build v5 with proven setups only (IB+VA) — **+$11,313, PF 2.358**
- [x] Test ADX approaches (4 variants) — all degraded performance, abandoned
- [x] 22-test manual parameter sweep across 4 rounds (TEMA, IB, VA, Session/Risk)
- [x] Build v5.1 with optimized defaults — +$12,315, PF 2.427, 55% WR
- [x] Test multi-contract scaling — confirmed pure linear (2x = exactly double), capital decision not strategy
- [x] Test max trades/dir/day — IB=2 adds +$1,050 (good), VA=2 hurts (bad)
- [x] Build v5.2 — **+$13,365, 57.5% WR, same $2,198 DD**
- [x] Convert AMT-TEMA v6 to thinkorSwim — `amt-tema-strategy-tos.ts` (study, Feb 13 2026)
- [x] Build Python backtester — exact Pine Script replica, 31 tests passing, validated vs TradingView
- [x] Download 1-year Databento ES data (70,331 bars, Feb 2025 - Feb 2026)
- [x] Run 1-year backtest — discovered v5.2 was overfit (PF 2.5 → 1.031 over 12 months)
- [x] A/B test 5 improvements — No VA, Noon Blackout, No Friday, Wider Stops, Earlier Cutoff
- [x] **Build v7** — combined best changes: +$28,936, PF 1.443, 47.1% WR (1yr only)
- [x] TEST v7 on TradingView — verified on ES1! 5m, +$4,275, PF 1.288, 40 trades
- [x] Walk-forward validation Year 1 — PASS: PF ratio 0.77, out-of-sample PF 1.224
- [x] Statistical significance Year 1 — p=0.047 (barely significant)
- [x] Downloaded Year 2 Databento data (Feb 2024 - Feb 2025, 70,635 bars)
- [x] 2-year validation of v7 — **FAILED**: +$18,445, PF 1.142, p=0.305 (longs + fixed stop killed it)
- [x] Investigate short-only — shorts +$24K over 2yr, longs -$5.6K (dead weight)
- [x] Investigate pct-based stops — 30bps scales correctly across ES 5000→6800
- [x] **Build v8** — short-only + 30bps pct stop: **+$30,084, PF 1.511, p=0.028 — SIGNIFICANT over 2yr**
- [x] Update SPX fork to v8 — `amt-tema-strategy-spx.pine` (short-only, pct stop, 18-row dashboard)
- [x] TEST v8 on TradingView — verified on ES1! 5m: +$3,027, 22 trades, 45.45% WR, PF 1.315
- [x] **Build v9** — exit overhaul + TEMA Cross: scaled TP, continuous re-arm, trail stop, TEMA exit, TX Cross setup
- [ ] Run v9 2-year Python backtest — compare vs v8 baseline
- [ ] A/B test TEMA Cross setup by day type / regime
- [ ] TEST v9 on TradingView — verify multiple IB entries, trail stop, TEMA exit, TX entries
- [ ] Improve VA levels — use volume.profile_session() instead of VWAP proxy
- [x] Set up automated execution Path 1 — TradingView→PickMyTrade→Tradovate (LIVE ON DEMO, Feb 15)
- [ ] Monitor demo for 1-2 weeks — verify entries match TradingView, SL/TP brackets correct, flatten at 15:55 works
- [ ] Go live — switch to funded Tradovate account, start with 1 contract
- [ ] Set up automated execution Path 2 — Python bot + IBKR API (permanent solution)
- [ ] Calibrate SPX fork — backtest/optimize parameters for SPX (ES params may not transfer directly)

## Team Assignments
- **Claude (Opus)**: Architecture, strategy design, complex Pine Script, debugging
- **Gemini**: Research validation, second opinion on setups, alt implementations
- **Grok**: Heavy reasoning on edge cases, refactoring, risk management logic
- **Local models**: Boilerplate code, simple edits, formatting only
