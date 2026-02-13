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

**Optimization-ready versions (current):**
- `proven-scalper-opt.pine` — Proven Scalper with minval/maxval/step on all inputs for TradingView Strategy Tester sweeps. **Fixed: lookahead_off** for accurate backtesting.
- `ichimoku-rsi-v2-opt.pine` — Ichimoku RSI v2 with full parameter sweep ranges on all inputs.
- `hybrid-scalper-ichimoku.pine` — **HYBRID v7** (current flagship). Combined hybrid strategy:
  - **Entry**: Direct structure break (v4 logic — best win rate at 41.1%)
  - **Trade reduction**: Higher lookback (8) + cooldown (6) + EMA alignment gate
  - From Proven Scalper: HTF 15m EMA bias, VWAP filter, split sessions
  - From Ichimoku: Cloud alignment + thickness filter, Chikou confirmation, Kijun trailing stop
  - **Mode Selector**: Conservative / Standard / Aggressive (controls gate strictness, RSI bands, SL/TP)
  - **8 toggleable entry gates**: HTF EMA, EMA Alignment, Cloud alignment, Cloud thickness, Chikou, RSI zone, VWAP, Volume
  - **Smart Exits**: Smart Kijun trailing (only improves stop, never worsens), fixed TP, session flatten
  - **14-row info panel** showing all filter states + cooldown + lookback
  - All inputs optimizable, lookahead_off, Pine Script v6, designed for ES 5m chart

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
1. IB Breakout + TEMA Confirmation (trend-day setup)
2. Value Area Fade + TEMA Slope (mean-reversion setup)
3. 80% Rule + TEMA Direction (high-probability setup)
4. TEMA Trend + Auction Context (adaptive setup)

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

## Current Tasks — Auction Theory + TEMA Strategy Build
- [ ] **ALL TEAM**: Read `research/auction-theory-tema-research.md` before doing ANY work
- [ ] Design and code the new AMT+TEMA strategy in Pine Script v6
- [ ] Implement TEMA engine (TEMA 9/21/55 calculations + crossover + slope)
- [ ] Implement Initial Balance tracker (IB High/Low/Range, first 60 min RTH)
- [ ] Implement Previous Day VA levels (POC/VAH/VAL via volume.profile_session)
- [ ] Build Setup 1: IB Breakout + TEMA Confirmation
- [ ] Build Setup 2: Value Area Fade + TEMA Slope
- [ ] Build Setup 3: 80% Rule + TEMA Direction
- [ ] Backtest on ES 5m, compare to v4-v7 results
- [ ] If profitable, run TradingView optimizer on key parameters

## Team Assignments
- **Claude (Opus)**: Architecture, strategy design, complex Pine Script, debugging
- **Gemini**: Research validation, second opinion on setups, alt implementations
- **Grok**: Heavy reasoning on edge cases, refactoring, risk management logic
- **Local models**: Boilerplate code, simple edits, formatting only
