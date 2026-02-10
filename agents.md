# AI Playground Project

## Overview
This is a multi-AI workflow test project where Claude Code, Gemini CLI, and Grok CLI work together on the same codebase. The current focus is ES/NQ futures scalping strategies in Pine Script v5.

## Project Structure
- Root directory for experimentation and testing multi-AI workflows

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
- `hybrid-scalper-ichimoku.pine` — **Combined hybrid strategy** merging the best of both:
  - From Proven Scalper: HTF 15m EMA bias, VWAP filter, EMA crossover trigger, split sessions
  - From Ichimoku: Cloud alignment + thickness filter, Chikou confirmation, Kijun trailing stop
  - **Mode Selector**: Conservative / Moderate / Aggressive (controls gate strictness, RSI bands, SL/TP)
  - **7 toggleable entry gates**: HTF EMA, Cloud alignment, Cloud thickness, Chikou, RSI zone, VWAP, Session
  - **Smart Exits**: Kijun trailing stop with buffer, fixed TP ceiling, max stop distance clamp, session flatten
  - **10-row info panel** showing all filter states
  - All inputs optimizable, lookahead_off, designed for ES 5m chart

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

## Current Tasks
- Backtest parameter optimization on all three -opt/hybrid strategies
- Verify trades fire on ES 5m in TradingView Strategy Tester
- Compare hybrid performance vs individual strategies across all 3 modes
