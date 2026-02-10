# AI Playground

Multi-AI workflow project where Claude Code, Gemini CLI, and Grok CLI collaborate on ES/NQ futures scalping strategies in Pine Script v5.

## Strategies

### Original
| File | Description |
|------|-------------|
| `proven-scalper.pine` | HTF EMA bias + EMA crossover + RSI + VWAP (1m chart) |
| `ichimoku-rsi-v2.pine` | Ichimoku cloud + TK cross + Chikou + Kijun trailing stop (5m chart) |
| `ichimoku-rsi-v1.pine` | First Ichimoku version |
| `futures-scalper.pine` — `v8` | Earlier scalper iterations |
| `claude-strategy.pine` | Claude-generated experiment |
| `gemini-strategy.pine` | Gemini-generated experiment |
| `combined-strategy.pine` | Combined AI experiment |

### Optimization-Ready
| File | Description |
|------|-------------|
| `proven-scalper-opt.pine` | Proven Scalper with full sweep ranges on all inputs. Fixed `lookahead_off` for accurate backtesting. |
| `ichimoku-rsi-v2-opt.pine` | Ichimoku RSI v2 with full parameter sweep ranges. |
| `hybrid-scalper-ichimoku.pine` | **Combined hybrid** — best of both strategies with mode selector. |

### Hybrid Strategy Details

The hybrid merges both strategies for the ES 5m chart:

**From Proven Scalper:** HTF 15m EMA bias, VWAP filter, EMA crossover trigger, split sessions (morning/afternoon)

**From Ichimoku:** Cloud alignment + thickness filter, Chikou confirmation, Kijun trailing stop

**Features:**
- **Mode Selector** — Conservative / Moderate / Aggressive
- **7 Toggleable Entry Gates** — HTF EMA, Cloud alignment, Cloud thickness, Chikou, RSI zone, VWAP, Session
- **Smart Exits** — Kijun trailing stop with buffer, fixed TP ceiling, max stop distance clamp, session flatten
- **10-Row Info Panel** — Live display of all filter states
- All inputs optimizable with `minval/maxval/step` for TradingView Strategy Tester

## Usage

1. Copy any `.pine` file contents into TradingView's Pine Editor
2. Add to an ES or NQ chart (5m recommended for hybrid/ichimoku, 1m for proven scalper)
3. Open Strategy Tester to run backtests
4. Use the optimization tab to sweep parameter ranges

## Docker Setup (Optional)

An isolated Docker environment is available for running AI agents safely.

```bash
docker-compose build
docker-compose up -d
docker exec -it ai-playground /bin/bash
```

All work inside the container happens in `/workspace` (maps to `./workspace`). Stop with `docker-compose down`.

## AI Agents

This project uses three AI coding assistants collaboratively:

- **Claude Code** — Deep analysis, long-form writing, specialized agents
- **Gemini CLI** — Fast web research, current info, quick iterations
- **Grok CLI** — Real-time X/Twitter data, different perspective

Shared context is maintained via `CLAUDE.md`, `GEMINI.md`, and `agents.md`.
