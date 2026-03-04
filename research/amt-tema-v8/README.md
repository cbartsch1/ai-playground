# AMT-TEMA v8 -- Peer Review

## Strategy Identity

| Field | Value |
|-------|-------|
| Name | AMT-TEMA v8 |
| File | `amt-tema-strategy.pine` (Pine Script v6) |
| Python Replica | `backtester/` (31 unit tests, exact Pine match) |
| Instrument | ES (E-mini S&P 500), 1 contract |
| Direction | SHORT ONLY |
| Timeframe | 5-minute chart |
| Data | Databento CME GLBX.MDP3, 140,966 bars, Feb 2024 -- Feb 2026 |
| Commission | $2.50/contract ($5 round trip) |
| Automation | TradingView -> PickMyTrade -> Tradovate (LIVE ON DEMO since Feb 15, 2026) |

## Key Performance Stats (2-Year Validated)

| Metric | Value |
|--------|-------|
| Net P&L | +$30,084 |
| Trades | 177 / 2yr (~88.5/yr) |
| Win Rate | 45.2% |
| Profit Factor | 1.511 |
| Sharpe Ratio | 2.65 |
| Max Drawdown | $5,055 |
| t-test p-value | 0.028 |
| Permutation p-value | 0.013 |
| Bootstrap P(profit) | 98.75% |
| 95% CI Avg Trade | $18.67 -- $321.27 (entirely positive) |
| Monthly Winners | 16/25 (64%) |

## Annual Breakdown

| Period | Trades | WR | PF | P&L | Max DD | Sharpe |
|--------|--------|------|------|------|--------|--------|
| Year 2 (Feb 24-25) | 87 | 41.4% | 1.129 | +$3,412 | $3,534 | 0.78 |
| Year 1 (Feb 25-26) | 90 | 48.9% | 1.820 | +$26,672 | $5,055 | 3.95 |
| **Combined** | **177** | **45.2%** | **1.511** | **+$30,084** | **$5,055** | **2.65** |

## Verdict

PASS -- Statistically significant at p < 0.05 by both t-test and permutation test across 2 years of out-of-sample Databento data. Walk-forward validation passed with PF ratio (out/in) = 0.77. The flagship strategy and FIRST to achieve full automation (TradingView -> PickMyTrade -> Tradovate chain live on demo). Year 2 performance (PF 1.129, +$3,412) is notably weaker than Year 1 (PF 1.820, +$26,672), raising questions about whether the edge is narrowing or whether Year 2's lower volatility (ES ~5000 vs ~6800) simply produced fewer trend days.

The strategy's natural frequency of 177 trades / 2 years (~0.25 trades/day) is confirmed: all attempts to add more trades degraded performance. 177 is the number, not a constraint.

## Document Index

| # | File | Contents |
|---|------|----------|
| 1 | `README.md` | This file -- summary, stats, verdict |
| 2 | `strategy-thesis.md` | Pattern definition, AMT theory, signal logic, all setups |
| 3 | `backtest-results.md` | Full stats, annual breakdown, statistical significance |
| 4 | `exit-analysis.md` | Exit types, IB extension target, pct stop, flatten logic |
| 5 | `risk-analysis.md` | Drawdowns, walk-forward validation, portfolio fit, worst-case |
| 6 | `limitations.md` | Known weaknesses, failed approaches, what could kill it |
| 7 | `pine-script-notes.md` | Full Pine Script details, automation setup, PickMyTrade integration |
