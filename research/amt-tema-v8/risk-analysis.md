# AMT-TEMA v8 -- Risk Analysis

## Drawdown Profile

| Metric | Value |
|--------|-------|
| Max Drawdown | $5,055 |
| Max DD % of Initial Capital | 5.1% (on $100K Pine default) |
| Max DD Occurrence | Year 1 (Feb 25-26) |

### Drawdown Context

The $5,055 max drawdown is moderate and manageable:
- On the $100K Pine Script default capital, it represents 5.1%
- At 1 contract, this is approximately 4 consecutive max-loss trades
- Recovery at the strategy's average P&L per trade (~$170) would take ~30 trades (~4 months)

Year 2's max DD of $3,534 is smaller than Year 1's $5,055, despite Year 2 having weaker performance. This suggests the drawdown comes from clustered losses during volatile moves, not from a grinding equity decline.

## Statistical Significance -- Full Analysis

### Primary Tests

| Test | Value | Threshold | Result |
|------|-------|-----------|--------|
| t-test p-value | 0.028 | < 0.05 | SIGNIFICANT |
| Permutation p-value | 0.013 | < 0.05 | SIGNIFICANT |
| Bootstrap P(profit) | 98.75% | > 95% | SIGNIFICANT |
| 95% CI Avg Trade | $18.67 -- $321.27 | > $0 | SIGNIFICANT |

### Interpretation

The confidence interval for average trade P&L ($18.67 -- $321.27) is entirely positive, meaning we can be 95% confident that the true expected P&L per trade is between $18.67 and $321.27. Even the lower bound is positive, confirming a real (if potentially small) edge.

However, $18.67 per trade at the lower bound represents only $3,305 per year (177 trades / 2yr x $18.67). After accounting for the manual effort and risk, this lower bound is barely worth trading. The point estimate of ~$170/trade ($30,084 / 177) is more encouraging but still modest.

### Comparison to Other Strategies

| Strategy | p-value | Trades | Edge Certainty |
|----------|---------|--------|----------------|
| Home Run | 0.0000 | 129/5yr | Overwhelming |
| Bear Breakdown | 0.0000 | 96/5yr | Overwhelming |
| Bull Credit Spread | 0.0000 | 405/5yr | Overwhelming |
| EMA Cross (credit) | 0.0000 | 1833/5yr | Overwhelming |
| **AMT-TEMA v8** | **0.028** | **177/2yr** | **Significant but modest** |
| EMA Cross Dir | 0.0205 | 176/5yr | Significant but modest |

AMT-TEMA v8 sits in the lower tier of statistical confidence. This is partly because it has only 2 years of data (vs 5 years for SPX strategies) and partly because the edge is narrower.

## Walk-Forward Validation -- Detailed Analysis

### Results

| Period | Config | Trades | WR | PF | P&L | Max DD | Sharpe |
|--------|--------|--------|------|------|------|--------|--------|
| In-sample (Feb-Aug 25) | v7 | 102 | 49.0% | 1.597 | +$22,909 | $5,168 | 3.03 |
| Out-of-sample (Aug 25-Feb 26) | v7 | 72 | 44.4% | 1.224 | +$6,028 | $6,145 | 1.37 |

### PF Ratio Analysis

**PF ratio = out-of-sample PF / in-sample PF = 1.224 / 1.597 = 0.77**

| PF Ratio | Interpretation |
|----------|----------------|
| > 0.85 | Excellent -- minimal degradation |
| 0.70 -- 0.85 | Good -- normal out-of-sample decay |
| 0.50 -- 0.70 | Concerning -- significant overfitting |
| < 0.50 | Failed -- strategy is overfit |

At 0.77, the strategy falls in the "Good" category. The out-of-sample degradation (PF dropping from 1.597 to 1.224) is within expected bounds and consistent with the filters being structural market patterns rather than curve-fitted artifacts.

### What Changed Out-of-Sample?

- Win rate dropped from 49.0% to 44.4% (reasonable -- some in-sample edge concentration)
- Sharpe dropped from 3.03 to 1.37 (more variable per-trade returns)
- Max DD increased from $5,168 to $6,145 (slightly worse drawdown profile)
- Still profitable: +$6,028 on 72 trades

### Important Caveat

Walk-forward was done on v7 filters. v8 added short-only + pct stop on top of v7. These v8 changes have structural justification (longs lost money, fixed stops don't scale) but have NOT been walk-forward validated independently.

## 2-Year Validation -- The v7 Failure That Led to v8

v7 passed 1-year walk-forward (PF ratio 0.77) but FAILED 2-year validation:
- Combined 2yr (v7): +$18,445, PF 1.142, p = 0.305 -- NOT SIGNIFICANT
- Year 2 (Feb 24-25, v7): -$10,491 (LOSING)

Root causes identified:
1. Longs lost -$5,589 over 2 years (PF 0.912)
2. Fixed 20pt stop didn't scale with ES price (0.4% at 5000, 0.29% at 6800)

v8 fixed both. The 2-year result improved from +$18,445 (PF 1.142, p=0.305) to +$30,084 (PF 1.511, p=0.028). This is a significant improvement from a straightforward structural fix, not additional optimization.

## Portfolio Fit

### Correlation with SPX Options Strategies

AMT-TEMA v8 trades ES futures (short only). The SPX options strategies trade SPX puts and credit spreads. The correlation structure:

- **AMT-TEMA v8 + HomeRun**: Both are bearish. They will draw down together during bull rallies. However, they use different instruments (ES futures vs SPX 0DTE puts) and different signals (IB breakout vs 30m structure break).
- **AMT-TEMA v8 + Bull Credit Spread**: Natural hedge. Bull Credit Spread profits during chop and bull periods where AMT-TEMA may not trade.
- **AMT-TEMA v8 alone**: The strategy trades ES at ~0.25 trades/day, which is low enough to coexist with any other strategy without capital conflicts.

### Capital Allocation

At 1 contract of ES, margin is approximately $500-$1,000 (day trading margin). The strategy does not compete for capital with the SPX options strategies. It can run on a separate broker account (Tradovate for ES, thinkorswim for SPX options).

## Worst-Case Scenarios

### 1. Extended Bull Rally with Low Volatility

If ES grinds higher with VIX < 15 for months:
- IB ranges shrink, fewer valid breakouts
- Trade frequency drops below 0.25/day
- The strategy may go weeks without a signal
- No direct loss, but opportunity cost on margined capital

### 2. Flash Crash During Position

A sudden 100+ point ES drop while short would be a massive win, but:
- Slippage on the limit order could be significant
- If the flash crash occurs overnight (position was not flattened), there is no exposure
- This is a tail-risk WIN scenario, not a loss scenario

### 3. Overnight Position (Flatten Failure)

If the 15:55 flatten alert fails:
- Position held overnight through ETH
- ES can gap 20-50 points overnight on news events
- A gap against (ES rallies 30pt overnight) would lose $1,500 (30pt x $50/pt)
- Mitigation: Secondary flatten alert, manual monitoring, broker-side time-based order

### 4. Strategy Degradation Over Time

Year 2's PF 1.129 is the canary in the coal mine. If Year 3 continues the degradation trend:
- Year 3 could be PF < 1.0 (losing)
- The 2-year statistical significance may not hold over 3 years
- The strategy should be re-evaluated after 6 months of live trading

### 5. Commission/Slippage Increase

Current costs: $5 round trip + 1 tick slippage (~$12.50) = $17.50/trade.
If costs double to $35/trade:
- Annual cost increase: ~$1,550 (88 trades x $17.50)
- Year 2's +$3,412 would become +$1,862 -- still positive but barely
- This is a real risk on a low-margin strategy

## Risk-Adjusted Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sharpe 2.65 | Annual risk-adjusted | Strong (>2.0 = very good) |
| Max DD $5,055 | Peak-to-trough | Moderate and manageable |
| DD / Annual P&L | $5,055 / $15,042 = 0.34x | Excellent (<1.0x) |
| P&L / Max Loss Trade | $30,084 / ~$1,000 = 30:1 | High reward per max risk |
| Monthly Win Rate 64% | 16 of 25 months | Adequate consistency |

## Recommendations

1. **Continue demo trading for 1-2 more weeks.** Verify automation chain (TradingView -> PMT -> Tradovate) works reliably, especially the 15:55 flatten.
2. **Go live with 1 contract.** The $5,055 max DD is manageable on any reasonably funded account.
3. **Monitor Year 3 performance monthly.** If PF drops below 1.0 for 3 consecutive months, reassess.
4. **Set a secondary flatten alert.** The single biggest risk is overnight exposure from a missed flatten.
5. **Do not add contracts until 3+ months of live data confirm backtest-like performance.**
6. **Track spread between Python backtest and live results.** Any systematic divergence indicates model assumptions are wrong.
