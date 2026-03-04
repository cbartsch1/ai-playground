
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                          MEDALLION 2.0
                  Market Regime Detection System

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


◆ WHAT IT IS

Medallion 2.0 is a local market regime detection system built on a Hidden Markov Model (HMM). It runs as a Streamlit web application on localhost, providing a personal command center for identifying the current state of the market and filtering trading decisions through that lens.

The system observes three measurable market characteristics — returns, price range, and volume volatility — and infers which of seven hidden market regimes is most likely active at any given time. These regimes range from Crash to Strong Bull, each with distinct statistical properties. The model doesn't predict where the market is going. It identifies where the market IS, right now, and how confident it is in that assessment.

The core output is simple: a regime label, a directional signal (bullish/bearish/neutral), and a confidence percentage. Everything else in the system exists to validate that those outputs are trustworthy and to make them actionable.


<div style="page-break-before: always;"></div>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ WHAT IT DOES

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


▸ 𝗥𝗲𝗴𝗶𝗺𝗲 𝗗𝗲𝘁𝗲𝗰𝘁𝗶𝗼𝗻

The 7-state Gaussian HMM trains on hourly SPY data and classifies every bar into one of seven regimes:

    ■ Crash (Panic) — High volatility, sharp negative returns. Rare but unmistakable.
    ■ Bear Trend — Sustained negative drift with elevated volatility.
    ■ Distribution — Weakening momentum, often a transition from bull to bear.
    ■ Accumulation (Chop) — Sideways, low-conviction price action. Neither side in control.
    ■ Recovery — Positive returns emerging from a bearish period.
    ■ Bull Run — Sustained positive drift with moderate volatility.
    ■ Strong Bull — High-conviction uptrend with broad participation.

The model assigns a probability to each regime at every bar. The highest probability determines the label. The probability itself becomes the confidence score.


▸ 𝟴-𝗖𝗼𝗻𝗳𝗶𝗿𝗺𝗮𝘁𝗶𝗼𝗻 𝗩𝗼𝘁𝗶𝗻𝗴 𝗦𝘆𝘀𝘁𝗲𝗺

Regime detection alone can produce false signals. The confirmation system cross-references the HMM output against eight independent technical indicators: RSI, momentum, volatility, volume, ADX, EMA 50, EMA 200, and MACD. A minimum of 7 out of 8 must agree before the system considers a regime signal actionable. This acts as a second layer of validation — the HMM says "this is a bull market" and the confirmations say "yes, the technicals agree."


▸ 𝗥𝗲𝗴𝗶𝗺𝗲 𝗤𝘂𝗮𝗹𝗶𝘁𝘆 𝗔𝗻𝗮𝗹𝘆𝘀𝗶𝘀

The system doesn't ask you to trust it blindly. The Regime Quality Analyzer measures four dimensions of usefulness:

    ■ Forward returns by regime — After the model identifies a regime, what actually happens to price over the next hour, 4 hours, day, and week? If Bull regimes don't produce positive forward returns, the model is useless regardless of how confident it appears.

    ■ Regime stability — How long does each regime last on average? How often does the model flip to a regime for only one or two bars before flipping back? Short-lived regimes are noise, not signal. The false alarm rate quantifies this directly.

    ■ Filter value — If you only held positions during bullish regimes, how does your risk-adjusted return compare to buy-and-hold? This is the central question — does knowing the regime actually help?

    ■ Regime separation — Are the returns during bullish regimes statistically different from returns during bearish regimes? A t-test provides a p-value. If Bull and Bear regimes produce indistinguishable returns, the labels are meaningless.

These four metrics combine into a single quality score from 0 to 100.


▸ 𝗪𝗮𝗹𝗸-𝗙𝗼𝗿𝘄𝗮𝗿𝗱 𝗩𝗮𝗹𝗶𝗱𝗮𝘁𝗶𝗼𝗻

The model trains on historical data. The obvious risk is overfitting — finding patterns in the past that don't persist into the future. Walk-forward validation addresses this directly.

The system trains on the first 6 months of data, then predicts the 7th month entirely out-of-sample. Then it trains on months 1-7 and predicts month 8. This expanding window continues through the entire dataset, producing 30 out-of-sample prediction periods.

The key questions it answers:

    ■ Do bullish regimes produce higher returns than bearish regimes in data the model has never seen?
    ■ Does the transition matrix (how regimes flow into each other) remain stable as more data is added?
    ■ Does model confidence remain high on unseen data, or does it deteriorate?


▸ 𝗠𝗼𝗱𝗲𝗹 𝗣𝗲𝗿𝘀𝗶𝘀𝘁𝗲𝗻𝗰𝗲

The HMM takes several seconds to fit. Without persistence, every dashboard load or regime check would require re-fitting from scratch. The system saves fitted models with a timestamp and configuration hash. On subsequent loads, the saved model is available instantly. Refitting is available on demand when you want to incorporate new data.


▸ 𝗥𝗲𝗴𝗶𝗺𝗲 𝗜𝗻𝘁𝗲𝗴𝗿𝗮𝘁𝗶𝗼𝗻 𝗔𝗣𝗜

The HMM runs on hourly bars. Trading strategies run on different timeframes — 5-minute bars for ES futures, 1-minute bars for SPY options. The RegimeFilter class bridges this gap. It forward-fills the hourly regime classification to any target timeframe, so a 5-minute strategy can query "what regime are we in?" at every bar without any timeframe mismatch.

The API provides:

    ■ Timeframe alignment — Forward-fill hourly regime to 5m, 1m, or any frequency.
    ■ Point-in-time queries — "What is the regime at this exact timestamp?"
    ■ Trade gating — "Should I take this short trade right now?" (checks regime + confidence)
    ■ Position sizing — A confidence-based multiplier from 0.0 to 1.0. 90%+ confidence gets full size. 70-90% gets three-quarter size. 50-70% gets half. Below 50%, the trade is skipped entirely.


▸ 𝗥𝗲𝗴𝗶𝗺𝗲 𝗖𝗵𝗮𝗻𝗴𝗲 𝗔𝗹𝗲𝗿𝘁𝘀

When the regime flips — particularly from bullish to bearish or vice versa — that's actionable information. The monitoring script compares the current regime to the last known regime and classifies the change by severity:

    ■ Critical — Bull to Bear or Bear to Bull (signal reversal)
    ■ Warning — Any cross between bullish/bearish/neutral boundaries
    ■ Info — Movement within the same signal group (e.g., Bull to Strong Bull)

Alerts arrive as desktop notifications and are logged to a persistent file. The monitor can run as a one-shot check or as a daemon on a schedule.


<div style="page-break-before: always;"></div>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ THE DASHBOARD

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Six tabs, one purpose: give you everything you need to trust or distrust the model's current assessment.

    ① Price & Regimes — Candlestick chart with regime-colored overlay and stacked regime probability chart. Shows what the model sees across the full data history.

    ② Confirmations — Current state of all 8 technical indicators with pass/fail status and a historical chart of how many confirmations were met over time.

    ③ Regime Quality — The quality score, filter value comparison, forward return tables, regime separation statistics, and stability metrics. This tab answers: is this model worth using?

    ④ Walk-Forward — Out-of-sample validation results with per-fold metrics, confidence charts, and the OOS regime separation test. This tab answers: will this model work on data it hasn't seen?

    ⑤ Transition Matrix — Heatmap of regime-to-regime transition probabilities and expected regime duration. Shows the structural dynamics — which regimes tend to follow which.

    ⑥ Model Selection — BIC/AIC comparison across 2-8 state models. Helps determine whether 7 states is optimal or whether a simpler model captures the same information.


<div style="page-break-before: always;"></div>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ HOW YOU USE IT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


▸ 𝗗𝗮𝘆-𝘁𝗼-𝗗𝗮𝘆 𝗪𝗼𝗿𝗸𝗳𝗹𝗼𝘄

The typical session starts by opening the dashboard and looking at the top row. Four cards tell you everything you need in under two seconds: the current signal (BULLISH, BEARISH, or NEUTRAL), the detected regime (e.g., Recovery), the confidence level (e.g., 90.1%), and how long the current regime has persisted.

That's the read. From there, the decision tree is straightforward.

If you run automated strategies — the regime filter is already integrated into your execution pipeline. Your strategy generates a signal. Before that signal becomes an order, the RegimeFilter checks the current regime and confidence. If the regime supports the trade direction and confidence exceeds your threshold, the order goes through. If not, the signal is discarded. You don't touch anything. The filter is always on.

If you trade discretionary or semi-automated — the dashboard gives you context before the market opens. You check the regime, note the confidence, look at the confirmation breakdown, and decide how aggressive to be today. A 95% confidence Bull Run with 7/8 confirmations passing means you lean into long setups. A 60% confidence Accumulation with 4/8 confirmations means you trade small or sit out.

If you manage a portfolio — the regime informs allocation. A Distribution regime is the early warning that a bull trend is losing steam. You don't need to sell everything, but you might reduce equity exposure, add hedges, or raise cash. The transition matrix shows you what typically follows Distribution — if the historical path is Distribution to Bear Trend 40% of the time, that's a quantified reason to get defensive.


▸ 𝗜𝗻𝘁𝗲𝗴𝗿𝗮𝘁𝗶𝗻𝗴 𝗪𝗶𝘁𝗵 𝗮 𝗧𝗿𝗮𝗱𝗶𝗻𝗴 𝗦𝘁𝗿𝗮𝘁𝗲𝗴𝘆

The integration is designed to be minimal.

Your strategy produces a DataFrame of signals — timestamps and trade directions. You load the regime filter, pass it your data, and add three columns: regime_label, regime_signal, and regime_confidence. Then you filter. Keep only the rows where the regime supports your direction. Adjust size by the confidence multiplier. Run the rest of your strategy logic as normal.

For a short-only ES futures strategy, this means: if the regime is Bear Trend or Crash with confidence above 50%, take the short. If the regime is Bull Run at 90% confidence, skip it. If the regime is Accumulation at 70%, take the short but at half size.

The filter doesn't change your strategy. It doesn't touch your entries, exits, stops, or targets. It wraps around the outside and decides whether the market environment makes this trade worth taking at all. Your strategy handles the tactical question — when and where to enter. The regime filter handles the strategic question — should you even be trading this direction right now.


▸ 𝗦𝗲𝘁𝘁𝗶𝗻𝗴 𝗨𝗽 𝗔𝗹𝗲𝗿𝘁𝘀

For traders who aren't staring at a dashboard all day, the regime monitor runs in the background. Set it to check every 5 minutes during market hours. When the regime flips from Recovery to Distribution, your machine sends a notification. You glance at it, open the dashboard if you want details, and decide whether to adjust your exposure.

The log file creates a historical record of every regime change. Over time, this becomes its own dataset. You can see how often the model correctly identified the shift before price confirmed it, how many false alarms occurred, and whether certain transitions (e.g., Bull to Distribution to Bear) follow a reliable sequence.


<div style="page-break-before: always;"></div>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ THE BENEFITS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


▸ 𝗬𝗼𝘂 𝗦𝘁𝗼𝗽 𝗙𝗶𝗴𝗵𝘁𝗶𝗻𝗴 𝘁𝗵𝗲 𝗠𝗮𝗿𝗸𝗲𝘁

Most trading losses don't come from bad entries or bad exits. They come from trading the wrong strategy in the wrong environment. A mean-reversion strategy in a trending market gives back everything it made. A momentum strategy in a choppy market gets whipsawed to death. Traders know this intuitively. They say things like "the market was choppy today" or "that was a trend day" — after the fact. Regime detection tells you before the fact, or at least at the same time, with a quantified confidence level.

When you know the regime, you stop running your trend strategy into a chop environment. You stop fading breakouts during a genuine trend day. You stop buying dips in a crash. These aren't edge cases. They're the primary source of preventable losses for active traders.


▸ 𝗬𝗼𝘂𝗿 𝗪𝗶𝗻 𝗥𝗮𝘁𝗲 𝗚𝗼𝗲𝘀 𝗨𝗽 𝗪𝗶𝘁𝗵𝗼𝘂𝘁 𝗖𝗵𝗮𝗻𝗴𝗶𝗻𝗴 𝗬𝗼𝘂𝗿 𝗦𝘁𝗿𝗮𝘁𝗲𝗴𝘆

Consider a strategy that takes 300 trades per year with a 40% win rate and a profit factor of 1.03. It's barely profitable. Commission drag might kill it. Now apply a regime filter that eliminates 120 of those trades — specifically the ones taken in hostile regimes. The remaining 180 trades might have a 47% win rate and a profit factor of 1.45. Nothing about the strategy logic changed. You didn't optimize a single parameter. You just stopped taking trades that the market environment was going to punish.

This is exactly what happened during the development of an ES futures strategy in this system. The unfiltered strategy took 305 trades at a 1.03 profit factor — essentially random after costs. Removing trades taken in hostile conditions dropped the trade count to 174 but pushed the profit factor to 1.44 and achieved statistical significance. The edge was always there. It was buried under noise trades.

Regime filtering applies this same principle at a higher level. Instead of filtering by time-of-day or day-of-week, you filter by the state of the market itself. The regime detector identifies the noise trades for you — the ones where your strategy's assumptions about market behavior don't hold.


▸ 𝗣𝗼𝘀𝗶𝘁𝗶𝗼𝗻 𝗦𝗶𝘇𝗶𝗻𝗴 𝗕𝗲𝗰𝗼𝗺𝗲𝘀 𝗔𝗱𝗮𝗽𝘁𝗶𝘃𝗲

Fixed position sizing treats every trade equally. A 1-contract trade in a 90% confidence Bull Run is the same as a 1-contract trade in a 55% confidence Accumulation. But these are not the same trade. The first has the full weight of market structure behind it. The second is a coin flip in fog.

Confidence-based position sizing makes this distinction mechanical. Full size when the model is highly confident. Three-quarter size when it's moderately confident. Half size when it's uncertain. Zero when it doesn't know. This isn't a new idea — professional traders have always sized by conviction. The difference is that conviction is now quantified by a statistical model rather than a gut feeling.

The practical effect is that your capital is concentrated in the trades most likely to work and pulled back from the ones most likely to fail. Over hundreds of trades, this concentration effect compounds. You don't need more winners. You need your winners to be bigger and your losers to be smaller. Regime-based sizing does both.


▸ 𝗗𝗿𝗮𝘄𝗱𝗼𝘄𝗻𝘀 𝗚𝗲𝘁 𝗦𝗵𝗮𝗹𝗹𝗼𝘄𝗲𝗿

The regime filter's biggest value isn't in the trades it takes. It's in the trades it prevents.

Consider a short-only strategy during a Strong Bull regime. Without the filter, the strategy takes shorts into a rising market. Some win, most lose, and the equity curve draws down. The trader watches their account shrink and eventually overrides the system or increases size to "make it back" — the classic drawdown spiral.

With the filter, those shorts never happen. The strategy sits flat during Strong Bull. No losses, no drawdown, no emotional spiral. When the regime shifts to Distribution or Bear, the strategy re-engages and catches the move it was designed for.

In validation, the regime filter reduced max drawdown from -19.4% (buy-and-hold) to -5.0% (filtered) while still capturing 48% of the total return. Put differently: you gave up 24 percentage points of return but you avoided 14 percentage points of drawdown. For any real trader using real money, that tradeoff is not close. Shallow drawdowns mean you stay in the game. Deep drawdowns mean you blow up or quit.


▸ 𝗬𝗼𝘂 𝗞𝗻𝗼𝘄 𝗪𝗵𝗲𝗻 𝘁𝗼 𝗦𝗶𝘁 𝗢𝘂𝘁

One of the hardest things in trading is doing nothing. Markets reward patience, but patience feels like missing out. Traders overtrade because inactivity feels like failure.

The regime detector gives you permission to sit out, backed by data. When the dashboard shows Accumulation at 65% confidence with 4/8 confirmations, the system is telling you: there's no edge right now. The market isn't trending. Momentum is flat. Volume is average. The regime could go either way. This is not a trade environment. Sit on your hands.

This is arguably more valuable than any entry signal. The trades you don't take when conditions are poor protect the capital you need for the trades you do take when conditions are good. The regime detector quantifies "poor conditions" so you don't have to decide subjectively.


▸ 𝗦𝘁𝗿𝗮𝘁𝗲𝗴𝘆 𝗗𝗲𝘃𝗲𝗹𝗼𝗽𝗺𝗲𝗻𝘁 𝗚𝗲𝘁𝘀 𝗙𝗮𝘀𝘁𝗲𝗿

Building a new strategy involves two questions: does this idea have edge, and where does that edge come from? Without regime decomposition, you backtest across the entire dataset and get aggregate numbers. The strategy made $30,000 over two years. Great. But did it make $50,000 in bull markets and lose $20,000 in bear markets? Or did it make $15,000 in both?

These are different strategies with different risk profiles. The first one will destroy you in the next bear market. The second one is genuinely robust. Without regime classification, you can't tell the difference until it's too late.

With Medallion 2.0, you decompose backtest results by regime from the start. You see that your strategy profits in five regimes and loses in two. Now you know exactly where the weakness is. You can either fix the strategy for those two regimes, or filter them out, or accept the risk with full knowledge of when it will arrive.

This decomposition also prevents overfitting. A strategy that only works in one regime is probably curve-fit to a specific market condition. A strategy that works across multiple regimes, even with varying profitability, is more likely to survive the next regime shift. The quality analyzer's forward returns table makes this visible immediately — you don't have to discover it the hard way with real money.


<div style="page-break-before: always;"></div>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ WHY IT MAKES THE DIFFERENCE BETWEEN PROFITABLE AND UNPROFITABLE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Most retail traders lose money. The statistics are well-documented: 70-90% of active traders are unprofitable over any multi-year period. The explanations usually focus on psychology — fear, greed, overtrading, revenge trading. These are real, but they're symptoms. The underlying cause is structural: retail traders apply fixed strategies to a market that changes state.

A strategy that works in one regime and fails in another will produce a breakeven or negative result over a full market cycle. The trader sees intermittent profits followed by drawdowns, concludes the strategy "stopped working," abandons it, and starts over with a new one. This cycle repeats. The strategy might have been fine. It just needed to know when to step aside.

Institutional traders and quantitative funds have used regime models for decades. They don't apply the same strategy in every market condition. They adapt — different strategies for trending markets, different strategies for mean-reverting markets, different position sizes for high-volatility vs. low-volatility environments. This adaptation is the primary structural advantage that separates consistently profitable operations from the rest.

Medallion 2.0 makes that adaptation available to anyone with a laptop. It doesn't require a team of PhDs, a Bloomberg terminal, or a proprietary data feed. It runs on free data, open-source libraries, and a single Python environment. The math is the same math that institutional funds use. The validation is the same validation. The only difference is scale.

The gap between a strategy that makes $3,700 per year (profit factor 1.03, essentially noise) and a strategy that makes $30,000 per year (profit factor 1.51, statistically significant) might be nothing more than knowing which 120 out of 300 trades to skip. Regime detection identifies those trades. That's the difference between an expensive hobby and a working business.

The system doesn't promise profits. No honest system can. What it does is remove the single largest source of preventable losses — trading into an environment that is working against you — and replace intuition with measurement. Whether you trade futures, options, equities, or crypto, the question is the same: what is the market doing right now, and does my strategy work in this condition?

Medallion 2.0 answers that question with a statistical model, a confidence score, and the validation to prove the answer is trustworthy.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
