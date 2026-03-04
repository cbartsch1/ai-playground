# Medallion Fund & HMM Regime Detection Research

## Jim Simons / Renaissance Technologies

### Performance
- **Founded**: 1988 by Jim Simons (NSA codebreaker, Chern-Simons theorem)
- **Gross annual returns**: ~66% per year (1988-2018, 30+ years)
- **Net returns**: ~39% after 5-and-44 fee structure (highest in industry)
- **Never had a losing year** after 1989. In 2008 crisis: +82% gross
- **Sharpe ratio**: 3.0-6.0 (most hedge funds target 1.0-2.0)
- **AUM**: Capped at ~$10B — excess profits returned to keep optimal size
- **Closed to outside investors since 1993** — employees only
- **Total profits**: $100B+ since inception
- **~300 employees**, half with PhDs in math/physics/CS/statistics (NOT finance)

### Known/Suspected Strategy Elements
- **Statistical arbitrage**: Short-term mispricings across related instruments
- **Mean reversion at short horizons**: Prices deviate from statistical norms, then revert
- **Momentum at medium horizons**: Some trend-following overlay
- **Signal processing (HMMs)**: Robert Mercer & Peter Brown from IBM speech recognition
- **Non-linear models**: Kernel methods, early neural networks
- **Massive feature engineering**: Thousands of signals with small edge, diversified
- **Market microstructure**: Order flow, bid-ask dynamics, liquidity patterns
- **High-frequency execution**: Not HFT, but very efficient market impact minimization

### Key Quotes
- Simons: "We search through historical data looking for anomalous patterns we wouldn't expect at random"
- Simons: "There are no gross inefficiencies. We look at anomalies that may be small in size and brief in time"
- Henry Laufer: Confirmed "pattern recognition" with individually small-edge signals combined in large numbers

### The HMM Connection
- Mercer & Brown adapted speech recognition HMMs to financial time series
- Speech: hidden states = phonemes, observations = acoustic features
- Finance: hidden states = market regimes, observations = return statistics
- Mathematical machinery is identical — the innovation was recognizing the transfer

## HMM Regime Detection in Finance

### How It Works
1. **Hidden states** = Market regimes (bull, bear, sideways, crisis)
2. **Observations** = Returns, volatility, volume, correlations
3. **Transition matrix** = P(regime_j at t+1 | regime_i at t)
4. **Emission distributions** = Statistical properties per regime (Gaussian)

### Three Core Algorithms
1. **Forward algorithm** (Evaluation): Real-time regime probabilities
2. **Viterbi algorithm** (Decoding): Most likely historical regime sequence
3. **Baum-Welch / EM** (Learning): Estimate model parameters from data

### Recommended Features
| Feature | Why |
|---------|-----|
| Log returns | Mean differs across regimes |
| Realized volatility (5d, 21d) | Volatility clusters |
| VIX level | Forward-looking fear gauge |
| Volume ratio (vs 20d SMA) | Conviction changes |
| Sector dispersion | Cross-sectional vol of sector returns |
| Bond-equity correlation | Flips sign across regimes |
| Credit spreads (HYG vs LQD) | Widens in risk-off |
| Yield curve slope (10Y-2Y) | Inversion = recession regime |
| Momentum (21d, 63d) | Trend vs. mean-reversion regimes |

### Number of States
- **2-state**: Bull/Bear — simplest, most robust
- **3-state**: Bull/Sideways/Bear — most common in practice
- **4-state**: Bull/Bear/Crisis/Sideways — more expressive
- **5+**: Rarely used — overfitting risk
- **Use BIC/AIC to select** — don't pick states because they "look better"

### Transition Matrix Properties
- Diagonal dominance: regimes are "sticky" (self-transition 0.85-0.98)
- Asymmetric: markets crash faster than they recover
- Expected duration = 1/(1-self_transition_prob)
- If self-transition=0.95, expected duration=20 periods

## Python Libraries

### hmmlearn (Recommended start)
- Simple, well-tested, CPU-based
- `GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)`
- `.fit()`, `.predict()`, `.predict_proba()`, `.score()`

### pomegranate 1.1 (Advanced)
- PyTorch backend, GPU support (5-10x speedup on complex models)
- More flexible emission distributions
- `DenseHMM` with `Normal` distributions

### statsmodels (Frequentist alternative)
- `MarkovRegression(data, k_regimes=3, switching_variance=True)`
- Built-in smoothed regime probabilities
- `MarkovAutoregression` for AR + regime switching

### arch (Complementary)
- GARCH models for volatility modeling
- Good complement to HMM regime detection

## Application to Trading

### Position Sizing by Regime
- 90%+ regime confidence → full size
- 55-70% → reduced size
- Below 50% → flat or minimum size

### Strategy Selection by Regime
- **Bull**: Trend-following, buy dips, long momentum
- **Bear**: Short bias, mean-reversion from oversold, defensive
- **Sideways**: Mean-reversion, sell volatility, pairs trading
- **Crisis**: Reduce exposure, long volatility, flight-to-quality

### Regime Change Detection
- When dominant regime probability drops below 70% → potential transition
- Require 3 consecutive days to confirm regime change (anti-whipsaw)
- Regime transitions are the highest-information events

## Key Academic Papers
1. Hamilton (1989) — "A New Approach to Economic Analysis of Nonstationary Time Series"
2. Rabiner (1989) — "A Tutorial on Hidden Markov Models" (canonical HMM reference)
3. Ang & Bekaert (2002) — "Regime Switches in Interest Rates"
4. Guidolin & Timmermann (2007) — "Asset Allocation Under Multivariate Regime Switching"
5. Nystrup et al. (2017) — "Long Memory of Financial Time Series and HMMs"
6. Lopez de Prado (2018) — "Advances in Financial Machine Learning"
7. Zuckerman (2019) — "The Man Who Solved the Market"

## State of the Art (2024-2026)
- **HMM-LSTM Hybrid**: Combine HMM with LSTM, entropy-weighted Bayesian model averaging
  - 50%+ volatility reduction, 15-17pp drawdown improvement
- **LSTM-Transformer**: mTrans-MLP model for regime-aware forecasting
- **Online/Streaming HMMs**: Incremental learning for real-time adaptation
- **5-minute bar HMMs**: Sufficient for intraday regime detection
