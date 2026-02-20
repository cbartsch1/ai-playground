# Medallion 2.0 / 2026 — HMM Market Regime Terminal

## Overview
7-state Gaussian HMM market regime detection with 13-confirmation voting system (8 technical + 5 macro),
FRED economic data, VIX term structure, credit spreads, market breadth, cross-asset regime detection,
and transition forecasting. Streamlit dashboard with 8 tabs.
Inspired by Jim Simons' Renaissance Technologies Medallion Fund approach.
**Machine**: M3 Ultra, 96 GB RAM
**Python**: 3.14, venv at `.venv/`

## Project Structure
```
medallion-2.0/
├── CLAUDE.md              # This file
├── requirements.txt       # Python dependencies
├── .env                   # FRED API key (gitignored)
├── .env.example           # Template for .env
├── .gitignore             # Protects .env, caches, models
├── config/
│   ├── __init__.py
│   └── settings.py        # 7-state config, 13 confirmations, macro events, forecast horizons
├── data/
│   ├── __init__.py
│   ├── data_loader.py     # Download OHLCV, compute HMM features + 13 confirmations
│   ├── macro_data.py      # FRED download, cache, alignment, yield curve, credit stress
│   ├── market_internals.py # VIX term structure, DXY, breadth, P/C ratio, cross-asset
│   ├── raw/               # Downloaded market data (parquet)
│   ├── processed/         # Feature matrices + cached data (parquet)
│   └── cache/             # Cached API responses (FRED + yfinance)
├── models/
│   ├── __init__.py
│   ├── hmm_regime.py      # RegimeDetector class (7-state Gaussian HMM, save/load)
│   ├── regime_api.py      # RegimeFilter — alignment, sizing, alerts (integration API)
│   ├── trained/           # Saved model files (pickle, auto-named with timestamp+hash)
│   └── checkpoints/       # Model checkpoints
├── backtester/
│   ├── __init__.py
│   ├── regime_quality.py  # RegimeQualityAnalyzer — forward returns, stability, separation
│   └── legacy_backtester.py  # Original RegimeBacktester (kept for reference)
├── dashboard/
│   ├── __init__.py
│   ├── app.py             # Streamlit Regime Terminal (dark theme, 8 tabs, model persistence)
│   ├── components/        # Dashboard widget modules
│   └── assets/            # CSS, images
├── scripts/
│   ├── download_data.py   # Legacy download script (yfinance + FRED, daily data)
│   ├── walk_forward.py    # Walk-forward OOS validation (expanding window)
│   ├── regime_monitor.py  # Regime change alerts (macOS notifications + log)
│   └── optimizer.py       # 3-phase HMM optimizer (n_regimes BIC + confirmation sensitivity)
├── tests/
│   ├── __init__.py
│   ├── test_hmm_regime.py        # 16 tests: fit, predict, save/load, model selection
│   ├── test_data_loader.py       # 10 tests: feature computation, confirmations
│   ├── test_regime_api.py        # 13 tests: alignment, sizing, alerts
│   ├── test_regime_quality.py    # 11 tests: forward returns, stability, separation
│   ├── test_macro_data.py        # 11 tests: FRED alignment, yield curve, credit stress
│   ├── test_market_internals.py  # 11 tests: VIX ratio, breadth, cross-asset features
│   └── test_transition_forecast.py # 8 tests: matrix power, P(change), alert levels
├── logs/                  # Regime change alerts log
├── notebooks/             # Jupyter exploration notebooks
└── research/
    └── medallion-fund-research.md  # Medallion Fund + HMM research
```

## Architecture

### Core: 7-State Gaussian HMM
- **Library**: `hmmlearn` (GaussianHMM) — simple, well-tested, CPU-based
- **Default**: 7 states, supports 2-7 with BIC/AIC model selection
- **3 HMM features**: returns (log), range (high-low/close), volume_vol (rolling std of log volume)
- **Training**: EM algorithm (Baum-Welch), 10 random restarts, best log-likelihood wins
- **Inference**: Forward algorithm for real-time regime probabilities
- **Auto-labels**: Sorts regimes by mean return → assigns names from Crash to Strong Bull

### 7 Regime States
| State | Label | Color | Signal |
|-------|-------|-------|--------|
| Crash | Crash (Panic) | Purple #8e44ad | Bearish |
| Bear | Bear Trend | Red #e74c3c | Bearish |
| Distribution | Distribution | Dark Orange #e67e22 | Bearish |
| Chop | Accumulation (Chop) | Orange #f39c12 | Neutral |
| Recovery | Recovery | Blue #3498db | Bullish |
| Bull | Bull Run (Trend) | Green #2ecc71 | Bullish |
| Strong Bull | Strong Bull (Trend) | Dark Green #27ae60 | Bullish |

### 13-Confirmation Voting System (8 Technical + 5 Macro)

**Technical (8):**
| # | Indicator | Threshold | Pass If |
|---|-----------|-----------|---------|
| 1 | RSI (14) | 90 | Below (not overbought) |
| 2 | Momentum (20) | 1% | Above |
| 3 | Volatility (20) | 6% | Below |
| 4 | Volume | 20-SMA | Above |
| 5 | ADX (14) | 25 | Above (strong trend) |
| 6 | EMA 50 | Price | Above |
| 7 | EMA 200 | Price | Above |
| 8 | MACD | Signal line | Above |

**Macro (5):**
| # | Indicator | Source | Threshold | Pass If |
|---|-----------|--------|-----------|---------|
| 9 | VIX Term Structure | yfinance | VIX/VIX3M < 1.0 | Contango (complacency) |
| 10 | Credit Spread | FRED HY OAS | < 5.0% | Credit calm |
| 11 | Yield Curve | FRED 10Y-2Y | > 0 | Not inverted |
| 12 | Market Breadth | RSP/SPY ratio | 20-SMA slope > 0 | Broad participation |
| 13 | Dollar Strength | DXY | Weekly change < 1% | Dollar not surging |

**Entry rule**: Bullish regime AND ≥10/13 confirmations (or ≥7/8 technical-only mode)
**Exit rule**: Regime flips to bearish (Bear/Crash/Distribution) → close immediately
**Cooldown**: 48 hours after any exit before re-entry allowed
**Leverage**: 2.5x (configurable)

### First Backtest Results (Feb 17, 2026 — SPY 1h, ~2yr, 5068 bars)
- Initial Capital: $100,000
- Final Value: $115,008
- Total Return: +15.0%
- Buy & Hold: +72.2% (strong bull market period)
- Alpha vs B&H: -57.2% (regime model is conservative — fewer trades)
- Trades: 98
- Win Rate: 45.9%
- Profit Factor: 1.27
- Max Drawdown: -12.3%
- Sharpe Ratio: 0.53
- Avg Hold: 8 bars
- **Note**: These are FIRST results, not optimized. The model trades conservatively.

### Dashboard: Streamlit (Dark Theme)
- **8 tabs**: Price & Regimes, Confirmations, Regime Quality, Walk-Forward, Transition Matrix, Model Selection, Macro Context, Market Internals
- **Model persistence**: Load Saved / Refit Model buttons (no auto-fit on page load)
- **Include Macro checkbox**: Toggles macro data download and 5 macro confirmations
- **Data freshness indicator**: Shows cache age in top-right
- Current signal card (BULLISH/BEARISH/NEUTRAL) with confidence %
- **Transition Risk card**: STABLE/WATCH/WARNING/CRITICAL with P(change 6h)
- Plotly candlestick chart with regime-colored background shading
- **FOMC/CPI/NFP event markers** on price chart (blue/orange/green dotted lines)
- Stacked regime probability area chart
- **13-confirmation voting breakdown**: 8 technical + 5 macro cards with pass/fail
- Confirmation count over time chart
- **Regime Quality tab**: Quality score (0-100), forward returns by regime, filter value (vs B&H), regime separation (t-test), stability metrics
- **Walk-Forward tab**: OOS validation results, per-fold confidence/changes, bullish vs bearish OOS returns
- Regime statistics table (mean return, vol, Sharpe, expected duration)
- **Transition Matrix tab**: Heatmap + P(change) forecast line chart + alert level badge
- BIC/AIC model selection chart (2-8 states)
- **Macro Context tab**: Yield curve, fed funds rate, credit spread (HY OAS), jobless claims, CPI YoY, event calendar, seasonal heatmap, macro health summary
- **Market Internals tab**: VIX vs VIX3M + ratio, RSP/SPY breadth, DXY + SMA50, put/call ratio, cross-asset regime panel (TLT/GLD/USO), alignment indicator

### Regime Integration API (`models/regime_api.py`)
- `RegimeFilter` class — bridge between hourly HMM and trading strategies
- `align_to_timeframe()` — forward-fill hourly regime to any bar frequency (5m, 1m, etc.)
- `add_regime_columns(df)` — adds regime_label, regime_signal, regime_confidence to any DataFrame
- `get_regime_at(timestamp)` — point-in-time regime query
- `should_trade(timestamp, direction)` — checks regime + confidence for trade gating
- `position_size_multiplier(timestamp)` — 0.0 to 1.0 based on confidence tiers
- `check_regime_change(prev, current)` — detect regime flips with severity classification
- `transition_risk(timestamp)` — P(regime change) at multiple horizons + alert level
- `should_reduce_size(timestamp)` — True if transition risk is warning/critical
- `is_macro_event_day(date)` — check for FOMC/CPI/NFP events
- `cross_asset_alignment(regimes)` — analyze regime alignment across SPY/TLT/GLD/USO

### Transition Forecasting (`models/hmm_regime.py`)
- `forecast_transitions(current_probs, horizons)` — matrix exponentiation: P(state at t+n) = P(t) @ T^n
- Returns P(change) at each horizon, most_likely_next, P(bearish), alert_level
- Alert levels: stable (<20%), watch (20-40%), warning (40-60%), critical (>60%)

### Confidence-Based Position Sizing
- 90%+ confidence → 1.0x (full size)
- 70-90% → 0.75x
- 50-70% → 0.5x
- <50% → 0.0 (skip)
- Thresholds configurable in `config/settings.py` as `CONFIDENCE_TIERS`

### Regime Change Alerts (`scripts/regime_monitor.py`)
- Checks current vs last saved regime, classifies severity (critical/warning/info)
- macOS desktop notification via `osascript`
- Logs to `logs/regime_changes.log`
- Can run as daemon (`--daemon 300` = check every 5 min)
- Suitable for cron/launchd scheduling during RTH

### Data Pipeline
- **Primary**: `data/data_loader.py` — downloads SPY hourly via yfinance (730d max)
- **Computes**: 3 HMM features + 8 confirmation indicators + pass/fail flags
- **Caches**: Processed data as parquet in `data/processed/`
- **Legacy**: `scripts/download_data.py` — daily data for all tickers + FRED

## Running
```bash
cd ~/projects/ai-playground/medallion-2.0
source .venv/bin/activate

# Quick test — download data + fit model
python data/data_loader.py

# Launch dashboard
streamlit run dashboard/app.py

# Run tests (80 tests)
pytest tests/

# Walk-forward validation
python scripts/walk_forward.py

# HMM optimizer (all 3 phases)
python scripts/optimizer.py

# Optimizer — single phase or forced n
python scripts/optimizer.py --phase 1          # n_regimes sweep only
python scripts/optimizer.py --phase 2          # confirmation sensitivity only
python scripts/optimizer.py --n-regimes 5      # force specific n for phase 2-3

# Regime monitor (one-shot check)
python scripts/regime_monitor.py

# Regime monitor (daemon mode, check every 5 min)
python scripts/regime_monitor.py --daemon 300
```

## Key Decisions
- 7-state model confirmed optimal by BIC sweep + OOS validation (Feb 17, 2026 optimizer run)
- 8 states overfits (p=1.0 OOS), fewer states lose quality monotonically
- Hourly timeframe for regime detection (daily too slow, 5m too noisy)
- 3 core HMM features only (returns, range, volume_vol) — keep it simple, avoid overfitting
- Macro data = confirmations layer, NOT HMM features (avoids curse of dimensionality)
- Regime is a FILTER not a signal — use with existing strategies (AMT-TEMA, Home Run)
- Auto-label regimes by sorting on mean return + volatility
- Walk-forward validation: expanding window, re-fit monthly
- Transition forecasting via matrix exponentiation (P(t) @ T^n)

### HMM Optimizer (`scripts/optimizer.py`) — Feb 17, 2026
- **Phase 1**: n_regimes sweep (2-8) — BIC/AIC + quality score + abbreviated walk-forward (6 folds)
- **Phase 2**: Confirmation sensitivity — leave-one-out impact + min_confirmations sweep
- **Phase 3**: Combined validation — full 30-fold walk-forward at best n
- **CLI**: `python scripts/optimizer.py`, `--phase 1|2|3`, `--n-regimes N`
- **Dashboard**: "Apply Best (N states)" button on Model Selection tab

**Optimizer Results (Feb 17, 2026 — SPY 1h, 5068 bars):**

| n | BIC | Quality | OOS p-value | Verdict |
|---|-----|---------|-------------|---------|
| 2 | 167.0M | 19 | 1.0000 | Underfit |
| 3 | 151.6M | 38 | 0.0011 | |
| 4 | 141.6M | 58 | 0.0000 | |
| 5 | 139.3M | 63 | 0.0000 | |
| 6 | 130.7M | 69 | 0.0000 | |
| **7** | **124.2M** | **74** | **0.0003** | **BEST** |
| 8 | 119.3M | 18 | 1.0000 | Overfit |

**Confirmation Sensitivity:**
- RSI (97% pass rate) — dead weight, always passes
- Volatility (99.6% pass rate) — dead weight, always passes
- Momentum (27% pass rate) — strictest gatekeeper (+54% bars if removed)
- MACD (47%), Volume (40%) — next most impactful
- Optimal min_confirmations: 4/8 (Sharpe 1.962) vs current 7/8 (Sharpe 1.324)

**Walk-Forward (30 folds):** OOS regime separation p=0.0000 (t=16.5), avg confidence 89.7%

## Connection to Existing Strategies
- **AMT-TEMA v8 (ES shorts)**: Skip shorts during Strong Bull/Bull Run regimes
- **Home Run (SPY 0DTE puts)**: Only take puts during Bear/Crash regimes, skip during Bull
- Regime-aware position sizing: full size in high-conviction regimes
- Skip trades when regime probability < 50% (uncertain transitions)

## Current Tasks
- [x] Project structure and venv setup
- [x] Core HMM RegimeDetector class (7-state, auto-labeling)
- [x] Data loader with 3 HMM features + 8 confirmations
- [x] Backtester with confirmations, cooldown, leverage
- [x] Full Streamlit dashboard (dark theme, 6 tabs)
- [x] Config with 7-state labels, colors, confirmation thresholds
- [x] Download and cache SPY 1h data (~5068 bars, 2yr)
- [x] First HMM fit and backtest validation
- [x] Model persistence — save_latest/load_latest with timestamp + config hash
- [x] Walk-forward validation script (expanding window, monthly re-fit)
- [x] Regime Quality Analyzer (replaces naive backtester — forward returns, stability, separation)
- [x] Regime Integration API (RegimeFilter — alignment, sizing, alerts)
- [x] Confidence-based position sizing (4-tier system)
- [x] Dashboard overhaul — regime quality + walk-forward tabs, model load/refit, no fake P&L
- [x] Regime change alerts (macOS notifications + log file)
- [x] Test suite — 50 tests across 4 files, all passing
- [x] Launch dashboard and verify all tabs render correctly
- [x] Run walk-forward validation and analyze OOS results (30 folds, p=0.0000, regime separation confirmed)
- [x] FRED API key and economic features (7 series: yield curve, credit spread, fed funds, claims, CPI)
- [x] Market internals (VIX term structure, DXY, RSP/SPY breadth, put/call ratio)
- [x] Expand confirmations 8 → 13 (8 technical + 5 macro)
- [x] Transition forecasting (matrix exponentiation, alert levels)
- [x] Cross-asset regime detection (TLT, GLD, USO with 3-state HMM)
- [x] Dashboard expanded to 8 tabs (Macro Context + Market Internals)
- [x] FOMC/CPI/NFP event markers on price chart
- [x] Transition risk card + forecast chart on Transition Matrix tab
- [x] Test suite expanded — 80 tests across 7 files, all passing
- [x] Optimize: n_regimes BIC sweep + confirmation sensitivity (`scripts/optimizer.py`)
  - 7 states confirmed optimal (BIC=124M, quality=74, OOS p=0.0003)
  - 8 states overfits (p=1.0 OOS), 2 states underfits (quality=19)
  - RSI (97% pass) and Volatility (99.6% pass) confirmations are dead weight
  - Momentum (27% pass) is the strictest gatekeeper
  - Optimal min_confirmations=4 (Sharpe 1.962) vs current 7 (Sharpe 1.324)
  - "Apply Best" button added to dashboard Model Selection tab
- [x] **OPTIMIZER REVIEWED**: min_confirmations stays at 7/8 — regime model must be accurate, not tuned for strategy throughput
- [ ] **CONSIDER**: Remove RSI + Volatility confirmations (dead weight — 97%/99.6% pass rate, but don't change to benefit strategies)
- [x] Integrate with AMT-TEMA and Home Run strategies (via RegimeFilter + regime_comparison.py)
  - AMT-TEMA: blocks Bull Run/Strong Bull → 177→155 trades, PF 1.51→1.69, Sharpe 2.65→3.39
  - Home Run: bearish-only filter → PF 3.60→6.96, Sharpe 6.71→8.95 (but 78/129 trades predate model)
  - Dashboard: Strategy Integration tab (9th tab) with comparison metrics + bar charts
- [ ] Set up regime_monitor.py on cron/launchd for RTH

## References
- Hamilton (1989) — Markov switching for business cycles (foundational paper)
- Rabiner (1989) — HMM tutorial (speech recognition, what Mercer/Brown knew)
- Zuckerman (2019) — "The Man Who Solved the Market" (Medallion Fund book)
- Nystrup et al. (2017) — Adaptive HMMs for financial time series
- Lopez de Prado (2018) — "Advances in Financial Machine Learning"
