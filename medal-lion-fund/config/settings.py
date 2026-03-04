"""
Medal-Lion Fund — Global Configuration

7-state Gaussian HMM regime detection with 13-confirmation voting system.
Inspired by Renaissance Technologies / Jim Simons' Medallion Fund.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"

# === HMM Configuration ===
DEFAULT_N_REGIMES = 7
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER = 200
HMM_N_RESTARTS = 10  # random restarts to avoid local optima
HMM_RANDOM_STATE = 42

# 3 core HMM features (what the model trains on)
HMM_FEATURES = [
    "returns",        # log returns
    "range",          # (high - low) / close — intrabar volatility
    "volume_vol",     # rolling std of volume — volume volatility
]

# === 7-State Regime Labels & Colors ===
REGIME_LABELS = {
    0: "Accumulation (Chop)",
    1: "Bull Run (Trend)",
    2: "Bear Trend",
    3: "Crash (Panic)",
    4: "Recovery",
    5: "Distribution",
    6: "Strong Bull (Trend)",
}

REGIME_COLORS = {
    0: "#f39c12",   # Orange — sideways chop
    1: "#2ecc71",   # Green — bull
    2: "#e74c3c",   # Red — bear
    3: "#8e44ad",   # Purple — crash/panic
    4: "#3498db",   # Blue — recovery
    5: "#e67e22",   # Dark orange — distribution
    6: "#27ae60",   # Dark green — strong bull
}

# Label-based colors — use this for dynamic HMM labeling (integers change per fit)
REGIME_LABEL_COLORS = {
    "Crash (Panic)": "#8e44ad",
    "Bear Trend": "#e74c3c",
    "Distribution": "#e67e22",
    "Accumulation (Chop)": "#f39c12",
    "Recovery": "#3498db",
    "Bull Run (Trend)": "#2ecc71",
    "Strong Bull (Trend)": "#27ae60",
    "Bear / Risk-Off": "#e74c3c",
    "Bull / Risk-On": "#2ecc71",
}

# Regime classification for trading signals
BULLISH_REGIMES = {"Bull Run (Trend)", "Strong Bull (Trend)", "Recovery"}
BEARISH_REGIMES = {"Bear Trend", "Crash (Panic)", "Distribution"}
NEUTRAL_REGIMES = {"Accumulation (Chop)"}

# === 8-Confirmation Voting System (Technical) ===
CONFIRMATIONS = {
    "rsi": {"threshold": 90, "direction": "below", "period": 14},
    "momentum": {"threshold": 0.01, "direction": "above", "period": 20},
    "volatility": {"threshold": 0.06, "direction": "below", "period": 20},
    "volume": {"threshold": 1.0, "direction": "above", "sma_period": 20},
    "adx": {"threshold": 25, "direction": "above", "period": 14},
    "ema_50": {"direction": "above", "period": 50},
    "ema_200": {"direction": "above", "period": 200},
    "macd": {"fast": 12, "slow": 26, "signal": 9, "direction": "above"},
}
MIN_CONFIRMATIONS = 7  # need 7/8 to enter (conservative — regime accuracy over strategy throughput)

# === 5 Macro Confirmations (new — extend to 13 total) ===
MACRO_CONFIRMATIONS = {
    "vix_term_structure": {
        "description": "VIX Term Structure",
        "threshold": 1.0,
        "direction": "below",
        "pass_label": "Contango (complacency)",
    },
    "credit_spread": {
        "description": "Credit Spread (HY OAS)",
        "threshold": 5.0,
        "direction": "below",
        "pass_label": "Credit calm",
    },
    "yield_curve": {
        "description": "Yield Curve (10Y-2Y)",
        "threshold": 0.0,
        "direction": "above",
        "pass_label": "Not inverted",
    },
    "market_breadth": {
        "description": "Market Breadth (RSP/SPY)",
        "threshold": 0.0,
        "direction": "above",
        "pass_label": "Broad participation",
    },
    "dollar_strength": {
        "description": "Dollar Strength (DXY)",
        "threshold": 1.0,
        "direction": "below",
        "pass_label": "Dollar not surging",
    },
}
TOTAL_CONFIRMATIONS = 13  # 8 technical + 5 macro
MIN_CONFIRMATIONS_EXPANDED = 10  # proportional: 7/8 ≈ 10/13

# === FRED Configuration ===
FRED_SERIES = {
    "yield_10y": "DGS10",
    "yield_2y": "DGS2",
    "yield_3m": "DTB3",
    "fed_funds": "FEDFUNDS",
    "credit_spread": "BAMLH0A0HYM2",
    "initial_claims": "ICSA",
    "cpi": "CPIAUCSL",
}
FRED_CACHE_HOURS = 24

# === Market Internals Configuration ===
MARKET_INTERNAL_TICKERS = {
    "vix": "^VIX",
    "vix3m": "^VIX3M",
    "dxy": "DX-Y.NYB",
    "rsp": "RSP",
    "spy": "SPY",
}
VIX_CACHE_HOURS = 4
INTERNALS_CACHE_HOURS = 24

# === Cross-Asset Regime ===
CROSS_ASSET_TICKERS = ["TLT", "GLD", "USO"]
CROSS_ASSET_N_REGIMES = 3  # simpler model for daily data

# === Transition Forecasting ===
TRANSITION_FORECAST_HORIZONS = [1, 3, 6, 12, 24]  # hours ahead

# === Macro Event Calendar (FOMC, CPI, NFP dates) ===
MACRO_EVENTS_2025_2026 = {
    "FOMC": [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
    ],
    "CPI": [
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
        "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
        "2025-09-10", "2025-10-14", "2025-11-13", "2025-12-10",
        "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-14",
    ],
    "NFP": [
        "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
        "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
        "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
        "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    ],
}

# === Seasonal Tendencies (historical monthly avg S&P 500 returns) ===
SEASONAL_TENDENCIES = {
    1: 1.1,    # January
    2: -0.1,   # February
    3: 1.2,    # March
    4: 1.5,    # April
    5: 0.1,    # May
    6: 0.0,    # June
    7: 1.0,    # July
    8: -0.2,   # August
    9: -0.7,   # September
    10: 0.9,   # October
    11: 1.5,   # November
    12: 1.3,   # December
}

# === Backtester Configuration ===
INITIAL_CAPITAL = 100_000
LEVERAGE = 2.5
COOLDOWN_HOURS = 48  # hours after exit before re-entry allowed
MIN_REGIME_CONFIDENCE = 0.50  # minimum probability to act on regime

# === Data Configuration ===
DEFAULT_TICKER = "SPY"
DEFAULT_INTERVAL = "1h"  # hourly bars
DEFAULT_PERIOD = "730d"  # ~2 years (yfinance max for hourly)

TICKERS = {
    "market": ["SPY", "QQQ", "IWM", "DIA"],
    "sectors": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"],
    "volatility": ["^VIX"],
    "bonds": ["TLT", "IEF", "SHY", "HYG", "LQD"],
    "commodities": ["GLD", "USO", "DBA"],
    "breadth": ["^GSPC"],
}

# === Model Persistence ===
MODEL_AUTO_SAVE = True  # auto-save model after fitting

# === Confidence-Based Position Sizing ===
CONFIDENCE_TIERS = {
    0.90: 1.0,   # 90%+ → full size
    0.70: 0.75,  # 70-90% → 3/4 size
    0.50: 0.5,   # 50-70% → half size
    0.00: 0.0,   # <50% → skip
}

# === Optimizer Configuration ===
OPTIMIZER_N_RANGE = range(2, 9)       # test 2-8 states
OPTIMIZER_MIN_OOS_P = 0.10            # OOS p-value must be below this
OPTIMIZER_WF_FOLDS = 6                # abbreviated walk-forward folds

# === Dashboard ===
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "Medal-Lion Fund — Market Regime Terminal"
REFRESH_INTERVAL_SECONDS = 300
