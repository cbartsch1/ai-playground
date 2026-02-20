"""
Medallion 2.0 — Market Regime Terminal

Dark-themed Streamlit dashboard with:
- Current regime signal + confidence + multi-TF
- Candlestick chart with regime-colored backgrounds + FOMC/CPI/NFP markers
- 13-confirmation voting breakdown (8 technical + 5 macro)
- Regime quality metrics (forward returns, stability, separation)
- Walk-forward validation results
- Transition matrix heatmap + transition forecast
- Model selection (BIC/AIC)
- Macro Context tab (yield curve, credit spread, fed funds, claims, CPI, events)
- Market Internals tab (VIX term structure, breadth, DXY, cross-asset regimes)

Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from config.settings import (
    DASHBOARD_TITLE,
    REGIME_LABEL_COLORS,
    BULLISH_REGIMES,
    BEARISH_REGIMES,
    DEFAULT_TICKER,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    DEFAULT_N_REGIMES,
    HMM_FEATURES,
    MIN_CONFIRMATIONS,
    MIN_CONFIRMATIONS_EXPANDED,
    TOTAL_CONFIRMATIONS,
    MODEL_AUTO_SAVE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    MACRO_EVENTS_2025_2026,
    SEASONAL_TENDENCIES,
    CROSS_ASSET_TICKERS,
    CROSS_ASSET_N_REGIMES,
)
from models.hmm_regime import RegimeDetector
from data.data_loader import load_data, compute_hmm_features, compute_confirmations
from backtester.regime_quality import RegimeQualityAnalyzer


# === Page Config ===
st.set_page_config(
    page_title="Medallion 2.0",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Dark Theme CSS ===
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    div[data-testid="stMetric"] {
        background-color: #1a1d24;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e;
    }
    .signal-card {
        background-color: #1a1d24;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2d3139;
        text-align: center;
    }
    .signal-bullish { border-left: 4px solid #2ecc71; }
    .signal-bearish { border-left: 4px solid #e74c3c; }
    .signal-neutral { border-left: 4px solid #f39c12; }
    .conf-pass { color: #2ecc71; font-weight: bold; }
    .conf-fail { color: #e74c3c; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1d24;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stDataFrame { background-color: #1a1d24; }
</style>
""", unsafe_allow_html=True)

# === Title ===
st.markdown(f"# {DASHBOARD_TITLE}")
st.markdown("*HMM-based regime detection with 13-confirmation voting system*")


# === Sidebar ===
with st.sidebar:
    st.header("Configuration")

    ticker = st.text_input("Ticker", value=DEFAULT_TICKER)
    interval = st.selectbox("Interval", ["1h", "1d", "30m", "15m"], index=0)

    if interval in ("30m", "15m"):
        period_options = ["60d", "30d", "14d", "7d"]
    elif interval == "1h":
        period_options = ["730d", "365d", "180d", "90d"]
    else:
        period_options = ["730d", "365d", "180d", "90d", "60d"]
    period = st.selectbox("Period", period_options, index=0)

    n_regimes = st.slider("Number of Regimes", 2, 7, DEFAULT_N_REGIMES)

    st.divider()
    st.subheader("Model")
    col_load, col_refit = st.columns(2)
    with col_load:
        load_btn = st.button("Load Saved", type="primary", use_container_width=True)
    with col_refit:
        refit_btn = st.button("Refit Model", use_container_width=True)

    st.divider()
    include_macro = st.checkbox("Include Macro Data", value=True)
    if include_macro:
        min_confs = st.slider("Min Confirmations", 1, TOTAL_CONFIRMATIONS, MIN_CONFIRMATIONS_EXPANDED)
    else:
        min_confs = st.slider("Min Confirmations", 1, 8, MIN_CONFIRMATIONS)

    st.divider()
    st.caption("Medallion 2.0 | Inspired by Chuck_MF_Norris")

# === Data Freshness Indicator ===
cache_path = PROCESSED_DATA_DIR / f"{ticker.lower()}_{interval}_{period}_cache.parquet"
if cache_path.exists():
    cache_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age_hours = (datetime.now() - cache_mtime).total_seconds() / 3600
    freshness_color = "#2ecc71" if age_hours < 2 else ("#f39c12" if age_hours < 24 else "#e74c3c")
    freshness_text = f"Data cached {cache_mtime.strftime('%Y-%m-%d %H:%M')} ({age_hours:.0f}h ago)"
else:
    freshness_color = "#e74c3c"
    freshness_text = "No cached data — click Load or Refit"

st.markdown(f'<div style="text-align:right;color:{freshness_color};font-size:12px;margin-bottom:8px;">{freshness_text}</div>', unsafe_allow_html=True)


# === Main Logic ===
def load_and_predict():
    """Load data and either load saved model or fit fresh."""
    with st.spinner("Loading data..."):
        try:
            result = load_data(ticker, interval, period, cache=True, include_macro=include_macro)
            if include_macro:
                ohlcv, hmm_features, confirmations, macro_context = result
            else:
                ohlcv, hmm_features, confirmations = result
                macro_context = None
        except Exception as e:
            st.error(f"Data download failed: {e}")
            st.stop()

    return ohlcv, hmm_features, confirmations, macro_context


def try_load_model(n_regimes):
    """Try to load a previously saved model."""
    return RegimeDetector.load_latest(n_regimes=n_regimes)


def fit_model(hmm_features, n_regimes):
    """Fit a fresh model."""
    with st.spinner(f"Fitting {n_regimes}-state HMM on {len(hmm_features)} bars..."):
        detector = RegimeDetector(n_regimes=n_regimes, n_restarts=5, n_iter=100)
        detector.fit(hmm_features, feature_cols=HMM_FEATURES)
        if MODEL_AUTO_SAVE:
            saved_path = detector.save_latest()
            st.toast(f"Model saved: {saved_path.name}")
    return detector


# Determine action
should_run = load_btn or refit_btn or "detector" in st.session_state

if should_run:
    ohlcv, hmm_features, confirmations, macro_context = load_and_predict()

    # Model loading logic
    if load_btn or ("detector" not in st.session_state and not refit_btn):
        detector = try_load_model(n_regimes)
        if detector is not None:
            st.toast(f"Loaded saved {n_regimes}-state model")
        else:
            st.info("No saved model found. Fitting fresh model...")
            detector = fit_model(hmm_features, n_regimes)
    else:
        detector = fit_model(hmm_features, n_regimes)

    st.session_state["detector"] = detector

    # Predict regimes
    regime_predictions = detector.predict(hmm_features)

    # Get current state
    current = detector.get_current_regime(hmm_features)

    # Transition forecast
    last_row = regime_predictions.dropna(subset=["confidence"]).iloc[-1]
    current_probs = np.array([last_row[f"prob_{i}"] for i in range(n_regimes)])
    transition_forecast = detector.forecast_transitions(current_probs)

    # Data info
    st.caption(f"Data: **{ticker}** | **{interval}** | **{len(ohlcv):,} bars** | {ohlcv.index[0].strftime('%Y-%m-%d')} to {ohlcv.index[-1].strftime('%Y-%m-%d')}")

    # === TOP ROW: Signal + Regime + Confidence + Transition Risk ===
    col1, col2, col3, col4 = st.columns(4)

    signal_color = "#2ecc71" if current["signal"] == "bullish" else (
        "#e74c3c" if current["signal"] == "bearish" else "#f39c12"
    )

    with col1:
        st.markdown(f"""
        <div class="signal-card signal-{current['signal']}">
            <h4 style="color: #8b949e; margin: 0;">Current Signal</h4>
            <h2 style="color: {signal_color}; margin: 5px 0;">{current['signal'].upper()}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        regime_color = REGIME_LABEL_COLORS.get(current["label"], "#888")
        st.markdown(f"""
        <div class="signal-card">
            <h4 style="color: #8b949e; margin: 0;">Detected Regime</h4>
            <h2 style="color: {regime_color}; margin: 5px 0;">{current['label']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        conf_pct = current["confidence"] * 100
        conf_color = "#2ecc71" if conf_pct > 70 else ("#f39c12" if conf_pct > 50 else "#e74c3c")
        st.markdown(f"""
        <div class="signal-card">
            <h4 style="color: #8b949e; margin: 0;">Confidence</h4>
            <h2 style="color: {conf_color}; margin: 5px 0;">{conf_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        alert_level = transition_forecast.get("alert_level", "stable")
        alert_colors = {"stable": "#2ecc71", "watch": "#f39c12", "warning": "#e67e22", "critical": "#e74c3c"}
        alert_color = alert_colors.get(alert_level, "#888")
        p6 = transition_forecast.get("p_change", {}).get(6, 0)
        st.markdown(f"""
        <div class="signal-card">
            <h4 style="color: #8b949e; margin: 0;">Transition Risk</h4>
            <h2 style="color: {alert_color}; margin: 5px 0;">{alert_level.upper()}</h2>
            <div style="color: #8b949e; font-size: 12px;">P(change 6h) = {p6:.0%} | Streak: {current['streak']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # === TABS ===
    tab_names = [
        "Price & Regimes",
        "Confirmations",
        "Regime Quality",
        "Walk-Forward",
        "Transition Matrix",
        "Model Selection",
    ]
    if include_macro:
        tab_names += ["Macro Context", "Market Internals"]
    tab_names.append("Strategy Integration")

    tabs = st.tabs(tab_names)

    # === TAB 1: Price Chart with Regime Overlay + Event Markers ===
    with tabs[0]:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.65, 0.35],
            subplot_titles=[f"{ticker} — Regime Overlay", "Regime Probabilities"],
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=ohlcv.index,
                open=ohlcv["Open"],
                high=ohlcv["High"],
                low=ohlcv["Low"],
                close=ohlcv["Close"],
                name=ticker,
                increasing_line_color="#2ecc71",
                decreasing_line_color="#e74c3c",
            ),
            row=1, col=1,
        )

        # Regime shading
        if "regime_label" in regime_predictions.columns:
            aligned = regime_predictions["regime_label"].reindex(ohlcv.index)
            price_range = ohlcv["High"].max() - ohlcv["Low"].min()
            bar_colors = [
                REGIME_LABEL_COLORS.get(label, "#888888") if pd.notna(label) else "rgba(0,0,0,0)"
                for label in aligned.values
            ]
            fig.add_trace(
                go.Bar(
                    x=ohlcv.index,
                    y=[price_range * 0.002] * len(ohlcv),
                    base=[ohlcv["Low"].min()] * len(ohlcv),
                    marker_color=bar_colors,
                    marker_line_width=0,
                    opacity=0.7,
                    name="Regime",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )

        # FOMC/CPI/NFP event markers
        if include_macro:
            event_colors = {"FOMC": "#3498db", "CPI": "#e67e22", "NFP": "#2ecc71"}
            data_start = ohlcv.index[0]
            data_end = ohlcv.index[-1]
            for event_type, dates in MACRO_EVENTS_2025_2026.items():
                color = event_colors.get(event_type, "#888")
                for date_str in dates:
                    event_date = pd.Timestamp(date_str)
                    if hasattr(data_start, 'tz') and data_start.tz is not None:
                        event_date = event_date.tz_localize(data_start.tz)
                    if data_start <= event_date <= data_end:
                        fig.add_vline(
                            x=event_date, line_dash="dot",
                            line_color=color, line_width=1,
                            opacity=0.5, row=1, col=1,
                        )

        # Regime probability stacked area
        for i in range(n_regimes):
            col_name = f"prob_{i}"
            if col_name in regime_predictions.columns:
                label = detector.regime_labels.get(i, f"Regime {i}")
                color = REGIME_LABEL_COLORS.get(label, "#888888")
                fig.add_trace(
                    go.Scatter(
                        x=regime_predictions.index,
                        y=regime_predictions[col_name],
                        name=label,
                        fill="tonexty" if i > 0 else "tozeroy",
                        line=dict(color=color, width=0.5),
                        stackgroup="probs",
                    ),
                    row=2, col=1,
                )

        fig.update_layout(
            height=700,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=10),
            ),
            margin=dict(l=50, r=50, t=60, b=50),
            xaxis_rangeslider_visible=False,
        )
        fig.update_xaxes(gridcolor="#1a1d24")
        fig.update_yaxes(gridcolor="#1a1d24")

        st.plotly_chart(fig, use_container_width=True)

        if include_macro:
            st.caption("Event markers: 🔵 FOMC | 🟠 CPI | 🟢 NFP")

    # === TAB 2: Confirmations ===
    with tabs[1]:
        total_available = TOTAL_CONFIRMATIONS if include_macro else 8
        st.subheader(f"{total_available}-Confirmation Voting System")
        st.markdown(f"**Minimum required**: {min_confs}/{total_available} to enter a position")

        last_confs = confirmations.iloc[-1]
        total_met = int(last_confs.get("total_confirmations_met", last_confs["confirmations_met"]))

        # Technical confirmations header
        st.markdown("#### Technical (8)")
        conf_cols = st.columns(4)
        conf_items = [
            ("RSI", "rsi", f"{last_confs['rsi']:.1f}", "< 90", last_confs["rsi_pass"]),
            ("Momentum", "momentum", f"{last_confs['momentum']*100:.2f}%", "> 1%", last_confs["momentum_pass"]),
            ("Volatility", "volatility", f"{last_confs['volatility']*100:.2f}%", "< 6%", last_confs["volatility_pass"]),
            ("Volume", "volume_ratio", f"{last_confs['volume_ratio']:.2f}x", "> 1.0x SMA", last_confs["volume_pass"]),
            ("ADX", "adx", f"{last_confs['adx']:.1f}", "> 25", last_confs["adx_pass"]),
            ("EMA 50", "ema_50", f"${last_confs['ema_50']:.2f}", "Price above", last_confs["ema_50_pass"]),
            ("EMA 200", "ema_200", f"${last_confs['ema_200']:.2f}", "Price above", last_confs["ema_200_pass"]),
            ("MACD", "macd_histogram", f"{last_confs['macd_histogram']:.4f}", "Line > Signal", last_confs["macd_pass"]),
        ]

        for i, (name, key, value, threshold, passed) in enumerate(conf_items):
            with conf_cols[i % 4]:
                color = "#2ecc71" if passed else "#e74c3c"
                icon = "PASS" if passed else "FAIL"
                st.markdown(f"""
                <div style="background: #1a1d24; border-radius: 8px; padding: 12px; margin-bottom: 8px;
                            border-left: 3px solid {color};">
                    <div style="color: #8b949e; font-size: 12px;">{name}</div>
                    <div style="color: white; font-size: 18px; font-weight: bold;">{icon} — {value}</div>
                    <div style="color: #8b949e; font-size: 11px;">Threshold: {threshold}</div>
                </div>
                """, unsafe_allow_html=True)

        # Macro confirmations (if available)
        if include_macro and "vix_term_structure_pass" in confirmations.columns:
            st.markdown("#### Macro (5)")
            macro_cols = st.columns(5)
            macro_items = [
                ("VIX Term", "vix_term_structure", "< 1.0", "vix_term_structure_pass"),
                ("Credit", "credit_spread_val", "< 5.0%", "credit_spread_pass"),
                ("Yield Curve", "yield_curve", "> 0", "yield_curve_pass"),
                ("Breadth", "breadth_slope", "Slope > 0", "breadth_pass"),
                ("Dollar", "dxy_weekly_chg", "< 1% wk", "dollar_pass"),
            ]
            for i, (name, val_col, threshold, pass_col) in enumerate(macro_items):
                with macro_cols[i]:
                    val = last_confs.get(val_col, np.nan)
                    passed = bool(last_confs.get(pass_col, False))
                    color = "#2ecc71" if passed else "#e74c3c"
                    icon = "PASS" if passed else "FAIL"
                    val_str = f"{val:.3f}" if pd.notna(val) else "N/A"
                    st.markdown(f"""
                    <div style="background: #1a1d24; border-radius: 8px; padding: 12px; margin-bottom: 8px;
                                border-left: 3px solid {color};">
                        <div style="color: #8b949e; font-size: 12px;">{name}</div>
                        <div style="color: white; font-size: 18px; font-weight: bold;">{icon} — {val_str}</div>
                        <div style="color: #8b949e; font-size: 11px;">Threshold: {threshold}</div>
                    </div>
                    """, unsafe_allow_html=True)

        summary_color = "#2ecc71" if total_met >= min_confs else "#e74c3c"
        st.markdown(f"""
        <div style="background: #1a1d24; border-radius: 8px; padding: 16px; text-align: center;
                    border: 2px solid {summary_color}; margin-top: 16px;">
            <h3 style="color: {summary_color}; margin: 0;">
                {total_met}/{total_available} Confirmations Met
                {"— ENTRY ALLOWED" if total_met >= min_confs else "— ENTRY BLOCKED"}
            </h3>
        </div>
        """, unsafe_allow_html=True)

        # Confirmation history chart
        st.subheader("Confirmation Count Over Time")
        conf_col_name = "total_confirmations_met" if "total_confirmations_met" in confirmations.columns else "confirmations_met"
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=confirmations.index,
            y=confirmations[conf_col_name],
            fill="tozeroy",
            line=dict(color="#3498db", width=1),
            fillcolor="rgba(52, 152, 219, 0.2)",
        ))
        fig_conf.add_hline(
            y=min_confs, line_dash="dash", line_color="#f39c12",
            annotation_text=f"Min: {min_confs}/{total_available}",
        )
        fig_conf.update_layout(
            height=250,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            yaxis_title="Confirmations Met",
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # === TAB 3: Regime Quality ===
    with tabs[2]:
        st.subheader("Regime Quality Analysis")
        st.markdown("*How useful is the regime detector as a trading filter?*")

        analyzer = RegimeQualityAnalyzer()
        quality = analyzer.analyze(ohlcv, regime_predictions)

        # Quality Score
        score = quality.summary_score
        score_color = "#2ecc71" if score >= 60 else ("#f39c12" if score >= 40 else "#e74c3c")
        st.markdown(f"""
        <div style="background: #1a1d24; border-radius: 10px; padding: 20px; text-align: center;
                    border: 2px solid {score_color}; margin-bottom: 20px;">
            <h4 style="color: #8b949e; margin: 0;">Regime Quality Score</h4>
            <h1 style="color: {score_color}; margin: 5px 0;">{score:.0f} / 100</h1>
        </div>
        """, unsafe_allow_html=True)

        # Filter Value
        st.subheader("Filter Value — Regime-Filtered vs Buy & Hold")
        fv = quality.filter_value
        fv1, fv2, fv3, fv4 = st.columns(4)
        with fv1:
            st.metric("Buy & Hold Return", f"{fv['buy_hold_return']:.1f}%")
        with fv2:
            st.metric("Filtered Return", f"{fv['filtered_return']:.1f}%")
        with fv3:
            st.metric("Filtered Sharpe", f"{fv['filtered_sharpe']:.2f}", f"B&H: {fv['buy_hold_sharpe']:.2f}")
        with fv4:
            st.metric("Time in Market", f"{fv['time_in_market']:.0f}%")

        dd1, dd2 = st.columns(2)
        with dd1:
            st.metric("B&H Max Drawdown", f"{fv['buy_hold_max_dd']:.1f}%")
        with dd2:
            st.metric("Filtered Max Drawdown", f"{fv['filtered_max_dd']:.1f}%")

        # Forward Returns by Regime
        st.subheader("Forward Returns by Regime")
        if quality.forward_returns is not None and len(quality.forward_returns) > 0:
            fwd = quality.forward_returns.copy()
            display_cols = ["label", "count", "fwd_1h_mean", "fwd_4h_mean", "fwd_1d_mean", "fwd_1w_mean"]
            available = [c for c in display_cols if c in fwd.columns]
            rename_map = {
                "label": "Regime", "count": "Bars",
                "fwd_1h_mean": "+1h (%)", "fwd_4h_mean": "+4h (%)",
                "fwd_1d_mean": "+1d (%)", "fwd_1w_mean": "+1w (%)",
            }
            st.dataframe(
                fwd[available].rename(columns=rename_map).style.format(
                    {k: "{:.4f}" for k in rename_map.values() if "%" in k},
                    na_rep="—",
                ),
                use_container_width=True,
            )

        # Regime Separation
        st.subheader("Regime Separation — Statistical Test")
        if quality.regime_separation is not None and len(quality.regime_separation) > 0:
            sep = quality.regime_separation
            st.dataframe(sep, use_container_width=True)

            bull_row = sep[sep["signal"] == "bullish"]
            if not bull_row.empty and "p_value_vs_bearish" in bull_row.columns:
                p = bull_row["p_value_vs_bearish"].iloc[0]
                if pd.notna(p):
                    sig_text = "SIGNIFICANT" if p < 0.05 else "NOT significant"
                    sig_color = "#2ecc71" if p < 0.05 else "#e74c3c"
                    st.markdown(f"**Bullish vs Bearish returns**: p = {p:.4f} — <span style='color:{sig_color}'>{sig_text}</span>", unsafe_allow_html=True)

        # Stability Metrics
        st.subheader("Regime Stability")
        if quality.stability_metrics is not None and len(quality.stability_metrics) > 0:
            stab = quality.stability_metrics
            stab_cols = ["label", "total_bars", "pct_time", "num_episodes", "avg_duration", "false_alarm_rate"]
            available = [c for c in stab_cols if c in stab.columns]
            st.dataframe(
                stab[available].rename(columns={
                    "label": "Regime", "total_bars": "Bars", "pct_time": "% Time",
                    "num_episodes": "Episodes", "avg_duration": "Avg Duration",
                    "false_alarm_rate": "False Alarm %",
                }),
                use_container_width=True,
            )

    # === TAB 4: Walk-Forward Results ===
    with tabs[3]:
        st.subheader("Walk-Forward Validation")
        st.markdown("*Out-of-sample validation with expanding training window*")

        run_wf = st.button("Run Walk-Forward Validation", type="primary")

        if run_wf:
            with st.spinner("Running walk-forward validation (this may take a minute)..."):
                try:
                    from scripts.walk_forward import run_walk_forward
                    wf_results = run_walk_forward(
                        n_regimes=n_regimes,
                        min_train_months=6,
                        ticker=ticker,
                        interval=interval,
                        period=period,
                    )
                    st.session_state["wf_results"] = wf_results
                except Exception as e:
                    st.error(f"Walk-forward failed: {e}")
                    wf_results = None
        elif "wf_results" in st.session_state:
            wf_results = st.session_state["wf_results"]
        else:
            wf_results = None

        if wf_results:
            folds = wf_results.get("folds", [])

            # OOS Regime Test
            oos = wf_results.get("oos_regime_test", {})
            if "bullish_mean" in oos:
                oos1, oos2, oos3 = st.columns(3)
                with oos1:
                    st.metric("Bullish OOS Return", f"{oos['bullish_mean']:.4f}%")
                with oos2:
                    st.metric("Bearish OOS Return", f"{oos['bearish_mean']:.4f}%")
                with oos3:
                    sig = oos.get("significant", False)
                    p = oos.get("p_value", 1.0)
                    st.metric("p-value", f"{p:.4f}", "SIGNIFICANT" if sig else "Not significant")

            # Transition stability
            ts = wf_results.get("transition_stability", {})
            ts1, ts2 = st.columns(2)
            with ts1:
                st.metric("Transition Stability", f"{ts.get('mean_diff', 0):.4f}", "Stable" if ts.get("stable") else "Unstable")
            with ts2:
                st.metric("Avg Confidence (OOS)", f"{wf_results.get('avg_confidence', 0):.1%}")

            # Per-fold table
            if folds:
                st.subheader("Per-Fold Results")
                fold_df = pd.DataFrame([{
                    "Fold": f["fold"],
                    "Test Month": f["test_month"],
                    "Bars": f["test_bars"],
                    "Changes": f["regime_changes"],
                    "Avg Duration": f"{f['avg_duration']:.1f}",
                    "Confidence": f"{f['avg_confidence']:.1%}",
                } for f in folds])
                st.dataframe(fold_df, use_container_width=True)

                # Charts
                fig_wf = make_subplots(rows=1, cols=2, subplot_titles=["OOS Confidence", "Regime Changes"])
                fig_wf.add_trace(go.Bar(
                    x=[f["fold"] for f in folds],
                    y=[f["avg_confidence"] for f in folds],
                    marker_color="#3498db",
                ), row=1, col=1)
                fig_wf.add_trace(go.Bar(
                    x=[f["fold"] for f in folds],
                    y=[f["regime_changes"] for f in folds],
                    marker_color="#e74c3c",
                ), row=1, col=2)
                fig_wf.update_layout(
                    height=300,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    showlegend=False,
                    margin=dict(l=50, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_wf, use_container_width=True)
        else:
            st.info("Click **Run Walk-Forward Validation** to analyze OOS regime consistency.")

    # === TAB 5: Transition Matrix + Forecast ===
    with tabs[4]:
        st.subheader("Regime Transition Probabilities")
        trans_matrix = detector.get_transition_matrix()

        fig_trans = go.Figure(data=go.Heatmap(
            z=trans_matrix.values,
            x=trans_matrix.columns,
            y=trans_matrix.index,
            colorscale="RdYlGn",
            text=np.round(trans_matrix.values, 3),
            texttemplate="%{text:.1%}",
            textfont={"size": 11},
            zmin=0, zmax=1,
        ))
        fig_trans.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            margin=dict(l=150, r=50, t=40, b=150),
        )
        st.plotly_chart(fig_trans, use_container_width=True)

        # Transition Forecast
        st.subheader("Transition Forecast")
        alert_level = transition_forecast.get("alert_level", "stable")
        alert_colors = {"stable": "#2ecc71", "watch": "#f39c12", "warning": "#e67e22", "critical": "#e74c3c"}

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown(f"""
            <div style="background:#1a1d24;border-radius:8px;padding:16px;text-align:center;
                        border:2px solid {alert_colors.get(alert_level, '#888')};">
                <div style="color:#8b949e;">Alert Level</div>
                <h2 style="color:{alert_colors.get(alert_level, '#888')};margin:5px 0;">{alert_level.upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
        with fc2:
            st.metric("Most Likely Next", transition_forecast.get("most_likely_next", "?"))
        with fc3:
            st.metric("P(Bearish)", f"{transition_forecast.get('p_bearish', 0):.1%}")

        # P(change) line chart
        p_change = transition_forecast.get("p_change", {})
        if p_change:
            fig_fc = go.Figure()
            horizons = sorted(p_change.keys())
            fig_fc.add_trace(go.Scatter(
                x=[f"{h}h" for h in horizons],
                y=[p_change[h] for h in horizons],
                mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=8),
                name="P(regime change)",
            ))
            fig_fc.add_hline(y=0.2, line_dash="dash", line_color="#2ecc71", annotation_text="Stable")
            fig_fc.add_hline(y=0.4, line_dash="dash", line_color="#f39c12", annotation_text="Watch")
            fig_fc.add_hline(y=0.6, line_dash="dash", line_color="#e74c3c", annotation_text="Warning")
            fig_fc.update_layout(
                height=300,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                yaxis_title="P(regime change)",
                xaxis_title="Forecast Horizon",
                yaxis_range=[0, 1],
                margin=dict(l=50, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        # Regime Statistics table
        st.subheader("Regime Statistics")
        regime_stats = detector.get_regime_stats(hmm_features, ohlcv)
        display_cols = ["label", "days", "pct_time", "mean_return", "annualized_vol", "sharpe", "expected_duration"]
        available_cols = [c for c in display_cols if c in regime_stats.columns]
        st.dataframe(
            regime_stats[available_cols].rename(columns={
                "label": "Regime", "days": "Bars", "pct_time": "% Time",
                "mean_return": "Mean Return", "annualized_vol": "Ann. Vol",
                "sharpe": "Sharpe", "expected_duration": "Exp. Duration",
            }),
            use_container_width=True,
        )

        # Persistence interpretation
        st.subheader("Regime Persistence (Diagonal)")
        for i in range(n_regimes):
            label = detector.regime_labels.get(i, f"Regime {i}")
            self_prob = trans_matrix.iloc[i, i] if i < len(trans_matrix) else 0
            expected_dur = 1 / (1 - self_prob) if self_prob < 1 else float("inf")
            st.markdown(f"**{label}**: {self_prob:.1%} self-transition = expected duration: **{expected_dur:.0f} bars**")

    # === TAB 6: Model Selection ===
    with tabs[5]:
        st.subheader("Model Selection — BIC/AIC Comparison")
        st.markdown("*Tests 2-8 regime models to find optimal state count.*")

        run_ms = st.button("Run Model Selection", type="primary")
        if not run_ms and "ms_results" in st.session_state:
            ms_results = st.session_state["ms_results"]
        elif run_ms:
            with st.spinner("Fitting 7 models (2-8 regimes)..."):
                ms_results = detector.model_selection(hmm_features, max_regimes=8)
                st.session_state["ms_results"] = ms_results
        else:
            st.info("Click **Run Model Selection** to compare 2-8 regime models.")
            ms_results = None

        if ms_results is not None and ("error" not in ms_results.columns or ms_results["error"].isna().all()):
            fig_ms = make_subplots(specs=[[{"secondary_y": True}]])

            fig_ms.add_trace(
                go.Scatter(
                    x=ms_results["n_regimes"], y=ms_results["bic"],
                    name="BIC", line=dict(color="#e74c3c", width=2),
                    mode="lines+markers",
                ),
                secondary_y=False,
            )
            fig_ms.add_trace(
                go.Scatter(
                    x=ms_results["n_regimes"], y=ms_results["aic"],
                    name="AIC", line=dict(color="#3498db", width=2),
                    mode="lines+markers",
                ),
                secondary_y=False,
            )
            fig_ms.add_trace(
                go.Scatter(
                    x=ms_results["n_regimes"], y=ms_results["log_likelihood"],
                    name="Log Likelihood", line=dict(color="#2ecc71", width=2, dash="dot"),
                    mode="lines+markers",
                ),
                secondary_y=True,
            )

            if "bic" in ms_results.columns:
                best_bic = ms_results.loc[ms_results["bic"].idxmin()]
                fig_ms.add_vline(
                    x=best_bic["n_regimes"],
                    line_dash="dash", line_color="#f39c12",
                    annotation_text=f"Best BIC: {int(best_bic['n_regimes'])} states",
                )

            fig_ms.update_layout(
                height=400,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                xaxis_title="Number of Regimes",
                margin=dict(l=50, r=50, t=40, b=40),
            )
            fig_ms.update_yaxes(title_text="BIC / AIC", secondary_y=False)
            fig_ms.update_yaxes(title_text="Log Likelihood", secondary_y=True)

            st.plotly_chart(fig_ms, use_container_width=True)
            st.dataframe(ms_results, use_container_width=True)

            # Apply Best button
            if "bic" in ms_results.columns and not ms_results["bic"].isna().all():
                best_bic_row = ms_results.loc[ms_results["bic"].idxmin()]
                best_n = int(best_bic_row["n_regimes"])
                if best_n != n_regimes:
                    if st.button(f"Apply Best ({best_n} states)", type="primary"):
                        with st.spinner(f"Refitting with {best_n} states..."):
                            new_detector = RegimeDetector(n_regimes=best_n, n_restarts=5, n_iter=100)
                            new_detector.fit(hmm_features, feature_cols=HMM_FEATURES)
                            if MODEL_AUTO_SAVE:
                                saved = new_detector.save_latest()
                                st.toast(f"Saved {best_n}-state model: {saved.name}")
                            st.session_state["detector"] = new_detector
                            st.rerun()
                else:
                    st.success(f"Current model ({n_regimes} states) already matches best BIC.")
        elif ms_results is not None:
            st.warning("Some models failed to converge:")
            st.dataframe(ms_results, use_container_width=True)

    # === TAB 7: Macro Context ===
    if include_macro and len(tabs) > 6:
        with tabs[6]:
            st.subheader("Macro Context — Economic Environment")

            if macro_context and macro_context.get("macro_raw") is not None and not macro_context["macro_raw"].empty:
                macro_raw = macro_context["macro_raw"]

                # Yield Curve Chart
                st.markdown("#### Yield Curve (10Y - 2Y)")
                if "yield_10y" in macro_raw.columns and "yield_2y" in macro_raw.columns:
                    yc = macro_raw["yield_10y"] - macro_raw["yield_2y"]
                    fig_yc = go.Figure()
                    fig_yc.add_trace(go.Scatter(
                        x=yc.index, y=yc.values,
                        fill="tozeroy",
                        line=dict(color="#3498db", width=1.5),
                        fillcolor="rgba(52,152,219,0.15)",
                        name="10Y - 2Y",
                    ))
                    fig_yc.add_hline(y=0, line_dash="dash", line_color="#e74c3c", annotation_text="Inversion")
                    # Red shading when inverted
                    inverted = yc[yc < 0]
                    if not inverted.empty:
                        fig_yc.add_trace(go.Scatter(
                            x=inverted.index, y=inverted.values,
                            fill="tozeroy",
                            fillcolor="rgba(231,76,60,0.3)",
                            line=dict(color="#e74c3c", width=0),
                            showlegend=False,
                        ))
                    fig_yc.update_layout(
                        height=250, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        yaxis_title="Spread (%)", margin=dict(l=50, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_yc, use_container_width=True)

                # Fed Funds Rate
                st.markdown("#### Fed Funds Rate")
                if "fed_funds" in macro_raw.columns:
                    ff = macro_raw["fed_funds"].dropna()
                    fig_ff = go.Figure()
                    fig_ff.add_trace(go.Scatter(
                        x=ff.index, y=ff.values,
                        line=dict(color="#f39c12", width=2, shape="hv"),
                        name="Fed Funds",
                    ))
                    # Add FOMC date annotations
                    for date_str in MACRO_EVENTS_2025_2026.get("FOMC", []):
                        fomc_date = pd.Timestamp(date_str)
                        if ff.index[0] <= fomc_date <= ff.index[-1]:
                            fig_ff.add_vline(x=fomc_date, line_dash="dot", line_color="#3498db", opacity=0.4)
                    fig_ff.update_layout(
                        height=250, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        yaxis_title="Rate (%)", margin=dict(l=50, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_ff, use_container_width=True)

                # Credit Spread
                st.markdown("#### Credit Spread (HY OAS)")
                if "credit_spread" in macro_raw.columns:
                    cs = macro_raw["credit_spread"].dropna()
                    p75 = cs.quantile(0.75)
                    fig_cs = go.Figure()
                    fig_cs.add_trace(go.Scatter(
                        x=cs.index, y=cs.values,
                        line=dict(color="#e67e22", width=1.5),
                        name="HY OAS",
                    ))
                    fig_cs.add_hline(y=p75, line_dash="dash", line_color="#e74c3c",
                                     annotation_text=f"75th pctl: {p75:.2f}%")
                    fig_cs.update_layout(
                        height=250, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        yaxis_title="Spread (%)", margin=dict(l=50, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_cs, use_container_width=True)

                # Initial Claims
                st.markdown("#### Initial Jobless Claims")
                if "initial_claims" in macro_raw.columns:
                    claims = macro_raw["initial_claims"].dropna()
                    fig_cl = go.Figure()
                    colors = ["#e74c3c" if v > 300000 else "#2ecc71" for v in claims.values]
                    fig_cl.add_trace(go.Bar(
                        x=claims.index, y=claims.values,
                        marker_color=colors, name="Claims",
                    ))
                    fig_cl.add_hline(y=300000, line_dash="dash", line_color="#f39c12",
                                     annotation_text="300K threshold")
                    fig_cl.update_layout(
                        height=250, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        yaxis_title="Claims", margin=dict(l=50, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_cl, use_container_width=True)

                # CPI
                st.markdown("#### CPI (Consumer Price Index)")
                if "cpi" in macro_raw.columns:
                    cpi = macro_raw["cpi"].dropna()
                    cpi_yoy = cpi.pct_change(12) * 100  # year-over-year
                    fig_cpi = go.Figure()
                    fig_cpi.add_trace(go.Scatter(
                        x=cpi_yoy.index, y=cpi_yoy.values,
                        line=dict(color="#8e44ad", width=1.5),
                        name="CPI YoY %",
                    ))
                    fig_cpi.add_hline(y=2.0, line_dash="dash", line_color="#2ecc71",
                                      annotation_text="2% Target")
                    fig_cpi.update_layout(
                        height=250, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        yaxis_title="YoY Change (%)", margin=dict(l=50, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_cpi, use_container_width=True)

                # Event Calendar
                st.markdown("#### Upcoming Macro Events")
                today = datetime.now().strftime("%Y-%m-%d")
                upcoming = []
                for etype, dates in MACRO_EVENTS_2025_2026.items():
                    for d in dates:
                        if d >= today:
                            upcoming.append({"Date": d, "Event": etype})
                if upcoming:
                    upcoming.sort(key=lambda x: x["Date"])
                    st.dataframe(pd.DataFrame(upcoming[:15]), use_container_width=True)

                # Seasonal Tendencies
                st.markdown("#### Seasonal Tendencies (Historical Monthly S&P 500 Returns)")
                months = list(range(1, 13))
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                returns = [SEASONAL_TENDENCIES.get(m, 0) for m in months]
                colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in returns]
                fig_seas = go.Figure()
                fig_seas.add_trace(go.Bar(
                    x=month_names, y=returns, marker_color=colors,
                ))
                fig_seas.update_layout(
                    height=250, template="plotly_dark",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    yaxis_title="Avg Return (%)", margin=dict(l=50, r=20, t=20, b=40),
                )
                st.plotly_chart(fig_seas, use_container_width=True)

                # Macro Health Summary
                st.markdown("#### Macro Health Summary")
                health_items = []
                if "yield_curve" in confirmations.columns:
                    yc_val = confirmations["yield_curve"].dropna().iloc[-1] if "yield_curve" in confirmations.columns else np.nan
                    health_items.append(("Yield Curve", "green" if yc_val > 0 else "red"))
                if "credit_spread_val" in confirmations.columns:
                    cs_val = confirmations["credit_spread_val"].dropna().iloc[-1] if "credit_spread_val" in confirmations.columns else np.nan
                    health_items.append(("Credit", "green" if cs_val < 5.0 else ("yellow" if cs_val < 7.0 else "red")))

                if health_items:
                    health_cols = st.columns(len(health_items))
                    traffic_colors = {"green": "#2ecc71", "yellow": "#f39c12", "red": "#e74c3c"}
                    for i, (name, status) in enumerate(health_items):
                        with health_cols[i]:
                            st.markdown(f"""
                            <div style="background:#1a1d24;border-radius:8px;padding:12px;text-align:center;
                                        border:2px solid {traffic_colors[status]};">
                                <div style="color:#8b949e;font-size:12px;">{name}</div>
                                <div style="color:{traffic_colors[status]};font-size:24px;">●</div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("No macro data available. Check FRED API key in .env file.")

    # === TAB 8: Market Internals ===
    if include_macro and len(tabs) > 7:
        with tabs[7]:
            st.subheader("Market Internals — VIX, Breadth, Dollar, Cross-Asset")

            if macro_context:
                # VIX Term Structure
                st.markdown("#### VIX Term Structure")
                vix_data = macro_context.get("vix")
                if vix_data is not None and not vix_data.empty:
                    fig_vix = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                            subplot_titles=["VIX vs VIX3M", "VIX/VIX3M Ratio"])

                    fig_vix.add_trace(go.Scatter(
                        x=vix_data.index, y=vix_data["vix"],
                        name="VIX", line=dict(color="#e74c3c", width=1.5),
                    ), row=1, col=1)
                    fig_vix.add_trace(go.Scatter(
                        x=vix_data.index, y=vix_data["vix3m"],
                        name="VIX3M", line=dict(color="#3498db", width=1.5),
                    ), row=1, col=1)

                    # Ratio with contango/backwardation coloring
                    ratio_colors = ["#2ecc71" if r < 1.0 else "#e74c3c" for r in vix_data["vix_ratio"]]
                    fig_vix.add_trace(go.Bar(
                        x=vix_data.index, y=vix_data["vix_ratio"],
                        marker_color=ratio_colors, name="Ratio",
                        showlegend=False,
                    ), row=2, col=1)
                    fig_vix.add_hline(y=1.0, line_dash="dash", line_color="#f39c12", row=2, col=1)

                    fig_vix.update_layout(
                        height=400, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        margin=dict(l=50, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_vix, use_container_width=True)

                    last_vix = vix_data.iloc[-1]
                    contango = "Contango (complacent)" if last_vix["vix_contango"] else "Backwardation (fear)"
                    st.caption(f"VIX: {last_vix['vix']:.1f} | VIX3M: {last_vix['vix3m']:.1f} | Ratio: {last_vix['vix_ratio']:.3f} | {contango}")

                # Market Breadth
                st.markdown("#### Market Breadth (RSP/SPY)")
                breadth_data = macro_context.get("breadth")
                if breadth_data is not None and not breadth_data.empty:
                    fig_br = go.Figure()
                    fig_br.add_trace(go.Scatter(
                        x=breadth_data.index, y=breadth_data["rsp_spy_ratio"],
                        name="RSP/SPY", line=dict(color="#3498db", width=1),
                    ))
                    if "breadth_sma20" in breadth_data.columns:
                        fig_br.add_trace(go.Scatter(
                            x=breadth_data.index, y=breadth_data["breadth_sma20"],
                            name="20-SMA", line=dict(color="#f39c12", width=1.5),
                        ))
                    fig_br.update_layout(
                        height=250, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        yaxis_title="RSP/SPY Ratio",
                        margin=dict(l=50, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_br, use_container_width=True)

                # Dollar Strength
                st.markdown("#### Dollar Strength (DXY)")
                dxy_data = macro_context.get("dxy")
                if dxy_data is not None and not dxy_data.empty:
                    fig_dxy = go.Figure()
                    fig_dxy.add_trace(go.Scatter(
                        x=dxy_data.index, y=dxy_data["dxy"],
                        name="DXY", line=dict(color="#2ecc71", width=1),
                    ))
                    if "dxy_sma50" in dxy_data.columns:
                        fig_dxy.add_trace(go.Scatter(
                            x=dxy_data.index, y=dxy_data["dxy_sma50"],
                            name="50-SMA", line=dict(color="#f39c12", width=1.5),
                        ))
                    fig_dxy.update_layout(
                        height=250, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        yaxis_title="DXY",
                        margin=dict(l=50, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_dxy, use_container_width=True)

                # Put/Call Ratio (if available)
                pc_data = macro_context.get("put_call")
                if pc_data is not None and not pc_data.empty:
                    st.markdown("#### Put/Call Ratio")
                    fig_pc = go.Figure()
                    fig_pc.add_trace(go.Scatter(
                        x=pc_data.index, y=pc_data["put_call"],
                        name="P/C Ratio", line=dict(color="#8e44ad", width=1),
                    ))
                    if "put_call_sma10" in pc_data.columns:
                        fig_pc.add_trace(go.Scatter(
                            x=pc_data.index, y=pc_data["put_call_sma10"],
                            name="10-SMA", line=dict(color="#f39c12", width=1.5),
                        ))
                    fig_pc.update_layout(
                        height=250, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        yaxis_title="P/C Ratio",
                        margin=dict(l=50, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_pc, use_container_width=True)

                # Cross-Asset Regime Panel
                st.markdown("#### Cross-Asset Regimes")
                st.caption("3-state HMM (Bull/Neutral/Bear) on daily data for TLT, GLD, USO")

                run_cross = st.button("Fit Cross-Asset Regimes", type="secondary")
                if run_cross:
                    with st.spinner("Fitting cross-asset HMMs..."):
                        try:
                            from data.market_internals import download_cross_asset, compute_cross_asset_features
                            cross_data = download_cross_asset(CROSS_ASSET_TICKERS)
                            cross_results = {}
                            for ca_ticker, ca_ohlcv in cross_data.items():
                                ca_features = compute_cross_asset_features(ca_ohlcv)
                                ca_features = ca_features.dropna()
                                if len(ca_features) < 100:
                                    continue
                                ca_detector = RegimeDetector(
                                    n_regimes=CROSS_ASSET_N_REGIMES,
                                    n_restarts=3, n_iter=50,
                                )
                                ca_detector.fit(ca_features, feature_cols=["returns", "range", "volume_vol"])
                                ca_current = ca_detector.get_current_regime(ca_features)
                                cross_results[ca_ticker] = ca_current
                            st.session_state["cross_results"] = cross_results
                        except Exception as e:
                            st.error(f"Cross-asset failed: {e}")

                cross_results = st.session_state.get("cross_results", {})
                if cross_results:
                    ca_cols = st.columns(len(cross_results) + 1)

                    # SPY from main model
                    with ca_cols[0]:
                        st.markdown(f"""
                        <div style="background:#1a1d24;border-radius:8px;padding:12px;text-align:center;
                                    border-left:3px solid {signal_color};">
                            <div style="color:#8b949e;font-size:12px;">SPY (Hourly)</div>
                            <div style="color:{signal_color};font-size:16px;font-weight:bold;">{current['label']}</div>
                            <div style="color:#8b949e;font-size:11px;">{current['confidence']:.0%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    for i, (ca_ticker, ca_state) in enumerate(cross_results.items()):
                        with ca_cols[i + 1]:
                            ca_signal = ca_state.get("signal", "neutral")
                            ca_color = "#2ecc71" if ca_signal == "bullish" else (
                                "#e74c3c" if ca_signal == "bearish" else "#f39c12"
                            )
                            st.markdown(f"""
                            <div style="background:#1a1d24;border-radius:8px;padding:12px;text-align:center;
                                        border-left:3px solid {ca_color};">
                                <div style="color:#8b949e;font-size:12px;">{ca_ticker} (Daily)</div>
                                <div style="color:{ca_color};font-size:16px;font-weight:bold;">{ca_state.get('label', '?')}</div>
                                <div style="color:#8b949e;font-size:11px;">{ca_state.get('confidence', 0):.0%}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Alignment indicator
                    all_signals = [current["signal"]] + [s.get("signal", "neutral") for s in cross_results.values()]
                    bullish_count = sum(1 for s in all_signals if s == "bullish")
                    bearish_count = sum(1 for s in all_signals if s == "bearish")
                    total_assets = len(all_signals)

                    if bullish_count >= total_assets - 1:
                        alignment_text = f"{bullish_count}/{total_assets} Bullish — ALIGNED"
                        alignment_color = "#2ecc71"
                    elif bearish_count >= total_assets - 1:
                        alignment_text = f"{bearish_count}/{total_assets} Bearish — ALIGNED"
                        alignment_color = "#e74c3c"
                    else:
                        alignment_text = f"{bullish_count}B / {bearish_count}R / {total_assets - bullish_count - bearish_count}N — DIVERGING"
                        alignment_color = "#f39c12"

                    st.markdown(f"""
                    <div style="background:#1a1d24;border-radius:8px;padding:12px;text-align:center;
                                border:2px solid {alignment_color};margin-top:12px;">
                        <div style="color:{alignment_color};font-size:18px;font-weight:bold;">{alignment_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Internals summary
                    st.markdown("#### Internals Summary")
                    bullish_internals = []
                    bearish_internals = []

                    if vix_data is not None and not vix_data.empty:
                        if vix_data.iloc[-1]["vix_contango"]:
                            bullish_internals.append("VIX Contango")
                        else:
                            bearish_internals.append("VIX Backwardation")

                    if breadth_data is not None and not breadth_data.empty:
                        slope = breadth_data["breadth_slope"].dropna()
                        if not slope.empty and slope.iloc[-1] > 0:
                            bullish_internals.append("Breadth Broadening")
                        else:
                            bearish_internals.append("Breadth Narrowing")

                    if dxy_data is not None and not dxy_data.empty:
                        wk_chg = dxy_data["dxy_weekly_chg"].dropna()
                        if not wk_chg.empty and abs(wk_chg.iloc[-1]) < 1.0:
                            bullish_internals.append("Dollar Stable")
                        else:
                            bearish_internals.append("Dollar Surging")

                    int_cols = st.columns(2)
                    with int_cols[0]:
                        st.markdown(f"**Bullish ({len(bullish_internals)})**: " + ", ".join(bullish_internals) if bullish_internals else "**Bullish (0)**")
                    with int_cols[1]:
                        st.markdown(f"**Bearish ({len(bearish_internals)})**: " + ", ".join(bearish_internals) if bearish_internals else "**Bearish (0)**")
            else:
                st.warning("No market internals data available. Check data downloads.")

    # === TAB: Strategy Integration ===
    strategy_tab_idx = len(tab_names) - 1
    with tabs[strategy_tab_idx]:
        st.subheader("Strategy Integration — Regime Filter Impact")

        # Load pre-computed comparison results
        import json as _json
        results_path = PROCESSED_DATA_DIR / "regime_comparison.json"

        if results_path.exists():
            with open(results_path) as _f:
                comp_results = _json.load(_f)

            st.caption(f"Last computed: {comp_results.get('generated_at', 'unknown')}")

            # ── AMT-TEMA v8 ──
            if "amt_tema" in comp_results:
                amt = comp_results["amt_tema"]
                b = amt["baseline"]
                f = amt["filtered"]

                st.markdown("### AMT-TEMA v8 (ES 5m, Short-Only)")
                st.markdown(f"**Blocked regimes**: {', '.join(amt['blocked_regimes'])}")

                # Comparison metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trades", f"{f['trades']}", f"{f['trades'] - b['trades']:+d}")
                c2.metric("Win Rate", f"{f['win_rate']:.1f}%", f"{f['win_rate'] - b['win_rate']:+.1f}%")
                c3.metric("Profit Factor", f"{f['profit_factor']:.2f}", f"{f['profit_factor'] - b['profit_factor']:+.2f}")
                c4.metric("Net P&L", f"${f['net_pnl']:,.0f}", f"${f['net_pnl'] - b['net_pnl']:+,.0f}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Sharpe", f"{f['sharpe']:.2f}", f"{f['sharpe'] - b['sharpe']:+.2f}")
                c6.metric("Max DD", f"${f['max_drawdown']:,.0f}", f"${f['max_drawdown'] - b['max_drawdown']:+,.0f}")
                c7.metric("Avg Trade", f"${f['avg_trade']:,.0f}", f"${f['avg_trade'] - b['avg_trade']:+,.0f}")
                pct_removed = (1 - f['trades'] / b['trades']) * 100 if b['trades'] > 0 else 0
                c8.metric("Trades Removed", f"{pct_removed:.1f}%")

                # Regime breakdown chart
                if "regime_breakdown" in amt:
                    bd = amt["regime_breakdown"]
                    regime_order = ["Crash (Panic)", "Bear Trend", "Distribution",
                                    "Accumulation (Chop)", "Recovery", "Bull Run (Trend)",
                                    "Strong Bull (Trend)"]
                    colors_map = {
                        "Crash (Panic)": "#8e44ad", "Bear Trend": "#e74c3c",
                        "Distribution": "#e67e22", "Accumulation (Chop)": "#f39c12",
                        "Recovery": "#3498db", "Bull Run (Trend)": "#2ecc71",
                        "Strong Bull (Trend)": "#27ae60", "No Data (before model)": "#555555",
                    }

                    labels = [r for r in regime_order if r in bd]
                    if "No Data (before model)" in bd:
                        labels.append("No Data (before model)")

                    fig_amt = make_subplots(rows=1, cols=2,
                                            subplot_titles=("Trade Count by Regime", "P&L by Regime"))

                    fig_amt.add_trace(go.Bar(
                        x=labels,
                        y=[bd[r]["count"] for r in labels],
                        marker_color=[colors_map.get(r, "#888") for r in labels],
                        name="Trades",
                        text=[f"{bd[r]['win_rate']:.0f}% WR" for r in labels],
                        textposition="outside",
                    ), row=1, col=1)

                    pnl_vals = [bd[r]["pnl"] for r in labels]
                    fig_amt.add_trace(go.Bar(
                        x=labels,
                        y=pnl_vals,
                        marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in pnl_vals],
                        name="P&L",
                        text=[f"${v:,.0f}" for v in pnl_vals],
                        textposition="outside",
                    ), row=1, col=2)

                    fig_amt.update_layout(
                        height=400, showlegend=False,
                        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d24",
                        font_color="#c9d1d9",
                    )
                    fig_amt.update_xaxes(tickangle=30, tickfont_size=10)
                    st.plotly_chart(fig_amt, use_container_width=True)

                st.divider()

            # ── Home Run ──
            if "home_run" in comp_results:
                hr = comp_results["home_run"]
                b = hr["baseline"]
                f = hr["filtered"]

                st.markdown("### Home Run (SPY 0DTE Puts, 30m Range >= 20)")
                st.markdown(f"**Allowed regimes**: {', '.join(hr['allowed_regimes'])}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trades", f"{f['trades']}", f"{f['trades'] - b['trades']:+d}")
                c2.metric("Win Rate", f"{f['win_rate']:.1f}%", f"{f['win_rate'] - b['win_rate']:+.1f}%")
                c3.metric("Profit Factor", f"{f['profit_factor']:.2f}", f"{f['profit_factor'] - b['profit_factor']:+.2f}")
                c4.metric("Net P&L", f"${f['net_pnl']:,.0f}", f"${f['net_pnl'] - b['net_pnl']:+,.0f}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Sharpe", f"{f['sharpe']:.2f}", f"{f['sharpe'] - b['sharpe']:+.2f}")
                c6.metric("Max DD", f"${f['max_drawdown']:,.0f}", f"${f['max_drawdown'] - b['max_drawdown']:+,.0f}")
                c7.metric("Avg Trade", f"${f['avg_trade']:,.0f}", f"${f['avg_trade'] - b['avg_trade']:+,.0f}")
                pct_removed = (1 - f['trades'] / b['trades']) * 100 if b['trades'] > 0 else 0
                c8.metric("Trades Removed", f"{pct_removed:.1f}%")

                # Regime breakdown chart
                if "regime_breakdown" in hr:
                    bd = hr["regime_breakdown"]
                    regime_order = ["Crash (Panic)", "Bear Trend", "Distribution",
                                    "Accumulation (Chop)", "Recovery", "Bull Run (Trend)",
                                    "Strong Bull (Trend)"]
                    colors_map = {
                        "Crash (Panic)": "#8e44ad", "Bear Trend": "#e74c3c",
                        "Distribution": "#e67e22", "Accumulation (Chop)": "#f39c12",
                        "Recovery": "#3498db", "Bull Run (Trend)": "#2ecc71",
                        "Strong Bull (Trend)": "#27ae60", "No Data (before model)": "#555555",
                    }

                    labels = [r for r in regime_order if r in bd]
                    if "No Data (before model)" in bd:
                        labels.append("No Data (before model)")

                    fig_hr = make_subplots(rows=1, cols=2,
                                           subplot_titles=("Trade Count by Regime", "P&L by Regime"))

                    fig_hr.add_trace(go.Bar(
                        x=labels,
                        y=[bd[r]["count"] for r in labels],
                        marker_color=[colors_map.get(r, "#888") for r in labels],
                        name="Trades",
                        text=[f"{bd[r]['win_rate']:.0f}% WR" for r in labels],
                        textposition="outside",
                    ), row=1, col=1)

                    pnl_vals = [bd[r]["pnl"] for r in labels]
                    fig_hr.add_trace(go.Bar(
                        x=labels,
                        y=pnl_vals,
                        marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in pnl_vals],
                        name="P&L",
                        text=[f"${v:,.0f}" for v in pnl_vals],
                        textposition="outside",
                    ), row=1, col=2)

                    fig_hr.update_layout(
                        height=400, showlegend=False,
                        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d24",
                        font_color="#c9d1d9",
                    )
                    fig_hr.update_xaxes(tickangle=30, tickfont_size=10)
                    st.plotly_chart(fig_hr, use_container_width=True)

            # Key insights
            st.markdown("### Key Insights")
            st.markdown("""
            - **AMT-TEMA**: Blocking Bull Run removes pure losers (18% WR, -$226/trade). Net +$5K improvement.
            - **Home Run**: Bearish-only filter doubles avg trade ($928 -> $1,685) and nearly doubles PF (3.6 -> 7.0).
            - **Crash regime** is the goldmine for both strategies (75%+ WR, highest avg P&L).
            - **Accumulation (Chop)** is a marginal loser for both — consider blocking it too.
            """)

            st.info("Re-run `python scripts/regime_comparison.py` to refresh these results.")

        else:
            st.warning("No comparison results found. Run the comparison script first:")
            st.code("cd ~/projects/ai-playground/medallion-2.0\n"
                     "source .venv/bin/activate\n"
                     "python scripts/regime_comparison.py", language="bash")

else:
    # Landing page
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h2 style="color: #8b949e;">Welcome to Medallion 2.0</h2>
        <p style="color: #6c757d; font-size: 18px;">
            Hidden Markov Model market regime detection<br>
            with 13-confirmation voting system
        </p>
        <p style="color: #6c757d;">
            Click <b>Load Saved</b> to load a previously fitted model,<br>
            or <b>Refit Model</b> to train a fresh model from data.
        </p>
        <div style="margin-top: 40px; padding: 20px; background: #1a1d24; border-radius: 10px; display: inline-block;">
            <p style="color: #8b949e; margin: 0;">Architecture</p>
            <code style="color: #2ecc71;">
            7-State HMM → Regime Detection → 13 Confirmations → Quality Analysis → Strategy Filter
            </code>
        </div>
    </div>
    """, unsafe_allow_html=True)
