#!/usr/bin/env python3
"""
Medallion 2.0 — Regime Change Monitor

Checks current regime vs last saved regime.
Sends macOS desktop notification and logs changes.

Usage:
    python scripts/regime_monitor.py              # check and notify
    python scripts/regime_monitor.py --daemon 300  # check every 5 min

Can be run via cron/launchd:
    */30 9-16 * * 1-5 cd ~/projects/ai-playground/medallion-2.0 && .venv/bin/python scripts/regime_monitor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import subprocess
import time
import numpy as np
from datetime import datetime

from config.settings import HMM_FEATURES, DEFAULT_N_REGIMES, MODELS_DIR
from models.hmm_regime import RegimeDetector
from data.data_loader import load_data
from models.regime_api import RegimeFilter

LOG_DIR = Path(__file__).parent.parent / "logs"
STATE_FILE = LOG_DIR / "last_regime.json"
LOG_FILE = LOG_DIR / "regime_changes.log"


def get_current_regime(n_regimes: int = DEFAULT_N_REGIMES) -> dict:
    """Load model, get fresh data, return current regime + transition forecast."""
    # Try loading saved model first
    detector = RegimeDetector.load_latest(n_regimes=n_regimes)

    if detector is None:
        print("No saved model found. Fitting fresh model...")
        _, hmm_features, _ = load_data(cache=True)
        detector = RegimeDetector(n_regimes=n_regimes, n_restarts=5, n_iter=100)
        detector.fit(hmm_features, feature_cols=HMM_FEATURES)
        detector.save_latest()
    else:
        print(f"Loaded saved {n_regimes}-state model.")

    # Get fresh data for prediction
    _, hmm_features, _ = load_data(cache=False)
    current = detector.get_current_regime(hmm_features)

    # Add transition forecast
    result = detector.predict(hmm_features)
    last_row = result.dropna(subset=["confidence"]).iloc[-1]
    current_probs = np.array([last_row[f"prob_{i}"] for i in range(n_regimes)])
    forecast = detector.forecast_transitions(current_probs)
    current["transition_forecast"] = forecast

    return current


def load_last_regime() -> dict | None:
    """Load last known regime from state file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def save_current_regime(regime: dict):
    """Save current regime to state file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(regime, f, indent=2, default=str)


def log_change(alert: dict):
    """Append regime change to log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{alert['severity'].upper()}] {alert['message']}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    print(f"Logged: {line.strip()}")


def send_notification(title: str, message: str, severity: str = "info"):
    """Send macOS desktop notification via osascript."""
    sound = "Basso" if severity == "critical" else ("Purr" if severity == "warning" else "Pop")
    script = f'display notification "{message}" with title "{title}" sound name "{sound}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )
        print(f"Notification sent: {title} — {message}")
    except Exception as e:
        print(f"Notification failed: {e}")


def check_regime(n_regimes: int = DEFAULT_N_REGIMES):
    """Main check: compare current regime to last saved, notify if changed."""
    print(f"\n--- Regime Check @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    current = get_current_regime(n_regimes)
    print(f"Current: {current['label']} ({current['signal']}) — {current['confidence']:.1%} confidence")

    last = load_last_regime()

    if last is None:
        print("First run — saving initial regime state.")
        save_current_regime(current)
        send_notification(
            "Medallion 2.0 — Monitor Started",
            f"Regime: {current['label']} ({current['signal']}, {current['confidence']:.0%})",
            "info",
        )
        return

    # Check for change
    rf = RegimeFilter.__new__(RegimeFilter)
    rf.bullish_regimes = {"Bull Run (Trend)", "Strong Bull (Trend)", "Recovery"}
    rf.bearish_regimes = {"Bear Trend", "Crash (Panic)", "Distribution"}

    # Use check_regime_change from RegimeFilter
    alert = rf.check_regime_change(last.get("label"), current["label"])

    if alert:
        log_change(alert)
        send_notification(
            f"Medallion 2.0 — Regime {alert['severity'].upper()}",
            alert["message"],
            alert["severity"],
        )
    else:
        print(f"No change (still {current['label']}).")

    # Print transition forecast
    forecast = current.get("transition_forecast")
    if forecast:
        p_change = forecast.get("p_change", {})
        alert_level = forecast.get("alert_level", "?")
        ref_h = 6 if 6 in p_change else (min(p_change.keys()) if p_change else None)
        if ref_h:
            print(f"Transition risk: P(change in {ref_h}h) = {p_change[ref_h]:.1%} — {alert_level.upper()}")
        print(f"Most likely next: {forecast.get('most_likely_next', '?')}")
        print(f"P(bearish): {forecast.get('p_bearish', 0):.1%}")

    save_current_regime(current)


def main():
    parser = argparse.ArgumentParser(description="Regime change monitor")
    parser.add_argument("--n-regimes", type=int, default=DEFAULT_N_REGIMES)
    parser.add_argument("--daemon", type=int, default=0, help="Run every N seconds (0 = once)")
    args = parser.parse_args()

    if args.daemon > 0:
        print(f"Running daemon mode, checking every {args.daemon}s. Ctrl+C to stop.")
        while True:
            try:
                check_regime(args.n_regimes)
                time.sleep(args.daemon)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        check_regime(args.n_regimes)


if __name__ == "__main__":
    main()
