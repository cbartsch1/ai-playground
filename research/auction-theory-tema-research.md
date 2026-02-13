# Auction Market Theory + TEMA: Research for ES 5m Strategy
**Date:** Feb 12, 2026 | **Compiled by:** Claude Opus 4.6
**Purpose:** Foundation for HYBRID v8+ — replacing structure break entry with auction theory + TEMA

---

## 1. AUCTION MARKET THEORY — CORE CONCEPTS FOR ES INTRADAY

### 1.1 Value Area (VAH, VAL, POC)

The **Value Area** is the price range where approximately **70% of trading volume** occurred in a given session. It represents one standard deviation of volume distribution and defines "fair value."

**Components:**
- **POC (Point of Control):** The single price level with the highest traded volume. The market's "fairest" price.
- **VAH (Value Area High):** Upper boundary of the 70% volume zone.
- **VAL (Value Area Low):** Lower boundary of the 70% volume zone.

**Calculation method:**
1. Build a volume histogram at each price level across the session.
2. Identify the price bin with the highest volume — that is the POC.
3. Starting from the POC, alternately add the next highest-volume bins above and below until 70% of total session volume is captured.
4. The highest bin included = VAH; the lowest = VAL.

**How they're used on ES:**
- **POC acts as a magnet.** Price tends to gravitate toward it during balanced/rotational days.
- **VAH/VAL act as support/resistance.** Responsive traders fade moves into VAH (sell) and VAL (buy). Initiative traders look for breakouts through them.
- **Previous day's VAH/VAL/POC** are key reference levels for the current session.

### 1.2 Initial Balance (IB)

The **Initial Balance** is the price range established during the **first hour of RTH** — for ES, that is **9:30–10:30 AM ET**.

**Why it matters:**
- The IB reflects the activity of **short-term/local traders** who establish the early range.
- Whether **longer-timeframe (institutional) traders** break the IB determines the day's character.
- A **wide IB** (relative to recent ATR) → trend days are less likely.
- A **narrow IB** → potential for large range extension when institutions enter.

**Key measurements:**
- **IB High / IB Low / IB Range** (IBR = IB High - IB Low)
- **IB Extension:** When price breaks above IB High or below IB Low after 10:30 AM.
- Typical ES IB range: ~10–20 points on average; <10 = narrow (trend potential); >20 = wide (rotation likely).

### 1.3 Day Types

| Day Type | IB Extension | Characteristics | Frequency |
|---|---|---|---|
| **Normal Day** | IB NOT broken | Wide IB; price stays in range all day | ~15% |
| **Normal Variation** | Broken ONE direction, <1x IB range | Moderate extension; directional bias | ~25% |
| **Trend Day** | Broken 2+ times; extension >2x IB range | One-sided flow; sharp moves | ~5-10% |
| **Double Distribution** | Broken; new value area forms away from IB | Two distinct balance areas | ~10% |
| **Neutral Day** | Broken BOTH directions; close inside IB | Both sides tested | ~20% |
| **Neutral Extreme** | Broken BOTH directions; close outside IB | One side wins decisively | ~10% |
| **Non-Trend Day** | Very tight IB; minimal extension | Low participation | ~10% |

**Key insight:** Non-trending/rotational days = ~80% of all sessions. A strategy that ONLY trades trends will miss most opportunities. Best approach: handle both balance (mean-reversion) AND imbalance (trend-following).

### 1.4 The 80% Rule

If price opens outside yesterday's Value Area, then moves back inside it and is accepted (two consecutive 30-min candles close inside VA), there is an **80% probability** price will travel to the opposite side of the VA.

**Mechanics:**
1. Yesterday's VAH/VAL identified
2. Today's open is OUTSIDE yesterday's VA
3. Price re-enters the VA
4. **Confirmation:** Two consecutive 30-min bars (or ~6 consecutive 5-min bars) close inside VA
5. **Entry:** On confirmation
6. **Target:** Opposite VA edge
7. **Stop:** Just outside the VA boundary where price re-entered

### 1.5 Open Types

| Open Type | Description | Implication |
|---|---|---|
| **Open-Drive (OD)** | Aggressive directional move from open; opening tick near day's extreme | Strong conviction; trade WITH direction |
| **Open-Test-Drive (OTD)** | Opens, tests a key level, then drives opposite direction | Institutional probing; trade the drive |
| **Open-Rejection-Reverse (ORR)** | Opens one direction, gets sharply rejected from HTF level | Reversal signal |
| **Open-Auction (OA)** | Balanced rotation near open; stays inside prior VA | No conviction; expect rotational day |

### 1.6 Key Trade Setups

**IB Breakout:**
- Wait for IB to form (first 60 min)
- Enter on break above IB High (long) or below IB Low (short)
- Confirmation: 5-min close beyond IB level
- Target: 1x to 1.5x IB range as measured move
- Stop: IB midpoint or opposite edge

**Value Area Fade (Responsive):**
- Price moves to prior day's VAH or VAL
- Look for rejection (wick, absorption)
- Enter fading the move, target POC then opposite VA edge
- Stop beyond the excess/wick

**Failed Auction:**
- IB is broken but price fails to sustain for 30 min, pulls back inside
- Trade reversal back toward opposite IB edge or POC
- The failed auction zone is typically revisited within 5 days

**Poor High/Poor Low:**
- Flat, untested extremes at session high or low (no tail/excess)
- "Unfinished business" — auction didn't complete
- Price will likely revisit and either break through or form proper excess

---

## 2. TEMA (TRIPLE EXPONENTIAL MOVING AVERAGE)

### 2.1 How TEMA Differs from EMA

**Formula:**
```
EMA1 = EMA(price, period)
EMA2 = EMA(EMA1, period)
EMA3 = EMA(EMA2, period)
TEMA = (3 * EMA1) - (3 * EMA2) + EMA3
```

**Key properties:**
- Hugs price much more closely than EMA or DEMA
- Generates crossover signals significantly earlier
- More responsive to sharp moves (like ES breakouts)
- BUT produces more false signals in choppy/ranging markets
- A 20-period TEMA reacts almost as fast as a 9-period EMA but is smoother

### 2.2 Common TEMA Strategies

**A. TEMA Crossover (Fast/Slow)**
- Fast TEMA (9 or 13) crosses above Slow TEMA (21 or 34): Long
- Crosses below: Short
- Earlier signals than EMA crossover but needs filters for whipsaws

**B. TEMA Slope/Momentum**
- `tema_slope = tema - tema[1]`
- Positive and increasing slope = strengthening uptrend
- Slope crossing zero = potential trend change
- Use slope magnitude as filter (skip flat/choppy conditions)

**C. Crossing TEMA Strategy (documented by WH SelfInvest)**
- Fast TEMA: 56 quarter-hours (~1 trading day on 15-min)
- Slow TEMA: 5x56 (5 trading days)
- Always-in-market with trailing stops

### 2.3 Optimal TEMA Periods for ES 5-Minute Chart

| Purpose | Fast TEMA | Slow TEMA | Notes |
|---|---|---|---|
| Scalping | 5–9 | 13–21 | Very responsive; needs strong filters |
| Day Trading (recommended) | 9–13 | 21–34 | Good speed vs. noise rejection |
| Swing/Position filter | 21–34 | 55–89 | Trend direction filter only |
| Slope/Momentum gauge | 13–21 | n/a | Single TEMA, measure slope |

**Recommendation:** Start with TEMA(9)/TEMA(21) crossover pair. Use TEMA(55) as trend filter.

---

## 3. PINE SCRIPT V6 IMPLEMENTATION NOTES

### Fully Implementable

| Concept | How |
|---|---|
| TEMA calculation | `ta.ema()` applied three times + formula |
| TEMA crossover | `ta.crossover()` / `ta.crossunder()` |
| TEMA slope | `tema - tema[1]` |
| Initial Balance (IB) | Track session high/low for first 60 min |
| IB Breakout detection | Compare price to stored IB High/Low after 10:30 |
| Previous Day High/Low/Close | `request.security()` with daily timeframe |
| Session Volume Profile | `volume.profile_session()` in v6 — returns POC, VAH, VAL |
| ATR-based stops/targets | `ta.atr()` |
| Session time filtering | `time(timeframe, session_string)` |

### Partially Implementable

| Concept | Workaround |
|---|---|
| Previous session POC/VAH/VAL | Store `profile.poc_price`, `profile.value_area_high`, `profile.value_area_low` in `var` variables at session change |
| Open Type classification | Time-based logic + stored prior VA levels |
| Failed Auction detection | Timer logic with bar counting after IB break |
| Day type reclassification | State machine pattern with `var` variables |

### NOT Implementable in Pine Script

- Order flow / Delta (no tick-level bid/ask data)
- TPO letters (approximate with volume profile instead)
- DOM / Level 2 data
- Machine learning classification

---

## 4. RECOMMENDED STRATEGY SETUPS (AMT + TEMA Combined)

### Setup 1: IB Breakout + TEMA Confirmation (Trend Setup)
- **Pre-condition:** TEMA(9) > TEMA(21) for longs; TEMA(9) < TEMA(21) for shorts
- **Optional:** TEMA(55) slope positive for longs, negative for shorts
- **Entry:** Price breaks above IB High (long) or below IB Low (short) after 10:30 AM ET
- **Confirmation:** 5-min candle closes beyond IB level
- **Stop:** IB midpoint or 1.5x ATR(14) inside IB
- **Target:** 1x IB range as measured move, or trail with TEMA(21)
- **Session:** 10:30 AM – 3:00 PM ET

### Setup 2: Value Area Fade + TEMA Slope (Mean-Reversion Setup)
- **Pre-condition:** Price reaches prior day's VAH (short) or VAL (long)
- **TEMA filter:** TEMA(13) slope is decelerating (slope < slope[3])
- **Entry:** On reversal candle at VA edge
- **Stop:** Beyond VA edge by ATR(14) * 0.5
- **Target 1:** Prior day's POC | **Target 2:** Opposite VA edge
- **Session:** 10:00 AM – 2:00 PM ET

### Setup 3: 80% Rule + TEMA Direction (High-Probability Setup)
- **Pre-condition:** Today's open is outside yesterday's VA
- **Trigger:** Price re-enters VA and 6 consecutive 5-min bars close inside it
- **TEMA filter:** TEMA(9) pointing toward opposite VA edge
- **Entry:** After confirmation
- **Stop:** Just outside VA boundary where price re-entered
- **Target:** Opposite VA edge

### Setup 4: TEMA Trend + Auction Context (Adaptive Setup)
- **Signal:** TEMA(9) crosses TEMA(21)
- **Context modifiers:**
  - Inside prior VA: Reduce size 50% (market in balance)
  - Outside prior VA (above VAH for longs, below VAL for shorts): Full size
  - At POC: Skip signal (too much gravitational pull)
- **Stop:** ATR-based, beyond nearest VA edge or POC
- **Target:** Trail via TEMA(21)

---

## 5. RISK MANAGEMENT FRAMEWORK

**Stop Placement Hierarchy:**
1. **Best:** Beyond auction structure (IB edge, VA edge, POC)
2. **Good:** ATR-based (1–2x ATR(14) from entry)
3. **Acceptable:** Fixed tick count (8–12 ticks on ES)

**Target Framework:**
- Minimum 2:1 reward-to-risk
- Use auction levels as targets (POC, opposite VA edge, IB measured move)
- Trail with TEMA(21) after reaching 1R profit

**Daily Risk Limits:**
- Max daily loss: 2–3R
- Max trades per day: 4–6
- No trading in first 5 minutes of RTH

---

## 6. BUILD ORDER (Recommended)

1. **TEMA Crossover Engine** — TEMA(9)/TEMA(21) + TEMA(55) trend filter
2. **Initial Balance Tracker** — IB High/Low/Range for first 60 min of RTH
3. **Previous Day VA Levels** — Store prior session POC, VAH, VAL
4. **IB Breakout + TEMA Setup** (Setup 1) — Trend-day strategy
5. **Value Area Fade + TEMA Setup** (Setup 2) — Balanced-day strategy
6. **80% Rule Module** (Setup 3) — High-probability overlay
7. **Day Type Classifier** — Compare IB range to 14-day average; track extensions

---

## Sources
- Kimatix Trading — AMT Primer for Day Traders
- FTMO — Market Profile, Volume Profile and AMT
- TRADEPRO Academy — Full Guide to Auction Market Theory
- PipSafe — The Value Area 80% Rule
- MyPivots — 80% Rule Definition
- Trader Dale — Failed Auctions Explained
- Axia Futures — Poor High Breakout & Failed Auction Strategies
- Tatanka Futures — Automated Auction Theory for ES/NQ
- StockCharts — TEMA ChartSchool
- TradingSim — TEMA Day Trading
- WH SelfInvest — Crossing TEMA Strategy
- Quantified Strategies — TEMA Backtest and Evaluation
- TradingCode — TEMA Pine Script Implementation
