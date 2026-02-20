# ============================================================================
# AMT-TEMA v6 — thinkorSwim Study
# Auction Market Theory + Triple EMA
# Converted from Pine Script v6 | For ES (/ES) 5m Chart
#
# SETUPS:
#   1. IB Breakout + TEMA — trend day entry after IB range break
#   2. Value Area Fade + TEMA Slope — mean reversion at prev-day VA edges
#   3. 80% Rule + TEMA (default OFF) — open outside VA, traverse opposite edge
#
# USAGE:
#   1. In thinkorSwim: Studies > Edit Studies > Create > thinkScript Editor
#   2. Paste this code, save as "AMT_TEMA_v6"
#   3. Apply to /ES 5-minute chart
#   4. Set chart timezone to US Eastern for correct time display
#
# NOTE: All time inputs are in US Eastern Time. The study uses
#       RegularTradingStart() for timezone-safe RTH detection.
# ============================================================================

declare upper;

# ============================================================================
# INPUTS
# ============================================================================

# --- TEMA ---
input temaFastLen = 9;       # [3-21]
input temaSlowLen = 21;      # [13-55]
input temaTrendLen = 55;     # [34-89]

# --- Session (Eastern Time HHMM) ---
input tradeStartTime = 1035; # [1030-1100]
input tradeEndTime = 1500;   # [1430-1555]
input flattenTime = 1555;    # [1530-1600]

# --- Day Type ---
input ibAvgLen = 20;         # [5-50]
input ibNarrowRatio = 0.8;   # [0.5-1.0]
input ibWideRatio = 1.2;     # [1.0-2.0]
input useDayType = yes;

# --- Value Area ---
input vaStdevMult = 1.0;     # [0.5-2.0]

# --- Risk ---
input atrLen = 14;           # [7-21]
input cooldownBars = 2;      # [1-5]

# --- Volatility ---
input useVolFilter = yes;
input atrAvgLen = 50;        # [20-100]
input volLowRatio = 0.5;     # [0.2-0.8]
input volHighRatio = 2.0;    # [1.5-3.0]

# --- IB Breakout ---
input useIBBreak = yes;
input useTrendFilter = yes;
input minIBRange = 8.0;      # [1-15] points
input maxIBRange = 25.0;     # [10-50] points
input ibStopType = {default IB_Mid, IB_Edge, ATR};

# --- VA Fade ---
input useVAFade = yes;
input vaBuffer = 4.0;        # [0-8] ticks
input vaStopMult = 0.5;      # [0.3-2.0] ATR mult
input vaMinRR = 0.5;         # [0.5-3.0]

# --- 80% Rule ---
input useEighty = no;
input eightyConfBars = 6;    # [3-10]
input eightyStopBuf = 0.5;   # [0.25-1.0] ATR mult

# --- Visuals ---
input showDashboard = yes;
input showLevels = yes;
input showTEMA = yes;

# ============================================================================
# TEMA ENGINE
# ============================================================================

def e1f = ExpAverage(close, temaFastLen);
def e2f = ExpAverage(e1f, temaFastLen);
def e3f = ExpAverage(e2f, temaFastLen);
def temaFast = 3 * e1f - 3 * e2f + e3f;

def e1s = ExpAverage(close, temaSlowLen);
def e2s = ExpAverage(e1s, temaSlowLen);
def e3s = ExpAverage(e2s, temaSlowLen);
def temaSlow = 3 * e1s - 3 * e2s + e3s;

def e1t = ExpAverage(close, temaTrendLen);
def e2t = ExpAverage(e1t, temaTrendLen);
def e3t = ExpAverage(e2t, temaTrendLen);
def temaTrend = 3 * e1t - 3 * e2t + e3t;

def temaBullish = temaFast > temaSlow;
def temaBearish = temaFast < temaSlow;
def temaSlope = temaFast - temaFast[3];
def slopeRising = temaSlope > temaSlope[3];
def slopeFalling = temaSlope < temaSlope[3];
def trendUp = close > temaTrend;
def trendDown = close < temaTrend;

# ============================================================================
# SESSION & TIME
# Uses RegularTradingStart() for timezone-safe RTH/IB detection.
# Trading window uses ms offset from RTH start so ET inputs work universally.
# ============================================================================

def rthStartMs = RegularTradingStart(GetYYYYMMDD());
def rthEndMs = RegularTradingEnd(GetYYYYMMDD());
def barMs = GetTime();
def msSinceRTH = barMs - rthStartMs;

def isRTH = barMs >= rthStartMs and barMs < rthEndMs;
def isIBPeriod = barMs >= rthStartMs and barMs < rthStartMs + 3600000;
def newRTH = isRTH and !isRTH[1];

# Convert HHMM ET inputs to ms offset from RTH start (9:30 ET = baseline)
def _tsHr = RoundDown(tradeStartTime / 100, 0);
def _tsMin = tradeStartTime - _tsHr * 100;
def tradeStartOff = ((_tsHr - 9) * 60 + (_tsMin - 30)) * 60000;

def _teHr = RoundDown(tradeEndTime / 100, 0);
def _teMin = tradeEndTime - _teHr * 100;
def tradeEndOff = ((_teHr - 9) * 60 + (_teMin - 30)) * 60000;

def _ftHr = RoundDown(flattenTime / 100, 0);
def _ftMin = flattenTime - _ftHr * 100;
def flattenOff = ((_ftHr - 9) * 60 + (_ftMin - 30)) * 60000;

def isTradingWindow = msSinceRTH >= tradeStartOff and msSinceRTH < tradeEndOff;
def isFlattenTime = msSinceRTH >= flattenOff;

# ============================================================================
# INITIAL BALANCE — First 60 min of RTH (9:30-10:30 ET)
# ============================================================================

rec ibHigh = if newRTH then high
             else if isIBPeriod then Max(if IsNaN(ibHigh[1]) then high else ibHigh[1], high)
             else if IsNaN(ibHigh[1]) then high
             else ibHigh[1];

rec ibLow = if newRTH then low
            else if isIBPeriod then Min(if IsNaN(ibLow[1]) then low else ibLow[1], low)
            else if IsNaN(ibLow[1]) then low
            else ibLow[1];

rec ibDone = if newRTH then no
             else if !ibDone[1] and isRTH and !isIBPeriod then yes
             else ibDone[1];

def ibRange = ibHigh - ibLow;
def ibMid = (ibHigh + ibLow) / 2;
def ibValid = ibRange >= minIBRange and ibRange <= maxIBRange;

# ============================================================================
# IB WIDTH DAY-TYPE CLASSIFICATION
# ============================================================================

def ibAlpha = 2.0 / (ibAvgLen + 1.0);

rec ibRangeAvg = if ibDone and !ibDone[1] then
                     if IsNaN(ibRangeAvg[1]) then ibRange
                     else ibRangeAvg[1] * (1.0 - ibAlpha) + ibRange * ibAlpha
                 else ibRangeAvg[1];

def ibRatio = if !IsNaN(ibRangeAvg) and ibRangeAvg > 0
              then ibRange / ibRangeAvg else 1.0;
def isNarrowIB = ibRatio < ibNarrowRatio;
def isWideIB = ibRatio > ibWideRatio;
# dayTypeStr removed — thinkScript def is numeric-only, strings go inline in AddLabel

# ============================================================================
# PREVIOUS DAY VALUE AREA — VWAP-BASED
# Running VWAP + stdev during RTH, stored at next session open
# ============================================================================

def hlc3Val = (high + low + close) / 3;

rec rthVwapSum = if newRTH then 0
                 else if isRTH then rthVwapSum[1] + hlc3Val * volume
                 else rthVwapSum[1];

rec rthVolSum = if newRTH then 0
                else if isRTH then rthVolSum[1] + volume
                else rthVolSum[1];

rec rthBarCnt = if newRTH then 0
                else if isRTH and rthVolSum > 0 then rthBarCnt[1] + 1
                else rthBarCnt[1];

rec rthSqDev = if newRTH then 0
               else if isRTH and rthVolSum > 0 then
                   rthSqDev[1] + Power(close - rthVwapSum / rthVolSum, 2)
               else rthSqDev[1];

# Store previous session levels when new RTH starts
rec prevPOC = if newRTH and rthVolSum[1] > 0
              then rthVwapSum[1] / rthVolSum[1]
              else if IsNaN(prevPOC[1]) then close
              else prevPOC[1];

rec prevVAH = if newRTH and rthVolSum[1] > 0 then
                  rthVwapSum[1] / rthVolSum[1] +
                  (if rthBarCnt[1] > 1
                   then Sqrt(rthSqDev[1] / rthBarCnt[1]) * vaStdevMult
                   else 10)
              else if IsNaN(prevVAH[1]) then close + 10
              else prevVAH[1];

rec prevVAL = if newRTH and rthVolSum[1] > 0 then
                  rthVwapSum[1] / rthVolSum[1] -
                  (if rthBarCnt[1] > 1
                   then Sqrt(rthSqDev[1] / rthBarCnt[1]) * vaStdevMult
                   else 10)
              else if IsNaN(prevVAL[1]) then close - 10
              else prevVAL[1];

def prevPOCVal = if IsNaN(prevPOC) then close else prevPOC;
def prevVAHVal = if IsNaN(prevVAH) then close + 10 else prevVAH;
def prevVALVal = if IsNaN(prevVAL) then close - 10 else prevVAL;

def aboveVA = close > prevVAHVal;
def belowVA = close < prevVALVal;
def insideVA = !aboveVA and !belowVA;

# ============================================================================
# ATR & VOLATILITY
# ============================================================================

def atrVal = ATR(atrLen);
def tick = TickSize();
def atrAvg = Average(atrVal, atrAvgLen);
def volRatio = if atrAvg > 0 then atrVal / atrAvg else 1.0;
def volOK = !useVolFilter or (volRatio >= volLowRatio and volRatio <= volHighRatio);

# ============================================================================
# RTH OPEN TRACKING (for 80% Rule)
# ============================================================================

rec rthOpen = if newRTH then open else if IsNaN(rthOpen[1]) then open else rthOpen[1];

rec openAboveVA = if newRTH then (!IsNaN(prevVAH) and open > prevVAHVal)
                  else if IsNaN(openAboveVA[1]) then no
                  else openAboveVA[1];

rec openBelowVA = if newRTH then (!IsNaN(prevVAL) and open < prevVALVal)
                  else if IsNaN(openBelowVA[1]) then no
                  else openBelowVA[1];

# ============================================================================
# 80% RULE STATE TRACKING
# Tracks market condition independently of position state
# ============================================================================

rec eightyReentered = if newRTH then no
    else if !eightyReentered[1] and isRTH and (openAboveVA or openBelowVA)
            and insideVA then yes
    else eightyReentered[1];

# eightyInsideCount: stop counting once threshold reached (avoids forward ref to eightyConfirmed)
rec eightyInsideCount = if newRTH then 0
    else if eightyInsideCount[1] >= eightyConfBars then eightyInsideCount[1]
    else if isRTH and (openAboveVA or openBelowVA) then
        (if insideVA then eightyInsideCount[1] + 1 else 0)
    else eightyInsideCount[1];

rec eightyConfirmed = if newRTH then no
    else if !eightyConfirmed[1] and eightyInsideCount >= eightyConfBars then yes
    else eightyConfirmed[1];

# ============================================================================
# RAW SIGNAL CONDITIONS (before position/cooldown checks)
# ============================================================================

# Day type gates
def ibDayTypeOK = !useDayType or !isWideIB;
def vaDayTypeOK = !useDayType or !isNarrowIB;

# IB Breakout
def ibCrossUp = ibDone and close crosses above ibHigh;
def ibCrossDown = ibDone and close crosses below ibLow;
def ibTrendOK_L = !useTrendFilter or trendUp;
def ibTrendOK_S = !useTrendFilter or trendDown;

def ibLongRaw = useIBBreak and ibCrossUp and temaBullish and ibTrendOK_L
    and ibValid and ibDayTypeOK and volOK and isTradingWindow;
def ibShortRaw = useIBBreak and ibCrossDown and temaBearish and ibTrendOK_S
    and ibValid and ibDayTypeOK and volOK and isTradingWindow;

# IB Stop/Target
def ibSL_L = if ibStopType == ibStopType.IB_Mid then ibMid
             else if ibStopType == ibStopType.IB_Edge then ibLow
             else close - atrVal * 1.5;
def ibTP_L = close + ibRange;
def ibSL_S = if ibStopType == ibStopType.IB_Mid then ibMid
             else if ibStopType == ibStopType.IB_Edge then ibHigh
             else close + atrVal * 1.5;
def ibTP_S = close - ibRange;

# VA Fade
def bufferPts = vaBuffer * tick;
def vaTouchLow = low <= prevVALVal + bufferPts and close > prevVALVal;
def vaTouchHigh = high >= prevVAHVal - bufferPts and close < prevVAHVal;

def vaSL_L = prevVALVal - atrVal * vaStopMult;
def vaTP_L = prevPOCVal;
def vaSL_S = prevVAHVal + atrVal * vaStopMult;
def vaTP_S = prevPOCVal;

def vaReward_L = AbsValue(vaTP_L - close);
def vaRisk_L = AbsValue(close - vaSL_L);
def vaReward_S = AbsValue(close - vaTP_S);
def vaRisk_S = AbsValue(vaSL_S - close);
def vaRROK_L = vaRisk_L > 0 and vaReward_L / vaRisk_L >= vaMinRR;
def vaRROK_S = vaRisk_S > 0 and vaReward_S / vaRisk_S >= vaMinRR;

def vaLongRaw = useVAFade and vaTouchLow and slopeRising and vaRROK_L
    and vaDayTypeOK and volOK and isTradingWindow;
def vaShortRaw = useVAFade and vaTouchHigh and slopeFalling and vaRROK_S
    and vaDayTypeOK and volOK and isTradingWindow;

# 80% Rule
def eightyLongRaw = useEighty and openBelowVA and eightyConfirmed
    and (temaBullish or slopeRising) and volOK and isTradingWindow;
def eightyShortRaw = useEighty and openAboveVA and eightyConfirmed
    and (temaBearish or slopeFalling) and volOK and isTradingWindow;

def eightySL_L = prevVALVal - atrVal * eightyStopBuf;
def eightyTP_L = prevVAHVal;
def eightySL_S = prevVAHVal + atrVal * eightyStopBuf;
def eightyTP_S = prevVALVal;

# ============================================================================
# POSITION TRACKING
# posDir: 0=flat, 1=IB_L, -1=IB_S, 2=VA_L, -2=VA_S, 3=80_L, -3=80_S
#
# Exit checks are computed INLINE from current-bar levels (no forward refs).
# IB levels (ibMid/ibHigh/ibLow) are constant after IB completes.
# VA levels (prevVAH/prevVAL/prevPOC) are constant within the day.
# IB target approximated as ibHigh+ibRange (entry ~ ibHigh at crossover).
# ============================================================================

rec posDir =
    # --- EXITS: IB Long ---
    if posDir[1] == 1 and (
        (if ibStopType == ibStopType.IB_Mid then low <= ibMid
         else if ibStopType == ibStopType.IB_Edge then low <= ibLow
         else low <= ibHigh - atrVal * 1.5)
        or high >= ibHigh + ibRange
    ) then 0
    # --- EXITS: IB Short ---
    else if posDir[1] == -1 and (
        (if ibStopType == ibStopType.IB_Mid then high >= ibMid
         else if ibStopType == ibStopType.IB_Edge then high >= ibHigh
         else high >= ibLow + atrVal * 1.5)
        or low <= ibLow - ibRange
    ) then 0
    # --- EXITS: VA Long ---
    else if posDir[1] == 2 and (
        low <= prevVALVal - atrVal * vaStopMult
        or high >= prevPOCVal
    ) then 0
    # --- EXITS: VA Short ---
    else if posDir[1] == -2 and (
        high >= prevVAHVal + atrVal * vaStopMult
        or low <= prevPOCVal
    ) then 0
    # --- EXITS: 80% Long ---
    else if posDir[1] == 3 and (
        low <= prevVALVal - atrVal * eightyStopBuf
        or high >= prevVAHVal
    ) then 0
    # --- EXITS: 80% Short ---
    else if posDir[1] == -3 and (
        high >= prevVAHVal + atrVal * eightyStopBuf
        or low <= prevVALVal
    ) then 0
    # --- EXITS: Session flatten ---
    else if posDir[1] != 0 and isFlattenTime then 0
    # --- ENTRIES (require flat + cooldown bars) ---
    else if posDir[1] == 0
            and (cooldownBars < 2 or posDir[2] == 0)
            and (cooldownBars < 3 or posDir[3] == 0)
            and (cooldownBars < 4 or posDir[4] == 0)
            and (cooldownBars < 5 or posDir[5] == 0) then
        (if ibLongRaw then 1
         else if ibShortRaw then -1
         else if vaLongRaw then 2
         else if vaShortRaw then -2
         else if eightyLongRaw then 3
         else if eightyShortRaw then -3
         else 0)
    # --- HOLD ---
    else posDir[1];

# Stop and target levels — for DISPLAY only (set on entry bar, held until exit)
rec stopLev =
    if posDir != 0 and posDir[1] == 0 then
        (if posDir == 1 then ibSL_L
         else if posDir == -1 then ibSL_S
         else if posDir == 2 then vaSL_L
         else if posDir == -2 then vaSL_S
         else if posDir == 3 then eightySL_L
         else eightySL_S)
    else if posDir != 0 then stopLev[1]
    else Double.NaN;

rec targLev =
    if posDir != 0 and posDir[1] == 0 then
        (if posDir == 1 then ibTP_L
         else if posDir == -1 then ibTP_S
         else if posDir == 2 then vaTP_L
         else if posDir == -2 then vaTP_S
         else if posDir == 3 then eightyTP_L
         else eightyTP_S)
    else if posDir != 0 then targLev[1]
    else Double.NaN;

# Entry price (for reference/exit markers)
rec entryPrice =
    if posDir != 0 and posDir[1] == 0 then close
    else if posDir != 0 then entryPrice[1]
    else Double.NaN;

# ============================================================================
# SIGNAL DETECTION (for arrows and alerts)
# ============================================================================

def ibLongFired = posDir == 1 and posDir[1] == 0;
def ibShortFired = posDir == -1 and posDir[1] == 0;
def vaLongFired = posDir == 2 and posDir[1] == 0;
def vaShortFired = posDir == -2 and posDir[1] == 0;
def eightyLongFired = posDir == 3 and posDir[1] == 0;
def eightyShortFired = posDir == -3 and posDir[1] == 0;

def anyLongEntry = ibLongFired or vaLongFired or eightyLongFired;
def anyShortEntry = ibShortFired or vaShortFired or eightyShortFired;

def exitLong = posDir[1] > 0 and posDir == 0;
def exitShort = posDir[1] < 0 and posDir == 0;

# ============================================================================
# PLOTS — TEMA LINES
# ============================================================================

plot pTemaFast = if showTEMA then temaFast else Double.NaN;
pTemaFast.SetDefaultColor(CreateColor(0, 255, 0));
pTemaFast.SetLineWeight(1);
pTemaFast.HideBubble();

plot pTemaSlow = if showTEMA then temaSlow else Double.NaN;
pTemaSlow.SetDefaultColor(CreateColor(255, 165, 0));
pTemaSlow.SetLineWeight(1);
pTemaSlow.HideBubble();

plot pTemaTrend = if showTEMA then temaTrend else Double.NaN;
pTemaTrend.SetDefaultColor(CreateColor(255, 255, 0));
pTemaTrend.SetLineWeight(2);
pTemaTrend.HideBubble();

# ============================================================================
# PLOTS — IB LEVELS
# ============================================================================

plot pIBHigh = if ibDone and showLevels then ibHigh else Double.NaN;
pIBHigh.SetDefaultColor(CreateColor(65, 105, 225));
pIBHigh.SetLineWeight(2);
pIBHigh.SetStyle(Curve.SHORT_DASH);
pIBHigh.HideBubble();

plot pIBLow = if ibDone and showLevels then ibLow else Double.NaN;
pIBLow.SetDefaultColor(CreateColor(65, 105, 225));
pIBLow.SetLineWeight(2);
pIBLow.SetStyle(Curve.SHORT_DASH);
pIBLow.HideBubble();

plot pIBMid = if ibDone and showLevels then ibMid else Double.NaN;
pIBMid.SetDefaultColor(CreateColor(65, 105, 225));
pIBMid.SetLineWeight(1);
pIBMid.SetStyle(Curve.MEDIUM_DASH);
pIBMid.HideBubble();

# ============================================================================
# PLOTS — VALUE AREA LEVELS
# ============================================================================

plot pVAH = if showLevels then prevVAHVal else Double.NaN;
pVAH.SetDefaultColor(CreateColor(255, 80, 80));
pVAH.SetLineWeight(1);
pVAH.SetStyle(Curve.SHORT_DASH);
pVAH.HideBubble();

plot pVAL = if showLevels then prevVALVal else Double.NaN;
pVAL.SetDefaultColor(CreateColor(80, 255, 80));
pVAL.SetLineWeight(1);
pVAL.SetStyle(Curve.SHORT_DASH);
pVAL.HideBubble();

plot pPOC = if showLevels then prevPOCVal else Double.NaN;
pPOC.SetDefaultColor(Color.WHITE);
pPOC.SetLineWeight(1);
pPOC.SetStyle(Curve.MEDIUM_DASH);
pPOC.HideBubble();

# ============================================================================
# PLOTS — STOP/TARGET WHILE IN POSITION
# ============================================================================

plot pStop = if posDir != 0 then stopLev else Double.NaN;
pStop.AssignValueColor(Color.RED);
pStop.SetLineWeight(3);
pStop.SetStyle(Curve.POINTS);
pStop.HideBubble();

plot pTarget = if posDir != 0 then targLev else Double.NaN;
pTarget.AssignValueColor(Color.GREEN);
pTarget.SetLineWeight(3);
pTarget.SetStyle(Curve.POINTS);
pTarget.HideBubble();

# ============================================================================
# ENTRY & EXIT SIGNALS
# Uses AddChartBubble ONLY (plot arrow colors get overridden by thinkorSwim UI).
# AddChartBubble colors are hardcoded and cannot be overridden.
# ============================================================================

# --- ENTRY BUBBLES: green for long, red for short, stem to exact fill (close) ---
AddChartBubble(ibLongFired, close, " IB LONG ", CreateColor(0, 255, 0), no);
AddChartBubble(ibShortFired, close, " IB SHORT ", CreateColor(255, 0, 0), yes);
AddChartBubble(vaLongFired, close, " VA LONG ", CreateColor(0, 255, 0), no);
AddChartBubble(vaShortFired, close, " VA SHORT ", CreateColor(255, 0, 0), yes);
AddChartBubble(eightyLongFired, close, " 80% LONG ", CreateColor(0, 255, 0), no);
AddChartBubble(eightyShortFired, close, " 80% SHORT ", CreateColor(255, 0, 0), yes);

# --- EXIT PRICE: stop if hit, target if hit, close otherwise ---
def exitPx =
    if exitLong then
        (if !IsNaN(stopLev[1]) and low <= stopLev[1] then stopLev[1]
         else if !IsNaN(targLev[1]) and high >= targLev[1] then targLev[1]
         else close)
    else if exitShort then
        (if !IsNaN(stopLev[1]) and high >= stopLev[1] then stopLev[1]
         else if !IsNaN(targLev[1]) and low <= targLev[1] then targLev[1]
         else close)
    else close;

# --- EXIT BUBBLES: white, stem to exit price ---
AddChartBubble(exitLong, exitPx, " EXIT ", CreateColor(255, 255, 255), yes);
AddChartBubble(exitShort, exitPx, " EXIT ", CreateColor(255, 255, 255), no);

# ============================================================================
# DASHBOARD — Vertical chart bubbles in expansion area (upper-right)
# Placed on a bar in the expansion area (right of price) at top of chart.
# Requires: Chart Settings > Time axis > Expansion area >= 10 bars
# ============================================================================

def _dataBar = if !IsNaN(close) then BarNumber() else 0;
def _lastBar = HighestAll(_dataBar);
# Place all bubbles 5 bars into the expansion area (right of last candle)
def _expBar = BarNumber() == _lastBar + 5;
def _show = _expBar and showDashboard;

# Carry last-bar values into expansion area (where close is NaN)
def _temaBullLast = HighestAll(if BarNumber() == _lastBar then temaBullish else 0) > 0;
def _trendUpLast = HighestAll(if BarNumber() == _lastBar then trendUp else 0) > 0;
def _temaSlopeLast = HighestAll(if BarNumber() == _lastBar then temaSlope else 0);
def _ibDoneLast = HighestAll(if BarNumber() == _lastBar then ibDone else 0) > 0;
def _ibRangeLast = HighestAll(if BarNumber() == _lastBar then ibRange else 0);
def _ibValidLast = HighestAll(if BarNumber() == _lastBar then ibValid else 0) > 0;
def _ibRatioLast = HighestAll(if BarNumber() == _lastBar then ibRatio else 0);
def _narrowLast = HighestAll(if BarNumber() == _lastBar then isNarrowIB else 0) > 0;
def _wideLast = HighestAll(if BarNumber() == _lastBar then isWideIB else 0) > 0;
def _prevPOCLast = HighestAll(if BarNumber() == _lastBar then prevPOCVal else 0);
def _aboveVALast = HighestAll(if BarNumber() == _lastBar then aboveVA else 0) > 0;
def _belowVALast = HighestAll(if BarNumber() == _lastBar then belowVA else 0) > 0;
def _volRatioLast = HighestAll(if BarNumber() == _lastBar then volRatio else 0);
def _volOKLast = HighestAll(if BarNumber() == _lastBar then volOK else 0) > 0;
def _isIBLast = HighestAll(if BarNumber() == _lastBar then isIBPeriod else 0) > 0;
def _isTWLast = HighestAll(if BarNumber() == _lastBar then isTradingWindow else 0) > 0;
def _isRTHLast = HighestAll(if BarNumber() == _lastBar then isRTH else 0) > 0;
# rec carries posDir into expansion area (HighestAll fails for negative values)
rec _posCarry = if !IsNaN(close) then posDir else _posCarry[1];
def _posDirLast = _posCarry;
def _openAboveLast = HighestAll(if BarNumber() == _lastBar then openAboveVA else 0) > 0;
def _openBelowLast = HighestAll(if BarNumber() == _lastBar then openBelowVA else 0) > 0;
def _eightyConfLast = HighestAll(if BarNumber() == _lastBar then eightyConfirmed else 0) > 0;
def _eightyReentLast = HighestAll(if BarNumber() == _lastBar then eightyReentered else 0) > 0;
def _eightyCntLast = HighestAll(if BarNumber() == _lastBar then eightyInsideCount else 0);

# IB status needs close vs ibHigh/ibLow from last bar
def _closeLast = HighestAll(if BarNumber() == _lastBar then close else 0);
def _ibHighLast = HighestAll(if BarNumber() == _lastBar then ibHigh else 0);
def _ibLowLast = HighestAll(if BarNumber() == _lastBar then ibLow else 0);

# Anchor at top of chart, scale row spacing to full chart range
def _chartHi = HighestAll(high);
def _chartLo = LowestAll(low);
def _s = (_chartHi - _chartLo) / 50;
def _y = _chartHi - _s;

AddChartBubble(_show, _y,
    " AMT-TEMA v6 ", CreateColor(220, 140, 0), yes);

AddChartBubble(_show, _y - _s,
    " TEMA     " + (if _temaBullLast then "BULL" else "BEAR") + " ",
    if _temaBullLast then CreateColor(0, 140, 0) else CreateColor(190, 0, 0), yes);

AddChartBubble(_show, _y - _s * 2,
    " Trend    " + (if _trendUpLast then "UP" else "DOWN") + " ",
    if _trendUpLast then CreateColor(0, 140, 0) else CreateColor(190, 0, 0), yes);

AddChartBubble(_show, _y - _s * 3,
    " Slope    " + AsText(_temaSlopeLast, NumberFormat.TWO_DECIMAL_PLACES) + " ",
    if _temaSlopeLast > 0 then CreateColor(0, 140, 0) else CreateColor(190, 0, 0), yes);

AddChartBubble(_show, _y - _s * 4,
    " IB Range " + (if _ibDoneLast then AsText(_ibRangeLast, NumberFormat.TWO_DECIMAL_PLACES) + "pt" else "FORMING") + " ",
    if !_ibDoneLast then CreateColor(180, 160, 0)
    else if _ibValidLast then CreateColor(0, 140, 0)
    else Color.ORANGE, yes);

AddChartBubble(_show, _y - _s * 5,
    " Day Type " + (if _narrowLast then "NARROW" else if _wideLast then "WIDE" else "NORMAL")
    + " (" + AsText(_ibRatioLast, NumberFormat.TWO_DECIMAL_PLACES) + "x) ",
    if _narrowLast then CreateColor(0, 160, 0)
    else if _wideLast then Color.ORANGE
    else CreateColor(80, 80, 80), yes);

AddChartBubble(_show, _y - _s * 6,
    " IB Status " + (if !_ibDoneLast then "---"
    else if _closeLast > _ibHighLast then "ABOVE"
    else if _closeLast < _ibLowLast then "BELOW"
    else "INSIDE") + " ",
    if !_ibDoneLast then Color.GRAY
    else if _closeLast > _ibHighLast then CreateColor(0, 140, 0)
    else if _closeLast < _ibLowLast then CreateColor(190, 0, 0)
    else CreateColor(80, 80, 80), yes);

AddChartBubble(_show, _y - _s * 7,
    " vs VA    " + (if _aboveVALast then "ABOVE" else if _belowVALast then "BELOW" else "INSIDE") + " ",
    if _aboveVALast then CreateColor(0, 140, 0)
    else if _belowVALast then CreateColor(190, 0, 0)
    else CreateColor(80, 80, 80), yes);

AddChartBubble(_show, _y - _s * 8,
    " POC      " + AsText(_prevPOCLast, NumberFormat.TWO_DECIMAL_PLACES) + " ",
    CreateColor(80, 80, 80), yes);

AddChartBubble(_show, _y - _s * 9,
    " Vol      " + AsText(_volRatioLast, NumberFormat.TWO_DECIMAL_PLACES) + "x ",
    if _volOKLast then CreateColor(0, 140, 0) else CreateColor(190, 0, 0), yes);

AddChartBubble(_show, _y - _s * 10,
    " Session  " + (if _isIBLast then "IB"
    else if _isTWLast then "ACTIVE"
    else if _isRTHLast then "RTH"
    else "CLOSED") + " ",
    if _isTWLast then CreateColor(0, 140, 0)
    else if _isIBLast then CreateColor(180, 160, 0)
    else if _isRTHLast then CreateColor(80, 80, 80)
    else Color.GRAY, yes);

AddChartBubble(_show, _y - _s * 11,
    " Position " + (if _posDirLast == 1 then "IB LONG"
    else if _posDirLast == -1 then "IB SHORT"
    else if _posDirLast == 2 then "VA LONG"
    else if _posDirLast == -2 then "VA SHORT"
    else if _posDirLast == 3 then "80% LONG"
    else if _posDirLast == -3 then "80% SHORT"
    else "FLAT") + " ",
    if _posDirLast == 0 then Color.GRAY
    else if _posDirLast > 0 then CreateColor(0, 140, 0)
    else CreateColor(190, 0, 0), yes);

AddChartBubble(_show and useEighty, _y - _s * 12,
    " 80% Rule " + (if !(_openAboveLast or _openBelowLast) then "INACTIVE"
    else if _eightyConfLast then "CONFIRMED"
    else if _eightyReentLast then "COUNT " + AsText(_eightyCntLast)
    else "WAITING") + " ",
    if _eightyConfLast then CreateColor(0, 160, 0)
    else if _eightyReentLast then CreateColor(180, 160, 0)
    else Color.GRAY, yes);

# ============================================================================
# ALERTS
# ============================================================================

Alert(ibLongFired, "AMT-TEMA: IB Breakout LONG", Alert.BAR, Sound.Ding);
Alert(ibShortFired, "AMT-TEMA: IB Breakout SHORT", Alert.BAR, Sound.Ding);
Alert(vaLongFired, "AMT-TEMA: VA Fade LONG", Alert.BAR, Sound.Ring);
Alert(vaShortFired, "AMT-TEMA: VA Fade SHORT", Alert.BAR, Sound.Ring);
Alert(eightyLongFired, "AMT-TEMA: 80% Rule LONG", Alert.BAR, Sound.Bell);
Alert(eightyShortFired, "AMT-TEMA: 80% Rule SHORT", Alert.BAR, Sound.Bell);
Alert(exitLong or exitShort, "AMT-TEMA: Position EXIT", Alert.BAR, Sound.Chimes);
