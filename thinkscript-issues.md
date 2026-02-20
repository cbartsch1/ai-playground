# thinkScript Conversion — Open Issues (Feb 13, 2026)

## File: `amt-tema-strategy-tos.ts`

## What WORKS
- Core logic: TEMA engine, IB tracker, VWAP-based VA, day type classification, volatility filter
- All 3 setups: IB Breakout, VA Fade, 80% Rule (entry/exit logic in posDir)
- Position tracking via `rec posDir` — inline exit conditions (no forward refs)
- TEMA line plots — green/orange/yellow, correct colors via `SetDefaultColor(CreateColor(...))`
- IB level plots (blue dashed) — correct
- VA level plots (red/green/white dashed) — correct
- Stop/target dots (red/green POINTS) — visible during positions
- Dashboard in expansion area — vertical bubbles, upper-right, correct colors
- Alerts — distinct sounds per signal type
- Session/time handling — `RegularTradingStart()` timezone-safe

## What's BROKEN: Entry/Exit Signal Markers

### Problem
Entry and exit signals are invisible or grey on the chart. User needs clear green (long entry), red (short entry), and white (exit) markers with bubbles pointing to exact price.

### What was tried and FAILED

1. **`plot` + `SetDefaultColor(Color.GREEN)` + `PaintingStrategy.ARROW_UP`**
   - Result: Grey arrows. thinkorSwim UI overrides SetDefaultColor on sparse plots.

2. **`plot` + `AssignValueColor(Color.GREEN)` + ARROW_UP**
   - Result: Still grey. Constant color treated same as SetDefaultColor.

3. **`plot` + `AssignValueColor(CreateColor(0,255,0))` + ARROW_UP**
   - Result: Still grey.

4. **`plot` + `AssignValueColor(if close > 0 then CreateColor(0,255,0) else CreateColor(0,255,0))` + ARROW_UP**
   - Result: Still grey. Conditional doesn't help.

5. **Consolidated 6 entry plots into 2 + used all above approaches**
   - Result: Still grey.

6. **Removed all arrow plots, used AddChartBubble only**
   - Result: Entry bubbles partially visible (small white stems seen in zoomed screenshot) but TOO SMALL to be useful. Exit bubbles possibly showing but not visible at chart zoom levels.

### What DOES work for colors
- `AddChartBubble` — colors hardcoded, cannot be overridden (proven by dashboard)
- `SetDefaultColor(CreateColor(...))` on CONTINUOUS plots (TEMA lines) — works
- `AssignValueColor(Color.RED)` on POINTS painting strategy (pStop) — works (red dots visible)

### Key insight
The color override issue appears specific to `PaintingStrategy.ARROW_UP/ARROW_DOWN` on sparse (mostly NaN) plots. Continuous plots and POINTS plots retain their colors.

### Approaches NOT yet tried
1. **Use POINTS or SQUARES painting strategy instead of ARROW** — pStop works with POINTS + AssignValueColor. Try making entry/exit markers as POINTS with large line weight.
2. **Use BOOLEAN_ARROW_UP/DOWN** — paints at fixed chart edge, not at specific price. Less precise but might work.
3. **Combine signal into single continuous plot** — e.g. always output 0, output +1 on entry, -1 on short entry. Use AssignValueColor with conditional. Since plot is "continuous" (never NaN), colors might stick.
4. **AddVerticalLine for entries/exits** — full vertical line at signal bar, hardcoded color. Very visible but cluttered.
5. **Larger AddChartBubble text** — pad with more spaces/characters to make bubbles bigger.
6. **Research thinkorSwim forums** — someone else must have solved colored sparse arrows.

### Dashboard posDir bug (FIXED)
- `HighestAll(if BarNumber() == _lastBar then posDir else Double.NEGATIVE_INFINITY)` was broken for negative posDir values
- Showed "80% SHORT" as catchall when value was unexpected
- Fixed: `rec _posCarry` carries posDir into expansion area; fallthrough now shows "FLAT" instead of "80% SHORT"

## thinkScript Language Gotchas
- `def` is numeric-only — cannot hold strings (use inline in AddLabel/AddChartBubble)
- NO forward references to `rec` variables, even with `[1]` offsets — strict top-to-bottom order
- `HighestAll()` returns max across ALL bars — doesn't work for carrying negative values to expansion area
- No equivalent to TradingView's `table.new(position.top_right)` — use expansion area bubbles
- `RegularTradingStart(GetYYYYMMDD())` for timezone-safe RTH detection
- Expansion area bars have NaN close — filter with `if !IsNaN(close)` before `HighestAll`
