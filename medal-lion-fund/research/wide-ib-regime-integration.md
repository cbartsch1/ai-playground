# Wide IB Day Type — Medallion 2.0 Integration Needed

## Problem

The IB Rejection setup (AMT-TEMA v10) is highly profitable ONLY on "wide" IB days
(IB ratio >= 1.2, where ratio = today's IB range / 20-day EMA of IB range).

- Wide days: PF 2.97, +$75K (2yr)
- Normal days: PF 0.76, -$19K
- Narrow days: PF 0.98, -$1.4K

"Wide" is NOT currently a defined regime within the Medallion 2.0 HMM model.
The 7-state HMM uses daily SPY data (log returns, HL range, volume volatility)
to classify market regimes. IB width is an intraday ES-specific concept that
operates on a completely different timescale and instrument.

## What Needs to Happen

1. **Determine if "wide IB" correlates with any existing HMM state**
   - Wide IB days might map to Trending or High Volatility regimes
   - If so, Medallion can deploy IB Rejection when those regimes are detected
   - If NOT, a new signal/feature is needed

2. **Consider adding IB day type as a confirmation or feature**
   - Could be a new confirmation signal (like VIX term structure or breadth)
   - Could be a separate intraday classifier that runs alongside the HMM
   - IB ratio is computable at 10:30 ET each day — it's a real-time signal

3. **Architecture question: intraday vs daily regime**
   - Medallion currently operates on hourly SPY data
   - IB width is computed once per day at 10:30 ET on ES
   - These are different instruments and timeframes
   - May need an "intraday context" layer that sits alongside the HMM

4. **Strategy deployment logic**
   - Medallion needs to know: "today is a wide IB day → deploy IB Rejection"
   - This is a DAILY decision made at 10:30 ET, not a regime transition
   - Different from current regime-based deployment (regime can last weeks)

## Key Data

- IB ratio = ib_range / ib_range_ema_20
- Wide threshold: ratio >= 1.2 (already in Pine Script as `isWideIB`)
- Computable at 10:30 ET when IB period ends
- ~30-35% of days qualify as "wide" based on 2yr Databento data

## Priority

This is a LATER project — first validate IB Rejection visually on TradingView,
then walk-forward validate, then integrate with Medallion for automated deployment.
