#!/usr/bin/env python3
"""VAR exhaustive parameter sweep — every combination."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtester.config import StrategyConfig
from backtester.data_loader import load_tos_csv
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics

df = load_tos_csv('data/es_5m_databento_2yr.csv', instrument='ES')
print(f"Loaded {len(df)} bars")

results = []

# Sweep: zone_pts, stop_buffer, min_rr, min_ib_periods, max_otf, target_pts, direction
zone_values = [2.0, 3.0, 5.0]
stop_values = [3.0, 4.0, 5.0, 7.0, 10.0]
rr_values = [0.5, 0.8, 1.0, 1.5]
period_values = [3, 4, 5, 6]
otf_values = [1, 2, 3]
target_values = [0.0, 5.0]  # 0=dynamic POC, 5=fixed
dir_values = ["both", "short"]

total = len(zone_values) * len(stop_values) * len(rr_values) * len(period_values) * len(otf_values) * len(target_values) * len(dir_values)
print(f"Total combinations: {total}")

count = 0
for zone in zone_values:
    for stop in stop_values:
        for rr in rr_values:
            for periods in period_values:
                for otf in otf_values:
                    for tgt in target_values:
                        for direction in dir_values:
                            count += 1
                            cfg = StrategyConfig()
                            cfg.direction_filter = direction
                            cfg.use_ib_break = False
                            cfg.use_va_fade = False
                            cfg.use_eighty = False
                            cfg.use_tema_cross = False
                            cfg.use_level_reject = False
                            cfg.use_level_reject_long = False
                            cfg.use_ib_reject = False
                            cfg.use_var = True
                            cfg.use_ptf = False
                            cfg.var_zone_pts = zone
                            cfg.var_target_pts = tgt
                            cfg.var_stop_buffer = stop
                            cfg.var_min_ib_periods = periods
                            cfg.var_require_rotation = True
                            cfg.var_max_otf = otf
                            cfg.max_var_trades = 8
                            cfg.var_min_rr = rr

                            trades = run_backtest(df.copy(), cfg)
                            if trades:
                                m = compute_metrics(trades)
                                results.append({
                                    'zone': zone, 'stop': stop, 'rr': rr,
                                    'periods': periods, 'otf': otf, 'tgt': tgt,
                                    'dir': direction,
                                    'trades': m.total_trades, 'wr': m.win_rate,
                                    'pf': m.profit_factor, 'pnl': m.net_pnl,
                                    'dd': m.max_drawdown, 'sharpe': m.sharpe,
                                    'avg': m.avg_trade
                                })
                            
                            if count % 100 == 0:
                                print(f"  {count}/{total} done...")

# Sort by PF (only configs with >= 100 trades)
viable = [r for r in results if r['trades'] >= 50]
viable.sort(key=lambda x: x['pf'], reverse=True)

print(f"\n{'='*120}")
print(f"TOP 20 BY PROFIT FACTOR (>= 50 trades)")
print(f"{'='*120}")
print(f"{'Zone':>5s} {'Stop':>5s} {'RR':>4s} {'Per':>4s} {'OTF':>4s} {'Tgt':>4s} {'Dir':>6s} "
      f"{'Trades':>7s} {'WR':>6s} {'PF':>7s} {'P&L':>10s} {'DD':>8s} {'Sharpe':>7s} {'Avg':>7s}")
print('-'*120)
for r in viable[:20]:
    tgt_label = 'POC' if r['tgt'] == 0 else f"{r['tgt']:.0f}pt"
    print(f"{r['zone']:>5.0f} {r['stop']:>5.0f} {r['rr']:>4.1f} {r['periods']:>4d} {r['otf']:>4d} {tgt_label:>4s} {r['dir']:>6s} "
          f"{r['trades']:>7d} {r['wr']:>5.1f}% {r['pf']:>7.3f} ${r['pnl']:>+9,.0f} ${r['dd']:>7,.0f} {r['sharpe']:>7.2f} ${r['avg']:>6,.0f}")

# Also show top by Sharpe
viable.sort(key=lambda x: x['sharpe'], reverse=True)
print(f"\nTOP 10 BY SHARPE (>= 50 trades)")
print('-'*120)
for r in viable[:10]:
    tgt_label = 'POC' if r['tgt'] == 0 else f"{r['tgt']:.0f}pt"
    print(f"{r['zone']:>5.0f} {r['stop']:>5.0f} {r['rr']:>4.1f} {r['periods']:>4d} {r['otf']:>4d} {tgt_label:>4s} {r['dir']:>6s} "
          f"{r['trades']:>7d} {r['wr']:>5.1f}% {r['pf']:>7.3f} ${r['pnl']:>+9,.0f} ${r['dd']:>7,.0f} {r['sharpe']:>7.2f} ${r['avg']:>6,.0f}")

# Top by net P&L
viable.sort(key=lambda x: x['pnl'], reverse=True)
print(f"\nTOP 10 BY NET P&L (>= 50 trades)")
print('-'*120)
for r in viable[:10]:
    tgt_label = 'POC' if r['tgt'] == 0 else f"{r['tgt']:.0f}pt"
    print(f"{r['zone']:>5.0f} {r['stop']:>5.0f} {r['rr']:>4.1f} {r['periods']:>4d} {r['otf']:>4d} {tgt_label:>4s} {r['dir']:>6s} "
          f"{r['trades']:>7d} {r['wr']:>5.1f}% {r['pf']:>7.3f} ${r['pnl']:>+9,.0f} ${r['dd']:>7,.0f} {r['sharpe']:>7.2f} ${r['avg']:>6,.0f}")
