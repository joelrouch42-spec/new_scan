#!/usr/bin/env python3
"""Debug combined signals for AAPL"""

import json
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from macd_analyzer import MACDAnalyzer
from squeeze_momentum_analyzer import SqueezeAnalyzer
from adx_analyzer import ADXAnalyzer

# Load settings
with open('settings.json', 'r') as f:
    settings = json.load(f)

# Get AAPL data
print("Downloading AAPL data...")
ticker = yf.Ticker('AAPL')
end_date = datetime.now(ZoneInfo('America/New_York'))
start_date = end_date - timedelta(days=400)
df = ticker.history(start=start_date, end=end_date, interval='1d')
df = df.tail(250)
df.reset_index(inplace=True)
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

print(f"Data: {len(df)} candles from {df.iloc[0]['Date']} to {df.iloc[-1]['Date']}")
print()

# Initialize analyzers
macd_config = settings['indicators']['macd']
squeeze_config = settings['indicators']['squeeze_momentum']
adx_config = settings['indicators']['adx']

macd_analyzer = MACDAnalyzer(macd_config)
squeeze_analyzer = SqueezeAnalyzer(squeeze_config)
adx_analyzer = ADXAnalyzer(adx_config)

# Analyze
macd_result = macd_analyzer.analyze(df)
squeeze_result = squeeze_analyzer.analyze(df)
adx_result = adx_analyzer.analyze(df)

print(f"MACD values: {len(macd_result['values'])}")
print(f"Squeeze values: {len(squeeze_result['values'])}")
print(f"ADX values: {len(adx_result['values'])}")
print()

# Create index maps
macd_by_idx = {v['index']: v for v in macd_result['values']}
squeeze_by_idx = {v['index']: v for v in squeeze_result['values']}
adx_by_idx = {v['index']: v for v in adx_result['values']}

common_indices = sorted(set(macd_by_idx.keys()) & set(squeeze_by_idx.keys()) & set(adx_by_idx.keys()))

print(f"Common indices: {len(common_indices)}")
print()

# Generate signals
buy_signals = []
sell_signals = []

prev_green = False
prev_red = False

print("=" * 120)
print(f"{'Date':<12} {'Close':>8} {'SQZ':>6} {'MACD':>5} {'ADX':>6} {'InTrend':>8} {'CombGreen':>10} {'CombRed':>9} {'Signal':>10}")
print("=" * 120)

for i, idx in enumerate(common_indices):
    if i == 0:
        continue

    sqz = squeeze_by_idx[idx]
    macd = macd_by_idx[idx]
    adx = adx_by_idx[idx]

    sqz_green = sqz['color'] == 'lime'
    sqz_red = sqz['color'] == 'red'
    macd_green = macd['line_color'] == 'green'
    macd_red = macd['line_color'] == 'red'
    in_trend = adx['in_trend']

    combined_green = sqz_green and macd_green and in_trend
    combined_red = sqz_red and macd_red and in_trend

    signal = ''

    # Transition detection
    if combined_green and not prev_green:
        signal = '🟢 BUY'
        buy_signals.append(idx)

    if combined_red and not prev_red:
        signal = '🔴 SELL'
        sell_signals.append(idx)

    # Print only signals or interesting transitions
    if signal or combined_green or combined_red:
        date = df.iloc[idx]['Date']
        close = df.iloc[idx]['Close']

        print(f"{date:<12} {close:>8.2f} {sqz['color']:>6} {macd['line_color']:>5} {adx['adx']:>6.1f} {str(in_trend):>8} "
              f"{str(combined_green):>10} {str(combined_red):>9} {signal:>10}")

    prev_green = combined_green
    prev_red = combined_red

print("=" * 120)
print()
print(f"Total BUY signals: {len(buy_signals)}")
print(f"Total SELL signals: {len(sell_signals)}")
print()

if buy_signals:
    print("BUY signals dates:")
    for idx in buy_signals:
        print(f"  {df.iloc[idx]['Date']} - ${df.iloc[idx]['Close']:.2f}")

if sell_signals:
    print("\nSELL signals dates:")
    for idx in sell_signals:
        print(f"  {df.iloc[idx]['Date']} - ${df.iloc[idx]['Close']:.2f}")
