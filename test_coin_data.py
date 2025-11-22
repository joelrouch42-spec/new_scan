#!/usr/bin/env python3
"""Test combined signals with COIN data from file"""

import json
import pandas as pd
from macd_analyzer import MACDAnalyzer
from squeeze_momentum_analyzer import SqueezeAnalyzer
from adx_analyzer import ADXAnalyzer

# Load settings
with open('settings.json', 'r') as f:
    settings = json.load(f)

# Load COIN data from file
df = pd.read_csv('data/2025-11-22_COIN_210_1d.csv')

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

# Create index maps
macd_by_idx = {v['index']: v for v in macd_result['values']}
squeeze_by_idx = {v['index']: v for v in squeeze_result['values']}
adx_by_idx = {v['index']: v for v in adx_result['values']}

common_indices = sorted(set(macd_by_idx.keys()) & set(squeeze_by_idx.keys()) & set(adx_by_idx.keys()))

print(f"Common indices: {len(common_indices)}")
print()

# Calculate combined states
print("="*120)
print(f"{'Date':<12} {'Close':>8} {'SQZ':>7} {'MACD':>5} {'ADX':>6} {'Trend':>6} {'CombG':>6} {'CombR':>6} {'Signal':<20}")
print("="*120)

combined_states = {}
for idx in common_indices:
    sqz = squeeze_by_idx[idx]
    macd = macd_by_idx[idx]
    adx = adx_by_idx[idx]

    sqz_green = sqz['color'] == 'lime'
    sqz_red = sqz['color'] == 'red'
    macd_green = macd['line_color'] == 'green'
    macd_red = macd['line_color'] == 'red'
    in_trend = adx['in_trend']

    combined_states[idx] = {
        'combined_green': sqz_green and macd_green,
        'combined_red': sqz_red and macd_red,
        'sqz_color': sqz['color'],
        'macd_color': macd['line_color'],
        'adx': adx['adx'],
        'in_trend': in_trend
    }

# Detect transitions
buy_signals = []
sell_signals = []

for i in range(1, len(common_indices)):
    idx = common_indices[i]
    prev_idx = common_indices[i-1]

    curr = combined_states[idx]
    prev = combined_states[prev_idx]

    signal = ''

    # Transition + ADX filter
    if curr['combined_green'] and not prev['combined_green'] and curr['in_trend']:
        buy_signals.append(idx)
        signal = '🟢 BUY SIGNAL'

    if curr['combined_red'] and not prev['combined_red'] and curr['in_trend']:
        sell_signals.append(idx)
        signal = '🔴 SELL SIGNAL'

    # Print interesting lines
    if curr['combined_green'] or curr['combined_red'] or signal:
        date = df.iloc[idx]['Date']
        close = df.iloc[idx]['Close']

        print(f"{date:<12} {close:>8.2f} {curr['sqz_color']:>7} {curr['macd_color']:>5} {curr['adx']:>6.1f} "
              f"{str(curr['in_trend']):>6} {str(curr['combined_green']):>6} {str(curr['combined_red']):>6} {signal:<20}")

print("="*120)
print()
print(f"Total BUY signals: {len(buy_signals)}")
print(f"Total SELL signals: {len(sell_signals)}")
print()

if buy_signals:
    print("BUY signals:")
    for idx in buy_signals:
        print(f"  {df.iloc[idx]['Date']} @ ${df.iloc[idx]['Close']:.2f}")

if sell_signals:
    print("\nSELL signals:")
    for idx in sell_signals:
        print(f"  {df.iloc[idx]['Date']} @ ${df.iloc[idx]['Close']:.2f}")
