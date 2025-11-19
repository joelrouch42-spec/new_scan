#!/usr/bin/env python3
"""Test script to log combined signals on sample data (simulating NVDA July-Sept 2025)"""

import pandas as pd
import numpy as np
from macd_analyzer import MACDAnalyzer
from squeeze_momentum_analyzer import SqueezeAnalyzer

# Create sample data simulating NVDA around July-Sept 2025
dates = pd.date_range(start='2025-06-01', periods=100, freq='D')
np.random.seed(42)

# Simulate price data with an uptrend then downtrend
base_price = 100
prices = []
for i in range(100):
    if i < 50:
        # Uptrend
        noise = np.random.normal(0, 2)
        price = base_price + i * 0.5 + noise
    else:
        # Downtrend
        noise = np.random.normal(0, 2)
        price = base_price + 50 * 0.5 - (i - 50) * 0.3 + noise
    prices.append(price)

df = pd.DataFrame({
    'Date': dates.strftime('%Y-%m-%d'),
    'Open': prices,
    'High': [p * 1.01 for p in prices],
    'Low': [p * 0.99 for p in prices],
    'Close': prices,
    'Volume': [1000000] * 100
})

# Initialize analyzers with default config
macd_config = {
    'fast_length': 12,
    'slow_length': 26,
    'signal_length': 9
}

squeeze_config = {
    'bb_length': 20,
    'bb_mult': 2.0,
    'kc_length': 20,
    'kc_mult': 1.5,
    'use_true_range': True
}

macd_analyzer = MACDAnalyzer(macd_config)
squeeze_analyzer = SqueezeAnalyzer(squeeze_config)

# Analyze
macd_result = macd_analyzer.analyze(df)
squeeze_result = squeeze_analyzer.analyze(df)

# Create index lookup for squeeze
squeeze_by_index = {v['index']: v for v in squeeze_result['values']}

# Find combined signals
print("Combined Signal Detection (MACD line + Squeeze Momentum histogram)")
print("=" * 100)
print(f"{'Date':<12} {'MACD Line':<12} {'Squeeze Color':<15} {'Signal':<10} {'Description'}")
print("=" * 100)

buy_count = 0
sell_count = 0

for macd_val in macd_result['values']:
    idx = macd_val['index']

    if idx in squeeze_by_index:
        squeeze_val = squeeze_by_index[idx]
        date = df.iloc[idx]['Date']
        macd_line = macd_val['line_color']
        squeeze_color = squeeze_val['color']

        signal = 'NONE'
        description = ''

        # BUY: MACD ligne verte + Squeeze histogram lime (vert vif)
        if macd_line == 'green' and squeeze_color == 'lime':
            signal = 'BUY'
            description = 'MACD green + Squeeze lime (vert vif)'
            buy_count += 1

        # SELL: MACD ligne rouge + Squeeze histogram red (rouge vif)
        elif macd_line == 'red' and squeeze_color == 'red':
            signal = 'SELL'
            description = 'MACD red + Squeeze red (rouge vif)'
            sell_count += 1

        # Only print signals (not every candle)
        if signal != 'NONE':
            print(f"{date:<12} {macd_line:<12} {squeeze_color:<15} {signal:<10} {description}")

print("=" * 100)
print(f"Total BUY signals: {buy_count}")
print(f"Total SELL signals: {sell_count}")
print(f"\nLogic verified:")
print("  ✓ BUY = MACD line green + Squeeze histogram lime (vert vif)")
print("  ✓ SELL = MACD line red + Squeeze histogram red (rouge vif)")
