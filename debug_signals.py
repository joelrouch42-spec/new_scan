#!/usr/bin/env python3
"""Debug script to check MACD line + Squeeze histogram colors for specific dates"""

import pandas as pd
import numpy as np
from macd_analyzer import MACDAnalyzer
from squeeze_momentum_analyzer import SqueezeAnalyzer

# Create sample data for March-May 2025 to include April dates
dates = pd.date_range(start='2025-03-01', periods=80, freq='D')
np.random.seed(100)  # Different seed for different pattern

# Simulate price data with realistic patterns
base_price = 120
prices = []
for i in range(80):
    if i < 30:
        # Uptrend
        noise = np.random.normal(0, 2)
        price = base_price + i * 0.8 + noise
    elif i < 50:
        # Sideways with decline
        noise = np.random.normal(0, 2.5)
        price = base_price + 30 * 0.8 - (i - 30) * 0.2 + noise
    else:
        # Further decline
        noise = np.random.normal(0, 2)
        price = base_price + 30 * 0.8 - 20 * 0.2 - (i - 50) * 0.5 + noise
    prices.append(max(price, 50))  # Floor at 50

df = pd.DataFrame({
    'Date': dates.strftime('%Y-%m-%d'),
    'Open': prices,
    'High': [p * 1.015 for p in prices],
    'Low': [p * 0.985 for p in prices],
    'Close': prices,
    'Volume': [1000000 + np.random.randint(-200000, 200000) for _ in range(80)]
})

# Initialize analyzers
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
print("Analyzing data...")
macd_result = macd_analyzer.analyze(df)
squeeze_result = squeeze_analyzer.analyze(df)

# Create index lookup for squeeze
squeeze_by_index = {v['index']: v for v in squeeze_result['values']}

# Print detailed info for April dates and surrounding periods
print("\n" + "=" * 120)
print(f"{'Date':<12} {'Price':<8} {'MACD Line':<12} {'Squeeze Hist':<15} {'Signal':<8} {'Momentum':<10} {'Note'}")
print("=" * 120)

for macd_val in macd_result['values']:
    idx = macd_val['index']

    if idx in squeeze_by_index:
        squeeze_val = squeeze_by_index[idx]
        date = df.iloc[idx]['Date']
        price = df.iloc[idx]['Close']
        macd_line = macd_val['line_color']
        squeeze_color = squeeze_val['color']
        momentum = squeeze_val['momentum']

        signal = 'NONE'
        note = ''

        # BUY: MACD ligne verte + Squeeze histogram lime
        if macd_line == 'green' and squeeze_color == 'lime':
            signal = 'BUY ✓'
            note = 'CORRECT: green + lime'

        # SELL: MACD ligne rouge + Squeeze histogram maroon
        elif macd_line == 'red' and squeeze_color == 'maroon':
            signal = 'SELL ✓'
            note = 'CORRECT: red + maroon'

        # WRONG combinations that should NOT signal
        elif macd_line == 'red' and squeeze_color == 'red':
            signal = 'NONE'
            note = 'NO SIGNAL: red + rouge sombre (not maroon)'

        # Only show March 15 onwards and especially April dates
        if '2025-03-1' in date or '2025-04' in date or '2025-05-0' in date:
            print(f"{date:<12} ${price:<7.2f} {macd_line:<12} {squeeze_color:<15} {signal:<8} {momentum:<10.4f} {note}")

print("=" * 120)
print("\nLégende:")
print("  - lime: vert vif (positif et croissant)")
print("  - green: vert sombre (positif mais décroissant)")
print("  - maroon: rouge vif (négatif et descendant = plus négatif)")
print("  - red: rouge sombre (négatif mais rebondissant = moins négatif)")
print("\nRègles de signaux:")
print("  ✓ BUY: MACD ligne verte + Squeeze histogram lime")
print("  ✓ SELL: MACD ligne rouge + Squeeze histogram maroon (rouge vif)")
print("  ✗ Pas de signal si Squeeze histogram est rouge sombre (red)")
