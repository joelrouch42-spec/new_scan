#!/usr/bin/env python3
"""
Script de test pour vérifier la détection des signaux MACD
"""
import pandas as pd
from macd_analyzer import MACDAnalyzer
import json

# Charger la configuration
with open('settings.json', 'r') as f:
    settings = json.load(f)

macd_config = settings['indicators']['macd']
analyzer = MACDAnalyzer(macd_config)

# Charger les données AAPL
import os
from datetime import datetime
from zoneinfo import ZoneInfo

now = datetime.now(ZoneInfo('America/New_York'))
today = now.strftime('%Y-%m-%d')
filename = f"data/{today}_AAPL_210_1d.csv"

if os.path.exists(filename):
    df = pd.read_csv(filename)
    print(f"Données chargées: {len(df)} bougies")
    print(f"Période: {df['Date'].iloc[0]} à {df['Date'].iloc[-1]}\n")

    # Analyser
    result = analyzer.analyze(df)

    print("=" * 80)
    print("SIGNAUX BUY DÉTECTÉS (Ligne verte + histogramme lime)")
    print("=" * 80)
    for signal in result['buy_signals']:
        idx = signal['index']
        date = df.iloc[idx]['Date']
        price = signal['price']
        hist = signal['histogram']
        print(f"  [{idx}] {date}: ${price:.2f} | histogram={hist:.4f}")

    print(f"\nTotal BUY: {len(result['buy_signals'])}\n")

    print("=" * 80)
    print("SIGNAUX SELL DÉTECTÉS (Ligne rouge + histogramme maroon)")
    print("=" * 80)
    for signal in result['sell_signals']:
        idx = signal['index']
        date = df.iloc[idx]['Date']
        price = signal['price']
        hist = signal['histogram']
        print(f"  [{idx}] {date}: ${price:.2f} | histogram={hist:.4f}")

    print(f"\nTotal SELL: {len(result['sell_signals'])}\n")

    # Afficher quelques valeurs autour du dernier signal pour analyse
    if result['buy_signals']:
        print("\n" + "=" * 80)
        print("DÉTAILS AUTOUR DU DERNIER SIGNAL BUY")
        print("=" * 80)
        last_buy = result['buy_signals'][-1]
        idx = last_buy['index']

        print(f"\n{'Index':<6} {'Date':<12} {'Close':<10} {'Line':<8} {'Hist Color':<12} {'Histogram':<12}")
        print("-" * 80)

        for i in range(max(0, idx-3), min(len(df), idx+4)):
            found = False
            for val in result['values']:
                if val['index'] == i:
                    marker = " >>> " if i == idx else "     "
                    print(f"{marker}{i:<6} {df.iloc[i]['Date']:<12} {df.iloc[i]['Close']:<10.2f} "
                          f"{val['line_color']:<8} {val['hist_color']:<12} {val['histogram']:<12.4f}")
                    found = True
                    break

    if result['sell_signals']:
        print("\n" + "=" * 80)
        print("DÉTAILS AUTOUR DU DERNIER SIGNAL SELL")
        print("=" * 80)
        last_sell = result['sell_signals'][-1]
        idx = last_sell['index']

        print(f"\n{'Index':<6} {'Date':<12} {'Close':<10} {'Line':<8} {'Hist Color':<12} {'Histogram':<12}")
        print("-" * 80)

        for i in range(max(0, idx-3), min(len(df), idx+4)):
            found = False
            for val in result['values']:
                if val['index'] == i:
                    marker = " >>> " if i == idx else "     "
                    print(f"{marker}{i:<6} {df.iloc[i]['Date']:<12} {df.iloc[i]['Close']:<10.2f} "
                          f"{val['line_color']:<8} {val['hist_color']:<12} {val['histogram']:<12.4f}")
                    found = True
                    break

else:
    print(f"Fichier non trouvé: {filename}")
    print("Lance d'abord: python3 new_scan.py --backtest --chart AAPL")
