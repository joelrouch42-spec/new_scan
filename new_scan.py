#!/usr/bin/env python3
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from zoneinfo import ZoneInfo
from ib_insync import IB, Stock
import time
import argparse
import numpy as np
from typing import List, Tuple, Optional, Dict
import plotly.graph_objects as go
from smc_analyzer import SMCAnalyzer

class StockScanner:
    def __init__(self, settings_file, is_backtest=False, patterns_file='patterns.json', chart_symbol=None):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        with open(patterns_file, 'r') as f:
            self.patterns_config = json.load(f)

        # Initialiser le SMC Analyzer
        smc_config = self.patterns_config['smc']
        self.smc_analyzer = SMCAnalyzer(smc_config)

        self.mode = 'backtest' if is_backtest else 'realtime'
        self.data_folder = self.settings['data_folder']
        self.config_file = 'config.txt'

        # Dossiers différents selon le mode
        if self.mode == 'backtest':
            self.patterns_folder = "patterns_backtest"
        else:
            self.patterns_folder = "patterns_realtime"

        self.chart_symbol = chart_symbol

        print("settings_file", settings_file)
        print("is_backtest: ", is_backtest)
        print("mode: ", self.mode)
        print("folder: ", self.data_folder)
        print("patterns_folder: ", self.patterns_folder)


    def get_data_filename(self, symbol, candle_nb, interval, date):
        """Génère le nom du fichier de données"""
        return os.path.join(
            self.data_folder,
            f"{date}_{symbol}_{candle_nb}_{interval}.csv"
        )

    def check_file_exists(self, filepath):
        """Vérifie si le fichier existe"""
        return os.path.exists(filepath)

    def load_watchlist(self):
        """Charge les symboles depuis le fichier de configuration"""
        symbols = []
        with open(self.config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        symbol = parts[0]
                        provider = parts[1]
                        symbols.append({'symbol': symbol, 'provider': provider})
        return symbols


    def download_ibkr_data(self, symbol, candle_nb, interval):
        """Télécharge les données depuis IBKR"""
        try:
            # Connexion IBKR
            realtime_config = self.settings['realtime']
            host = realtime_config['ibkr_host']
            port = realtime_config['ibkr_port']
            client_id = realtime_config['ibkr_client_id']

            ib = IB()
            ib.connect(host, port, clientId=client_id)

            # Créer le contrat
            contract = Stock(symbol, 'SMART', 'USD')
            qualified = ib.qualifyContracts(contract)

            if not qualified:
                ib.disconnect()
                return None

            contract = qualified[0]

            # Calculer la durée
            if interval == '1d':
                duration_str = f"{candle_nb} D"
                bar_size = "1 day"
            elif interval == '1h':
                duration_str = f"{candle_nb} S"
                bar_size = "1 hour"
            else:
                duration_str = f"{candle_nb} D"
                bar_size = "1 day"

            # Télécharger les données
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                timeout=10
            )

            ib.disconnect()

            if not bars or len(bars) < candle_nb:
                return None

            # Convertir en DataFrame avec conversion timezone EST
            data = []
            est_tz = ZoneInfo('America/New_York')

            for bar in bars[-candle_nb:]:
                # Convertir la date en EST si elle a une timezone, sinon assumer qu'elle est déjà en EST
                if hasattr(bar.date, 'tzinfo') and bar.date.tzinfo is not None:
                    date_est = bar.date.astimezone(est_tz)
                else:
                    date_est = bar.date

                data.append({
                    'Date': date_est.strftime('%Y-%m-%d'),
                    'Open': bar.open,
                    'High': bar.high,
                    'Low': bar.low,
                    'Close': bar.close,
                    'Volume': bar.volume
                })

            df = pd.DataFrame(data)
            return df

        except Exception as e:
            print(f"Erreur IBKR: {e}")
            return None


    def run_backtest(self):
        """Execute le mode backtest"""
        backtest_config = self.settings['backtest']
        candle_nb = backtest_config['candle_nb']
        interval = backtest_config['interval']
        test_start = backtest_config['test_candle_start']
        test_stop = backtest_config['test_candle_stop']

        # Calculer le nombre total de bougies à charger
        total_candles_needed = candle_nb + test_stop

        # Créer les dossiers data et patterns_backtest s'ils n'existent pas
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.patterns_folder, exist_ok=True)

        watchlist = self.load_watchlist()

        # Filtrer la watchlist si --chart est spécifié
        if self.chart_symbol:
            watchlist = [item for item in watchlist if item['symbol'].upper() == self.chart_symbol.upper()]
            if not watchlist:
                print(f"Erreur: Symbole {self.chart_symbol} non trouvé dans la watchlist")
                return

        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
        print(today)

        for item in watchlist:
            symbol = item['symbol']
            filename = self.get_data_filename(symbol, total_candles_needed, interval, today)
            print(filename)

            # Télécharge ou charge les données
            if self.check_file_exists(filename):
                df = pd.read_csv(filename)
            else:
                df = self.download_ibkr_data(symbol, total_candles_needed, interval)
                if df is not None:
                    df.to_csv(filename, index=False)
                else:
                    continue

            if df is None or len(df) == 0:
                continue

            total_candles = len(df)

            print(f"\n{'='*60}")
            print(f"ANALYSE SMC pour {symbol}")
            print(f"{'='*60}")
            print(f"Total candles: {total_candles}")
            print(f"Test range: candle {test_start} à {test_stop}")

            # ANALYSE SMC sur l'ensemble du dataset
            smc_result = self.smc_analyzer.analyze(df)

            # DETECTION DES ALERTES DE TRADING
            alerts = self.smc_analyzer.detect_setups(df, smc_result)

            # AFFICHAGE DES ALERTES
            print(f"\n{'*'*60}")
            print(f"   ALERTES DE TRADING")
            print(f"{'*'*60}")

            if alerts:
                for i, alert in enumerate(alerts, 1):
                    print(f"\n🔔 ALERTE #{i}: {alert['type']}")
                    print(f"   Raison: {alert['reason']}")
                    print(f"   Entry: ${alert['entry']:.2f}")
                    print(f"   Stop Loss: ${alert['stop']:.2f}")
                    print(f"   Target: ${alert['target']:.2f}")
                    print(f"   Risk/Reward: 1:{alert['risk_reward']}")
            else:
                print("\nAucune alerte détectée pour le moment.")

            # Prix actuel et zones
            pd_zones = smc_result['premium_discount']
            print(f"\n--- PRIX ACTUEL ---")
            print(f"Prix: ${pd_zones['current_price']:.2f}")
            print(f"Zone: {pd_zones['current_zone'].upper()}")
            print(f"  Range Low: ${pd_zones['range_low']:.2f}")
            print(f"  Discount: ${pd_zones['discount']:.2f}")
            print(f"  Equilibrium: ${pd_zones['equilibrium']:.2f}")
            print(f"  Premium: ${pd_zones['premium']:.2f}")
            print(f"  Range High: ${pd_zones['range_high']:.2f}")

            # Afficher les résultats SMC (résumé)
            print(f"\n--- RÉSUMÉ SMC ---")
            print(f"Order Blocks: {len(smc_result['order_blocks']['bullish'])} bullish, {len(smc_result['order_blocks']['bearish'])} bearish")
            print(f"Fair Value Gaps: {len(smc_result['fvg']['bullish'])} bullish, {len(smc_result['fvg']['bearish'])} bearish")
            print(f"Break of Structure: {len(smc_result['bos'])} détectés")
            print(f"Change of Character: {len(smc_result['choch'])} détectés")

            if smc_result['liquidity']['strong_highs']:
                strong_highs = [f"${h['price']:.2f}" for h in smc_result['liquidity']['strong_highs'][:3]]
                print(f"Strong Highs: {strong_highs}")
            if smc_result['liquidity']['strong_lows']:
                strong_lows = [f"${l['price']:.2f}" for l in smc_result['liquidity']['strong_lows'][:3]]
                print(f"Strong Lows: {strong_lows}")


    def run(self):
        """Point d'entrée principal"""
        if self.mode == 'backtest':
            self.run_backtest()
        elif self.mode == 'realtime':
            print("Mode realtime pas encore implémenté")
        else:
            print(f"Mode inconnu: {self.mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scanner de stocks avec Smart Money Concepts')
    parser.add_argument('--backtest', action='store_true', help='Lance en mode backtest')
    parser.add_argument('--chart', type=str, metavar='SYMBOL', help='Génère un graphique HTML pour le symbole spécifié (ex: --chart AAPL)')
    args = parser.parse_args()

    scanner = StockScanner('settings.json', is_backtest=args.backtest, chart_symbol=args.chart)
    scanner.run()
