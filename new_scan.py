#!/usr/bin/env python3
import json
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from zoneinfo import ZoneInfo
from ib_insync import IB, Stock
import time
import argparse
import numpy as np
from scipy.signal import argrelextrema
from typing import List, Tuple, Optional, Dict
import plotly.graph_objects as go
from sr_analyzer import SRAnalyzer

class StockScanner:
    def __init__(self, settings_file, is_backtest=False, patterns_file='patterns.json', chart_symbol=None):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        with open(patterns_file, 'r') as f:
            self.patterns_config = json.load(f)
            
        sr_config = self.patterns_config['support_resistance']
        self.sr_analyzer = SRAnalyzer(sr_config)

        self.mode = 'backtest' if is_backtest else 'realtime'
        self.data_folder = self.settings['data_folder']
        self.config_file = 'config.txt'

        # Dossiers différents selon le mode
        base_patterns_folder = self.patterns_config['support_resistance']['patterns_folder']
        if self.mode == 'backtest':
            self.patterns_folder = f"{base_patterns_folder}_backtest"
        else:
            self.patterns_folder = f"{base_patterns_folder}_realtime"

        self.chart_symbol = chart_symbol
        
        print("settings_file", settings_file);
        print("is_backtest: ", is_backtest)
        print("mode: ", self.mode)
        print("folder: ", self.data_folder)
        print("patterns_folder: ", self.patterns_folder);
        
        
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
                duration_str = f"{candle_nb} S"  # S pour secondes (heures)
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
                # Fallback sur Yahoo si pas assez de données
                return none

            # Convertir en DataFrame avec conversion timezone EST
            data = []
            est_tz = ZoneInfo('America/New_York')

            for bar in bars[-candle_nb:]:
                # Convertir la date en EST si elle a une timezone, sinon assumer qu'elle est déjà en EST
                if hasattr(bar.date, 'tzinfo') and bar.date.tzinfo is not None:
                    date_est = bar.date.astimezone(est_tz)
                else:
                    # Si pas de timezone, on assume que c'est déjà en EST
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
            return nono
            
    def find_support_resistance(self, df: pd.DataFrame, filter_high: float = None, filter_low: float = None) -> Dict:
        return self.sr_analyzer.find_levels(df, filter_high, filter_low)

        
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
            print (filename)

            # Télécharge ou charge les données
            if self.check_file_exists(filename):
                df = pd.read_csv(filename)
            else:
                # TEMPORAIRE: Utilise IBKR au lieu de Yahoo pour éviter les gaps
                df = self.download_ibkr_data(symbol, total_candles_needed, interval)
                if df is not None:
                    df.to_csv(filename, index=False)
                else:
                    continue
            
            if df is None or len(df) == 0:
                continue

            total_candles = len(df)
            
            # Liste pour stocker les patterns détectés (pour le graphique)
            detected_patterns = []
            
            print("total_candles_needed", total_candles_needed)
            print("total_candles", total_candles)
            print("test_stop", test_stop)
            
            idx = 0
            for candle_nb in range(test_stop, test_start - 1, -1):   # 10.9.8. ... 1
                sr_calc_pos = total_candles - candle_nb

                if sr_calc_pos >= total_candles:
                    break

                df_for_sr = df.iloc[idx:sr_calc_pos].copy()
                idx = idx + 1

                # Récupérer le MAX high et MIN low de TOUTES les bougies
                # depuis le calcul S/R jusqu'à la fin du dataset
                max_high = df['High'].iloc[sr_calc_pos:].max()
                min_low = df['Low'].iloc[sr_calc_pos:].min()

                # Calcul S/R avec filtrage basé sur le max/min depuis le calcul
                sr_result = self.find_support_resistance(
                    df_for_sr,
                    filter_high=max_high,
                    filter_low=min_low
                )

                # Extraire les niveaux valides et cassés
                valid_supports = sr_result['valid']['supports']
                valid_resistances = sr_result['valid']['resistances']
                broken_supports = sr_result['broken']['supports']
                broken_resistances = sr_result['broken']['resistances']

                print("Valid supports:", valid_supports)
                print("Valid resistances:", valid_resistances)
                print("Broken supports:", broken_supports)
                print("Broken resistances:", broken_resistances)

        
    def run(self):
        """Point d'entrée principal"""
        if self.mode == 'backtest':
            self.run_backtest()
        elif self.mode == 'realtime':
            self.run_realtime()
        else:
            print(f"Mode inconnu: {self.mode}")
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scanner de stocks')
    parser.add_argument('--backtest', action='store_true', help='Lance en mode backtest')
    parser.add_argument('--chart', type=str, metavar='SYMBOL', help='Génère un graphique HTML pour le symbole spécifié (ex: --chart AAPL)')
    args = parser.parse_args()

    scanner = StockScanner('settings.json', is_backtest=args.backtest, chart_symbol=args.chart)   
    scanner.run()
    
    
    
