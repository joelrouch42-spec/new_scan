#!/usr/bin/env python3
import json
import os
from datetime import datetime
import yfinance as yf
import pandas as pd


class StockScanner:
    def __init__(self, settings_file):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        self.mode = self.settings['mode']
        self.data_folder = self.settings['data_folder']
        self.config_file = self.settings['config_file']

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

    def get_data_filename(self, symbol, start_date, end_date):
        """Génère le nom du fichier de données"""
        return os.path.join(
            self.data_folder,
            f"{symbol}_{start_date}_{end_date}.csv"
        )

    def check_file_date(self, filepath, expected_start_date):
        """Vérifie si le fichier existe et si la date correspond"""
        if not os.path.exists(filepath):
            return False

        try:
            df = pd.read_csv(filepath, nrows=1)
            if len(df) > 0 and 'Date' in df.columns:
                file_start_date = df['Date'].iloc[0]
                return file_start_date == expected_start_date
        except Exception as e:
            print(f"Erreur lors de la vérification du fichier {filepath}: {e}")
            return False

        return False

    def download_yahoo_data(self, symbol, start_date, end_date):
        """Télécharge les données depuis Yahoo Finance"""
        try:
            print(f"Téléchargement des données pour {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                print(f"Aucune donnée disponible pour {symbol}")
                return None

            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            return df
        except Exception as e:
            print(f"Erreur lors du téléchargement de {symbol}: {e}")
            return None

    def run_backtest(self):
        """Execute le mode backtest"""
        backtest_config = self.settings['backtest']
        start_date = backtest_config['start_date']
        end_date = backtest_config['end_date']

        watchlist = self.load_watchlist()
        print(f"Mode: {self.mode}")
        print(f"Période: {start_date} à {end_date}")
        print(f"Nombre de symboles: {len(watchlist)}\n")

        for item in watchlist:
            symbol = item['symbol']
            filename = self.get_data_filename(symbol, start_date, end_date)

            if self.check_file_date(filename, start_date):
                print(f"Skip {symbol} - fichier déjà existant avec la bonne date")
                continue

            df = self.download_yahoo_data(symbol, start_date, end_date)
            if df is not None:
                df.to_csv(filename, index=False)
                print(f"Données sauvegardées: {filename}\n")

    def run_realtime(self):
        """Execute le mode temps réel"""
        print("Mode temps réel - à implémenter")
        pass

    def run(self):
        """Point d'entrée principal"""
        if self.mode == 'backtest':
            self.run_backtest()
        elif self.mode == 'realtime':
            self.run_realtime()
        else:
            print(f"Mode inconnu: {self.mode}")


if __name__ == '__main__':
    scanner = StockScanner('settings.json')
    scanner.run()
