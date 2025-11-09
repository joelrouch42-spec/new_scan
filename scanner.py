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


class StockScanner:
    def __init__(self, settings_file, is_backtest=False):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        self.mode = 'backtest' if is_backtest else 'realtime'
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

    def get_data_filename(self, symbol, candle_nb, interval, date):
        """Génère le nom du fichier de données"""
        return os.path.join(
            self.data_folder,
            f"{date}_{symbol}_{candle_nb}_{interval}.csv"
        )

    def check_file_exists(self, filepath):
        """Vérifie si le fichier existe"""
        return os.path.exists(filepath)

    def download_yahoo_data(self, symbol, candle_nb, interval):
        """Télécharge les données depuis Yahoo Finance"""
        try:
            print(f"Téléchargement des données pour {symbol}...")
            ticker = yf.Ticker(symbol)

            # Calculer la période nécessaire avec marge de sécurité
            if interval == '1d':
                days_needed = int(candle_nb * 1.6)  # Marge pour weekends/jours fériés
            elif interval == '1h':
                days_needed = int(candle_nb / 6.5)  # ~6.5h de trading par jour
            elif interval == '1wk':
                days_needed = candle_nb * 7 * 2
            else:
                days_needed = candle_nb * 2  # Marge par défaut

            end_date = datetime.now(ZoneInfo('America/New_York'))
            start_date = end_date - timedelta(days=days_needed)

            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                print(f"Aucune donnée disponible pour {symbol}")
                return None

            # Prendre les N dernières bougies
            df = df.tail(candle_nb)

            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            return df
        except Exception as e:
            print(f"Erreur lors du téléchargement de {symbol}: {e}")
            return None

    def run_backtest(self):
        """Execute le mode backtest"""
        backtest_config = self.settings['backtest']
        candle_nb = backtest_config['candle_nb']
        interval = backtest_config['interval']

        # Créer le dossier data s'il n'existe pas
        os.makedirs(self.data_folder, exist_ok=True)

        watchlist = self.load_watchlist()
        print(f"Mode: {self.mode}")
        print(f"Nombre de bougies: {candle_nb}")
        print(f"Interval: {interval}")
        print(f"Nombre de symboles: {len(watchlist)}\n")

        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

        for item in watchlist:
            symbol = item['symbol']
            filename = self.get_data_filename(symbol, candle_nb, interval, today)

            if self.check_file_exists(filename):
                print(f"Skip {symbol} - fichier déjà existant")
                continue

            df = self.download_yahoo_data(symbol, candle_nb, interval)
            if df is not None:
                df.to_csv(filename, index=False)
                print(f"Données sauvegardées: {filename}\n")

    def connect_ibkr(self):
        """Connecte à Interactive Brokers"""
        realtime_config = self.settings['realtime']
        host = realtime_config['ibkr_host']
        port = realtime_config['ibkr_port']
        client_id = realtime_config['ibkr_client_id']

        try:
            ib = IB()
            ib.connect(host, port, clientId=client_id)
            print(f"Connecté à IBKR {host}:{port}")
            return ib
        except Exception as e:
            print(f"Erreur connexion IBKR: {e}")
            return None

    def get_last_close_ibkr(self, ib, symbol):
        """Récupère le close de la dernière bougie depuis IBKR"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            qualified = ib.qualifyContracts(contract)

            if not qualified:
                print(f"Contrat non trouvé pour {symbol}")
                return None

            contract = qualified[0]

            # Demande la dernière bougie daily
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='2 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                timeout=5
            )

            if not bars:
                print(f"Aucune donnée pour {symbol}")
                return None

            # Dernière bougie close
            last_close = bars[-1].close
            return last_close

        except Exception as e:
            print(f"Erreur récupération {symbol}: {e}")
            return None

    def run_realtime(self):
        """Execute le mode temps réel"""
        realtime_config = self.settings['realtime']
        update_interval = realtime_config['update_interval_seconds']

        watchlist = self.load_watchlist()
        print(f"Mode: {self.mode}")
        print(f"Interval de mise à jour: {update_interval}s")
        print(f"Nombre de symboles: {len(watchlist)}\n")

        # Connexion IBKR
        ib = self.connect_ibkr()
        if not ib:
            print("Impossible de se connecter à IBKR. Arrêt.")
            return

        try:
            while True:
                print(f"\n[{datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}] Scan en cours...")

                for item in watchlist:
                    symbol = item['symbol']
                    last_close = self.get_last_close_ibkr(ib, symbol)

                    if last_close:
                        print(f"{symbol}: ${last_close:.2f}")

                print(f"\nProchaine mise à jour dans {update_interval}s...")
                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\nArrêt du scanner temps réel")
        finally:
            if ib and ib.isConnected():
                ib.disconnect()
                print("Déconnecté d'IBKR")

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
    args = parser.parse_args()

    scanner = StockScanner('settings.json', is_backtest=args.backtest)
    scanner.run()
