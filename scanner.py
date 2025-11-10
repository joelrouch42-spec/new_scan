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


class StockScanner:
    def __init__(self, settings_file, is_backtest=False, patterns_file='patterns.json'):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        with open(patterns_file, 'r') as f:
            self.patterns_config = json.load(f)

        self.mode = 'backtest' if is_backtest else 'realtime'
        self.data_folder = self.settings['data_folder']
        self.config_file = 'config.txt'
        self.patterns_folder = self.patterns_config['output']['patterns_folder']

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

    def find_support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Trouve les niveaux de support et résistance"""
        sr_config = self.patterns_config['support_resistance']
        order = sr_config['order']
        cluster_threshold = sr_config['cluster_threshold']

        highs = df['High'].values
        lows = df['Low'].values
        n = len(df)

        if order < 1:
            order = 1
        if n <= (2 * order):
            order = max(1, (n - 1) // 2)

        resistance_idx = argrelextrema(highs, np.greater, order=order)[0]
        support_idx = argrelextrema(lows, np.less, order=order)[0]

        resistance_levels = highs[resistance_idx] if resistance_idx.size else np.array([])
        support_levels = lows[support_idx] if support_idx.size else np.array([])

        def cluster_levels(levels: np.ndarray) -> List[float]:
            if len(levels) == 0:
                return []
            levels_sorted = sorted(levels)
            clusters: List[float] = []
            current_cluster = [levels_sorted[0]]
            for level in levels_sorted[1:]:
                denom = current_cluster[-1] if current_cluster[-1] != 0 else 1.0
                if abs(level - current_cluster[-1]) / denom < cluster_threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(float(np.mean(current_cluster)))
                    current_cluster = [level]
            clusters.append(float(np.mean(current_cluster)))
            return clusters

        support_clusters = cluster_levels(support_levels)
        resistance_clusters = cluster_levels(resistance_levels)

        return support_clusters, resistance_clusters

    def detect_breakouts(self, df: pd.DataFrame, support_levels: List[float], resistance_levels: List[float]) -> Optional[Dict]:
        """Détecte les breakouts de support/résistance"""
        if len(df) < 2:
            return None

        last_idx = len(df) - 1
        current_high = float(df['High'].iloc[last_idx])
        current_low = float(df['Low'].iloc[last_idx])
        prev_close = float(df['Close'].iloc[last_idx - 1])
        current_close = float(df['Close'].iloc[last_idx])

        # Détection breakout résistance (vers le haut)
        for resistance in resistance_levels:
            if prev_close < resistance and current_high > resistance:
                return {
                    'type': 'resistance_breakout',
                    'level': resistance,
                    'close': current_close,
                    'timestamp': df.index[last_idx] if hasattr(df.index[last_idx], 'strftime') else str(df.index[last_idx])
                }

        # Détection breakdown support (vers le bas)
        for support in support_levels:
            if prev_close > support and current_low < support:
                return {
                    'type': 'support_breakdown',
                    'level': support,
                    'close': current_close,
                    'timestamp': df.index[last_idx] if hasattr(df.index[last_idx], 'strftime') else str(df.index[last_idx])
                }

        return None

    def save_sr_levels(self, symbol: str, support_levels: List[float], resistance_levels: List[float], date: str):
        """Sauvegarde les niveaux S/R pour un symbole"""
        os.makedirs(self.patterns_folder, exist_ok=True)

        filename = os.path.join(self.patterns_folder, f"{date}_{symbol}_sr.json")

        data = {
            'symbol': symbol,
            'date': date,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'updated': datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_sr_levels(self, symbol: str, date: str) -> Tuple[List[float], List[float]]:
        """Charge les niveaux S/R depuis le fichier"""
        filename = os.path.join(self.patterns_folder, f"{date}_{symbol}_sr.json")

        if not os.path.exists(filename):
            return [], []

        with open(filename, 'r') as f:
            data = json.load(f)

        return data.get('support_levels', []), data.get('resistance_levels', [])

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

            # Télécharge ou charge les données
            if self.check_file_exists(filename):
                print(f"Chargement {symbol} depuis fichier existant")
                df = pd.read_csv(filename)
            else:
                df = self.download_yahoo_data(symbol, candle_nb, interval)
                if df is not None:
                    df.to_csv(filename, index=False)
                    print(f"Données sauvegardées: {filename}")
                else:
                    continue

            # Analyse des patterns
            if df is not None and len(df) > 0:
                # Détecte les S/R
                support_levels, resistance_levels = self.find_support_resistance(df)
                print(f"{symbol}: {len(support_levels)} supports, {len(resistance_levels)} résistances")

                # Sauvegarde TOUJOURS les S/R avec la date
                self.save_sr_levels(symbol, support_levels, resistance_levels, today)

                # Détecte les breakouts
                breakout = self.detect_breakouts(df, support_levels, resistance_levels)
                if breakout:
                    print(f"  BREAKOUT: {breakout['type']} à {breakout['level']:.2f}")
                else:
                    print(f"  Pas de breakout")

            print()

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

    def get_last_bars_ibkr(self, ib, symbol):
        """Récupère les 2 dernières bougies depuis IBKR"""
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            qualified = ib.qualifyContracts(contract)

            if not qualified:
                print(f"Contrat non trouvé pour {symbol}")
                return None

            contract = qualified[0]

            # Demande les 2 dernières bougies daily
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

            if not bars or len(bars) < 2:
                print(f"Pas assez de données pour {symbol}")
                return None

            # Retourne les 2 dernières bougies
            return {
                'prev_close': bars[-2].close,
                'current_high': bars[-1].high,
                'current_low': bars[-1].low,
                'current_close': bars[-1].close
            }

        except Exception as e:
            print(f"Erreur récupération {symbol}: {e}")
            return None

    def check_realtime_breakout(self, bars_data: Dict, support_levels: List[float], resistance_levels: List[float]) -> Optional[Dict]:
        """Vérifie si un breakout est en cours avec les données temps réel"""
        prev_close = bars_data['prev_close']
        current_high = bars_data['current_high']
        current_low = bars_data['current_low']
        current_close = bars_data['current_close']

        # Détection breakout résistance (vers le haut)
        for resistance in resistance_levels:
            if prev_close < resistance and current_high > resistance:
                return {
                    'type': 'resistance_breakout',
                    'level': resistance,
                    'close': current_close,
                    'timestamp': datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
                }

        # Détection breakdown support (vers le bas)
        for support in support_levels:
            if prev_close > support and current_low < support:
                return {
                    'type': 'support_breakdown',
                    'level': support,
                    'close': current_close,
                    'timestamp': datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
                }

        return None

    def run_realtime(self):
        """Execute le mode temps réel"""
        realtime_config = self.settings['realtime']
        update_interval = realtime_config['update_interval_seconds']

        watchlist = self.load_watchlist()
        print(f"Mode: {self.mode}")
        print(f"Interval de mise à jour: {update_interval}s")
        print(f"Nombre de symboles: {len(watchlist)}\n")

        # Charge les S/R UNE FOIS à l'initialisation
        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
        print(f"Chargement des S/R pour la date: {today}\n")

        sr_data = {}
        for item in watchlist:
            symbol = item['symbol']
            support_levels, resistance_levels = self.load_sr_levels(symbol, today)

            if not support_levels and not resistance_levels:
                print(f"{symbol}: Pas de S/R pour {today} - lancer --backtest d'abord")
                sr_data[symbol] = None
            else:
                sr_data[symbol] = {
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels
                }
                print(f"{symbol}: {len(support_levels)} supports, {len(resistance_levels)} résistances chargés")

        print()

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

                    # Utilise les S/R chargés en mémoire
                    if sr_data[symbol] is None:
                        continue

                    support_levels = sr_data[symbol]['support_levels']
                    resistance_levels = sr_data[symbol]['resistance_levels']

                    # Récupère les données IBKR
                    bars_data = self.get_last_bars_ibkr(ib, symbol)

                    if not bars_data:
                        continue

                    # Affiche le prix
                    print(f"{symbol}: ${bars_data['current_close']:.2f} (S:{len(support_levels)} R:{len(resistance_levels)})")

                    # Vérifie breakout
                    breakout = self.check_realtime_breakout(bars_data, support_levels, resistance_levels)
                    if breakout:
                        print(f"  BREAKOUT: {breakout['type']} à {breakout['level']:.2f}")

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
