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
        self.patterns_folder = self.patterns_config['support_resistance']['patterns_folder']

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

    def is_pattern_enabled(self, pattern_name: str) -> bool:
        """Vérifie si un pattern est activé dans la configuration"""
        for pattern in self.patterns_config.get('patterns', []):
            if pattern['name'] == pattern_name:
                return pattern.get('enabled', False)
        return False

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

    def detect_breakouts(self, df: pd.DataFrame, support_levels: List[float], resistance_levels: List[float], last_breakout_direction: Optional[str] = None, symbol: str = None) -> Optional[Dict]:
        """Détecte les breakouts de support/résistance

        Args:
            last_breakout_direction: 'up' si dernier breakout était résistance, 'down' si c'était support, None si aucun
        """
        if len(df) < 2:
            return None

        last_idx = len(df) - 1
        current_high = float(df['High'].iloc[last_idx])
        current_low = float(df['Low'].iloc[last_idx])
        prev_close = float(df['Close'].iloc[last_idx - 1])
        current_close = float(df['Close'].iloc[last_idx])

        # Détection breakout résistance (vers le haut) - seulement si dernier mouvement n'était pas vers le haut
        if last_breakout_direction != 'up':
            for resistance in resistance_levels:
                if prev_close < resistance and current_high > resistance:
                    return {
                        'type': 'resistance_breakout',
                        'level': resistance,
                        'close': current_close,
                        'direction': 'up',
                        'timestamp': df.index[last_idx] if hasattr(df.index[last_idx], 'strftime') else str(df.index[last_idx])
                    }

        # Détection breakdown support (vers le bas) - seulement si dernier mouvement n'était pas vers le bas
        if last_breakout_direction != 'down':
            for support in support_levels:
                if prev_close > support and current_low < support:
                    return {
                        'type': 'support_breakdown',
                        'level': support,
                        'close': current_close,
                        'direction': 'down',
                        'timestamp': df.index[last_idx] if hasattr(df.index[last_idx], 'strftime') else str(df.index[last_idx])
                    }

        return None

    def detect_flips(self, df: pd.DataFrame, breakout_history: List[Dict], symbol: str = None) -> Optional[Dict]:
        """Détecte les role reversals (flip) - quand un ancien support devient résistance ou vice versa"""
        if len(df) < 2 or not breakout_history:
            return None

        flip_tolerance = self.patterns_config['support_resistance']['flip_tolerance']

        last_idx = len(df) - 1
        current_high = float(df['High'].iloc[last_idx])
        current_low = float(df['Low'].iloc[last_idx])
        prev_close = float(df['Close'].iloc[last_idx - 1])
        current_close = float(df['Close'].iloc[last_idx])

        # Parcourt l'historique des breakouts pour détecter les flips
        for breakout in breakout_history:
            level = breakout['level']
            original_type = breakout['original_type']

            # Vérifie si on est proche du niveau (dans la tolérance)
            tolerance_range = level * flip_tolerance

            # Cas 1: Ancien resistance → nouveau support
            # Après un breakout de résistance, le prix revient tester le niveau
            if original_type == 'resistance':
                # Le LOW de la bougie touche le niveau (retest par le haut)
                if (current_low <= level + tolerance_range and
                    current_low >= level - tolerance_range):
                    return {
                        'type': 'flip_resistance_to_support',
                        'level': level,
                        'original_type': 'resistance',
                        'new_type': 'support',
                        'close': current_close
                    }

            # Cas 2: Ancien support → nouveau resistance
            # Après un breakdown de support, le prix revient tester le niveau
            elif original_type == 'support':
                # Le HIGH de la bougie touche le niveau (retest par le bas)
                if (current_high >= level - tolerance_range and
                    current_high <= level + tolerance_range):
                    return {
                        'type': 'flip_support_to_resistance',
                        'level': level,
                        'original_type': 'support',
                        'new_type': 'resistance',
                        'close': current_close
                    }

        return None

    def save_sr_levels(self, symbol: str, support_levels: List[float], resistance_levels: List[float], date: str, breakout_history: Optional[List[Dict]] = None, last_breakout_direction: Optional[str] = None):
        """Sauvegarde les niveaux S/R pour un symbole"""
        os.makedirs(self.patterns_folder, exist_ok=True)

        filename = os.path.join(self.patterns_folder, f"{date}_{symbol}_sr.json")

        # Charge l'historique existant si présent
        existing_history = []
        existing_direction = None
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
                    existing_history = existing_data.get('breakout_history', [])
                    existing_direction = existing_data.get('last_breakout_direction')
            except:
                pass

        # Utilise l'historique fourni ou conserve l'existant
        if breakout_history is not None:
            final_history = breakout_history
        else:
            final_history = existing_history

        # Utilise la direction fournie ou conserve l'existante
        if last_breakout_direction is not None:
            final_direction = last_breakout_direction
        else:
            final_direction = existing_direction

        data = {
            'symbol': symbol,
            'date': date,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'breakout_history': final_history,
            'last_breakout_direction': final_direction,
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

    def load_breakout_history(self, symbol: str, date: str) -> List[Dict]:
        """Charge l'historique des breakouts depuis le fichier"""
        filename = os.path.join(self.patterns_folder, f"{date}_{symbol}_sr.json")

        if not os.path.exists(filename):
            return []

        with open(filename, 'r') as f:
            data = json.load(f)

        return data.get('breakout_history', [])

    def load_last_breakout_direction(self, symbol: str, date: str) -> Optional[str]:
        """Charge la direction du dernier breakout depuis le fichier"""
        filename = os.path.join(self.patterns_folder, f"{date}_{symbol}_sr.json")

        if not os.path.exists(filename):
            return None

        with open(filename, 'r') as f:
            data = json.load(f)

        return data.get('last_breakout_direction')

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
                return self.download_yahoo_data(symbol, candle_nb, interval)

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
            return self.download_yahoo_data(symbol, candle_nb, interval)

    def download_yahoo_data(self, symbol, candle_nb, interval):
        """Télécharge les données depuis Yahoo Finance"""
        try:
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
        test_start = backtest_config['test_candle_start']
        test_stop = backtest_config['test_candle_stop']

        # Calculer le nombre total de bougies à charger
        total_candles_needed = candle_nb + test_stop

        # Créer le dossier data s'il n'existe pas
        os.makedirs(self.data_folder, exist_ok=True)

        watchlist = self.load_watchlist()
        print(f"Mode: {self.mode}")
        print(f"Nombre de bougies pour S/R: {candle_nb}")
        print(f"Nombre total de bougies chargées: {total_candles_needed}")
        print(f"Interval: {interval}")
        print(f"Test range: {test_start} à {test_stop}")
        print(f"Nombre de symboles: {len(watchlist)}\n")

        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

        for item in watchlist:
            symbol = item['symbol']
            filename = self.get_data_filename(symbol, total_candles_needed, interval, today)

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

            # Initialise l'historique des breakouts pour ce symbole
            breakout_history = []
            last_breakout_direction = None  # 'up', 'down', ou None

            # Boucle de test: du passé vers le présent (de test_stop vers test_start)
            for candle_nb in range(test_stop, test_start - 1, -1):
                # Position dans le df: on enlève les N dernières bougies
                current_pos = total_candles - candle_nb

                if current_pos <= 0:
                    break

                # Prend les données jusqu'à cette position (exclut les N dernières bougies)
                df_until_pos = df.iloc[:current_pos].copy()

                # Calcule S/R sur les données jusqu'à cette position
                support_levels, resistance_levels = self.find_support_resistance(df_until_pos)

                # Détecte les flips (role reversals) si activé
                if self.is_pattern_enabled('flips'):
                    flip = self.detect_flips(df_until_pos, breakout_history, symbol)
                    if flip:
                        last_row = df_until_pos.iloc[-1]
                        date_str = last_row['Date'] if 'Date' in df_until_pos.columns else ''
                        print(f"{symbol}: Bougie {candle_nb} ({date_str}): FLIP {flip['original_type']}→{flip['new_type']} à {flip['level']:.2f}")

                # Détecte breakout sur la dernière bougie de cette position si activé
                if self.is_pattern_enabled('breakouts'):
                    breakout = self.detect_breakouts(df_until_pos, support_levels, resistance_levels, last_breakout_direction, symbol)
                    if breakout:
                        last_row = df_until_pos.iloc[-1]
                        date_str = last_row['Date'] if 'Date' in df_until_pos.columns else ''
                        print(f"{symbol}: Bougie {candle_nb} ({date_str}): BREAKOUT {breakout['type']} à {breakout['level']:.2f}")

                        # Met à jour la direction du dernier breakout
                        last_breakout_direction = breakout['direction']

                        # Ajoute le breakout à l'historique
                        breakout_history.append({
                            'level': breakout['level'],
                            'original_type': 'resistance' if breakout['type'] == 'resistance_breakout' else 'support',
                            'breakout_candle': candle_nb,
                            'breakout_date': today
                        })

                # Sauvegarde les S/R pour la première bougie testée (la plus récente) avec l'historique
                if candle_nb == test_start:
                    self.save_sr_levels(symbol, support_levels, resistance_levels, today, breakout_history, last_breakout_direction)

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
            # Convertir la date en EST
            est_tz = ZoneInfo('America/New_York')
            current_date = bars[-1].date
            if hasattr(current_date, 'tzinfo') and current_date.tzinfo is not None:
                date_est = current_date.astimezone(est_tz)
            else:
                date_est = current_date

            return {
                'prev_close': bars[-2].close,
                'current_high': bars[-1].high,
                'current_low': bars[-1].low,
                'current_close': bars[-1].close,
                'current_date': date_est.strftime('%Y-%m-%d')
            }

        except Exception as e:
            print(f"Erreur récupération {symbol}: {e}")
            return None

    def check_realtime_breakout(self, bars_data: Dict, support_levels: List[float], resistance_levels: List[float], last_breakout_direction: Optional[str] = None) -> Optional[Dict]:
        """Vérifie si un breakout est en cours avec les données temps réel

        Args:
            last_breakout_direction: 'up' si dernier breakout était résistance, 'down' si c'était support, None si aucun
        """
        prev_close = bars_data['prev_close']
        current_high = bars_data['current_high']
        current_low = bars_data['current_low']
        current_close = bars_data['current_close']

        # Détection breakout résistance (vers le haut) - seulement si dernier mouvement n'était pas vers le haut
        if last_breakout_direction != 'up':
            for resistance in resistance_levels:
                if prev_close < resistance and current_high > resistance:
                    return {
                        'type': 'resistance_breakout',
                        'level': resistance,
                        'close': current_close,
                        'direction': 'up',
                        'timestamp': datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
                    }

        # Détection breakdown support (vers le bas) - seulement si dernier mouvement n'était pas vers le bas
        if last_breakout_direction != 'down':
            for support in support_levels:
                if prev_close > support and current_low < support:
                    return {
                        'type': 'support_breakdown',
                        'level': support,
                        'close': current_close,
                        'direction': 'down',
                        'timestamp': datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
                    }

        return None

    def check_realtime_flip(self, bars_data: Dict, breakout_history: List[Dict]) -> Optional[Dict]:
        """Vérifie si un flip est en cours avec les données temps réel"""
        if not breakout_history:
            return None

        flip_tolerance = self.patterns_config['support_resistance']['flip_tolerance']

        prev_close = bars_data['prev_close']
        current_high = bars_data['current_high']
        current_low = bars_data['current_low']
        current_close = bars_data['current_close']

        # Parcourt l'historique des breakouts pour détecter les flips
        for breakout in breakout_history:
            level = breakout['level']
            original_type = breakout['original_type']

            # Vérifie si on est proche du niveau (dans la tolérance)
            tolerance_range = level * flip_tolerance

            # Cas 1: Ancien resistance → nouveau support
            # Après un breakout de résistance, le prix revient tester le niveau
            if original_type == 'resistance':
                # Le LOW de la bougie touche le niveau (retest par le haut)
                if (current_low <= level + tolerance_range and
                    current_low >= level - tolerance_range):
                    return {
                        'type': 'flip_resistance_to_support',
                        'level': level,
                        'original_type': 'resistance',
                        'new_type': 'support',
                        'close': current_close
                    }

            # Cas 2: Ancien support → nouveau resistance
            # Après un breakdown de support, le prix revient tester le niveau
            elif original_type == 'support':
                # Le HIGH de la bougie touche le niveau (retest par le bas)
                if (current_high >= level - tolerance_range and
                    current_high <= level + tolerance_range):
                    return {
                        'type': 'flip_support_to_resistance',
                        'level': level,
                        'original_type': 'support',
                        'new_type': 'resistance',
                        'close': current_close
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
            breakout_history = self.load_breakout_history(symbol, today)
            last_breakout_direction = self.load_last_breakout_direction(symbol, today)

            if not support_levels and not resistance_levels:
                print(f"{symbol}: Pas de S/R pour {today} - lancer --backtest d'abord")
                sr_data[symbol] = None
            else:
                sr_data[symbol] = {
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels,
                    'breakout_history': breakout_history,
                    'last_breakout_direction': last_breakout_direction
                }
                print(f"{symbol}: {len(support_levels)} supports, {len(resistance_levels)} résistances, {len(breakout_history)} breakouts chargés")

        print()

        # Connexion IBKR
        ib = self.connect_ibkr()
        if not ib:
            print("Impossible de se connecter à IBKR. Arrêt.")
            return

        try:
            while True:
                for item in watchlist:
                    symbol = item['symbol']

                    # Utilise les S/R chargés en mémoire
                    if sr_data[symbol] is None:
                        continue

                    support_levels = sr_data[symbol]['support_levels']
                    resistance_levels = sr_data[symbol]['resistance_levels']
                    breakout_history = sr_data[symbol]['breakout_history']
                    last_breakout_direction = sr_data[symbol]['last_breakout_direction']

                    # Récupère les données IBKR
                    bars_data = self.get_last_bars_ibkr(ib, symbol)

                    if not bars_data:
                        continue

                    # Vérifie flip si activé
                    if self.is_pattern_enabled('flips'):
                        flip = self.check_realtime_flip(bars_data, breakout_history)
                        if flip:
                            print(f"FLIP: {symbol} ({bars_data['current_date']}) ${bars_data['current_close']:.2f} {flip['original_type']}→{flip['new_type']} à {flip['level']:.2f}")

                    # Vérifie breakout si activé
                    if self.is_pattern_enabled('breakouts'):
                        breakout = self.check_realtime_breakout(bars_data, support_levels, resistance_levels, last_breakout_direction)
                        if breakout:
                            print(f"BREAKOUT: {symbol} ({bars_data['current_date']}) ${bars_data['current_close']:.2f} {breakout['type']} à {breakout['level']:.2f}")

                            # Met à jour la direction et sauvegarde
                            last_breakout_direction = breakout['direction']
                            sr_data[symbol]['last_breakout_direction'] = last_breakout_direction

                            # Ajoute le breakout à l'historique
                            breakout_history.append({
                                'level': breakout['level'],
                                'original_type': 'resistance' if breakout['type'] == 'resistance_breakout' else 'support',
                                'breakout_candle': 0,
                                'breakout_date': today
                            })

                            # Sauvegarde le nouveau statut
                            self.save_sr_levels(symbol, support_levels, resistance_levels, today, breakout_history, last_breakout_direction)

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
