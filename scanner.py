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


class StockScanner:
    def __init__(self, settings_file, is_backtest=False, patterns_file='patterns.json', chart_symbol=None):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        with open(patterns_file, 'r') as f:
            self.patterns_config = json.load(f)

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

    def should_print_pattern(self, pattern_name: str) -> bool:
        """Vérifie si les alertes d'un pattern doivent être affichées"""
        for pattern in self.patterns_config.get('patterns', []):
            if pattern['name'] == pattern_name:
                return pattern.get('print_alerts', True)  # Par défaut True pour compatibilité
        return True

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
        min_touches = sr_config.get('min_touches', 2)

        highs = df['High'].values
        lows = df['Low'].values
        n = len(df)

        original_order = order
        if order < 1:
            order = 1
        if n <= (2 * order):
            order = max(1, (n - 1) // 2)
            print(f"AVERTISSEMENT: Order ajusté de {original_order} à {order} pour n={n} bougies")

        resistance_idx = argrelextrema(highs, np.greater, order=order)[0]
        support_idx = argrelextrema(lows, np.less, order=order)[0]

        resistance_levels = highs[resistance_idx] if resistance_idx.size else np.array([])
        support_levels = lows[support_idx] if support_idx.size else np.array([])

        def cluster_levels(levels: np.ndarray, min_touches: int) -> List[float]:
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
                    # Ne garde le cluster que s'il a assez d'extrema
                    if len(current_cluster) >= min_touches:
                        clusters.append(float(np.mean(current_cluster)))
                    current_cluster = [level]
            # Dernier cluster
            if len(current_cluster) >= min_touches:
                clusters.append(float(np.mean(current_cluster)))
            return clusters

        support_clusters = cluster_levels(support_levels, min_touches)
        resistance_clusters = cluster_levels(resistance_levels, min_touches)

        return support_clusters, resistance_clusters

    def detect_breakouts(self, df: pd.DataFrame, support_levels: List[float], resistance_levels: List[float]) -> Optional[Dict]:
        """Détecte les breakouts de support/résistance avec logique stricte

        CRITÈRES STRICTS:
        - Le CLOSE doit casser le niveau (pas juste high/low avec mèche)
        - Le close précédent doit être de l'autre côté du niveau
        - Minimum 0.3% de marge au-delà du niveau pour confirmer la cassure
        """
        if len(df) < 2 or not resistance_levels or not support_levels:
            return None

        last_idx = len(df) - 1
        prev_close = float(df['Close'].iloc[last_idx - 1])
        current_close = float(df['Close'].iloc[last_idx])

        # Marge de cassure: 0.3% au-delà du niveau pour confirmer
        breakout_margin = 0.003

        # Détection breakout résistance (vers le haut)
        # Le CLOSE actuel doit être au-dessus de la résistance (pas juste le high)
        for resistance in resistance_levels:
            # Vérifier que le close précédent était en-dessous
            if prev_close < resistance:
                # Vérifier que le close actuel est au-dessus AVEC marge
                if current_close > resistance * (1 + breakout_margin):
                    return {
                        'type': 'resistance_breakout',
                        'level': resistance,
                        'close': current_close,
                        'direction': 'up'
                    }

        # Détection breakdown support (vers le bas)
        # Le CLOSE actuel doit être en-dessous du support (pas juste le low)
        for support in support_levels:
            # Vérifier que le close précédent était au-dessus
            if prev_close > support:
                # Vérifier que le close actuel est en-dessous AVEC marge
                if current_close < support * (1 - breakout_margin):
                    return {
                        'type': 'support_breakdown',
                        'level': support,
                        'close': current_close,
                        'direction': 'down'
                    }

        return None

    def detect_engulfing(self, df: pd.DataFrame, support_levels: List[float] = None, resistance_levels: List[float] = None) -> Optional[Dict]:
        """Détecte les Engulfing Patterns (Bullish et Bearish)

        Bullish Engulfing: bougie verte englobe complètement une bougie rouge précédente
        Bearish Engulfing: bougie rouge englobe complètement une bougie verte précédente

        Args:
            df: DataFrame avec les données OHLCV
            support_levels: Liste des niveaux de support (optionnel)
            resistance_levels: Liste des niveaux de résistance (optionnel)
        """
        if len(df) < 2:
            return None

        last_idx = len(df) - 1
        prev_idx = last_idx - 1

        # Bougie précédente
        prev_open = float(df['Open'].iloc[prev_idx])
        prev_close = float(df['Close'].iloc[prev_idx])
        prev_high = float(df['High'].iloc[prev_idx])
        prev_low = float(df['Low'].iloc[prev_idx])

        # Bougie actuelle
        current_open = float(df['Open'].iloc[last_idx])
        current_close = float(df['Close'].iloc[last_idx])
        current_high = float(df['High'].iloc[last_idx])
        current_low = float(df['Low'].iloc[last_idx])

        # Body sizes (pour éviter les bougies doji insignifiantes)
        prev_body = abs(prev_close - prev_open)
        current_body = abs(current_close - current_open)

        # Minimum body size (0.5% du prix pour éviter les doji)
        min_body_size = current_close * 0.005

        if prev_body < min_body_size or current_body < min_body_size:
            return None

        # Vérifier configuration S/R proximity
        engulfing_config = self.patterns_config.get('engulfing', {})
        require_sr_proximity = engulfing_config.get('require_sr_proximity', True)
        sr_tolerance_percent = engulfing_config.get('sr_tolerance_percent', 2) / 100.0

        # BULLISH ENGULFING
        # Bougie 1: bearish (close < open)
        # Bougie 2: bullish (close > open) et englobe complètement le body de bougie 1
        if prev_close < prev_open and current_close > current_open:
            if current_open <= prev_close and current_close >= prev_open:
                # Si require_sr_proximity est activé, vérifier qu'on est près d'un support
                if require_sr_proximity:
                    if not support_levels:
                        return None  # Pas de support disponible, rejet

                    near_support = False
                    nearest_level = None
                    for support in support_levels:
                        # Vérifier si le low de l'engulfing est proche du support
                        distance = abs(current_low - support) / support
                        if distance <= sr_tolerance_percent:
                            near_support = True
                            nearest_level = support
                            break

                    if not near_support:
                        return None  # Ne touche pas un support, rejet

                    return {
                        'type': 'bullish_engulfing',
                        'direction': 'up',
                        'price': current_close,
                        'sr_level': nearest_level,
                        'prev_candle': {'open': prev_open, 'close': prev_close, 'high': prev_high, 'low': prev_low},
                        'current_candle': {'open': current_open, 'close': current_close, 'high': current_high, 'low': current_low}
                    }
                else:
                    # Si le filtre S/R n'est pas requis, accepter le pattern
                    return {
                        'type': 'bullish_engulfing',
                        'direction': 'up',
                        'price': current_close,
                        'prev_candle': {'open': prev_open, 'close': prev_close, 'high': prev_high, 'low': prev_low},
                        'current_candle': {'open': current_open, 'close': current_close, 'high': current_high, 'low': current_low}
                    }

        # BEARISH ENGULFING
        # Bougie 1: bullish (close > open)
        # Bougie 2: bearish (close < open) et englobe complètement le body de bougie 1
        if prev_close > prev_open and current_close < current_open:
            if current_open >= prev_close and current_close <= prev_open:
                # Si require_sr_proximity est activé, vérifier qu'on est près d'une résistance
                if require_sr_proximity:
                    if not resistance_levels:
                        return None  # Pas de résistance disponible, rejet

                    near_resistance = False
                    nearest_level = None
                    for resistance in resistance_levels:
                        # Vérifier si le high de l'engulfing est proche de la résistance
                        distance = abs(current_high - resistance) / resistance
                        if distance <= sr_tolerance_percent:
                            near_resistance = True
                            nearest_level = resistance
                            break

                    if not near_resistance:
                        return None  # Ne touche pas une résistance, rejet

                    return {
                        'type': 'bearish_engulfing',
                        'direction': 'down',
                        'price': current_close,
                        'sr_level': nearest_level,
                        'prev_candle': {'open': prev_open, 'close': prev_close, 'high': prev_high, 'low': prev_low},
                        'current_candle': {'open': current_open, 'close': current_close, 'high': current_high, 'low': current_low}
                    }
                else:
                    # Si le filtre S/R n'est pas requis, accepter le pattern
                    return {
                        'type': 'bearish_engulfing',
                        'direction': 'down',
                        'price': current_close,
                        'prev_candle': {'open': prev_open, 'close': prev_close, 'high': prev_high, 'low': prev_low},
                        'current_candle': {'open': current_open, 'close': current_close, 'high': current_high, 'low': current_low}
                    }

        return None

    def detect_pinbar(self, df: pd.DataFrame, support_levels: List[float] = None, resistance_levels: List[float] = None) -> Optional[Dict]:
        """Détecte les Pin Bars / Rejection Candles (Hammer et Shooting Star)

        Bullish Pin Bar (Hammer): longue mèche basse + petit body en haut = rejet du support
        Bearish Pin Bar (Shooting Star): longue mèche haute + petit body en bas = rejet de la résistance

        Args:
            df: DataFrame avec les données OHLCV
            support_levels: Liste des niveaux de support (optionnel)
            resistance_levels: Liste des niveaux de résistance (optionnel)
        """
        if len(df) < 1:
            return None

        last_idx = len(df) - 1

        # Bougie actuelle
        open_price = float(df['Open'].iloc[last_idx])
        close_price = float(df['Close'].iloc[last_idx])
        high_price = float(df['High'].iloc[last_idx])
        low_price = float(df['Low'].iloc[last_idx])

        # Calcul du body et des mèches
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        body_size = abs(close_price - open_price)

        upper_wick = high_price - body_top
        lower_wick = body_bottom - low_price

        total_range = high_price - low_price

        # Éviter division par zéro
        if total_range == 0 or body_size == 0:
            return None

        # Configuration
        pinbar_config = self.patterns_config.get('pinbar', {})
        require_sr_proximity = pinbar_config.get('require_sr_proximity', True)
        sr_tolerance_percent = pinbar_config.get('sr_tolerance_percent', 2) / 100.0
        min_wick_body_ratio = pinbar_config.get('min_wick_body_ratio', 2.0)

        # BULLISH PIN BAR (Hammer)
        # - Longue mèche basse (au moins min_wick_body_ratio * body_size)
        # - Body dans le 1/3 supérieur de la bougie
        # - Petite ou pas de mèche haute
        if lower_wick >= min_wick_body_ratio * body_size:
            # Vérifier que le body est dans le tiers supérieur
            body_position = (body_bottom - low_price) / total_range
            if body_position >= 0.66:  # Body dans les 33% supérieurs
                # Vérifier que la mèche haute est petite
                if upper_wick <= body_size:
                    # Filtre S/R si activé
                    if require_sr_proximity:
                        if not support_levels:
                            return None  # Pas de support disponible, rejet

                        near_support = False
                        nearest_level = None
                        for support in support_levels:
                            # La mèche basse doit toucher le support
                            distance = abs(low_price - support) / support
                            if distance <= sr_tolerance_percent:
                                near_support = True
                                nearest_level = support
                                break

                        if not near_support:
                            return None  # Ne touche pas un support, rejet

                        return {
                            'type': 'bullish_pinbar',
                            'direction': 'up',
                            'price': close_price,
                            'low': low_price,
                            'high': high_price,
                            'sr_level': nearest_level,
                            'wick_ratio': lower_wick / body_size if body_size > 0 else 0
                        }
                    else:
                        # Si le filtre S/R n'est pas requis, accepter le pattern
                        return {
                            'type': 'bullish_pinbar',
                            'direction': 'up',
                            'price': close_price,
                            'low': low_price,
                            'high': high_price,
                            'wick_ratio': lower_wick / body_size if body_size > 0 else 0
                        }

        # BEARISH PIN BAR (Shooting Star)
        # - Longue mèche haute (au moins min_wick_body_ratio * body_size)
        # - Body dans le 1/3 inférieur de la bougie
        # - Petite ou pas de mèche basse
        if upper_wick >= min_wick_body_ratio * body_size:
            # Vérifier que le body est dans le tiers inférieur
            body_position = (high_price - body_top) / total_range
            if body_position >= 0.66:  # Body dans les 33% inférieurs
                # Vérifier que la mèche basse est petite
                if lower_wick <= body_size:
                    # Filtre S/R si activé
                    if require_sr_proximity:
                        if not resistance_levels:
                            return None  # Pas de résistance disponible, rejet

                        near_resistance = False
                        nearest_level = None
                        for resistance in resistance_levels:
                            # La mèche haute doit toucher la résistance
                            distance = abs(high_price - resistance) / resistance
                            if distance <= sr_tolerance_percent:
                                near_resistance = True
                                nearest_level = resistance
                                break

                        if not near_resistance:
                            return None  # Ne touche pas une résistance, rejet

                        return {
                            'type': 'bearish_pinbar',
                            'direction': 'down',
                            'price': close_price,
                            'low': low_price,
                            'high': high_price,
                            'sr_level': nearest_level,
                            'wick_ratio': upper_wick / body_size if body_size > 0 else 0
                        }
                    else:
                        # Si le filtre S/R n'est pas requis, accepter le pattern
                        return {
                            'type': 'bearish_pinbar',
                            'direction': 'down',
                            'price': close_price,
                            'low': low_price,
                            'high': high_price,
                            'wick_ratio': upper_wick / body_size if body_size > 0 else 0
                        }

        return None

    def detect_combo_pattern(self, df: pd.DataFrame, support_levels: List[float] = None, resistance_levels: List[float] = None) -> Optional[Dict]:
        """Détecte les combos Pin Bar + Engulfing (signal TRÈS fort)

        LOGIQUE STRICTE:
        - L'engulfing DOIT être sur la dernière bougie (confirmation récente)
        - Le pin bar doit être dans les 2-3 bougies précédentes (setup)
        - Ils doivent être de la même direction (bullish ou bearish)
        - Le pin bar doit précéder l'engulfing (logique de confirmation)

        Bullish Combo: Hammer sur support → Bullish Engulfing confirme
        Bearish Combo: Shooting Star sur résistance → Bearish Engulfing confirme

        Args:
            df: DataFrame avec les données OHLCV
            support_levels: Liste des niveaux de support (optionnel)
            resistance_levels: Liste des niveaux de résistance (optionnel)
        """
        combo_config = self.patterns_config.get('combo', {})
        window_size = combo_config.get('window_size', 3)

        if len(df) < 3:  # Minimum 3 bougies (pin bar peut être sur N-2, engulfing sur N)
            return None

        # ÉTAPE 1: Vérifier si la DERNIÈRE bougie est un engulfing
        engulfing = self.detect_engulfing(df, support_levels, resistance_levels)
        if not engulfing:
            return None  # Pas d'engulfing sur la dernière bougie = pas de combo

        engulfing_type = engulfing['type']

        # ÉTAPE 2: Chercher un pin bar dans les 2-3 bougies PRÉCÉDENTES
        # On regarde les bougies N-3 à N-1 (avant la dernière)
        lookback = min(window_size, len(df) - 1)
        pinbar_found = None

        for i in range(1, lookback + 1):
            # Créer un sous-dataframe jusqu'à la bougie N-i
            temp_df = df.iloc[:-i].copy()
            if len(temp_df) < 1:
                continue

            pinbar = self.detect_pinbar(temp_df, support_levels, resistance_levels)
            if pinbar:
                pinbar_type = pinbar['type']

                # Vérifier que le pin bar et l'engulfing sont de la même direction
                if (pinbar_type == 'bullish_pinbar' and engulfing_type == 'bullish_engulfing') or \
                   (pinbar_type == 'bearish_pinbar' and engulfing_type == 'bearish_engulfing'):
                    pinbar_found = pinbar
                    break  # Prendre le pin bar le plus récent qui correspond

        # ÉTAPE 3: Si on a trouvé pin bar ET engulfing de même direction = COMBO
        if pinbar_found:
            pinbar_type = pinbar_found['type']

            # BULLISH COMBO: Hammer précède Bullish Engulfing
            if pinbar_type == 'bullish_pinbar' and engulfing_type == 'bullish_engulfing':
                sr_level = pinbar_found.get('sr_level') or engulfing.get('sr_level')

                # VÉRIFICATION STRICTE: Les deux doivent avoir un sr_level valide (= près d'un S/R)
                # Si require_sr_proximity est activé, on ne garde que les combos avec S/R
                if combo_config.get('require_sr_proximity', True):
                    if not sr_level or (not pinbar_found.get('sr_level') and not engulfing.get('sr_level')):
                        return None  # Pas de S/R valide = pas un vrai combo

                return {
                    'type': 'bullish_combo',
                    'direction': 'up',
                    'price': engulfing['price'],
                    'low': pinbar_found['low'],
                    'high': engulfing.get('current_candle', {}).get('high', pinbar_found['high']),
                    'sr_level': sr_level,
                    'components': ['hammer', 'bullish_engulfing']
                }

            # BEARISH COMBO: Shooting Star précède Bearish Engulfing
            elif pinbar_type == 'bearish_pinbar' and engulfing_type == 'bearish_engulfing':
                sr_level = pinbar_found.get('sr_level') or engulfing.get('sr_level')

                # VÉRIFICATION STRICTE: Les deux doivent avoir un sr_level valide (= près d'un S/R)
                if combo_config.get('require_sr_proximity', True):
                    if not sr_level or (not pinbar_found.get('sr_level') and not engulfing.get('sr_level')):
                        return None  # Pas de S/R valide = pas un vrai combo

                return {
                    'type': 'bearish_combo',
                    'direction': 'down',
                    'price': engulfing['price'],
                    'low': engulfing.get('current_candle', {}).get('low', pinbar_found['low']),
                    'high': pinbar_found['high'],
                    'sr_level': sr_level,
                    'components': ['shooting_star', 'bearish_engulfing']
                }

        return None

    def save_sr_levels(self, symbol: str, support_levels: List[float], resistance_levels: List[float], date: str):
        """Sauvegarde les niveaux S/R pour un symbole

        IMPORTANT: Sauvegarde TOUJOURS dans patterns_backtest car les S/R sont calculés en backtest
        """
        base_patterns_folder = self.patterns_config['support_resistance']['patterns_folder']
        backtest_folder = f"{base_patterns_folder}_backtest"
        os.makedirs(backtest_folder, exist_ok=True)

        filename = os.path.join(backtest_folder, f"{date}_{symbol}_sr.json")

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
        """Charge les niveaux S/R depuis le fichier

        IMPORTANT: Charge toujours depuis patterns_backtest car c'est là que les S/R sont calculés
        """
        base_patterns_folder = self.patterns_config['support_resistance']['patterns_folder']
        backtest_folder = f"{base_patterns_folder}_backtest"
        filename = os.path.join(backtest_folder, f"{date}_{symbol}_sr.json")

        if not os.path.exists(filename):
            return [], []

        with open(filename, 'r') as f:
            data = json.load(f)

        return data.get('support_levels', []), data.get('resistance_levels', [])

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

    def generate_chart(self, symbol: str, df: pd.DataFrame, support_levels: List[float],
                       resistance_levels: List[float], detected_patterns: List[Dict], date: str):
        """Génère un graphique HTML interactif pour un symbole

        Args:
            symbol: Symbole du titre
            df: DataFrame avec les données OHLCV
            support_levels: Liste des niveaux de support
            resistance_levels: Liste des niveaux de résistance
            detected_patterns: Liste des patterns détectés (breakouts, combos, etc.)
            date: Date du scan
        """
        print(f"\nGénération du graphique pour {symbol}...")

        # Créer un graphique simple sans volume
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            )
        )

        # Lignes de support (vertes)
        for support in support_levels:
            fig.add_hline(
                y=support,
                line_dash="dash",
                line_color="green",
                annotation_text=f"S: ${support:.2f}",
                annotation_position="right"
            )

        # Lignes de résistance (rouges)
        for resistance in resistance_levels:
            fig.add_hline(
                y=resistance,
                line_dash="dash",
                line_color="red",
                annotation_text=f"R: ${resistance:.2f}",
                annotation_position="right"
            )

        # Marqueurs pour les breakouts
        breakout_up_dates = []
        breakout_up_prices = []
        breakout_up_texts = []

        breakout_down_dates = []
        breakout_down_prices = []
        breakout_down_texts = []

        # Marqueurs pour les engulfing patterns
        engulfing_bull_dates = []
        engulfing_bull_prices = []
        engulfing_bull_texts = []

        engulfing_bear_dates = []
        engulfing_bear_prices = []
        engulfing_bear_texts = []

        # Marqueurs pour les pin bars
        pinbar_bull_dates = []
        pinbar_bull_prices = []
        pinbar_bull_texts = []

        pinbar_bear_dates = []
        pinbar_bear_prices = []
        pinbar_bear_texts = []

        # Marqueurs pour les combos
        combo_bull_dates = []
        combo_bull_prices = []
        combo_bull_texts = []

        combo_bear_dates = []
        combo_bear_prices = []
        combo_bear_texts = []

        for pattern in detected_patterns:
            pattern_date = pattern['date']
            pattern_type = pattern['type']
            direction = pattern['direction']

            if pattern_type == 'resistance_breakout':
                pattern_level = pattern['level']
                pattern_close = pattern['close']
                breakout_up_dates.append(pattern_date)
                breakout_up_prices.append(pattern_level)
                breakout_up_texts.append(f"BREAKOUT UP<br>Résistance @ ${pattern_level:.2f}<br>Close: ${pattern_close:.2f}")

            elif pattern_type == 'support_breakdown':
                pattern_level = pattern['level']
                pattern_close = pattern['close']
                breakout_down_dates.append(pattern_date)
                breakout_down_prices.append(pattern_level)
                breakout_down_texts.append(f"BREAKDOWN<br>Support @ ${pattern_level:.2f}<br>Close: ${pattern_close:.2f}")

            elif pattern_type == 'bullish_engulfing':
                price = pattern['price']
                engulfing_bull_dates.append(pattern_date)
                engulfing_bull_prices.append(price)
                engulfing_bull_texts.append(f"BULLISH ENGULFING<br>Price: ${price:.2f}<br>Retournement haussier")

            elif pattern_type == 'bearish_engulfing':
                price = pattern['price']
                engulfing_bear_dates.append(pattern_date)
                engulfing_bear_prices.append(price)
                engulfing_bear_texts.append(f"BEARISH ENGULFING<br>Price: ${price:.2f}<br>Retournement baissier")

            elif pattern_type == 'bullish_pinbar':
                low = pattern.get('low', pattern['price'])
                pinbar_bull_dates.append(pattern_date)
                pinbar_bull_prices.append(low)
                pinbar_bull_texts.append(f"BULLISH PIN BAR<br>Hammer @ ${pattern['price']:.2f}<br>Rejet du support")

            elif pattern_type == 'bearish_pinbar':
                high = pattern.get('high', pattern['price'])
                pinbar_bear_dates.append(pattern_date)
                pinbar_bear_prices.append(high)
                pinbar_bear_texts.append(f"BEARISH PIN BAR<br>Shooting Star @ ${pattern['price']:.2f}<br>Rejet de la résistance")

            elif pattern_type == 'bullish_combo':
                price = pattern['price']
                combo_bull_dates.append(pattern_date)
                combo_bull_prices.append(price)
                sr_level = pattern.get('sr_level')
                sr_text = f"<br>S/R @ ${sr_level:.2f}" if sr_level else ""
                combo_bull_texts.append(f"⭐ COMBO BULLISH ⭐<br>Hammer + Engulfing<br>@ ${price:.2f}{sr_text}<br>SIGNAL TRÈS FORT")

            elif pattern_type == 'bearish_combo':
                price = pattern['price']
                combo_bear_dates.append(pattern_date)
                combo_bear_prices.append(price)
                sr_level = pattern.get('sr_level')
                sr_text = f"<br>S/R @ ${sr_level:.2f}" if sr_level else ""
                combo_bear_texts.append(f"⭐ COMBO BEARISH ⭐<br>Shooting Star + Engulfing<br>@ ${price:.2f}{sr_text}<br>SIGNAL TRÈS FORT")

        # Ajouter les marqueurs de breakout UP (vert)
        if breakout_up_dates:
            fig.add_trace(
                go.Scatter(
                    x=breakout_up_dates,
                    y=breakout_up_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='lime',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Breakout UP',
                    hovertext=breakout_up_texts,
                    hoverinfo='text'
                )
            )

        # Ajouter les marqueurs de breakout DOWN (rouge)
        if breakout_down_dates:
            fig.add_trace(
                go.Scatter(
                    x=breakout_down_dates,
                    y=breakout_down_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Breakdown',
                    hovertext=breakout_down_texts,
                    hoverinfo='text'
                )
            )

        # Ajouter les marqueurs d'engulfing bullish (étoile verte)
        if engulfing_bull_dates:
            fig.add_trace(
                go.Scatter(
                    x=engulfing_bull_dates,
                    y=engulfing_bull_prices,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=18,
                        color='lightgreen',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Bullish Engulfing',
                    hovertext=engulfing_bull_texts,
                    hoverinfo='text'
                )
            )

        # Ajouter les marqueurs d'engulfing bearish (étoile rouge)
        if engulfing_bear_dates:
            fig.add_trace(
                go.Scatter(
                    x=engulfing_bear_dates,
                    y=engulfing_bear_prices,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=18,
                        color='orange',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Bearish Engulfing',
                    hovertext=engulfing_bear_texts,
                    hoverinfo='text'
                )
            )

        # Ajouter les marqueurs de pin bar bullish (triangle vers bas, pointant vers le rejet)
        if pinbar_bull_dates:
            fig.add_trace(
                go.Scatter(
                    x=pinbar_bull_dates,
                    y=pinbar_bull_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=16,
                        color='cyan',
                        line=dict(width=2, color='blue')
                    ),
                    name='Bullish Pin Bar',
                    hovertext=pinbar_bull_texts,
                    hoverinfo='text'
                )
            )

        # Ajouter les marqueurs de pin bar bearish (triangle vers haut, pointant vers le rejet)
        if pinbar_bear_dates:
            fig.add_trace(
                go.Scatter(
                    x=pinbar_bear_dates,
                    y=pinbar_bear_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=16,
                        color='magenta',
                        line=dict(width=2, color='purple')
                    ),
                    name='Bearish Pin Bar',
                    hovertext=pinbar_bear_texts,
                    hoverinfo='text'
                )
            )

        # Ajouter les marqueurs de combo bullish (diamant doré - signal premium)
        if combo_bull_dates:
            fig.add_trace(
                go.Scatter(
                    x=combo_bull_dates,
                    y=combo_bull_prices,
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=22,
                        color='gold',
                        line=dict(width=3, color='orange')
                    ),
                    name='⭐ Combo Bullish',
                    hovertext=combo_bull_texts,
                    hoverinfo='text'
                )
            )

        # Ajouter les marqueurs de combo bearish (diamant violet - signal premium)
        if combo_bear_dates:
            fig.add_trace(
                go.Scatter(
                    x=combo_bear_dates,
                    y=combo_bear_prices,
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=22,
                        color='darkviolet',
                        line=dict(width=3, color='purple')
                    ),
                    name='⭐ Combo Bearish',
                    hovertext=combo_bear_texts,
                    hoverinfo='text'
                )
            )

        # Mise en forme
        fig.update_layout(
            title=f'{symbol} - Support/Resistance & Patterns - {date}',
            xaxis_title="Date",
            yaxis_title="Prix ($)",
            xaxis_rangeslider_visible=False,
            width=2400,
            height=1200,
            hovermode='x unified',
            template='plotly_dark'
        )

        # Sauvegarder
        output_file = os.path.join(self.patterns_folder, f"{date}_{symbol}_chart.html")
        fig.write_html(output_file)
        print(f"Graphique sauvegardé: {output_file}")

        return output_file

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

            # Validation: vérifier les colonnes requises
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Colonnes manquantes pour {symbol}: {missing_columns}")
                return None

            # Prendre les N dernières bougies
            df = df.tail(candle_nb)

            df.reset_index(inplace=True)

            # Vérifier que Date existe après reset_index
            if 'Date' not in df.columns:
                print(f"Colonne 'Date' manquante après reset_index pour {symbol}")
                return None

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

        # Filtrer la watchlist si --chart est spécifié
        if self.chart_symbol:
            watchlist = [item for item in watchlist if item['symbol'].upper() == self.chart_symbol.upper()]
            if not watchlist:
                print(f"Erreur: Symbole {self.chart_symbol} non trouvé dans la watchlist")
                return

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

            # Liste pour stocker les patterns détectés (pour le graphique)
            detected_patterns = []

            # CALCULER LES S/R UNE SEULE FOIS sur les données à test_stop
            # Position de référence pour le calcul des S/R
            sr_calc_pos = total_candles - test_stop
            if sr_calc_pos <= 0:
                continue

            df_for_sr = df.iloc[:sr_calc_pos].copy()
            support_levels, resistance_levels = self.find_support_resistance(df_for_sr)

            print(f"\n{symbol}: S/R calculés sur {len(df_for_sr)} bougies")
            print(f"  Supports: {[f'{s:.2f}' for s in support_levels[:5]]}")
            print(f"  Résistances: {[f'{r:.2f}' for r in resistance_levels[:5]]}\n")

            # Boucle de test: du passé vers le présent (de test_stop vers test_start)
            for candle_nb in range(test_stop, test_start - 1, -1):
                # Position dans le df: on enlève les N dernières bougies
                current_pos = total_candles - candle_nb

                if current_pos <= 0:
                    break

                # Prend les données jusqu'à cette position (exclut les N dernières bougies)
                df_until_pos = df.iloc[:current_pos].copy()

                # UTILISER LES S/R CALCULÉS UNE SEULE FOIS (pas de recalcul dans la boucle)

                # Flag pour éviter détections individuelles si combo trouvé
                combo_detected = False

                # Détecte combos EN PREMIER (priorité)
                if self.is_pattern_enabled('combo'):
                    combo = self.detect_combo_pattern(df_until_pos, support_levels, resistance_levels)
                    if combo:
                        # Extrait la date de la dernière bougie
                        current_date = df_until_pos['Date'].iloc[-1]

                        # Ajoute au graphique (avec le S/R utilisé pour la détection)
                        detected_patterns.append({
                            'type': combo['type'],
                            'date': current_date,
                            'price': combo['price'],
                            'low': combo.get('low'),
                            'high': combo.get('high'),
                            'direction': combo['direction'],
                            'sr_level': combo.get('sr_level')  # Stocke le S/R détecté
                        })

                        if self.should_print_pattern('combo'):
                            pattern_name = '⭐ COMBO BULLISH (Hammer + Engulfing)' if combo['type'] == 'bullish_combo' else '⭐ COMBO BEARISH (Shooting Star + Engulfing)'
                            sr_info = f" (près S/R {combo['sr_level']:.2f})" if 'sr_level' in combo and combo['sr_level'] is not None else " (NO S/R!)"

                            # Debug: montrer tous les S/R disponibles
                            all_sr = f" [Supports: {[f'{s:.2f}' for s in support_levels[:3]]}] [Résistances: {[f'{r:.2f}' for r in resistance_levels[:3]]}]"

                            print(f"{symbol}: Bougie {candle_nb} ({current_date}): {pattern_name} à ${combo['price']:.2f}{sr_info}{all_sr}")

                        combo_detected = True  # Ne pas détecter les patterns individuels

                # Détecte breakouts
                if self.is_pattern_enabled('breakouts'):
                    breakout = self.detect_breakouts(df_until_pos, support_levels, resistance_levels)
                    if breakout:
                        # Extrait la date de la dernière bougie
                        current_date = df_until_pos['Date'].iloc[-1]

                        # Ajoute au graphique
                        detected_patterns.append({
                            'type': breakout['type'],
                            'date': current_date,
                            'level': breakout['level'],
                            'close': breakout['close'],
                            'direction': breakout['direction']
                        })

                        if self.should_print_pattern('breakouts'):
                            direction = 'UP' if breakout['direction'] == 'up' else 'DOWN'
                            label = 'résistance' if breakout['type'] == 'resistance_breakout' else 'support'
                            print(f"{symbol}: Bougie {candle_nb} ({current_date}): BREAKOUT {direction} {label} à {breakout['level']:.2f}")

                # Détecte engulfing patterns (seulement si pas de combo)
                if not combo_detected and self.is_pattern_enabled('engulfing'):
                    engulfing = self.detect_engulfing(df_until_pos, support_levels, resistance_levels)
                    if engulfing:
                        # Extrait la date de la dernière bougie
                        current_date = df_until_pos['Date'].iloc[-1]

                        # Ajoute au graphique
                        detected_patterns.append({
                            'type': engulfing['type'],
                            'date': current_date,
                            'price': engulfing['price'],
                            'direction': engulfing['direction']
                        })

                        if self.should_print_pattern('engulfing'):
                            pattern_name = 'BULLISH ENGULFING' if engulfing['type'] == 'bullish_engulfing' else 'BEARISH ENGULFING'
                            sr_info = f" (près S/R {engulfing['sr_level']:.2f})" if 'sr_level' in engulfing and engulfing['sr_level'] is not None else ""
                            print(f"{symbol}: Bougie {candle_nb} ({current_date}): {pattern_name} à ${engulfing['price']:.2f}{sr_info}")

                # Détecte pin bars (seulement si pas de combo)
                if not combo_detected and self.is_pattern_enabled('pinbar'):
                    pinbar = self.detect_pinbar(df_until_pos, support_levels, resistance_levels)
                    if pinbar:
                        # Extrait la date de la dernière bougie
                        current_date = df_until_pos['Date'].iloc[-1]

                        # Ajoute au graphique
                        detected_patterns.append({
                            'type': pinbar['type'],
                            'date': current_date,
                            'price': pinbar['price'],
                            'low': pinbar['low'],
                            'high': pinbar['high'],
                            'direction': pinbar['direction']
                        })

                        if self.should_print_pattern('pinbar'):
                            pattern_name = 'BULLISH PIN BAR (Hammer)' if pinbar['type'] == 'bullish_pinbar' else 'BEARISH PIN BAR (Shooting Star)'
                            sr_info = f" (près S/R {pinbar['sr_level']:.2f})" if 'sr_level' in pinbar and pinbar['sr_level'] is not None else ""
                            wick_ratio = pinbar.get('wick_ratio', 0)
                            print(f"{symbol}: Bougie {candle_nb} ({current_date}): {pattern_name} à ${pinbar['price']:.2f}{sr_info} (ratio: {wick_ratio:.1f}x)")

                # Sauvegarde les S/R pour la première bougie testée (la plus récente)
                if candle_nb == test_start:
                    self.save_sr_levels(symbol, support_levels, resistance_levels, today)

            # Génère le graphique si ce symbole correspond à celui demandé
            if self.chart_symbol and self.chart_symbol.upper() == symbol.upper():
                # UTILISER LES MÊMES S/R calculés avant la boucle (pas de recalcul)
                self.generate_chart(symbol, df, support_levels, resistance_levels, detected_patterns, today)

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
                'prev_high': bars[-2].high,
                'prev_low': bars[-2].low,
                'prev_close': bars[-2].close,
                'current_high': bars[-1].high,
                'current_low': bars[-1].low,
                'current_close': bars[-1].close,
                'current_date': date_est.strftime('%Y-%m-%d')
            }

        except Exception as e:
            print(f"Erreur récupération {symbol}: {e}")
            return None

    def check_realtime_breakout(self, bars_data: Dict, support_levels: List[float], resistance_levels: List[float]) -> Optional[Dict]:
        """Vérifie si un breakout est en cours avec les données temps réel

        Breakout = prev_close d'un côté du niveau + current_high/low de l'autre côté
        """
        prev_close = bars_data.get('prev_close')
        current_high = bars_data['current_high']
        current_low = bars_data['current_low']
        current_close = bars_data['current_close']

        # Si les données de la bougie précédente ne sont pas complètes, on ne peut pas confirmer
        if prev_close is None:
            return None

        # Détection breakout résistance (vers le haut)
        for resistance in resistance_levels:
            if prev_close < resistance and current_high > resistance:
                return {
                    'type': 'resistance_breakout',
                    'level': resistance,
                    'close': current_close,
                    'direction': 'up',
                    'timestamp': datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
                }

        # Détection breakdown support (vers le bas)
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
                print(f"{symbol}: {len(support_levels)} supports, {len(resistance_levels)} résistances")

        print()

        # Connexion IBKR
        ib = self.connect_ibkr()
        if not ib:
            print("Impossible de se connecter à IBKR. Arrêt.")
            return

        try:
            while True:
                # Vérifier et reconnecter IBKR si nécessaire
                if not ib.isConnected():
                    print("Connexion IBKR perdue, reconnexion...")
                    ib = self.connect_ibkr()
                    if not ib:
                        print("Reconnexion échouée. Arrêt.")
                        break

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

                    # Convertir bars_data en DataFrame pour les détections
                    df = pd.DataFrame([{
                        'Date': bars_data['prev_date'],
                        'Open': bars_data['prev_open'],
                        'High': bars_data['prev_high'],
                        'Low': bars_data['prev_low'],
                        'Close': bars_data['prev_close'],
                        'Volume': 0
                    }, {
                        'Date': bars_data['current_date'],
                        'Open': bars_data['current_open'],
                        'High': bars_data['current_high'],
                        'Low': bars_data['current_low'],
                        'Close': bars_data['current_close'],
                        'Volume': 0
                    }])

                    combo_detected = False

                    # Détecte combos EN PREMIER
                    if self.is_pattern_enabled('combo'):
                        combo = self.detect_combo_pattern(df, support_levels, resistance_levels)
                        if combo and self.should_print_pattern('combo'):
                            pattern_name = '⭐ COMBO BULLISH (Hammer + Engulfing)' if combo['type'] == 'bullish_combo' else '⭐ COMBO BEARISH (Shooting Star + Engulfing)'
                            sr_info = f" (près S/R {combo['sr_level']:.2f})" if 'sr_level' in combo and combo['sr_level'] is not None else ""
                            print(f"{symbol}: {bars_data['current_date']}: {pattern_name} à ${combo['price']:.2f}{sr_info}")
                            combo_detected = True

                    # Vérifie breakout si activé
                    if self.is_pattern_enabled('breakouts'):
                        breakout = self.check_realtime_breakout(bars_data, support_levels, resistance_levels)
                        if breakout and self.should_print_pattern('breakouts'):
                            # Indiquer la direction du breakout
                            direction = 'UP' if breakout['direction'] == 'up' else 'DOWN'
                            breakout_label = 'résistance' if breakout['type'] == 'resistance_breakout' else 'support'
                            print(f"BREAKOUT: {symbol} ({bars_data['current_date']}) ${bars_data['current_close']:.2f} {direction} {breakout_label} à {breakout['level']:.2f}")

                    # Détecte engulfing (si pas de combo et si activé)
                    if not combo_detected and self.is_pattern_enabled('engulfing'):
                        engulfing = self.detect_engulfing(df, support_levels, resistance_levels)
                        if engulfing and self.should_print_pattern('engulfing'):
                            pattern_name = 'BULLISH ENGULFING' if engulfing['type'] == 'bullish_engulfing' else 'BEARISH ENGULFING'
                            sr_info = f" (près S/R {engulfing['sr_level']:.2f})" if 'sr_level' in engulfing and engulfing['sr_level'] is not None else ""
                            print(f"{symbol}: {bars_data['current_date']}: {pattern_name} à ${engulfing['price']:.2f}{sr_info}")

                    # Détecte pin bars (si pas de combo et si activé)
                    if not combo_detected and self.is_pattern_enabled('pinbar'):
                        pinbar = self.detect_pinbar(df, support_levels, resistance_levels)
                        if pinbar and self.should_print_pattern('pinbar'):
                            pattern_name = 'BULLISH PIN BAR (Hammer)' if pinbar['type'] == 'bullish_pinbar' else 'BEARISH PIN BAR (Shooting Star)'
                            sr_info = f" (près S/R {pinbar['sr_level']:.2f})" if 'sr_level' in pinbar and pinbar['sr_level'] is not None else ""
                            wick_ratio = pinbar.get('wick_ratio', 0)
                            print(f"{symbol}: {bars_data['current_date']}: {pattern_name} à ${pinbar['price']:.2f}{sr_info} (ratio: {wick_ratio:.1f}x)")

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
    parser.add_argument('--chart', type=str, metavar='SYMBOL', help='Génère un graphique HTML pour le symbole spécifié (ex: --chart AAPL)')
    args = parser.parse_args()

    scanner = StockScanner('settings.json', is_backtest=args.backtest, chart_symbol=args.chart)
    scanner.run()
