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
        self.patterns_folder = self.patterns_config['support_resistance']['patterns_folder']
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

    def detect_step_1_breakout(self, df: pd.DataFrame, support_levels: List[float], resistance_levels: List[float]) -> Optional[Dict]:
        """Détecte les cassures step 1: bougie ENTIÈRE au-delà du niveau"""
        if len(df) < 1:
            return None

        last_idx = len(df) - 1
        current_high = float(df['High'].iloc[last_idx])
        current_low = float(df['Low'].iloc[last_idx])
        current_date = df['Date'].iloc[last_idx] if 'Date' in df.columns else None

        # Détection breakout résistance: bougie ENTIÈRE au-dessus (LOW > résistance)
        for resistance in resistance_levels:
            if current_low > resistance:
                return {
                    'type': 'step_1',
                    'level': resistance,
                    'original_type': 'resistance',
                    'direction': 'up',
                    'date': current_date
                }

        # Détection breakdown support: bougie ENTIÈRE en dessous (HIGH < support)
        for support in support_levels:
            if current_high < support:
                return {
                    'type': 'step_1',
                    'level': support,
                    'original_type': 'support',
                    'direction': 'down',
                    'date': current_date
                }

        return None

    def detect_step_2_pullback(self, df: pd.DataFrame, step_1_list: List[Dict]) -> Optional[Dict]:
        """Détecte les retournements step 2: bougie ENTIÈRE de l'autre côté du niveau"""
        if len(df) < 1 or not step_1_list:
            return None

        last_idx = len(df) - 1
        current_high = float(df['High'].iloc[last_idx])
        current_low = float(df['Low'].iloc[last_idx])
        current_date = df['Date'].iloc[last_idx] if 'Date' in df.columns else None

        # Parcourt les step 1 qui n'ont pas encore de step 2
        for i, step_1 in enumerate(step_1_list):
            if step_1.get('step_2_detected', False):
                continue

            level = step_1['level']
            original_type = step_1['original_type']

            # Cas 1: Après breakout résistance UP, retournement DOWN
            # Bougie ENTIÈRE en dessous de la résistance (HIGH < résistance)
            if original_type == 'resistance' and step_1['direction'] == 'up':
                if current_high < level:
                    step_1_list[i]['step_2_detected'] = True
                    return {
                        'type': 'step_2',
                        'level': level,
                        'original_type': original_type,
                        'direction': step_1['direction'],
                        'date': current_date
                    }

            # Cas 2: Après breakdown support DOWN, retournement UP
            # Bougie ENTIÈRE au-dessus du support (LOW > support)
            elif original_type == 'support' and step_1['direction'] == 'down':
                if current_low > level:
                    step_1_list[i]['step_2_detected'] = True
                    return {
                        'type': 'step_2',
                        'level': level,
                        'original_type': original_type,
                        'direction': step_1['direction'],
                        'date': current_date
                    }

        return None

    def detect_step_3_retest(self, df: pd.DataFrame, step_1_list: List[Dict]) -> Optional[Dict]:
        """Détecte les retests step 3: bougie touche le niveau"""
        if len(df) < 1 or not step_1_list:
            return None

        flip_tolerance = self.patterns_config['support_resistance']['flip_tolerance']
        last_idx = len(df) - 1
        current_high = float(df['High'].iloc[last_idx])
        current_low = float(df['Low'].iloc[last_idx])
        current_date = df['Date'].iloc[last_idx] if 'Date' in df.columns else None

        # Parcourt les step 1 qui ont un step 2 mais pas encore de step 3
        for i, step_1 in enumerate(step_1_list):
            if not step_1.get('step_2_detected', False) or step_1.get('step_3_detected', False):
                continue

            level = step_1['level']
            original_type = step_1['original_type']
            tolerance_range = level * flip_tolerance

            # Cas 1: Après breakout résistance, retest du niveau (devenu support)
            # Touche par le bas: LOW proche du niveau
            if original_type == 'resistance' and step_1['direction'] == 'up':
                if (current_low <= level + tolerance_range and
                    current_low >= level - tolerance_range):
                    step_1_list[i]['step_3_detected'] = True
                    return {
                        'type': 'step_3',
                        'level': level,
                        'original_type': original_type,
                        'new_type': 'support',
                        'direction': 'up',
                        'date': current_date
                    }

            # Cas 2: Après breakdown support, retest du niveau (devenu résistance)
            # Touche par le haut: HIGH proche du niveau
            elif original_type == 'support' and step_1['direction'] == 'down':
                if (current_high >= level - tolerance_range and
                    current_high <= level + tolerance_range):
                    step_1_list[i]['step_3_detected'] = True
                    return {
                        'type': 'step_3',
                        'level': level,
                        'original_type': original_type,
                        'new_type': 'resistance',
                        'direction': 'down',
                        'date': current_date
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
        for i, breakout in enumerate(breakout_history):
            # Ignore si un flip a déjà été détecté pour ce breakout
            if breakout.get('flip_detected', False):
                continue

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
                    # Marque le flip comme détecté
                    breakout_history[i]['flip_detected'] = True
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
                    # Marque le flip comme détecté
                    breakout_history[i]['flip_detected'] = True
                    return {
                        'type': 'flip_support_to_resistance',
                        'level': level,
                        'original_type': 'support',
                        'new_type': 'resistance',
                        'close': current_close
                    }

        return None

    def save_sr_levels(self, symbol: str, support_levels: List[float], resistance_levels: List[float], date: str, breakout_history: Optional[List[Dict]] = None, last_breakout_direction: Optional[str] = None, last_breakout_level: Optional[float] = None):
        """Sauvegarde les niveaux S/R pour un symbole"""
        os.makedirs(self.patterns_folder, exist_ok=True)

        filename = os.path.join(self.patterns_folder, f"{date}_{symbol}_sr.json")

        # Charge l'historique existant si présent
        existing_history = []
        existing_direction = None
        existing_level = None
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
                    existing_history = existing_data.get('breakout_history', [])
                    existing_direction = existing_data.get('last_breakout_direction')
                    existing_level = existing_data.get('last_breakout_level')
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

        # Utilise le niveau fourni ou conserve l'existant
        if last_breakout_level is not None:
            final_level = last_breakout_level
        else:
            final_level = existing_level

        # Limiter l'historique à 50 breakouts max pour éviter croissance infinie
        MAX_HISTORY = 50
        if len(final_history) > MAX_HISTORY:
            final_history = final_history[-MAX_HISTORY:]

        data = {
            'symbol': symbol,
            'date': date,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'breakout_history': final_history,
            'last_breakout_direction': final_direction,
            'last_breakout_level': final_level,
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

    def load_last_breakout_level(self, symbol: str, date: str) -> Optional[float]:
        """Charge le niveau du dernier breakout depuis le fichier"""
        filename = os.path.join(self.patterns_folder, f"{date}_{symbol}_sr.json")

        if not os.path.exists(filename):
            return None

        with open(filename, 'r') as f:
            data = json.load(f)

        return data.get('last_breakout_level')

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
            detected_patterns: Liste des patterns détectés (breakouts et flips)
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

        # Marqueurs numérotés pour visualiser les étapes
        step_1_dates = []
        step_1_prices = []
        step_1_texts = []

        step_2_dates = []
        step_2_prices = []
        step_2_texts = []

        step_3_dates = []
        step_3_prices = []
        step_3_texts = []

        for pattern in detected_patterns:
            pattern_date = pattern['date']
            pattern_type = pattern['type']
            pattern_level = pattern['level']

            if pattern_type == 'step_1':
                step_1_dates.append(pattern_date)
                step_1_prices.append(pattern['price'])
                direction = 'UP' if pattern['direction'] == 'up' else 'DOWN'
                step_1_texts.append(f"1: Cassure {direction}<br>@ ${pattern_level:.2f}")

            elif pattern_type == 'step_2':
                step_2_dates.append(pattern_date)
                step_2_prices.append(pattern['price'])
                direction = 'UP' if pattern['direction'] == 'up' else 'DOWN'
                step_2_texts.append(f"2: Confirmation {direction}<br>@ ${pattern_level:.2f}")

            elif pattern_type == 'step_3':
                step_3_dates.append(pattern_date)
                step_3_prices.append(pattern['price'])
                direction = 'UP' if pattern['direction'] == 'up' else 'DOWN'
                step_3_texts.append(f"3: Flip {direction}<br>{pattern['from']}->{pattern['to']}<br>@ ${pattern_level:.2f}")

        # Ajouter les marqueurs numéro 1 (cassure)
        if step_1_dates:
            for date, price, text in zip(step_1_dates, step_1_prices, step_1_texts):
                fig.add_trace(
                    go.Scatter(
                        x=[date],
                        y=[price],
                        mode='markers+text',
                        marker=dict(
                            symbol='circle',
                            size=20,
                            color='yellow',
                            line=dict(width=2, color='black')
                        ),
                        text='1',
                        textfont=dict(size=14, color='black', family='Arial Black'),
                        textposition='middle center',
                        showlegend=False,
                        hovertext=text,
                        hoverinfo='text'
                    )
                )

        # Ajouter les marqueurs numéro 2 (confirmation)
        if step_2_dates:
            for date, price, text in zip(step_2_dates, step_2_prices, step_2_texts):
                fig.add_trace(
                    go.Scatter(
                        x=[date],
                        y=[price],
                        mode='markers+text',
                        marker=dict(
                            symbol='circle',
                            size=20,
                            color='orange',
                            line=dict(width=2, color='black')
                        ),
                        text='2',
                        textfont=dict(size=14, color='black', family='Arial Black'),
                        textposition='middle center',
                        showlegend=False,
                        hovertext=text,
                        hoverinfo='text'
                    )
                )

        # Ajouter les marqueurs numéro 3 (flip/retest)
        if step_3_dates:
            for date, price, text in zip(step_3_dates, step_3_prices, step_3_texts):
                fig.add_trace(
                    go.Scatter(
                        x=[date],
                        y=[price],
                        mode='markers+text',
                        marker=dict(
                            symbol='circle',
                            size=20,
                            color='cyan',
                            line=dict(width=2, color='black')
                        ),
                        text='3',
                        textfont=dict(size=14, color='black', family='Arial Black'),
                        textposition='middle center',
                        showlegend=False,
                        hovertext=text,
                        hoverinfo='text'
                    )
                )

        # Mise en forme
        fig.update_layout(
            title=f'{symbol} - Support/Resistance & Patterns - {date}',
            xaxis_title="Date",
            yaxis_title="Prix ($)",
            xaxis_rangeslider_visible=False,
            height=700,
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

            # Liste pour tracker les breakouts (step_1) et leurs états
            step_1_list = []

            # Liste pour stocker les patterns détectés (pour le graphique)
            detected_patterns = []

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

                # STEP 1: Détecte cassure (bougie ENTIÈRE au-delà du niveau)
                if self.is_pattern_enabled('breakouts'):
                    step_1 = self.detect_step_1_breakout(df_until_pos, support_levels, resistance_levels)
                    if step_1:
                        # Ajoute à la liste de tracking
                        step_1['step_2_detected'] = False
                        step_1['step_3_detected'] = False
                        step_1_list.append(step_1)

                        # Ajoute au graphique
                        detected_patterns.append({
                            'type': 'step_1',
                            'date': step_1['date'],
                            'price': step_1['level'],
                            'level': step_1['level'],
                            'direction': step_1['direction']
                        })

                        if self.should_print_pattern('breakouts'):
                            direction = 'UP' if step_1['direction'] == 'up' else 'DOWN'
                            label = step_1['original_type']
                            print(f"{symbol}: Bougie {candle_nb} ({step_1['date']}): STEP 1 - Cassure {direction} {label} à {step_1['level']:.2f}")

                # STEP 2: Détecte retournement (bougie ENTIÈRE de l'autre côté)
                if self.is_pattern_enabled('breakouts'):
                    step_2 = self.detect_step_2_pullback(df_until_pos, step_1_list)
                    if step_2:
                        # Ajoute au graphique
                        detected_patterns.append({
                            'type': 'step_2',
                            'date': step_2['date'],
                            'price': step_2['level'],
                            'level': step_2['level'],
                            'direction': step_2['direction']
                        })

                        if self.should_print_pattern('breakouts'):
                            direction = 'UP' if step_2['direction'] == 'up' else 'DOWN'
                            print(f"{symbol}: Bougie {candle_nb} ({step_2['date']}): STEP 2 - Retournement pour cassure {direction} à {step_2['level']:.2f}")

                # STEP 3: Détecte retest (bougie touche le niveau)
                if self.is_pattern_enabled('flips'):
                    step_3 = self.detect_step_3_retest(df_until_pos, step_1_list)
                    if step_3:
                        # Ajoute au graphique
                        detected_patterns.append({
                            'type': 'step_3',
                            'date': step_3['date'],
                            'price': step_3['level'],
                            'level': step_3['level'],
                            'direction': step_3['direction'],
                            'from': step_3['original_type'],
                            'to': step_3['new_type']
                        })

                        if self.should_print_pattern('flips'):
                            direction = 'UP' if step_3['direction'] == 'up' else 'DOWN'
                            print(f"{symbol}: Bougie {candle_nb} ({step_3['date']}): STEP 3 - Retest {direction} {step_3['original_type']}->{step_3['new_type']} à {step_3['level']:.2f}")

                # Sauvegarde les S/R pour la première bougie testée (la plus récente)
                if candle_nb == test_start:
                    # Convertir step_1_list en breakout_history pour compatibilité
                    breakout_history = []
                    for s1 in step_1_list:
                        breakout_history.append({
                            'level': s1['level'],
                            'original_type': s1['original_type'],
                            'breakout_candle': 0,
                            'breakout_date': today,
                            'flip_detected': s1.get('step_3_detected', False)
                        })
                    self.save_sr_levels(symbol, support_levels, resistance_levels, today, breakout_history, None, None)

            # Génère le graphique si ce symbole correspond à celui demandé
            if self.chart_symbol and self.chart_symbol.upper() == symbol.upper():
                # Recalcule les S/R sur le DataFrame complet pour l'affichage
                final_support_levels, final_resistance_levels = self.find_support_resistance(df)
                self.generate_chart(symbol, df, final_support_levels, final_resistance_levels, detected_patterns, today)

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

    def check_realtime_breakout(self, bars_data: Dict, support_levels: List[float], resistance_levels: List[float], last_breakout_direction: Optional[str] = None, last_breakout_level: Optional[float] = None) -> Optional[Dict]:
        """Vérifie si un breakout est en cours avec les données temps réel avec confirmation

        Breakout = bougie N-1 CLOSE casse + bougie N CLOSE confirme

        Args:
            last_breakout_direction: 'up' si dernier breakout était résistance, 'down' si c'était support, None si aucun
            last_breakout_level: Niveau du dernier breakout (pour détecter les retracements)
        """
        prev_close = bars_data.get('prev_close')
        current_close = bars_data['current_close']

        # Si les données de la bougie précédente ne sont pas complètes, on ne peut pas confirmer
        if prev_close is None:
            return None

        # Reset de la direction si retracement significatif (3%)
        RETRACEMENT_THRESHOLD = 0.03
        if last_breakout_direction and last_breakout_level:
            if last_breakout_direction == 'up' and current_close < last_breakout_level * (1 - RETRACEMENT_THRESHOLD):
                # Prix a retracé de plus de 3% sous le dernier niveau de breakout up
                last_breakout_direction = None
                last_breakout_level = None
            elif last_breakout_direction == 'down' and current_close > last_breakout_level * (1 + RETRACEMENT_THRESHOLD):
                # Prix a remonté de plus de 3% au-dessus du dernier niveau de breakdown
                last_breakout_direction = None
                last_breakout_level = None

        # Détection breakout résistance (vers le haut) avec confirmation
        # Bougie N-1: CLOSE > résistance (vraie cassure)
        # Bougie N: CLOSE > résistance (confirme)
        if last_breakout_direction != 'up':
            for resistance in resistance_levels:
                if prev_close > resistance and current_close > resistance:
                    return {
                        'type': 'resistance_breakout',
                        'level': resistance,
                        'close': current_close,
                        'direction': 'up',
                        'timestamp': datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
                    }

        # Détection breakdown support (vers le bas) avec confirmation
        # Bougie N-1: CLOSE < support (vraie cassure)
        # Bougie N: CLOSE < support (confirme)
        if last_breakout_direction != 'down':
            for support in support_levels:
                if prev_close < support and current_close < support:
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
        for i, breakout in enumerate(breakout_history):
            # Ignore si un flip a déjà été détecté pour ce breakout
            if breakout.get('flip_detected', False):
                continue

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
                    # Marque le flip comme détecté
                    breakout_history[i]['flip_detected'] = True
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
                    # Marque le flip comme détecté
                    breakout_history[i]['flip_detected'] = True
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
            last_breakout_level = self.load_last_breakout_level(symbol, today)

            if not support_levels and not resistance_levels:
                print(f"{symbol}: Pas de S/R pour {today} - lancer --backtest d'abord")
                sr_data[symbol] = None
            else:
                sr_data[symbol] = {
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels,
                    'breakout_history': breakout_history,
                    'last_breakout_direction': last_breakout_direction,
                    'last_breakout_level': last_breakout_level
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
                    breakout_history = sr_data[symbol]['breakout_history']
                    last_breakout_direction = sr_data[symbol]['last_breakout_direction']
                    last_breakout_level = sr_data[symbol]['last_breakout_level']

                    # Récupère les données IBKR
                    bars_data = self.get_last_bars_ibkr(ib, symbol)

                    if not bars_data:
                        continue

                    # Vérifie flip si activé
                    if self.is_pattern_enabled('flips'):
                        flip = self.check_realtime_flip(bars_data, breakout_history)
                        if flip and self.should_print_pattern('flips'):
                            # Indiquer la direction attendue après le flip
                            if flip['type'] == 'flip_resistance_to_support':
                                direction = 'UP'  # Ancien résistance devient support → prix soutenu UP
                            else:
                                direction = 'DOWN'  # Ancien support devient résistance → prix rejeté DOWN
                            print(f"FLIP: {symbol} ({bars_data['current_date']}) ${bars_data['current_close']:.2f} {direction} {flip['original_type']}->{flip['new_type']} à {flip['level']:.2f}")

                    # Vérifie breakout si activé
                    if self.is_pattern_enabled('breakouts'):
                        breakout = self.check_realtime_breakout(bars_data, support_levels, resistance_levels, last_breakout_direction, last_breakout_level)
                        if breakout:
                            if self.should_print_pattern('breakouts'):
                                # Indiquer la direction du breakout
                                direction = 'UP' if breakout['direction'] == 'up' else 'DOWN'
                                breakout_label = 'résistance' if breakout['type'] == 'resistance_breakout' else 'support'
                                print(f"BREAKOUT: {symbol} ({bars_data['current_date']}) ${bars_data['current_close']:.2f} {direction} {breakout_label} à {breakout['level']:.2f}")

                            # Met à jour la direction, le niveau et sauvegarde
                            last_breakout_direction = breakout['direction']
                            last_breakout_level = breakout['level']
                            sr_data[symbol]['last_breakout_direction'] = last_breakout_direction
                            sr_data[symbol]['last_breakout_level'] = last_breakout_level

                            # Ajoute le breakout à l'historique
                            breakout_history.append({
                                'level': breakout['level'],
                                'original_type': 'resistance' if breakout['type'] == 'resistance_breakout' else 'support',
                                'breakout_candle': 0,
                                'breakout_date': today,
                                'flip_detected': False  # Track si un flip a déjà été détecté pour ce breakout
                            })

                            # Sauvegarde le nouveau statut
                            self.save_sr_levels(symbol, support_levels, resistance_levels, today, breakout_history, last_breakout_direction, last_breakout_level)

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
