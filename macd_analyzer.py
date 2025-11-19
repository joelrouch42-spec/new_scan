#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import List, Dict


class MACDAnalyzer:
    """
    MACD Analyzer - Détecte les croisements MACD/Signal
    Port du code TradingView CM_MacD_Ult_MTF
    """

    def __init__(self, config: dict):
        """
        Initialise l'analyseur MACD

        Args:
            config: Configuration depuis settings.json
        """
        self.config = config
        self.fast_length = config.get('fast_length', 12)
        self.slow_length = config.get('slow_length', 26)
        self.signal_length = config.get('signal_length', 9)


    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyse complète MACD sur le DataFrame

        Args:
            df: DataFrame avec colonnes Open, High, Low, Close, Volume

        Returns:
            Dict contenant les croisements MACD/Signal et les signaux buy/sell
        """
        result = {
            'crossovers': [],  # Liste de {index, price, type}
            'buy_signals': [],  # Ligne verte + histogramme lime
            'sell_signals': [], # Ligne rouge + histogramme maroon
            'values': []       # Toutes les valeurs pour analyse
        }

        if len(df) < max(self.slow_length, self.signal_length) + 1:
            return result

        # Calculer MACD
        closes = df['Close'].values
        fast_ema = self._ema(closes, self.fast_length)
        slow_ema = self._ema(closes, self.slow_length)
        macd = fast_ema - slow_ema

        # Calculer Signal (SMA du MACD)
        signal = self._sma(macd, self.signal_length)

        # Calculer l'histogramme
        histogram = macd - signal

        # Détecter les croisements
        crossovers = self._detect_crossovers(df, macd, signal)
        result['crossovers'] = crossovers

        # Détecter les signaux buy/sell basés sur ligne + histogramme
        start_idx = max(self.slow_length, self.signal_length)

        for i in range(start_idx + 1, len(df)):
            hist_curr = histogram[i]
            hist_prev = histogram[i-1]

            # Déterminer la couleur de la ligne épaisse
            # Verte si MACD > Signal, Rouge si MACD < Signal
            line_green = macd[i] > signal[i]
            line_red = macd[i] < signal[i]

            # Déterminer la couleur de l'histogramme
            # Lime: hist > 0 et hist > hist_prev
            # Green: hist > 0 et hist <= hist_prev
            # Red: hist < 0 et hist < hist_prev
            # Maroon: hist < 0 et hist >= hist_prev

            if hist_curr > 0:
                if hist_curr > hist_prev:
                    hist_color = 'lime'
                else:
                    hist_color = 'green'
            else:
                if hist_curr < hist_prev:
                    hist_color = 'red'
                else:
                    hist_color = 'maroon'

            # Signal d'achat: Ligne verte ET histogramme lime
            if line_green and hist_color == 'lime':
                # Vérifier si c'est un nouveau signal (pas déjà actif)
                is_new = True
                if i > start_idx + 1:
                    prev_line_green = macd[i-1] > signal[i-1]
                    prev_hist_curr = histogram[i-1]
                    prev_hist_prev = histogram[i-2]
                    prev_hist_color = 'lime' if prev_hist_curr > 0 and prev_hist_curr > prev_hist_prev else None
                    if prev_line_green and prev_hist_color == 'lime':
                        is_new = False

                if is_new:
                    result['buy_signals'].append({
                        'index': i,
                        'price': df.iloc[i]['Close'],
                        'macd': macd[i],
                        'signal': signal[i],
                        'histogram': hist_curr
                    })

            # Signal de vente: Ligne rouge ET histogramme maroon
            elif line_red and hist_color == 'maroon':
                # Vérifier si c'est un nouveau signal (pas déjà actif)
                is_new = True
                if i > start_idx + 1:
                    prev_line_red = macd[i-1] < signal[i-1]
                    prev_hist_curr = histogram[i-1]
                    prev_hist_prev = histogram[i-2]
                    prev_hist_color = 'maroon' if prev_hist_curr < 0 and prev_hist_curr >= prev_hist_prev else None
                    if prev_line_red and prev_hist_color == 'maroon':
                        is_new = False

                if is_new:
                    result['sell_signals'].append({
                        'index': i,
                        'price': df.iloc[i]['Close'],
                        'macd': macd[i],
                        'signal': signal[i],
                        'histogram': hist_curr
                    })

            # Stocker toutes les valeurs
            result['values'].append({
                'index': i,
                'macd': macd[i],
                'signal': signal[i],
                'histogram': hist_curr,
                'line_color': 'green' if line_green else 'red',
                'hist_color': hist_color
            })

        return result


    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calcule l'EMA (Exponential Moving Average)
        """
        ema = np.zeros_like(data, dtype=float)
        multiplier = 2.0 / (period + 1)

        # Première valeur = SMA
        ema[period-1] = np.mean(data[:period])

        # Calcul EMA
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema


    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calcule la SMA (Simple Moving Average)
        """
        sma = np.zeros_like(data, dtype=float)

        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])

        return sma


    def _detect_crossovers(self, df: pd.DataFrame, macd: np.ndarray, signal: np.ndarray) -> List[Dict]:
        """
        Détecte les croisements entre MACD et Signal

        Returns:
            Liste de croisements avec index, prix, et type
        """
        crossovers = []

        # Parcourir les données pour détecter les croisements
        # Commencer après que l'EMA slow soit initialisée
        start_idx = max(self.slow_length, self.signal_length)

        for i in range(start_idx + 1, len(macd)):
            macd_prev = macd[i-1]
            macd_curr = macd[i]
            signal_prev = signal[i-1]
            signal_curr = signal[i]

            # Croisement: MACD croise Signal
            # cross(outMacD, outSignal) dans PineScript
            crossed = False

            # Bullish cross: MACD passe au-dessus de Signal
            if macd_prev <= signal_prev and macd_curr > signal_curr:
                crossed = True
                cross_type = 'bullish'

            # Bearish cross: MACD passe en-dessous de Signal
            elif macd_prev >= signal_prev and macd_curr < signal_curr:
                crossed = True
                cross_type = 'bearish'

            if crossed:
                crossovers.append({
                    'index': i,
                    'price': df.iloc[i]['Close'],
                    'type': cross_type,
                    'macd': macd_curr,
                    'signal': signal_curr
                })

        return crossovers
