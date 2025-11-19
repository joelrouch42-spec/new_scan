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
            Dict contenant les signaux buy/sell
        """
        result = {
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

            # Signal d'achat: Ligne passe de rouge à verte
            # (MACD croise au-dessus de Signal)
            prev_line_green = macd[i-1] > signal[i-1]
            if line_green and not prev_line_green:
                result['buy_signals'].append({
                    'index': i,
                    'price': df.iloc[i]['Close'],
                    'macd': macd[i],
                    'signal': signal[i],
                    'histogram': hist_curr
                })

            # Signal de vente: Ligne passe de verte à rouge
            # (MACD croise en-dessous de Signal)
            prev_line_red = macd[i-1] < signal[i-1]
            if line_red and not prev_line_red:
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
