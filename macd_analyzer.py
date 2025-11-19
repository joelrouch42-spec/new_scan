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
        Retourne uniquement l'état de la ligne MACD (verte ou rouge)

        Args:
            df: DataFrame avec colonnes Open, High, Low, Close, Volume

        Returns:
            Dict contenant les valeurs MACD et l'état de la ligne
        """
        result = {
            'values': []  # Valeurs MACD avec état de la ligne (green/red)
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

        # Analyser l'état de la ligne MACD
        start_idx = max(self.slow_length, self.signal_length)

        for i in range(start_idx, len(df)):
            # Déterminer la couleur de la ligne MACD
            # Verte si MACD > Signal, Rouge si MACD < Signal
            line_color = 'green' if macd[i] > signal[i] else 'red'

            # Stocker toutes les valeurs
            result['values'].append({
                'index': i,
                'macd': macd[i],
                'signal': signal[i],
                'histogram': histogram[i],
                'line_color': line_color
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
