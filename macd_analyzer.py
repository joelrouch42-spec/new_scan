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
            Dict contenant les croisements MACD/Signal
        """
        result = {
            'crossovers': []  # Liste de {index, price, type}
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

        # Détecter les croisements
        crossovers = self._detect_crossovers(df, macd, signal)
        result['crossovers'] = crossovers

        print(f"DEBUG MACD: {len(crossovers)} crossovers détectés")
        if len(crossovers) > 0:
            print(f"  Premier: index={crossovers[0]['index']}, type={crossovers[0]['type']}, price={crossovers[0]['price']:.2f}")
        print(f"  MACD range: {macd[macd != 0].min():.4f} to {macd.max():.4f}")
        print(f"  Signal range: {signal[signal != 0].min():.4f} to {signal.max():.4f}")

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
