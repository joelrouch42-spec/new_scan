#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class SRLevelsAnalyzer:
    """
    Support and Resistance Levels Analyzer with Breaks Detection
    Port du code TradingView LuxAlgo
    """

    def __init__(self, config: dict):
        """
        Initialise l'analyseur S/R

        Args:
            config: Configuration depuis patterns.json
        """
        self.config = config
        self.left_bars = config.get('left_bars', 15)
        self.right_bars = config.get('right_bars', 15)
        self.volume_threshold = config.get('volume_threshold', 20)
        self.show_breaks = config.get('show_breaks', True)


    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyse complète S/R sur le DataFrame

        Args:
            df: DataFrame avec colonnes Open, High, Low, Close, Volume

        Returns:
            Dict contenant les niveaux S/R et les cassures détectées
        """
        result = {
            'resistance': None,
            'support': None,
            'breaks': []
        }

        if len(df) < self.left_bars + self.right_bars + 1:
            return result

        # Détecter les pivots (support/résistance)
        resistance_level = self._detect_resistance(df)
        support_level = self._detect_support(df)

        result['resistance'] = resistance_level
        result['support'] = support_level

        # Détecter les cassures si activé
        if self.show_breaks:
            breaks = self._detect_breaks(df, resistance_level, support_level)
            result['breaks'] = breaks

        return result


    def _detect_resistance(self, df: pd.DataFrame) -> Optional[float]:
        """
        Détecte le niveau de résistance actuel (pivot high)

        PineScript: highUsePivot = fixnan(pivothigh(leftBars, rightBars)[1])
        """
        # pivothigh: trouve les sommets locaux
        for i in range(len(df) - self.right_bars - 1, self.left_bars - 1, -1):
            high_val = df['High'].iloc[i]

            # Vérifier si c'est un pivot high
            left_ok = all(df['High'].iloc[i] >= df['High'].iloc[j]
                         for j in range(i - self.left_bars, i))
            right_ok = all(df['High'].iloc[i] >= df['High'].iloc[j]
                          for j in range(i + 1, min(i + self.right_bars + 1, len(df))))

            if left_ok and right_ok:
                return high_val

        return None


    def _detect_support(self, df: pd.DataFrame) -> Optional[float]:
        """
        Détecte le niveau de support actuel (pivot low)

        PineScript: lowUsePivot = fixnan(pivotlow(leftBars, rightBars)[1])
        """
        # pivotlow: trouve les creux locaux
        for i in range(len(df) - self.right_bars - 1, self.left_bars - 1, -1):
            low_val = df['Low'].iloc[i]

            # Vérifier si c'est un pivot low
            left_ok = all(df['Low'].iloc[i] <= df['Low'].iloc[j]
                         for j in range(i - self.left_bars, i))
            right_ok = all(df['Low'].iloc[i] <= df['Low'].iloc[j]
                          for j in range(i + 1, min(i + self.right_bars + 1, len(df))))

            if left_ok and right_ok:
                return low_val

        return None


    def _calculate_volume_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule l'oscillateur de volume

        PineScript:
        short = ema(volume, 5)
        long = ema(volume, 10)
        osc = 100 * (short - long) / long
        """
        short_ema = df['Volume'].ewm(span=5, adjust=False).mean()
        long_ema = df['Volume'].ewm(span=10, adjust=False).mean()
        osc = 100 * (short_ema - long_ema) / long_ema
        return osc


    def _detect_breaks(self, df: pd.DataFrame, resistance: Optional[float],
                      support: Optional[float]) -> List[Dict]:
        """
        Détecte les cassures de support/résistance

        Returns:
            Liste de cassures avec type, index, et description
        """
        breaks = []

        if resistance is None and support is None:
            return breaks

        # Calculer oscillateur de volume
        vol_osc = self._calculate_volume_oscillator(df)

        # Parcourir les dernières bougies pour détecter les cassures
        for i in range(1, len(df)):
            close_prev = df['Close'].iloc[i-1]
            close_curr = df['Close'].iloc[i]
            open_curr = df['Open'].iloc[i]
            high_curr = df['High'].iloc[i]
            low_curr = df['Low'].iloc[i]
            vol_curr = vol_osc.iloc[i]

            # Cassure de support
            if support is not None and close_prev >= support and close_curr < support:
                # Bear Wick: longue mèche haute (open - close < high - open)
                if open_curr - close_curr < high_curr - open_curr:
                    breaks.append({
                        'type': 'bear_wick',
                        'level': support,
                        'index': i,
                        'price': close_curr,
                        'description': 'Bear Wick'
                    })
                # Cassure normale avec volume
                elif vol_curr > self.volume_threshold:
                    breaks.append({
                        'type': 'support_break',
                        'level': support,
                        'index': i,
                        'price': close_curr,
                        'description': 'Support Break'
                    })

            # Cassure de résistance
            if resistance is not None and close_prev <= resistance and close_curr > resistance:
                # Bull Wick: longue mèche basse (open - low > close - open)
                if open_curr - low_curr > close_curr - open_curr:
                    breaks.append({
                        'type': 'bull_wick',
                        'level': resistance,
                        'index': i,
                        'price': close_curr,
                        'description': 'Bull Wick'
                    })
                # Cassure normale avec volume
                elif vol_curr > self.volume_threshold:
                    breaks.append({
                        'type': 'resistance_break',
                        'level': resistance,
                        'index': i,
                        'price': close_curr,
                        'description': 'Resistance Break'
                    })

        return breaks
