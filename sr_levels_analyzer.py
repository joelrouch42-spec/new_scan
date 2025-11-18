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
            Dict contenant les niveaux S/R (historique complet) et les cassures détectées
        """
        result = {
            'resistance_levels': [],  # Liste de {level, start_idx, end_idx}
            'support_levels': [],     # Liste de {level, start_idx, end_idx}
            'breaks': []
        }

        if len(df) < self.left_bars + self.right_bars + 1:
            return result

        # Détecter tous les pivots S/R au fil du temps
        resistance_levels = self._detect_all_resistances(df)
        support_levels = self._detect_all_supports(df)

        result['resistance_levels'] = resistance_levels
        result['support_levels'] = support_levels

        # Détecter les cassures si activé
        if self.show_breaks:
            breaks = self._detect_breaks_from_levels(df, resistance_levels, support_levels)
            result['breaks'] = breaks

        return result


    def _detect_all_resistances(self, df: pd.DataFrame) -> List[Dict]:
        """
        Détecte tous les niveaux de résistance au fil du temps

        Returns:
            Liste de {level, start_idx, end_idx}
        """
        resistances = []
        current_resistance = None
        start_idx = None

        # Convertir en numpy pour performance
        highs = df['High'].values

        # Parcourir toutes les bougies pour détecter les pivots
        for i in range(self.left_bars, len(highs) - self.right_bars):
            high_val = highs[i]

            # Vérifier si c'est un pivot high (optimisé avec numpy)
            left_ok = np.all(highs[i] >= highs[i - self.left_bars:i])
            right_ok = np.all(highs[i] >= highs[i + 1:i + self.right_bars + 1])

            if left_ok and right_ok:
                # Nouveau pivot détecté
                if current_resistance is not None:
                    # Terminer le segment précédent
                    resistances.append({
                        'level': current_resistance,
                        'start_idx': start_idx,
                        'end_idx': i
                    })

                # Commencer nouveau segment
                current_resistance = high_val
                start_idx = i

        # Ajouter le dernier segment jusqu'à la fin
        if current_resistance is not None:
            resistances.append({
                'level': current_resistance,
                'start_idx': start_idx,
                'end_idx': len(highs) - 1
            })

        return resistances

    def _detect_all_supports(self, df: pd.DataFrame) -> List[Dict]:
        """
        Détecte tous les niveaux de support au fil du temps

        Returns:
            Liste de {level, start_idx, end_idx}
        """
        supports = []
        current_support = None
        start_idx = None

        # Convertir en numpy pour performance
        lows = df['Low'].values

        # Parcourir toutes les bougies pour détecter les pivots
        for i in range(self.left_bars, len(lows) - self.right_bars):
            low_val = lows[i]

            # Vérifier si c'est un pivot low (optimisé avec numpy)
            left_ok = np.all(lows[i] <= lows[i - self.left_bars:i])
            right_ok = np.all(lows[i] <= lows[i + 1:i + self.right_bars + 1])

            if left_ok and right_ok:
                # Nouveau pivot détecté
                if current_support is not None:
                    # Terminer le segment précédent
                    supports.append({
                        'level': current_support,
                        'start_idx': start_idx,
                        'end_idx': i
                    })

                # Commencer nouveau segment
                current_support = low_val
                start_idx = i

        # Ajouter le dernier segment jusqu'à la fin
        if current_support is not None:
            supports.append({
                'level': current_support,
                'start_idx': start_idx,
                'end_idx': len(lows) - 1
            })

        return supports

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


    def _detect_breaks_from_levels(self, df: pd.DataFrame,
                                   resistance_levels: List[Dict],
                                   support_levels: List[Dict]) -> List[Dict]:
        """
        Détecte les cassures de support/résistance pour tous les niveaux

        Returns:
            Liste de cassures avec type, index, et description
        """
        breaks = []

        # Calculer oscillateur de volume
        vol_osc = self._calculate_volume_oscillator(df)

        # Convertir en numpy arrays pour performance
        closes = df['Close'].values
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        vol_osc_arr = vol_osc.values

        # Parcourir toutes les bougies pour détecter les cassures
        for i in range(1, len(df)):
            close_prev = closes[i-1]
            close_curr = closes[i]
            open_curr = opens[i]
            high_curr = highs[i]
            low_curr = lows[i]
            vol_curr = vol_osc_arr[i]

            # Trouver le niveau de support actif à cet instant
            active_support = None
            for lvl in support_levels:
                if lvl['start_idx'] <= i <= lvl['end_idx']:
                    active_support = lvl['level']
                    break

            # Cassure de support
            if active_support is not None and close_prev >= active_support and close_curr < active_support:
                # Bear Wick: longue mèche haute (open - close < high - open)
                if open_curr - close_curr < high_curr - open_curr:
                    breaks.append({
                        'type': 'bear_wick',
                        'level': active_support,
                        'index': i,
                        'price': close_curr,
                        'description': 'Bear Wick'
                    })
                # Cassure normale avec volume
                elif vol_curr > self.volume_threshold:
                    breaks.append({
                        'type': 'support_break',
                        'level': active_support,
                        'index': i,
                        'price': close_curr,
                        'description': 'Support Break'
                    })

            # Trouver le niveau de résistance actif à cet instant
            active_resistance = None
            for lvl in resistance_levels:
                if lvl['start_idx'] <= i <= lvl['end_idx']:
                    active_resistance = lvl['level']
                    break

            # Cassure de résistance
            if active_resistance is not None and close_prev <= active_resistance and close_curr > active_resistance:
                # Bull Wick: longue mèche basse (open - low > close - open)
                if open_curr - low_curr > close_curr - open_curr:
                    breaks.append({
                        'type': 'bull_wick',
                        'level': active_resistance,
                        'index': i,
                        'price': close_curr,
                        'description': 'Bull Wick'
                    })
                # Cassure normale avec volume
                elif vol_curr > self.volume_threshold:
                    breaks.append({
                        'type': 'resistance_break',
                        'level': active_resistance,
                        'index': i,
                        'price': close_curr,
                        'description': 'Resistance Break'
                    })

        return breaks

    def _detect_breaks(self, df: pd.DataFrame, resistance: Optional[float],
                      support: Optional[float]) -> List[Dict]:
        """
        DEPRECATED: Ancienne méthode gardée pour compatibilité
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
