#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class SMCAnalyzer:
    """
    Smart Money Concepts Analyzer avec Zigzag
    Détecte: Order Blocks basés sur Market Structure
    """

    def __init__(self, config: dict):
        """
        Initialise l'analyseur SMC avec la configuration

        Args:
            config: Configuration SMC depuis patterns.json
        """
        self.config = config
        self.ob_config = config.get('order_blocks', {})


    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyse complète SMC sur le DataFrame

        Args:
            df: DataFrame avec colonnes Open, High, Low, Close, Volume

        Returns:
            Dict contenant les Order Blocks détectés
        """
        result = {
            'order_blocks': {'bullish': [], 'bearish': []}
        }

        if len(df) < 10:
            return result

        # Détection Order Blocks avec zigzag
        if self.ob_config.get('enabled', True):
            result['order_blocks'] = self._detect_order_blocks_zigzag(df)

        return result


    def _detect_order_blocks_zigzag(self, df: pd.DataFrame) -> Dict:
        """
        Détecte les Order Blocks basés sur la structure zigzag du marché

        Logique:
        1. Calcule zigzag (pivots hauts/bas)
        2. Détecte Market Structure Break (MSB)
        3. OB Bullish = dernière bougie rouge entre pivot bas précédent et pivot haut actuel
        4. OB Bearish = dernière bougie verte entre pivot haut précédent et pivot bas actuel

        Returns:
            Dict avec 'bullish' et 'bearish' order blocks VALIDES uniquement
        """
        zigzag_len = self.ob_config.get('zigzag_length', 9)
        fib_factor = self.ob_config.get('fib_factor', 0.33)

        bullish_obs = []
        bearish_obs = []

        # Arrays pour stocker les pivots
        high_points = []
        high_indices = []
        low_points = []
        low_indices = []

        # Détection des pivots zigzag
        trend = 1  # 1 = up, -1 = down

        for i in range(zigzag_len, len(df) - zigzag_len):
            # Check si c'est un pivot haut
            window_high = df['High'].iloc[i-zigzag_len:i+zigzag_len+1]
            if df['High'].iloc[i] >= window_high.max():
                if trend == 1:
                    # Update dernier pivot haut
                    if high_points and high_indices[-1] > i - 2*zigzag_len:
                        if df['High'].iloc[i] > high_points[-1]:
                            high_points[-1] = df['High'].iloc[i]
                            high_indices[-1] = i
                    else:
                        high_points.append(df['High'].iloc[i])
                        high_indices.append(i)
                else:
                    # Changement de tendance down -> up
                    high_points.append(df['High'].iloc[i])
                    high_indices.append(i)
                    trend = 1

            # Check si c'est un pivot bas
            window_low = df['Low'].iloc[i-zigzag_len:i+zigzag_len+1]
            if df['Low'].iloc[i] <= window_low.min():
                if trend == -1:
                    # Update dernier pivot bas
                    if low_points and low_indices[-1] > i - 2*zigzag_len:
                        if df['Low'].iloc[i] < low_points[-1]:
                            low_points[-1] = df['Low'].iloc[i]
                            low_indices[-1] = i
                    else:
                        low_points.append(df['Low'].iloc[i])
                        low_indices.append(i)
                else:
                    # Changement de tendance up -> down
                    low_points.append(df['Low'].iloc[i])
                    low_indices.append(i)
                    trend = -1

        # Besoin d'au moins 2 pivots hauts et 2 pivots bas
        if len(high_points) < 2 or len(low_points) < 2:
            return {'bullish': [], 'bearish': []}

        # Détecter les Market Structure Breaks et Order Blocks
        market_structure = 1  # 1 = bullish, -1 = bearish

        for i in range(1, min(len(high_points), len(low_points))):
            # MSB Bullish: pivot haut actuel > pivot haut précédent
            if len(high_points) > i and len(low_points) > i-1:
                h0 = high_points[i]
                h0i = high_indices[i]
                h1 = high_points[i-1]
                h1i = high_indices[i-1]
                l0 = low_points[i-1] if i-1 < len(low_points) else low_points[-1]
                l0i = low_indices[i-1] if i-1 < len(low_indices) else low_indices[-1]
                l1 = low_points[i-2] if i-2 >= 0 else l0
                l1i = low_indices[i-2] if i-2 >= 0 else l0i

                # Check MSB Bullish
                if l0 < l1 and h0 > h1 + (h1 - l1) * fib_factor:
                    if market_structure != 1:
                        # Chercher dernière bougie rouge entre h1 et l0
                        ob_index = None
                        for j in range(h1i, min(l0i + 1, len(df))):
                            if df['Close'].iloc[j] < df['Open'].iloc[j]:
                                ob_index = j

                        if ob_index is not None:
                            ob = {
                                'index': ob_index,
                                'low': df['Low'].iloc[ob_index],
                                'high': df['High'].iloc[ob_index],
                                'open': df['Open'].iloc[ob_index],
                                'close': df['Close'].iloc[ob_index]
                            }
                            bullish_obs.append(ob)

                        market_structure = 1

            # MSB Bearish: pivot bas actuel < pivot bas précédent
            if len(low_points) > i and len(high_points) > i-1:
                l0 = low_points[i]
                l0i = low_indices[i]
                l1 = low_points[i-1]
                l1i = low_indices[i-1]
                h0 = high_points[i-1] if i-1 < len(high_points) else high_points[-1]
                h0i = high_indices[i-1] if i-1 < len(high_indices) else high_indices[-1]
                h1 = high_points[i-2] if i-2 >= 0 else h0
                h1i = high_indices[i-2] if i-2 >= 0 else h0i

                # Check MSB Bearish
                if h0 > h1 and l0 < l1 - (h1 - l1) * fib_factor:
                    if market_structure != -1:
                        # Chercher dernière bougie verte entre l1 et h0
                        ob_index = None
                        for j in range(l1i, min(h0i + 1, len(df))):
                            if df['Close'].iloc[j] > df['Open'].iloc[j]:
                                ob_index = j

                        if ob_index is not None:
                            ob = {
                                'index': ob_index,
                                'low': df['Low'].iloc[ob_index],
                                'high': df['High'].iloc[ob_index],
                                'open': df['Open'].iloc[ob_index],
                                'close': df['Close'].iloc[ob_index]
                            }
                            bearish_obs.append(ob)

                        market_structure = -1

        return {'bullish': bullish_obs, 'bearish': bearish_obs}
