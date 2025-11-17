#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class SMCAnalyzer:
    """
    Smart Money Concepts Analyzer - Port exact du PineScript TradingView
    Détecte: Order Blocks basés sur Market Structure Breaks
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

        if len(df) < 20:
            return result

        # Détection Order Blocks avec zigzag (port exact TradingView)
        if self.ob_config.get('enabled', True):
            result['order_blocks'] = self._detect_order_blocks_tv(df)

        return result


    def _detect_order_blocks_tv(self, df: pd.DataFrame) -> Dict:
        """
        Port exact du code TradingView pour détecter les Order Blocks

        Logique (identique au PineScript):
        1. Détecte zigzag pivots (to_up, to_down)
        2. Track trend changes
        3. Stocke high_points/low_points avec indices
        4. Détecte Market Structure Breaks (market variable)
        5. Quand MSB bullish: trouve dernière bougie rouge entre h1i et l0i
        6. Quand MSB bearish: trouve dernière bougie verte entre l1i et h0i

        Returns:
            Dict avec 'bullish' et 'bearish' order blocks
        """
        zigzag_len = self.ob_config.get('zigzag_length', 9)
        fib_factor = self.ob_config.get('fib_factor', 0.33)

        bullish_obs = []
        bearish_obs = []

        # Arrays pour stocker les pivots (comme dans PineScript)
        high_points_arr = []
        high_index_arr = []
        low_points_arr = []
        low_index_arr = []

        # Variables d'état
        trend = 1  # 1 = up, -1 = down
        market = 1  # market structure: 1 = bullish, -1 = bearish

        # Étape 1: Détection des pivots zigzag (comme PineScript)
        for i in range(zigzag_len, len(df)):
            # to_up = high >= ta.highest(zigzag_len)
            to_up = df['High'].iloc[i] >= df['High'].iloc[i-zigzag_len:i+1].max()

            # to_down = low <= ta.lowest(zigzag_len)
            to_down = df['Low'].iloc[i] <= df['Low'].iloc[i-zigzag_len:i+1].min()

            # Changement de trend
            prev_trend = trend
            if trend == 1 and to_down:
                trend = -1
            elif trend == -1 and to_up:
                trend = 1

            # Si trend a changé, stocke le pivot
            if prev_trend != trend:
                if trend == 1:
                    # On vient de passer en uptrend, donc on a un pivot bas
                    # Cherche le plus bas depuis le dernier pivot haut
                    last_trend_up_since = 0
                    for j in range(i, -1, -1):
                        if j < len(high_index_arr) and j >= high_index_arr[-1] if high_index_arr else True:
                            last_trend_up_since += 1
                        else:
                            break

                    if last_trend_up_since > 0:
                        low_val = df['Low'].iloc[max(0, i-last_trend_up_since):i+1].min()
                        low_index = i - (df['Low'].iloc[max(0, i-last_trend_up_since):i+1][::-1] == low_val).idxmax()
                    else:
                        low_val = df['Low'].iloc[i]
                        low_index = i

                    low_points_arr.append(low_val)
                    low_index_arr.append(low_index)

                if trend == -1:
                    # On vient de passer en downtrend, donc on a un pivot haut
                    last_trend_down_since = 0
                    for j in range(i, -1, -1):
                        if j < len(low_index_arr) and j >= low_index_arr[-1] if low_index_arr else True:
                            last_trend_down_since += 1
                        else:
                            break

                    if last_trend_down_since > 0:
                        high_val = df['High'].iloc[max(0, i-last_trend_down_since):i+1].max()
                        high_index = i - (df['High'].iloc[max(0, i-last_trend_down_since):i+1][::-1] == high_val).idxmax()
                    else:
                        high_val = df['High'].iloc[i]
                        high_index = i

                    high_points_arr.append(high_val)
                    high_index_arr.append(high_index)

        # Besoin d'au moins 2 pivots de chaque type
        if len(high_points_arr) < 2 or len(low_points_arr) < 2:
            return {'bullish': [], 'bearish': []}

        # Étape 2: Détection des Market Structure Breaks (comme PineScript)
        # On itère sur toutes les combinaisons de pivots
        market = 1

        for i in range(2, min(len(high_points_arr), len(low_points_arr))):
            # Get pivots (notation PineScript: 0=latest, 1=previous)
            h0 = high_points_arr[-1] if i >= len(high_points_arr) else high_points_arr[i]
            h0i = high_index_arr[-1] if i >= len(high_index_arr) else high_index_arr[i]
            h1 = high_points_arr[i-1]
            h1i = high_index_arr[i-1]

            l0 = low_points_arr[-1] if i >= len(low_points_arr) else low_points_arr[i]
            l0i = low_index_arr[-1] if i >= len(low_index_arr) else low_index_arr[i]
            l1 = low_points_arr[i-1]
            l1i = low_index_arr[i-1]

            prev_market = market

            # MSB Bullish: market == -1 and h0 > h1 and h0 > h1 + abs(h1 - l0) * fib_factor
            if market == -1 and h0 > h1 and h0 > h1 + abs(h1 - l0) * fib_factor:
                market = 1

                # Bu-OB: dernière bougie rouge entre h1i et l0i
                bu_ob_index = None
                for j in range(h1i, min(l0i + 1, len(df))):
                    if df['Open'].iloc[j] > df['Close'].iloc[j]:  # Bougie rouge
                        bu_ob_index = j

                if bu_ob_index is not None:
                    bullish_obs.append({
                        'index': bu_ob_index,
                        'low': df['Low'].iloc[bu_ob_index],
                        'high': df['High'].iloc[bu_ob_index],
                        'open': df['Open'].iloc[bu_ob_index],
                        'close': df['Close'].iloc[bu_ob_index]
                    })

            # MSB Bearish: market == 1 and l0 < l1 and l0 < l1 - abs(h0 - l1) * fib_factor
            if market == 1 and l0 < l1 and l0 < l1 - abs(h0 - l1) * fib_factor:
                market = -1

                # Be-OB: dernière bougie verte entre l1i et h0i
                be_ob_index = None
                for j in range(l1i, min(h0i + 1, len(df))):
                    if df['Open'].iloc[j] < df['Close'].iloc[j]:  # Bougie verte
                        be_ob_index = j

                if be_ob_index is not None:
                    bearish_obs.append({
                        'index': be_ob_index,
                        'low': df['Low'].iloc[be_ob_index],
                        'high': df['High'].iloc[be_ob_index],
                        'open': df['Open'].iloc[be_ob_index],
                        'close': df['Close'].iloc[be_ob_index]
                    })

        return {'bullish': bullish_obs, 'bearish': bearish_obs}
