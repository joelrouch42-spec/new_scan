#!/usr/bin/env python3
import pandas as pd
from typing import Dict


class SMCAnalyzer:
    """
    Smart Money Concepts Analyzer
    Détecte: Order Blocks (zones de support/résistance institutionnelles)
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

        # Détection Order Blocks
        if self.ob_config.get('enabled', True):
            result['order_blocks'] = self._detect_order_blocks(df)

        return result


    def _detect_order_blocks(self, df: pd.DataFrame) -> Dict:
        """
        Détecte les Order Blocks (bullish et bearish)
        ET les invalide s'ils ont été cassés par le prix

        Order Block = dernière bougie opposée avant une forte impulsion

        Returns:
            Dict avec 'bullish' et 'bearish' order blocks VALIDES uniquement
        """
        impulse_threshold = self.ob_config.get('impulse_threshold', 2.0)
        min_body_percent = self.ob_config.get('min_body_percent', 0.3)
        min_volume_ratio = self.ob_config.get('min_volume_ratio', 1.0)

        bullish_obs = []
        bearish_obs = []

        # Calculer le body moyen sur 20 bougies pour référence
        df_copy = df.copy()
        df_copy['body'] = abs(df_copy['Close'] - df_copy['Open'])
        df_copy['range'] = df_copy['High'] - df_copy['Low']
        avg_body = df_copy['body'].rolling(window=20, min_periods=1).mean()
        avg_volume = df_copy['Volume'].rolling(window=20, min_periods=1).mean()

        for i in range(1, len(df)):
            current_body = abs(df.iloc[i]['Close'] - df.iloc[i]['Open'])
            current_range = df.iloc[i]['High'] - df.iloc[i]['Low']
            current_volume = df.iloc[i]['Volume']

            # Bougie d'impulsion = body > threshold * moyenne
            is_impulse = current_body > (impulse_threshold * avg_body.iloc[i])

            # Body doit être significatif (pas une doji)
            has_significant_body = (current_body / current_range) > min_body_percent if current_range > 0 else False

            # Volume doit être supérieur à la moyenne
            has_volume = current_volume > (min_volume_ratio * avg_volume.iloc[i])

            if not (is_impulse and has_significant_body and has_volume):
                continue

            # Bullish Order Block: bougie i-1 baissière + bougie i haussière forte
            if df.iloc[i]['Close'] > df.iloc[i]['Open']:  # Bougie i haussière
                if df.iloc[i-1]['Close'] < df.iloc[i-1]['Open']:  # Bougie i-1 baissière
                    ob = {
                        'index': i-1,
                        'low': df.iloc[i-1]['Low'],
                        'high': df.iloc[i-1]['High'],
                        'open': df.iloc[i-1]['Open'],
                        'close': df.iloc[i-1]['Close']
                    }

                    # Vérifier si l'OB est toujours valide (pas cassé depuis)
                    # Un OB bullish est cassé si le prix CLOSE EN DESSOUS de son low
                    is_valid = True
                    for j in range(i, len(df)):
                        if df.iloc[j]['Close'] < ob['low']:
                            is_valid = False
                            break

                    if is_valid:
                        bullish_obs.append(ob)

            # Bearish Order Block: bougie i-1 haussière + bougie i baissière forte
            elif df.iloc[i]['Close'] < df.iloc[i]['Open']:  # Bougie i baissière
                if df.iloc[i-1]['Close'] > df.iloc[i-1]['Open']:  # Bougie i-1 haussière
                    ob = {
                        'index': i-1,
                        'low': df.iloc[i-1]['Low'],
                        'high': df.iloc[i-1]['High'],
                        'open': df.iloc[i-1]['Open'],
                        'close': df.iloc[i-1]['Close']
                    }

                    # Vérifier si l'OB est toujours valide (pas cassé depuis)
                    # Un OB bearish est cassé si le prix CLOSE AU DESSUS de son high
                    is_valid = True
                    for j in range(i, len(df)):
                        if df.iloc[j]['Close'] > ob['high']:
                            is_valid = False
                            break

                    if is_valid:
                        bearish_obs.append(ob)

        return {'bullish': bullish_obs, 'bearish': bearish_obs}
