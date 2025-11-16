#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import List, Dict, Tuple, Optional


class SMCAnalyzer:
    """
    Smart Money Concepts Analyzer
    Détecte: Order Blocks, Fair Value Gaps, BOS, CHoCH, Liquidity Levels, Premium/Discount zones
    """

    def __init__(self, config: dict):
        """
        Initialise l'analyseur SMC avec la configuration

        Args:
            config: Configuration SMC depuis patterns.json
        """
        self.config = config
        self.ob_config = config.get('order_blocks', {})
        self.fvg_config = config.get('fair_value_gaps', {})
        self.bos_config = config.get('break_of_structure', {})
        self.liq_config = config.get('liquidity', {})
        self.pd_config = config.get('premium_discount', {})


    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyse complète SMC sur le DataFrame

        Args:
            df: DataFrame avec colonnes Open, High, Low, Close, Volume

        Returns:
            Dict contenant tous les éléments SMC détectés
        """
        result = {
            'order_blocks': {'bullish': [], 'bearish': []},
            'fvg': {'bullish': [], 'bearish': []},
            'bos': [],
            'choch': [],
            'liquidity': {
                'weak_highs': [],
                'weak_lows': [],
                'strong_highs': [],
                'strong_lows': []
            },
            'premium_discount': {}
        }

        if len(df) < 10:
            return result

        # Détection Order Blocks
        if self.ob_config.get('enabled', True):
            result['order_blocks'] = self._detect_order_blocks(df)

        # Détection Fair Value Gaps
        if self.fvg_config.get('enabled', True):
            result['fvg'] = self._detect_fair_value_gaps(df)

        # Détection BOS et CHoCH
        if self.bos_config.get('enabled', True):
            swings = self._find_swing_points(df)
            result['bos'] = self._detect_bos(df, swings)
            result['choch'] = self._detect_choch(df, swings)

        # Détection Liquidity Levels
        if self.liq_config.get('enabled', True):
            result['liquidity'] = self._detect_liquidity_levels(df)

        # Calcul Premium/Discount zones
        if self.pd_config.get('enabled', True):
            result['premium_discount'] = self._calculate_premium_discount(df)

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

        bullish_obs = []
        bearish_obs = []

        # Calculer le body moyen sur 20 bougies pour référence
        df_copy = df.copy()
        df_copy['body'] = abs(df_copy['Close'] - df_copy['Open'])
        df_copy['range'] = df_copy['High'] - df_copy['Low']
        avg_body = df_copy['body'].rolling(window=20, min_periods=1).mean()

        for i in range(1, len(df)):
            current_body = abs(df.iloc[i]['Close'] - df.iloc[i]['Open'])
            prev_body = abs(df.iloc[i-1]['Close'] - df.iloc[i-1]['Open'])
            current_range = df.iloc[i]['High'] - df.iloc[i]['Low']

            # Bougie d'impulsion = body > threshold * moyenne
            is_impulse = current_body > (impulse_threshold * avg_body.iloc[i])

            # Body doit être significatif (pas une doji)
            has_significant_body = (current_body / current_range) > min_body_percent if current_range > 0 else False

            if not (is_impulse and has_significant_body):
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


    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> Dict:
        """
        Détecte les Fair Value Gaps (FVG)

        FVG = gap entre la bougie i-1 et la bougie i+1
        - Bullish FVG: low[i+1] > high[i-1]
        - Bearish FVG: high[i+1] < low[i-1]

        Returns:
            Dict avec 'bullish' et 'bearish' FVGs
        """
        min_gap_percent = self.fvg_config.get('min_gap_percent', 0.001)

        bullish_fvgs = []
        bearish_fvgs = []

        for i in range(1, len(df) - 1):
            high_prev = df.iloc[i-1]['High']
            low_prev = df.iloc[i-1]['Low']
            high_next = df.iloc[i+1]['High']
            low_next = df.iloc[i+1]['Low']

            # Bullish FVG
            if low_next > high_prev:
                gap_size = low_next - high_prev
                gap_percent = gap_size / high_prev

                if gap_percent >= min_gap_percent:
                    bullish_fvgs.append({
                        'index': i,
                        'low': high_prev,
                        'high': low_next,
                        'gap_size': gap_size
                    })

            # Bearish FVG
            if high_next < low_prev:
                gap_size = low_prev - high_next
                gap_percent = gap_size / low_prev

                if gap_percent >= min_gap_percent:
                    bearish_fvgs.append({
                        'index': i,
                        'low': high_next,
                        'high': low_prev,
                        'gap_size': gap_size
                    })

        return {'bullish': bullish_fvgs, 'bearish': bearish_fvgs}


    def _find_swing_points(self, df: pd.DataFrame) -> Dict:
        """
        Trouve les swing highs et swing lows

        Returns:
            Dict avec 'highs' et 'lows' (indices et valeurs)
        """
        swing_order = self.bos_config.get('swing_order', 5)

        highs = df['High'].values
        lows = df['Low'].values

        # Trouver les extrema locaux
        swing_high_idx = argrelextrema(highs, np.greater, order=swing_order)[0]
        swing_low_idx = argrelextrema(lows, np.less, order=swing_order)[0]

        swing_highs = [{'index': idx, 'price': highs[idx]} for idx in swing_high_idx]
        swing_lows = [{'index': idx, 'price': lows[idx]} for idx in swing_low_idx]

        return {'highs': swing_highs, 'lows': swing_lows}


    def _detect_bos(self, df: pd.DataFrame, swings: Dict) -> List[Dict]:
        """
        Détecte les Break of Structure (BOS)

        BOS = cassure d'un swing high/low précédent dans la direction de la tendance

        Args:
            df: DataFrame
            swings: Dict des swing points

        Returns:
            List de BOS détectés
        """
        bos_list = []
        swing_highs = swings['highs']
        swing_lows = swings['lows']

        # Détecter BOS haussiers (cassure de swing highs)
        for i in range(1, len(swing_highs)):
            prev_high = swing_highs[i-1]['price']
            current_idx = swing_highs[i]['index']

            # Vérifier si le prix a cassé le swing high précédent
            if df.iloc[current_idx]['Close'] > prev_high:
                bos_list.append({
                    'index': current_idx,
                    'type': 'bullish',
                    'price': prev_high,
                    'broken_swing_idx': swing_highs[i-1]['index']
                })

        # Détecter BOS baissiers (cassure de swing lows)
        for i in range(1, len(swing_lows)):
            prev_low = swing_lows[i-1]['price']
            current_idx = swing_lows[i]['index']

            # Vérifier si le prix a cassé le swing low précédent
            if df.iloc[current_idx]['Close'] < prev_low:
                bos_list.append({
                    'index': current_idx,
                    'type': 'bearish',
                    'price': prev_low,
                    'broken_swing_idx': swing_lows[i-1]['index']
                })

        return sorted(bos_list, key=lambda x: x['index'])


    def _detect_choch(self, df: pd.DataFrame, swings: Dict) -> List[Dict]:
        """
        Détecte les Change of Character (CHoCH)

        CHoCH = cassure d'un swing dans la direction OPPOSEE à la tendance
        (signal de retournement potentiel)

        Args:
            df: DataFrame
            swings: Dict des swing points

        Returns:
            List de CHoCH détectés
        """
        choch_list = []
        swing_highs = swings['highs']
        swing_lows = swings['lows']

        # CHoCH baissier: en tendance haussière, on casse un swing low
        for i in range(1, len(swing_lows)):
            prev_low = swing_lows[i-1]['price']
            current_idx = swing_lows[i]['index']

            # Chercher si on était en tendance haussière avant
            recent_highs = [h for h in swing_highs if h['index'] < current_idx]
            if len(recent_highs) >= 2:
                # Tendance haussière = highs de plus en plus hauts
                if recent_highs[-1]['price'] > recent_highs[-2]['price']:
                    # Et maintenant on casse un low précédent
                    if df.iloc[current_idx]['Close'] < prev_low:
                        choch_list.append({
                            'index': current_idx,
                            'type': 'bearish',
                            'price': prev_low,
                            'broken_swing_idx': swing_lows[i-1]['index']
                        })

        # CHoCH haussier: en tendance baissière, on casse un swing high
        for i in range(1, len(swing_highs)):
            prev_high = swing_highs[i-1]['price']
            current_idx = swing_highs[i]['index']

            # Chercher si on était en tendance baissière avant
            recent_lows = [l for l in swing_lows if l['index'] < current_idx]
            if len(recent_lows) >= 2:
                # Tendance baissière = lows de plus en plus bas
                if recent_lows[-1]['price'] < recent_lows[-2]['price']:
                    # Et maintenant on casse un high précédent
                    if df.iloc[current_idx]['Close'] > prev_high:
                        choch_list.append({
                            'index': current_idx,
                            'type': 'bullish',
                            'price': prev_high,
                            'broken_swing_idx': swing_highs[i-1]['index']
                        })

        return sorted(choch_list, key=lambda x: x['index'])


    def _detect_liquidity_levels(self, df: pd.DataFrame) -> Dict:
        """
        Détecte les niveaux de liquidité (Weak/Strong Highs/Lows)

        Weak = touché peu de fois (facile à casser = liquidity grab)
        Strong = touché plusieurs fois (support/résistance fort)

        Returns:
            Dict avec weak_highs, weak_lows, strong_highs, strong_lows
        """
        swing_order = self.liq_config.get('swing_order', 5)
        weak_threshold = self.liq_config.get('weak_threshold', 1)
        strong_threshold = self.liq_config.get('strong_threshold', 3)

        highs = df['High'].values
        lows = df['Low'].values

        # Trouver les extrema
        swing_high_idx = argrelextrema(highs, np.greater, order=swing_order)[0]
        swing_low_idx = argrelextrema(lows, np.less, order=swing_order)[0]

        # Compter combien de fois chaque niveau est testé
        tolerance = 0.005  # 0.5% de tolérance

        def count_tests(level, prices):
            """Compte combien de fois un niveau est testé"""
            count = 0
            for price in prices:
                if abs(price - level) / level <= tolerance:
                    count += 1
            return count

        weak_highs = []
        strong_highs = []
        for idx in swing_high_idx:
            level = highs[idx]
            tests = count_tests(level, highs)

            if tests <= weak_threshold:
                weak_highs.append({'index': idx, 'price': level, 'tests': tests})
            elif tests >= strong_threshold:
                strong_highs.append({'index': idx, 'price': level, 'tests': tests})

        weak_lows = []
        strong_lows = []
        for idx in swing_low_idx:
            level = lows[idx]
            tests = count_tests(level, lows)

            if tests <= weak_threshold:
                weak_lows.append({'index': idx, 'price': level, 'tests': tests})
            elif tests >= strong_threshold:
                strong_lows.append({'index': idx, 'price': level, 'tests': tests})

        return {
            'weak_highs': weak_highs,
            'weak_lows': weak_lows,
            'strong_highs': strong_highs,
            'strong_lows': strong_lows
        }


    def _calculate_premium_discount(self, df: pd.DataFrame) -> Dict:
        """
        Calcule les zones Premium/Discount

        Premium = au-dessus de 50% (zone de vente)
        Discount = en-dessous de 50% (zone d'achat)

        Returns:
            Dict avec range_high, range_low, equilibrium, premium, discount, current_zone
        """
        lookback = self.pd_config.get('lookback', 50)
        premium_level = self.pd_config.get('premium_level', 0.618)
        discount_level = self.pd_config.get('discount_level', 0.382)

        # Prendre les N dernières bougies
        recent_df = df.tail(min(lookback, len(df)))

        range_high = recent_df['High'].max()
        range_low = recent_df['Low'].min()
        range_size = range_high - range_low

        equilibrium = range_low + (range_size * 0.5)
        premium = range_low + (range_size * premium_level)
        discount = range_low + (range_size * discount_level)

        # Prix actuel
        current_price = df.iloc[-1]['Close']

        # Déterminer la zone actuelle
        if current_price >= premium:
            current_zone = 'premium'
        elif current_price <= discount:
            current_zone = 'discount'
        else:
            current_zone = 'equilibrium'

        return {
            'range_high': range_high,
            'range_low': range_low,
            'equilibrium': equilibrium,
            'premium': premium,
            'discount': discount,
            'current_price': current_price,
            'current_zone': current_zone
        }


    def detect_setups(self, df: pd.DataFrame, smc_result: Dict) -> List[Dict]:
        """
        Détecte les setups de trading valides basés sur les concepts SMC

        Args:
            df: DataFrame des prix
            smc_result: Résultat de analyze()

        Returns:
            List d'alertes de trading avec type, raison, entry, stop, target
        """
        alerts = []

        if len(df) < 10:
            return alerts

        current_price = df.iloc[-1]['Close']
        current_idx = len(df) - 1

        # Configuration des alertes
        alert_config = self.config.get('alerts', {})
        proximity_percent = alert_config.get('proximity_percent', 0.02)  # 2%
        recent_lookback = alert_config.get('recent_lookback', 20)  # 20 bougies

        # Récupérer les éléments SMC
        bullish_obs = smc_result['order_blocks']['bullish']
        bearish_obs = smc_result['order_blocks']['bearish']
        bullish_fvgs = smc_result['fvg']['bullish']
        bearish_fvgs = smc_result['fvg']['bearish']
        bos_list = smc_result['bos']
        choch_list = smc_result['choch']
        pd_zones = smc_result['premium_discount']

        # Fonction helper: est-ce que le prix est proche d'une zone?
        def is_near_zone(price, zone_low, zone_high, tolerance=proximity_percent):
            """Vérifie si le prix est dans ou près d'une zone"""
            zone_mid = (zone_low + zone_high) / 2
            zone_range = zone_high - zone_low
            extended_low = zone_low - (zone_mid * tolerance)
            extended_high = zone_high + (zone_mid * tolerance)
            return extended_low <= price <= extended_high

        # Fonction helper: trouver le dernier BOS/CHoCH récent
        def get_recent_structure(structure_list, lookback):
            """Retourne les structures dans les N dernières bougies"""
            cutoff_idx = current_idx - lookback
            return [s for s in structure_list if s['index'] >= cutoff_idx]

        # Tendance actuelle basée sur les BOS récents
        recent_bos = get_recent_structure(bos_list, recent_lookback)
        recent_choch = get_recent_structure(choch_list, recent_lookback)

        bullish_trend = any(b['type'] == 'bullish' for b in recent_bos[-3:]) if recent_bos else False
        bearish_trend = any(b['type'] == 'bearish' for b in recent_bos[-3:]) if recent_bos else False

        # === ALERT LONG ===
        # Conditions: Bullish OB/FVG + Tendance haussière + Zone discount
        if pd_zones.get('current_zone') == 'discount' and bullish_trend:
            # Chercher OB bullish proche
            for ob in bullish_obs[-10:]:  # Derniers 10 OB
                if is_near_zone(current_price, ob['low'], ob['high']):
                    alerts.append({
                        'type': 'LONG',
                        'reason': f"Prix dans Bullish OB [{ob['low']:.2f}-{ob['high']:.2f}] + Tendance UP + Zone Discount",
                        'entry': current_price,
                        'stop': ob['low'] - (ob['high'] - ob['low']) * 0.1,  # 10% sous l'OB
                        'target': pd_zones['premium'],  # Objectif = zone premium
                        'zone': 'order_block',
                        'zone_index': ob['index']
                    })
                    break  # Une seule alerte par type

            # Chercher FVG bullish proche
            if not alerts:  # Si pas d'OB, chercher FVG
                for fvg in bullish_fvgs[-10:]:
                    if is_near_zone(current_price, fvg['low'], fvg['high']):
                        alerts.append({
                            'type': 'LONG',
                            'reason': f"Prix dans Bullish FVG [{fvg['low']:.2f}-{fvg['high']:.2f}] + Tendance UP + Zone Discount",
                            'entry': current_price,
                            'stop': fvg['low'] - (fvg['high'] - fvg['low']) * 0.1,
                            'target': pd_zones['premium'],
                            'zone': 'fvg',
                            'zone_index': fvg['index']
                        })
                        break

        # === ALERT SHORT ===
        # Conditions: Bearish OB/FVG + Tendance baissière + Zone premium
        if pd_zones.get('current_zone') == 'premium' and bearish_trend:
            # Chercher OB bearish proche
            for ob in bearish_obs[-10:]:
                if is_near_zone(current_price, ob['low'], ob['high']):
                    alerts.append({
                        'type': 'SHORT',
                        'reason': f"Prix dans Bearish OB [{ob['low']:.2f}-{ob['high']:.2f}] + Tendance DOWN + Zone Premium",
                        'entry': current_price,
                        'stop': ob['high'] + (ob['high'] - ob['low']) * 0.1,  # 10% au-dessus de l'OB
                        'target': pd_zones['discount'],  # Objectif = zone discount
                        'zone': 'order_block',
                        'zone_index': ob['index']
                    })
                    break

            # Chercher FVG bearish proche
            if not any(a['type'] == 'SHORT' for a in alerts):
                for fvg in bearish_fvgs[-10:]:
                    if is_near_zone(current_price, fvg['low'], fvg['high']):
                        alerts.append({
                            'type': 'SHORT',
                            'reason': f"Prix dans Bearish FVG [{fvg['low']:.2f}-{fvg['high']:.2f}] + Tendance DOWN + Zone Premium",
                            'entry': current_price,
                            'stop': fvg['high'] + (fvg['high'] - fvg['low']) * 0.1,
                            'target': pd_zones['discount'],
                            'zone': 'fvg',
                            'zone_index': fvg['index']
                        })
                        break

        # === ALERT REVERSAL ===
        # CHoCH récent + prix dans OB/FVG opposé
        if recent_choch:
            last_choch = recent_choch[-1]

            # CHoCH bullish (retournement à la hausse)
            if last_choch['type'] == 'bullish':
                for ob in bullish_obs[-5:]:
                    if is_near_zone(current_price, ob['low'], ob['high']):
                        alerts.append({
                            'type': 'REVERSAL_LONG',
                            'reason': f"CHoCH Bullish détecté + Prix dans Bullish OB [{ob['low']:.2f}-{ob['high']:.2f}]",
                            'entry': current_price,
                            'stop': ob['low'] - (ob['high'] - ob['low']) * 0.15,
                            'target': pd_zones['equilibrium'],
                            'zone': 'order_block',
                            'zone_index': ob['index']
                        })
                        break

            # CHoCH bearish (retournement à la baisse)
            elif last_choch['type'] == 'bearish':
                for ob in bearish_obs[-5:]:
                    if is_near_zone(current_price, ob['low'], ob['high']):
                        alerts.append({
                            'type': 'REVERSAL_SHORT',
                            'reason': f"CHoCH Bearish détecté + Prix dans Bearish OB [{ob['low']:.2f}-{ob['high']:.2f}]",
                            'entry': current_price,
                            'stop': ob['high'] + (ob['high'] - ob['low']) * 0.15,
                            'target': pd_zones['equilibrium'],
                            'zone': 'order_block',
                            'zone_index': ob['index']
                        })
                        break

        # Calculer Risk/Reward pour chaque alerte
        for alert in alerts:
            risk = abs(alert['entry'] - alert['stop'])
            reward = abs(alert['target'] - alert['entry'])
            alert['risk_reward'] = round(reward / risk, 2) if risk > 0 else 0

        return alerts
