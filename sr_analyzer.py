# Fichier : sr_analyzer.py

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import List, Tuple, Dict, Any

class SRAnalyzer:
    """
    Analyse les niveaux de support et de résistance basés sur les extrema locaux
    et le clustering des niveaux.
    """
    
    def __init__(self, sr_config: Dict[str, Any]):
        """Initialise l'analyseur avec la configuration S/R."""
        self.config = sr_config
        self.order = sr_config.get('order', 5)
        self.cluster_threshold = sr_config.get('cluster_threshold', 0.005) # Exemple: 0.5%
        self.min_touches = sr_config.get('min_touches', 2)
        self.breakout_tolerance = sr_config.get('breakout_tolerance', 0.005)
        print("order: ", self.order)
        print("touches", self.min_touches)
        print("threshold: ", self.cluster_threshold)
        print("breakout_tolerance: ", self.breakout_tolerance)

    def _adjust_order(self, n: int) -> int:
        """Ajuste le paramètre 'order' pour argrelextrema en cas de données insuffisantes."""
        order = self.order
        if order < 1:
            order = 1
        if n <= (2 * order):
            original_order = self.order
            # Utilise au maximum (n-1)/2 comme ordre
            order = max(1, (n - 1) // 2)
            print(f"AVERTISSEMENT: Order ajusté de {original_order} à {order} pour n={n} bougies")
        return order
        
    def _cluster_levels(self, levels: np.ndarray) -> List[float]:
        """Regroupe les niveaux d'extrema proches en clusters."""
        if len(levels) == 0:
            return []
            
        levels_sorted = sorted(levels)
        clusters: List[float] = []
        current_cluster = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            # Utilisation du seuil relatif
            denom = current_cluster[-1] if current_cluster[-1] != 0 else 1.0
            if abs(level - current_cluster[-1]) / denom < self.cluster_threshold:
                current_cluster.append(level)
            else:
                # Ne garde le cluster que s'il a assez d'extrema
                if len(current_cluster) >= self.min_touches:
                    clusters.append(float(np.mean(current_cluster)))
                current_cluster = [level]
                
        # Dernier cluster
        if len(current_cluster) >= self.min_touches:
            clusters.append(float(np.mean(current_cluster)))
            
        return clusters

    def find_levels(self, df: pd.DataFrame, filter_high: float = None, filter_low: float = None) -> Tuple[List[float], List[float]]:
        """
        Calcule et retourne les niveaux de support et résistance clusterisés.
        Les niveaux cassés (avec breakout_tolerance) sont invalidés.

        Args:
            df (pd.DataFrame): DataFrame contenant les colonnes 'High' et 'Low'.
            filter_high (float, optional): High de la bougie à utiliser pour filtrage. Par défaut: dernière bougie du df.
            filter_low (float, optional): Low de la bougie à utiliser pour filtrage. Par défaut: dernière bougie du df.

        Returns:
            Tuple[List[float], List[float]]: (niveaux de support, niveaux de résistance)
        """
        highs = df['High'].values
        lows = df['Low'].values
        n = len(df)

        # Ajuster 'order' avant de calculer les extrema
        adjusted_order = self._adjust_order(n)

        # 1. Extraction des Extrema
        resistance_idx = argrelextrema(highs, np.greater, order=adjusted_order)[0]
        support_idx = argrelextrema(lows, np.less, order=adjusted_order)[0]

        resistance_levels = highs[resistance_idx] if resistance_idx.size else np.array([])
        support_levels = lows[support_idx] if support_idx.size else np.array([])

        # 2. Clustering des Niveaux
        support_clusters = self._cluster_levels(support_levels)
        resistance_clusters = self._cluster_levels(resistance_levels)

        # 3. Filtrer les niveaux cassés basés sur la bougie spécifiée (ou dernière bougie)
        last_high = filter_high if filter_high is not None else df['High'].iloc[-1]
        last_low = filter_low if filter_low is not None else df['Low'].iloc[-1]

        # Invalider résistances cassées : prix a dépassé résistance + tolérance
        valid_resistances = [
            r for r in resistance_clusters
            if last_high <= r * (1 + self.breakout_tolerance)
        ]

        # Invalider supports cassés : prix est descendu sous support - tolérance
        valid_supports = [
            s for s in support_clusters
            if last_low >= s * (1 - self.breakout_tolerance)
        ]

        return valid_supports, valid_resistances
# --- Fin du fichier sr_analyzer.py ---