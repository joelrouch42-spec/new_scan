#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict


class ADXAnalyzer:
    """
    ADX (Average Directional Index) Analyzer
    Measures trend strength - values above threshold indicate strong trend
    """

    def __init__(self, config: dict):
        self.length = config.get('length', 14)
        self.threshold = config.get('threshold', 25)


    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Complete ADX analysis on DataFrame
        Returns ADX values and trend detection
        """
        result = {
            'values': []
        }

        if len(df) < self.length * 2:
            return result

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values

        tr = self._calculate_true_range(highs, lows, closes)
        plus_dm = self._calculate_plus_dm(highs, lows)
        minus_dm = self._calculate_minus_dm(highs, lows)

        tr_smooth = self._wilder_smooth(tr, self.length)
        plus_dm_smooth = self._wilder_smooth(plus_dm, self.length)
        minus_dm_smooth = self._wilder_smooth(minus_dm, self.length)

        plus_di = np.zeros(len(df))
        minus_di = np.zeros(len(df))

        for i in range(len(df)):
            if tr_smooth[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]

        dx = np.zeros(len(df))
        for i in range(len(df)):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        adx = self._wilder_smooth(dx, self.length)

        start_idx = self.length * 2

        for i in range(start_idx, len(df)):
            result['values'].append({
                'index': i,
                'adx': adx[i],
                'in_trend': adx[i] > self.threshold
            })

        return result


    def _calculate_true_range(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        tr = np.zeros(len(highs))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(highs)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )

        return tr


    def _calculate_plus_dm(self, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
        """
        Calculate +DM (Plus Directional Movement)
        TradingView logic:
        up = change(high) = high - high[1]
        down = -change(low) = -(low - low[1]) = low[1] - low
        +DM = (up > down and up > 0) ? up : 0
        """
        plus_dm = np.zeros(len(highs))

        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0

        return plus_dm


    def _calculate_minus_dm(self, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
        """
        Calculate -DM (Minus Directional Movement)
        TradingView logic:
        up = change(high) = high - high[1]
        down = -change(low) = low[1] - low
        -DM = (down > up and down > 0) ? down : 0
        """
        minus_dm = np.zeros(len(lows))

        for i in range(1, len(lows)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0

        return minus_dm


    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        smoothed = np.zeros_like(data, dtype=float)
        smoothed[period-1] = np.mean(data[:period])

        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period - 1) + data[i]) / period

        return smoothed
