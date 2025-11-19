#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import List, Dict


class SqueezeAnalyzer:
    """
    Squeeze Momentum Indicator Analyzer - LazyBear
    Detects squeeze conditions and momentum changes
    Port of TradingView SQZMOM_LB indicator
    """

    def __init__(self, config: dict):
        """
        Initialize Squeeze Momentum Analyzer

        Args:
            config: Configuration from settings.json
        """
        self.config = config
        self.bb_length = config.get('bb_length', 20)
        self.bb_mult = config.get('bb_mult', 2.0)
        self.kc_length = config.get('kc_length', 20)
        self.kc_mult = config.get('kc_mult', 1.5)
        self.use_true_range = config.get('use_true_range', True)


    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Complete Squeeze Momentum analysis on DataFrame

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume

        Returns:
            Dict containing squeeze signals and momentum values
        """
        result = {
            'squeeze_on': [],      # List of squeeze on signals
            'squeeze_off': [],     # List of squeeze off signals
            'momentum_change': [], # List of momentum direction changes
            'values': []          # All calculated values for charting
        }

        min_length = max(self.bb_length, self.kc_length)
        if len(df) < min_length + 1:
            return result

        # Calculate Bollinger Bands
        closes = df['Close'].values
        bb_basis = self._sma(closes, self.bb_length)
        bb_dev = self.bb_mult * self._stdev(closes, self.bb_length)
        upper_bb = bb_basis + bb_dev
        lower_bb = bb_basis - bb_dev

        # Calculate Keltner Channels
        kc_ma = self._sma(closes, self.kc_length)

        if self.use_true_range:
            range_vals = self._true_range(df)
        else:
            range_vals = df['High'].values - df['Low'].values

        range_ma = self._sma(range_vals, self.kc_length)
        upper_kc = kc_ma + range_ma * self.kc_mult
        lower_kc = kc_ma - range_ma * self.kc_mult

        # Detect squeeze conditions
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
        no_sqz = ~sqz_on & ~sqz_off

        # Calculate momentum value using linear regression
        momentum = self._calculate_momentum(df, self.kc_length)

        # Detect squeeze on/off signals and momentum changes
        start_idx = min_length

        for i in range(start_idx + 1, len(df)):
            # Squeeze ON signal (transition from not squeezed to squeezed)
            if sqz_on[i] and not sqz_on[i-1]:
                result['squeeze_on'].append({
                    'index': i,
                    'price': df.iloc[i]['Close'],
                    'type': 'squeeze_on'
                })

            # Squeeze OFF signal (transition from squeezed to not squeezed)
            if sqz_off[i] and not sqz_off[i-1]:
                result['squeeze_off'].append({
                    'index': i,
                    'price': df.iloc[i]['Close'],
                    'type': 'squeeze_off'
                })

            # Momentum direction change
            if i > start_idx + 1:
                # Bullish momentum change (negative to positive)
                if momentum[i-1] < 0 and momentum[i] > 0:
                    result['momentum_change'].append({
                        'index': i,
                        'price': df.iloc[i]['Close'],
                        'type': 'bullish',
                        'momentum': momentum[i]
                    })
                # Bearish momentum change (positive to negative)
                elif momentum[i-1] > 0 and momentum[i] < 0:
                    result['momentum_change'].append({
                        'index': i,
                        'price': df.iloc[i]['Close'],
                        'type': 'bearish',
                        'momentum': momentum[i]
                    })

                # Momentum increasing (bullish strengthening)
                if momentum[i] > 0 and momentum[i] > momentum[i-1]:
                    color = 'lime'
                # Momentum positive but decreasing (bullish weakening)
                elif momentum[i] > 0 and momentum[i] <= momentum[i-1]:
                    color = 'green'
                # Momentum negative but increasing (bearish weakening)
                elif momentum[i] < 0 and momentum[i] > momentum[i-1]:
                    color = 'maroon'
                # Momentum decreasing (bearish strengthening)
                else:
                    color = 'red'
            else:
                color = 'gray'

            # Store values for charting
            result['values'].append({
                'index': i,
                'momentum': momentum[i],
                'sqz_on': sqz_on[i],
                'sqz_off': sqz_off[i],
                'no_sqz': no_sqz[i],
                'color': color
            })

        return result


    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average
        """
        sma = np.zeros_like(data, dtype=float)

        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])

        return sma


    def _stdev(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Standard Deviation
        """
        stdev = np.zeros_like(data, dtype=float)

        for i in range(period - 1, len(data)):
            stdev[i] = np.std(data[i - period + 1:i + 1], ddof=0)

        return stdev


    def _true_range(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate True Range
        """
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        tr = np.zeros(len(df))

        # First bar: high - low
        tr[0] = high[0] - low[0]

        # Subsequent bars: max of three values
        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        return tr


    def _calculate_momentum(self, df: pd.DataFrame, length: int) -> np.ndarray:
        """
        Calculate momentum using linear regression
        Based on: linreg(source - avg(avg(highest(high, lengthKC), lowest(low, lengthKC)), sma(close, lengthKC)), lengthKC, 0)
        """
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values

        momentum = np.zeros(len(df))

        for i in range(length - 1, len(df)):
            # Get data window
            window_highs = highs[i - length + 1:i + 1]
            window_lows = lows[i - length + 1:i + 1]
            window_closes = closes[i - length + 1:i + 1]

            # Calculate highest and lowest
            highest_high = np.max(window_highs)
            lowest_low = np.min(window_lows)

            # Calculate average of highest and lowest
            avg_hl = (highest_high + lowest_low) / 2

            # Calculate SMA of close
            sma_close = np.mean(window_closes)

            # Average of both
            avg_val = (avg_hl + sma_close) / 2

            # Calculate source - avg
            source_vals = window_closes - avg_val

            # Linear regression (slope at current point)
            # y = mx + b, we want the value at the most recent point
            x = np.arange(length)

            # Linear regression using least squares
            x_mean = np.mean(x)
            y_mean = np.mean(source_vals)

            numerator = np.sum((x - x_mean) * (source_vals - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                # Value at offset 0 (most recent) is at x = length - 1
                momentum[i] = slope * (length - 1) + intercept
            else:
                momentum[i] = 0

        return momentum
