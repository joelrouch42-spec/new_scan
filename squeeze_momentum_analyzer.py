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
            'zero_cross_positive': [],  # Momentum crosses from negative to positive
            'zero_cross_negative': [],  # Momentum crosses from positive to negative
            'values': []                # All momentum values with their colors
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

        # Analyze momentum values
        start_idx = min_length

        for i in range(start_idx, len(df)):
            # Determine bar color based on Pine Script logic:
            # bcolor = iff( val > 0,
            #             iff( val > nz(val[1]), lime, green),
            #             iff( val < nz(val[1]), red, maroon))

            if i > start_idx:
                val = momentum[i]
                val_prev = momentum[i-1]

                if val > 0:
                    # Positive momentum
                    if val > val_prev:
                        color = 'lime'    # Bright green - becoming more positive (longer bars)
                    else:
                        color = 'green'   # Dark green - becoming less positive (shorter bars)
                else:
                    # Negative momentum
                    # Match Pine Script exactly: iff( val < nz(val[1]), red, maroon)
                    if val < val_prev:
                        color = 'red'     # Pine 'red' - becoming more negative (longer bars)
                    else:
                        color = 'maroon'  # Pine 'maroon' - becoming less negative (shorter bars)

                # Detect zero crossings
                # Cross from negative to positive (0 à +)
                if val_prev <= 0 and val > 0:
                    result['zero_cross_positive'].append({
                        'index': i,
                        'price': df.iloc[i]['Close'],
                        'momentum': val
                    })

                # Cross from positive to negative (0 à -)
                elif val_prev >= 0 and val < 0:
                    result['zero_cross_negative'].append({
                        'index': i,
                        'price': df.iloc[i]['Close'],
                        'momentum': val
                    })
            else:
                color = 'gray'
                val = momentum[i]

            # Determine squeeze color for the dot
            if sqz_on[i]:
                squeeze_color = 'black'
            elif sqz_off[i]:
                squeeze_color = 'gray'
            else:
                squeeze_color = 'blue'  # no squeeze

            # Store all values
            result['values'].append({
                'index': i,
                'momentum': val,
                'color': color,
                'squeeze_color': squeeze_color,
                'sqz_on': sqz_on[i],
                'sqz_off': sqz_off[i],
                'no_sqz': no_sqz[i]
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
        Calculate momentum using Pine Script's exact linear regression implementation
        Pine Script: linreg(source - avg(avg(highest(high, length), lowest(low, length)), sma(close, length)), length, 0)
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

            # Pine Script calculation exactly:
            # highest(high, length)
            highest_high = np.max(window_highs)
            # lowest(low, length)
            lowest_low = np.min(window_lows)
            # avg(highest, lowest)
            avg_hl = (highest_high + lowest_low) / 2
            # sma(close, length)
            sma_close = np.mean(window_closes)
            # avg(avg_hl, sma_close)
            avg_val = (avg_hl + sma_close) / 2

            # source - avg_val (this is what we regress on)
            source_vals = window_closes - avg_val

            # Pine Script ta.linreg() exact implementation
            # This calculates linear regression and returns the value at offset 0 (current bar)
            momentum[i] = self._pine_linreg(source_vals, length, 0)

        return momentum

    def _pine_linreg(self, src: np.ndarray, length: int, offset: int) -> float:
        """
        Exact Pine Script ta.linreg() implementation
        Calculates linear regression and returns value at specified offset
        """
        if len(src) != length:
            return 0.0
        
        if length <= 1:
            return 0.0
            
        # Pine Script ta.linreg() exact implementation
        x = np.arange(length, dtype=float)  # 0, 1, 2, ..., length-1
        y = np.array(src, dtype=float)
        
        # Standard linear regression calculation
        n = float(length)
        sum_x = np.sum(x)
        sum_y = np.sum(y) 
        sum_xx = np.sum(x * x)
        sum_xy = np.sum(x * y)
        
        # Calculate slope and intercept
        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate value at specified offset
        # offset=0 means current bar, which is at x = length-1
        target_x = length - 1 - offset
        regression_value = slope * target_x + intercept
        
        # Return raw regression value (price adjustment applied above if needed)
        return regression_value
