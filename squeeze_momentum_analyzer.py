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
            'values': [],               # All momentum values with their colors
            'momentum': [],             # Momentum values array (for compatibility)
            'momentum_colors': [],      # Momentum colors array (for compatibility) 
            'signals': []               # Combined signals (for compatibility)
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
        start_idx = max(self.bb_length, self.kc_length) * 2 - 2  # Match original behavior

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

                # Zero crossings kept for backward compatibility but not used for signals
                if val_prev <= 0 and val > 0:
                    result['zero_cross_positive'].append({
                        'index': i,
                        'price': df.iloc[i]['Close'],
                        'momentum': val
                    })
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

        # Populate compatibility arrays for SqueezeMomentumIndicator interface  
        # Create full-length arrays with NaN/gray for early values
        result['momentum'] = [np.nan] * len(df)
        result['momentum_colors'] = ['gray'] * len(df)
        
        # Fill in the calculated values
        for val in result['values']:
            idx = val['index']
            result['momentum'][idx] = val['momentum']
            result['momentum_colors'][idx] = val['color']
        
        # Detect momentum color change signals (correct LazyBear logic)
        colors = [val['color'] for val in result['values']]
        
        for i in range(1, len(colors)):
            prev_color = colors[i-1]
            curr_color = colors[i]
            
            # BUY signal: momentum becomes maroon (increasing negative)
            if curr_color == 'maroon' and prev_color != 'maroon':
                result['signals'].append({
                    'index': result['values'][i]['index'],
                    'type': 'bullish',
                    'momentum': result['values'][i]['momentum'],
                    'color': curr_color
                })
            # SELL signal: momentum becomes green (decreasing positive)  
            elif curr_color == 'green' and prev_color != 'green':
                result['signals'].append({
                    'index': result['values'][i]['index'],
                    'type': 'bearish',
                    'momentum': result['values'][i]['momentum'],
                    'color': curr_color
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
        Calculate momentum using EXACT original SqueezeMomentumIndicator logic
        """
        highs = df['High'].values
        lows = df['Low'].values  
        closes = df['Close'].values
        
        # Use original's helper functions logic
        highest_high = self._highest(highs, length)
        lowest_low = self._lowest(lows, length)
        sma_close = self._sma(closes, length)
        
        avg1 = (highest_high + lowest_low) / 2
        avg2 = (avg1 + sma_close) / 2
        
        source_diff = closes - avg2
        momentum = self._linreg(source_diff, length)
        
        return momentum
    
    def _highest(self, data: np.array, period: int) -> np.array:
        """Highest value over period - copied from original"""
        return pd.Series(data).rolling(window=period, min_periods=period).max().values

    def _lowest(self, data: np.array, period: int) -> np.array:
        """Lowest value over period - copied from original"""
        return pd.Series(data).rolling(window=period, min_periods=period).min().values
    
    def _linreg(self, data: np.array, period: int) -> np.array:
        """Linear regression value - copied from original"""
        result = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            y_vals = data[i - period + 1:i + 1]
            x_vals = np.arange(period)
            
            if len(y_vals) == period and not np.any(np.isnan(y_vals)):
                A = np.vstack([x_vals, np.ones(len(x_vals))]).T
                m, b = np.linalg.lstsq(A, y_vals, rcond=None)[0]
                result[i] = m * (period - 1) + b
                
        return result

    def _pine_linreg(self, src: np.ndarray, length: int, offset: int) -> float:
        """
        Exact Pine Script ta.linreg() implementation
        Calculates linear regression and returns value at specified offset
        """
        if len(src) != length:
            return 0.0
        
        if length <= 1:
            return 0.0
            
        # Pine Script ta.linreg() exact implementation using np.linalg.lstsq
        x = np.arange(length, dtype=float)  # 0, 1, 2, ..., length-1
        y = np.array(src, dtype=float)
        
        # Use np.linalg.lstsq for better performance and numerical stability
        try:
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0
        
        # Calculate value at specified offset
        # offset=0 means current bar, which is at x = length-1
        target_x = length - 1 - offset
        regression_value = slope * target_x + intercept
        
        # Return raw regression value (price adjustment applied above if needed)
        return regression_value
