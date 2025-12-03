#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class SqueezeMomentumIndicator:
    """
    Squeeze Momentum Indicator - LazyBear (Independent)
    Port direct du code Pine Script SQZMOM_LB
    """

    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5, use_true_range=True):
        """
        Initialize Squeeze Momentum Indicator
        
        Args:
            bb_length: Bollinger Bands period (default 20)
            bb_mult: Bollinger Bands multiplier (default 2.0)  
            kc_length: Keltner Channel period (default 20)
            kc_mult: Keltner Channel multiplier (default 1.5)
            use_true_range: Use True Range for KC calculation (default True)
        """
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.use_true_range = use_true_range

    def calculate_bollinger_bands(self, closes: np.array) -> Tuple[np.array, np.array, np.array]:
        """Calculate Bollinger Bands"""
        basis = self._sma(closes, self.bb_length)
        std_dev = self._stdev(closes, self.bb_length)
        dev = self.bb_mult * std_dev
        upper_bb = basis + dev
        lower_bb = basis - dev
        return upper_bb, lower_bb, basis

    def calculate_keltner_channel(self, highs: np.array, lows: np.array, closes: np.array) -> Tuple[np.array, np.array]:
        """Calculate Keltner Channel"""
        ma = self._sma(closes, self.kc_length)
        
        if self.use_true_range:
            range_values = self._true_range(highs, lows, closes)
        else:
            range_values = highs - lows
            
        range_ma = self._sma(range_values, self.kc_length)
        upper_kc = ma + range_ma * self.kc_mult
        lower_kc = ma - range_ma * self.kc_mult
        return upper_kc, lower_kc

    def calculate_momentum_value(self, highs: np.array, lows: np.array, closes: np.array) -> np.array:
        """Calculate momentum value using linear regression"""
        # avg(avg(highest(high, lengthKC), lowest(low, lengthKC)), sma(close, lengthKC))
        highest_high = self._highest(highs, self.kc_length)
        lowest_low = self._lowest(lows, self.kc_length)
        sma_close = self._sma(closes, self.kc_length)
        
        avg1 = (highest_high + lowest_low) / 2
        avg2 = (avg1 + sma_close) / 2
        
        source_diff = closes - avg2
        momentum = self._linreg(source_diff, self.kc_length)
        
        return momentum

    def get_squeeze_state(self, upper_bb: np.array, lower_bb: np.array, upper_kc: np.array, lower_kc: np.array) -> Dict:
        """Determine squeeze state"""
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
        no_sqz = ~sqz_on & ~sqz_off
        
        return {
            'squeeze_on': sqz_on,
            'squeeze_off': sqz_off, 
            'no_squeeze': no_sqz
        }

    def get_momentum_color(self, momentum: np.array) -> List[str]:
        """Get momentum bar colors according to LazyBear logic"""
        colors = []
        prev_val = 0
        
        for i, val in enumerate(momentum):
            if np.isnan(val):
                colors.append('gray')
                continue
                
            if val > 0:
                if val > prev_val:
                    colors.append('lime')  # Increasing positive
                else:
                    colors.append('green')  # Decreasing positive
            else:
                if val < prev_val:
                    colors.append('red')    # Decreasing negative
                else:
                    colors.append('maroon')  # Increasing negative
                    
            prev_val = val
            
        return colors

    def get_squeeze_color(self, squeeze_state: Dict) -> List[str]:
        """Get squeeze indicator colors"""
        colors = []
        for i in range(len(squeeze_state['squeeze_on'])):
            if squeeze_state['no_squeeze'][i]:
                colors.append('blue')
            elif squeeze_state['squeeze_on'][i]:
                colors.append('black')
            else:
                colors.append('gray')
        return colors

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Complete Squeeze Momentum analysis
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict with momentum values, colors, and squeeze states
        """
        if len(df) < max(self.bb_length, self.kc_length):
            return {
                'momentum': [],
                'momentum_colors': [],
                'squeeze_colors': [],
                'squeeze_on': [],
                'squeeze_off': [],
                'signals': []
            }
        
        highs = df['High'].values
        lows = df['Low'].values 
        closes = df['Close'].values
        
        # Calculate components
        upper_bb, lower_bb, bb_basis = self.calculate_bollinger_bands(closes)
        upper_kc, lower_kc = self.calculate_keltner_channel(highs, lows, closes)
        momentum = self.calculate_momentum_value(highs, lows, closes)
        squeeze_state = self.get_squeeze_state(upper_bb, lower_bb, upper_kc, lower_kc)
        
        # Get colors
        momentum_colors = self.get_momentum_color(momentum)
        squeeze_colors = self.get_squeeze_color(squeeze_state)
        
        # Detect signals (momentum crosses zero)
        signals = self._detect_signals(momentum)
        
        return {
            'momentum': momentum.tolist(),
            'momentum_colors': momentum_colors,
            'squeeze_colors': squeeze_colors,
            'squeeze_on': squeeze_state['squeeze_on'].tolist(),
            'squeeze_off': squeeze_state['squeeze_off'].tolist(),
            'signals': signals,
            'upper_bb': upper_bb.tolist(),
            'lower_bb': lower_bb.tolist(),
            'upper_kc': upper_kc.tolist(), 
            'lower_kc': lower_kc.tolist()
        }

    def _detect_signals(self, momentum: np.array) -> List[Dict]:
        """Detect momentum zero crossings"""
        signals = []
        for i in range(1, len(momentum)):
            if np.isnan(momentum[i-1]) or np.isnan(momentum[i]):
                continue
                
            # Bullish: momentum crosses from negative to positive
            if momentum[i-1] <= 0 and momentum[i] > 0:
                signals.append({
                    'index': i,
                    'type': 'bullish',
                    'momentum': momentum[i]
                })
            # Bearish: momentum crosses from positive to negative  
            elif momentum[i-1] >= 0 and momentum[i] < 0:
                signals.append({
                    'index': i,
                    'type': 'bearish', 
                    'momentum': momentum[i]
                })
                
        return signals

    # Helper functions
    def _sma(self, data: np.array, period: int) -> np.array:
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period, min_periods=period).mean().values

    def _stdev(self, data: np.array, period: int) -> np.array:
        """Standard Deviation"""
        return pd.Series(data).rolling(window=period, min_periods=period).std().values

    def _true_range(self, highs: np.array, lows: np.array, closes: np.array) -> np.array:
        """True Range calculation"""
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]  # Handle first value
        
        tr1 = highs - lows
        tr2 = np.abs(highs - prev_close)
        tr3 = np.abs(lows - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))

    def _highest(self, data: np.array, period: int) -> np.array:
        """Highest value over period"""
        return pd.Series(data).rolling(window=period, min_periods=period).max().values

    def _lowest(self, data: np.array, period: int) -> np.array:
        """Lowest value over period"""
        return pd.Series(data).rolling(window=period, min_periods=period).min().values

    def _linreg(self, data: np.array, period: int) -> np.array:
        """Linear regression value"""
        result = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            y_vals = data[i - period + 1:i + 1]
            x_vals = np.arange(period)
            
            if len(y_vals) == period and not np.any(np.isnan(y_vals)):
                # Linear regression: y = mx + b
                A = np.vstack([x_vals, np.ones(len(x_vals))]).T
                m, b = np.linalg.lstsq(A, y_vals, rcond=None)[0]
                # Return the value at the end of the regression line
                result[i] = m * (period - 1) + b
                
        return result


# Example usage
if __name__ == '__main__':
    # Test with sample data
    import yfinance as yf
    
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="6mo", interval="1d")
    
    # Reset column names to match expected format
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    
    indicator = SqueezeMomentumIndicator()
    result = indicator.analyze(df)
    
    print(f"Analyzed {len(df)} candles")
    print(f"Found {len(result['signals'])} momentum signals")
    
    # Show recent momentum values and colors
    for i in range(-5, 0):
        momentum = result['momentum'][i]
        color = result['momentum_colors'][i]
        print(f"Bar {i}: Momentum={momentum:.4f}, Color={color}")