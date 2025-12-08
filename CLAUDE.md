# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based stock scanning and technical analysis system that analyzes NASDAQ 100 stocks using multiple indicators:
- **Squeeze Momentum** (LazyBear's SQZMOM_LB indicator port)
- **MACD** (Moving Average Convergence Divergence)
- **SMC** (Smart Money Concepts - Order Blocks)
- **Support/Resistance Levels**
- **ADX** (Average Directional Index for trend filtering)

The system generates combined trading signals when multiple indicators align and can operate in both backtest and real-time modes with intelligent market hours detection and automatic cache management.

## Core Architecture

- **new_scan.py**: Main orchestrator & scanner with market hours intelligence
- **trading_manager.py**: IBKR integration & order management
- **squeeze_momentum_analyzer.py**: LazyBear SQZMOM_LB indicator
- **macd_analyzer.py**: MACD calculations & signals
- **smc_analyzer.py**: Smart Money Concepts (Order Blocks)
- **sr_levels_analyzer.py**: Support/Resistance detection
- **adx_analyzer.py**: ADX trend strength filtering
- **settings.json**: System configuration
- **trading_settings.json**: Trading parameters
- **config.txt**: Symbol watchlists

### Key Modules
```
new_scan/
├── new_scan.py              # Main orchestrator & scanner
├── trading_manager.py       # IBKR integration & order management
├── squeeze_momentum_analyzer.py  # LazyBear SQZMOM_LB indicator
├── macd_analyzer.py         # MACD calculations & signals
├── smc_analyzer.py          # Smart Money Concepts (Order Blocks)
├── sr_levels_analyzer.py   # Support/Resistance detection
├── adx_analyzer.py          # ADX trend strength filtering
├── settings.json           # System configuration
├── trading_settings.json   # Trading parameters
└── config.txt              # Symbol watchlists
```

## Running the System

### Development/Testing Commands
```bash
# Run backtest mode (uses Yahoo Finance data) - scans NASDAQ 100
python3 new_scan.py --backtest

# Generate chart for specific symbol
python3 new_scan.py --chart AAPL

# Run real-time scanning (requires IBKR connection) - scans NASDAQ 100
python3 new_scan.py

# Clean cached data
python3 new_scan.py --cleanup
```

### Stock Coverage
The system automatically scans **NASDAQ 100 stocks** by default. No additional flags required.

### Dependencies
Install requirements:
```bash
pip install -r requirements.txt
```

### Market Hours Intelligence
The system automatically detects NYSE market hours and adjusts behavior:
- **Market Open (9:30-16:00 EST)**: 60-second scan intervals
- **Pre-market/After-hours**: 5-minute intervals to conserve resources  
- **Weekend**: 30-minute intervals with minimal activity
- **Market Opening**: Automatic cache cleanup and fresh data download

## Signal Logic

The system implements exact Pine Script logic for combined signals:
- **Buy Signal**: Squeeze Momentum turns LIME AND MACD turns GREEN (transition required)
- **Sell Signal**: Squeeze Momentum turns RED AND MACD turns RED (transition required)
- **ADX Filter**: Optional trend strength filtering (when enabled)
- **Alternating Signals**: System enforces buy→sell→buy pattern to avoid duplicate signals

### Market Hours Detection
```python
def is_market_open():
    """Detect NYSE market hours and return status"""
    now = datetime.now(EST)
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    current_time = now.time()
    
    # Weekend check
    if weekday >= 5:  # Saturday=5, Sunday=6
        days_until_monday = (7 - weekday) % 7
        if days_until_monday == 0:  # Si c'est dimanche
            days_until_monday = 1
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=days_until_monday)
        return False, "Weekend", next_open
    
    # Trading hours (9:30-16:00 EST)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    if current_time < market_open:
        # Pre-market
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return False, "Pre-market", next_open
    elif current_time > market_close:
        # After-hours
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
        return False, "After-hours", next_open
    else:
        # Market open
        next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return True, "Open", next_close
```

## Configuration

The system is configured via `settings.json`:
- **data_folder**: Where historical data is cached
- **backtest**: Parameters for backtesting mode (candle count, interval)
- **realtime**: IBKR connection settings and real-time parameters
- **indicators**: Enable/disable and configure each indicator

Individual stock watchlist is defined in `config.txt` (symbol + provider format).

## Data Sources

- **Backtest Mode**: Yahoo Finance (`yfinance`)
- **Real-time Mode**: Interactive Brokers (`ib_insync`)

Data is cached in the `data/` folder with filename format: `{date}_{symbol}_{candle_count}_{interval}.csv`

### Caching System
- **Location**: `data/` folder
- **Format**: `{date}_{symbol}_{candle_count}_{interval}.csv`
- **Example**: `data/2025-12-05_AAPL_200_1d.csv`
- **Automatic Cleanup**: Cache refresh on market opening using existing `cleanup_data()` method

### EST Timezone Handling
```python
from zoneinfo import ZoneInfo

EST = ZoneInfo('America/New_York')
now = datetime.now(EST)
```

### Automatic Cache Management
- **Market Opening**: System automatically calls `cleanup_data()` when market opens
- **Resource Optimization**: Intelligent scan intervals based on market status
- **Fresh Data**: Ensures clean cache for accurate trading decisions

## Output

- **Console**: Real-time alerts with emoji indicators (🟢 for buy, 🔴 for sell)
- **Charts**: HTML charts generated in `chart/` folder using Plotly when signals trigger
- **Charts include**: Candlesticks, Order Blocks, Support/Resistance levels, and signal markers

## Calibration Files

The `squeeze_momentum_analyzer.py` includes calibration factors optimized to match TradingView's exact values:
- Multiple analysis and optimization scripts in repo (`analyze_*.py`, `optimize_*.py`, `verify_*.py`)
- Calibration ensures momentum calculations match TradingView reference implementation

## Trading Integration

### TradingManager Features
- **Bracket Orders**: Parent market order with child stop-loss order
- **Risk Management**: Automatic stop-loss calculation based on price action
- **Paper Trading**: Safe testing environment before live trading
- **Exchange-Managed Stops**: 24/7 protection without script dependency

### Stop-Loss Logic
- **BUY positions**: SL = min(signal_low, previous_low)  
- **SELL positions**: SL = max(previous_high, pre_previous_high) - Uses 2 candles before signal

### Key Trading Functions
- `smart_trade_with_bracket()`: Execute bracket order with automatic stop-loss
- `calculate_sl_level()`: Calculate stop-loss based on price action
- `get_account_equity()`: Monitor account balance and position sizing
- `can_trade()`: Check positions and pending orders to prevent duplicates
- `get_pending_orders()`: Monitor active/pending orders at IBKR
- `display_positions_summary()`: Show current positions with P&L

## Testing Framework

### Available Tests
- `test_market_hours.py`: Validate market hours detection logic
- `test_bracket_order.py`: Test bracket order functionality with IBKR
- Various analyzer test files for indicator validation

### Running Tests
```bash
# Test market hours detection
python3 test_market_hours.py

# Test bracket order functionality  
python3 test_bracket_order.py

# Test specific indicators
python3 test_*.py
```

## Current Implementation Status

✅ **Complete:**
- Market hours intelligence with automatic cache management
- Multiple technical indicator implementations (Squeeze Momentum, MACD, SMC, S/R, ADX)
- Combined signal logic with strict alternation
- IBKR integration with bracket order support
- Interactive chart generation with signal markers
- Comprehensive configuration system
- Automatic cache cleanup on market opening

🚧 **Active Features:**
- Real-time scanning with intelligent intervals based on market hours
- Paper trading integration for safe testing
- Cache optimization and resource management
- EST timezone handling with ZoneInfo

📋 **Recent Improvements:**
- Added `is_market_open()` function with fake_market mode support
- Implemented automatic cache management with intelligent cleanup
- Enhanced duplicate order prevention with pending order checks
- Fixed stop-loss calculation to use correct candle references
- Simplified system to focus on NASDAQ 100 (removed crypto options)
- Added position summary display with real-time P&L
- Integrated comprehensive order status monitoring