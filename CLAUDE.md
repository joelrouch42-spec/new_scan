# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based stock scanning and technical analysis system that analyzes NASDAQ 100 stocks and cryptocurrencies using the Squeeze Momentum indicator:
- **Squeeze Momentum** (LazyBear's SQZMOM_LB indicator port)
- **Support/Resistance Levels**
- **ADX** (Average Directional Index for trend filtering)

The system generates trading signals when momentum crosses zero and can operate in both backtest and real-time modes with automatic trading capabilities.

## Core Architecture

- **StockScanner** (`new_scan.py`): Main orchestrator that coordinates all analyzers
- **Squeeze Momentum Indicator** (`squeeze_momentum_indicator.py`): Independent LazyBear implementation
- **Trading Manager** (`trading_manager.py`): Automatic trading via IBKR API
- **Support/Resistance Analyzer** (`sr_levels_analyzer.py`): Support/Resistance level identification
- **ADX Analyzer** (`adx_analyzer.py`): ADX trend strength filtering

## Running the System

### Development/Testing Commands
```bash
# Run backtest mode 
python3 new_scan.py --nasdaq --backtest
python3 new_scan.py --crypto --backtest

# Generate chart for specific symbol (implies backtest)
python3 new_scan.py --chart AAPL --nasdaq

# Generate charts for all symbols (implies backtest)
python3 new_scan.py --nasdaq --allcharts

# Run real-time scanning (requires IBKR connection)
python3 new_scan.py --nasdaq
python3 new_scan.py --crypto

# Override test candle for realtime
python3 new_scan.py --nasdaq --test_candle 2

# Clean data cache
python3 new_scan.py --cleanup
```

### Dependencies
Install requirements:
```bash
pip install -r requirements.txt
```

## Configuration

### Main Settings (`settings.json`)
- **data_folder**: Where historical data is cached
- **backtest**: Parameters for backtesting mode (candle count, interval, test range)
- **realtime**: IBKR connection settings and real-time parameters
- **indicators**: Enable/disable and configure each indicator

### Trading Settings (`trading_settings.json`)
- **enabled**: Enable/disable automatic trading
- **trade_percent**: Percentage of equity to use per trade
- **order_settings**: Market/limit orders, time in force, outside RTH

## Signal Logic

The system implements exact Pine Script logic for momentum signals:
- **Buy Signal**: Momentum crosses from negative to positive (zero crossing)
- **Sell Signal**: Momentum crosses from positive to negative (zero crossing)
- **Visual Indicators**: LIME/RED dots on candles, green/red arrows for zero crossings
- **ADX Filter**: Optional trend strength filtering (when enabled)

## Trading System

### Automatic Trading
- **smart_trade()**: Single API for all trading operations
- **Dynamic Position Checking**: Reads real IBKR positions to prevent double orders
- **Risk Management**: Configurable trade_percent and position sizing
- **Paper Trading**: Supports IBKR paper trading accounts

### Position Management
- Long positions: smart_trade(symbol, 'BUY') 
- Close positions: smart_trade(symbol, 'SELL') - automatically sells full position
- Position verification via IBKR API prevents duplicate trades

## Data Sources

- **Backtest Mode**: Yahoo Finance (`yfinance`) for both stocks and crypto
- **Real-time Mode**: Interactive Brokers (`ib_insync`) for stocks, Yahoo Finance for crypto
- **Crypto Support**: XRP-USD, XDC-USD, BTC-USD, ETH-USD, and other major cryptocurrencies

Data is cached in the `data/` folder with filename format: `{date}_{symbol}_{candle_count}_{interval}.csv`

## Output

- **Console**: Real-time alerts with BUY/SELL notifications
- **Charts**: HTML charts generated in `chart/` folder using Plotly when signals trigger
- **Charts include**: Candlesticks, momentum histogram, support/resistance levels, and signal markers
- **Trading**: Automatic order placement via IBKR when enabled

## Key Features

- **Real-time Market Data**: Live scanning every 60 seconds
- **Multiple Asset Classes**: NASDAQ 100 stocks and major cryptocurrencies
- **Visual Analysis**: Interactive charts with crosshair functionality
- **Configurable Test Ranges**: Flexible backtest periods and real-time test candles
- **Professional Trading Integration**: Full IBKR API integration with position management