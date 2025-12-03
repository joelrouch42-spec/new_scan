# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based stock scanning and technical analysis system that analyzes NASDAQ 100 stocks using multiple indicators:
- **Squeeze Momentum** (LazyBear's SQZMOM_LB indicator port)
- **MACD** (Moving Average Convergence Divergence)
- **SMC** (Smart Money Concepts - Order Blocks)
- **Support/Resistance Levels**
- **ADX** (Average Directional Index for trend filtering)

The system generates combined trading signals when multiple indicators align and can operate in both backtest and real-time modes.

## Core Architecture

- **StockScanner** (`new_scan.py`): Main orchestrator that coordinates all analyzers
- **Indicator Analyzers**: Each technical indicator has its own analyzer module:
  - `squeeze_momentum_analyzer.py`: LazyBear Squeeze Momentum implementation
  - `macd_analyzer.py`: MACD calculations and signals
  - `smc_analyzer.py`: Smart Money Concepts Order Block detection
  - `sr_levels_analyzer.py`: Support/Resistance level identification
  - `adx_analyzer.py`: ADX trend strength filtering

## Running the System

### Development/Testing Commands
```bash
# Run backtest mode (uses Yahoo Finance data)
python3 new_scan.py --backtest

# Generate chart for specific symbol
python3 new_scan.py --chart AAPL

# Run real-time scanning (requires IBKR connection)
python3 new_scan.py
```

### Dependencies
Install requirements:
```bash
pip install -r requirements.txt
```

## Configuration

The system is configured via `settings.json`:
- **data_folder**: Where historical data is cached
- **backtest**: Parameters for backtesting mode (candle count, interval)
- **realtime**: IBKR connection settings and real-time parameters
- **indicators**: Enable/disable and configure each indicator

Individual stock watchlist is defined in `config.txt` (symbol + provider format).

## Signal Logic

The system implements exact Pine Script logic for combined signals:
- **Buy Signal**: Squeeze Momentum turns LIME AND MACD turns GREEN (transition required)
- **Sell Signal**: Squeeze Momentum turns RED AND MACD turns RED (transition required)
- **ADX Filter**: Optional trend strength filtering (when enabled)
- **Alternating Signals**: System enforces buy→sell→buy pattern to avoid duplicate signals

## Data Sources

- **Backtest Mode**: Yahoo Finance (`yfinance`)
- **Real-time Mode**: Interactive Brokers (`ib_insync`)

Data is cached in the `data/` folder with filename format: `{date}_{symbol}_{candle_count}_{interval}.csv`

## Output

- **Console**: Real-time alerts with emoji indicators (🟢 for buy, 🔴 for sell)
- **Charts**: HTML charts generated in `chart/` folder using Plotly when signals trigger
- **Charts include**: Candlesticks, Order Blocks, Support/Resistance levels, and signal markers

## Calibration Files

The `squeeze_momentum_analyzer.py` includes calibration factors optimized to match TradingView's exact values:
- Multiple analysis and optimization scripts in repo (`analyze_*.py`, `optimize_*.py`, `verify_*.py`)
- Calibration ensures momentum calculations match TradingView reference implementation