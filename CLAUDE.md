# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **C-based trading system** that analyzes NASDAQ 100 stocks using IBKR API integration. The system was ported from Python to C for performance and includes comprehensive data caching, market hours validation, and signal processing capabilities.

The core functionality includes:
- **Real-time NASDAQ 100 scanning** (99 symbols)
- **IBKR Gateway integration** for market data and trading
- **EST timezone-aware operations** for US market hours
- **Smart caching system** with opening/continuous scan logic
- **Memory-safe implementation** with proper signal handling

## Core Architecture

### Main Components
- **trading_system.c**: Main trading loop and orchestration with global memory management
- **data_manager.c**: Market data caching and EST timezone management  
- **ibkr_connector.c**: Interactive Brokers API interface (stub implementation)
- **nasdaq_symbols.c**: Hardcoded NASDAQ 100 symbol list
- **settings_parser.c**: JSON configuration file parser

### Key Modules
```c
// Main trading system
./trading_system              // Main executable

// Core modules
trading_system.h/c            // Main trading loop and signal handlers
data_manager.h/c              // Data caching with EST timezone support
ibkr_connector.h/c            // IBKR API interface (currently stubs)
nasdaq_symbols.h/c            // NASDAQ 100 symbol management
settings_parser.h/c           // JSON settings configuration parser
```

## Running the System

### Build and Run
```bash
# Build the system
make

# Run trading system
./trading_system

# Kill with Ctrl+C (proper memory cleanup via signal handler)
```

### Build System
- **Makefile**: Comprehensive build system with debug support
- **Dependencies**: Standard C libraries, no external dependencies required
- **Debug mode**: Extensive logging for development and troubleshooting

## Opening vs Continuous Scan Logic

The system implements two scan modes:

### Opening Scan
Triggered when:
- No `data/` folder exists
- No `.scan` file exists  
- Date in `.scan` differs from today (EST)
- `scanned=0` flag in `.scan` file

**Actions:**
- Cleans old cache files (7+ days)
- Downloads/caches all 99 NASDAQ symbols via `prepare_cache()`
- Updates `.scan` file with current date and `scanned=1`
- Ignores market hours for testing purposes

### Continuous Scan  
Triggered when:
- `.scan` file exists with today's date (EST)
- `scanned=1` flag is set

**Actions:**
- Respects market hours (9:30-16:00 EST, Mon-Fri)
- Uses existing cached data
- Performs analysis on cached data without re-downloading

## Configuration

The system is configured via `settings.json`:
- **realtime.ibkr_host**: IBKR Gateway host (default: 127.0.0.1)
- **realtime.ibkr_port**: IBKR Gateway port (default: 4002)  
- **realtime.ibkr_client_id**: Client ID for IBKR connection
- **realtime.candle_nb**: Number of candles to download (default: 200)
- **indicators**: Configuration for future indicator implementations

## Data Management

### Caching System
- **Location**: `data/` folder
- **Format**: `{date}_{symbol}_{candle_count}_{interval}.csv`
- **Example**: `data/2025-12-05_AAPL_200_1d.csv`
- **Cleanup**: Automatic removal of files older than 7 days

### EST Timezone Handling
```c
// Proper EST time calculation (UTC-5)
int data_manager_get_est_time(struct tm* tm_est) {
    time_t now = time(NULL);
    time_t est_timestamp = now - (5 * 60 * 60); // UTC-5
    struct tm* est_time = gmtime(&est_timestamp);
    *tm_est = *est_time;
    return 1;
}
```

### Market Hours Validation
- **Trading Hours**: 9:30-16:00 EST, Monday-Friday
- **Outside Hours**: System sleeps 5 minutes between checks
- **Test Mode**: Opening scan ignores market hours for development

## Memory Management

### Global Buffer System
The system now uses a **global buffer allocation strategy** to eliminate repeated malloc/free operations:

```c
// Global reusable buffer
static IBKRMarketData* g_market_data = NULL;

// Allocate once at startup
int init_global_market_data_buffer() {
    g_market_data = malloc(sizeof(IBKRMarketData));
    g_market_data->candles = malloc(sizeof(IBKRCandle) * g_settings.realtime.candle_nb);
    return 1;
}

// Reuse buffer for all symbols
int data_manager_load_into_buffer(IBKRConnection* conn, const char* symbol, 
                                  int candle_count, const char* interval,
                                  int force_refresh, IBKRMarketData* buffer);
```

### Signal Handling
```c
void signal_handler(int sig) {
    // Clean up global variables
    if(g_symbols) {
        free(g_symbols);
        g_symbols = NULL;
    }
    if(g_conn) {
        ibkr_free_connection(g_conn);
        g_conn = NULL;
    }
    if(g_market_data) {
        if(g_market_data->candles) {
            free(g_market_data->candles);
        }
        free(g_market_data);
        g_market_data = NULL;
    }
    exit(0);
}
```

### Memory Efficiency
- **Single Allocation**: One global buffer allocated at startup (no malloc/free in scan loops)
- **Buffer Reuse**: Same buffer used for all 99 symbols during scanning
- **Proper Cleanup**: Memory freed only under Ctrl+C signal handling

## IBKR Integration

### Current Status
- **Stub Implementation**: All IBKR functions are currently stubs for development
- **Connection**: Simulated connection to 127.0.0.1:4002
- **Data Generation**: Fake market data for testing

### Future Implementation
- Replace stubs with actual IBKR API calls
- Implement real market data downloads
- Add order placement and position management

## Status Tracking

### .scan File Format
```
date=2025-12-05
scanned=1
```

### Debug Output
Extensive logging with debug modes:
- `data_manager_set_debug_mode(1)`: Data operations logging
- `ibkr_set_debug_mode(1)`: IBKR operations logging  
- `main_set_debug_mode(0)`: Main system logging (currently disabled)

## Key Functions

### Core System Functions
- `main_trading_loop()`: Main system loop with market hours and scan logic
- `prepare_cache()`: Downloads and caches market data for all symbols (opening scan only)
- `is_opening_scan()`: Determines scan type based on .scan file and current date
- `signal_handler()`: Proper cleanup on Ctrl+C with memory deallocation
- `init_global_market_data_buffer()`: Initialize global buffer once at startup

### Data Management Functions  
- `data_manager_get_market_data()`: Primary interface for market data (cache-first)
- `data_manager_load_into_buffer()`: Load data into existing buffer (no malloc/free)
- `data_manager_get_est_time()`: EST timezone conversion from UTC
- `data_manager_is_market_hours()`: Market hours validation (9:30-16:00 EST)
- `data_manager_cleanup_old_cache()`: Automatic cleanup of old cache files

### Settings Management Functions
- `settings_load()`: Load configuration from settings.json
- `settings_init_defaults()`: Initialize with default values
- `settings_parse_json_line()`: Parse simple JSON key-value pairs
- `settings_parse_int/double/bool()`: Type-safe parsing with validation

### NASDAQ Symbol Management
- `get_nasdaq100_symbols()`: Load all 99 NASDAQ symbols into dynamic array
- `get_nasdaq100_count()`: Get symbol count for dynamic allocation
- `is_nasdaq100_symbol()`: Validate symbol membership

## Current Implementation Status

✅ **Complete:**
- EST timezone handling and market hours validation
- Global buffer memory management (eliminates malloc/free in loops)
- JSON settings parser with type-safe validation
- Opening/continuous scan logic with .scan file tracking
- Comprehensive data caching system
- NASDAQ 100 symbol management
- IBKR connection framework (stubs)
- Signal handling with proper memory cleanup

🚧 **In Development:**
- Actual IBKR API integration
- Signal analysis algorithms (TODO sections in main loop)
- Order placement and position management
- Stop-loss calculation and tracking

📋 **Planned:**
- Real-time signal detection
- Chart generation and visualization
- Performance optimization and testing
- Comprehensive error handling
- Unit testing framework