#ifndef TRADING_SYSTEM_H
#define TRADING_SYSTEM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// ===== CONSTANTES =====
#define MAX_SYMBOL_LEN 16
#define MAX_ACTION_LEN 8
#define MAX_TYPE_LEN 16
#define MAX_COLOR_LEN 16
#define MAX_DATE_LEN 32
#define MAX_CANDLES 500
#define MAX_SIGNALS 100
#define MAX_POSITIONS 200

typedef enum {
	TRADE_SUCCESS,
	TRADE_FAIL,
	TRADE_CONTINUOUS_SCAN,
	TRADE_OPENING_SCAN
}TRADE_STATUS;

// ===== STRUCTURES =====

typedef struct {
    char symbol[MAX_SYMBOL_LEN];
    char action[MAX_ACTION_LEN];     // "BUY" or "SELL"
    double sl_level;
    char entry_date[MAX_DATE_LEN];
} Position;

typedef struct {
    int index;
    char type[MAX_TYPE_LEN];         // "bullish" or "bearish"
    double momentum;
    char color[MAX_COLOR_LEN];       // "maroon", "green", etc.
} Signal;

typedef struct {
    char date[MAX_DATE_LEN];
    double open;
    double high;
    double low;
    double close;
    long volume;
} Candle;

typedef struct {
    char symbol[MAX_SYMBOL_LEN];
    Candle* candles;
    int candle_count;
} MarketData;

typedef struct {
    char date[MAX_DATE_LEN];
    int opening_scan_done;
} ScanStatus;

// ===== PROTOTYPES DONNÉES MARCHÉ =====
void free_market_data(MarketData* data);

// ===== PROTOTYPES INDICATEURS =====
double* calculate_squeeze_momentum(MarketData* data);
double* calculate_sma(double* data, int length, int period);
double* calculate_bollinger_bands(double* closes, int length, int period, double mult);
double* calculate_keltner_channel(double* highs, double* lows, double* closes, int length, int period, double mult);

// ===== PROTOTYPES SIGNAUX =====
Signal* detect_signals(double* momentum, int count, int* signal_count);
void free_signals(Signal* signals);

// ===== PROTOTYPES TRADING IBKR =====
int execute_trade(const char* symbol, const char* action);
double get_current_price(const char* symbol);
int get_position_size(const char* symbol);

// ===== PROTOTYPES STOP-LOSS =====
double calculate_sl_level(const char* action, double high_signal, double low_signal, 
                         double high_prev, double low_prev);
int save_position_tracking(const char* symbol, const char* action, 
                          double sl_level, const char* entry_date);
Position* load_position_tracking(int* position_count);
void free_positions(Position* positions);
int check_sl_hits_opening(Position* positions, int position_count, MarketData* market_data);
int check_sl_hits_continuous(Position* positions, int position_count, MarketData* market_data);
int remove_position_tracking(const char* symbol);
int update_position_sl(const char* symbol, double new_sl_level);

// ===== PROTOTYPES SCAN STATUS =====
ScanStatus* load_scan_status(void);
int is_opening_scan(char *);
void update_scan_status(const char* date);
void save_scan_status(ScanStatus* status);

// ===== PROTOTYPES UTILITAIRES =====
void cleanup_data(void);
char* get_current_date(void);
void print_signal(Signal* signal);
void print_position(Position* position);
void log_message(const char* level, const char* format, ...);

// ===== PROTOTYPES MAIN =====
void main_trading_loop(void);

#endif // TRADING_SYSTEM_H
