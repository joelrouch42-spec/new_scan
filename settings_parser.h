#ifndef SETTINGS_PARSER_H
#define SETTINGS_PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===== STRUCTURES =====

typedef struct {
    char data_folder[256];
} GlobalSettings;

typedef struct {
    int candle_nb;
    char interval[16];
    char data_source[32];
    int test_candle_start;
    int test_candle_stop;
} BacktestSettings;

typedef struct {
    int candle_nb;
    char interval[16];
    int update_interval_seconds;
    int test_candle;
    char ibkr_host[128];
    int ibkr_port;
    int ibkr_client_id;
} RealtimeSettings;

typedef struct {
    int enabled;
    int left_bars;
    int right_bars;
    int volume_threshold;
    int show_breaks;
} SupportResistanceSettings;

typedef struct {
    int enabled;
    int bb_length;
    double bb_mult;
    int kc_length;
    double kc_mult;
    int use_true_range;
} SqueezeMomentumSettings;

typedef struct {
    SupportResistanceSettings support_resistance;
    SqueezeMomentumSettings squeeze_momentum;
} IndicatorsSettings;

typedef struct {
    GlobalSettings global;
    BacktestSettings backtest;
    RealtimeSettings realtime;
    IndicatorsSettings indicators;
} AppSettings;

// ===== FONCTIONS =====

/**
 * Load the settings from the file JSON
 * @param filename: Nom of the file settings.json
 * @param settings: Pointeur to structure to remplir
 * @return: 1 if success, 0 if error
 */
int settings_load(const char* filename, AppSettings* settings);

/**
 * Initialize the settings with des valeurs by default
 * @param settings: Pointeur to structure to initialiser
 */
void settings_init_defaults(AppSettings* settings);

/**
 * Display the settings (mode debug)
 * @param settings: Pointer to structure to display
 */
void settings_print(const AppSettings* settings);

/**
 * Parse a simple JSON line (key: value)
 * @param line: Line to parse
 * @param key: Buffer to store the key
 * @param value: Buffer to store the value
 * @return: 1 if success, 0 if error
 */
int settings_parse_json_line(const char* line, char* key, char* value);

/**
 * Convert string value to int with validation
 * @param value: String to convert
 * @param result: Pointer to store the result
 * @return: 1 if success, 0 if error
 */
int settings_parse_int(const char* value, int* result);

/**
 * Convert string value to double with validation
 * @param value: String to convert
 * @param result: Pointer to store the result
 * @return: 1 if success, 0 if error
 */
int settings_parse_double(const char* value, double* result);

/**
 * Convert string value to boolean with validation
 * @param value: String to convert (true/false)
 * @param result: Pointer to store the result
 * @return: 1 if success, 0 if error
 */
int settings_parse_bool(const char* value, int* result);

/**
 * Enable/disable debug mode of the parser
 * @param enable: 1 to enable, 0 to disable
 */
void settings_set_debug_mode(int enable);

#endif // SETTINGS_PARSER_H