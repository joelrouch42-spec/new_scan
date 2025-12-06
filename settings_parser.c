#include "settings_parser.h"
#include <ctype.h>
#include <limits.h>

// Variables globales
static int debug_mode = 0;

// ===== UTILITAIRES =====

void settings_set_debug_mode(int enable) {
    debug_mode = enable;
    if(debug_mode) {
        printf("[SETTINGS] Debug mode enabled\n");
    }
}

int settings_parse_int(const char* value, int* result) {
    if(!value || !result) return 0;
    
    char* endptr;
    long val = strtol(value, &endptr, 10);
    
    if(*endptr != '\0' || val > INT_MAX || val < INT_MIN) {
        return 0;
    }
    
    *result = (int)val;
    return 1;
}

int settings_parse_double(const char* value, double* result) {
    if(!value || !result) return 0;
    
    char* endptr;
    double val = strtod(value, &endptr);
    
    if(*endptr != '\0') {
        return 0;
    }
    
    *result = val;
    return 1;
}

int settings_parse_bool(const char* value, int* result) {
    if(!value || !result) return 0;
    
    if(strcmp(value, "true") == 0) {
        *result = 1;
        return 1;
    } else if(strcmp(value, "false") == 0) {
        *result = 0;
        return 1;
    }
    
    return 0;
}

// Function to clean spaces and quotes
void settings_trim_quotes(char* str) {
    if(!str) return;
    
    int len = strlen(str);
    
    // Remove spaces at start
    int start = 0;
    while(start < len && isspace(str[start])) start++;
    
    // Remove spaces at end
    int end = len - 1;
    while(end > start && (isspace(str[end]) || str[end] == ',')) end--;
    
    // Remove quotes
    if(str[start] == '"') start++;
    if(str[end] == '"') end--;
    
    // Shift and null-terminate
    memmove(str, str + start, end - start + 1);
    str[end - start + 1] = '\0';
}

int settings_parse_json_line(const char* line, char* key, char* value) {
    if(!line || !key || !value) return 0;
    
    // Search for the ':'
    const char* colon = strchr(line, ':');
    if(!colon) return 0;
    
    // Extract the key
    int key_len = colon - line;
    strncpy(key, line, key_len);
    key[key_len] = '\0';
    settings_trim_quotes(key);
    
    // Extract the value
    strcpy(value, colon + 1);
    settings_trim_quotes(value);
    
    return 1;
}

// ===== INITIALISATION =====

void settings_init_defaults(AppSettings* settings) {
    if(!settings) return;
    
    // Global settings
    strcpy(settings->global.data_folder, "data");
    
    // Backtest settings
    settings->backtest.candle_nb = 200;
    strcpy(settings->backtest.interval, "1d");
    strcpy(settings->backtest.data_source, "ibkr");
    settings->backtest.test_candle_start = 0;
    settings->backtest.test_candle_stop = 100;
    
    // Realtime settings
    settings->realtime.candle_nb = 200;
    strcpy(settings->realtime.interval, "1d");
    settings->realtime.update_interval_seconds = 60;
    settings->realtime.test_candle = 0;
    strcpy(settings->realtime.ibkr_host, "127.0.0.1");
    settings->realtime.ibkr_port = 4002;
    settings->realtime.ibkr_client_id = 2;
    
    // Support/Resistance settings
    settings->indicators.support_resistance.enabled = 1;
    settings->indicators.support_resistance.left_bars = 15;
    settings->indicators.support_resistance.right_bars = 15;
    settings->indicators.support_resistance.volume_threshold = 20;
    settings->indicators.support_resistance.show_breaks = 1;
    
    // Squeeze Momentum settings
    settings->indicators.squeeze_momentum.enabled = 1;
    settings->indicators.squeeze_momentum.bb_length = 20;
    settings->indicators.squeeze_momentum.bb_mult = 2.0;
    settings->indicators.squeeze_momentum.kc_length = 20;
    settings->indicators.squeeze_momentum.kc_mult = 1.5;
    settings->indicators.squeeze_momentum.use_true_range = 1;
    
    if(debug_mode) {
        printf("[SETTINGS] Initialized with default values\n");
    }
}

// ===== PARSING =====

int settings_load(const char* filename, AppSettings* settings) {
    if(!filename || !settings) return 0;
    
    if(debug_mode) {
        printf("[SETTINGS] Loading from %s\n", filename);
    }
    
    FILE* file = fopen(filename, "r");
    if(!file) {
        printf("ERROR: Cannot open settings file %s\n", filename);
        return 0;
    }
    
    // Initialize with defaults
    settings_init_defaults(settings);
    
    char line[512];
    char current_section[64] = "";
    char subsection[64] = "";
    
    while(fgets(line, sizeof(line), file)) {
        // Ignore empty lines and comments
        if(line[0] == '\n' || line[0] == '\r' || line[0] == '#') continue;
        
        // Détecter the sections
        if(strstr(line, "\"data_folder\"")) {
            strcpy(current_section, "global");
        } else if(strstr(line, "\"backtest\"")) {
            strcpy(current_section, "backtest");
        } else if(strstr(line, "\"realtime\"")) {
            strcpy(current_section, "realtime");
        } else if(strstr(line, "\"indicators\"")) {
            strcpy(current_section, "indicators");
        } else if(strstr(line, "\"support_resistance\"")) {
            strcpy(current_section, "indicators");
            strcpy(subsection, "support_resistance");
        } else if(strstr(line, "\"squeeze_momentum\"")) {
            strcpy(current_section, "indicators");
            strcpy(subsection, "squeeze_momentum");
        }
        
        // Parser the clés-valeurs
        char key[128], value[128];
        if(settings_parse_json_line(line, key, value)) {
            if(debug_mode) {
                printf("[SETTINGS] [%s.%s] %s = %s\n", current_section, subsection, key, value);
            }
            
            // Global settings
            if(strcmp(current_section, "global") == 0) {
                if(strcmp(key, "data_folder") == 0) {
                    strncpy(settings->global.data_folder, value, sizeof(settings->global.data_folder) - 1);
                }
            }
            // Backtest settings
            else if(strcmp(current_section, "backtest") == 0) {
                if(strcmp(key, "candle_nb") == 0) {
                    settings_parse_int(value, &settings->backtest.candle_nb);
                } else if(strcmp(key, "interval") == 0) {
                    strncpy(settings->backtest.interval, value, sizeof(settings->backtest.interval) - 1);
                } else if(strcmp(key, "data_source") == 0) {
                    strncpy(settings->backtest.data_source, value, sizeof(settings->backtest.data_source) - 1);
                } else if(strcmp(key, "test_candle_start") == 0) {
                    settings_parse_int(value, &settings->backtest.test_candle_start);
                } else if(strcmp(key, "test_candle_stop") == 0) {
                    settings_parse_int(value, &settings->backtest.test_candle_stop);
                }
            }
            // Realtime settings
            else if(strcmp(current_section, "realtime") == 0) {
                if(strcmp(key, "candle_nb") == 0) {
                    settings_parse_int(value, &settings->realtime.candle_nb);
                } else if(strcmp(key, "interval") == 0) {
                    strncpy(settings->realtime.interval, value, sizeof(settings->realtime.interval) - 1);
                } else if(strcmp(key, "update_interval_seconds") == 0) {
                    settings_parse_int(value, &settings->realtime.update_interval_seconds);
                } else if(strcmp(key, "test_candle") == 0) {
                    settings_parse_int(value, &settings->realtime.test_candle);
                } else if(strcmp(key, "ibkr_host") == 0) {
                    strncpy(settings->realtime.ibkr_host, value, sizeof(settings->realtime.ibkr_host) - 1);
                } else if(strcmp(key, "ibkr_port") == 0) {
                    settings_parse_int(value, &settings->realtime.ibkr_port);
                } else if(strcmp(key, "ibkr_client_id") == 0) {
                    settings_parse_int(value, &settings->realtime.ibkr_client_id);
                }
            }
            // Indicators settings
            else if(strcmp(current_section, "indicators") == 0) {
                if(strcmp(subsection, "support_resistance") == 0) {
                    if(strcmp(key, "enabled") == 0) {
                        settings_parse_bool(value, &settings->indicators.support_resistance.enabled);
                    } else if(strcmp(key, "left_bars") == 0) {
                        settings_parse_int(value, &settings->indicators.support_resistance.left_bars);
                    } else if(strcmp(key, "right_bars") == 0) {
                        settings_parse_int(value, &settings->indicators.support_resistance.right_bars);
                    } else if(strcmp(key, "volume_threshold") == 0) {
                        settings_parse_int(value, &settings->indicators.support_resistance.volume_threshold);
                    } else if(strcmp(key, "show_breaks") == 0) {
                        settings_parse_bool(value, &settings->indicators.support_resistance.show_breaks);
                    }
                } else if(strcmp(subsection, "squeeze_momentum") == 0) {
                    if(strcmp(key, "enabled") == 0) {
                        settings_parse_bool(value, &settings->indicators.squeeze_momentum.enabled);
                    } else if(strcmp(key, "bb_length") == 0) {
                        settings_parse_int(value, &settings->indicators.squeeze_momentum.bb_length);
                    } else if(strcmp(key, "bb_mult") == 0) {
                        settings_parse_double(value, &settings->indicators.squeeze_momentum.bb_mult);
                    } else if(strcmp(key, "kc_length") == 0) {
                        settings_parse_int(value, &settings->indicators.squeeze_momentum.kc_length);
                    } else if(strcmp(key, "kc_mult") == 0) {
                        settings_parse_double(value, &settings->indicators.squeeze_momentum.kc_mult);
                    } else if(strcmp(key, "use_true_range") == 0) {
                        settings_parse_bool(value, &settings->indicators.squeeze_momentum.use_true_range);
                    }
                }
            }
        }
    }
    
    fclose(file);
    
    if(debug_mode) {
        printf("[SETTINGS] Successfully loaded from %s\n", filename);
    }
    
    return 1;
}

void settings_print(const AppSettings* settings) {
    if(!settings) return;
    
    printf("=== SETTINGS CONFIGURATION ===\n");
    
    printf("[GLOBAL]\n");
    printf("  data_folder: %s\n", settings->global.data_folder);
    
    printf("[BACKTEST]\n");
    printf("  candle_nb: %d\n", settings->backtest.candle_nb);
    printf("  interval: %s\n", settings->backtest.interval);
    printf("  data_source: %s\n", settings->backtest.data_source);
    printf("  test_candle_start: %d\n", settings->backtest.test_candle_start);
    printf("  test_candle_stop: %d\n", settings->backtest.test_candle_stop);
    
    printf("[REALTIME]\n");
    printf("  candle_nb: %d\n", settings->realtime.candle_nb);
    printf("  interval: %s\n", settings->realtime.interval);
    printf("  update_interval_seconds: %d\n", settings->realtime.update_interval_seconds);
    printf("  test_candle: %d\n", settings->realtime.test_candle);
    printf("  ibkr_host: %s\n", settings->realtime.ibkr_host);
    printf("  ibkr_port: %d\n", settings->realtime.ibkr_port);
    printf("  ibkr_client_id: %d\n", settings->realtime.ibkr_client_id);
    
    printf("[INDICATORS]\n");
    printf("  Support/Resistance:\n");
    printf("    enabled: %s\n", settings->indicators.support_resistance.enabled ? "true" : "false");
    printf("    left_bars: %d\n", settings->indicators.support_resistance.left_bars);
    printf("    right_bars: %d\n", settings->indicators.support_resistance.right_bars);
    printf("    volume_threshold: %d\n", settings->indicators.support_resistance.volume_threshold);
    printf("    show_breaks: %s\n", settings->indicators.support_resistance.show_breaks ? "true" : "false");
    
    printf("  Squeeze Momentum:\n");
    printf("    enabled: %s\n", settings->indicators.squeeze_momentum.enabled ? "true" : "false");
    printf("    bb_length: %d\n", settings->indicators.squeeze_momentum.bb_length);
    printf("    bb_mult: %.1f\n", settings->indicators.squeeze_momentum.bb_mult);
    printf("    kc_length: %d\n", settings->indicators.squeeze_momentum.kc_length);
    printf("    kc_mult: %.1f\n", settings->indicators.squeeze_momentum.kc_mult);
    printf("    use_true_range: %s\n", settings->indicators.squeeze_momentum.use_true_range ? "true" : "false");
    
    printf("===============================\n");
}