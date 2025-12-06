#include "trading_system.h"
#include "nasdaq_symbols.h"
#include "ibkr_connector.h"
#include "data_manager.h"
#include "settings_parser.h"
#include <signal.h>

/* Global variables for debug */
static int debug_mode = 0;

// ===== VARIABLES GLOBALES =====
static char (*g_symbols)[MAX_SYMBOL_LEN] = NULL;
static IBKRConnection* g_conn = NULL;
static AppSettings g_settings;
static IBKRMarketData* g_market_data = NULL; /* Global reusable buffer  */

void signal_handler(int sig) {
    printf("\n[TRADING] Received signal %d, cleaning up...\n", sig);
    
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
    
    printf("[TRADING] Cleanup complete, exiting...\n");
    exit(0);
}

void main_set_debug_mode(int enable) {
    debug_mode = enable;
    if(debug_mode)
        printf("[IBKR] Debug mode enabled\n");
}

/* Indicctors */

double* calculate_squeeze_momentum(MarketData* data) {
    printf("[STUB] Calculating squeeze momentum for %s\n", data->symbol);
    
    double* momentum = malloc(sizeof(double) * data->candle_count);
    
    // Stub: valeurs fictives
    for(int i = 0; i < data->candle_count; i++) {
        momentum[i] = (double)(i % 10) - 5.0;  // Oscillation -5 to +5
    }
    
    return momentum;
}

/* Signaux
Signal* detect_signals(double* momentum, int count, int* signal_count) {
    printf("[STUB] Detecting signals from momentum\n");
    
    // Stub: create some fake signals
    Signal* signals = malloc(sizeof(Signal) * 10);
    *signal_count = 2;
    
    // Signal BUY fictif
    signals[0].index = count - 3;
    strcpy(signals[0].type, "bullish");
    strcpy(signals[0].color, "maroon");
    signals[0].momentum = -2.5;
    
    // Signal SELL fictif
    signals[1].index = count - 1;
    strcpy(signals[1].type, "bearish");
    strcpy(signals[1].color, "green");
    signals[1].momentum = 3.2;
    
    return signals;
} */

int execute_trade(const char* symbol, const char* action) {
    printf("[STUB] Executing %s order for %s\n", action, symbol);
    return 1; // Success
}

// Stop-Loss
double calculate_sl_level(const char* action, double high_signal, double low_signal, 
                         double high_prev, double low_prev) {
    printf("[STUB] Calculating SL for %s action\n", action);
    
    if(strcmp(action, "BUY") == 0) {
        return (high_signal > high_prev) ? high_signal : high_prev;
    } else {
        return (low_signal < low_prev) ? low_signal : low_prev;
    }
}

/*

int save_position_tracking(const char* symbol, const char* action, 
                          double sl_level, const char* entry_date) {
    printf("[STUB] Saving position tracking: %s %s SL=%.2f\n", 
           symbol, action, sl_level);
    return 1; // Success
} */

Position* load_position_tracking(int* position_count) {
    printf("[STUB] Loading position tracking\n");
    
    // Stub: no tracked positions
    *position_count = 0;
    return NULL;
}

/*
int check_sl_hits(Position* positions, int position_count, MarketData* market_data) {
    printf("[STUB] Checking SL hits for %d positions\n", position_count);
    return 0; // No SL hit
}
*/

// Scan status
int is_opening_scan(char *current_date) {

	if (! data_manager_cache_exists("data"))
		return 1;
    
    struct tm tm_est;
    if(!data_manager_get_est_time(&tm_est))
        return 1;
    
    /* Read the file .scan */
    FILE* scan_file = fopen(".scan", "r");
    if(!scan_file)
        return 1; /* No file = opening scan */
    
    int scanned_flag = 0;
    char file_date[32];
    strftime(current_date, sizeof(current_date), "%Y-%m-%d", &tm_est);
    
    /* Format: date=YYYY-MM-DD\nscanned=1  */
    if(fscanf(scan_file, "date=%31s\nscanned=%d", file_date, &scanned_flag) != 2) {
        fclose(scan_file);
        return 1; /* Read error = opening scan */
    }

    fclose(scan_file);
    
    // If different date or not yet scanned today
    if(strcmp(file_date, current_date) != 0 || !scanned_flag)
        return 1; /* Good for download */
    
    return 0; /* Continuous scan  */
}

void update_scan_status(const char* date) {
    if(!date) return;
    
    FILE* scan_file = fopen(".scan", "w");
    if(!scan_file) {
        printf("ERROR: Cannot create .scan file\n");
        return;
    }
    
    // Write date and scanned flag
    fprintf(scan_file, "date=%s\nscanned=1\n", date);
    fclose(scan_file);
    
    if(debug_mode) {
        printf("[TRADING] Updated .scan file for %s\n", date);
    }
}

// Nettoyage
void cleanup_data() {
    printf("[TRADING] Cleaning up old cache files\n");
    int deleted = data_manager_cleanup_old_cache(7); // 7 jours
    printf("[TRADING] Deleted %d old cache files\n", deleted);
}

void free_market_data(MarketData* data) {
    if(data) {
        free(data->candles);
        free(data);
    }
}

static int prepare_cache(int loaded_count) {

	int i;
	int processed = 0;

	// Scan all the symbols
	for (i = 0; i < loaded_count; i++) {

		// Load market data into global buffer (no malloc)
		if (!data_manager_load_into_buffer(g_conn,
				g_symbols[i], g_settings.realtime.candle_nb, 
				g_settings.realtime.interval, 0, g_market_data)) {
			printf("FAILED (no data)\n");
			continue; // Continue with next symbol instead of returning
		}

		if (debug_mode)
			printf("OK (%d candles)\n", g_market_data->candle_count);

		processed++;

	}

	return 0;

}

// ===== ALLOCATION GLOBALE =====

int init_global_market_data_buffer() {
	if(debug_mode) {
		printf("[TRADING] Initializing global market data buffer\n");
	}
	
	// Allocate the main structure
	g_market_data = malloc(sizeof(IBKRMarketData));
	if(!g_market_data) {
		printf("ERROR: Failed to allocate global market data buffer\n");
		return 0;
	}
	
	// Allocate the buffer for candles for maximum possible
	g_market_data->candles = malloc(sizeof(IBKRCandle) * g_settings.realtime.candle_nb);
	if(!g_market_data->candles) {
		printf("ERROR: Failed to allocate global candles buffer\n");
		free(g_market_data);
		g_market_data = NULL;
		return 0;
	}
	
	// Initialize default values
	strcpy(g_market_data->symbol, "");
	g_market_data->candle_count = 0;
	
	if(debug_mode) {
		printf("[TRADING] Global buffer allocated: %d candles max\n", g_settings.realtime.candle_nb);
	}
	
	return 1;
}

// ===== MAIN LOOP =====

void main_trading_loop() {

	printf("=== TRADING SYSTEM STARTED ===\n");

	// Installer signal handler
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	// Charger settings depuis settings.json
	printf("[TRADING] Loading settings from settings.json...\n");
	if(!settings_load("settings.json", &g_settings)) {
		printf("WARNING: Cannot load settings.json, using defaults\n");
		settings_init_defaults(&g_settings);
	} else {
		printf("[TRADING] Settings loaded successfully\n");
	}
	
	// Afficher settings importantes
	printf("[TRADING] Configuration:\n");
	printf("  Data folder: %s\n", g_settings.global.data_folder);
	printf("  IBKR: %s:%d (client_id=%d)\n", 
	       g_settings.realtime.ibkr_host,
	       g_settings.realtime.ibkr_port,
	       g_settings.realtime.ibkr_client_id);
	printf("  Candles: %d (%s)\n",
	       g_settings.realtime.candle_nb,
	       g_settings.realtime.interval);

	// Initialize global buffer for market data
	if(!init_global_market_data_buffer()) {
		printf("ERROR: Failed to initialize global market data buffer\n");
		return;
	}

	/* Initialize data manager - create /data */

	if (data_manager_init() == TRADE_FAIL) {
		printf("ERROR: Failed to initialize data manager\n");
		return;
	}

	// Enable debug
	data_manager_set_debug_mode(1);
	ibkr_set_debug_mode(1);
	main_set_debug_mode(0);

	// IBKR connection from settings
	g_conn = ibkr_init(g_settings.realtime.ibkr_host, 
	                   g_settings.realtime.ibkr_port, 
	                   g_settings.realtime.ibkr_client_id);
	if (!g_conn || !ibkr_connect(g_conn)) {
		printf("ERROR: Cannot connect to IBKR\n");
		return;
	}

	/* Load all the symbols NASDAQ */

	int symbol_count = get_nasdaq100_count();
	g_symbols = malloc(symbol_count * sizeof(*g_symbols));
	if (!g_symbols) {
		printf("ERROR: Failed to allocate memory for symbols\n");
		ibkr_free_connection(g_conn);
		return;
	}

	/* Load the array of symbols */
	int loaded_count = get_nasdaq100_symbols(g_symbols, symbol_count);

	printf("[TRADING] Loaded %d/%d NASDAQ symbols\n", loaded_count,
			symbol_count);

	while (1) {
		struct tm tm_est;
		if (data_manager_get_est_time(&tm_est)) {
			printf("\n=== SCAN %02d:%02d:%02d EST ===\n", tm_est.tm_hour,
					tm_est.tm_min, tm_est.tm_sec);
		}
#if 0
		if (!data_manager_is_market_hours()) {
			printf("[TRADING] Outside market hours (9:30-16:00 EST, Mon-Fri)\n");
			printf("Sleeping 5 minutes...\n");
			sleep(300); /* 5 minutes */
			continue;
		}
#endif
		/* Check scan type */

		char current_date[32];
		if (is_opening_scan(current_date)) {
			printf("*** OPENING SCAN ***\n");
			cleanup_data();
			update_scan_status(current_date);
			if (prepare_cache(loaded_count))
				if (debug_mode)
					printf("Loading symbols failed\n");

			continue;

		} else
			printf("*** CONTINUOUS SCAN ***\n");

		printf("\nSleeping 5 seconds...\n");
		sleep(5);

		/* Close  connection */

		ibkr_free_connection(g_conn);
		g_conn = NULL;
	}
}

// ===== ENTRY POINT =====

int main(int argc, char* argv[]) {
    printf("Trading System v1.0 (C Implementation)\n");
    printf("Press Ctrl+C to stop\n\n");
    
    main_trading_loop();
    
    return 0;
}

