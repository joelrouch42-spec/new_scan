#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <time.h>
#include "ibkr_connector.h"
#include "trading_system.h"

#define MAX_FILENAME_LEN 512
#define MAX_DATE_LEN 32
#define DATA_FOLDER "data"

// ===== STRUCTURES =====

typedef struct {
    char filename[MAX_FILENAME_LEN];
    char symbol[MAX_SYMBOL_LEN];
    char date[MAX_DATE_LEN];
    int candle_count;
    char interval[16];
    time_t created_time;
} CacheInfo;

// ===== PROTOTYPES CACHE =====

/**
 * Initialize data/ cache folder
 * @return: 1 if success, 0 if error
 */
TRADE_STATUS data_manager_init(void);

/**
 * Generate cache filename for a symbol
 * @param symbol: Symbol (ex: "AAPL")
 * @param candle_count: Number of candles
 * @param interval: Interval (ex: "1d")
 * @param date: Date (ex: "2025-12-04")
 * @param filename: Buffer to store the name (must be MAX_FILENAME_LEN)
 * @return: 1 if success, 0 if error
 */
void data_manager_get_cache_filename(const char* symbol, int candle_count,
                                   const char* interval, const char* date,
                                   char* filename);

/**
 * Check if cache file exists
 * @param filename: Cache filename
 * @return: 1 if exists, 0 otherwise
 */
int data_manager_cache_exists(const char* filename);

/**
 * Save market data to cache
 * @param data: Data to save
 * @param filename: Cache filename
 * @return: 1 if success, 0 if error
 */
int data_manager_save_to_cache(IBKRMarketData* data, const char* filename);

/**
 * Load data from cache
 * @param filename: Cache filename
 * @return: Loaded data or NULL if error
 */
IBKRMarketData* data_manager_load_from_cache(const char* filename);

/**
 * Load data with smart cache (cache first, then IBKR if necessary)
 * @param conn: Active IBKR connection
 * @param symbol: Symbol
 * @param candle_count: Number of candles
 * @param interval: Interval
 * @param force_refresh: 1 to force download, 0 to use cache if possible
 * @return: Data or NULL if error
 */
IBKRMarketData* data_manager_get_market_data(IBKRConnection* conn,
                                            const char* symbol,
                                            int candle_count,
                                            const char* interval,
                                            int force_refresh);

/**
 * Load data into existing buffer (no malloc/free)
 * @param conn: IBKR connection
 * @param symbol: Symbol
 * @param candle_count: Number of candles
 * @param interval: Interval
 * @param force_refresh: 1 to force download
 * @param buffer: Existing buffer to fill
 * @return: 1 if success, 0 if error
 */
int data_manager_load_into_buffer(IBKRConnection* conn,
                                  const char* symbol,
                                  int candle_count,
                                  const char* interval,
                                  int force_refresh,
                                  IBKRMarketData* buffer);

// ===== PROTOTYPES NETTOYAGE =====

/**
 * Remove all files from data/ folder
 * @return: Number of files removed
 */
int data_manager_cleanup_cache(void);

/**
 * Remove cache files older than N days
 * @param days: Number of days (ex: 7 to remove > 1 week)
 * @return: Number of files removed
 */
int data_manager_cleanup_old_cache(int days);

/**
 * List all cache files
 * @param cache_list: Array to store info (must be allocated)
 * @param max_count: Max array size
 * @return: Number of files found
 */
int data_manager_list_cache_files(CacheInfo* cache_list, int max_count);

// ===== PROTOTYPES UTILITAIRES =====

/**
 * Get current date in format YYYY-MM-DD
 * @param date_buffer: Buffer to store date (must be MAX_DATE_LEN)
 * @return: 1 if success, 0 if error
 */
int data_manager_get_current_date(char* date_buffer);

/**
 * Check if folder exists
 * @param folder_path: Path to folder
 * @return: 1 if exists, 0 otherwise
 */
int data_manager_folder_exists(const char* folder_path);

/**
 * Create folder (with parents if necessary)
 * @param folder_path: Path to folder
 * @return: 1 if success, 0 if error
 */
int data_manager_create_folder(const char* folder_path);

/**
 * Get file size
 * @param filename: Filename
 * @return: Size in bytes or -1 if error
 */
long data_manager_get_file_size(const char* filename);

/**
 * Enable/disable debug mode
 * @param enable: 1 to enable, 0 to disable
 */
void data_manager_set_debug_mode(int enable);

/**
 * Get current time in EST
 * @param tm_est: tm structure to store EST time
 * @return: 1 if success, 0 if error
 */
int data_manager_get_est_time(struct tm* tm_est);

/**
 * Check if in market hours (9:30-16:00 EST, Mon-Fri)
 * @return: 1 if market hours, 0 otherwise
 */
int data_manager_is_market_hours(void);

#endif // DATA_MANAGER_H
