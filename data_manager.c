#include "data_manager.h"
#include "trading_system.h"
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

// Variables globales
static int debug_mode = 0;

// ===== FONCTIONS CACHE =====

TRADE_STATUS data_manager_init(void) {
    if(debug_mode) {
        printf("[DATA] Initializing data manager\n");
    }
    
    // Create data/ folder if it does not exist
    if(!data_manager_folder_exists(DATA_FOLDER)) {
        if(!data_manager_create_folder(DATA_FOLDER)) {
            printf("ERROR: Failed to create %s folder\n", DATA_FOLDER);
            return TRADE_FAIL;
        }
        if(debug_mode) {
            printf("[DATA] Created %s folder\n", DATA_FOLDER);
        }
    }
    
    return TRADE_SUCCESS;
}

/* define a name file for the stock */

void data_manager_get_cache_filename(const char* symbol, int candle_count,
                                   const char* interval, const char* date,
                                   char* filename) {
    
    /* Format: data/YYYY-MM-DD_SYMBOL_COUNT_INTERVAL.csv */

    snprintf(filename, MAX_FILENAME_LEN, "%s/%s_%s_%d_%s.csv", 
             DATA_FOLDER, date, symbol, candle_count, interval);
    
    if(debug_mode) {
        printf("[DATA] Cache filename: %s\n", filename);
    }
}

int data_manager_cache_exists(const char* dirname) {
    if(!dirname)
    	return 0;
    
    struct stat st;
    if(stat(dirname, &st) != 0)
        return 0;

    // Check that it is a directory
    if(!S_ISDIR(st.st_mode))
        return 0;

    // Open directory
    DIR *dir = opendir(dirname);
    if(!dir)
        return 0;

    struct dirent *entry;
    int file_count = 0;

    // Count the files (ignore "." and "..")
    while((entry = readdir(dir)) != NULL) {
        if(entry->d_name[0] != '.') {
            file_count++;
            break;  // Stop at first file found
        }
    }

    closedir(dir);
    
    if(debug_mode)
        printf("[DATA] Directory %s: %s\n", dirname, file_count > 0 ? "HAS FILES" : "EMPTY");
    
    return file_count > 0;
}

/* Save record */

int data_manager_save_to_cache(IBKRMarketData* data, const char* filename) {

    if(!data || !filename)
    	return 0;
    
    if(debug_mode) {
        printf("[DATA] Saving %d candles to %s\n", data->candle_count, filename);
    }
    
    FILE* file = fopen(filename, "w");
    if(!file) {
        printf("ERROR: Cannot create cache file %s\n", filename);
        return 0;
    }
    
    // Header CSV
    fprintf(file, "Date,Open,High,Low,Close,Volume\n");
    
    // Data
    for(int i = 0; i < data->candle_count; i++) {
        fprintf(file, "%s,%.2f,%.2f,%.2f,%.2f,%ld\n",
                data->candles[i].date,
                data->candles[i].open,
                data->candles[i].high,
                data->candles[i].low,
                data->candles[i].close,
                data->candles[i].volume);
    }
    
    fclose(file);
    
    if(debug_mode) {
        printf("[DATA] Saved %d candles to cache\n", data->candle_count);
    }
    
    return 1;
}

IBKRMarketData* data_manager_load_from_cache(const char* filename) {

    if(!filename)
    	return NULL;
    
    if(debug_mode)
        printf("[DATA] Loading from cache: %s\n", filename);
    
    FILE* file = fopen(filename, "r");
    if(!file) {
        if(debug_mode) {
            printf("[DATA] Cannot open cache file %s\n", filename);
        }
        return NULL;
    }
    
    // Create data structure
    IBKRMarketData* data = malloc(sizeof(IBKRMarketData));
    if(!data) {
        fclose(file);
        return NULL;
    }
    
    // Extract the symbol of the filename
    // Format: data/YYYY-MM-DD_SYMBOL_COUNT_INTERVAL.csv
    char* basename = strrchr(filename, '/');
    if(basename) basename++; // Skip '/'
    else basename = (char*)filename;
    
    char temp[MAX_FILENAME_LEN];
    strncpy(temp, basename, sizeof(temp)-1);
    
    // Parser: DATE_SYMBOL_COUNT_INTERVAL.csv
    char* token = strtok(temp, "_");
    if(token) token = strtok(NULL, "_"); // Skip date, get symbol
    if(token) {
        strncpy(data->symbol, token, sizeof(data->symbol) - 1);
        data->symbol[sizeof(data->symbol) - 1] = '\0';
    } else {
        strcpy(data->symbol, "UNKNOWN");
    }
    
    // Count the lines (exclude header)
    char line[512];
    int line_count = 0;
    
    // Skip header
    if(fgets(line, sizeof(line), file)) {
        // Count data lines
        while(fgets(line, sizeof(line), file)) {
            line_count++;
        }
    }
    
    if(line_count == 0) {
        if(debug_mode) {
            printf("[DATA] Empty cache file\n");
        }
        free(data);
        fclose(file);
        return NULL;
    }
    
    // Allouer tableau of bougies
    data->candle_count = line_count;
    data->candles = malloc(sizeof(IBKRCandle) * line_count);
    if(!data->candles) {
        free(data);
        fclose(file);
        return NULL;
    }
    
    // Go back to beginning and skip header
    rewind(file);
    fgets(line, sizeof(line), file);
    
    // Read data
    int i = 0;
    while(fgets(line, sizeof(line), file) && i < line_count) {
        char* date = strtok(line, ",");
        char* open = strtok(NULL, ",");
        char* high = strtok(NULL, ",");
        char* low = strtok(NULL, ",");
        char* close = strtok(NULL, ",");
        char* volume = strtok(NULL, ",\n");
        
        if(date && open && high && low && close && volume) {
            strncpy(data->candles[i].date, date, sizeof(data->candles[i].date) - 1);
            data->candles[i].date[sizeof(data->candles[i].date) - 1] = '\0';
            data->candles[i].open = atof(open);
            data->candles[i].high = atof(high);
            data->candles[i].low = atof(low);
            data->candles[i].close = atof(close);
            data->candles[i].volume = atol(volume);
            i++;
        }
    }
    
    data->candle_count = i; // Adjust to actually read count
    fclose(file);
    
    if(debug_mode) {
        printf("[DATA] Loaded %d candles from cache\n", data->candle_count);
    }
    
    return data;
}

IBKRMarketData* data_manager_get_market_data(IBKRConnection* conn,
                                            const char* symbol,
                                            int candle_count,
                                            const char* interval,
                                            int force_refresh) {

    if(!conn || !symbol || !interval)
    	return NULL;

    /* Get the current date */
    
    char current_date[MAX_DATE_LEN];
    if(!data_manager_get_current_date(current_date))
        return NULL;
    
    // Generate cache filename
    char cache_filename[MAX_FILENAME_LEN];
    
    data_manager_get_cache_filename(symbol, candle_count, interval,
                                       current_date, cache_filename);

    /*  If no force refresh and cache exists, load from cache
     * force_refresh = 0 - Use the cache if the file exists
     * force_refresh = 1 - Ignore cache and always download from IBKR */

    if(!force_refresh && data_manager_cache_exists(cache_filename)) {
        if(debug_mode)
            printf("[DATA] Loading %s from cache\n", symbol);

        return data_manager_load_from_cache(cache_filename);
    }
    
    // Otherwise download from IBKR
    if(debug_mode) {
        printf("[DATA] Downloading %s from IBKR\n", symbol);
    }
    
    IBKRMarketData* data = ibkr_download_historical_data(conn, symbol, 
                                                        candle_count, interval);
    if(!data) {
        printf("ERROR: Failed to download data for %s\n", symbol);
        return NULL;
    }
    
    // Save to cache
    if(!data_manager_save_to_cache(data, cache_filename)) {
        printf("WARNING: Failed to save cache for %s\n", symbol);
        // Continue anyway with data
    }
    
    return data;
}

int data_manager_load_into_buffer(IBKRConnection* conn,
                                  const char* symbol,
                                  int candle_count,
                                  const char* interval,
                                  int force_refresh,
                                  IBKRMarketData* buffer) {

    if(!conn || !symbol || !interval || !buffer) {
        return 0;
    }

    // Get the current date
    char current_date[MAX_DATE_LEN];
    if(!data_manager_get_current_date(current_date)) {
        return 0;
    }
    
    // Generate cache filename
    char cache_filename[MAX_FILENAME_LEN];
    data_manager_get_cache_filename(symbol, candle_count, interval,
                                   current_date, cache_filename);

    // If no force refresh and cache exists, load from cache in buffer
    if(!force_refresh && data_manager_cache_exists(cache_filename)) {
        if(debug_mode) {
            printf("[DATA] Loading %s from cache into buffer\n", symbol);
        }

        // Load from cache directly into buffer
        IBKRMarketData* temp_data = data_manager_load_from_cache(cache_filename);
        if(!temp_data) {
            return 0;
        }
        
        // Copy to our global buffer
        strcpy(buffer->symbol, temp_data->symbol);
        buffer->candle_count = temp_data->candle_count;
        
        // Copy candles (buffer->candles already allocated)
        memcpy(buffer->candles, temp_data->candles, 
               sizeof(IBKRCandle) * temp_data->candle_count);
        
        // Free temporary data
        free(temp_data->candles);
        free(temp_data);
        
        return 1;
    }
    
    // Otherwise download from IBKR and copy to buffer
    if(debug_mode) {
        printf("[DATA] Downloading %s from IBKR into buffer\n", symbol);
    }
    
    IBKRMarketData* temp_data = ibkr_download_historical_data(conn, symbol, 
                                                             candle_count, interval);
    if(!temp_data) {
        printf("ERROR: Failed to download data for %s\n", symbol);
        return 0;
    }
    
    // Copy to our global buffer
    strcpy(buffer->symbol, temp_data->symbol);
    buffer->candle_count = temp_data->candle_count;
    memcpy(buffer->candles, temp_data->candles, 
           sizeof(IBKRCandle) * temp_data->candle_count);
    
    // Save to cache
    if(!data_manager_save_to_cache(temp_data, cache_filename)) {
        printf("WARNING: Failed to save cache for %s\n", symbol);
    }
    
    // Free temporary data
    free(temp_data->candles);
    free(temp_data);
    
    return 1;
}

// ===== FONCTIONS NETTOYAGE =====

int data_manager_cleanup_cache(void) {
    if(debug_mode) {
        printf("[DATA] Cleaning up cache folder\n");
    }
    
    DIR* dir = opendir(DATA_FOLDER);
    if(!dir) {
        printf("ERROR: Cannot open %s folder\n", DATA_FOLDER);
        return 0;
    }
    
    struct dirent* entry;
    int deleted_count = 0;
    
    while((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        char filepath[MAX_FILENAME_LEN];
        snprintf(filepath, sizeof(filepath), "%s/%s", DATA_FOLDER, entry->d_name);
        
        if(remove(filepath) == 0) {
            deleted_count++;
            if(debug_mode) {
                printf("[DATA] Deleted: %s\n", entry->d_name);
            }
        }
    }
    
    closedir(dir);
    
    if(debug_mode) {
        printf("[DATA] Deleted %d cache files\n", deleted_count);
    }
    
    return deleted_count;
}

int data_manager_cleanup_old_cache(int days) {
    if(debug_mode) {
        printf("[DATA] Cleaning cache files older than %d days\n", days);
    }
    
    DIR* dir = opendir(DATA_FOLDER);
    if(!dir) return 0;
    
    // Utiliser l'heure EST
    struct tm tm_est;
    if(!data_manager_get_est_time(&tm_est)) {
        closedir(dir);
        return 0;
    }
    
    time_t now_est = mktime(&tm_est);
    time_t cutoff = now_est - (days * 24 * 60 * 60); // days en secondes EST
    
    struct dirent* entry;
    int deleted_count = 0;
    
    while((entry = readdir(dir)) != NULL) {
        if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        char filepath[MAX_FILENAME_LEN];
        snprintf(filepath, sizeof(filepath), "%s/%s", DATA_FOLDER, entry->d_name);
        
        struct stat file_stat;
        if(stat(filepath, &file_stat) == 0) {
            if(file_stat.st_mtime < cutoff) {
                if(remove(filepath) == 0) {
                    deleted_count++;
                    if(debug_mode) {
                        printf("[DATA] Deleted old file: %s\n", entry->d_name);
                    }
                }
            }
        }
    }
    
    closedir(dir);
    return deleted_count;
}

int data_manager_list_cache_files(CacheInfo* cache_list, int max_count) {
    if(!cache_list || max_count <= 0) return 0;
    
    if(debug_mode) {
        printf("[DATA] Listing cache files\\n");
    }
    
    DIR* dir = opendir(DATA_FOLDER);
    if(!dir) return 0;
    
    struct dirent* entry;
    int count = 0;
    
    while((entry = readdir(dir)) != NULL && count < max_count) {
        if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        // Check that it is a CSV file
        if(!strstr(entry->d_name, ".csv")) {
            continue;
        }
        
        char filepath[MAX_FILENAME_LEN];
        snprintf(filepath, sizeof(filepath), "%s/%s", DATA_FOLDER, entry->d_name);
        
        // Parse the filename: DATE_SYMBOL_COUNT_INTERVAL.csv
        char temp[MAX_FILENAME_LEN];
        strncpy(temp, entry->d_name, sizeof(temp));
        
        char* date = strtok(temp, "_");
        char* symbol = strtok(NULL, "_");
        char* count_str = strtok(NULL, "_");
        char* interval = strtok(NULL, ".");
        
        if(date && symbol && count_str && interval) {
            strncpy(cache_list[count].filename, entry->d_name, sizeof(cache_list[count].filename) - 1);
            cache_list[count].filename[sizeof(cache_list[count].filename) - 1] = '\0';
            
            strncpy(cache_list[count].date, date, sizeof(cache_list[count].date) - 1);
            cache_list[count].date[sizeof(cache_list[count].date) - 1] = '\0';
            
            strncpy(cache_list[count].symbol, symbol, sizeof(cache_list[count].symbol) - 1);
            cache_list[count].symbol[sizeof(cache_list[count].symbol) - 1] = '\0';
            
            cache_list[count].candle_count = atoi(count_str);
            
            snprintf(cache_list[count].interval, sizeof(cache_list[count].interval), "%s", interval);
            
            // Get creation timestamp
            struct stat file_stat;
            if(stat(filepath, &file_stat) == 0) {
                cache_list[count].created_time = file_stat.st_mtime;
            } else {
                cache_list[count].created_time = 0;
            }
            
            count++;
        }
    }
    
    closedir(dir);
    
    if(debug_mode) {
        printf("[DATA] Found %d cache files\\n", count);
    }
    
    return count;
}

// ===== FONCTIONS UTILITAIRES =====

int data_manager_get_current_date(char* date_buffer) {
    if(!date_buffer) return 0;
    
    // Utiliser l'heure EST
    struct tm tm_est;
    if(!data_manager_get_est_time(&tm_est)) {
        return 0;
    }
    
    strftime(date_buffer, MAX_DATE_LEN, "%Y-%m-%d", &tm_est);
    
    if(debug_mode) {
        printf("[DATA] Current date (EST): %s\n", date_buffer);
    }
    
    return 1;
}

int data_manager_folder_exists(const char* folder_path) {
    if(!folder_path) return 0;
    
    struct stat st;
    return (stat(folder_path, &st) == 0 && S_ISDIR(st.st_mode));
}

int data_manager_create_folder(const char* folder_path) {
    if(!folder_path) return 0;
    
    if(mkdir(folder_path, 0755) == 0) {
        return 1;
    }
    
    // Error but maybe the folder already exists
    return data_manager_folder_exists(folder_path);
}

long data_manager_get_file_size(const char* filename) {
    if(!filename) return -1;
    
    struct stat st;
    if(stat(filename, &st) == 0) {
        return st.st_size;
    }
    
    return -1;
}

void data_manager_set_debug_mode(int enable) {
    debug_mode = enable;
    if(debug_mode) {
        printf("[DATA] Debug mode enabled\n");
    }
}

int data_manager_get_est_time(struct tm* tm_est) {

    if(!tm_est)
    	return 0;
    
    time_t now = time(NULL);
    time_t est_timestamp = now - (5 * 60 * 60); // UTC-5 for EST
    
    struct tm* est_time = gmtime(&est_timestamp);
    if(!est_time)
    	return 0;
    
    *tm_est = *est_time;
    return 1;
}

int data_manager_is_market_hours(void) {
    struct tm tm_est;
    if(!data_manager_get_est_time(&tm_est)) {
        return 0;
    }
    
    /* Check day of the week (0=Sunday, 1=Monday, ..., 6=Saturday) */

    if(tm_est.tm_wday == 0 || tm_est.tm_wday == 6)
        return 0; /* Weekend */
    
    // Convert to minutes since midnight
    int current_minutes = tm_est.tm_hour * 60 + tm_est.tm_min;
    int market_open = 9 * 60 + 30;  // 9h30
    int market_close = 16 * 60;     // 16h00
    
    return (current_minutes >= market_open && current_minutes <= market_close);
}
