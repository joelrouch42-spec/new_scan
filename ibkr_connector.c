#include "ibkr_connector.h"

// Global variables for debug
static int debug_mode = 0;

// ===== FONCTIONS CONNEXION =====

IBKRConnection* ibkr_init(const char* host, int port, int client_id) {
    if(!host) return NULL;
    
    IBKRConnection* conn = malloc(sizeof(IBKRConnection));
    if(!conn) return NULL;
    
    strncpy(conn->host, host, sizeof(conn->host) - 1);
    conn->host[sizeof(conn->host) - 1] = '\0';
    conn->port = port;
    conn->client_id = client_id;
    conn->connected = 0;
    conn->error_msg[0] = '\0';
    
    if(debug_mode) {
        printf("[IBKR] Initialized connection: %s:%d (client_id=%d)\n", 
               host, port, client_id);
    }
    
    return conn;
}

int ibkr_connect(IBKRConnection* conn) {
    if(!conn) return 0;
    
    if(debug_mode) {
        printf("[IBKR] Connecting to %s:%d...\n", conn->host, conn->port);
    }
    
    // TODO: Implémentation vraie connexion IBKR
    // Pour l'instant: STUB
    printf("[STUB] Connecting to IBKR Gateway %s:%d (client_id=%d)\n", 
           conn->host, conn->port, conn->client_id);
    
    conn->connected = 1;
    return 1;
}

void ibkr_disconnect(IBKRConnection* conn) {
    if(!conn || !conn->connected) return;
    
    if(debug_mode) {
        printf("[IBKR] Disconnecting from %s:%d\n", conn->host, conn->port);
    }
    
    // TODO: Implémentation vraie disconnect
    printf("[STUB] Disconnecting from IBKR\n");
    
    conn->connected = 0;
}

int ibkr_is_connected(IBKRConnection* conn) {
    return conn ? conn->connected : 0;
}

void ibkr_free_connection(IBKRConnection* conn) {
    if(conn) {
        if(conn->connected) {
            ibkr_disconnect(conn);
        }
        free(conn);
    }
}

// ===== FONCTIONS DONNÉES MARCHÉ =====

IBKRMarketData* ibkr_download_historical_data(IBKRConnection* conn, 
                                             const char* symbol, 
                                             int candle_count,
                                             const char* interval) {
    if(!conn || !conn->connected || !symbol) return NULL;
    
    if(debug_mode) {
        printf("[IBKR] Downloading %d candles of %s (%s)\n", 
               candle_count, symbol, interval);
    }
    
    IBKRMarketData* data = malloc(sizeof(IBKRMarketData));
    if(!data) return NULL;
    
    strncpy(data->symbol, symbol, sizeof(data->symbol) - 1);
    data->symbol[sizeof(data->symbol) - 1] = '\0';
    data->candle_count = candle_count;
    data->candles = malloc(sizeof(IBKRCandle) * candle_count);
    
    if(!data->candles) {
        free(data);
        return NULL;
    }
    
    // Générer data fictives for test
    for(int i = 0; i < candle_count; i++) {
        snprintf(data->candles[i].date, sizeof(data->candles[i].date), 
                "2025-12-%02d", (i % 30) + 1);
        data->candles[i].open = 100.0 + i * 0.5;
        data->candles[i].high = 102.0 + i * 0.5;
        data->candles[i].low = 98.0 + i * 0.5;
        data->candles[i].close = 101.0 + i * 0.5;
        data->candles[i].volume = 1000000 + (i * 10000);
    }
    
    return data;
}

double ibkr_get_current_price(IBKRConnection* conn, const char* symbol) {
    if(!conn || !conn->connected || !symbol) return -1.0;
    
    if(debug_mode) {
        printf("[IBKR] Getting current price for %s\n", symbol);
    }
    
    // TODO: Vraie implémentation
    printf("[STUB] Getting current price for %s\n", symbol);
    
    // Prix fictif basé on the hash of the symbol
    double price = 100.0 + (strlen(symbol) * 10.5);
    return price;
}

void ibkr_free_market_data(IBKRMarketData* data) {
    if(data) {
        free(data->candles);
        free(data);
    }
}

// ===== FONCTIONS TRADING =====

int ibkr_place_market_order(IBKRConnection* conn, 
                           const char* symbol,
                           const char* action, 
                           int quantity) {
    if(!conn || !conn->connected || !symbol || !action) return -1;
    
    if(debug_mode) {
        printf("[IBKR] Placing market order: %s %d %s\n", 
               action, quantity, symbol);
    }
    
    // TODO: Vraie implémentation
    printf("[STUB] Market order: %s %d shares of %s\n", 
           action, quantity, symbol);
    
    // Retourner order ID fictif
    static int next_order_id = 1000;
    return next_order_id++;
}

int ibkr_place_limit_order(IBKRConnection* conn,
                          const char* symbol,
                          const char* action,
                          int quantity,
                          double limit_price) {
    if(!conn || !conn->connected || !symbol || !action) return -1;
    
    if(debug_mode) {
        printf("[IBKR] Placing limit order: %s %d %s @ %.2f\n", 
               action, quantity, symbol, limit_price);
    }
    
    // TODO: Vraie implémentation  
    printf("[STUB] Limit order: %s %d shares of %s @ $%.2f\n", 
           action, quantity, symbol, limit_price);
    
    static int next_order_id = 1000;
    return next_order_id++;
}

// ===== FONCTIONS COMPTE =====

double ibkr_get_account_equity(IBKRConnection* conn) {
    if(!conn || !conn->connected) return -1.0;
    
    if(debug_mode) {
        printf("[IBKR] Getting account equity\n");
    }
    
    // TODO: Vraie implémentation
    printf("[STUB] Getting account equity\n");
    
    return 50000.0; // Equity fictive
}

IBKRPosition* ibkr_get_positions(IBKRConnection* conn, int* position_count) {
    if(!conn || !conn->connected || !position_count) return NULL;
    
    if(debug_mode) {
        printf("[IBKR] Getting current positions\n");
    }
    
    // TODO: Vraie implémentation
    printf("[STUB] Getting positions\n");
    
    // Aucune position for l'instant
    *position_count = 0;
    return NULL;
}

int ibkr_get_position_size(IBKRConnection* conn, const char* symbol) {
    if(!conn || !conn->connected || !symbol) return 0;
    
    if(debug_mode) {
        printf("[IBKR] Getting position size for %s\n", symbol);
    }
    
    // TODO: Vraie implémentation
    printf("[STUB] Getting position size for %s\n", symbol);
    
    return 0; // Aucune position
}

void ibkr_free_positions(IBKRPosition* positions) {
    if(positions) {
        free(positions);
    }
}

// ===== FONCTIONS UTILITAIRES =====

const char* ibkr_get_last_error(IBKRConnection* conn) {
    return conn ? conn->error_msg : "Invalid connection";
}

void ibkr_set_debug_mode(int enable) {
    debug_mode = enable;
    if(debug_mode) {
        printf("[IBKR] Debug mode enabled\n");
    }
}
