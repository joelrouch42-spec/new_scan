#ifndef IBKR_CONNECTOR_H
#define IBKR_CONNECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SYMBOL_LEN 16
#define MAX_CANDLES 500
#define MAX_ERROR_MSG 256

// ===== STRUCTURES =====

typedef struct {
    char date[32];
    double open;
    double high;
    double low;
    double close;
    long volume;
} IBKRCandle;

typedef struct {
    char symbol[MAX_SYMBOL_LEN];
    IBKRCandle* candles;
    int candle_count;
} IBKRMarketData;

typedef struct {
    char symbol[MAX_SYMBOL_LEN];
    int shares;
    double avg_cost;
    double market_price;
    double market_value;
    double unrealized_pnl;
} IBKRPosition;

typedef struct {
    char host[64];
    int port;
    int client_id;
    int connected;
    char error_msg[MAX_ERROR_MSG];
} IBKRConnection;

// ===== PROTOTYPES CONNEXION =====

/**
 * Initialise the connexion IBKR
 * @param host: IP of the Gateway IBKR (ex: "127.0.0.1")
 * @param port: Port of the Gateway (ex: 4002)
 * @param client_id: ID client unique
 * @return: Pointeur to structure of connexion or NULL if erreur
 */
IBKRConnection* ibkr_init(const char* host, int port, int client_id);

/**
 * Se connecte au Gateway IBKR
 * @param conn: Structure of connexion
 * @return: 1 if succès, 0 if erreur
 */
int ibkr_connect(IBKRConnection* conn);

/**
 * Se déconnecte of the Gateway IBKR
 * @param conn: Structure of connexion
 */
void ibkr_disconnect(IBKRConnection* conn);

/**
 * Vérifie if the connexion est active
 * @param conn: Structure of connexion
 * @return: 1 if connecté, 0 sinon
 */
int ibkr_is_connected(IBKRConnection* conn);

/**
 * Libère the ressources of connexion
 * @param conn: Structure of connexion
 */
void ibkr_free_connection(IBKRConnection* conn);

// ===== PROTOTYPES DONNÉES MARCHÉ =====

/**
 * Télécharge the données historiques for a symbole
 * @param conn: Connexion IBKR active
 * @param symbol: Symbole to télécharger (ex: "AAPL")
 * @param candle_count: Nombre of bougies (ex: 200)
 * @param interval: Intervalle (ex: "1d", "1h", "30m")
 * @return: Pointeur to données or NULL if erreur
 */
IBKRMarketData* ibkr_download_historical_data(IBKRConnection* conn, 
                                             const char* symbol, 
                                             int candle_count,
                                             const char* interval);

/**
 * Obtient the prix actuel d'a symbole
 * @param conn: Connexion IBKR active
 * @param symbol: Symbole (ex: "AAPL")
 * @return: Prix actuel or -1.0 if erreur
 */
double ibkr_get_current_price(IBKRConnection* conn, const char* symbol);

/**
 * Libère the données of marché
 * @param data: Données to libérer
 */
void ibkr_free_market_data(IBKRMarketData* data);

// ===== PROTOTYPES TRADING =====

/**
 * Passe a ordre of marché
 * @param conn: Connexion IBKR active
 * @param symbol: Symbole (ex: "AAPL")
 * @param action: "BUY" or "SELL"
 * @param quantity: Nombre d'actions
 * @return: Order ID or -1 if erreur
 */
int ibkr_place_market_order(IBKRConnection* conn, 
                           const char* symbol,
                           const char* action, 
                           int quantity);

/**
 * Passe a ordre limite
 * @param conn: Connexion IBKR active
 * @param symbol: Symbole
 * @param action: "BUY" or "SELL"
 * @param quantity: Nombre d'actions
 * @param limit_price: Prix limite
 * @return: Order ID or -1 if erreur
 */
int ibkr_place_limit_order(IBKRConnection* conn,
                          const char* symbol,
                          const char* action,
                          int quantity,
                          double limit_price);

// ===== PROTOTYPES COMPTE =====

/**
 * Obtient l'equity of the compte
 * @param conn: Connexion IBKR active
 * @return: Equity en USD or -1.0 if erreur
 */
double ibkr_get_account_equity(IBKRConnection* conn);

/**
 * Obtient the positions actuelles
 * @param conn: Connexion IBKR active
 * @param position_count: Pointeur for retourner the nombre of positions
 * @return: Tableau of positions or NULL if erreur
 */
IBKRPosition* ibkr_get_positions(IBKRConnection* conn, int* position_count);

/**
 * Obtient the taille of position for a symbole
 * @param conn: Connexion IBKR active
 * @param symbol: Symbole
 * @return: Nombre d'actions (+ for long, - for short, 0 for aucune)
 */
int ibkr_get_position_size(IBKRConnection* conn, const char* symbol);

/**
 * Libère the tableau of positions
 * @param positions: Positions to libérer
 */
void ibkr_free_positions(IBKRPosition* positions);

// ===== PROTOTYPES UTILITAIRES =====

/**
 * Obtient the dernier message d'erreur
 * @param conn: Connexion IBKR
 * @return: Message d'erreur or chaîne vide
 */
const char* ibkr_get_last_error(IBKRConnection* conn);

/**
 * Active/désactive the mode debug
 * @param enable: 1 for activer, 0 for désactiver
 */
void ibkr_set_debug_mode(int enable);

#endif // IBKR_CONNECTOR_H