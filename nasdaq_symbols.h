#ifndef NASDAQ_SYMBOLS_H
#define NASDAQ_SYMBOLS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SYMBOL_LEN 16
#define NASDAQ100_COUNT 100
#define CANDLE_COUNT 200

// ===== PROTOTYPES =====

/**
 * Remplit a tableau with the symbols NASDAQ 100
 * @param symbols: Tableau 2D for stocker the symbols
 * @param max_symbols: Taille maximum of the tableau
 * @return: Nombre of symboles copiés
 */
int get_nasdaq100_symbols(char symbols[][MAX_SYMBOL_LEN], int max_symbols);

/**
 * Affiche all the symbols NASDAQ 100 formatés
 */
void print_nasdaq100_symbols(void);

/**
 * Vérifie if a symbole fait partie of the NASDAQ 100
 * @param symbol: Symbole to vérifier
 * @return: 1 if trouvé, 0 sinon
 */
int is_nasdaq100_symbol(const char* symbol);

/**
 * Retourne the nombre total of symboles NASDAQ 100
 * @return: Nombre of symboles
 */
int get_nasdaq100_count(void);

#endif // NASDAQ_SYMBOLS_H