#include "nasdaq_symbols.h"

// Liste hardcodée des 100 symbols NASDAQ
static const char* nasdaq100_list[] = {
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "AMAT", "AMD", "AMGN", "AMZN", "ANSS",
    "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR", "CCEP", "CDNS", "CEG", "CHTR",
    "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DDOG",
    "DLTR", "DXCM", "EA", "EBAY", "ENPH", "EXC", "FANG", "FAST", "FTNT", "GILD",
    "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC", "INTU", "ISRG", "JD", "KDP",
    "KHC", "KLAC", "LCID", "LRCX", "LULU", "MAR", "MCHP", "MDLZ", "MELI", "META",
    "MNST", "MRNA", "MRVL", "MSFT", "MU", "NFLX", "NVDA", "NXPI", "ODFL", "ON",
    "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PYPL", "QCOM", "REGN", "RIVN",
    "ROST", "SBUX", "SGEN", "SIRI", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO",
    "TXN", "VRSK", "VRTX", "WBA", "WBD", "WDAY", "XEL", "ZM", "ZS", NULL
};

int get_nasdaq100_symbols(char symbols[][MAX_SYMBOL_LEN], int max_symbols) {
    int count = 0;
    
    // Copier the symbols in the tableau fourni
    for(int i = 0; nasdaq100_list[i] != NULL && count < max_symbols; i++) {
        strncpy(symbols[count], nasdaq100_list[i], MAX_SYMBOL_LEN - 1);
        symbols[count][MAX_SYMBOL_LEN - 1] = '\0'; // Assurer null termination
        count++;
    }
    
    return count;
}

void print_nasdaq100_symbols(void) {
    printf("NASDAQ 100 Symbols:\n");
    printf("==================\n");
    
    int count = 0;
    for(int i = 0; nasdaq100_list[i] != NULL; i++) {
        printf("%-8s", nasdaq100_list[i]);
        count++;
        
        // New line every 10 symbols
        if(count % 10 == 0) {
            printf("\n");
        }
    }
    
    if(count % 10 != 0) {
        printf("\n");
    }
    
    printf("\nTotal: %d symbols\n", count);
}

int is_nasdaq100_symbol(const char* symbol) {
    if(!symbol) return 0;
    
    for(int i = 0; nasdaq100_list[i] != NULL; i++) {
        if(strcmp(nasdaq100_list[i], symbol) == 0) {
            return 1; // Trouvé
        }
    }
    
    return 0; // Pas trouvé
}

int get_nasdaq100_count(void) {
    int count = 0;
    while(nasdaq100_list[count] != NULL) {
        count++;
    }
    return count;
}