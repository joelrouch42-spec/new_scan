#include "settings_parser.h"

int main() {
    AppSettings settings;
    
    // Activer the mode debug for voir the parsing
    settings_set_debug_mode(1);
    
    // Charger the settings
    if(!settings_load("settings.json", &settings)) {
        printf("ERREUR: Impossible of charger settings.json\n");
        return 1;
    }
    
    // Afficher the settings chargés
    settings_print(&settings);
    
    printf("\n=== TEST VALIDATIONS ===\n");
    printf("Data folder: %s\n", settings.global.data_folder);
    printf("IBKR Host: %s:%d (client_id=%d)\n", 
           settings.realtime.ibkr_host, 
           settings.realtime.ibkr_port,
           settings.realtime.ibkr_client_id);
    printf("Candles: %d (%s)\n", 
           settings.realtime.candle_nb,
           settings.realtime.interval);
    printf("Squeeze Momentum: %s (BB: %d/%.1f, KC: %d/%.1f)\n",
           settings.indicators.squeeze_momentum.enabled ? "ENABLED" : "DISABLED",
           settings.indicators.squeeze_momentum.bb_length,
           settings.indicators.squeeze_momentum.bb_mult,
           settings.indicators.squeeze_momentum.kc_length,
           settings.indicators.squeeze_momentum.kc_mult);
    
    return 0;
}