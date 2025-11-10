#!/usr/bin/env python3
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import yfinance as yf
import pandas as pd
from zoneinfo import ZoneInfo
from ib_insync import IB, Stock
import time
import argparse
import logging
import sys


class StockScanner:
    # Constantes pour le calcul des périodes
    DAILY_MARGIN = 1.6
    HOURLY_HOURS_PER_DAY = 6.5
    WEEKLY_DAYS = 7
    DEFAULT_MARGIN = 2
    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY = 5  # secondes

    def __init__(self, settings_file: str, is_backtest: bool = False):
        """
        Initialise le scanner de stocks

        Args:
            settings_file: Chemin vers le fichier de configuration JSON
            is_backtest: True pour le mode backtest, False pour le mode temps réel

        Raises:
            FileNotFoundError: Si le fichier de configuration n'existe pas
            json.JSONDecodeError: Si le fichier JSON est mal formaté
            KeyError: Si des clés requises sont manquantes dans la configuration
        """
        # Configuration du logging
        self.logger = logging.getLogger(__name__)

        try:
            with open(settings_file, 'r') as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Fichier de configuration non trouvé: {settings_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Erreur de parsing JSON dans {settings_file}: {e}")
            raise

        # Validation des clés requises
        required_keys = ['data_folder', 'config_file']
        missing_keys = [key for key in required_keys if key not in self.settings]
        if missing_keys:
            raise KeyError(f"Clés manquantes dans la configuration: {missing_keys}")

        self.mode = 'backtest' if is_backtest else 'realtime'
        self.data_folder = self.settings['data_folder']
        self.config_file = self.settings['config_file']
        self.timezone = ZoneInfo('America/New_York')

    def load_watchlist(self) -> List[Dict[str, str]]:
        """
        Charge les symboles depuis le fichier de configuration

        Returns:
            Liste de dictionnaires contenant 'symbol' et 'provider'

        Raises:
            FileNotFoundError: Si le fichier de configuration n'existe pas
        """
        symbols = []
        try:
            with open(self.config_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            symbol = parts[0]
                            provider = parts[1]
                            symbols.append({'symbol': symbol, 'provider': provider})
                        else:
                            self.logger.warning(
                                f"Ligne {line_num} mal formatée dans {self.config_file}: '{line}'"
                            )
        except FileNotFoundError:
            self.logger.error(f"Fichier de watchlist non trouvé: {self.config_file}")
            raise

        self.logger.info(f"Chargé {len(symbols)} symboles depuis {self.config_file}")
        return symbols

    def get_data_filename(self, symbol: str, candle_nb: int, interval: str, date: str) -> str:
        """
        Génère le nom du fichier de données

        Args:
            symbol: Symbole du titre
            candle_nb: Nombre de bougies
            interval: Intervalle (1d, 1h, etc.)
            date: Date au format YYYY-MM-DD

        Returns:
            Chemin complet du fichier
        """
        return os.path.join(
            self.data_folder,
            f"{date}_{symbol}_{candle_nb}_{interval}.csv"
        )

    def check_file_exists(self, filepath: str) -> bool:
        """
        Vérifie si le fichier existe

        Args:
            filepath: Chemin du fichier à vérifier

        Returns:
            True si le fichier existe, False sinon
        """
        return os.path.exists(filepath)

    def download_yahoo_data(self, symbol: str, candle_nb: int, interval: str) -> Optional[pd.DataFrame]:
        """
        Télécharge les données depuis Yahoo Finance

        Args:
            symbol: Symbole du titre
            candle_nb: Nombre de bougies à récupérer
            interval: Intervalle (1d, 1h, 1wk, etc.)

        Returns:
            DataFrame avec les données ou None en cas d'erreur
        """
        try:
            self.logger.info(f"Téléchargement des données pour {symbol}...")
            ticker = yf.Ticker(symbol)

            # Calculer la période nécessaire avec marge de sécurité
            if interval == '1d':
                days_needed = int(candle_nb * self.DAILY_MARGIN)
            elif interval == '1h':
                days_needed = int(candle_nb / self.HOURLY_HOURS_PER_DAY) + 1
            elif interval == '1wk':
                days_needed = candle_nb * self.WEEKLY_DAYS * 2
            else:
                days_needed = candle_nb * self.DEFAULT_MARGIN

            end_date = datetime.now(self.timezone)
            start_date = end_date - timedelta(days=days_needed)

            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                self.logger.warning(f"Aucune donnée disponible pour {symbol}")
                return None

            # Prendre les N dernières bougies
            df = df.tail(candle_nb)

            # Validation: vérifier qu'on a au moins quelques bougies
            if len(df) < candle_nb * 0.5:  # Au moins 50% des bougies demandées
                self.logger.warning(
                    f"Nombre de bougies insuffisant pour {symbol}: "
                    f"{len(df)}/{candle_nb} récupérées"
                )

            # Validation: vérifier les colonnes requises
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Colonnes manquantes pour {symbol}: {missing_columns}")
                return None

            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

            self.logger.info(f"Téléchargement réussi: {len(df)} bougies pour {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Erreur lors du téléchargement de {symbol}: {type(e).__name__}: {e}")
            return None

    def run_backtest(self) -> None:
        """
        Execute le mode backtest

        Télécharge les données historiques pour tous les symboles de la watchlist
        et les sauvegarde en CSV.
        """
        try:
            backtest_config = self.settings['backtest']
            candle_nb = backtest_config['candle_nb']
            interval = backtest_config['interval']
        except KeyError as e:
            self.logger.error(f"Configuration backtest manquante: {e}")
            raise

        # Créer le dossier data s'il n'existe pas
        os.makedirs(self.data_folder, exist_ok=True)

        watchlist = self.load_watchlist()
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Nombre de bougies: {candle_nb}")
        self.logger.info(f"Interval: {interval}")
        self.logger.info(f"Nombre de symboles: {len(watchlist)}")

        today = datetime.now(self.timezone).strftime('%Y-%m-%d')

        success_count = 0
        skip_count = 0
        error_count = 0

        for item in watchlist:
            symbol = item['symbol']
            filename = self.get_data_filename(symbol, candle_nb, interval, today)

            if self.check_file_exists(filename):
                self.logger.info(f"Skip {symbol} - fichier déjà existant")
                skip_count += 1
                continue

            df = self.download_yahoo_data(symbol, candle_nb, interval)
            if df is not None:
                try:
                    df.to_csv(filename, index=False)
                    self.logger.info(f"Données sauvegardées: {filename}")
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"Erreur sauvegarde {symbol}: {e}")
                    error_count += 1
            else:
                error_count += 1

        self.logger.info(
            f"\nRésumé: {success_count} succès, {skip_count} skippés, {error_count} erreurs"
        )

    def connect_ibkr(self) -> Optional[IB]:
        """
        Connecte à Interactive Brokers

        Returns:
            Instance IB connectée ou None en cas d'erreur

        Raises:
            KeyError: Si la configuration IBKR est manquante
        """
        try:
            realtime_config = self.settings['realtime']
            host = realtime_config['ibkr_host']
            port = realtime_config['ibkr_port']
            client_id = realtime_config['ibkr_client_id']
        except KeyError as e:
            self.logger.error(f"Configuration IBKR manquante: {e}")
            raise

        for attempt in range(self.MAX_RECONNECT_ATTEMPTS):
            try:
                ib = IB()
                ib.connect(host, port, clientId=client_id)
                self.logger.info(f"Connecté à IBKR {host}:{port} (tentative {attempt + 1})")
                return ib
            except Exception as e:
                self.logger.warning(
                    f"Tentative {attempt + 1}/{self.MAX_RECONNECT_ATTEMPTS} échouée: {e}"
                )
                if attempt < self.MAX_RECONNECT_ATTEMPTS - 1:
                    self.logger.info(f"Nouvelle tentative dans {self.RECONNECT_DELAY}s...")
                    time.sleep(self.RECONNECT_DELAY)

        self.logger.error("Impossible de se connecter à IBKR après plusieurs tentatives")
        return None

    def get_last_close_ibkr(self, ib: IB, symbol: str) -> Optional[float]:
        """
        Récupère le close de la dernière bougie depuis IBKR

        Args:
            ib: Instance IB connectée
            symbol: Symbole du titre

        Returns:
            Prix de clôture ou None en cas d'erreur
        """
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            qualified = ib.qualifyContracts(contract)

            if not qualified:
                self.logger.warning(f"Contrat non trouvé pour {symbol}")
                return None

            contract = qualified[0]

            # Demande la dernière bougie daily
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='2 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                timeout=5
            )

            if not bars:
                self.logger.warning(f"Aucune donnée pour {symbol}")
                return None

            # Dernière bougie close
            last_close = bars[-1].close
            return last_close

        except Exception as e:
            self.logger.error(f"Erreur récupération {symbol}: {type(e).__name__}: {e}")
            return None

    def run_realtime(self) -> None:
        """
        Execute le mode temps réel

        Scanne les symboles en continu et affiche les derniers prix
        """
        try:
            realtime_config = self.settings['realtime']
            update_interval = realtime_config['update_interval_seconds']
        except KeyError as e:
            self.logger.error(f"Configuration realtime manquante: {e}")
            raise

        watchlist = self.load_watchlist()
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Interval de mise à jour: {update_interval}s")
        self.logger.info(f"Nombre de symboles: {len(watchlist)}")

        # Connexion IBKR
        ib = self.connect_ibkr()
        if not ib:
            self.logger.error("Impossible de se connecter à IBKR. Arrêt.")
            return

        try:
            while True:
                # Vérifier la connexion et reconnecter si nécessaire
                if not ib.isConnected():
                    self.logger.warning("Connexion IBKR perdue, reconnexion...")
                    ib = self.connect_ibkr()
                    if not ib:
                        self.logger.error("Reconnexion échouée. Arrêt.")
                        break

                timestamp = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S')
                self.logger.info(f"\n[{timestamp}] Scan en cours...")

                for item in watchlist:
                    symbol = item['symbol']
                    last_close = self.get_last_close_ibkr(ib, symbol)

                    if last_close:
                        self.logger.info(f"{symbol}: ${last_close:.2f}")

                self.logger.info(f"\nProchaine mise à jour dans {update_interval}s...")
                time.sleep(update_interval)

        except KeyboardInterrupt:
            self.logger.info("\nArrêt du scanner temps réel")
        finally:
            if ib and ib.isConnected():
                ib.disconnect()
                self.logger.info("Déconnecté d'IBKR")

    def run(self) -> None:
        """
        Point d'entrée principal

        Execute le mode approprié selon la configuration
        """
        if self.mode == 'backtest':
            self.run_backtest()
        elif self.mode == 'realtime':
            self.run_realtime()
        else:
            self.logger.error(f"Mode inconnu: {self.mode}")
            raise ValueError(f"Mode inconnu: {self.mode}")


def setup_logging(verbose: bool = False) -> None:
    """
    Configure le système de logging

    Args:
        verbose: Si True, affiche les logs de niveau DEBUG
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Format des logs
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Configuration du handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Configuration du logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Réduire la verbosité des librairies tierces
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('ib_insync').setLevel(logging.WARNING)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scanner de stocks')
    parser.add_argument('--backtest', action='store_true', help='Lance en mode backtest')
    parser.add_argument('--verbose', '-v', action='store_true', help='Active les logs détaillés')
    parser.add_argument('--settings', default='settings.json', help='Fichier de configuration (défaut: settings.json)')
    args = parser.parse_args()

    # Configuration du logging
    setup_logging(verbose=args.verbose)

    try:
        scanner = StockScanner(args.settings, is_backtest=args.backtest)
        scanner.run()
    except Exception as e:
        logging.error(f"Erreur fatale: {e}", exc_info=True)
        sys.exit(1)
