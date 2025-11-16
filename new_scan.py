#!/usr/bin/env python3
import json
import os
from datetime import datetime
import pandas as pd
from zoneinfo import ZoneInfo
from ib_insync import IB, Stock
import argparse
import plotly.graph_objects as go
from smc_analyzer import SMCAnalyzer
import logging

# Désactiver les logs ib_insync
logging.getLogger('ib_insync').setLevel(logging.CRITICAL)

class StockScanner:
    def __init__(self, settings_file, is_backtest=False, patterns_file='patterns.json', chart_symbol=None):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        with open(patterns_file, 'r') as f:
            self.patterns_config = json.load(f)

        # Initialiser le SMC Analyzer
        smc_config = self.patterns_config['smc']
        self.smc_analyzer = SMCAnalyzer(smc_config)

        self.mode = 'backtest' if is_backtest else 'realtime'
        self.data_folder = self.settings['data_folder']
        self.config_file = 'config.txt'

        # Dossiers différents selon le mode
        if self.mode == 'backtest':
            self.patterns_folder = "patterns_backtest"
        else:
            self.patterns_folder = "patterns_realtime"

        self.chart_symbol = chart_symbol


    def get_data_filename(self, symbol, candle_nb, interval, date):
        """Génère le nom du fichier de données"""
        return os.path.join(
            self.data_folder,
            f"{date}_{symbol}_{candle_nb}_{interval}.csv"
        )

    def check_file_exists(self, filepath):
        """Vérifie si le fichier existe"""
        return os.path.exists(filepath)

    def get_nasdaq100_symbols(self):
        """Retourne la liste des symboles NASDAQ 100"""
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST',
            'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'CMCSA', 'INTC', 'TXN', 'QCOM',
            'INTU', 'AMGN', 'HON', 'AMAT', 'SBUX', 'ISRG', 'BKNG', 'VRTX', 'GILD', 'ADI',
            'ADP', 'MDLZ', 'REGN', 'PANW', 'LRCX', 'MU', 'PYPL', 'KLAC', 'SNPS', 'CDNS',
            'MELI', 'ASML', 'MAR', 'ABNB', 'CRWD', 'CSX', 'MRVL', 'FTNT', 'ORLY', 'MNST',
            'NXPI', 'ADSK', 'DASH', 'WDAY', 'ROP', 'CHTR', 'PCAR', 'AEP', 'CPRT', 'ROST',
            'PAYX', 'KDP', 'FAST', 'ODFL', 'BKR', 'EA', 'CTSH', 'VRSK', 'DXCM', 'AZN',
            'KHC', 'GEHC', 'LULU', 'IDXX', 'EXC', 'CSGP', 'XEL', 'ON', 'TTWO', 'ANSS',
            'FANG', 'BIIB', 'ZS', 'DDOG', 'CDW', 'GFS', 'ILMN', 'WBD', 'MRNA', 'MDB',
            'TEAM', 'ALGN', 'CTAS', 'DLTR', 'SMCI', 'ARM', 'COIN', 'APP', 'HOOD', 'RIVN'
        ]
        return [{'symbol': s, 'provider': 'IBKR'} for s in symbols]

    def load_watchlist(self):
        """Charge les symboles depuis le fichier de configuration"""
        symbols = []
        with open(self.config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        symbol = parts[0]
                        provider = parts[1]
                        symbols.append({'symbol': symbol, 'provider': provider})
        return symbols


    def download_ibkr_data(self, symbol, candle_nb, interval):
        """Télécharge les données depuis IBKR"""
        try:
            # Connexion IBKR
            realtime_config = self.settings['realtime']
            host = realtime_config['ibkr_host']
            port = realtime_config['ibkr_port']
            client_id = realtime_config['ibkr_client_id']

            ib = IB()
            ib.connect(host, port, clientId=client_id)

            # Créer le contrat
            contract = Stock(symbol, 'SMART', 'USD')
            qualified = ib.qualifyContracts(contract)

            if not qualified:
                ib.disconnect()
                return None

            contract = qualified[0]

            # Calculer la durée
            if interval == '1d':
                duration_str = f"{candle_nb} D"
                bar_size = "1 day"
            elif interval == '1h':
                duration_str = f"{candle_nb} S"
                bar_size = "1 hour"
            else:
                duration_str = f"{candle_nb} D"
                bar_size = "1 day"

            # Télécharger les données
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                timeout=10
            )

            ib.disconnect()

            if not bars or len(bars) < candle_nb:
                return None

            # Convertir en DataFrame avec conversion timezone EST
            data = []
            est_tz = ZoneInfo('America/New_York')

            for bar in bars[-candle_nb:]:
                # Convertir la date en EST si elle a une timezone, sinon assumer qu'elle est déjà en EST
                if hasattr(bar.date, 'tzinfo') and bar.date.tzinfo is not None:
                    date_est = bar.date.astimezone(est_tz)
                else:
                    date_est = bar.date

                data.append({
                    'Date': date_est.strftime('%Y-%m-%d'),
                    'Open': bar.open,
                    'High': bar.high,
                    'Low': bar.low,
                    'Close': bar.close,
                    'Volume': bar.volume
                })

            df = pd.DataFrame(data)
            return df

        except Exception as e:
            # Silencieux pour les erreurs IBKR (connexion refusée, etc.)
            return None


    def generate_chart(self, symbol, df):
        """Génère un graphique HTML avec les Order Blocks"""
        # Créer le dossier chart s'il n'existe pas
        chart_folder = 'chart'
        os.makedirs(chart_folder, exist_ok=True)

        # Analyse SMC
        smc_result = self.smc_analyzer.analyze(df)
        bullish_obs = smc_result['order_blocks']['bullish']
        bearish_obs = smc_result['order_blocks']['bearish']

        # Créer le graphique de chandelles
        fig = go.Figure()

        # Ajouter les chandelles
        fig.add_trace(go.Candlestick(
            x=df['Date'] if 'Date' in df.columns else list(range(len(df))),
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))

        # Ajouter les zones BLEUES (Bullish Order Blocks)
        for ob in bullish_obs:
            idx = ob['index']
            date = df.iloc[idx]['Date'] if 'Date' in df.columns else idx

            # Zone rectangulaire qui s'étend jusqu'à la fin
            fig.add_shape(
                type="rect",
                x0=date,
                x1=df.iloc[-1]['Date'] if 'Date' in df.columns else len(df) - 1,
                y0=ob['low'],
                y1=ob['high'],
                fillcolor="blue",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

        # Ajouter les zones ROUGES (Bearish Order Blocks)
        for ob in bearish_obs:
            idx = ob['index']
            date = df.iloc[idx]['Date'] if 'Date' in df.columns else idx

            # Zone rectangulaire qui s'étend jusqu'à la fin
            fig.add_shape(
                type="rect",
                x0=date,
                x1=df.iloc[-1]['Date'] if 'Date' in df.columns else len(df) - 1,
                y0=ob['low'],
                y1=ob['high'],
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

        # Mise en forme
        fig.update_layout(
            title=f'{symbol} - Order Blocks (Bullish=Bleu, Bearish=Rouge)',
            yaxis_title='Prix',
            xaxis_title='Date',
            template='plotly_dark',
            height=800,
            xaxis_rangeslider_visible=False
        )

        # Sauvegarder dans le dossier chart
        filename = os.path.join(chart_folder, f'{symbol}_order_blocks.html')
        fig.write_html(filename)
        print(f"Graphique: {filename}")
        print(f"Bullish OB (bleu): {len(bullish_obs)}")
        print(f"Bearish OB (rouge): {len(bearish_obs)}")

        return filename


    def run_backtest(self):
        """Execute le mode backtest"""
        backtest_config = self.settings['backtest']
        candle_nb = backtest_config['candle_nb']
        interval = backtest_config['interval']
        test_start = backtest_config['test_candle_start']
        test_stop = backtest_config['test_candle_stop']

        # Calculer le nombre total de bougies à charger
        total_candles_needed = candle_nb + test_stop

        # Créer les dossiers data et patterns_backtest s'ils n'existent pas
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.patterns_folder, exist_ok=True)

        watchlist = self.load_watchlist()

        # Filtrer la watchlist si --chart est spécifié
        if self.chart_symbol:
            watchlist = [item for item in watchlist if item['symbol'].upper() == self.chart_symbol.upper()]
            if not watchlist:
                print(f"Erreur: Symbole {self.chart_symbol} non trouvé")
                return

        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

        for item in watchlist:
            symbol = item['symbol']
            filename = self.get_data_filename(symbol, total_candles_needed, interval, today)

            # Télécharge ou charge les données
            if self.check_file_exists(filename):
                df = pd.read_csv(filename)
            else:
                df = self.download_ibkr_data(symbol, total_candles_needed, interval)
                if df is not None:
                    df.to_csv(filename, index=False)
                else:
                    continue

            if df is None or len(df) == 0:
                continue

            # Générer le graphique
            self.generate_chart(symbol, df)


    def run_realtime(self):
        """Execute le mode realtime avec alertes sur NASDAQ 100"""
        realtime_config = self.settings['realtime']
        candle_nb = realtime_config['candle_nb']
        interval = realtime_config['interval']

        # Créer les dossiers
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.patterns_folder, exist_ok=True)

        # Scanne NASDAQ 100
        watchlist = self.get_nasdaq100_symbols()
        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

        for item in watchlist:
            symbol = item['symbol']
            filename = self.get_data_filename(symbol, candle_nb, interval, today)

            # Télécharge ou charge les données
            if self.check_file_exists(filename):
                df = pd.read_csv(filename)
            else:
                df = self.download_ibkr_data(symbol, candle_nb, interval)
                if df is not None:
                    df.to_csv(filename, index=False)
                else:
                    continue

            if df is None or len(df) == 0:
                continue

            # Analyse SMC
            smc_result = self.smc_analyzer.analyze(df)
            bullish_obs = smc_result['order_blocks']['bullish']
            bearish_obs = smc_result['order_blocks']['bearish']

            # Prix actuel
            current_price = df.iloc[-1]['Close']

            # Vérifier si le prix touche une zone bleue (Bullish OB)
            for ob in bullish_obs:
                if ob['low'] <= current_price <= ob['high']:
                    print(f"🔵 {symbol} @ ${current_price:.2f} touche zone BLEUE [{ob['low']:.2f}-{ob['high']:.2f}]")
                    break

            # Vérifier si le prix touche une zone rouge (Bearish OB)
            for ob in bearish_obs:
                if ob['low'] <= current_price <= ob['high']:
                    print(f"🔴 {symbol} @ ${current_price:.2f} touche zone ROUGE [{ob['low']:.2f}-{ob['high']:.2f}]")
                    break


    def run(self):
        """Point d'entrée principal"""
        if self.mode == 'backtest':
            self.run_backtest()
        elif self.mode == 'realtime':
            self.run_realtime()
        else:
            print(f"Mode inconnu: {self.mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scanner de stocks avec Order Blocks')
    parser.add_argument('--backtest', action='store_true', help='Lance en mode backtest')
    parser.add_argument('--chart', type=str, metavar='SYMBOL', help='Génère un graphique pour le symbole (ex: --chart AAPL)')
    args = parser.parse_args()

    scanner = StockScanner('settings.json', is_backtest=args.backtest, chart_symbol=args.chart)
    scanner.run()
