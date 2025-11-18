#!/usr/bin/env python3
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from zoneinfo import ZoneInfo
from ib_insync import IB, Stock
import argparse
import plotly.graph_objects as go
from smc_analyzer import SMCAnalyzer
import logging
import yfinance as yf
import warnings

# Désactiver les logs ib_insync et yfinance
logging.getLogger('ib_insync').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

class StockScanner:
    def __init__(self, settings_file, is_backtest=False, chart_symbol=None):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        # Récupérer les indicateurs depuis settings
        self.indicators_config = self.settings.get('indicators', {})

        # Initialiser les analyzers selon les indicateurs activés
        self.smc_analyzer = None
        if self.indicators_config.get('smc', {}).get('enabled', False):
            smc_config = self.indicators_config['smc']
            self.smc_analyzer = SMCAnalyzer(smc_config)

        self.sr_analyzer = None
        if self.indicators_config.get('support_resistance', {}).get('enabled', False):
            from sr_levels_analyzer import SRLevelsAnalyzer
            sr_config = self.indicators_config['support_resistance']
            self.sr_analyzer = SRLevelsAnalyzer(sr_config)

        self.macd_analyzer = None
        if self.indicators_config.get('macd', {}).get('enabled', False):
            from macd_analyzer import MACDAnalyzer
            macd_config = self.indicators_config['macd']
            self.macd_analyzer = MACDAnalyzer(macd_config)

        self.mode = 'backtest' if is_backtest else 'realtime'
        self.data_folder = self.settings['data_folder']
        self.config_file = 'config.txt'
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
            return None


    def download_yahoo_data(self, symbol, candle_nb, interval):
        """Télécharge les données depuis Yahoo Finance"""
        import sys
        from io import StringIO

        old_stderr = sys.stderr
        try:
            # Rediriger stderr pour supprimer les erreurs yfinance
            sys.stderr = StringIO()

            ticker = yf.Ticker(symbol)

            # Calculer la période nécessaire avec marge de sécurité
            if interval == '1d':
                days_needed = int(candle_nb * 1.6)  # Marge pour weekends/jours fériés
            elif interval == '1h':
                days_needed = int(candle_nb / 6.5)  # ~6.5h de trading par jour
            elif interval == '1wk':
                days_needed = candle_nb * 7 * 2
            else:
                days_needed = candle_nb * 2  # Marge par défaut

            end_date = datetime.now(ZoneInfo('America/New_York'))
            start_date = end_date - timedelta(days=days_needed)

            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                return None

            # Validation: vérifier les colonnes requises
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return None

            # Prendre les N dernières bougies
            df = df.tail(candle_nb)

            df.reset_index(inplace=True)

            # Vérifier que Date existe après reset_index
            if 'Date' not in df.columns:
                return None

            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            return df
        except Exception as e:
            return None
        finally:
            # Restaurer stderr dans tous les cas
            sys.stderr = old_stderr


    def generate_chart(self, symbol, df):
        """Génère un graphique HTML avec les indicateurs activés"""
        # Créer le dossier chart s'il n'existe pas
        chart_folder = 'chart'
        os.makedirs(chart_folder, exist_ok=True)

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

        title_parts = [symbol]
        bullish_count = 0
        bearish_count = 0

        # Ajouter SMC si activé
        if self.smc_analyzer:
            smc_result = self.smc_analyzer.analyze(df)
            bullish_obs = smc_result['order_blocks']['bullish']
            bearish_obs = smc_result['order_blocks']['bearish']
            bullish_count = len(bullish_obs)
            bearish_count = len(bearish_obs)

            # Ajouter les zones BLEUES (Bullish Order Blocks)
            for ob in bullish_obs:
                idx = ob['index']
                date = df.iloc[idx]['Date'] if 'Date' in df.columns else idx
                end_date = df.iloc[-1]['Date'] if 'Date' in df.columns else len(df) - 1

                # Zone rectangulaire qui s'étend jusqu'à la fin
                fig.add_shape(
                    type="rect",
                    x0=date,
                    x1=end_date,
                    y0=ob['low'],
                    y1=ob['high'],
                    fillcolor="blue",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

                # Label des prix à droite
                mid_price = (ob['low'] + ob['high']) / 2
                fig.add_annotation(
                    x=end_date,
                    y=mid_price,
                    text=f"{ob['low']:.2f}-{ob['high']:.2f}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(color="white", size=14),
                    bgcolor="rgba(0,0,255,0.3)"
                )

            # Ajouter les zones ROUGES (Bearish Order Blocks)
            for ob in bearish_obs:
                idx = ob['index']
                date = df.iloc[idx]['Date'] if 'Date' in df.columns else idx
                end_date = df.iloc[-1]['Date'] if 'Date' in df.columns else len(df) - 1

                # Zone rectangulaire qui s'étend jusqu'à la fin
                fig.add_shape(
                    type="rect",
                    x0=date,
                    x1=end_date,
                    y0=ob['low'],
                    y1=ob['high'],
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )

                # Label des prix à droite
                mid_price = (ob['low'] + ob['high']) / 2
                fig.add_annotation(
                    x=end_date,
                    y=mid_price,
                    text=f"{ob['low']:.2f}-{ob['high']:.2f}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(color="red", size=14),
                    bgcolor="rgba(0,0,0,0.5)"
                )

            title_parts.append(f'OB (Bullish={bullish_count}, Bearish={bearish_count})')

        # Ajouter S/R si activé
        if self.sr_analyzer:
            sr_result = self.sr_analyzer.analyze(df)

            # Ajouter toutes les lignes de résistance (rouge, pointillée)
            for lvl in sr_result['resistance_levels']:
                start_date = df.iloc[lvl['start_idx']]['Date'] if 'Date' in df.columns else lvl['start_idx']
                end_date = df.iloc[lvl['end_idx']]['Date'] if 'Date' in df.columns else lvl['end_idx']

                fig.add_shape(
                    type="line",
                    x0=start_date,
                    x1=end_date,
                    y0=lvl['level'],
                    y1=lvl['level'],
                    line=dict(color="red", width=1, dash="dash")
                )

                # Annotation seulement sur le dernier segment
                if lvl == sr_result['resistance_levels'][-1]:
                    fig.add_annotation(
                        x=end_date,
                        y=lvl['level'],
                        text=f"${lvl['level']:.2f}",
                        showarrow=False,
                        xanchor="left",
                        font=dict(color="red", size=10),
                        bgcolor="rgba(255,0,0,0.2)"
                    )

            # Ajouter toutes les lignes de support (bleue, pointillée)
            for lvl in sr_result['support_levels']:
                start_date = df.iloc[lvl['start_idx']]['Date'] if 'Date' in df.columns else lvl['start_idx']
                end_date = df.iloc[lvl['end_idx']]['Date'] if 'Date' in df.columns else lvl['end_idx']

                fig.add_shape(
                    type="line",
                    x0=start_date,
                    x1=end_date,
                    y0=lvl['level'],
                    y1=lvl['level'],
                    line=dict(color="blue", width=1, dash="dash")
                )

                # Annotation seulement sur le dernier segment
                if lvl == sr_result['support_levels'][-1]:
                    fig.add_annotation(
                        x=end_date,
                        y=lvl['level'],
                        text=f"${lvl['level']:.2f}",
                        showarrow=False,
                        xanchor="left",
                        font=dict(color="blue", size=10),
                        bgcolor="rgba(0,0,255,0.2)"
                    )

            # Ajouter les marqueurs de cassures avec annotations
            for break_info in sr_result['breaks']:
                idx = break_info['index']
                date = df.iloc[idx]['Date'] if 'Date' in df.columns else idx

                # Couleur et texte selon le type
                if break_info['type'] == 'support_break':
                    bgcolor = 'rgba(255,0,0,0.7)'
                    text = 'S Break'
                elif break_info['type'] == 'resistance_break':
                    bgcolor = 'rgba(0,255,0,0.7)'
                    text = 'R Break'
                elif break_info['type'] == 'bear_wick':
                    bgcolor = 'rgba(255,0,0,0.7)'
                    text = 'Bear Wick'
                elif break_info['type'] == 'bull_wick':
                    bgcolor = 'rgba(0,255,0,0.7)'
                    text = 'Bull Wick'
                else:
                    bgcolor = 'rgba(128,128,128,0.7)'
                    text = break_info['description']

                # Ajouter annotation avec rectangle
                fig.add_annotation(
                    x=date,
                    y=break_info['price'],
                    text=text,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=bgcolor.replace('0.7', '1.0'),
                    ax=0,
                    ay=-40 if 'bull' in break_info['type'] or break_info['type'] == 'resistance_break' else 40,
                    font=dict(color="white", size=10),
                    bgcolor=bgcolor,
                    bordercolor="white",
                    borderwidth=1
                )

            title_parts.append('S/R Levels')

        # Ajouter MACD crossovers si activé
        if self.macd_analyzer:
            macd_result = self.macd_analyzer.analyze(df)

            # Ajouter des flèches aux croisements
            for cross in macd_result['crossovers']:
                idx = cross['index']
                date = df.iloc[idx]['Date'] if 'Date' in df.columns else idx

                if cross['type'] == 'bullish':
                    # Flèche bleue vers le haut EN DESSOUS de la bougie
                    color = 'blue'
                    marker_symbol = 'triangle-up'
                    y_offset = -0.015  # Décalage vers le bas en % du prix
                else:
                    # Flèche rouge vers le bas AU-DESSUS de la bougie
                    color = 'red'
                    marker_symbol = 'triangle-down'
                    y_offset = 0.015  # Décalage vers le haut en % du prix

                # Calculer la position Y de la flèche
                arrow_y = cross['price'] * (1 + y_offset)

                fig.add_trace(go.Scatter(
                    x=[date],
                    y=[arrow_y],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=color,
                        symbol=marker_symbol,
                        line=dict(width=1, color=color)
                    ),
                    name='MACD Cross',
                    showlegend=False
                ))

            title_parts.append(f'MACD ({len(macd_result["crossovers"])} crosses)')

        # Mise en forme
        chart_title = f"{' - '.join(title_parts)}"
        fig.update_layout(
            title=chart_title,
            yaxis_title='Prix',
            xaxis_title='Date',
            template='plotly_dark',
            height=800,
            xaxis_rangeslider_visible=False
        )

        # Ligne horizontale pour le prix actuel
        current_price = df.iloc[-1]['Close']
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            xref="paper",
            y0=current_price,
            y1=current_price,
            line=dict(color="yellow", width=2, dash="dash")
        )
        fig.add_annotation(
            x=1,
            xref="paper",
            y=current_price,
            text=f"${current_price:.2f}",
            showarrow=False,
            xanchor="left",
            font=dict(color="yellow", size=14, weight="bold"),
            bgcolor="rgba(0,0,0,0.7)"
        )

        # Sauvegarder dans le dossier chart
        filename = os.path.join(chart_folder, f'{symbol}_indicators.html')
        fig.write_html(filename)
        return filename


    def run_backtest(self):
        """Execute le mode backtest"""
        import time

        backtest_config = self.settings['backtest']
        candle_nb = backtest_config['candle_nb']
        interval = backtest_config['interval']
        test_start = backtest_config['test_candle_start']
        test_stop = backtest_config['test_candle_stop']

        # Calculer le nombre total de bougies à charger
        total_candles_needed = candle_nb + test_stop

        # Créer le dossier data s'il n'existe pas
        os.makedirs(self.data_folder, exist_ok=True)

        # Si --chart spécifié, utiliser directement le symbole
        if self.chart_symbol:
            watchlist = [{'symbol': self.chart_symbol, 'provider': 'IBKR'}]
        else:
            # Scanner tous les stocks NASDAQ 100
            watchlist = self.get_nasdaq100_symbols()

        # Utiliser l'intervalle de scan du mode realtime
        update_interval = self.settings['realtime']['update_interval_seconds']

        print(f"Scanner backtest démarré - scanne {len(watchlist)} symboles toutes les {update_interval}s")
        print("Appuyez sur Ctrl+C pour arrêter\n")

        while True:
            try:
                now = datetime.now(ZoneInfo('America/New_York'))
                today = now.strftime('%Y-%m-%d')
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

                print(f"=== Scan {timestamp} ===")

                for item in watchlist:
                    symbol = item['symbol']
                    filename = self.get_data_filename(symbol, total_candles_needed, interval, today)

                    # Télécharge ou charge les données (backtest = toujours Yahoo)
                    if self.check_file_exists(filename):
                        df = pd.read_csv(filename)
                    else:
                        df = self.download_yahoo_data(symbol, total_candles_needed, interval)
                        if df is not None:
                            df.to_csv(filename, index=False)
                        else:
                            continue

                    if df is None or len(df) == 0:
                        continue

                    # Prix actuel
                    current_price = df.iloc[-1]['Close']
                    alert_triggered = False

                    # Alertes MACD uniquement (croisements sur la dernière bougie)
                    if self.macd_analyzer:
                        macd_result = self.macd_analyzer.analyze(df)
                        for cross in macd_result['crossovers']:
                            if cross['index'] == len(df) - 1:  # Croisement sur la dernière bougie
                                if cross['type'] == 'bullish':
                                    print(f"🔵 {symbol} @ ${current_price:.2f} - MACD Bullish Cross")
                                else:
                                    print(f"🔴 {symbol} @ ${current_price:.2f} - MACD Bearish Cross")
                                alert_triggered = True

                    # Générer le graphique si --chart spécifié OU si alerte déclenchée
                    if self.chart_symbol or alert_triggered:
                        self.generate_chart(symbol, df)

                print(f"Prochain scan dans {update_interval}s...\n")
                time.sleep(update_interval)

            except KeyboardInterrupt:
                print("\nScanner arrêté par l'utilisateur")
                break
            except Exception as e:
                print(f"Erreur: {e}")
                time.sleep(update_interval)


    def run_realtime(self):
        """Execute le mode realtime avec alertes sur NASDAQ 100"""
        import time

        realtime_config = self.settings['realtime']
        candle_nb = realtime_config['candle_nb']
        interval = realtime_config['interval']
        update_interval = realtime_config['update_interval_seconds']

        # Créer le dossier data
        os.makedirs(self.data_folder, exist_ok=True)

        # Scanne NASDAQ 100
        watchlist = self.get_nasdaq100_symbols()

        print(f"Scanner realtime démarré - scanne {len(watchlist)} symboles toutes les {update_interval}s")
        print("Appuyez sur Ctrl+C pour arrêter\n")

        while True:
            try:
                now = datetime.now(ZoneInfo('America/New_York'))
                today = now.strftime('%Y-%m-%d')
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

                print(f"=== Scan {timestamp} ===")

                for item in watchlist:
                    symbol = item['symbol']
                    filename = self.get_data_filename(symbol, candle_nb, interval, today)

                    # Télécharge les données fraîches (realtime = toujours IBKR)
                    df = self.download_ibkr_data(symbol, candle_nb, interval)
                    if df is not None:
                        df.to_csv(filename, index=False)
                    else:
                        continue

                    if df is None or len(df) == 0:
                        continue

                    # Prix actuel
                    current_price = df.iloc[-1]['Close']
                    alert_triggered = False

                    # Alertes MACD uniquement (croisements sur la dernière bougie)
                    if self.macd_analyzer:
                        macd_result = self.macd_analyzer.analyze(df)
                        for cross in macd_result['crossovers']:
                            if cross['index'] == len(df) - 1:  # Croisement sur la dernière bougie
                                if cross['type'] == 'bullish':
                                    print(f"🔵 {symbol} @ ${current_price:.2f} - MACD Bullish Cross")
                                else:
                                    print(f"🔴 {symbol} @ ${current_price:.2f} - MACD Bearish Cross")
                                alert_triggered = True

                    # Générer le graphique si une alerte a été déclenchée
                    if alert_triggered:
                        self.generate_chart(symbol, df)

                print(f"Prochain scan dans {update_interval}s...\n")
                time.sleep(update_interval)

            except KeyboardInterrupt:
                print("\nScanner arrêté par l'utilisateur")
                break
            except Exception as e:
                print(f"Erreur: {e}")
                time.sleep(update_interval)


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
