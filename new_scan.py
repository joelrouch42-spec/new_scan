#!/usr/bin/env python3
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from zoneinfo import ZoneInfo
from ib_insync import IB, Stock
import time
import argparse
import numpy as np
from typing import List, Tuple, Optional, Dict
import plotly.graph_objects as go
from smc_analyzer import SMCAnalyzer

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

        print("settings_file", settings_file)
        print("is_backtest: ", is_backtest)
        print("mode: ", self.mode)
        print("folder: ", self.data_folder)
        print("patterns_folder: ", self.patterns_folder)


    def get_data_filename(self, symbol, candle_nb, interval, date):
        """Génère le nom du fichier de données"""
        return os.path.join(
            self.data_folder,
            f"{date}_{symbol}_{candle_nb}_{interval}.csv"
        )

    def check_file_exists(self, filepath):
        """Vérifie si le fichier existe"""
        return os.path.exists(filepath)

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
            print(f"Erreur IBKR: {e}")
            return None


    def generate_chart(self, symbol, df):
        """Génère un graphique HTML avec les Order Blocks"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

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

        # Sauvegarder
        filename = f'{symbol}_order_blocks.html'
        fig.write_html(filename)
        print(f"\nGraphique généré: {filename}")
        print(f"Bullish Order Blocks (bleu): {len(bullish_obs)}")
        print(f"Bearish Order Blocks (rouge): {len(bearish_obs)}")

        return filename


    def run_backtest(self):
        """Execute le mode backtest avec simulation de trades"""
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
                print(f"Erreur: Symbole {self.chart_symbol} non trouvé dans la watchlist")
                return

        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
        print(today)

        for item in watchlist:
            symbol = item['symbol']
            filename = self.get_data_filename(symbol, total_candles_needed, interval, today)
            print(filename)

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

            total_candles = len(df)

            print(f"\n{'='*60}")
            print(f"BACKTEST SMC pour {symbol}")
            print(f"{'='*60}")
            print(f"Total candles: {total_candles}")
            print(f"Période backtest: {candle_nb} bougies de training, {test_stop - test_start + 1} bougies de test")

            # SIMULATION DE TRADING
            trades = []
            active_trade = None

            # Commencer à la bougie candle_nb (après training)
            start_idx = candle_nb
            end_idx = total_candles - test_start + 1

            for i in range(start_idx, end_idx):
                # Données jusqu'à la bougie actuelle
                df_current = df.iloc[:i].copy()
                current_candle = df.iloc[i]

                # Si on a un trade actif, vérifier stop/target
                if active_trade:
                    trade_type = active_trade['type']

                    # Check LONG
                    if 'LONG' in trade_type:
                        # Stop hit
                        if current_candle['Low'] <= active_trade['stop']:
                            active_trade['exit_price'] = active_trade['stop']
                            active_trade['exit_candle'] = i
                            active_trade['exit_date'] = df.iloc[i]['Date'] if 'Date' in df.columns else f'Candle {i}'
                            active_trade['result'] = 'LOSS'
                            active_trade['pnl'] = active_trade['exit_price'] - active_trade['entry']
                            trades.append(active_trade)
                            active_trade = None
                        # Target hit
                        elif current_candle['High'] >= active_trade['target']:
                            active_trade['exit_price'] = active_trade['target']
                            active_trade['exit_candle'] = i
                            active_trade['exit_date'] = df.iloc[i]['Date'] if 'Date' in df.columns else f'Candle {i}'
                            active_trade['result'] = 'WIN'
                            active_trade['pnl'] = active_trade['exit_price'] - active_trade['entry']
                            trades.append(active_trade)
                            active_trade = None

                    # Check SHORT
                    elif 'SHORT' in trade_type:
                        # Stop hit
                        if current_candle['High'] >= active_trade['stop']:
                            active_trade['exit_price'] = active_trade['stop']
                            active_trade['exit_candle'] = i
                            active_trade['exit_date'] = df.iloc[i]['Date'] if 'Date' in df.columns else f'Candle {i}'
                            active_trade['result'] = 'LOSS'
                            active_trade['pnl'] = active_trade['entry'] - active_trade['exit_price']
                            trades.append(active_trade)
                            active_trade = None
                        # Target hit
                        elif current_candle['Low'] <= active_trade['target']:
                            active_trade['exit_price'] = active_trade['target']
                            active_trade['exit_candle'] = i
                            active_trade['exit_date'] = df.iloc[i]['Date'] if 'Date' in df.columns else f'Candle {i}'
                            active_trade['result'] = 'WIN'
                            active_trade['pnl'] = active_trade['entry'] - active_trade['exit_price']
                            trades.append(active_trade)
                            active_trade = None

                # Si pas de trade actif, chercher une alerte
                if not active_trade:
                    smc_result = self.smc_analyzer.analyze(df_current)
                    alerts = self.smc_analyzer.detect_setups(df_current, smc_result)

                    if alerts:
                        # Prendre la première alerte
                        alert = alerts[0]

                        # Ouvrir le trade
                        active_trade = {
                            'type': alert['type'],
                            'entry': alert['entry'],
                            'stop': alert['stop'],
                            'target': alert['target'],
                            'entry_candle': i,
                            'entry_date': df.iloc[i]['Date'] if 'Date' in df.columns else f'Candle {i}',
                            'reason': alert['reason']
                        }

            # Clore le trade actif si encore ouvert à la fin
            if active_trade:
                last_candle = df.iloc[end_idx - 1]
                active_trade['exit_price'] = last_candle['Close']
                active_trade['exit_candle'] = end_idx - 1
                active_trade['exit_date'] = df.iloc[end_idx - 1]['Date'] if 'Date' in df.columns else f'Candle {end_idx - 1}'
                active_trade['result'] = 'OPEN'
                if 'LONG' in active_trade['type']:
                    active_trade['pnl'] = active_trade['exit_price'] - active_trade['entry']
                else:
                    active_trade['pnl'] = active_trade['entry'] - active_trade['exit_price']
                trades.append(active_trade)

            # AFFICHER LES RÉSULTATS
            print(f"\n{'*'*60}")
            print(f"   RÉSULTATS BACKTEST")
            print(f"{'*'*60}")

            if trades:
                wins = [t for t in trades if t['result'] == 'WIN']
                losses = [t for t in trades if t['result'] == 'LOSS']
                total_trades = len(trades)
                win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0

                total_pnl = sum(t['pnl'] for t in trades)
                avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
                avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0

                print(f"\nTotal Trades: {total_trades}")
                print(f"Wins: {len(wins)} | Losses: {len(losses)}")
                print(f"Win Rate: {win_rate:.1f}%")
                print(f"Total P&L: ${total_pnl:.2f}")
                print(f"Average Win: ${avg_win:.2f}")
                print(f"Average Loss: ${avg_loss:.2f}")

                if avg_loss != 0:
                    profit_factor = abs(avg_win / avg_loss)
                    print(f"Profit Factor: {profit_factor:.2f}")

                print(f"\n--- DÉTAIL DES TRADES ---")
                for i, trade in enumerate(trades, 1):
                    print(f"\nTrade #{i}: {trade['type']} - {trade['result']}")
                    print(f"  Entry: ${trade['entry']:.2f} @ {trade['entry_date']} (candle {trade['entry_candle']})")
                    print(f"  Exit: ${trade['exit_price']:.2f} @ {trade['exit_date']} (candle {trade['exit_candle']})")
                    print(f"  Stop: ${trade['stop']:.2f} | Target: ${trade['target']:.2f}")
                    print(f"  P&L: ${trade['pnl']:.2f}")
                    print(f"  Raison: {trade['reason']}")
            else:
                print("\nAucun trade exécuté pendant la période de backtest.")

            # Générer le graphique si demandé
            if self.chart_symbol:
                self.generate_chart(symbol, df)
            else:
                # Analyse SMC actuelle (dernière bougie)
                smc_result = self.smc_analyzer.analyze(df)
                pd_zones = smc_result['premium_discount']

                print(f"\n--- SITUATION ACTUELLE ---")
                print(f"Prix: ${pd_zones['current_price']:.2f}")
                print(f"Zone: {pd_zones['current_zone'].upper()}")
                print(f"Order Blocks: {len(smc_result['order_blocks']['bullish'])} bullish, {len(smc_result['order_blocks']['bearish'])} bearish")
                print(f"Fair Value Gaps: {len(smc_result['fvg']['bullish'])} bullish, {len(smc_result['fvg']['bearish'])} bearish")


    def run(self):
        """Point d'entrée principal"""
        if self.mode == 'backtest':
            self.run_backtest()
        elif self.mode == 'realtime':
            print("Mode realtime pas encore implémenté")
        else:
            print(f"Mode inconnu: {self.mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scanner de stocks avec Smart Money Concepts')
    parser.add_argument('--backtest', action='store_true', help='Lance en mode backtest')
    parser.add_argument('--chart', type=str, metavar='SYMBOL', help='Génère un graphique HTML pour le symbole spécifié (ex: --chart AAPL)')
    args = parser.parse_args()

    scanner = StockScanner('settings.json', is_backtest=args.backtest, chart_symbol=args.chart)
    scanner.run()
