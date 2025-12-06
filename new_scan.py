#!/usr/bin/env python3
import json
import os
from datetime import datetime, time, timedelta
import pandas as pd
from zoneinfo import ZoneInfo
from ib_insync import IB, Stock
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from squeeze_momentum_indicator import SqueezeMomentumIndicator
import logging
import yfinance as yf
import warnings

# Désactiver les logs ib_insync et yfinance
logging.getLogger('ib_insync').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

class StockScanner:
    def __init__(self, settings_file, is_backtest=False, chart_symbol=None, all_charts=False, nasdaq=False, crypto=False, test_candle_override=None):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        # Récupérer les indicateurs depuis settings
        self.indicators_config = self.settings.get('indicators', {})

        # Initialiser les analyzers selon les indicateurs activés

        self.sr_analyzer = None
        if self.indicators_config.get('support_resistance', {}).get('enabled', False):
            from sr_levels_analyzer import SRLevelsAnalyzer
            sr_config = self.indicators_config['support_resistance']
            self.sr_analyzer = SRLevelsAnalyzer(sr_config)


        self.squeeze_analyzer = None
        if self.indicators_config.get('squeeze_momentum', {}).get('enabled', False):
            from squeeze_momentum_analyzer import SqueezeAnalyzer
            squeeze_config = self.indicators_config['squeeze_momentum']
            self.squeeze_analyzer = SqueezeAnalyzer(squeeze_config)

        self.adx_analyzer = None
        if 'adx' in self.indicators_config and self.indicators_config['adx'].get('enabled', True):
            from adx_analyzer import ADXAnalyzer
            adx_config = self.indicators_config['adx']
            self.adx_analyzer = ADXAnalyzer(adx_config)


        self.mode = 'backtest' if is_backtest else 'realtime'
        self.data_folder = self.settings['data_folder']
        self.config_file = 'config.txt'
        self.chart_symbol = chart_symbol
        self.all_charts = all_charts
        self.nasdaq = nasdaq
        self.crypto = crypto
        self.test_candle_override = test_candle_override


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

    def get_crypto_symbols(self):
        """Retourne la liste des cryptomonnaies et actions crypto-liées"""
        symbols = [
            # Essayer cryptos directes
            'BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD', 'MATICUSD',
            # Actions crypto-liées (qui fonctionnent)  
            'MSTR', 'COIN', 'RIOT', 'MARA', 'CLSK', 'BITF', 'HUT', 'IREN'
        ]
        ibkr_symbols = [{'symbol': s, 'provider': 'IBKR'} for s in symbols]
        
        # Ajouter les cryptos yfinance
        crypto_yf = [
            'XRP-USD', 'XDC-USD', 'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 
            'DOT-USD', 'MATIC-USD', 'AVAX-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD',
            'BCH-USD', 'XLM-USD', 'ALGO-USD', 'ATOM-USD', 'ICP-USD', 'APT-USD',
            'NEAR-USD', 'FTM-USD', 'SAND-USD', 'MANA-USD', 'CRV-USD', 'AAVE-USD'
        ]
        yf_symbols = [{'symbol': s, 'provider': 'YFINANCE'} for s in crypto_yf]
        
        return ibkr_symbols + yf_symbols

    def download_crypto_yfinance(self, symbol, candle_nb, interval):
        """Télécharge les données crypto via yfinance"""
        try:
            import yfinance as yf
            
            # Mapping des intervalles
            interval_map = {'1d': '1d', '1h': '1h', '5m': '5m'}
            yf_interval = interval_map.get(interval, '1d')
            
            # Calculer la période
            if candle_nb <= 7:
                period = "7d"
            elif candle_nb <= 30:
                period = "1mo"
            elif candle_nb <= 90:
                period = "3mo"
            elif candle_nb <= 180:
                period = "6mo"
            else:
                period = "1y"
            
            # Télécharger
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=yf_interval)
            
            if df.empty:
                return None
                
            # Limiter au nombre de bougies demandées
            if len(df) > candle_nb:
                df = df.tail(candle_nb)
            
            # Reset de l'index et renommage des colonnes
            df = df.reset_index()
            if 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'Date'})
            
            return df
            
        except Exception as e:
            print(f"Erreur téléchargement crypto {symbol}: {e}")
            return None

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
            
            if not ib.isConnected():
                print(f"❌ IBKR Gateway non connecté ({host}:{port})")
                return None

            # Créer le contrat
            contract = Stock(symbol, 'SMART', 'USD')
            qualified = ib.qualifyContracts(contract)

            if not qualified:
                ib.disconnect()
                return None

            contract = qualified[0]

            # Calculer la durée
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
            print(f"❌ Erreur IBKR {symbol}: {e}")
            return None
        finally:
            # Restaurer stderr dans tous les cas
            sys.stderr = old_stderr


    def _generate_combined_signals(self, df):
        """
        Génère les signaux basés sur Squeeze Momentum
        """
        buy_signals = []
        sell_signals = []
        
        # Utiliser Squeeze Momentum pour les signaux si activé
        if self.indicators_config.get('squeeze_momentum', {}).get('enabled', False):
            squeeze_config = self.indicators_config['squeeze_momentum']
            squeeze_indicator = SqueezeMomentumIndicator(
                bb_length=squeeze_config.get('bb_length', 20),
                bb_mult=squeeze_config.get('bb_mult', 2.0),
                kc_length=squeeze_config.get('kc_length', 20),
                kc_mult=squeeze_config.get('kc_mult', 1.5),
                use_true_range=squeeze_config.get('use_true_range', True)
            )
            
            squeeze_result = squeeze_indicator.analyze(df)
            
            # Utiliser directement les signaux de l'indicateur
            for signal in squeeze_result['signals']:
                if signal['type'] == 'bullish':
                    buy_signals.append(signal)
                elif signal['type'] == 'bearish':
                    sell_signals.append(signal)
        
        return buy_signals, sell_signals


    def generate_chart(self, symbol, df):
        """Génère un graphique HTML avec les indicateurs activés"""
        # Créer le dossier chart s'il n'existe pas
        chart_folder = 'chart'
        os.makedirs(chart_folder, exist_ok=True)

        # Créer des subplots : prix en haut, momentum en bas
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price & Indicators', 'Squeeze Momentum'),
            row_heights=[0.7, 0.3]
        )

        # Ajouter les chandelles au premier subplot
        fig.add_trace(go.Candlestick(
            x=df['Date'] if 'Date' in df.columns else list(range(len(df))),
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)

        title_parts = [symbol]

        # S/R supprimé
        if False and self.sr_analyzer:
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

                # Ignorer les Bull Wick, Bear Wick, et Support Breaks
                if break_info['type'] in ['bull_wick', 'bear_wick', 'support_break']:
                    continue

                # Couleur et texte selon le type
                if break_info['type'] == 'support_break':
                    bgcolor = 'rgba(255,0,0,0.7)'
                    text = 'S Break'
                elif break_info['type'] == 'resistance_break':
                    bgcolor = 'rgba(0,255,0,0.7)'
                    text = 'R Break'
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
                    ay=-40 if break_info['type'] == 'resistance_break' else 40,
                    font=dict(color="white", size=10),
                    bgcolor=bgcolor,
                    bordercolor="white",
                    borderwidth=1
                )

            title_parts.append('S/R Levels')


        # Ajouter Squeeze Momentum si activé
        if self.indicators_config.get('squeeze_momentum', {}).get('enabled', False):
            squeeze_config = self.indicators_config['squeeze_momentum']
            squeeze_indicator = SqueezeMomentumIndicator(
                bb_length=squeeze_config.get('bb_length', 20),
                bb_mult=squeeze_config.get('bb_mult', 2.0),
                kc_length=squeeze_config.get('kc_length', 20),
                kc_mult=squeeze_config.get('kc_mult', 1.5),
                use_true_range=squeeze_config.get('use_true_range', True)
            )
            
            squeeze_result = squeeze_indicator.analyze(df)
            x_axis = df['Date'] if 'Date' in df.columns else list(range(len(df)))
            
            # Ajouter l'histogramme momentum au deuxième subplot
            momentum_values = squeeze_result['momentum']
            momentum_colors = squeeze_result['momentum_colors']
            
            # Convertir les couleurs pour Plotly
            color_map = {'lime': '#00FF00', 'green': '#008000', 'red': '#FF0000', 'maroon': '#800000', 'gray': '#808080'}
            
            for i, (momentum, color) in enumerate(zip(momentum_values, momentum_colors)):
                if not pd.isna(momentum) and i < len(x_axis):
                    fig.add_trace(go.Bar(
                        x=[x_axis[i]],
                        y=[momentum],
                        marker_color=color_map.get(color, color),
                        name='Momentum',
                        showlegend=False,
                        width=86400000 if 'Date' in df.columns else 0.8  # Width for daily bars
                    ), row=2, col=1)
            
            # Ajouter ligne de zéro pour le momentum
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=[0] * len(x_axis),
                mode='lines',
                line=dict(color='white', width=1),
                name='Zero Line',
                showlegend=False
            ), row=2, col=1)
            
            
            # Ajouter des flèches pour les signaux de couleur
            signals = squeeze_result['signals']
            for signal in signals:
                i = signal['index']
                if i < len(x_axis):
                    if signal['type'] == 'bullish':  # MAROON = BUY
                        # arrow_y = df.iloc[i]['Low'] - (df.iloc[i]['High'] - df.iloc[i]['Low']) * 0.8
                        # fig.add_trace(go.Scatter(
                        #     x=[x_axis[i]],
                        #     y=[arrow_y],
                        #     mode='markers',
                        #     marker=dict(
                        #         symbol='triangle-up',
                        #         size=15,
                        #         color='lime',
                        #         line=dict(width=1, color='green')
                        #     ),
                        #     name='BUY Signal',
                        #     showlegend=False,
                        #     hoverinfo='skip'
                        # ), row=1, col=1)
                        
                        # SL pour BUY: min des bas de cette bougie et la précédente
                        if i > 0:
                            sl_level = min(df.iloc[i]['Low'], df.iloc[i-1]['Low'])
                            # SL avec tolérance (1% plus bas par défaut)
                            sl_tolerance = 1  # TODO: récupérer depuis trading_settings.json
                            sl_with_tolerance = sl_level * (1 - sl_tolerance / 100)
                            
                            # Tracer ligne horizontale SL vers le passé
                            fig.add_shape(
                                type="line",
                                x0=x_axis[max(i-2, 0)],  # 2 bougies vers la gauche
                                x1=x_axis[i],
                                y0=sl_level,
                                y1=sl_level,
                                line=dict(color="green", width=4, dash="solid"),
                                row=1, col=1
                            )
                            
                            # Tracer ligne de tolérance (pointillée)
                            fig.add_shape(
                                type="line",
                                x0=x_axis[max(i-2, 0)],
                                x1=x_axis[i],
                                y0=sl_with_tolerance,
                                y1=sl_with_tolerance,
                                line=dict(color="lightgreen", width=2, dash="dot"),
                                row=1, col=1
                            )
                    
                    elif signal['type'] == 'bearish':  # GREEN = SELL
                        # arrow_y = df.iloc[i]['High'] + (df.iloc[i]['High'] - df.iloc[i]['Low']) * 0.8
                        # fig.add_trace(go.Scatter(
                        #     x=[x_axis[i]],
                        #     y=[arrow_y],
                        #     mode='markers',
                        #     marker=dict(
                        #         symbol='triangle-down',
                        #         size=15,
                        #         color='red',
                        #         line=dict(width=1, color='darkred')
                        #     ),
                        #     name='SELL Signal',
                        #     showlegend=False,
                        #     hoverinfo='skip'
                        # ), row=1, col=1)
                        
                        # SL pour SELL: max des hauts de cette bougie et la précédente
                        if i > 0:
                            sl_level = max(df.iloc[i]['High'], df.iloc[i-1]['High'])
                            # SL avec tolérance (1% plus haut par défaut)
                            sl_tolerance = 1  # TODO: récupérer depuis trading_settings.json
                            sl_with_tolerance = sl_level * (1 + sl_tolerance / 100)
                            
                            # Tracer ligne horizontale SL vers le passé
                            fig.add_shape(
                                type="line",
                                x0=x_axis[max(i-2, 0)],  # 2 bougies vers la gauche
                                x1=x_axis[i],
                                y0=sl_level,
                                y1=sl_level,
                                line=dict(color="red", width=4, dash="solid"),
                                row=1, col=1
                            )
                            
                            # Tracer ligne de tolérance (pointillée)
                            fig.add_shape(
                                type="line",
                                x0=x_axis[max(i-2, 0)],
                                x1=x_axis[i],
                                y0=sl_with_tolerance,
                                y1=sl_with_tolerance,
                                line=dict(color="lightcoral", width=2, dash="dot"),
                                row=1, col=1
                            )
            
            title_parts.append('SqzMom')



        # Mise en forme
        chart_title = f"{' - '.join(title_parts)}"
        fig.update_layout(
            title=chart_title,
            template='plotly_dark',
            height=900,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # Mettre à jour les axes
        fig.update_yaxes(title_text="Prix", row=1, col=1)
        fig.update_yaxes(title_text="Momentum", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        # Ligne horizontale pour le prix actuel
        current_price = df.iloc[-1]['Close']
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            xref="paper",
            y0=current_price,
            y1=current_price,
            yref="y1",
            line=dict(color="yellow", width=2, dash="dash")
        )
        fig.add_annotation(
            x=1,
            xref="paper",
            y=current_price,
            yref="y1",
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
        elif self.nasdaq:
            # Scanner NASDAQ 100
            watchlist = self.get_nasdaq100_symbols()
        elif self.crypto:
            # Scanner cryptomonnaies
            watchlist = self.get_crypto_symbols()
        else:
            print("❌ Erreur: Veuillez spécifier soit --nasdaq soit --crypto")
            print("   Exemple: python3 new_scan.py --nasdaq --backtest")
            print("   Exemple: python3 new_scan.py --crypto --chart MSTR")
            return

        print(f"Scanner backtest démarré - scanne {len(watchlist)} symboles (all_charts: {self.all_charts})\n")

        now = datetime.now(ZoneInfo('America/New_York'))
        today = now.strftime('%Y-%m-%d')
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

        print(f"=== Scan {timestamp} ===")

        for item in watchlist:
            symbol = item['symbol']
            provider = item.get('provider', 'IBKR')
            self.current_symbol = symbol  # Stocker pour multi-timeframe
            filename = self.get_data_filename(symbol, total_candles_needed, interval, today)

            # Télécharge ou charge les données
            if self.check_file_exists(filename):
                df = pd.read_csv(filename)
            else:
                # Choisir la méthode selon le provider
                if provider == 'YFINANCE':
                    df = self.download_crypto_yfinance(symbol, total_candles_needed, interval)
                else:
                    # Utiliser IBKR pour backtest (limite: ~600 bougies max)
                    candles_to_request = min(total_candles_needed, 600)
                    df = self.download_ibkr_data(symbol, candles_to_request, interval)
                
                if df is not None:
                    df.to_csv(filename, index=False)
                else:
                    # Si téléchargement échoue, chercher un fichier de cache plus ancien
                    cache_files = []
                    import glob
                    cache_pattern = os.path.join(self.data_folder, f"*_{symbol}_*.csv")
                    cache_files = glob.glob(cache_pattern)
                    
                    if cache_files:
                        # Prendre le fichier le plus récent
                        latest_cache = max(cache_files, key=os.path.getmtime)
                        df = pd.read_csv(latest_cache)
                    else:
                        continue

            if df is None or len(df) == 0:
                continue

            # Prix actuel
            current_price = df.iloc[-1]['Close']
            alert_triggered = False

            # Analyser tout le DataFrame comme en realtime
            buy_signals, sell_signals = self._generate_combined_signals(df)
            
            # Filtrer les signaux selon la plage de test (depuis la fin)
            total_candles = len(df)
            if test_stop >= total_candles:
                test_stop = total_candles - 1
            
            # Calculer les indices depuis la fin (bougie 0 = dernière)
            start_idx = total_candles - test_stop - 1
            end_idx = total_candles - test_start
            
            # Filtrer les signaux dans la plage
            filtered_buy = [s for s in buy_signals if start_idx <= s['index'] < end_idx]
            filtered_sell = [s for s in sell_signals if start_idx <= s['index'] < end_idx]
            
            total_signals = len(filtered_buy) + len(filtered_sell)

            if total_signals > 0:
                alert_triggered = True
                
            # Générer le graphique de la bougie 0 à test_stop
            if self.chart_symbol or self.all_charts or alert_triggered:
                # Afficher les dernières test_stop+1 bougies (0 à test_stop)
                df_chart_range = df.tail(test_stop + 1).copy().reset_index(drop=True)
                self.generate_chart(symbol, df_chart_range)

        print("\nScan terminé.")


    def run_realtime(self):
        """Execute le mode realtime avec alertes sur NASDAQ 100"""
        import time

        realtime_config = self.settings['realtime']
        candle_nb = realtime_config['candle_nb']
        interval = realtime_config['interval']
        update_interval = realtime_config['update_interval_seconds']
        test_candle = self.test_candle_override if self.test_candle_override is not None else realtime_config.get('test_candle', 0)

        # Créer le dossier data
        os.makedirs(self.data_folder, exist_ok=True)

        # Déterminer la watchlist selon les options
        if self.nasdaq:
            watchlist = self.get_nasdaq100_symbols()
        elif self.crypto:
            watchlist = self.get_crypto_symbols()
        else:
            print("❌ Erreur: Veuillez spécifier soit --nasdaq soit --crypto")
            print("   Exemple: python3 new_scan.py --nasdaq --backtest")
            print("   Exemple: python3 new_scan.py --crypto --realtime")
            return

        print(f"Scanner realtime démarré - scanne {len(watchlist)} symboles toutes les {update_interval}s")
        print("Appuyez sur Ctrl+C pour arrêter\n")

        while True:
            try:
                now = datetime.now(ZoneInfo('America/New_York'))
                today = now.strftime('%Y-%m-%d')
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

                print(f"=== Scan {timestamp} (test_candle: {test_candle}) ===")

                for item in watchlist:
                    symbol = item['symbol']
                    provider = item.get('provider', 'IBKR')
                    filename = self.get_data_filename(symbol, candle_nb, interval, today)

                    # Si test_candle == 0, télécharger données fraîches, sinon utiliser cache
                    if test_candle == 0:
                        # Télécharge les données fraîches selon le provider
                        if provider == 'YFINANCE':
                            df = self.download_crypto_yfinance(symbol, candle_nb, interval)
                        else:
                            df = self.download_ibkr_data(symbol, candle_nb, interval)
                        
                        if df is not None:
                            df.to_csv(filename, index=False)
                        else:
                            continue
                    else:
                        # Utiliser les fichiers de cache pour tester les bougies historiques
                        cache_files = []
                        import glob
                        cache_pattern = os.path.join(self.data_folder, f"*_{symbol}_*.csv")
                        cache_files = glob.glob(cache_pattern)
                        
                        if cache_files:
                            # Prendre le fichier le plus récent
                            latest_cache = max(cache_files, key=os.path.getmtime)
                            df = pd.read_csv(latest_cache)
                        else:
                            continue

                    if df is None or len(df) == 0:
                        continue

                    # Prix actuel
                    current_price = df.iloc[-1]['Close']
                    alert_triggered = False

                    # Détecter si c'est le premier scan de l'ouverture (nouvelle bougie)
                    is_opening_scan = self._is_opening_scan(symbol, df)
                    
                    # Gestion SL selon type de scan
                    if self.mode == 'realtime' and test_candle == 0:
                        from trading_manager import TradingManager
                        trader = TradingManager('settings.json')
                        if trader.connect():
                            
                            if is_opening_scan:
                                # SCAN OUVERTURE: SL exact + vérification signaux + mise à jour JSON
                                open_price = df.iloc[-1]['Open']
                                current_prices = {symbol: {'Open': open_price}}
                                
                                sl_hits = trader.check_sl_hits_opening(current_prices)
                                
                                for sl_hit in sl_hits:
                                    hit_symbol = sl_hit['symbol']
                                    action = sl_hit['action']
                                    
                                    # Générer signaux pour vérifier nouveau signal même sens
                                    buy_signals, sell_signals = self._generate_combined_signals(df)
                                    target_candle_idx = len(df) - 1 - test_candle
                                    
                                    has_same_direction_signal = False
                                    
                                    if action == 'BUY':
                                        for signal in buy_signals:
                                            if signal['index'] == target_candle_idx:
                                                has_same_direction_signal = True
                                                # Mise à jour SL avec nouveau signal
                                                signal_idx = signal['index']
                                                high_price = df.iloc[signal_idx]['High']
                                                low_price = df.iloc[signal_idx]['Low']
                                                previous_high = df.iloc[signal_idx - 1]['High'] if signal_idx > 0 else high_price
                                                previous_low = df.iloc[signal_idx - 1]['Low'] if signal_idx > 0 else low_price
                                                trader.handle_new_signal(hit_symbol, 'BUY', high_price, low_price, previous_high, previous_low, timestamp)
                                                print(f"🔄 Nouveau signal BUY {hit_symbol} - SL mis à jour")
                                                break
                                    else:  # SELL
                                        for signal in sell_signals:
                                            if signal['index'] == target_candle_idx:
                                                has_same_direction_signal = True
                                                # Mise à jour SL avec nouveau signal
                                                signal_idx = signal['index']
                                                high_price = df.iloc[signal_idx]['High']
                                                low_price = df.iloc[signal_idx]['Low']
                                                previous_high = df.iloc[signal_idx - 1]['High'] if signal_idx > 0 else high_price
                                                previous_low = df.iloc[signal_idx - 1]['Low'] if signal_idx > 0 else low_price
                                                trader.handle_new_signal(hit_symbol, 'SELL', high_price, low_price, previous_high, previous_low, timestamp)
                                                print(f"🔄 Nouveau signal SELL {hit_symbol} - SL mis à jour")
                                                break
                                    
                                    # Si pas de nouveau signal même sens: fermer position
                                    if not has_same_direction_signal:
                                        opposite_action = 'SELL' if action == 'BUY' else 'BUY'
                                        trade_result = trader.smart_trade(hit_symbol, opposite_action)
                                        if trade_result:
                                            print(f"🚨 SL OUVERTURE déclenché: {hit_symbol} position fermée")
                                            trader.remove_position_tracking(hit_symbol)
                            
                            else:
                                # SCAN CONTINU: SL + tolérance, fermeture immédiate
                                current_price = df.iloc[-1]['Close']  # Prix temps réel
                                current_prices = {symbol: {'Current': current_price}}
                                
                                sl_hits = trader.check_sl_hits_continuous(current_prices)
                                
                                for sl_hit in sl_hits:
                                    hit_symbol = sl_hit['symbol']
                                    action = sl_hit['action']
                                    opposite_action = 'SELL' if action == 'BUY' else 'BUY'
                                    
                                    trade_result = trader.smart_trade(hit_symbol, opposite_action)
                                    if trade_result:
                                        print(f"🚨 CRASH PROTECTION: {hit_symbol} position fermée immédiatement")
                                        trader.remove_position_tracking(hit_symbol)
                            
                            trader.disconnect()
                    
                    # Alertes combinées Squeeze + MACD + ADX
                    buy_signals, sell_signals = self._generate_combined_signals(df)
                    target_candle_idx = len(df) - 1 - test_candle

                    for signal in buy_signals:
                        if signal['index'] == target_candle_idx:
                            signal_date = df.iloc[signal['index']]['Date'] if 'Date' in df.columns else f"Index {signal['index']}"
                            print(f"🟢 BUY {symbol} @ ${current_price:.2f} - {signal_date}")
                            
                            # Calculer SL et gérer position
                            if self.mode == 'realtime':
                                from trading_manager import TradingManager
                                trader = TradingManager('settings.json')
                                if trader.connect():
                                    # TOUJOURS calculer et sauvegarder le SL (même si trade échoue)
                                    signal_idx = signal['index']
                                    high_price = df.iloc[signal_idx]['High']
                                    low_price = df.iloc[signal_idx]['Low']
                                    
                                    if signal_idx > 0:
                                        previous_high = df.iloc[signal_idx - 1]['High']
                                        previous_low = df.iloc[signal_idx - 1]['Low']
                                    else:
                                        previous_high = high_price
                                        previous_low = low_price
                                    
                                    entry_date = signal_date
                                    trader.handle_new_signal(symbol, 'BUY', high_price, low_price, 
                                                           previous_high, previous_low, entry_date)
                                    
                                    # Passer l'ordre d'achat (après sauvegarde SL)
                                    trade_result = trader.smart_trade(symbol, 'BUY')
                                    
                                    trader.disconnect()
                            
                            alert_triggered = True
                            break

                    for signal in sell_signals:
                        if signal['index'] == target_candle_idx:
                            signal_date = df.iloc[signal['index']]['Date'] if 'Date' in df.columns else f"Index {signal['index']}"
                            print(f"🔴 SELL {symbol} @ ${current_price:.2f} - {signal_date}")
                            
                            # Calculer SL et gérer position
                            if self.mode == 'realtime':
                                from trading_manager import TradingManager
                                trader = TradingManager('settings.json')
                                if trader.connect():
                                    # TOUJOURS calculer et sauvegarder le SL (même si trade échoue)
                                    signal_idx = signal['index']
                                    high_price = df.iloc[signal_idx]['High']
                                    low_price = df.iloc[signal_idx]['Low']
                                    
                                    if signal_idx > 0:
                                        previous_high = df.iloc[signal_idx - 1]['High']
                                        previous_low = df.iloc[signal_idx - 1]['Low']
                                    else:
                                        previous_high = high_price
                                        previous_low = low_price
                                    
                                    entry_date = signal_date
                                    trader.handle_new_signal(symbol, 'SELL', high_price, low_price,
                                                           previous_high, previous_low, entry_date)
                                    
                                    # Passer l'ordre de vente (après sauvegarde SL)
                                    trade_result = trader.smart_trade(symbol, 'SELL')
                                    
                                    trader.disconnect()
                            
                            alert_triggered = True
                            break

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


    def cleanup_data(self):
        """Supprime tous les fichiers dans data/ et chart/"""
        import shutil
        
        folders_to_clean = ['data', 'chart']
        
        for folder in folders_to_clean:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    print(f"Dossier {folder}/ supprimé")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {folder}/: {e}")
            else:
                print(f"Dossier {folder}/ n'existe pas")
        
        print("Nettoyage terminé.")

    def _is_opening_scan(self, symbol, df):
        """
        Détermine si c'est le premier scan du jour (scan d'ouverture) vs scan continu
        
        Utilise scan_status.json pour tracker si on a déjà fait le scan d'ouverture aujourd'hui
        """
        scan_status_file = 'scan_status.json'
        today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
        
        try:
            # Charger le status de scan
            if os.path.exists(scan_status_file):
                with open(scan_status_file, 'r') as f:
                    scan_status = json.load(f)
            else:
                scan_status = {}
            
            # Vérifier si on a déjà fait le scan d'ouverture aujourd'hui
            if scan_status.get('date') < today:
                # Nouvelle journée - nettoyer les données et marquer comme scan d'ouverture
                print(f"🗓️ Nouvelle journée détectée: {today}")
                self.cleanup_data()
                
                # Recréer le dossier data après cleanup
                os.makedirs(self.data_folder, exist_ok=True)
                
                # Marquer comme scanné
                scan_status = {
                    'date': today,
                    'opening_scan_done': True
                }
                
                with open(scan_status_file, 'w') as f:
                    json.dump(scan_status, f, indent=2)
                
                return True  # Premier scan du jour
            
            elif not scan_status.get('opening_scan_done', False):
                # Même jour mais pas encore fait le scan d'ouverture
                scan_status['opening_scan_done'] = True
                
                with open(scan_status_file, 'w') as f:
                    json.dump(scan_status, f, indent=2)
                
                return True  # Premier scan du jour
            else:
                # Déjà fait le scan d'ouverture aujourd'hui
                return False  # Scan continu
                
        except Exception as e:
            print(f"⚠️ Erreur vérification scan status: {e}")
            return False  # En cas de doute, scan continu

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
    parser.add_argument('--allcharts', action='store_true', help='Génère des graphiques pour tous les symboles (utiliser avec --backtest)')
    parser.add_argument('--nasdaq', action='store_true', help='Scanne les symboles NASDAQ 100')
    parser.add_argument('--crypto', action='store_true', help='Scanne les principales cryptomonnaies')
    parser.add_argument('--cleanup', action='store_true', help='Supprime tous les fichiers dans data/ et chart/')
    parser.add_argument('--test_candle', type=int, metavar='N', help='Override test_candle setting (default: read from JSON)')
    args = parser.parse_args()

    if args.cleanup:
        # Create scanner instance just for cleanup
        scanner = StockScanner('settings.json')
        scanner.cleanup_data()
    else:
        # --chart ou --allcharts impliquent automatiquement --backtest
        is_backtest = args.backtest or bool(args.chart) or args.allcharts
        scanner = StockScanner('settings.json', is_backtest=is_backtest, chart_symbol=args.chart, all_charts=args.allcharts, nasdaq=args.nasdaq, crypto=args.crypto, test_candle_override=args.test_candle)
        scanner.run()
