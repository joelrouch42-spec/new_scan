#!/usr/bin/env python3
import json
import os
import subprocess
import asyncio
import signal
import sys
from datetime import datetime, time, timedelta
import time as time_module
import pandas as pd
from zoneinfo import ZoneInfo
from ib_insync import IB, Stock
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import yfinance as yf
import warnings

# Disable logs for ib_insync and yfinance to keep console clean
logging.getLogger('ib_insync').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# Timezone constants
EST = ZoneInfo('America/New_York')

def is_market_open(fake_market=False):
    """
    Vérifie si le marché NYSE est ouvert
    Args: fake_market - Si True, court-circuite la vérification et retourne toujours ouvert
    Returns: (is_open, market_status, next_open_time)
    """
    if fake_market:
        now = datetime.now(EST)
        next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return True, "Fake Market (Always Open)", next_close
    
    now = datetime.now(EST)
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    current_time = now.time()
    
    # Weekend
    if weekday >= 5:  # Saturday=5, Sunday=6
        # Prochain lundi 9:30
        days_until_monday = (7 - weekday) % 7
        if days_until_monday == 0:  # Si c'est dimanche
            days_until_monday = 1
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=days_until_monday)
        return False, "Weekend", next_open
    
    # Heures de trading (9:30-16:00 EST)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    if current_time < market_open:
        # Pre-market
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return False, "Pre-market", next_open
    elif current_time > market_close:
        # After-hours
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
        return False, "After-hours", next_open
    else:
        # Marché ouvert
        next_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return True, "Open", next_close

class StockScanner:
    def __init__(self, settings_file, is_backtest=False, chart_symbol=None, all_charts=False, nasdaq=False, crypto=False, test_candle_override=None):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        # Load trading settings
        try:
            with open('trading_settings.json', 'r') as f:
                self.trading_settings = json.load(f)
        except FileNotFoundError:
            self.trading_settings = {"trading": {"fake_market": False}}

        # Load indicators from settings
        self.indicators_config = self.settings.get('indicators', {})

        # Initialize analyzers based on config
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
        
        # IB connection management for async operations
        self.ib = None
        self.ib_connected = False
        
        # Market status tracking
        self.last_market_check = None
        self.market_was_closed = True
        self.cache_refreshed_today = False


    def get_data_filename(self, symbol, candle_nb, interval, date):
        """Generates data filename"""
        return os.path.join(
            self.data_folder,
            f"{date}_{symbol}_{candle_nb}_{interval}.csv"
        )

    def check_file_exists(self, filepath):
        """Checks if file exists"""
        return os.path.exists(filepath)

    def get_nasdaq100_symbols(self):
        """Returns NASDAQ 100 symbols"""
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
        """Returns Crypto symbols (Ignored based on user request but kept for code stability)"""
        symbols = [
            'BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD', 'MATICUSD',
            'MSTR', 'COIN', 'RIOT', 'MARA', 'CLSK', 'BITF', 'HUT', 'IREN'
        ]
        ibkr_symbols = [{'symbol': s, 'provider': 'IBKR'} for s in symbols]
        return ibkr_symbols

    def check_market_and_refresh_cache(self):
        """
        Vérifie le statut du marché et renouvelle le cache si nécessaire
        """
        fake_market = self.trading_settings.get('trading', {}).get('fake_market', False)
        is_open, status, next_time = is_market_open(fake_market)
        now = datetime.now(EST)
        
        print(f"📊 Market Status: {status}")
        
        # Détection de l'ouverture du marché
        if is_open and self.market_was_closed:
            print("🔔 Marché vient de s'ouvrir - Nettoyage complet du cache!")
            self.cleanup_data()
            os.makedirs(self.data_folder, exist_ok=True)
            self.cache_refreshed_today = True
            self.market_was_closed = False
            
        elif not is_open:
            if status == "Weekend":
                print(f"📅 Weekend - Prochain trading: {next_time.strftime('%A %H:%M')}")
                self.cache_refreshed_today = False  # Reset pour la semaine
            elif status == "Pre-market":
                print(f"🌅 Pre-market - Ouverture à {next_time.strftime('%H:%M')}")
            elif status == "After-hours": 
                print(f"🌙 After-hours - Prochain trading: {next_time.strftime('%A %H:%M')}")
            
            self.market_was_closed = True
            
        return is_open, status, next_time
    

    def check_ibgateway_status(self):
        """Check if IBGateway is running and return status"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if 'ibgateway' in result.stdout.lower():
                status = "🟢 IBGateway: RUNNING"
            else:
                status = "🔴 IBGateway: NOT RUNNING"
            
            print(f"{status}")
            return 'ibgateway' in result.stdout.lower()
        except Exception as e:
            print(f"❌ Error checking IBGateway status: {e}")
            return False

    async def connect_ib(self):
        """Establish and maintain IB connection for realtime operations"""
        try:
            if self.ib is not None and self.ib.isConnected():
                return True
                
            realtime_config = self.settings['realtime']
            host = realtime_config['ibkr_host']
            port = realtime_config['ibkr_port']
            client_id = realtime_config['ibkr_client_id']
            
            self.ib = IB()
            await self.ib.connectAsync(host, port, clientId=client_id)
            
            if self.ib.isConnected():
                self.ib_connected = True
                print(f"🟢 IBKR connected: {host}:{port} (client_id={client_id})")
                return True
            else:
                print(f"❌ IBKR connection failed: {host}:{port}")
                return False
                
        except Exception as e:
            print(f"❌ IBKR connection error: {e}")
            self.ib_connected = False
            return False

    async def disconnect_ib(self):
        """Properly disconnect IB connection"""
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()
            self.ib_connected = False
            print("🔴 IBKR disconnected")

    async def download_ibkr_data_async(self, symbol, candle_nb, interval):
        """Async download using persistent IB connection"""
        try:
            if not self.ib_connected or not self.ib.isConnected():
                if not await self.connect_ib():
                    return None
                    
            contract = Stock(symbol, 'SMART', 'USD')
            try:
                qualified = await self.ib.qualifyContractsAsync(contract)
                if not qualified:
                    print(f"❌ Contract not found for {symbol}")
                    return None
                    
                contract = qualified[0]
            except Exception as e:
                print(f"❌ Contract qualification failed for {symbol}: {e}")
                return None
            
            # Map intervals
            duration_map = {
                '1d': f"{candle_nb} D",
                '1h': f"{max(candle_nb * 60, 1440)} S",
                '30m': f"{max(candle_nb * 30, 1440)} S",
                '15m': f"{max(candle_nb * 15, 1440)} S",
                '5m': f"{max(candle_nb * 5, 1440)} S",
                '1m': f"{max(candle_nb, 1440)} S"
            }
            
            bar_size_map = {
                '1d': '1 day',
                '1h': '1 hour', 
                '30m': '30 mins',
                '15m': '15 mins',
                '5m': '5 mins',
                '1m': '1 min'
            }
            
            duration = duration_map.get(interval, f"{candle_nb} D")
            bar_size = bar_size_map.get(interval, '1 day')
            
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                print(f"❌ No data for {symbol}")
                return None
                
            df = pd.DataFrame([{
                'Date': bar.date,
                'Open': float(bar.open),
                'High': float(bar.high), 
                'Low': float(bar.low),
                'Close': float(bar.close),
                'Volume': int(bar.volume)
            } for bar in bars])
            
            return df
            
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            return None

    def download_crypto_yfinance(self, symbol, candle_nb, interval):
        # Stub for compatibility, removed logic as requested
        return None

    def load_watchlist(self):
        """Loads symbols from config file"""
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
        """Synchronous download for backtesting"""
        try:
            realtime_config = self.settings['realtime']
            host = realtime_config['ibkr_host']
            port = realtime_config['ibkr_port']
            client_id = realtime_config['ibkr_client_id']

            ib = IB()
            ib.connect(host, port, clientId=client_id)
            
            if not ib.isConnected():
                print(f"❌ IBKR Gateway not connected ({host}:{port})")
                return None

            contract = Stock(symbol, 'SMART', 'USD')
            qualified = ib.qualifyContracts(contract)

            if not qualified:
                ib.disconnect()
                return None

            contract = qualified[0]
            duration_str = f"{candle_nb} D"
            bar_size = "1 day"

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

            data = []
            est_tz = EST

            for bar in bars[-candle_nb:]:
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

    def _generate_combined_signals(self, df):
        """Generates signals based on Squeeze Momentum"""
        buy_signals = []
        sell_signals = []
        
        if self.squeeze_analyzer is not None:
            squeeze_result = self.squeeze_analyzer.analyze(df)
            
            for signal in squeeze_result['signals']:
                if signal['type'] == 'bullish':
                    buy_signals.append(signal)
                elif signal['type'] == 'bearish':
                    sell_signals.append(signal)
        
        return buy_signals, sell_signals


    def generate_chart(self, symbol, df):
        """Generates HTML chart with indicators"""
        chart_folder = 'chart'
        os.makedirs(chart_folder, exist_ok=True)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price & Indicators', 'Squeeze Momentum'),
            row_heights=[0.7, 0.3]
        )

        fig.add_trace(go.Candlestick(
            x=df['Date'] if 'Date' in df.columns else list(range(len(df))),
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)

        title_parts = [symbol]

        if self.squeeze_analyzer is not None:
            squeeze_result = self.squeeze_analyzer.analyze(df)
            x_axis = df['Date'] if 'Date' in df.columns else list(range(len(df)))
            
            momentum_values = squeeze_result['momentum']
            momentum_colors = squeeze_result['momentum_colors']
            
            color_map = {'lime': '#00FF00', 'green': '#008000', 'red': '#FF0000', 'maroon': '#800000', 'gray': '#808080'}
            
            for i, (momentum, color) in enumerate(zip(momentum_values, momentum_colors)):
                if not pd.isna(momentum) and i < len(x_axis):
                    fig.add_trace(go.Bar(
                        x=[x_axis[i]],
                        y=[momentum],
                        marker_color=color_map.get(color, color),
                        name='Momentum',
                        showlegend=False,
                        width=86400000 if 'Date' in df.columns else 0.8 
                    ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=[0] * len(x_axis),
                mode='lines',
                line=dict(color='white', width=1),
                name='Zero Line',
                showlegend=False
            ), row=2, col=1)
            
            signals = squeeze_result['signals']
            for signal in signals:
                i = signal['index']
                if i < len(x_axis):
                    if signal['type'] == 'bullish': 
                        if i > 0:
                            sl_level = min(df.iloc[i]['Low'], df.iloc[i-1]['Low'])
                            sl_tolerance = 1 
                            sl_with_tolerance = sl_level * (1 - sl_tolerance / 100)
                            
                            fig.add_shape(type="line", x0=x_axis[max(i-2, 0)], x1=x_axis[i], y0=sl_level, y1=sl_level,
                                line=dict(color="green", width=4, dash="solid"), row=1, col=1)
                            fig.add_shape(type="line", x0=x_axis[max(i-2, 0)], x1=x_axis[i], y0=sl_with_tolerance, y1=sl_with_tolerance,
                                line=dict(color="green", width=2, dash="dash"), row=1, col=1)
                    
                    elif signal['type'] == 'bearish':
                        if i > 0:
                            sl_level = max(df.iloc[i]['High'], df.iloc[i-1]['High'])
                            sl_tolerance = 1
                            sl_with_tolerance = sl_level * (1 + sl_tolerance / 100)
                            
                            fig.add_shape(type="line", x0=x_axis[max(i-2, 0)], x1=x_axis[i], y0=sl_level, y1=sl_level,
                                line=dict(color="red", width=4, dash="solid"), row=1, col=1)
                            fig.add_shape(type="line", x0=x_axis[max(i-2, 0)], x1=x_axis[i], y0=sl_with_tolerance, y1=sl_with_tolerance,
                                line=dict(color="red", width=2, dash="dash"), row=1, col=1)
            
            title_parts.append('SqzMom')

        chart_title = f"{' - '.join(title_parts)}"
        fig.update_layout(
            title=chart_title,
            template='plotly_dark',
            height=900,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        current_price = df.iloc[-1]['Close']
        fig.add_shape(
            type="line", x0=0, x1=1, xref="paper", y0=current_price, y1=current_price, yref="y1",
            line=dict(color="yellow", width=2, dash="dash")
        )
        fig.add_annotation(
            x=1, xref="paper", y=current_price, yref="y1", text=f"${current_price:.2f}",
            showarrow=False, xanchor="left", font=dict(color="yellow", size=14, weight="bold"),
            bgcolor="rgba(0,0,0,0.7)"
        )

        filename = os.path.join(chart_folder, f'{symbol}_indicators.html')
        fig.write_html(filename)
        return filename


    def run_backtest(self):
        """Executes backtest mode"""
        backtest_config = self.settings['backtest']
        candle_nb = backtest_config['candle_nb']
        interval = backtest_config['interval']
        test_start = backtest_config['test_candle_start']
        test_stop = backtest_config['test_candle_stop']

        total_candles_needed = candle_nb + test_stop
        os.makedirs(self.data_folder, exist_ok=True)

        if self.chart_symbol:
            watchlist = [{'symbol': self.chart_symbol, 'provider': 'IBKR'}]
        elif self.nasdaq:
            watchlist = self.get_nasdaq100_symbols()
        elif self.crypto:
            watchlist = self.get_crypto_symbols()
        else:
            print("❌ Error: Specify either --nasdaq, --crypto or --chart")
            return

        print(f"Backtest scanner started - {len(watchlist)} symbols (all_charts: {self.all_charts})\n")

        has_ibkr_symbols = any(item.get('provider', 'IBKR') == 'IBKR' for item in watchlist)
        if has_ibkr_symbols:
            ibgateway_running = self.check_ibgateway_status()
            if not ibgateway_running:
                print("⚠️  Warning: IBGateway not running - IBKR data downloads will fail\n")

        now = datetime.now(EST)
        today = now.strftime('%Y-%m-%d')
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

        print(f"=== Scan {timestamp} ===")

        for item in watchlist:
            symbol = item['symbol']
            provider = item.get('provider', 'IBKR')
            self.current_symbol = symbol 
            filename = self.get_data_filename(symbol, total_candles_needed, interval, today)

            if self.check_file_exists(filename):
                df = pd.read_csv(filename)
            else:
                if provider == 'YFINANCE':
                    df = self.download_crypto_yfinance(symbol, total_candles_needed, interval)
                else:
                    candles_to_request = min(total_candles_needed, 600)
                    df = self.download_ibkr_data(symbol, candles_to_request, interval)
                
                if df is not None:
                    df.to_csv(filename, index=False)
                else:
                    import glob
                    cache_pattern = os.path.join(self.data_folder, f"*_{symbol}_*.csv")
                    cache_files = glob.glob(cache_pattern)
                    
                    if cache_files:
                        latest_cache = max(cache_files, key=os.path.getmtime)
                        df = pd.read_csv(latest_cache)
                    else:
                        continue

            if df is None or len(df) == 0:
                continue

            current_price = df.iloc[-1]['Close']
            alert_triggered = False

            buy_signals, sell_signals = self._generate_combined_signals(df)
            
            total_candles = len(df)
            if test_stop >= total_candles:
                test_stop = total_candles - 1
            
            start_idx = total_candles - test_stop - 1
            end_idx = total_candles - test_start
            
            filtered_buy = [s for s in buy_signals if start_idx <= s['index'] < end_idx]
            filtered_sell = [s for s in sell_signals if start_idx <= s['index'] < end_idx]
            
            total_signals = len(filtered_buy) + len(filtered_sell)

            if total_signals > 0:
                alert_triggered = True
                
            if self.chart_symbol or self.all_charts or alert_triggered:
                # Show only the analyzed range (test_candle_start to test_candle_stop)
                chart_start = total_candles - test_stop - 1
                chart_end = total_candles - test_start
                df_chart_range = df.iloc[chart_start:chart_end].copy().reset_index(drop=True)
                self.generate_chart(symbol, df_chart_range)

        print("\nScan complete.")


    def run_realtime(self):
        """Executes realtime mode with alerts"""
        try:
            realtime_config = self.settings['realtime']
            candle_nb = realtime_config['candle_nb']
            interval = realtime_config['interval']
            update_interval = realtime_config['update_interval_seconds']
            test_candle = self.test_candle_override if self.test_candle_override is not None else realtime_config.get('test_candle', 0)

            os.makedirs(self.data_folder, exist_ok=True)

            if self.nasdaq:
                watchlist = self.get_nasdaq100_symbols()
            elif self.crypto:
                watchlist = self.get_crypto_symbols()
            else:
                print("❌ Error: Specify either --nasdaq or --crypto")
                return

            print(f"Realtime scanner started - {len(watchlist)} symbols every {update_interval}s")
            print("Press Ctrl+C to stop\n")
            
            has_ibkr_symbols = any(item.get('provider', 'IBKR') == 'IBKR' for item in watchlist)
            if has_ibkr_symbols:
                print("🔄 Establishing persistent IB connection...")
                # Connection synchrone IBKR (pas de connexion persistante en mode sync)
                pass

            while True:
                try:
                    now = datetime.now(EST)
                    today = now.strftime('%Y-%m-%d')
                    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    
                    print(f"=== Scan {timestamp} (test_candle: {test_candle}) ===")
                    
                    # Vérifier statut du marché et gérer le cache
                    market_is_open, market_status, next_time = self.check_market_and_refresh_cache()
                    
                    # Si marché fermé, attendre plus longtemps
                    if not market_is_open:
                        if market_status == "Weekend":
                            print(f"💤 Weekend - Pause de 30 minutes")
                            time_module.sleep(1800)  # 30 minutes
                            continue
                        elif market_status in ["Pre-market", "After-hours"]:
                            print(f"💤 {market_status} - Pause de 5 minutes")
                            time_module.sleep(300)  # 5 minutes
                            continue
                    
                    ibgateway_running = self.check_ibgateway_status()
                    if not ibgateway_running:
                        print("⚠️  Warning: IBGateway not running - IBKR data downloads will fail")
    
                    for item in watchlist:
                        # SAFETY PAUSE: Prevent IBKR Pacing Violation (Error 162)
                        # Downloading history for many symbols in a loop is fast.
                        # We add a delay to stay within the 60 requests/10min limit if possible.
                        time_module.sleep(2.0) 
    
                        symbol = item['symbol']
                        provider = item.get('provider', 'IBKR')
                        filename = self.get_data_filename(symbol, candle_nb, interval, today)
    
                        if test_candle == 0:
                            if provider == 'YFINANCE':
                                df = self.download_crypto_yfinance(symbol, candle_nb, interval)
                            else:
                                df = self.download_ibkr_data(symbol, candle_nb, interval)
                            
                            if df is not None:
                                df.to_csv(filename, index=False)
                            else:
                                continue
                        else:
                            import glob
                            cache_pattern = os.path.join(self.data_folder, f"*_{symbol}_*.csv")
                            cache_files = glob.glob(cache_pattern)
                            
                            if cache_files:
                                latest_cache = max(cache_files, key=os.path.getmtime)
                                df = pd.read_csv(latest_cache)
                            else:
                                continue
    
                        if df is None or len(df) == 0:
                            continue
    
                        current_price = df.iloc[-1]['Close']
                        alert_triggered = False
                        is_opening_scan = self._is_opening_scan(symbol, df)
                        
                        if self.mode == 'realtime' and test_candle == 0:
                            # TODO: Fix async trading manager integration
                            # Trading disabled temporarily to avoid event loop conflicts
                            pass
                            if False:  # Disabled
                                from trading_manager import TradingManager
                                trader = TradingManager('settings.json')
                                if trader.connect():
                                    
                                    if is_opening_scan:
                                        # OPENING SCAN: Check strict SL matches to see if we should re-enter or exit
                                        open_price = df.iloc[-1]['Open']
                                        current_prices = {symbol: {'Open': open_price}}
                                        
                                        sl_hits = trader.check_sl_hits_opening(current_prices)
                                        
                                        for sl_hit in sl_hits:
                                            hit_symbol = sl_hit['symbol']
                                            action = sl_hit['action']
                                            
                                            # Generate signals to check if we have a NEW signal in same direction
                                            buy_signals, sell_signals = self._generate_combined_signals(df)
                                            target_candle_idx = len(df) - 1 - test_candle
                                            
                                            has_same_direction_signal = False
                                            
                                            if action == 'BUY':
                                                for signal in buy_signals:
                                                    if signal['index'] == target_candle_idx:
                                                        has_same_direction_signal = True
                                                        signal_idx = signal['index']
                                                        high_price = df.iloc[signal_idx]['High']
                                                        low_price = df.iloc[signal_idx]['Low']
                                                        previous_high = df.iloc[signal_idx - 1]['High'] if signal_idx > 0 else high_price
                                                        previous_low = df.iloc[signal_idx - 1]['Low'] if signal_idx > 0 else low_price
                                                        trader.handle_new_signal(hit_symbol, 'BUY', high_price, low_price, previous_high, previous_low, timestamp)
                                                        print(f"🔄 New BUY signal {hit_symbol} - SL updated")
                                                        break
                                            else:  # SELL
                                                for signal in sell_signals:
                                                    if signal['index'] == target_candle_idx:
                                                        has_same_direction_signal = True
                                                        signal_idx = signal['index']
                                                        high_price = df.iloc[signal_idx]['High']
                                                        low_price = df.iloc[signal_idx]['Low']
                                                        previous_high = df.iloc[signal_idx - 1]['High'] if signal_idx > 0 else high_price
                                                        previous_low = df.iloc[signal_idx - 1]['Low'] if signal_idx > 0 else low_price
                                                        trader.handle_new_signal(hit_symbol, 'SELL', high_price, low_price, previous_high, previous_low, timestamp)
                                                        print(f"🔄 New SELL signal {hit_symbol} - SL updated")
                                                        break
                                            
                                            if not has_same_direction_signal:
                                                # Close position if no new signal confirms the trend
                                                opposite_action = 'SELL' if action == 'BUY' else 'BUY'
                                                trade_result = trader.smart_trade(hit_symbol, opposite_action)
                                                if trade_result:
                                                    print(f"🚨 OPENING SL Hit: {hit_symbol} position closed")
                                                    trader.remove_position_tracking(hit_symbol)
                                
                                else:
                                        # CONTINUOUS SCAN: 
                                        # CRITICAL FIX: Removed manual SL check here. 
                                        # Since we use Bracket Orders (smart_trade_with_bracket), 
                                        # IBKR manages the stop loss. We do NOT want to manually send a Sell order
                                        # while IBKR holds a Stop Sell order, or we risk selling twice.
                                        pass
                                
                                        trader.disconnect()
                        
                        # Generate signals
                        buy_signals, sell_signals = self._generate_combined_signals(df)
                        target_candle_idx = len(df) - 1 - test_candle
    
                        for signal in buy_signals:
                            if signal['index'] == target_candle_idx:
                                signal_date = df.iloc[signal['index']]['Date'] if 'Date' in df.columns else f"Index {signal['index']}"
                                
                                # Get SL level (same logic as chart)
                                i = signal['index']
                                sl_level = min(df.iloc[i]['Low'], df.iloc[i-1]['Low']) if i > 0 else df.iloc[i]['Low']
                                sl_tolerance = 1 
                                sl_with_tolerance = sl_level * (1 - sl_tolerance / 100)
                                
                                print(f"🟢 BUY {symbol} @ ${current_price:.2f} - {signal_date} - SL: ${sl_with_tolerance:.2f}")
                                
                                if self.mode == 'realtime':
                                    # Import et initialise TradingManager pour exécution
                                    try:
                                        from trading_manager import TradingManager
                                        trader = TradingManager('settings.json')
                                        if trader.connect():
                                            result = trader.smart_trade_with_bracket(
                                                symbol, 'BUY',
                                                df.iloc[i]['High'], df.iloc[i]['Low'],
                                                df.iloc[i-1]['High'] if i > 0 else df.iloc[i]['High'], 
                                                df.iloc[i-1]['Low'] if i > 0 else df.iloc[i]['Low'],
                                                current_price_hint=current_price
                                            )
                                            trader.disconnect()
                                            print(f"✅ BUY order executed: {result}")
                                        else:
                                            print("❌ Failed to connect to IBKR")
                                    except Exception as e:
                                        print(f"❌ Trading execution error: {e}")
                                
                                alert_triggered = True
                                break
    
                        for signal in sell_signals:
                            if signal['index'] == target_candle_idx:
                                signal_date = df.iloc[signal['index']]['Date'] if 'Date' in df.columns else f"Index {signal['index']}"
                                
                                # Get SL level (same logic as chart)
                                i = signal['index']
                                sl_level = max(df.iloc[i]['High'], df.iloc[i-1]['High']) if i > 0 else df.iloc[i]['High']
                                sl_tolerance = 1
                                sl_with_tolerance = sl_level * (1 + sl_tolerance / 100)
                                
                                print(f"🔴 SELL {symbol} @ ${current_price:.2f} - {signal_date} - SL: ${sl_with_tolerance:.2f}")
                                
                                if self.mode == 'realtime':
                                    # Import et initialise TradingManager pour exécution
                                    try:
                                        from trading_manager import TradingManager
                                        trader = TradingManager('settings.json')
                                        if trader.connect():
                                            result = trader.smart_trade_with_bracket(
                                                symbol, 'SELL',
                                                df.iloc[i]['High'], df.iloc[i]['Low'],
                                                df.iloc[i-1]['High'] if i > 0 else df.iloc[i]['High'], 
                                                df.iloc[i-1]['Low'] if i > 0 else df.iloc[i]['Low'],
                                                current_price_hint=current_price
                                            )
                                            trader.disconnect()
                                            print(f"✅ SELL order executed: {result}")
                                        else:
                                            print("❌ Failed to connect to IBKR")
                                    except Exception as e:
                                        print(f"❌ Trading execution error: {e}")
                                
                                alert_triggered = True
                                break
    
                        if alert_triggered:
                            # Show only last 50 candles for realtime charts
                            df_chart = df.tail(50).copy().reset_index(drop=True)
                            self.generate_chart(symbol, df_chart)
    
                    print(f"Next scan in {update_interval}s...\n")
                    time_module.sleep(update_interval)
    
                except KeyboardInterrupt:
                    print("\n\nScanner stopped by user")
                    return
                except Exception as e:
                    print(f"Error: {e}")
                    time_module.sleep(update_interval)
                    
        except KeyboardInterrupt:
            print("\n\nScanner stopped by user")
            sys.exit(0)


    def cleanup_data(self):
        """Cleans up data and chart folders"""
        import shutil
        folders_to_clean = ['data', 'chart']
        for folder in folders_to_clean:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    print(f"Folder {folder}/ deleted")
                except Exception as e:
                    print(f"Error deleting {folder}/: {e}")
            else:
                print(f"Folder {folder}/ does not exist")
        print("Cleanup complete.")

    def _is_opening_scan(self, symbol, df):
        """Checks if this is the first scan of the day"""
        scan_status_file = 'scan_status.json'
        today = datetime.now(EST).strftime('%Y-%m-%d')
        
        try:
            if os.path.exists(scan_status_file):
                with open(scan_status_file, 'r') as f:
                    scan_status = json.load(f)
            else:
                scan_status = {}
            
            if scan_status.get('date') is None or scan_status.get('date') < today:
                print(f"🗓️ New day detected: {today}")
                self.cleanup_data()
                os.makedirs(self.data_folder, exist_ok=True)
                
                scan_status = {'date': today, 'opening_scan_done': True}
                with open(scan_status_file, 'w') as f:
                    json.dump(scan_status, f, indent=2)
                return True 
            
            elif not scan_status.get('opening_scan_done', False):
                scan_status['opening_scan_done'] = True
                with open(scan_status_file, 'w') as f:
                    json.dump(scan_status, f, indent=2)
                return True
            else:
                return False 
                
        except Exception as e:
            print(f"⚠️ Error checking scan status: {e}")
            return False

    def run(self):
        """Main entry point"""
        if self.mode == 'backtest':
            self.run_backtest()
        elif self.mode == 'realtime':
            self.run_realtime()
        else:
            print(f"Unknown mode: {self.mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Scanner with Bracket Orders')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--chart', type=str, metavar='SYMBOL', help='Generate chart for symbol')
    parser.add_argument('--allcharts', action='store_true', help='Generate charts for all symbols')
    parser.add_argument('--nasdaq', action='store_true', help='Scan NASDAQ 100')
    parser.add_argument('--crypto', action='store_true', help='Scan Crypto (Ignored)')
    parser.add_argument('--cleanup', action='store_true', help='Delete data/ and chart/ files')
    parser.add_argument('--test_candle', type=int, metavar='N', help='Override test_candle setting')
    args = parser.parse_args()

    if args.cleanup:
        scanner = StockScanner('settings.json')
        scanner.cleanup_data()
    else:
        is_backtest = args.backtest or bool(args.chart) or args.allcharts
        scanner = StockScanner('settings.json', is_backtest=is_backtest, chart_symbol=args.chart, all_charts=args.allcharts, nasdaq=args.nasdaq, crypto=args.crypto, test_candle_override=args.test_candle)
        scanner.run()