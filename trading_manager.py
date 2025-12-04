#!/usr/bin/env python3
import json
import os
from datetime import datetime
from ib_insync import *

class TradingManager:
    """
    Gestionnaire de trading automatique via IBKR
    """
    
    def __init__(self, settings_file, trading_settings_file='trading_settings.json'):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)
        
        with open(trading_settings_file, 'r') as f:
            self.trading_settings = json.load(f)
        
        self.ib = IB()
        self.connected = False
    
    def connect(self):
        """Se connecter à IBKR Gateway"""
        try:
            realtime_config = self.settings['realtime']
            host = realtime_config['ibkr_host']
            port = realtime_config['ibkr_port']
            client_id = realtime_config['ibkr_client_id']
            
            self.ib.connect(host, port, clientId=client_id)
            
            if self.ib.isConnected():
                self.connected = True
                print(f"✅ Connecté à IBKR Gateway ({host}:{port})")
                return True
            else:
                print(f"❌ Échec de connexion IBKR ({host}:{port})")
                return False
                
        except Exception as e:
            print(f"❌ Erreur connexion IBKR: {e}")
            return False
    
    def disconnect(self):
        """Se déconnecter d'IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("🔌 Déconnecté d'IBKR")
    
    def get_account_equity(self):
        """Lire l'equity du compte"""
        if not self.connected:
            print("❌ Pas connecté à IBKR")
            return None
        
        try:
            # Récupérer les informations du compte
            account_values = self.ib.accountValues()
            
            equity = None
            for value in account_values:
                if value.tag == 'NetLiquidation' and value.currency == 'USD':
                    equity = float(value.value)
                    break
            
            if equity is not None:
                print(f"💰 Equity: ${equity:,.2f}")
                
                # Afficher unrealized/realized PNL
                self._display_pnl_summary(account_values)
                
                return equity
            else:
                print("❌ Impossible de lire l'equity")
                return None
                
        except Exception as e:
            print(f"❌ Erreur lecture equity: {e}")
            return None
    
    def get_account_summary(self):
        """Afficher un résumé du compte"""
        if not self.connected:
            print("❌ Pas connecté à IBKR")
            return None
        
        try:
            account_values = self.ib.accountValues()
            
            summary = {}
            for value in account_values:
                if value.currency == 'USD':
                    if value.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 'AvailableFunds']:
                        summary[value.tag] = float(value.value)
            
            print("📊 Résumé du compte:")
            for key, val in summary.items():
                print(f"   {key}: ${val:,.2f}")
            
            return summary
            
        except Exception as e:
            print(f"❌ Erreur lecture compte: {e}")
            return None
    
    def calculate_position_size(self, equity, symbol_price):
        """Calcule la taille de position basée sur trade_percent"""
        trade_percent = self.trading_settings['trading']['trade_percent']
        
        # Montant à investir = equity * trade_percent / 100
        trade_amount = equity * (trade_percent / 100)
        
        # Nombre d'actions = montant / prix par action
        shares = int(trade_amount / symbol_price)
        
        print(f"💹 Calcul position pour {symbol_price}:")
        print(f"   Equity: ${equity:,.2f}")
        print(f"   Trade %: {trade_percent}%")
        print(f"   Montant: ${trade_amount:,.2f}")
        print(f"   Actions: {shares}")
        
        return shares
    
    
    def get_current_positions(self):
        """Récupère les positions actuelles d'IBKR"""
        if not self.connected:
            print("❌ Pas connecté à IBKR")
            return {}
        
        try:
            positions = self.ib.positions()
            current_positions = {}
            
            for position in positions:
                symbol = position.contract.symbol
                shares = int(position.position)
                if shares != 0:  # Ignorer les positions fermées
                    current_positions[symbol] = {
                        'shares': shares,
                        'avgCost': position.avgCost,
                        'marketPrice': position.marketPrice,
                        'marketValue': position.marketValue,
                        'unrealizedPNL': position.unrealizedPNL
                    }
            
            return current_positions
            
        except Exception as e:
            print(f"❌ Erreur lecture positions: {e}")
            return {}
    
    def has_position(self, symbol):
        """Vérifie si on a déjà une position sur ce symbole"""
        positions = self.get_current_positions()
        return symbol in positions
    
    def get_position_size(self, symbol):
        """Retourne le nombre d'actions détenues pour un symbole"""
        positions = self.get_current_positions()
        return positions.get(symbol, {}).get('shares', 0)
    
    def can_trade(self, symbol, action):
        """
        Vérifie dynamiquement si on peut trader (basé sur positions IBKR réelles)
        """
        current_shares = self.get_position_size(symbol)
        
        if action == 'BUY' and current_shares > 0:
            print(f"⚠️ Position longue existante {symbol}: {current_shares} actions")
            return False
        elif action == 'SELL' and current_shares < 0:
            print(f"⚠️ Position courte existante {symbol}: {current_shares} actions")
            return False
        
        return True
    
    def smart_trade(self, symbol, action, price=None):
        """
        API unique pour acheter/vendre avec vérification automatique des positions
        
        Args:
            symbol: Symbole de l'action (ex: 'AAPL')
            action: 'BUY' ou 'SELL'
            price: Prix limite (si None, ordre market)
        """
        if not self.connected:
            print("❌ Pas connecté à IBKR")
            return None
        
        if not self.trading_settings['trading']['enabled']:
            print("❌ Trading désactivé dans trading_settings.json")
            return None
        
        if not self.can_trade(symbol, action):
            return None
        
        try:
            # Créer le contrat
            contract = Stock(symbol, 'SMART', 'USD')
            qualified = self.ib.qualifyContracts(contract)
            
            if not qualified:
                print(f"❌ Contrat invalide pour {symbol}")
                return None
            
            contract = qualified[0]
            
            # Calculer automatiquement les shares selon l'action
            if action == 'BUY':
                # Pour BUY: calculer selon trade_percent
                equity = self.get_account_equity()
                if equity is None:
                    return None
                
                # Obtenir le prix actuel
                ticker = self.ib.reqMktData(contract)
                self.ib.sleep(2)  # Attendre les données
                current_price = ticker.last if ticker.last else ticker.close
                
                if current_price:
                    shares = self.calculate_position_size(equity, current_price)
                else:
                    print(f"❌ Impossible d'obtenir le prix pour {symbol}")
                    return None
                    
            elif action == 'SELL':
                # Pour SELL: vendre toute la position
                shares = self.get_position_size(symbol)
                if shares <= 0:
                    print(f"❌ Aucune position à vendre pour {symbol}")
                    return None
            else:
                print(f"❌ Action invalide: {action}")
                return None
            
            if shares <= 0:
                print(f"❌ Taille de position invalide: {shares}")
                return None
            
            # Créer l'ordre
            order_type = self.trading_settings['trading']['order_settings']['order_type']
            time_in_force = self.trading_settings['trading']['order_settings']['time_in_force']
            
            if order_type == 'MARKET' or price is None:
                order = MarketOrder(action, shares)
            else:
                order = LimitOrder(action, shares, price)
            
            order.tif = time_in_force
            order.outsideRth = self.trading_settings['trading']['order_settings']['outside_rth']
            
            # Passer l'ordre
            trade = self.ib.placeOrder(contract, order)
            
            print(f"📈 Ordre passé:")
            print(f"   Symbole: {symbol}")
            print(f"   Action: {action}")
            print(f"   Quantité: {shares}")
            print(f"   Type: {order_type}")
            print(f"   OrderId: {trade.order.orderId}")
            
            return trade
            
        except Exception as e:
            print(f"❌ Erreur passage d'ordre {symbol}: {e}")
            return None
    
    def _display_pnl_summary(self, account_values):
        """Affiche le résumé des profits/pertes"""
        unrealized_pnl = 0
        realized_pnl = 0
        interest = 0
        
        for value in account_values:
            if value.currency == 'USD':
                if value.tag == 'UnrealizedPnL':
                    unrealized_pnl = float(value.value)
                elif value.tag == 'RealizedPnL':
                    realized_pnl = float(value.value)
                elif value.tag in ['AccruedCash', 'AccruedDividend']:
                    interest += float(value.value)
        
        print(f"📈 Unrealized PnL: ${unrealized_pnl:,.2f}")
        print(f"💵 Realized PnL: ${realized_pnl:,.2f}")
        print(f"💰 Interest/Dividends: ${interest:,.2f}")
        
        total_pnl = unrealized_pnl + realized_pnl
        color = "🟢" if total_pnl >= 0 else "🔴"
        print(f"{color} Trading PnL: ${total_pnl:,.2f}")
        
        total_with_interest = total_pnl + interest
        color_total = "🟢" if total_with_interest >= 0 else "🔴"
        print(f"{color_total} Total (with interest): ${total_with_interest:,.2f}")


# Test du module
if __name__ == '__main__':
    trader = TradingManager('settings.json')
    
    if trader.connect():
        trader.get_account_equity()
        trader.get_account_summary()
        trader.disconnect()