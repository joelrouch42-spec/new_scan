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
        self.positions_tracking_file = 'position_tracking.json'
    
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
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Erreur connexion IBKR: {e}")
            return False
    
    def disconnect(self):
        """Se déconnecter d'IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
    
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
            
            # Vérifier si trading activé JUSTE avant de passer l'ordre
            if not self.trading_settings['trading']['enabled']:
                print("❌ Trading désactivé - ordre simulé")
                return True  # Retourner True pour continuer le logging
            
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
    
    def save_position_tracking(self, symbol, action, sl_level, entry_date):
        """Sauvegarde une nouvelle position avec son SL dans le fichier JSON"""
        try:
            # Charger le fichier existant ou créer nouveau dict
            if os.path.exists(self.positions_tracking_file):
                with open(self.positions_tracking_file, 'r') as f:
                    positions = json.load(f)
            else:
                positions = {}
            
            # Ajouter/mettre à jour la position
            positions[symbol] = {
                'action': action,
                'sl_level': sl_level,
                'entry_date': entry_date
            }
            
            # Sauvegarder
            with open(self.positions_tracking_file, 'w') as f:
                json.dump(positions, f, indent=2)
            
            print(f"📝 Position trackée: {symbol} {action} SL={sl_level:.2f}")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde position {symbol}: {e}")
    
    def update_sl_tracking(self, symbol, new_sl_level):
        """Met à jour le SL d'une position existante"""
        try:
            if not os.path.exists(self.positions_tracking_file):
                return False
            
            with open(self.positions_tracking_file, 'r') as f:
                positions = json.load(f)
            
            if symbol in positions:
                positions[symbol]['sl_level'] = new_sl_level
                
                with open(self.positions_tracking_file, 'w') as f:
                    json.dump(positions, f, indent=2)
                
                print(f"🔄 SL mis à jour: {symbol} nouveau SL={new_sl_level:.2f}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Erreur mise à jour SL {symbol}: {e}")
            return False
    
    def check_sl_hits_continuous(self, current_prices):
        """
        Scan CONTINU: Protection crash avec tolérance - fermeture immédiate
        
        Args:
            current_prices: dict {symbol: {'Current': prix_temps_reel}}
            
        Returns:
            List des positions à fermer immédiatement
        """
        sl_hits = []
        
        try:
            if not os.path.exists(self.positions_tracking_file):
                return sl_hits
            
            with open(self.positions_tracking_file, 'r') as f:
                tracked_positions = json.load(f)
            
            ibkr_positions = self.get_current_positions()
            sl_tolerance = self.trading_settings['trading'].get('sl_tolerance_percent', 1)
            
            for symbol, tracking_info in tracked_positions.items():
                if symbol not in ibkr_positions or symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]['Current']
                sl_level = tracking_info['sl_level']
                action = tracking_info['action']
                
                # SL AVEC tolérance pour protection crash
                if action == 'BUY':
                    sl_with_tolerance = sl_level * (1 - sl_tolerance / 100)
                    sl_hit = current_price <= sl_with_tolerance
                else:  # SELL
                    sl_with_tolerance = sl_level * (1 + sl_tolerance / 100)
                    sl_hit = current_price >= sl_with_tolerance
                
                if sl_hit:
                    sl_hits.append({
                        'symbol': symbol,
                        'action': action,
                        'current_price': current_price,
                        'sl_level': sl_level,
                        'sl_with_tolerance': sl_with_tolerance,
                        'reason': 'crash_protection'
                    })
                    print(f"🚨 CRASH PROTECTION: {symbol} {action} prix={current_price:.2f} SL={sl_with_tolerance:.2f}")
            
            return sl_hits
            
        except Exception as e:
            print(f"❌ Erreur scan continu SL: {e}")
            return []
    
    def check_sl_hits_opening(self, current_prices):
        """
        Scan OUVERTURE: Nouvelle bougie avec SL exact - vérification signaux
        
        Args:
            current_prices: dict {symbol: {'Open': prix_ouverture}}
            
        Returns:
            List des positions où SL exact touché (pour vérification signaux)
        """
        sl_hits = []
        
        try:
            if not os.path.exists(self.positions_tracking_file):
                return sl_hits
            
            with open(self.positions_tracking_file, 'r') as f:
                tracked_positions = json.load(f)
            
            ibkr_positions = self.get_current_positions()
            
            for symbol, tracking_info in tracked_positions.items():
                if symbol not in ibkr_positions or symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]['Open']
                sl_level = tracking_info['sl_level']
                action = tracking_info['action']
                
                # SL EXACT (sans tolérance) pour scan ouverture
                if action == 'BUY':
                    sl_hit = current_price <= sl_level
                else:  # SELL
                    sl_hit = current_price >= sl_level
                
                if sl_hit:
                    sl_hits.append({
                        'symbol': symbol,
                        'action': action,
                        'current_price': current_price,
                        'sl_level': sl_level,
                        'reason': 'opening_check'
                    })
                    print(f"📊 OUVERTURE SL touché: {symbol} {action} prix={current_price:.2f} SL exact={sl_level:.2f}")
            
            return sl_hits
            
        except Exception as e:
            print(f"❌ Erreur scan ouverture SL: {e}")
            return []
    
    def calculate_sl_level(self, symbol, action, high_price, low_price, previous_high, previous_low):
        """
        Calcule le niveau de SL selon la logique définie:
        - BUY: max(high de la bougie signal, high de la bougie précédente)  
        - SELL: min(low de la bougie signal, low de la bougie précédente)
        
        Args:
            symbol: Symbole de l'action
            action: 'BUY' ou 'SELL'
            high_price: High de la bougie du signal
            low_price: Low de la bougie du signal
            previous_high: High de la bougie précédente
            previous_low: Low de la bougie précédente
            
        Returns:
            float: Niveau de SL calculé
        """
        try:
            if action == 'BUY':
                # Pour position longue: SL = max(high signal, high précédent)
                sl_level = max(high_price, previous_high)
            elif action == 'SELL':
                # Pour position courte: SL = min(low signal, low précédent) 
                sl_level = min(low_price, previous_low)
            else:
                print(f"❌ Action invalide pour calcul SL: {action}")
                return None
            
            print(f"💹 SL calculé pour {symbol} {action}: {sl_level:.2f}")
            return sl_level
            
        except Exception as e:
            print(f"❌ Erreur calcul SL {symbol}: {e}")
            return None
    
    def handle_new_signal(self, symbol, action, high_price, low_price, previous_high, previous_low, entry_date):
        """
        Gère un nouveau signal: sauvegarde nouvelle position OU met à jour SL existant
        
        Args:
            symbol: Symbole de l'action
            action: 'BUY' ou 'SELL' 
            high_price: High de la bougie du signal
            low_price: Low de la bougie du signal
            previous_high: High de la bougie précédente
            previous_low: Low de la bougie précédente
            entry_date: Date du signal
        """
        try:
            # Calculer le nouveau niveau de SL
            new_sl_level = self.calculate_sl_level(symbol, action, high_price, low_price, previous_high, previous_low)
            if new_sl_level is None:
                return
                
            # Vérifier si position déjà trackée
            if os.path.exists(self.positions_tracking_file):
                with open(self.positions_tracking_file, 'r') as f:
                    positions = json.load(f)
                    
                if symbol in positions and positions[symbol]['action'] == action:
                    # Signal dans le même sens: mettre à jour SL
                    old_sl = positions[symbol]['sl_level']
                    
                    # Mettre à jour seulement si nouveau SL est plus favorable
                    should_update = False
                    if action == 'BUY' and new_sl_level > old_sl:
                        should_update = True  # SL plus haut pour position longue
                    elif action == 'SELL' and new_sl_level < old_sl:
                        should_update = True  # SL plus bas pour position courte
                    
                    if should_update:
                        self.update_sl_tracking(symbol, new_sl_level)
                        print(f"🔄 SL mis à jour: {symbol} {old_sl:.2f} → {new_sl_level:.2f}")
                    else:
                        print(f"↔️ SL inchangé: {symbol} nouveau={new_sl_level:.2f} vs ancien={old_sl:.2f}")
                    return
            
            # Nouvelle position: sauvegarder
            self.save_position_tracking(symbol, action, new_sl_level, entry_date)
            
        except Exception as e:
            print(f"❌ Erreur gestion nouveau signal {symbol}: {e}")
    
    def remove_position_tracking(self, symbol):
        """
        Supprime une position du tracking (quand fermée)
        
        Args:
            symbol: Symbole à supprimer
        """
        try:
            if not os.path.exists(self.positions_tracking_file):
                return False
            
            with open(self.positions_tracking_file, 'r') as f:
                positions = json.load(f)
            
            if symbol in positions:
                del positions[symbol]
                
                with open(self.positions_tracking_file, 'w') as f:
                    json.dump(positions, f, indent=2)
                
                print(f"🗑️ Position supprimée du tracking: {symbol}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Erreur suppression tracking {symbol}: {e}")
            return False


# Test du module
if __name__ == '__main__':
    trader = TradingManager('settings.json')
    
    if trader.connect():
        trader.get_account_equity()
        trader.get_account_summary()
        trader.disconnect()