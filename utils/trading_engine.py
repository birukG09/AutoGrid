import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

class EnergyTradingEngine:
    def __init__(self):
        self.active_trades = []
        self.trade_history = []
        self.market_price = 0.15  # $/kWh base price
        self.price_volatility = 0.02
        self.participants = [
            {'id': 'GRID_01', 'name': 'Main Grid', 'type': 'utility', 'capacity': 1000},
            {'id': 'SOLAR_01', 'name': 'Solar Farm A', 'type': 'generator', 'capacity': 100},
            {'id': 'WIND_01', 'name': 'Wind Farm B', 'type': 'generator', 'capacity': 75},
            {'id': 'BATTERY_01', 'name': 'Community Storage', 'type': 'storage', 'capacity': 50},
            {'id': 'INDUSTRIAL_01', 'name': 'Factory Complex', 'type': 'consumer', 'capacity': 200},
            {'id': 'RESIDENTIAL_01', 'name': 'Neighborhood A', 'type': 'consumer', 'capacity': 150}
        ]
        
    def get_current_market_price(self):
        """Get current market price with volatility"""
        # Simulate price changes based on time and random factors
        hour = datetime.now().hour
        
        # Higher prices during peak hours
        time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 12)
        
        # Add random volatility
        volatility_factor = 1.0 + np.random.normal(0, self.price_volatility)
        
        current_price = self.market_price * time_factor * volatility_factor
        return max(0.05, current_price)  # Minimum price floor
    
    def create_energy_offers(self, current_data):
        """Generate energy trading offers based on current grid status"""
        offers = []
        current_price = self.get_current_market_price()
        
        # Generation offers (sellers)
        if current_data['solar_generation'] > 0:
            offers.append({
                'id': str(uuid.uuid4())[:8],
                'participant_id': 'SOLAR_01',
                'participant_name': 'Solar Farm A',
                'type': 'sell',
                'energy_amount': current_data['solar_generation'] * 0.8,  # Keep some for own use
                'price_per_kwh': current_price * 0.9,  # Slightly below market
                'duration_hours': 1,
                'timestamp': datetime.now(),
                'status': 'active'
            })
        
        if current_data['wind_generation'] > 0:
            offers.append({
                'id': str(uuid.uuid4())[:8],
                'participant_id': 'WIND_01', 
                'participant_name': 'Wind Farm B',
                'type': 'sell',
                'energy_amount': current_data['wind_generation'] * 0.7,
                'price_per_kwh': current_price * 0.95,
                'duration_hours': 1,
                'timestamp': datetime.now(),
                'status': 'active'
            })
        
        # Battery offers (can buy or sell)
        if current_data['battery_soc'] > 80:
            offers.append({
                'id': str(uuid.uuid4())[:8],
                'participant_id': 'BATTERY_01',
                'participant_name': 'Community Storage',
                'type': 'sell',
                'energy_amount': 15,
                'price_per_kwh': current_price * 1.1,  # Premium for storage
                'duration_hours': 2,
                'timestamp': datetime.now(),
                'status': 'active'
            })
        elif current_data['battery_soc'] < 40:
            offers.append({
                'id': str(uuid.uuid4())[:8],
                'participant_id': 'BATTERY_01',
                'participant_name': 'Community Storage', 
                'type': 'buy',
                'energy_amount': 20,
                'price_per_kwh': current_price * 0.8,  # Below market for storage
                'duration_hours': 3,
                'timestamp': datetime.now(),
                'status': 'active'
            })
        
        # Demand offers (buyers)
        if current_data['total_demand'] > current_data['total_generation']:
            deficit = current_data['total_demand'] - current_data['total_generation']
            
            offers.append({
                'id': str(uuid.uuid4())[:8],
                'participant_id': 'INDUSTRIAL_01',
                'participant_name': 'Factory Complex',
                'type': 'buy',
                'energy_amount': min(deficit * 0.6, 25),
                'price_per_kwh': current_price * 1.05,
                'duration_hours': 1,
                'timestamp': datetime.now(),
                'status': 'active'
            })
            
            offers.append({
                'id': str(uuid.uuid4())[:8],
                'participant_id': 'RESIDENTIAL_01',
                'participant_name': 'Neighborhood A',
                'type': 'buy',
                'energy_amount': min(deficit * 0.4, 15),
                'price_per_kwh': current_price,
                'duration_hours': 2,
                'timestamp': datetime.now(),
                'status': 'active'
            })
        
        return offers
    
    def match_trades(self, offers):
        """Match buy and sell offers to create trades"""
        sell_offers = [o for o in offers if o['type'] == 'sell']
        buy_offers = [o for o in offers if o['type'] == 'buy']
        
        # Sort sell offers by price (ascending)
        sell_offers.sort(key=lambda x: x['price_per_kwh'])
        
        # Sort buy offers by price (descending)
        buy_offers.sort(key=lambda x: x['price_per_kwh'], reverse=True)
        
        trades = []
        
        for buy_offer in buy_offers:
            remaining_demand = buy_offer['energy_amount']
            
            for sell_offer in sell_offers:
                if (sell_offer['price_per_kwh'] <= buy_offer['price_per_kwh'] and 
                    sell_offer['energy_amount'] > 0 and remaining_demand > 0):
                    
                    # Calculate trade amount
                    trade_amount = min(remaining_demand, sell_offer['energy_amount'])
                    
                    # Create trade
                    trade = {
                        'id': str(uuid.uuid4())[:8],
                        'buyer_id': buy_offer['participant_id'],
                        'buyer_name': buy_offer['participant_name'],
                        'seller_id': sell_offer['participant_id'],
                        'seller_name': sell_offer['participant_name'],
                        'energy_amount': trade_amount,
                        'price_per_kwh': (buy_offer['price_per_kwh'] + sell_offer['price_per_kwh']) / 2,
                        'total_value': trade_amount * ((buy_offer['price_per_kwh'] + sell_offer['price_per_kwh']) / 2),
                        'timestamp': datetime.now(),
                        'status': 'executed'
                    }
                    
                    trades.append(trade)
                    
                    # Update offer amounts
                    sell_offer['energy_amount'] -= trade_amount
                    remaining_demand -= trade_amount
                    
                    if remaining_demand <= 0:
                        break
        
        return trades
    
    def simulate_trading_session(self, current_data):
        """Simulate a complete trading session"""
        
        # Generate offers
        offers = self.create_energy_offers(current_data)
        
        # Match trades
        new_trades = self.match_trades(offers)
        
        # Add to history
        self.trade_history.extend(new_trades)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.trade_history = [t for t in self.trade_history if t['timestamp'] > cutoff_time]
        
        return {
            'offers': offers,
            'executed_trades': new_trades,
            'market_summary': self._get_market_summary(new_trades)
        }
    
    def _get_market_summary(self, trades):
        """Generate market summary statistics"""
        if not trades:
            return {
                'total_volume': 0,
                'total_value': 0,
                'average_price': self.get_current_market_price(),
                'num_trades': 0
            }
        
        total_volume = sum(t['energy_amount'] for t in trades)
        total_value = sum(t['total_value'] for t in trades)
        average_price = total_value / total_volume if total_volume > 0 else 0
        
        return {
            'total_volume': total_volume,
            'total_value': total_value,
            'average_price': average_price,
            'num_trades': len(trades)
        }
    
    def get_trading_analytics(self):
        """Get comprehensive trading analytics"""
        if not self.trade_history:
            return {'status': 'no_data'}
        
        recent_trades = [t for t in self.trade_history if t['timestamp'] > datetime.now() - timedelta(hours=24)]
        
        # Price analytics
        prices = [t['price_per_kwh'] for t in recent_trades]
        volumes = [t['energy_amount'] for t in recent_trades]
        
        # Participant analytics
        participant_stats = {}
        for trade in recent_trades:
            # Buyer stats
            buyer_id = trade['buyer_id']
            if buyer_id not in participant_stats:
                participant_stats[buyer_id] = {'bought': 0, 'sold': 0, 'trades': 0}
            participant_stats[buyer_id]['bought'] += trade['energy_amount']
            participant_stats[buyer_id]['trades'] += 1
            
            # Seller stats
            seller_id = trade['seller_id']
            if seller_id not in participant_stats:
                participant_stats[seller_id] = {'bought': 0, 'sold': 0, 'trades': 0}
            participant_stats[seller_id]['sold'] += trade['energy_amount']
            participant_stats[seller_id]['trades'] += 1
        
        analytics = {
            'status': 'success',
            'market_metrics': {
                'total_trades': len(recent_trades),
                'total_volume': sum(volumes),
                'total_value': sum(t['total_value'] for t in recent_trades),
                'average_price': np.mean(prices) if prices else 0,
                'price_volatility': np.std(prices) if len(prices) > 1 else 0,
                'min_price': min(prices) if prices else 0,
                'max_price': max(prices) if prices else 0
            },
            'participant_stats': participant_stats,
            'hourly_volume': self._get_hourly_volume(recent_trades)
        }
        
        return analytics
    
    def _get_hourly_volume(self, trades):
        """Calculate trading volume by hour"""
        hourly_data = {}
        
        for trade in trades:
            hour = trade['timestamp'].hour
            if hour not in hourly_data:
                hourly_data[hour] = 0
            hourly_data[hour] += trade['energy_amount']
        
        return hourly_data
    
    def get_market_participants(self):
        """Get list of market participants with current status"""
        current_price = self.get_current_market_price()
        
        participants_with_status = []
        for participant in self.participants:
            # Add some dynamic status
            status = np.random.choice(['active', 'active', 'active', 'maintenance'], p=[0.7, 0.2, 0.08, 0.02])
            
            participants_with_status.append({
                **participant,
                'status': status,
                'last_trade_price': current_price * np.random.uniform(0.9, 1.1),
                'current_offer_count': np.random.randint(0, 5)
            })
        
        return participants_with_status
    
    def create_trade_order(self, participant_id, order_type, energy_amount, price_per_kwh, duration_hours):
        """Create a new trade order"""
        
        participant = next((p for p in self.participants if p['id'] == participant_id), None)
        if not participant:
            return {'status': 'error', 'message': 'Invalid participant ID'}
        
        order = {
            'id': str(uuid.uuid4())[:8],
            'participant_id': participant_id,
            'participant_name': participant['name'],
            'type': order_type,
            'energy_amount': energy_amount,
            'price_per_kwh': price_per_kwh,
            'duration_hours': duration_hours,
            'timestamp': datetime.now(),
            'status': 'pending'
        }
        
        self.active_trades.append(order)
        
        return {
            'status': 'success',
            'order_id': order['id'],
            'message': f"Order created successfully for {energy_amount} kWh at ${price_per_kwh:.3f}/kWh"
        }
