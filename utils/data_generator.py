import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class MicrogridDataGenerator:
    def __init__(self):
        self.base_time = datetime.now()
        self.solar_base = 25  # Base solar generation
        self.wind_base = 15   # Base wind generation
        self.battery_soc = 75  # Battery state of charge
        self.previous_data = None
        self.events_log = []
        
    def get_time_factor(self):
        """Get time-based factors for solar and wind generation"""
        current_time = datetime.now()
        hour = current_time.hour
        
        # Solar factor (peak at noon)
        solar_factor = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        
        # Wind factor (more variable, slightly higher at night)
        wind_factor = 0.3 + 0.7 * (1 + np.sin(hour * np.pi / 12 + np.pi)) / 2
        
        return solar_factor, wind_factor
    
    def get_demand_factor(self):
        """Get time-based demand factors"""
        current_time = datetime.now()
        hour = current_time.hour
        
        # Residential demand (peaks in evening)
        residential_factor = 0.4 + 0.6 * (1 + np.sin((hour - 6) * np.pi / 12)) / 2
        
        # Commercial demand (peaks during business hours)
        commercial_factor = 0.8 if 8 <= hour <= 17 else 0.3
        
        # Industrial demand (more constant)
        industrial_factor = 0.9 + 0.1 * np.sin(hour * np.pi / 12)
        
        # EV charging (peaks in evening and early morning)
        ev_factor = 0.9 if (17 <= hour <= 23) or (6 <= hour <= 8) else 0.2
        
        return residential_factor, commercial_factor, industrial_factor, ev_factor
    
    def get_current_status(self):
        """Generate current microgrid status"""
        solar_factor, wind_factor = self.get_time_factor()
        residential_factor, commercial_factor, industrial_factor, ev_factor = self.get_demand_factor()
        
        # Add some randomness
        solar_noise = np.random.normal(0, 0.1)
        wind_noise = np.random.normal(0, 0.2)
        
        # Generate power values
        solar_generation = max(0, self.solar_base * solar_factor * (1 + solar_noise))
        wind_generation = max(0, self.wind_base * wind_factor * (1 + wind_noise))
        
        # Loads
        residential_load = 20 * residential_factor * (1 + np.random.normal(0, 0.05))
        commercial_load = 25 * commercial_factor * (1 + np.random.normal(0, 0.05))
        industrial_load = 35 * industrial_factor * (1 + np.random.normal(0, 0.03))
        ev_load = 15 * ev_factor * (1 + np.random.normal(0, 0.1))
        
        total_demand = residential_load + commercial_load + industrial_load + ev_load
        renewable_generation = solar_generation + wind_generation
        
        # Battery management
        net_renewable = renewable_generation - total_demand
        battery_discharge = 0
        battery_charge = 0
        
        if net_renewable < 0:  # Need more power
            if self.battery_soc > 20:
                battery_discharge = min(abs(net_renewable), 20, self.battery_soc * 0.5)
                self.battery_soc -= battery_discharge / 100 * 2  # Simplified battery model
        else:  # Excess power
            if self.battery_soc < 95:
                battery_charge = min(net_renewable, 20, (100 - self.battery_soc) * 0.5)
                self.battery_soc += battery_charge / 100 * 2
        
        total_generation = renewable_generation + battery_discharge
        net_flow = total_generation - total_demand
        
        # Calculate changes from previous data
        generation_change = 0
        demand_change = 0
        battery_change = 0
        
        if self.previous_data:
            generation_change = total_generation - self.previous_data['total_generation']
            demand_change = total_demand - self.previous_data['total_demand']
            battery_change = self.battery_soc - self.previous_data['battery_soc']
        
        current_data = {
            'timestamp': datetime.now(),
            'solar_generation': solar_generation,
            'wind_generation': wind_generation,
            'battery_discharge': battery_discharge,
            'battery_charge': battery_charge,
            'battery_soc': self.battery_soc,
            'residential_load': residential_load,
            'commercial_load': commercial_load,
            'industrial_load': industrial_load,
            'ev_load': ev_load,
            'total_generation': total_generation,
            'total_demand': total_demand,
            'net_flow': net_flow,
            'generation_change': generation_change,
            'demand_change': demand_change,
            'battery_change': battery_change
        }
        
        # Log significant events
        self._log_events(current_data)
        
        self.previous_data = current_data
        return current_data
    
    def _log_events(self, data):
        """Log significant system events"""
        events = []
        
        if data['battery_soc'] < 20:
            events.append("Low battery warning")
        elif data['battery_soc'] > 95:
            events.append("Battery fully charged")
        
        if abs(data['net_flow']) > 10:
            if data['net_flow'] > 0:
                events.append("Excess energy - exporting to grid")
            else:
                events.append("Energy deficit - importing from grid")
        
        if data['solar_generation'] < 2 and 10 <= datetime.now().hour <= 16:
            events.append("Low solar generation detected")
        
        for event in events:
            self.events_log.append({
                'timestamp': data['timestamp'],
                'event': event,
                'severity': 'Info'
            })
        
        # Keep only recent events
        self.events_log = self.events_log[-50:]
    
    def get_recent_events(self):
        """Get recent system events as DataFrame"""
        if not self.events_log:
            return pd.DataFrame({'Timestamp': [], 'Event': [], 'Severity': []})
        
        events_df = pd.DataFrame(self.events_log)
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        events_df = events_df.sort_values('timestamp', ascending=False).head(10)
        
        events_df = events_df.rename(columns={
            'timestamp': 'Timestamp',
            'event': 'Event', 
            'severity': 'Severity'
        })
        return events_df
    
    def get_historical_data(self, hours=24):
        """Generate historical data for the specified number of hours"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')
        data = []
        
        for ts in timestamps:
            hour = ts.hour
            
            # Time-based factors
            solar_factor = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            wind_factor = 0.3 + 0.7 * (1 + np.sin(hour * np.pi / 12 + np.pi)) / 2
            
            # Demand factors
            residential_factor = 0.4 + 0.6 * (1 + np.sin((hour - 6) * np.pi / 12)) / 2
            commercial_factor = 0.8 if 8 <= hour <= 17 else 0.3
            industrial_factor = 0.9 + 0.1 * np.sin(hour * np.pi / 12)
            ev_factor = 0.9 if (17 <= hour <= 23) or (6 <= hour <= 8) else 0.2
            
            # Add noise
            solar_noise = np.random.normal(0, 0.1)
            wind_noise = np.random.normal(0, 0.2)
            
            data.append({
                'timestamp': ts,
                'solar_generation': max(0, self.solar_base * solar_factor * (1 + solar_noise)),
                'wind_generation': max(0, self.wind_base * wind_factor * (1 + wind_noise)),
                'residential_load': 20 * residential_factor * (1 + np.random.normal(0, 0.05)),
                'commercial_load': 25 * commercial_factor * (1 + np.random.normal(0, 0.05)),
                'industrial_load': 35 * industrial_factor * (1 + np.random.normal(0, 0.03)),
                'ev_load': 15 * ev_factor * (1 + np.random.normal(0, 0.1))
            })
        
        df = pd.DataFrame(data)
        df['total_generation'] = df['solar_generation'] + df['wind_generation']
        df['total_demand'] = df['residential_load'] + df['commercial_load'] + df['industrial_load'] + df['ev_load']
        df['net_flow'] = df['total_generation'] - df['total_demand']
        
        return df
