import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import joblib

class DemandForecaster:
    def __init__(self):
        self.demand_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.generation_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature', 'humidity', 'cloud_cover'
        ]
        
    def create_features(self, timestamps):
        """Create features from timestamps and weather data"""
        features = []
        
        for ts in timestamps:
            # Simulate weather data (in real implementation, this would come from weather API)
            temp = 20 + 10 * np.sin(2 * np.pi * ts.hour / 24) + np.random.normal(0, 2)
            humidity = 50 + 20 * np.sin(2 * np.pi * ts.hour / 24 + np.pi/4) + np.random.normal(0, 5)
            cloud_cover = max(0, min(100, 30 + 40 * np.sin(2 * np.pi * ts.hour / 24 + np.pi) + np.random.normal(0, 15)))
            
            features.append([
                ts.hour,
                ts.weekday(),
                ts.month,
                1 if ts.weekday() >= 5 else 0,  # is_weekend
                temp,
                humidity,
                cloud_cover
            ])
        
        return np.array(features)
    
    def train_models(self, historical_data):
        """Train forecasting models on historical data"""
        
        # Create features
        features = self.create_features(historical_data['timestamp'])
        
        # Prepare targets
        demand_target = historical_data['total_demand'].values
        generation_target = historical_data['total_generation'].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train models
        self.demand_model.fit(features_scaled, demand_target)
        self.generation_model.fit(features_scaled, generation_target)
        
        self.is_trained = True
        
        # Calculate training metrics
        demand_pred = self.demand_model.predict(features_scaled)
        generation_pred = self.generation_model.predict(features_scaled)
        
        metrics = {
            'demand_mae': mean_absolute_error(demand_target, demand_pred),
            'demand_rmse': np.sqrt(mean_squared_error(demand_target, demand_pred)),
            'generation_mae': mean_absolute_error(generation_target, generation_pred),
            'generation_rmse': np.sqrt(mean_squared_error(generation_target, generation_pred))
        }
        
        return metrics
    
    def forecast_demand(self, hours_ahead=24):
        """Forecast demand for the next specified hours"""
        
        if not self.is_trained:
            # Generate synthetic training data if not trained
            self._train_with_synthetic_data()
        
        # Generate future timestamps
        start_time = datetime.now()
        future_times = [start_time + timedelta(hours=h) for h in range(1, hours_ahead + 1)]
        
        # Create features for future times
        future_features = self.create_features(future_times)
        future_features_scaled = self.scaler.transform(future_features)
        
        # Make predictions
        demand_forecast = self.demand_model.predict(future_features_scaled)
        generation_forecast = self.generation_model.predict(future_features_scaled)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'timestamp': future_times,
            'predicted_demand': demand_forecast,
            'predicted_generation': generation_forecast,
            'predicted_net_flow': generation_forecast - demand_forecast,
            'hour': [t.hour for t in future_times],
            'day_of_week': [t.weekday() for t in future_times]
        })
        
        # Add confidence intervals (simplified)
        forecast_df['demand_lower'] = demand_forecast * 0.9
        forecast_df['demand_upper'] = demand_forecast * 1.1
        forecast_df['generation_lower'] = generation_forecast * 0.8
        forecast_df['generation_upper'] = generation_forecast * 1.2
        
        return forecast_df
    
    def _train_with_synthetic_data(self):
        """Train models with synthetic data if no historical data is available"""
        
        # Generate 7 days of synthetic historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
        
        synthetic_data = []
        for ts in timestamps:
            hour = ts.hour
            
            # Synthetic patterns
            solar_factor = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            wind_factor = 0.3 + 0.7 * (1 + np.sin(hour * np.pi / 12 + np.pi)) / 2
            demand_factor = 0.6 + 0.4 * (1 + np.sin((hour - 6) * np.pi / 12)) / 2
            
            # Add weekly patterns
            if ts.weekday() >= 5:  # Weekend
                demand_factor *= 0.8
            
            synthetic_data.append({
                'timestamp': ts,
                'total_demand': 60 * demand_factor * (1 + np.random.normal(0, 0.1)),
                'total_generation': (25 * solar_factor + 15 * wind_factor) * (1 + np.random.normal(0, 0.1))
            })
        
        synthetic_df = pd.DataFrame(synthetic_data)
        self.train_models(synthetic_df)
    
    def detect_anomalies(self, current_data, threshold=2.0):
        """Detect anomalies in current data compared to predictions"""
        
        if not self.is_trained:
            return {'anomalies': [], 'scores': {}}
        
        # Get current prediction
        current_time = datetime.now()
        features = self.create_features([current_time])
        features_scaled = self.scaler.transform(features)
        
        predicted_demand = self.demand_model.predict(features_scaled)[0]
        predicted_generation = self.generation_model.predict(features_scaled)[0]
        
        # Calculate anomaly scores
        demand_score = abs(current_data['total_demand'] - predicted_demand) / predicted_demand
        generation_score = abs(current_data['total_generation'] - predicted_generation) / max(predicted_generation, 0.1)
        
        anomalies = []
        
        if demand_score > threshold:
            anomalies.append({
                'type': 'demand',
                'severity': 'High' if demand_score > 3.0 else 'Medium',
                'description': f"Demand anomaly detected: {current_data['total_demand']:.1f} kW vs predicted {predicted_demand:.1f} kW",
                'score': demand_score
            })
        
        if generation_score > threshold:
            anomalies.append({
                'type': 'generation',
                'severity': 'High' if generation_score > 3.0 else 'Medium', 
                'description': f"Generation anomaly detected: {current_data['total_generation']:.1f} kW vs predicted {predicted_generation:.1f} kW",
                'score': generation_score
            })
        
        return {
            'anomalies': anomalies,
            'scores': {
                'demand_score': demand_score,
                'generation_score': generation_score,
                'predicted_demand': predicted_demand,
                'predicted_generation': predicted_generation
            }
        }
    
    def get_forecast_accuracy(self, actual_data, forecast_data):
        """Calculate forecast accuracy metrics"""
        
        if len(actual_data) == 0 or len(forecast_data) == 0:
            return {'status': 'insufficient_data'}
        
        # Align data by timestamp (simplified)
        actual_demand = actual_data['total_demand'].values
        forecast_demand = forecast_data['predicted_demand'].values[:len(actual_demand)]
        
        if len(forecast_demand) == 0:
            return {'status': 'no_forecasts'}
        
        # Calculate metrics
        mae = mean_absolute_error(actual_demand, forecast_demand)
        rmse = np.sqrt(mean_squared_error(actual_demand, forecast_demand))
        mape = np.mean(np.abs((actual_demand - forecast_demand) / np.maximum(actual_demand, 0.1))) * 100
        
        return {
            'status': 'success',
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'accuracy_percentage': max(0, 100 - mape)
        }
    
    def get_peak_predictions(self, forecast_df):
        """Identify predicted peak demand and generation periods"""
        
        peaks = {
            'demand_peak': {
                'time': forecast_df.loc[forecast_df['predicted_demand'].idxmax(), 'timestamp'],
                'value': forecast_df['predicted_demand'].max(),
                'hour': forecast_df.loc[forecast_df['predicted_demand'].idxmax(), 'hour']
            },
            'generation_peak': {
                'time': forecast_df.loc[forecast_df['predicted_generation'].idxmax(), 'timestamp'],
                'value': forecast_df['predicted_generation'].max(),
                'hour': forecast_df.loc[forecast_df['predicted_generation'].idxmax(), 'hour']
            },
            'max_deficit': {
                'time': forecast_df.loc[forecast_df['predicted_net_flow'].idxmin(), 'timestamp'],
                'value': forecast_df['predicted_net_flow'].min(),
                'hour': forecast_df.loc[forecast_df['predicted_net_flow'].idxmin(), 'hour']
            },
            'max_surplus': {
                'time': forecast_df.loc[forecast_df['predicted_net_flow'].idxmax(), 'timestamp'],
                'value': forecast_df['predicted_net_flow'].max(),
                'hour': forecast_df.loc[forecast_df['predicted_net_flow'].idxmax(), 'hour']
            }
        }
        
        return peaks
