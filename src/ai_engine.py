#!/usr/bin/env python3
"""
AutoGrid AI Engine (Python)
Machine Learning and Predictive Analytics
"""

import numpy as np
import pandas as pd
import json
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import pickle
import os

class ForecastType(Enum):
    DEMAND = "demand"
    GENERATION = "generation"
    PRICE = "price"
    WEATHER = "weather"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class ForecastResult:
    timestamp: datetime
    forecast_type: ForecastType
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    model_accuracy: float

@dataclass
class AnomalyDetection:
    timestamp: datetime
    component: str
    anomaly_score: float
    expected_value: float
    actual_value: float
    severity: AlertLevel
    description: str

@dataclass
class EVOptimization:
    ev_id: str
    recommended_action: str
    target_power: float
    estimated_duration: int
    priority_score: float
    reason: str

class MLModelManager:
    def __init__(self):
        self.models = {}
        self.model_accuracies = {}
        self.last_training = {}
        
    def load_model(self, model_name: str, model_path: Optional[str] = None):
        """Load or create ML model"""
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"[Python] Loaded model: {model_name}")
            except Exception as e:
                print(f"[Python] Failed to load {model_name}: {e}")
                self._create_synthetic_model(model_name)
        else:
            self._create_synthetic_model(model_name)
            
    def _create_synthetic_model(self, model_name: str):
        """Create synthetic model for simulation"""
        # In real implementation, this would be a trained ML model
        class SyntheticModel:
            def __init__(self, model_type):
                self.model_type = model_type
                self.weights = np.random.randn(10)
                self.bias = np.random.randn()
                
            def predict(self, features):
                # Simple linear combination for simulation
                if len(features) != len(self.weights):
                    features = np.resize(features, len(self.weights))
                return np.dot(features, self.weights) + self.bias
                
            def predict_with_confidence(self, features):
                prediction = self.predict(features)
                confidence = 0.85 + np.random.random() * 0.1  # 85-95% confidence
                error_margin = abs(prediction) * (1 - confidence)
                return prediction, (prediction - error_margin, prediction + error_margin), confidence
        
        self.models[model_name] = SyntheticModel(model_name)
        self.model_accuracies[model_name] = 0.85 + np.random.random() * 0.1
        print(f"[Python] Created synthetic model: {model_name}")

class AIEngine:
    def __init__(self):
        self.running = False
        self.model_manager = MLModelManager()
        self.historical_data = []
        self.current_predictions = {}
        self.anomalies = []
        self.ev_recommendations = {}
        self.weather_data = {}
        
        # Initialize models
        self._initialize_models()
        
        # Data collection thread
        self.data_thread = None
        self.prediction_thread = None
        self.anomaly_thread = None
        
    def _initialize_models(self):
        """Initialize ML models"""
        models_to_load = [
            "demand_forecaster",
            "solar_predictor", 
            "wind_predictor",
            "price_forecaster",
            "anomaly_detector",
            "ev_optimizer",
            "weather_predictor"
        ]
        
        for model_name in models_to_load:
            self.model_manager.load_model(model_name)
            
    def start(self):
        """Start AI engine services"""
        print("[Python] Starting AI Engine...")
        self.running = True
        
        # Start data collection
        self.data_thread = threading.Thread(target=self._data_collection_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start prediction loop
        self.prediction_thread = threading.Thread(target=self._prediction_loop)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        
        # Start anomaly detection
        self.anomaly_thread = threading.Thread(target=self._anomaly_detection_loop)
        self.anomaly_thread.daemon = True
        self.anomaly_thread.start()
        
        print("[Python] AI Engine started successfully")
        
    def stop(self):
        """Stop AI engine services"""
        self.running = False
        print("[Python] AI Engine stopped")
        
    def _data_collection_loop(self):
        """Collect historical data for training"""
        while self.running:
            try:
                # Simulate data collection from sensors and external APIs
                current_time = datetime.now()
                
                # Generate synthetic sensor data
                data_point = {
                    'timestamp': current_time,
                    'solar_generation': self._simulate_solar(current_time),
                    'wind_generation': self._simulate_wind(current_time),
                    'total_demand': self._simulate_demand(current_time),
                    'temperature': self._simulate_weather(current_time)['temperature'],
                    'humidity': self._simulate_weather(current_time)['humidity'],
                    'cloud_cover': self._simulate_weather(current_time)['cloud_cover'],
                    'grid_frequency': 60.0 + np.random.normal(0, 0.05),
                    'battery_soc': 75.0 + np.random.normal(0, 5),
                }
                
                self.historical_data.append(data_point)
                
                # Keep only last 1000 data points
                if len(self.historical_data) > 1000:
                    self.historical_data = self.historical_data[-1000:]
                    
                time.sleep(10)  # Collect data every 10 seconds
                
            except Exception as e:
                print(f"[Python] Data collection error: {e}")
                time.sleep(5)
                
    def _prediction_loop(self):
        """Generate predictions using ML models"""
        while self.running:
            try:
                if len(self.historical_data) < 10:
                    time.sleep(5)
                    continue
                    
                current_time = datetime.now()
                
                # Generate features from recent data
                features = self._extract_features(current_time)
                
                # Make predictions
                predictions = {}
                
                # Demand forecasting
                demand_model = self.model_manager.models.get('demand_forecaster')
                if demand_model:
                    pred, conf_int, conf_level = demand_model.predict_with_confidence(features)
                    predictions['demand'] = ForecastResult(
                        timestamp=current_time + timedelta(hours=1),
                        forecast_type=ForecastType.DEMAND,
                        predicted_value=max(0, pred),
                        confidence_interval=conf_int,
                        confidence_level=conf_level,
                        model_accuracy=self.model_manager.model_accuracies.get('demand_forecaster', 0.85)
                    )
                
                # Solar generation forecasting
                solar_model = self.model_manager.models.get('solar_predictor')
                if solar_model:
                    pred, conf_int, conf_level = solar_model.predict_with_confidence(features)
                    predictions['solar'] = ForecastResult(
                        timestamp=current_time + timedelta(hours=1),
                        forecast_type=ForecastType.GENERATION,
                        predicted_value=max(0, pred),
                        confidence_interval=conf_int,
                        confidence_level=conf_level,
                        model_accuracy=self.model_manager.model_accuracies.get('solar_predictor', 0.85)
                    )
                
                # Price forecasting
                price_model = self.model_manager.models.get('price_forecaster')
                if price_model:
                    pred, conf_int, conf_level = price_model.predict_with_confidence(features)
                    predictions['price'] = ForecastResult(
                        timestamp=current_time + timedelta(hours=1),
                        forecast_type=ForecastType.PRICE,
                        predicted_value=max(0.05, pred * 0.15),  # Base price $0.15/kWh
                        confidence_interval=(max(0.05, conf_int[0] * 0.15), conf_int[1] * 0.15),
                        confidence_level=conf_level,
                        model_accuracy=self.model_manager.model_accuracies.get('price_forecaster', 0.85)
                    )
                
                self.current_predictions = predictions
                
                # Generate EV optimization recommendations
                self._optimize_ev_fleet()
                
                time.sleep(30)  # Update predictions every 30 seconds
                
            except Exception as e:
                print(f"[Python] Prediction error: {e}")
                time.sleep(10)
                
    def _anomaly_detection_loop(self):
        """Detect anomalies in system operation"""
        while self.running:
            try:
                if len(self.historical_data) < 20:
                    time.sleep(10)
                    continue
                    
                recent_data = self.historical_data[-10:]
                current_time = datetime.now()
                
                # Check for anomalies
                anomaly_model = self.model_manager.models.get('anomaly_detector')
                if anomaly_model:
                    for component in ['solar_generation', 'wind_generation', 'total_demand', 'grid_frequency']:
                        values = [dp[component] for dp in recent_data]
                        current_value = values[-1]
                        expected_value = np.mean(values[:-1])
                        
                        features = [current_value, expected_value, np.std(values), len(values)]
                        anomaly_score = abs(anomaly_model.predict(features))
                        
                        if anomaly_score > 2.0:  # Threshold for anomaly
                            severity = AlertLevel.CRITICAL if anomaly_score > 5.0 else AlertLevel.WARNING
                            
                            anomaly = AnomalyDetection(
                                timestamp=current_time,
                                component=component,
                                anomaly_score=anomaly_score,
                                expected_value=float(expected_value),
                                actual_value=current_value,
                                severity=severity,
                                description=f"Anomalous {component}: {current_value:.2f} (expected {expected_value:.2f})"
                            )
                            
                            self.anomalies.append(anomaly)
                            print(f"[Python] Anomaly detected: {anomaly.description}")
                
                # Keep only recent anomalies
                cutoff_time = current_time - timedelta(hours=1)
                self.anomalies = [a for a in self.anomalies if a.timestamp > cutoff_time]
                
                time.sleep(15)  # Check for anomalies every 15 seconds
                
            except Exception as e:
                print(f"[Python] Anomaly detection error: {e}")
                time.sleep(10)
                
    def _optimize_ev_fleet(self):
        """Generate EV optimization recommendations"""
        try:
            ev_model = self.model_manager.models.get('ev_optimizer')
            if not ev_model or not self.current_predictions:
                return
                
            # Simulate EV fleet status
            ev_fleet = {}
            for i in range(1, 9):
                ev_id = f"EV_{i:03d}"
                ev_fleet[ev_id] = {
                    'connected': np.random.choice([True, False], p=[0.7, 0.3]),
                    'soc': np.random.uniform(20, 95),
                    'max_power': 11.0,
                    'v2g_enabled': np.random.choice([True, False], p=[0.6, 0.4]),
                    'priority': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], p=[0.4, 0.3, 0.2, 0.1])
                }
            
            # Get predicted demand and generation
            demand_forecast = self.current_predictions.get('demand')
            solar_forecast = self.current_predictions.get('solar')
            
            if not (demand_forecast and solar_forecast):
                return
                
            predicted_imbalance = solar_forecast.predicted_value - demand_forecast.predicted_value
            
            recommendations = {}
            
            for ev_id, status in ev_fleet.items():
                if not status['connected']:
                    continue
                    
                features = [
                    status['soc'],
                    status['max_power'],
                    predicted_imbalance,
                    1.0 if status['v2g_enabled'] else 0.0,
                    {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}[status['priority']]
                ]
                
                optimization_score = ev_model.predict(features)
                
                if predicted_imbalance > 5.0:  # Excess generation
                    if status['soc'] < 90:
                        recommendations[ev_id] = EVOptimization(
                            ev_id=ev_id,
                            recommended_action="CHARGE",
                            target_power=min(status['max_power'], predicted_imbalance * 0.3),
                            estimated_duration=int((90 - status['soc']) / 10 * 60),  # minutes
                            priority_score=optimization_score,
                            reason="Excess generation available"
                        )
                elif predicted_imbalance < -5.0:  # Energy deficit
                    if status['v2g_enabled'] and status['soc'] > 50 and status['priority'] != 'CRITICAL':
                        recommendations[ev_id] = EVOptimization(
                            ev_id=ev_id,
                            recommended_action="V2G_DISCHARGE",
                            target_power=min(status['max_power'] * 0.8, abs(predicted_imbalance) * 0.3),
                            estimated_duration=int((status['soc'] - 50) / 10 * 60),  # minutes
                            priority_score=optimization_score,
                            reason="Grid support needed"
                        )
                        
            self.ev_recommendations = recommendations
            
        except Exception as e:
            print(f"[Python] EV optimization error: {e}")
            
    def _extract_features(self, timestamp: datetime) -> List[float]:
        """Extract features for ML models"""
        if not self.historical_data:
            return [0.0] * 10
            
        # Time-based features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Statistical features from recent data
        recent_data = self.historical_data[-10:]
        
        features = [
            hour,
            day_of_week,
            month,
            1.0 if day_of_week >= 5 else 0.0,  # is_weekend
            np.mean([dp['temperature'] for dp in recent_data]),
            np.mean([dp['humidity'] for dp in recent_data]),
            np.mean([dp['cloud_cover'] for dp in recent_data]),
            np.mean([dp['solar_generation'] for dp in recent_data]),
            np.mean([dp['wind_generation'] for dp in recent_data]),
            np.mean([dp['total_demand'] for dp in recent_data])
        ]
        
        return features
        
    def _simulate_solar(self, timestamp: datetime) -> float:
        """Simulate solar generation based on time"""
        hour = timestamp.hour
        if 6 <= hour <= 18:
            solar_factor = np.sin(np.pi * (hour - 6) / 12)
            return max(0, 25 * solar_factor * (0.8 + 0.4 * np.random.random()))
        return 0.0
        
    def _simulate_wind(self, timestamp: datetime) -> float:
        """Simulate wind generation"""
        hour = timestamp.hour
        wind_factor = 0.3 + 0.7 * (1 + np.sin(hour * np.pi / 12 + np.pi)) / 2
        return 15 * wind_factor * (0.7 + 0.6 * np.random.random())
        
    def _simulate_demand(self, timestamp: datetime) -> float:
        """Simulate energy demand"""
        hour = timestamp.hour
        # Peak in evening, lower at night
        demand_factor = 0.4 + 0.6 * (1 + np.sin((hour - 6) * np.pi / 12)) / 2
        return 60 * demand_factor * (0.8 + 0.4 * np.random.random())
        
    def _simulate_weather(self, timestamp: datetime) -> Dict[str, float]:
        """Simulate weather conditions"""
        hour = timestamp.hour
        
        # Daily temperature cycle
        temp = 20 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
        
        # Humidity inversely related to temperature
        humidity = 50 + 20 * np.sin(2 * np.pi * hour / 24 + np.pi/4) + np.random.normal(0, 5)
        humidity = np.clip(humidity, 0, 100)
        
        # Cloud cover
        cloud_cover = max(0, min(100, 30 + 40 * np.sin(2 * np.pi * hour / 24 + np.pi) + np.random.normal(0, 15)))
        
        return {
            'temperature': temp,
            'humidity': humidity,
            'cloud_cover': cloud_cover
        }
        
    def get_predictions(self) -> Dict[str, ForecastResult]:
        """Get current predictions"""
        return self.current_predictions
        
    def get_anomalies(self) -> List[AnomalyDetection]:
        """Get recent anomalies"""
        return self.anomalies[-10:]  # Last 10 anomalies
        
    def get_ev_recommendations(self) -> Dict[str, EVOptimization]:
        """Get EV optimization recommendations"""
        return self.ev_recommendations
        
    def get_model_status(self) -> Dict[str, Dict]:
        """Get ML model status"""
        status = {}
        for model_name in self.model_manager.models.keys():
            status[model_name] = {
                'loaded': True,
                'accuracy': self.model_manager.model_accuracies.get(model_name, 0.85),
                'last_prediction': datetime.now().isoformat()
            }
        return status
        
    def retrain_model(self, model_name: str) -> bool:
        """Retrain a specific model"""
        try:
            if not self.historical_data or len(self.historical_data) < 100:
                print(f"[Python] Insufficient data for retraining {model_name}")
                return False
                
            print(f"[Python] Retraining model: {model_name}")
            
            # Simulate retraining with improved accuracy
            current_accuracy = self.model_manager.model_accuracies.get(model_name, 0.85)
            new_accuracy = min(0.95, current_accuracy + np.random.uniform(0.01, 0.05))
            self.model_manager.model_accuracies[model_name] = new_accuracy
            self.model_manager.last_training[model_name] = datetime.now()
            
            print(f"[Python] Model {model_name} retrained. Accuracy: {new_accuracy:.3f}")
            return True
            
        except Exception as e:
            print(f"[Python] Error retraining {model_name}: {e}")
            return False

# Global AI engine instance for external interface
ai_engine = None

def init_ai_engine():
    """Initialize AI engine"""
    global ai_engine
    if ai_engine is None:
        ai_engine = AIEngine()
        ai_engine.start()
        print("[Python] AI Engine initialized")

def stop_ai_engine():
    """Stop AI engine"""
    global ai_engine
    if ai_engine:
        ai_engine.stop()
        ai_engine = None
        print("[Python] AI Engine stopped")

def get_ai_status():
    """Get AI engine status"""
    global ai_engine
    if ai_engine:
        return {
            'running': ai_engine.running,
            'models': ai_engine.get_model_status(),
            'predictions': len(ai_engine.current_predictions),
            'anomalies': len(ai_engine.get_anomalies()),
            'data_points': len(ai_engine.historical_data)
        }
    return {'running': False}

if __name__ == "__main__":
    print("AutoGrid AI Engine Test")
    print("=======================")
    
    init_ai_engine()
    
    try:
        # Run for 60 seconds
        for i in range(12):
            time.sleep(5)
            status = get_ai_status()
            predictions = ai_engine.get_predictions() if ai_engine else {}
            anomalies = ai_engine.get_anomalies() if ai_engine else []
            
            print(f"\n--- Status Update {i+1} ---")
            print(f"Models loaded: {len(status.get('models', {}))}")
            print(f"Active predictions: {status.get('predictions', 0)}")
            print(f"Recent anomalies: {len(anomalies)}")
            
            if predictions:
                for pred_type, forecast in predictions.items():
                    print(f"{pred_type}: {forecast.predicted_value:.2f} (confidence: {forecast.confidence_level:.1%})")
                    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_ai_engine()