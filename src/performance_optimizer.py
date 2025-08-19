#!/usr/bin/env python3
"""
AutoGrid Performance Optimization Engine
Real-time system optimization and resource management
"""

import time
import threading
import psutil
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import statistics

class OptimizationTarget(Enum):
    ENERGY_EFFICIENCY = "energy_efficiency"
    COST_MINIMIZATION = "cost_minimization"
    GRID_STABILITY = "grid_stability"
    CARBON_REDUCTION = "carbon_reduction"
    PEAK_SHAVING = "peak_shaving"
    LOAD_BALANCING = "load_balancing"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    ENERGY = "energy"
    THERMAL = "thermal"

@dataclass
class PerformanceMetric:
    metric_id: str
    timestamp: datetime
    resource_type: ResourceType
    value: float
    unit: str
    target_value: Optional[float]
    threshold_warning: float
    threshold_critical: float
    optimization_weight: float

@dataclass
class OptimizationResult:
    optimization_id: str
    timestamp: datetime
    target: OptimizationTarget
    parameters_before: Dict[str, float]
    parameters_after: Dict[str, float]
    improvement_percentage: float
    energy_savings: float
    cost_savings: float
    execution_time: float
    success: bool

class SystemMonitor:
    """Advanced system performance monitoring"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=10000)
        self.resource_alerts = deque(maxlen=1000)
        self.baseline_metrics = {}
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("[Performance] System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        print("[Performance] System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                for metric in metrics:
                    self.metrics_history.append(metric)
                    
                    # Check thresholds
                    self._check_thresholds(metric)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"[Performance] Monitoring error: {e}")
                time.sleep(10)
    
    def _collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect comprehensive system metrics"""
        current_time = datetime.now()
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(PerformanceMetric(
            metric_id="cpu_utilization",
            timestamp=current_time,
            resource_type=ResourceType.CPU,
            value=cpu_percent,
            unit="%",
            target_value=70.0,
            threshold_warning=80.0,
            threshold_critical=95.0,
            optimization_weight=1.0
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(PerformanceMetric(
            metric_id="memory_utilization",
            timestamp=current_time,
            resource_type=ResourceType.MEMORY,
            value=memory.percent,
            unit="%",
            target_value=60.0,
            threshold_warning=80.0,
            threshold_critical=90.0,
            optimization_weight=0.8
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(PerformanceMetric(
            metric_id="disk_utilization",
            timestamp=current_time,
            resource_type=ResourceType.DISK,
            value=disk.percent,
            unit="%",
            target_value=70.0,
            threshold_warning=85.0,
            threshold_critical=95.0,
            optimization_weight=0.6
        ))
        
        # Network metrics (simulated)
        network_utilization = np.random.uniform(10, 40)  # Simulated
        metrics.append(PerformanceMetric(
            metric_id="network_utilization",
            timestamp=current_time,
            resource_type=ResourceType.NETWORK,
            value=network_utilization,
            unit="%",
            target_value=50.0,
            threshold_warning=70.0,
            threshold_critical=90.0,
            optimization_weight=0.7
        ))
        
        # Energy consumption (simulated)
        energy_consumption = 1500 + np.random.normal(0, 100)  # Watts
        metrics.append(PerformanceMetric(
            metric_id="energy_consumption",
            timestamp=current_time,
            resource_type=ResourceType.ENERGY,
            value=energy_consumption,
            unit="W",
            target_value=1200.0,
            threshold_warning=2000.0,
            threshold_critical=2500.0,
            optimization_weight=1.2
        ))
        
        # Thermal metrics (simulated)
        cpu_temperature = 45 + np.random.normal(0, 5)  # °C
        metrics.append(PerformanceMetric(
            metric_id="cpu_temperature",
            timestamp=current_time,
            resource_type=ResourceType.THERMAL,
            value=cpu_temperature,
            unit="°C",
            target_value=40.0,
            threshold_warning=70.0,
            threshold_critical=85.0,
            optimization_weight=0.9
        ))
        
        return metrics
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds"""
        if metric.value >= metric.threshold_critical:
            alert = {
                'level': 'critical',
                'metric': metric.metric_id,
                'value': metric.value,
                'threshold': metric.threshold_critical,
                'timestamp': metric.timestamp
            }
            self.resource_alerts.append(alert)
            print(f"[Performance] CRITICAL: {metric.metric_id} = {metric.value}{metric.unit}")
            
        elif metric.value >= metric.threshold_warning:
            alert = {
                'level': 'warning',
                'metric': metric.metric_id,
                'value': metric.value,
                'threshold': metric.threshold_warning,
                'timestamp': metric.timestamp
            }
            self.resource_alerts.append(alert)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        if not self.metrics_history:
            return {}
        
        # Get latest metrics for each type
        latest_metrics = {}
        for metric in reversed(self.metrics_history):
            if metric.metric_id not in latest_metrics:
                latest_metrics[metric.metric_id] = metric.value
                
        return latest_metrics
    
    def get_performance_trends(self, hours: int = 1) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        trends = defaultdict(list)
        
        for metric in self.metrics_history:
            if metric.timestamp > cutoff_time:
                trends[metric.metric_id].append(metric.value)
        
        return dict(trends)

class IntelligentOptimizer:
    """AI-powered system optimization"""
    
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
        self.optimization_history = deque(maxlen=1000)
        self.optimization_strategies = {}
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        
    def register_optimization_strategy(self, target: OptimizationTarget, strategy_func):
        """Register optimization strategy"""
        self.optimization_strategies[target] = strategy_func
        print(f"[Performance] Strategy registered for {target.value}")
    
    def optimize_system(self, target: OptimizationTarget, 
                       constraints: Dict[str, Any] = None) -> OptimizationResult:
        """Perform intelligent system optimization"""
        start_time = time.time()
        optimization_id = f"opt_{int(start_time)}_{target.value}"
        
        # Get current system state
        current_metrics = self.monitor.get_current_metrics()
        parameters_before = current_metrics.copy()
        
        # Apply optimization strategy
        if target in self.optimization_strategies:
            optimization_params = self.optimization_strategies[target](
                current_metrics, constraints or {}
            )
        else:
            optimization_params = self._default_optimization(target, current_metrics)
        
        # Simulate optimization execution
        success = self._execute_optimization(optimization_params)
        
        # Measure results
        time.sleep(2)  # Allow time for changes to take effect
        new_metrics = self.monitor.get_current_metrics()
        parameters_after = new_metrics.copy()
        
        # Calculate improvements
        improvement = self._calculate_improvement(
            parameters_before, parameters_after, target
        )
        
        execution_time = time.time() - start_time
        
        result = OptimizationResult(
            optimization_id=optimization_id,
            timestamp=datetime.now(),
            target=target,
            parameters_before=parameters_before,
            parameters_after=parameters_after,
            improvement_percentage=improvement,
            energy_savings=self._calculate_energy_savings(parameters_before, parameters_after),
            cost_savings=self._calculate_cost_savings(parameters_before, parameters_after),
            execution_time=execution_time,
            success=success
        )
        
        self.optimization_history.append(result)
        
        if success:
            print(f"[Performance] Optimization complete: {improvement:.1f}% improvement")
        else:
            print(f"[Performance] Optimization failed for {target.value}")
        
        return result
    
    def _default_optimization(self, target: OptimizationTarget, 
                            metrics: Dict[str, float]) -> Dict[str, Any]:
        """Default optimization strategy"""
        params = {}
        
        if target == OptimizationTarget.ENERGY_EFFICIENCY:
            # Reduce CPU frequency if utilization is low
            if metrics.get('cpu_utilization', 0) < 50:
                params['cpu_frequency'] = 0.8  # Scale down
            
            # Adjust memory allocation
            if metrics.get('memory_utilization', 0) > 80:
                params['memory_compression'] = True
                
        elif target == OptimizationTarget.COST_MINIMIZATION:
            # Shift non-critical loads to off-peak hours
            params['load_shifting'] = True
            params['priority_scaling'] = 0.7
            
        elif target == OptimizationTarget.GRID_STABILITY:
            # Balance load distribution
            params['load_balancing'] = True
            params['reactive_power_control'] = True
            
        return params
    
    def _execute_optimization(self, params: Dict[str, Any]) -> bool:
        """Execute optimization parameters"""
        try:
            # Simulate optimization execution
            for param, value in params.items():
                print(f"[Performance] Applying {param}: {value}")
                time.sleep(0.5)  # Simulate execution time
            
            return True
            
        except Exception as e:
            print(f"[Performance] Optimization execution failed: {e}")
            return False
    
    def _calculate_improvement(self, before: Dict[str, float], 
                             after: Dict[str, float], 
                             target: OptimizationTarget) -> float:
        """Calculate optimization improvement percentage"""
        if target == OptimizationTarget.ENERGY_EFFICIENCY:
            energy_before = before.get('energy_consumption', 1500)
            energy_after = after.get('energy_consumption', energy_before * 0.95)
            return max(0, (energy_before - energy_after) / energy_before * 100)
            
        elif target == OptimizationTarget.GRID_STABILITY:
            # Use CPU as proxy for system stability
            cpu_before = before.get('cpu_utilization', 50)
            cpu_after = after.get('cpu_utilization', cpu_before * 0.9)
            if cpu_before > 70:  # High utilization improved
                return max(0, (cpu_before - cpu_after) / cpu_before * 100)
            else:
                return 5.0  # Baseline improvement
        
        # Default improvement simulation
        return np.random.uniform(2, 15)
    
    def _calculate_energy_savings(self, before: Dict[str, float], 
                                after: Dict[str, float]) -> float:
        """Calculate energy savings in kWh"""
        energy_before = before.get('energy_consumption', 1500) / 1000  # Convert to kW
        energy_after = after.get('energy_consumption', energy_before * 1000 * 0.95) / 1000
        return max(0, energy_before - energy_after)
    
    def _calculate_cost_savings(self, before: Dict[str, float], 
                              after: Dict[str, float]) -> float:
        """Calculate cost savings in dollars"""
        energy_savings = self._calculate_energy_savings(before, after)
        cost_per_kwh = 0.15  # $0.15 per kWh
        return energy_savings * cost_per_kwh

class AdaptiveTuner:
    """Machine learning-based adaptive tuning"""
    
    def __init__(self, optimizer: IntelligentOptimizer):
        self.optimizer = optimizer
        self.parameter_history = defaultdict(list)
        self.performance_correlation = {}
        self.auto_tuning_active = False
        
    def start_adaptive_tuning(self):
        """Start adaptive tuning process"""
        self.auto_tuning_active = True
        self.tuning_thread = threading.Thread(target=self._adaptive_loop)
        self.tuning_thread.daemon = True
        self.tuning_thread.start()
        print("[Performance] Adaptive tuning started")
    
    def stop_adaptive_tuning(self):
        """Stop adaptive tuning"""
        self.auto_tuning_active = False
        print("[Performance] Adaptive tuning stopped")
    
    def _adaptive_loop(self):
        """Main adaptive tuning loop"""
        while self.auto_tuning_active:
            try:
                # Analyze recent performance
                self._analyze_performance_patterns()
                
                # Determine if optimization is needed
                if self._should_optimize():
                    target = self._select_optimization_target()
                    self.optimizer.optimize_system(target)
                
                time.sleep(300)  # Adaptive tuning every 5 minutes
                
            except Exception as e:
                print(f"[Performance] Adaptive tuning error: {e}")
                time.sleep(60)
    
    def _analyze_performance_patterns(self):
        """Analyze performance patterns for learning"""
        trends = self.optimizer.monitor.get_performance_trends(hours=1)
        
        for metric, values in trends.items():
            if len(values) > 10:
                # Calculate trend
                trend = np.polyfit(range(len(values)), values, 1)[0]
                self.parameter_history[metric].append({
                    'trend': trend,
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'timestamp': datetime.now()
                })
    
    def _should_optimize(self) -> bool:
        """Determine if optimization is needed"""
        current_metrics = self.optimizer.monitor.get_current_metrics()
        
        # Check if any metric is above warning threshold
        warning_thresholds = {
            'cpu_utilization': 80,
            'memory_utilization': 80,
            'energy_consumption': 2000
        }
        
        for metric, threshold in warning_thresholds.items():
            if current_metrics.get(metric, 0) > threshold:
                return True
        
        # Check for degrading trends
        for metric, history in self.parameter_history.items():
            if history and len(history) >= 3:
                recent_trends = [h['trend'] for h in history[-3:]]
                if all(t > 0 for t in recent_trends):  # Consistently increasing
                    return True
        
        return False
    
    def _select_optimization_target(self) -> OptimizationTarget:
        """Select best optimization target based on current conditions"""
        current_metrics = self.optimizer.monitor.get_current_metrics()
        
        # Priority-based selection
        if current_metrics.get('energy_consumption', 0) > 2000:
            return OptimizationTarget.ENERGY_EFFICIENCY
        elif current_metrics.get('cpu_utilization', 0) > 85:
            return OptimizationTarget.LOAD_BALANCING
        elif datetime.now().hour in [17, 18, 19, 20]:  # Peak hours
            return OptimizationTarget.PEAK_SHAVING
        else:
            return OptimizationTarget.COST_MINIMIZATION

class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.optimizer = IntelligentOptimizer(self.monitor)
        self.adaptive_tuner = AdaptiveTuner(self.optimizer)
        self.running = False
        
        # Register optimization strategies
        self._register_strategies()
    
    def start(self):
        """Start performance optimization system"""
        self.running = True
        self.monitor.start_monitoring()
        self.adaptive_tuner.start_adaptive_tuning()
        print("[Performance] Performance optimization system started")
    
    def stop(self):
        """Stop optimization system"""
        self.running = False
        self.monitor.stop_monitoring()
        self.adaptive_tuner.stop_adaptive_tuning()
        print("[Performance] Performance optimization system stopped")
    
    def _register_strategies(self):
        """Register optimization strategies"""
        
        def energy_efficiency_strategy(metrics: Dict[str, float], 
                                     constraints: Dict[str, Any]) -> Dict[str, Any]:
            """Energy efficiency optimization strategy"""
            params = {}
            
            cpu_util = metrics.get('cpu_utilization', 50)
            energy = metrics.get('energy_consumption', 1500)
            
            if cpu_util < 30:
                params['cpu_power_scaling'] = 0.7
            elif cpu_util > 80:
                params['task_scheduling'] = 'load_balance'
            
            if energy > 1800:
                params['cooling_optimization'] = True
                params['non_critical_shutdown'] = True
            
            return params
        
        def cost_minimization_strategy(metrics: Dict[str, float], 
                                     constraints: Dict[str, Any]) -> Dict[str, Any]:
            """Cost minimization strategy"""
            hour = datetime.now().hour
            params = {}
            
            # Time-of-use optimization
            if 17 <= hour <= 21:  # Peak hours
                params['demand_reduction'] = 0.8
                params['battery_discharge'] = True
            elif 22 <= hour <= 6:  # Off-peak
                params['battery_charging'] = True
                params['deferred_tasks'] = False
            
            return params
        
        def grid_stability_strategy(metrics: Dict[str, float], 
                                  constraints: Dict[str, Any]) -> Dict[str, Any]:
            """Grid stability optimization"""
            params = {}
            
            network_util = metrics.get('network_utilization', 30)
            
            if network_util > 60:
                params['communication_throttling'] = True
                params['data_compression'] = True
            
            params['voltage_regulation'] = True
            params['frequency_control'] = True
            
            return params
        
        # Register strategies
        self.optimizer.register_optimization_strategy(
            OptimizationTarget.ENERGY_EFFICIENCY, energy_efficiency_strategy
        )
        self.optimizer.register_optimization_strategy(
            OptimizationTarget.COST_MINIMIZATION, cost_minimization_strategy
        )
        self.optimizer.register_optimization_strategy(
            OptimizationTarget.GRID_STABILITY, grid_stability_strategy
        )
    
    def manual_optimization(self, target: OptimizationTarget, 
                          constraints: Dict[str, Any] = None) -> OptimizationResult:
        """Perform manual optimization"""
        return self.optimizer.optimize_system(target, constraints)
    
    def get_performance_status(self) -> Dict:
        """Get current performance status"""
        current_metrics = self.monitor.get_current_metrics()
        
        # Calculate overall performance score
        score_factors = {
            'cpu_utilization': (100 - current_metrics.get('cpu_utilization', 50)) / 100,
            'memory_utilization': (100 - current_metrics.get('memory_utilization', 50)) / 100,
            'energy_efficiency': max(0, (2000 - current_metrics.get('energy_consumption', 1500)) / 2000)
        }
        
        overall_score = sum(score_factors.values()) / len(score_factors) * 100
        
        return {
            'overall_score': round(overall_score, 1),
            'current_metrics': current_metrics,
            'recent_optimizations': len(self.optimizer.optimization_history),
            'adaptive_tuning_active': self.adaptive_tuner.auto_tuning_active,
            'alerts': len([a for a in self.monitor.resource_alerts 
                          if (datetime.now() - a['timestamp']).seconds < 3600])
        }
    
    def get_optimization_history(self, limit: int = 10) -> List[OptimizationResult]:
        """Get recent optimization history"""
        return list(self.optimizer.optimization_history)[-limit:]
    
    def get_recommendations(self) -> List[Dict[str, str]]:
        """Get optimization recommendations"""
        current_metrics = self.monitor.get_current_metrics()
        recommendations = []
        
        # CPU optimization
        cpu_util = current_metrics.get('cpu_utilization', 50)
        if cpu_util > 80:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'high',
                'description': 'CPU utilization is high. Consider load balancing or upgrading resources.',
                'action': 'Optimize CPU usage'
            })
        
        # Energy optimization
        energy = current_metrics.get('energy_consumption', 1500)
        if energy > 1800:
            recommendations.append({
                'type': 'energy_optimization',
                'priority': 'medium',
                'description': 'Energy consumption is above optimal levels.',
                'action': 'Run energy efficiency optimization'
            })
        
        # Memory optimization
        memory_util = current_metrics.get('memory_utilization', 50)
        if memory_util > 85:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'description': 'Memory utilization is critically high.',
                'action': 'Enable memory compression or reduce memory usage'
            })
        
        return recommendations

# Global performance optimizer
performance_optimizer = None

def init_performance_optimizer():
    """Initialize performance optimization system"""
    global performance_optimizer
    if performance_optimizer is None:
        performance_optimizer = PerformanceOptimizer()
        performance_optimizer.start()
        print("[Performance] Performance optimization system initialized")

def stop_performance_optimizer():
    """Stop performance optimizer"""
    global performance_optimizer
    if performance_optimizer:
        performance_optimizer.stop()
        performance_optimizer = None

def get_performance_status():
    """Get performance optimization status"""
    global performance_optimizer
    if performance_optimizer:
        return performance_optimizer.get_performance_status()
    return {'running': False}

if __name__ == "__main__":
    print("AutoGrid Performance Optimization Test")
    print("=====================================")
    
    init_performance_optimizer()
    
    try:
        # Run for 60 seconds
        time.sleep(60)
        
        # Get status
        status = get_performance_status()
        print(f"Performance Status: {status}")
        
        # Run manual optimization
        result = performance_optimizer.manual_optimization(OptimizationTarget.ENERGY_EFFICIENCY)
        print(f"Optimization Result: {result.improvement_percentage:.1f}% improvement")
        
        # Get recommendations
        recommendations = performance_optimizer.get_recommendations()
        for rec in recommendations:
            print(f"Recommendation: {rec['description']}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_performance_optimizer()