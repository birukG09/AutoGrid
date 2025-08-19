import numpy as np
import pandas as pd
from scipy.optimize import linprog
from datetime import datetime, timedelta

class GridOptimizer:
    def __init__(self):
        self.load_priorities = {
            'critical': 1.0,    # Industrial, hospitals
            'high': 0.8,        # Commercial
            'medium': 0.6,      # Residential
            'low': 0.4          # EV charging, optional loads
        }
        
        self.storage_capacity = 100  # kWh
        self.max_charge_rate = 20    # kW
        self.max_discharge_rate = 20 # kW
        
    def optimize_dispatch(self, current_data, forecast_data=None):
        """Optimize energy dispatch using linear programming"""
        
        # Extract current state
        generation = current_data['total_generation']
        demand = current_data['total_demand']
        battery_soc = current_data['battery_soc']
        
        # Define loads with priorities
        loads = {
            'industrial': {'demand': current_data['industrial_load'], 'priority': 'critical'},
            'commercial': {'demand': current_data['commercial_load'], 'priority': 'high'},
            'residential': {'demand': current_data['residential_load'], 'priority': 'medium'},
            'ev_charging': {'demand': current_data['ev_load'], 'priority': 'low'}
        }
        
        # Calculate optimal load allocation
        optimization_result = self._solve_dispatch_optimization(generation, loads, battery_soc)
        
        return optimization_result
    
    def _solve_dispatch_optimization(self, generation, loads, battery_soc):
        """Solve the dispatch optimization problem"""
        
        # Objective: Maximize load satisfaction weighted by priority
        # Variables: [industrial_served, commercial_served, residential_served, ev_served, battery_charge, battery_discharge]
        
        # Coefficients for objective function (negative because linprog minimizes)
        c = [
            -self.load_priorities['critical'],   # industrial
            -self.load_priorities['high'],       # commercial  
            -self.load_priorities['medium'],     # residential
            -self.load_priorities['low'],        # ev charging
            -0.1,  # battery charge (small incentive)
            0.1    # battery discharge (small penalty)
        ]
        
        # Constraints
        # Power balance: generation + battery_discharge = loads_served + battery_charge
        A_eq = [[1, 1, 1, 1, 1, -1]]  # sum of served loads + charge - discharge = generation
        b_eq = [generation]
        
        # Inequality constraints
        A_ub = []
        b_ub = []
        
        # Load constraints (cannot serve more than demanded)
        for i, (load_name, load_data) in enumerate(loads.items()):
            constraint = [0] * 6
            constraint[i] = 1
            A_ub.append(constraint)
            b_ub.append(load_data['demand'])
        
        # Battery constraints
        # Battery charge rate constraint
        A_ub.append([0, 0, 0, 0, 1, 0])
        b_ub.append(self.max_charge_rate)
        
        # Battery discharge rate constraint  
        A_ub.append([0, 0, 0, 0, 0, 1])
        b_ub.append(self.max_discharge_rate)
        
        # Battery capacity constraint (simplified)
        # Current energy + charge - discharge <= capacity
        current_energy = battery_soc * self.storage_capacity / 100
        A_ub.append([0, 0, 0, 0, 1, -1])
        b_ub.append(self.storage_capacity - current_energy)
        
        # Lower bounds (all variables >= 0)
        bounds = [(0, None)] * 6
        
        try:
            # Solve optimization
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if result.success:
                industrial_served, commercial_served, residential_served, ev_served, battery_charge, battery_discharge = result.x
                
                return {
                    'status': 'optimal',
                    'load_allocation': {
                        'industrial': industrial_served,
                        'commercial': commercial_served,  
                        'residential': residential_served,
                        'ev_charging': ev_served
                    },
                    'battery_action': {
                        'charge': battery_charge,
                        'discharge': battery_discharge
                    },
                    'total_served': sum(result.x[:4]),
                    'load_shedding': sum([loads[k]['demand'] for k in loads.keys()]) - sum(result.x[:4]),
                    'objective_value': -result.fun
                }
            else:
                return self._fallback_optimization(generation, loads)
                
        except Exception as e:
            return self._fallback_optimization(generation, loads)
    
    def _fallback_optimization(self, generation, loads):
        """Fallback optimization using simple priority-based allocation"""
        
        available_power = generation
        load_allocation = {}
        
        # Sort loads by priority
        sorted_loads = sorted(loads.items(), 
                            key=lambda x: self.load_priorities[x[1]['priority']], 
                            reverse=True)
        
        for load_name, load_data in sorted_loads:
            allocated = min(load_data['demand'], available_power)
            load_allocation[load_name] = allocated
            available_power -= allocated
        
        total_demand = sum([loads[k]['demand'] for k in loads.keys()])
        total_served = sum(load_allocation.values())
        
        return {
            'status': 'fallback',
            'load_allocation': load_allocation,
            'battery_action': {'charge': 0, 'discharge': 0},
            'total_served': total_served,
            'load_shedding': total_demand - total_served,
            'objective_value': sum([load_allocation[k] * self.load_priorities[loads[k]['priority']] 
                                  for k in load_allocation.keys()])
        }
    
    def get_optimization_recommendations(self, current_data):
        """Get optimization recommendations for grid operators"""
        
        optimization_result = self.optimize_dispatch(current_data)
        recommendations = []
        
        # Check for load shedding
        if optimization_result['load_shedding'] > 0:
            recommendations.append({
                'priority': 'High',
                'action': 'Load Shedding Required',
                'description': f"Shed {optimization_result['load_shedding']:.1f} kW of low-priority loads",
                'target': 'EV Charging and Optional Loads'
            })
        
        # Battery recommendations
        battery_action = optimization_result['battery_action']
        if battery_action['charge'] > 0:
            recommendations.append({
                'priority': 'Medium',
                'action': 'Charge Battery',
                'description': f"Charge battery at {battery_action['charge']:.1f} kW",
                'target': 'Energy Storage System'
            })
        elif battery_action['discharge'] > 0:
            recommendations.append({
                'priority': 'Medium', 
                'action': 'Discharge Battery',
                'description': f"Discharge battery at {battery_action['discharge']:.1f} kW",
                'target': 'Energy Storage System'
            })
        
        # Generation recommendations
        generation_ratio = current_data['total_generation'] / current_data['total_demand']
        if generation_ratio < 0.8:
            recommendations.append({
                'priority': 'High',
                'action': 'Increase Generation',
                'description': "Consider starting backup generators or requesting grid import",
                'target': 'Generation Sources'
            })
        elif generation_ratio > 1.2:
            recommendations.append({
                'priority': 'Low',
                'action': 'Excess Generation',
                'description': "Consider reducing generation or increasing storage",
                'target': 'Generation Sources'
            })
        
        # Efficiency recommendations
        if current_data['solar_generation'] < 5 and 10 <= datetime.now().hour <= 16:
            recommendations.append({
                'priority': 'Medium',
                'action': 'Check Solar Panels',
                'description': "Low solar generation detected during peak hours",
                'target': 'Solar Generation System'
            })
        
        return recommendations
    
    def calculate_efficiency_metrics(self, current_data):
        """Calculate system efficiency metrics"""
        
        total_generation = current_data['total_generation']
        total_demand = current_data['total_demand']
        renewable_generation = current_data['solar_generation'] + current_data['wind_generation']
        
        metrics = {
            'generation_efficiency': min(100, total_generation / max(total_demand, 0.1) * 100),
            'renewable_penetration': renewable_generation / max(total_generation, 0.1) * 100,
            'load_factor': total_demand / 100 * 100,  # Assuming 100kW is max capacity
            'storage_utilization': abs(100 - current_data['battery_soc']) / 100 * 100,
            'grid_stability': max(0, 100 - abs(current_data['net_flow']) * 2)  # Penalize large imbalances
        }
        
        return metrics
