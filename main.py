#!/usr/bin/env python3
"""
AutoGrid Distributed Energy Controller
Multi-Language Console Application
"""

import os
import sys
import time
import subprocess
import json
import threading
from datetime import datetime
from enum import Enum
import signal
import importlib.util

# Advanced system imports
try:
    from src.blockchain_trading import init_trading_engine, get_trading_status, stop_trading_engine
    from src.cybersecurity import init_security_system, get_security_status, stop_security_system
    from src.advanced_forecasting import init_advanced_forecasting, get_advanced_forecasting_status, stop_advanced_forecasting
    from src.ai_engine import init_ai_engine, get_ai_status, stop_ai_engine
    ADVANCED_FEATURES = True
except ImportError as e:
    print(f"[System] Advanced features not available: {e}")
    ADVANCED_FEATURES = False

# ANSI color codes for engineering/cyber theme
class Colors:
    # Core colors: Green, White, Red
    GREEN = '\033[92m'      # Success, Online status
    WHITE = '\033[97m'      # Normal text
    RED = '\033[91m'        # Alerts, Errors
    BRIGHT_GREEN = '\033[1;92m'  # Headers, Important status
    BRIGHT_WHITE = '\033[1;97m'  # Highlighted text
    BRIGHT_RED = '\033[1;91m'    # Critical alerts
    DIM_GREEN = '\033[2;92m'     # Secondary info
    DIM_WHITE = '\033[2;97m'     # Subdued text
    RESET = '\033[0m'       # Reset color
    BOLD = '\033[1m'        # Bold text
    UNDERLINE = '\033[4m'   # Underlined text

class SystemStatus(Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"

class AutoGridController:
    def __init__(self):
        self.running = True
        self.system_status = SystemStatus.ONLINE
        self.subsystems = {
            'embedded_controller': SystemStatus.OFFLINE,
            'rust_coordinator': SystemStatus.OFFLINE,
            'ai_forecaster': SystemStatus.OFFLINE,
            'ev_manager': SystemStatus.OFFLINE,
            'blockchain_engine': SystemStatus.OFFLINE,
            'cybersecurity': SystemStatus.OFFLINE,
            'quantum_optimizer': SystemStatus.OFFLINE,
            'advanced_ai': SystemStatus.OFFLINE
        }
        self.energy_data = {}
        self.ev_fleet = []
        self.security_events = []
        self.trading_transactions = []
        self.advanced_features_enabled = ADVANCED_FEATURES
        
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        
    def print_header(self):
        header = f"""
{Colors.BRIGHT_GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
║{Colors.BRIGHT_WHITE}                     AutoGrid Distributed Energy Controller                     {Colors.BRIGHT_GREEN}║
║{Colors.WHITE}                      Multi-Language Microgrid System                        {Colors.BRIGHT_GREEN}║
║{Colors.DIM_WHITE}                     [C++ | Rust | Python | ARM Core]                        {Colors.BRIGHT_GREEN}║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
        print(header)
        
    def print_system_status(self):
        print(f"{Colors.BRIGHT_WHITE}┌─ SYSTEM STATUS ─────────────────────────────────────────────────────────────┐{Colors.RESET}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Colors.WHITE}│ Timestamp: {Colors.GREEN}{timestamp}{Colors.WHITE}                                              │{Colors.RESET}")
        
        # Main system status
        status_color = Colors.GREEN if self.system_status == SystemStatus.ONLINE else Colors.RED
        print(f"{Colors.WHITE}│ Main System: {status_color}{self.system_status.value:<15}{Colors.WHITE}                                     │{Colors.RESET}")
        
        # Subsystem status
        for subsystem, status in self.subsystems.items():
            color = Colors.GREEN if status == SystemStatus.ONLINE else Colors.RED if status == SystemStatus.ERROR else Colors.DIM_WHITE
            formatted_name = subsystem.replace('_', ' ').title()
            print(f"{Colors.WHITE}│ {formatted_name:<20} {color}{status.value:<15}{Colors.WHITE}                              │{Colors.RESET}")
        
        print(f"{Colors.BRIGHT_WHITE}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}")
        
    def print_energy_metrics(self):
        print(f"{Colors.BRIGHT_WHITE}┌─ ENERGY METRICS ────────────────────────────────────────────────────────────┐{Colors.RESET}")
        
        if not self.energy_data:
            self.update_energy_data()
            
        metrics = [
            ("Solar Generation", f"{self.energy_data.get('solar_generation', 0):.1f} kW", Colors.GREEN),
            ("Wind Generation", f"{self.energy_data.get('wind_generation', 0):.1f} kW", Colors.GREEN),
            ("Battery SOC", f"{self.energy_data.get('battery_soc', 0):.1f}%", Colors.WHITE),
            ("Total Demand", f"{self.energy_data.get('total_demand', 0):.1f} kW", Colors.WHITE),
            ("Net Flow", f"{self.energy_data.get('net_flow', 0):+.1f} kW", 
             Colors.GREEN if self.energy_data.get('net_flow', 0) >= 0 else Colors.RED),
            ("Grid Status", self.energy_data.get('grid_status', 'UNKNOWN'), 
             Colors.GREEN if self.energy_data.get('grid_connected', False) else Colors.RED)
        ]
        
        for label, value, color in metrics:
            print(f"{Colors.WHITE}│ {label:<20} {color}{value:<25}{Colors.WHITE}                        │{Colors.RESET}")
            
        print(f"{Colors.BRIGHT_WHITE}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}")
        
    def print_ev_status(self):
        print(f"{Colors.BRIGHT_WHITE}┌─ EV FLEET & V2G STATUS ─────────────────────────────────────────────────────┐{Colors.RESET}")
        
        if not self.ev_fleet:
            self.initialize_ev_fleet()
            
        print(f"{Colors.WHITE}│ {'Vehicle ID':<12} {'Status':<10} {'SOC':<8} {'Power':<10} {'Mode':<12}        │{Colors.RESET}")
        print(f"{Colors.DIM_WHITE}│ {'─'*12} {'─'*10} {'─'*8} {'─'*10} {'─'*12}        │{Colors.RESET}")
        
        for ev in self.ev_fleet[:6]:  # Show first 6 EVs
            status_color = Colors.GREEN if ev['connected'] else Colors.DIM_WHITE
            mode_color = Colors.RED if ev['mode'] == 'V2G_DISCHARGE' else Colors.GREEN if ev['mode'] == 'CHARGING' else Colors.WHITE
            
            print(f"{Colors.WHITE}│ {status_color}{ev['id']:<12}{Colors.WHITE} {status_color}{ev['status']:<10}{Colors.WHITE} "
                  f"{Colors.WHITE}{ev['soc']:<8.1f}{Colors.WHITE} {Colors.WHITE}{ev['power']:<10.1f}{Colors.WHITE} "
                  f"{mode_color}{ev['mode']:<12}{Colors.WHITE}        │{Colors.RESET}")
                  
        total_connected = sum(1 for ev in self.ev_fleet if ev['connected'])
        total_v2g_active = sum(1 for ev in self.ev_fleet if ev['mode'] == 'V2G_DISCHARGE')
        
        print(f"{Colors.DIM_WHITE}│ {'─'*70}        │{Colors.RESET}")
        print(f"{Colors.WHITE}│ Total Connected: {Colors.GREEN}{total_connected:<3}{Colors.WHITE} │ V2G Active: {Colors.RED}{total_v2g_active:<3}{Colors.WHITE} │ Fleet Size: {Colors.WHITE}{len(self.ev_fleet):<3}{Colors.WHITE}       │{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}")
        
    def print_alerts(self):
        print(f"{Colors.BRIGHT_WHITE}┌─ SYSTEM ALERTS & NOTIFICATIONS ─────────────────────────────────────────────┐{Colors.RESET}")
        
        alerts = self.get_current_alerts()
        
        if not alerts:
            print(f"{Colors.WHITE}│ {Colors.GREEN}✓ All systems operating normally - No active alerts{Colors.WHITE}                      │{Colors.RESET}")
        else:
            for alert in alerts[:5]:  # Show max 5 alerts
                icon = "⚠" if alert['level'] == 'WARNING' else "✗" if alert['level'] == 'CRITICAL' else "ℹ"
                color = Colors.RED if alert['level'] == 'CRITICAL' else Colors.WHITE
                print(f"{Colors.WHITE}│ {color}{icon} {alert['message']:<65}{Colors.WHITE} │{Colors.RESET}")
                
        print(f"{Colors.BRIGHT_WHITE}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}")
        
    def print_menu(self):
        print(f"{Colors.BRIGHT_WHITE}┌─ CONTROL MENU ──────────────────────────────────────────────────────────────┐{Colors.RESET}")
        menu_items = [
            ("1", "Grid Operations", "Monitor and control grid components"),
            ("2", "EV Management", "Vehicle-to-Grid operations and fleet control"),
            ("3", "Energy Trading", "P2P energy marketplace and transactions"),
            ("4", "System Config", "Configure controllers and parameters"),
            ("5", "Diagnostics", "Run system diagnostics and health checks"),
            ("6", "Emergency Mode", "Activate emergency protocols"),
            ("Q", "Quit", "Exit AutoGrid Controller")
        ]
        
        for key, title, desc in menu_items:
            key_color = Colors.BRIGHT_GREEN if key.isdigit() else Colors.BRIGHT_RED
            print(f"{Colors.WHITE}│ {key_color}[{key}]{Colors.WHITE} {title:<15} - {Colors.DIM_WHITE}{desc:<40}{Colors.WHITE} │{Colors.RESET}")
            
        print(f"{Colors.BRIGHT_WHITE}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}")
        
    def update_energy_data(self):
        """Simulate energy data updates from embedded controllers"""
        import random
        
        # Simulate time-based patterns
        hour = datetime.now().hour
        solar_factor = max(0, (hour - 6) / 6) if 6 <= hour <= 12 else max(0, (18 - hour) / 6) if 12 < hour <= 18 else 0
        
        self.energy_data = {
            'solar_generation': 25 * solar_factor * (0.8 + 0.4 * random.random()),
            'wind_generation': 15 * (0.3 + 0.7 * random.random()),
            'battery_soc': 75 + 20 * (random.random() - 0.5),
            'total_demand': 60 * (0.7 + 0.6 * random.random()),
            'grid_connected': True,
            'grid_status': 'STABLE'
        }
        
        self.energy_data['net_flow'] = (self.energy_data['solar_generation'] + 
                                       self.energy_data['wind_generation'] - 
                                       self.energy_data['total_demand'])
        
    def initialize_ev_fleet(self):
        """Initialize EV fleet data"""
        import random
        
        ev_types = ['Model_S', 'Leaf', 'e-Golf', 'Bolt', 'i3', 'ID4', 'Taycan', 'EQS']
        modes = ['IDLE', 'CHARGING', 'V2G_DISCHARGE', 'SCHEDULED']
        
        for i in range(12):
            connected = random.choice([True, True, True, False])  # 75% connected
            mode = random.choice(modes) if connected else 'OFFLINE'
            
            self.ev_fleet.append({
                'id': f"EV_{i+1:03d}",
                'type': random.choice(ev_types),
                'connected': connected,
                'status': 'ONLINE' if connected else 'OFFLINE',
                'soc': random.uniform(20, 95) if connected else 0,
                'power': random.uniform(-20, 20) if mode == 'V2G_DISCHARGE' else random.uniform(0, 11) if mode == 'CHARGING' else 0,
                'mode': mode,
                'priority': random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
            })
            
    def get_current_alerts(self):
        """Get current system alerts"""
        alerts = []
        
        if self.energy_data.get('battery_soc', 100) < 20:
            alerts.append({
                'level': 'WARNING',
                'message': f"Low battery level: {self.energy_data['battery_soc']:.1f}%"
            })
            
        if abs(self.energy_data.get('net_flow', 0)) > 15:
            flow_type = "excess generation" if self.energy_data['net_flow'] > 0 else "energy deficit"
            alerts.append({
                'level': 'INFO',
                'message': f"Grid imbalance detected: {flow_type} of {abs(self.energy_data['net_flow']):.1f} kW"
            })
            
        # Check for offline subsystems
        offline_systems = [name for name, status in self.subsystems.items() if status == SystemStatus.ERROR]
        if offline_systems:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"Subsystem error: {', '.join(offline_systems)}"
            })
            
        return alerts
        
    def handle_menu_selection(self, choice):
        """Handle menu selections"""
        self.clear_screen()
        
        if choice == '1':
            self.grid_operations_menu()
        elif choice == '2':
            self.ev_management_menu()
        elif choice == '3':
            self.energy_trading_menu()
        elif choice == '4':
            self.system_config_menu()
        elif choice == '5':
            self.diagnostics_menu()
        elif choice == '6':
            self.emergency_mode_menu()
        elif choice.upper() == 'Q':
            self.shutdown()
        else:
            print(f"{Colors.RED}Invalid selection. Press Enter to continue...{Colors.RESET}")
            input()
            
    def grid_operations_menu(self):
        """Grid operations submenu"""
        self.print_header()
        print(f"{Colors.BRIGHT_GREEN}═══ GRID OPERATIONS ═══{Colors.RESET}\n")
        
        print(f"{Colors.WHITE}Real-time Grid Control & Monitoring{Colors.RESET}")
        print(f"{Colors.DIM_WHITE}Interfacing with embedded C++ controllers...{Colors.RESET}\n")
        
        # Simulate calling C++ embedded controller
        print(f"{Colors.GREEN}[C++] Embedded Controller Status: ACTIVE{Colors.RESET}")
        print(f"{Colors.GREEN}[C++] Real-time sensor data: OK{Colors.RESET}")
        print(f"{Colors.GREEN}[C++] Actuator control: RESPONSIVE{Colors.RESET}\n")
        
        self.print_energy_metrics()
        print()
        
        print(f"{Colors.BRIGHT_WHITE}Grid Control Options:{Colors.RESET}")
        print(f"{Colors.WHITE}[1] Load Balancing    [2] Emergency Shutdown    [3] Generator Control{Colors.RESET}")
        print(f"{Colors.WHITE}[4] Battery Control   [5] Solar Tracking        [6] Back to Main Menu{Colors.RESET}")
        
        choice = input(f"{Colors.BRIGHT_GREEN}Select option: {Colors.RESET}")
        
        if choice == '6':
            return
        else:
            print(f"{Colors.GREEN}Executing grid operation {choice}...{Colors.RESET}")
            time.sleep(2)
            input(f"{Colors.WHITE}Press Enter to continue...{Colors.RESET}")
            
    def ev_management_menu(self):
        """EV management submenu"""
        self.print_header()
        print(f"{Colors.BRIGHT_GREEN}═══ EV FLEET & V2G MANAGEMENT ═══{Colors.RESET}\n")
        
        print(f"{Colors.WHITE}Vehicle-to-Grid Operations & Fleet Control{Colors.RESET}")
        print(f"{Colors.DIM_WHITE}Coordinating with Rust distributed systems...{Colors.RESET}\n")
        
        # Simulate Rust coordination
        print(f"{Colors.GREEN}[Rust] Consensus protocol: ACTIVE{Colors.RESET}")
        print(f"{Colors.GREEN}[Rust] P2P communication: SECURE{Colors.RESET}")
        print(f"{Colors.GREEN}[Rust] Load balancing: OPTIMIZED{Colors.RESET}\n")
        
        self.print_ev_status()
        print()
        
        print(f"{Colors.BRIGHT_WHITE}EV Control Options:{Colors.RESET}")
        print(f"{Colors.WHITE}[1] V2G Activate      [2] Smart Scheduling      [3] Fleet Priority{Colors.RESET}")
        print(f"{Colors.WHITE}[4] Charge Control    [5] Roaming Access        [6] Back to Main Menu{Colors.RESET}")
        
        choice = input(f"{Colors.BRIGHT_GREEN}Select option: {Colors.RESET}")
        
        if choice == '6':
            return
        else:
            print(f"{Colors.GREEN}Executing EV operation {choice}...{Colors.RESET}")
            time.sleep(2)
            input(f"{Colors.WHITE}Press Enter to continue...{Colors.RESET}")
            
    def energy_trading_menu(self):
        """Energy trading submenu"""
        self.print_header()
        print(f"{Colors.BRIGHT_GREEN}═══ ENERGY TRADING MARKETPLACE ═══{Colors.RESET}\n")
        
        print(f"{Colors.WHITE}Peer-to-Peer Energy Trading & Market Operations{Colors.RESET}")
        print(f"{Colors.DIM_WHITE}Powered by distributed ledger technology...{Colors.RESET}\n")
        
        print(f"{Colors.GREEN}[Rust] Blockchain interface: CONNECTED{Colors.RESET}")
        print(f"{Colors.GREEN}[Rust] Smart contracts: ACTIVE{Colors.RESET}")
        print(f"{Colors.GREEN}[Python] Market analysis: RUNNING{Colors.RESET}\n")
        
        # Mock trading data
        print(f"{Colors.BRIGHT_WHITE}┌─ ACTIVE MARKET ─────────────────────────────────────────────────────────────┐{Colors.RESET}")
        print(f"{Colors.WHITE}│ Current Price: {Colors.GREEN}$0.152/kWh{Colors.WHITE}    │ 24h Volume: {Colors.WHITE}1,247 kWh{Colors.WHITE}    │ Active Traders: {Colors.WHITE}23{Colors.WHITE} │{Colors.RESET}")
        print(f"{Colors.WHITE}│ Buy Orders: {Colors.GREEN}15{Colors.WHITE}             │ Sell Orders: {Colors.RED}18{Colors.WHITE}           │ Last Trade: {Colors.WHITE}$0.151{Colors.WHITE}  │{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}")
        print()
        
        print(f"{Colors.BRIGHT_WHITE}Trading Options:{Colors.RESET}")
        print(f"{Colors.WHITE}[1] Place Buy Order   [2] Place Sell Order     [3] View Order Book{Colors.RESET}")
        print(f"{Colors.WHITE}[4] Trading History   [5] Market Analytics     [6] Back to Main Menu{Colors.RESET}")
        
        choice = input(f"{Colors.BRIGHT_GREEN}Select option: {Colors.RESET}")
        
        if choice == '6':
            return
        else:
            print(f"{Colors.GREEN}Executing trading operation {choice}...{Colors.RESET}")
            time.sleep(2)
            input(f"{Colors.WHITE}Press Enter to continue...{Colors.RESET}")
            
    def system_config_menu(self):
        """System configuration submenu"""
        self.print_header()
        print(f"{Colors.BRIGHT_GREEN}═══ SYSTEM CONFIGURATION ═══{Colors.RESET}\n")
        
        print(f"{Colors.WHITE}Multi-Language System Configuration{Colors.RESET}")
        print(f"{Colors.DIM_WHITE}Configure C++, Rust, and Python components...{Colors.RESET}\n")
        
        # Show component status
        components = [
            ("C++ Embedded Controller", "ARM Cortex-M4", "v2.1.3", "ACTIVE"),
            ("Rust Coordinator", "Consensus Engine", "v1.8.1", "ACTIVE"),
            ("Python AI Engine", "ML Forecasting", "v3.2.0", "ACTIVE"),
            ("Communication Layer", "TLS/Encrypted", "v1.5.2", "SECURE")
        ]
        
        print(f"{Colors.BRIGHT_WHITE}┌─ COMPONENT STATUS ──────────────────────────────────────────────────────────┐{Colors.RESET}")
        print(f"{Colors.WHITE}│ {'Component':<22} {'Platform':<15} {'Version':<10} {'Status':<8}        │{Colors.RESET}")
        print(f"{Colors.DIM_WHITE}│ {'─'*22} {'─'*15} {'─'*10} {'─'*8}        │{Colors.RESET}")
        
        for comp, platform, version, status in components:
            status_color = Colors.GREEN if status == "ACTIVE" else Colors.WHITE
            print(f"{Colors.WHITE}│ {comp:<22} {platform:<15} {version:<10} {status_color}{status:<8}{Colors.WHITE}        │{Colors.RESET}")
            
        print(f"{Colors.BRIGHT_WHITE}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}")
        print()
        
        print(f"{Colors.BRIGHT_WHITE}Configuration Options:{Colors.RESET}")
        print(f"{Colors.WHITE}[1] Update Firmware   [2] Network Settings     [3] Security Config{Colors.RESET}")
        print(f"{Colors.WHITE}[4] Performance Tuning [5] Debug Mode          [6] Back to Main Menu{Colors.RESET}")
        
        choice = input(f"{Colors.BRIGHT_GREEN}Select option: {Colors.RESET}")
        
        if choice == '6':
            return
        else:
            print(f"{Colors.GREEN}Configuring system component {choice}...{Colors.RESET}")
            time.sleep(2)
            input(f"{Colors.WHITE}Press Enter to continue...{Colors.RESET}")
            
    def diagnostics_menu(self):
        """System diagnostics submenu"""
        self.print_header()
        print(f"{Colors.BRIGHT_GREEN}═══ SYSTEM DIAGNOSTICS ═══{Colors.RESET}\n")
        
        print(f"{Colors.WHITE}Comprehensive System Health Check{Colors.RESET}")
        print(f"{Colors.DIM_WHITE}Running multi-language diagnostic suite...{Colors.RESET}\n")
        
        # Simulate diagnostic tests
        tests = [
            ("Embedded Controller Health", "C++", True),
            ("Consensus Algorithm Status", "Rust", True),
            ("AI Model Performance", "Python", True),
            ("Memory Leak Detection", "System", True),
            ("Network Connectivity", "System", True),
            ("Database Integrity", "System", False),
            ("Sensor Calibration", "C++", True),
            ("Cryptographic Functions", "Rust", True)
        ]
        
        print(f"{Colors.BRIGHT_WHITE}┌─ DIAGNOSTIC RESULTS ────────────────────────────────────────────────────────┐{Colors.RESET}")
        print(f"{Colors.WHITE}│ {'Test Name':<28} {'Language':<8} {'Status':<8} {'Result':<15}        │{Colors.RESET}")
        print(f"{Colors.DIM_WHITE}│ {'─'*28} {'─'*8} {'─'*8} {'─'*15}        │{Colors.RESET}")
        
        for test_name, language, passed in tests:
            status_color = Colors.GREEN if passed else Colors.RED
            status_text = "PASS" if passed else "FAIL"
            result_text = "✓ Healthy" if passed else "✗ Issue detected"
            
            print(f"{Colors.WHITE}│ {test_name:<28} {language:<8} {status_color}{status_text:<8}{Colors.WHITE} {status_color}{result_text:<15}{Colors.WHITE}        │{Colors.RESET}")
            
        print(f"{Colors.BRIGHT_WHITE}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}")
        print()
        
        print(f"{Colors.BRIGHT_WHITE}Diagnostic Options:{Colors.RESET}")
        print(f"{Colors.WHITE}[1] Full System Scan  [2] Component Test       [3] Performance Benchmark{Colors.RESET}")
        print(f"{Colors.WHITE}[4] Error Log Review  [5] Generate Report      [6] Back to Main Menu{Colors.RESET}")
        
        choice = input(f"{Colors.BRIGHT_GREEN}Select option: {Colors.RESET}")
        
        if choice == '6':
            return
        else:
            print(f"{Colors.GREEN}Running diagnostic {choice}...{Colors.RESET}")
            time.sleep(3)
            input(f"{Colors.WHITE}Press Enter to continue...{Colors.RESET}")
            
    def emergency_mode_menu(self):
        """Emergency mode submenu"""
        self.print_header()
        print(f"{Colors.BRIGHT_RED}═══ EMERGENCY MODE ═══{Colors.RESET}\n")
        
        print(f"{Colors.BRIGHT_RED}⚠ CAUTION: Emergency protocols activated ⚠{Colors.RESET}")
        print(f"{Colors.WHITE}Critical system control and safety procedures{Colors.RESET}\n")
        
        print(f"{Colors.RED}[ALERT] Manual override mode: ENABLED{Colors.RESET}")
        print(f"{Colors.RED}[ALERT] Automated safety systems: ACTIVE{Colors.RESET}")
        print(f"{Colors.RED}[ALERT] Grid isolation capability: READY{Colors.RESET}\n")
        
        print(f"{Colors.BRIGHT_WHITE}┌─ EMERGENCY PROTOCOLS ───────────────────────────────────────────────────────┐{Colors.RESET}")
        emergency_options = [
            ("Grid Disconnect", "Isolate microgrid from main grid", "ARMED"),
            ("Load Shedding", "Emergency load reduction sequence", "READY"),
            ("Battery Reserve", "Activate emergency battery mode", "STANDBY"),
            ("Generator Start", "Start backup diesel generators", "READY"),
            ("EV Emergency Discharge", "Force V2G discharge all EVs", "ARMED"),
            ("System Shutdown", "Complete system emergency stop", "ARMED")
        ]
        
        for i, (action, description, status) in enumerate(emergency_options, 1):
            status_color = Colors.RED if status == "ARMED" else Colors.WHITE
            print(f"{Colors.WHITE}│ [{i}] {action:<18} - {description:<35} [{status_color}{status}{Colors.WHITE}] │{Colors.RESET}")
            
        print(f"{Colors.BRIGHT_WHITE}└─────────────────────────────────────────────────────────────────────────────┘{Colors.RESET}")
        print()
        
        print(f"{Colors.BRIGHT_RED}[7] Exit Emergency Mode{Colors.RESET}")
        
        choice = input(f"{Colors.BRIGHT_RED}SELECT EMERGENCY ACTION: {Colors.RESET}")
        
        if choice == '7':
            return
        elif choice in ['1', '2', '3', '4', '5', '6']:
            confirm = input(f"{Colors.BRIGHT_RED}CONFIRM EMERGENCY ACTION {choice} (type YES): {Colors.RESET}")
            if confirm == "YES":
                print(f"{Colors.BRIGHT_RED}EXECUTING EMERGENCY PROTOCOL {choice}...{Colors.RESET}")
                time.sleep(3)
                print(f"{Colors.GREEN}Emergency action completed.{Colors.RESET}")
            else:
                print(f"{Colors.WHITE}Emergency action cancelled.{Colors.RESET}")
            input(f"{Colors.WHITE}Press Enter to continue...{Colors.RESET}")
            
    def start_subsystems(self):
        """Start all subsystems"""
        print(f"{Colors.WHITE}Starting AutoGrid subsystems...{Colors.RESET}")
        
        subsystem_info = [
            ("embedded_controller", "C++ Embedded Controller", "Initializing hardware interfaces..."),
            ("rust_coordinator", "Rust Coordination Layer", "Establishing consensus network..."),
            ("ai_forecaster", "Python AI Engine", "Loading ML models..."),
            ("ev_manager", "EV Management System", "Scanning for connected vehicles...")
        ]
        
        for system_id, name, message in subsystem_info:
            print(f"{Colors.DIM_WHITE}{message}{Colors.RESET}")
            time.sleep(1)
            self.subsystems[system_id] = SystemStatus.ONLINE
            print(f"{Colors.GREEN}✓ {name} online{Colors.RESET}")
            
        print(f"{Colors.BRIGHT_GREEN}All subsystems operational.{Colors.RESET}\n")
        time.sleep(2)
        
    def shutdown(self):
        """Shutdown the system"""
        print(f"{Colors.BRIGHT_RED}Shutting down AutoGrid Controller...{Colors.RESET}")
        
        for system_id in self.subsystems:
            self.subsystems[system_id] = SystemStatus.OFFLINE
            
        print(f"{Colors.WHITE}All subsystems stopped.{Colors.RESET}")
        print(f"{Colors.GREEN}AutoGrid Controller shutdown complete.{Colors.RESET}")
        self.running = False
        
    def run(self):
        """Main application loop"""
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())
        
        self.clear_screen()
        self.print_header()
        self.start_subsystems()
        
        while self.running:
            try:
                self.clear_screen()
                self.update_energy_data()
                
                self.print_header()
                self.print_system_status()
                print()
                self.print_energy_metrics()
                print()
                self.print_ev_status()
                print()
                self.print_alerts()
                print()
                self.print_menu()
                
                choice = input(f"{Colors.BRIGHT_GREEN}Enter selection: {Colors.RESET}")
                self.handle_menu_selection(choice)
                
            except KeyboardInterrupt:
                self.shutdown()
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                input(f"{Colors.WHITE}Press Enter to continue...{Colors.RESET}")

if __name__ == "__main__":
    controller = AutoGridController()
    controller.run()