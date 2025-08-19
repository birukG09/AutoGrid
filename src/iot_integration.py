#!/usr/bin/env python3
"""
AutoGrid IoT Integration Engine
Real-time sensor data and device management
"""

import json
import time
import socket
import threading
import struct
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import hashlib
import random
import numpy as np
from collections import deque, defaultdict

class DeviceType(Enum):
    SMART_METER = "smart_meter"
    SENSOR_NODE = "sensor_node"
    RELAY_SWITCH = "relay_switch"
    WEATHER_STATION = "weather_station"
    EV_CHARGER = "ev_charger"
    BATTERY_BMS = "battery_bms"
    INVERTER = "inverter"
    TRANSFORMER = "transformer"

class SensorType(Enum):
    VOLTAGE = "voltage"
    CURRENT = "current"
    POWER = "power"
    FREQUENCY = "frequency"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    FLOW_RATE = "flow_rate"
    VIBRATION = "vibration"
    GAS = "gas_concentration"

@dataclass
class IoTDevice:
    device_id: str
    device_type: DeviceType
    location: str
    ip_address: str
    protocol: str
    status: str
    last_seen: datetime
    firmware_version: str
    battery_level: Optional[float]
    signal_strength: Optional[float]
    sensors: List[SensorType]
    capabilities: List[str]

@dataclass
class SensorReading:
    device_id: str
    sensor_type: SensorType
    timestamp: datetime
    value: float
    unit: str
    quality: float  # 0-1 quality score
    calibration_offset: float
    raw_value: float

@dataclass
class DeviceCommand:
    command_id: str
    device_id: str
    command_type: str
    parameters: Dict[str, Any]
    timestamp: datetime
    priority: int
    timeout: float
    status: str

class ModbusSimulator:
    """Simplified Modbus RTU/TCP simulator"""
    
    def __init__(self):
        self.registers = defaultdict(lambda: defaultdict(int))
        self.coils = defaultdict(lambda: defaultdict(bool))
        
    def read_holding_registers(self, device_id: str, start_addr: int, count: int) -> List[int]:
        """Read holding registers from device"""
        values = []
        for addr in range(start_addr, start_addr + count):
            values.append(self.registers[device_id][addr])
        return values
    
    def write_holding_register(self, device_id: str, addr: int, value: int) -> bool:
        """Write single holding register"""
        self.registers[device_id][addr] = value
        return True
    
    def read_coils(self, device_id: str, start_addr: int, count: int) -> List[bool]:
        """Read coil status"""
        values = []
        for addr in range(start_addr, start_addr + count):
            values.append(self.coils[device_id][addr])
        return values
    
    def write_coil(self, device_id: str, addr: int, value: bool) -> bool:
        """Write single coil"""
        self.coils[device_id][addr] = value
        return True

class DNP3Simulator:
    """Simplified DNP3 protocol simulator"""
    
    def __init__(self):
        self.analog_inputs = defaultdict(lambda: defaultdict(float))
        self.binary_inputs = defaultdict(lambda: defaultdict(bool))
        self.binary_outputs = defaultdict(lambda: defaultdict(bool))
        
    def read_analog_input(self, device_id: str, point: int) -> float:
        """Read analog input point"""
        return self.analog_inputs[device_id][point]
    
    def read_binary_input(self, device_id: str, point: int) -> bool:
        """Read binary input point"""
        return self.binary_inputs[device_id][point]
    
    def operate_binary_output(self, device_id: str, point: int, value: bool) -> bool:
        """Operate binary output"""
        self.binary_outputs[device_id][point] = value
        return True

class WeatherDataCollector:
    """Advanced weather data collection and processing"""
    
    def __init__(self):
        self.weather_stations = {}
        self.historical_data = deque(maxlen=10000)
        self.forecasts = {}
        
    def add_weather_station(self, station_id: str, location: Dict[str, float]):
        """Add weather monitoring station"""
        self.weather_stations[station_id] = {
            'location': location,  # {'lat': float, 'lon': float, 'elevation': float}
            'sensors': [
                SensorType.TEMPERATURE,
                SensorType.HUMIDITY,
                SensorType.PRESSURE,
                SensorType.FLOW_RATE  # wind speed
            ],
            'last_reading': None
        }
        print(f"[IoT] Weather station {station_id} added at {location}")
    
    def collect_weather_data(self, station_id: str) -> Dict[str, SensorReading]:
        """Collect comprehensive weather data"""
        if station_id not in self.weather_stations:
            return {}
        
        station = self.weather_stations[station_id]
        current_time = datetime.now()
        hour = current_time.hour
        
        # Simulate realistic weather patterns
        readings = {}
        
        # Temperature (seasonal and daily variation)
        base_temp = 20 + 10 * np.sin(2 * np.pi * (current_time.timetuple().tm_yday - 80) / 365)
        daily_temp = base_temp + 8 * np.sin(2 * np.pi * (hour - 6) / 24)
        temp_noise = np.random.normal(0, 1.5)
        temperature = daily_temp + temp_noise
        
        readings['temperature'] = SensorReading(
            device_id=station_id,
            sensor_type=SensorType.TEMPERATURE,
            timestamp=current_time,
            value=round(temperature, 2),
            unit="°C",
            quality=0.95 + np.random.uniform(-0.05, 0.05),
            calibration_offset=0.1,
            raw_value=temperature + 0.1
        )
        
        # Humidity (inverse relationship with temperature)
        base_humidity = 60 - (temperature - 20) * 1.5
        humidity_noise = np.random.normal(0, 5)
        humidity = max(20, min(95, base_humidity + humidity_noise))
        
        readings['humidity'] = SensorReading(
            device_id=station_id,
            sensor_type=SensorType.HUMIDITY,
            timestamp=current_time,
            value=round(humidity, 1),
            unit="%RH",
            quality=0.90 + np.random.uniform(-0.05, 0.05),
            calibration_offset=-0.2,
            raw_value=humidity - 0.2
        )
        
        # Atmospheric pressure
        base_pressure = 1013.25 + np.sin(2 * np.pi * hour / 24) * 2
        pressure_noise = np.random.normal(0, 1.5)
        pressure = base_pressure + pressure_noise
        
        readings['pressure'] = SensorReading(
            device_id=station_id,
            sensor_type=SensorType.PRESSURE,
            timestamp=current_time,
            value=round(pressure, 2),
            unit="hPa",
            quality=0.98 + np.random.uniform(-0.02, 0.02),
            calibration_offset=0.5,
            raw_value=pressure + 0.5
        )
        
        # Wind speed (flow rate sensor)
        base_wind = 5 + 3 * np.sin(2 * np.pi * hour / 24)
        wind_noise = np.random.exponential(2)
        wind_speed = max(0, base_wind + wind_noise)
        
        readings['wind_speed'] = SensorReading(
            device_id=station_id,
            sensor_type=SensorType.FLOW_RATE,
            timestamp=current_time,
            value=round(wind_speed, 1),
            unit="m/s",
            quality=0.85 + np.random.uniform(-0.10, 0.10),
            calibration_offset=0.3,
            raw_value=wind_speed + 0.3
        )
        
        # Store historical data
        for reading in readings.values():
            self.historical_data.append(reading)
        
        station['last_reading'] = current_time
        return readings

class SmartMeterNetwork:
    """Advanced smart meter data collection"""
    
    def __init__(self):
        self.meters = {}
        self.readings_buffer = deque(maxlen=50000)
        self.demand_profiles = {}
        
    def add_smart_meter(self, meter_id: str, meter_type: str, location: str):
        """Add smart meter to network"""
        self.meters[meter_id] = {
            'type': meter_type,  # residential, commercial, industrial
            'location': location,
            'sensors': [
                SensorType.VOLTAGE,
                SensorType.CURRENT,
                SensorType.POWER,
                SensorType.FREQUENCY
            ],
            'tariff_schedule': self._generate_tariff_schedule(),
            'demand_profile': self._generate_demand_profile(meter_type)
        }
        print(f"[IoT] Smart meter {meter_id} ({meter_type}) added at {location}")
    
    def _generate_tariff_schedule(self) -> Dict[str, float]:
        """Generate time-of-use tariff schedule"""
        return {
            'peak': 0.25,      # 17:00-21:00
            'off_peak': 0.12,  # 22:00-06:00
            'standard': 0.18   # 07:00-16:00
        }
    
    def _generate_demand_profile(self, meter_type: str) -> Dict[int, float]:
        """Generate typical demand profile by hour"""
        if meter_type == 'residential':
            # Typical home load pattern
            base_profile = [
                0.5, 0.4, 0.3, 0.3, 0.4, 0.6,  # 00:00-05:00
                0.8, 1.2, 1.0, 0.7, 0.6, 0.5,  # 06:00-11:00
                0.6, 0.7, 0.6, 0.5, 0.8, 1.5,  # 12:00-17:00
                1.8, 2.0, 1.7, 1.3, 0.9, 0.7   # 18:00-23:00
            ]
        elif meter_type == 'commercial':
            # Office building pattern
            base_profile = [
                0.3, 0.2, 0.2, 0.2, 0.3, 0.5,  # 00:00-05:00
                0.8, 1.5, 2.0, 2.2, 2.0, 1.8,  # 06:00-11:00
                1.5, 1.8, 2.1, 2.0, 1.9, 1.6,  # 12:00-17:00
                1.2, 0.8, 0.6, 0.5, 0.4, 0.3   # 18:00-23:00
            ]
        else:  # industrial
            # 24/7 industrial load
            base_profile = [
                0.8, 0.7, 0.7, 0.8, 0.9, 1.0,  # 00:00-05:00
                1.2, 1.4, 1.5, 1.5, 1.4, 1.3,  # 06:00-11:00
                1.2, 1.3, 1.4, 1.5, 1.4, 1.3,  # 12:00-17:00
                1.2, 1.1, 1.0, 0.9, 0.8, 0.8   # 18:00-23:00
            ]
        
        return {hour: multiplier for hour, multiplier in enumerate(base_profile)}
    
    def collect_meter_readings(self, meter_id: str) -> Dict[str, SensorReading]:
        """Collect smart meter readings"""
        if meter_id not in self.meters:
            return {}
        
        meter = self.meters[meter_id]
        current_time = datetime.now()
        hour = current_time.hour
        
        # Get demand multiplier for current hour
        demand_multiplier = meter['demand_profile'][hour]
        
        # Add some randomness
        demand_multiplier *= (1 + np.random.normal(0, 0.1))
        
        readings = {}
        
        # Voltage (should be relatively stable)
        nominal_voltage = 230.0  # Volts
        voltage_variation = np.random.normal(0, 2.0)
        voltage = nominal_voltage + voltage_variation
        
        readings['voltage'] = SensorReading(
            device_id=meter_id,
            sensor_type=SensorType.VOLTAGE,
            timestamp=current_time,
            value=round(voltage, 2),
            unit="V",
            quality=0.99,
            calibration_offset=0.5,
            raw_value=voltage + 0.5
        )
        
        # Power consumption based on demand profile
        base_power = 5.0  # kW
        if meter['type'] == 'commercial':
            base_power = 50.0
        elif meter['type'] == 'industrial':
            base_power = 200.0
        
        power = base_power * demand_multiplier
        power_noise = np.random.normal(0, power * 0.05)
        power = max(0, power + power_noise)
        
        readings['power'] = SensorReading(
            device_id=meter_id,
            sensor_type=SensorType.POWER,
            timestamp=current_time,
            value=round(power, 3),
            unit="kW",
            quality=0.95,
            calibration_offset=0.01,
            raw_value=power + 0.01
        )
        
        # Current from power and voltage
        current = power * 1000 / voltage  # P = V * I
        current_noise = np.random.normal(0, current * 0.02)
        current = max(0, current + current_noise)
        
        readings['current'] = SensorReading(
            device_id=meter_id,
            sensor_type=SensorType.CURRENT,
            timestamp=current_time,
            value=round(current, 2),
            unit="A",
            quality=0.96,
            calibration_offset=0.05,
            raw_value=current + 0.05
        )
        
        # Frequency (should be very stable)
        nominal_frequency = 50.0  # Hz
        frequency_variation = np.random.normal(0, 0.05)
        frequency = nominal_frequency + frequency_variation
        
        readings['frequency'] = SensorReading(
            device_id=meter_id,
            sensor_type=SensorType.FREQUENCY,
            timestamp=current_time,
            value=round(frequency, 3),
            unit="Hz",
            quality=0.995,
            calibration_offset=0.001,
            raw_value=frequency + 0.001
        )
        
        # Store readings
        for reading in readings.values():
            self.readings_buffer.append(reading)
        
        return readings

class IoTDeviceManager:
    """Comprehensive IoT device management"""
    
    def __init__(self):
        self.devices = {}
        self.device_registry = {}
        self.command_queue = deque()
        self.communication_protocols = {
            'modbus': ModbusSimulator(),
            'dnp3': DNP3Simulator()
        }
        self.weather_collector = WeatherDataCollector()
        self.meter_network = SmartMeterNetwork()
        self.running = False
        
    def start(self):
        """Start IoT device management"""
        self.running = True
        
        # Start data collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        # Start command processing thread
        self.command_thread = threading.Thread(target=self._command_loop)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        print("[IoT] Device management started")
    
    def stop(self):
        """Stop IoT device management"""
        self.running = False
        print("[IoT] Device management stopped")
    
    def register_device(self, device: IoTDevice) -> bool:
        """Register new IoT device"""
        try:
            self.devices[device.device_id] = device
            self.device_registry[device.device_id] = {
                'registered_at': datetime.now(),
                'total_readings': 0,
                'last_command': None,
                'status_history': []
            }
            
            # Initialize device-specific collectors
            if device.device_type == DeviceType.WEATHER_STATION:
                location = {'lat': 40.7128, 'lon': -74.0060, 'elevation': 10}  # NYC
                self.weather_collector.add_weather_station(device.device_id, location)
            elif device.device_type == DeviceType.SMART_METER:
                meter_type = 'residential'  # Default
                self.meter_network.add_smart_meter(device.device_id, meter_type, device.location)
            
            print(f"[IoT] Device {device.device_id} ({device.device_type.value}) registered")
            return True
            
        except Exception as e:
            print(f"[IoT] Device registration failed: {e}")
            return False
    
    def send_command(self, command: DeviceCommand) -> bool:
        """Send command to IoT device"""
        if command.device_id not in self.devices:
            return False
        
        command.status = "queued"
        self.command_queue.append(command)
        print(f"[IoT] Command {command.command_id} queued for {command.device_id}")
        return True
    
    def get_device_readings(self, device_id: str) -> Dict[str, SensorReading]:
        """Get latest readings from device"""
        if device_id not in self.devices:
            return {}
        
        device = self.devices[device_id]
        
        if device.device_type == DeviceType.WEATHER_STATION:
            return self.weather_collector.collect_weather_data(device_id)
        elif device.device_type == DeviceType.SMART_METER:
            return self.meter_network.collect_meter_readings(device_id)
        else:
            return self._simulate_generic_readings(device)
    
    def _simulate_generic_readings(self, device: IoTDevice) -> Dict[str, SensorReading]:
        """Simulate generic sensor readings"""
        readings = {}
        current_time = datetime.now()
        
        for sensor_type in device.sensors:
            if sensor_type == SensorType.TEMPERATURE:
                value = 25.0 + np.random.normal(0, 2.0)
                unit = "°C"
            elif sensor_type == SensorType.VIBRATION:
                value = np.random.exponential(0.5)
                unit = "mm/s"
            elif sensor_type == SensorType.GAS:
                value = np.random.uniform(0, 100)
                unit = "ppm"
            else:
                value = np.random.uniform(0, 100)
                unit = "units"
            
            readings[sensor_type.value] = SensorReading(
                device_id=device.device_id,
                sensor_type=sensor_type,
                timestamp=current_time,
                value=round(value, 2),
                unit=unit,
                quality=0.90 + np.random.uniform(-0.05, 0.05),
                calibration_offset=np.random.uniform(-0.1, 0.1),
                raw_value=value + np.random.uniform(-0.1, 0.1)
            )
        
        return readings
    
    def _collection_loop(self):
        """Main data collection loop"""
        while self.running:
            try:
                for device_id, device in self.devices.items():
                    if device.status == "online":
                        readings = self.get_device_readings(device_id)
                        if readings:
                            self.device_registry[device_id]['total_readings'] += len(readings)
                            device.last_seen = datetime.now()
                
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                print(f"[IoT] Collection error: {e}")
                time.sleep(10)
    
    def _command_loop(self):
        """Process device commands"""
        while self.running:
            try:
                if self.command_queue:
                    command = self.command_queue.popleft()
                    self._execute_command(command)
                
                time.sleep(1)
                
            except Exception as e:
                print(f"[IoT] Command processing error: {e}")
                time.sleep(5)
    
    def _execute_command(self, command: DeviceCommand):
        """Execute device command"""
        try:
            device = self.devices.get(command.device_id)
            if not device:
                command.status = "failed"
                return
            
            print(f"[IoT] Executing {command.command_type} on {command.device_id}")
            
            # Simulate command execution
            if command.command_type == "reboot":
                device.status = "rebooting"
                time.sleep(2)
                device.status = "online"
            elif command.command_type == "calibrate":
                print(f"[IoT] Calibrating sensors on {command.device_id}")
            elif command.command_type == "update_firmware":
                device.firmware_version = command.parameters.get("version", "1.0.0")
            
            command.status = "completed"
            self.device_registry[command.device_id]['last_command'] = command
            
        except Exception as e:
            print(f"[IoT] Command execution failed: {e}")
            command.status = "failed"
    
    def get_device_status(self) -> Dict:
        """Get overall device status"""
        online_devices = sum(1 for d in self.devices.values() if d.status == "online")
        total_readings = sum(reg['total_readings'] for reg in self.device_registry.values())
        
        device_types = defaultdict(int)
        for device in self.devices.values():
            device_types[device.device_type.value] += 1
        
        return {
            'total_devices': len(self.devices),
            'online_devices': online_devices,
            'total_readings': total_readings,
            'device_types': dict(device_types),
            'data_collection_active': self.running
        }
    
    def get_device_list(self) -> List[IoTDevice]:
        """Get list of all registered devices"""
        return list(self.devices.values())

# Global IoT manager
iot_manager = None

def init_iot_system():
    """Initialize IoT device management"""
    global iot_manager
    if iot_manager is None:
        iot_manager = IoTDeviceManager()
        iot_manager.start()
        
        # Register some demo devices
        demo_devices = [
            IoTDevice(
                device_id="WS001",
                device_type=DeviceType.WEATHER_STATION,
                location="Grid Center",
                ip_address="192.168.1.100",
                protocol="modbus",
                status="online",
                last_seen=datetime.now(),
                firmware_version="2.1.0",
                battery_level=85.0,
                signal_strength=95.0,
                sensors=[SensorType.TEMPERATURE, SensorType.HUMIDITY, SensorType.PRESSURE, SensorType.FLOW_RATE],
                capabilities=["weather_forecast", "alert_generation"]
            ),
            IoTDevice(
                device_id="SM001",
                device_type=DeviceType.SMART_METER,
                location="Building A",
                ip_address="192.168.1.101",
                protocol="dnp3",
                status="online",
                last_seen=datetime.now(),
                firmware_version="1.5.2",
                battery_level=None,
                signal_strength=88.0,
                sensors=[SensorType.VOLTAGE, SensorType.CURRENT, SensorType.POWER, SensorType.FREQUENCY],
                capabilities=["demand_response", "tariff_management"]
            ),
            IoTDevice(
                device_id="EVC001",
                device_type=DeviceType.EV_CHARGER,
                location="Parking Lot",
                ip_address="192.168.1.102",
                protocol="modbus",
                status="online",
                last_seen=datetime.now(),
                firmware_version="3.0.1",
                battery_level=None,
                signal_strength=92.0,
                sensors=[SensorType.VOLTAGE, SensorType.CURRENT, SensorType.POWER, SensorType.TEMPERATURE],
                capabilities=["fast_charging", "v2g_support", "load_balancing"]
            )
        ]
        
        for device in demo_devices:
            iot_manager.register_device(device)
        
        print("[IoT] IoT integration system initialized")

def stop_iot_system():
    """Stop IoT system"""
    global iot_manager
    if iot_manager:
        iot_manager.stop()
        iot_manager = None

def get_iot_status():
    """Get IoT system status"""
    global iot_manager
    if iot_manager:
        return iot_manager.get_device_status()
    return {'running': False}

if __name__ == "__main__":
    print("AutoGrid IoT Integration Test")
    print("=============================")
    
    init_iot_system()
    
    try:
        # Run for 30 seconds
        time.sleep(30)
        
        # Get status
        status = get_iot_status()
        print(f"IoT Status: {status}")
        
        # Get device readings
        for device in iot_manager.get_device_list():
            readings = iot_manager.get_device_readings(device.device_id)
            print(f"\n{device.device_id} readings:")
            for sensor, reading in readings.items():
                print(f"  {sensor}: {reading.value} {reading.unit}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_iot_system()