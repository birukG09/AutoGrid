#!/usr/bin/env python3
"""
AutoGrid Cybersecurity Engine
Advanced threat detection and protection
"""

import hashlib
import hmac
import secrets
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import json
import re
import socket
import ipaddress
from collections import defaultdict, deque
import numpy as np

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AttackType(Enum):
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    MAN_IN_MIDDLE = "mitm"
    INJECTION = "injection"
    MALWARE = "malware"
    INSIDER_THREAT = "insider"
    PROTOCOL_ANOMALY = "protocol_anomaly"
    DATA_EXFILTRATION = "data_exfiltration"

@dataclass
class SecurityEvent:
    event_id: str
    timestamp: datetime
    source_ip: str
    target_component: str
    attack_type: AttackType
    threat_level: ThreatLevel
    description: str
    indicators: Dict
    remediation_actions: List[str]
    blocked: bool

@dataclass
class AccessAttempt:
    timestamp: datetime
    source_ip: str
    user_agent: str
    endpoint: str
    success: bool
    credentials: str
    session_id: Optional[str]

class NetworkMonitor:
    def __init__(self):
        self.connection_log = deque(maxlen=10000)
        self.failed_attempts = defaultdict(list)
        self.blocked_ips = set()
        self.rate_limits = defaultdict(deque)
        self.suspicious_patterns = []
        
    def log_connection(self, source_ip: str, port: int, protocol: str):
        """Log network connection"""
        self.connection_log.append({
            'timestamp': datetime.now(),
            'source_ip': source_ip,
            'port': port,
            'protocol': protocol
        })
        
        # Rate limiting check
        self._check_rate_limit(source_ip)
        
    def _check_rate_limit(self, source_ip: str):
        """Check for rate limiting violations"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Add current request
        self.rate_limits[source_ip].append(now)
        
        # Remove old requests
        while self.rate_limits[source_ip] and self.rate_limits[source_ip][0] < minute_ago:
            self.rate_limits[source_ip].popleft()
        
        # Check if rate limit exceeded
        if len(self.rate_limits[source_ip]) > 100:  # 100 requests per minute
            self.blocked_ips.add(source_ip)
            return True
        return False
    
    def detect_ddos(self) -> List[str]:
        """Detect potential DDoS attacks"""
        suspicious_ips = []
        now = datetime.now()
        last_minute = now - timedelta(minutes=1)
        
        # Count connections per IP in the last minute
        ip_counts = defaultdict(int)
        for conn in self.connection_log:
            if conn['timestamp'] > last_minute:
                ip_counts[conn['source_ip']] += 1
        
        # Flag IPs with excessive connections
        for ip, count in ip_counts.items():
            if count > 50:  # Threshold for DDoS detection
                suspicious_ips.append(ip)
                self.blocked_ips.add(ip)
        
        return suspicious_ips
    
    def detect_port_scan(self) -> List[str]:
        """Detect port scanning attempts"""
        suspicious_ips = []
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        
        # Track unique ports accessed per IP
        ip_ports = defaultdict(set)
        for conn in self.connection_log:
            if conn['timestamp'] > last_hour:
                ip_ports[conn['source_ip']].add(conn['port'])
        
        # Flag IPs accessing many different ports
        for ip, ports in ip_ports.items():
            if len(ports) > 20:  # Threshold for port scan detection
                suspicious_ips.append(ip)
        
        return suspicious_ips

class AuthenticationManager:
    def __init__(self):
        self.active_sessions = {}
        self.failed_logins = defaultdict(list)
        self.locked_accounts = set()
        self.session_timeout = 3600  # 1 hour
        self.max_failed_attempts = 5
        
    def create_session(self, user_id: str, source_ip: str) -> str:
        """Create authenticated session"""
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'source_ip': source_ip,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        return session_id
    
    def validate_session(self, session_id: str, source_ip: str) -> bool:
        """Validate session"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        now = datetime.now()
        
        # Check timeout
        if (now - session['last_activity']).seconds > self.session_timeout:
            del self.active_sessions[session_id]
            return False
        
        # Check IP consistency (simple session hijacking protection)
        if session['source_ip'] != source_ip:
            del self.active_sessions[session_id]
            return False
        
        # Update last activity
        session['last_activity'] = now
        return True
    
    def record_failed_login(self, user_id: str, source_ip: str):
        """Record failed login attempt"""
        now = datetime.now()
        self.failed_logins[user_id].append({
            'timestamp': now,
            'source_ip': source_ip
        })
        
        # Remove old attempts (last hour only)
        hour_ago = now - timedelta(hours=1)
        self.failed_logins[user_id] = [
            attempt for attempt in self.failed_logins[user_id]
            if attempt['timestamp'] > hour_ago
        ]
        
        # Check if account should be locked
        if len(self.failed_logins[user_id]) >= self.max_failed_attempts:
            self.locked_accounts.add(user_id)
    
    def is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked"""
        return user_id in self.locked_accounts
    
    def unlock_account(self, user_id: str):
        """Unlock account (admin action)"""
        self.locked_accounts.discard(user_id)
        if user_id in self.failed_logins:
            del self.failed_logins[user_id]

class ProtocolAnalyzer:
    def __init__(self):
        self.known_protocols = {
            'modbus': {'ports': [502], 'patterns': [b'\x00\x01', b'\x00\x03']},
            'dnp3': {'ports': [20000], 'patterns': [b'\x05\x64']},
            'iec61850': {'ports': [102], 'patterns': [b'\x68']},
            'mqtt': {'ports': [1883, 8883], 'patterns': [b'\x10', b'\x20']},
            'coap': {'ports': [5683], 'patterns': [b'\x40', b'\x50']}
        }
        self.message_buffer = deque(maxlen=1000)
        
    def analyze_message(self, data: bytes, source_ip: str, port: int) -> Optional[Dict]:
        """Analyze protocol message for anomalies"""
        self.message_buffer.append({
            'timestamp': datetime.now(),
            'data': data,
            'source_ip': source_ip,
            'port': port
        })
        
        # Check for malformed messages
        anomalies = []
        
        # Check message length
        if len(data) > 1024:  # Unusual message size
            anomalies.append("oversized_message")
        
        # Check for known protocol patterns
        protocol_detected = None
        for protocol, info in self.known_protocols.items():
            if port in info['ports']:
                protocol_detected = protocol
                if not any(pattern in data for pattern in info['patterns']):
                    anomalies.append("protocol_mismatch")
                break
        
        # Check for suspicious byte patterns
        if b'\x00' * 100 in data:  # Null byte flooding
            anomalies.append("null_flood")
        
        if len(set(data)) < 5 and len(data) > 50:  # Low entropy
            anomalies.append("low_entropy")
        
        if anomalies:
            return {
                'protocol': protocol_detected,
                'anomalies': anomalies,
                'severity': 'high' if len(anomalies) > 2 else 'medium'
            }
        
        return None

class MalwareDetector:
    def __init__(self):
        self.known_signatures = [
            b'Stuxnet',
            b'TRITON',
            b'CRASHOVERRIDE',
            b'INDUSTROYER',
            b'Havex',
            b'BlackEnergy'
        ]
        self.behavioral_patterns = {
            'excessive_modbus_writes': 50,
            'rapid_connection_attempts': 100,
            'unusual_file_access': 20
        }
        self.quarantined_files = set()
        
    def scan_data(self, data: bytes, source: str) -> Dict:
        """Scan data for malware signatures"""
        threats = []
        
        # Signature-based detection
        for signature in self.known_signatures:
            if signature in data:
                threats.append({
                    'type': 'signature_match',
                    'signature': signature.decode('utf-8', errors='ignore'),
                    'severity': 'critical'
                })
        
        # Heuristic detection
        entropy = self._calculate_entropy(data)
        if entropy > 7.5:  # High entropy might indicate packed/encrypted malware
            threats.append({
                'type': 'high_entropy',
                'entropy': entropy,
                'severity': 'medium'
            })
        
        # Check for suspicious strings
        suspicious_strings = [b'cmd.exe', b'powershell', b'eval(', b'exec(']
        for sus_str in suspicious_strings:
            if sus_str in data:
                threats.append({
                    'type': 'suspicious_string',
                    'string': sus_str.decode('utf-8', errors='ignore'),
                    'severity': 'high'
                })
        
        return {
            'source': source,
            'threats_found': len(threats),
            'threats': threats,
            'clean': len(threats) == 0
        }
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate probabilities and entropy
        length = len(data)
        entropy = 0
        for count in byte_counts:
            if count > 0:
                p = count / length
                entropy -= p * np.log2(p)
        
        return entropy

class SecurityOrchestrator:
    def __init__(self):
        self.network_monitor = NetworkMonitor()
        self.auth_manager = AuthenticationManager()
        self.protocol_analyzer = ProtocolAnalyzer()
        self.malware_detector = MalwareDetector()
        
        self.security_events = deque(maxlen=1000)
        self.running = False
        self.monitoring_thread = None
        
        # Security policies
        self.policies = {
            'auto_block_ddos': True,
            'quarantine_malware': True,
            'lock_brute_force': True,
            'alert_anomalies': True,
            'log_all_access': True
        }
        
    def start_monitoring(self):
        """Start security monitoring"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        print("[Security] Cybersecurity monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("[Security] Cybersecurity monitoring stopped")
    
    def _monitoring_loop(self):
        """Main security monitoring loop"""
        while self.running:
            try:
                # Check for DDoS attacks
                ddos_ips = self.network_monitor.detect_ddos()
                for ip in ddos_ips:
                    self._create_security_event(
                        ip, "network", AttackType.DDoS, ThreatLevel.HIGH,
                        f"DDoS attack detected from {ip}",
                        {"blocked": True}
                    )
                
                # Check for port scans
                scan_ips = self.network_monitor.detect_port_scan()
                for ip in scan_ips:
                    self._create_security_event(
                        ip, "network", AttackType.BRUTE_FORCE, ThreatLevel.MEDIUM,
                        f"Port scanning detected from {ip}",
                        {"ports_scanned": "multiple"}
                    )
                
                # Clean up expired sessions
                self._cleanup_expired_sessions()
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                print(f"[Security] Monitoring error: {e}")
                time.sleep(5)
    
    def _create_security_event(self, source_ip: str, target: str, attack_type: AttackType,
                             threat_level: ThreatLevel, description: str, indicators: Dict):
        """Create security event"""
        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            timestamp=datetime.now(),
            source_ip=source_ip,
            target_component=target,
            attack_type=attack_type,
            threat_level=threat_level,
            description=description,
            indicators=indicators,
            remediation_actions=self._get_remediation_actions(attack_type),
            blocked=source_ip in self.network_monitor.blocked_ips
        )
        
        self.security_events.append(event)
        
        # Auto-execute remediation if enabled
        if self.policies.get('auto_block_ddos') and attack_type == AttackType.DDoS:
            self.network_monitor.blocked_ips.add(source_ip)
        
        print(f"[Security] {threat_level.value.upper()} threat detected: {description}")
    
    def _get_remediation_actions(self, attack_type: AttackType) -> List[str]:
        """Get recommended remediation actions"""
        actions = {
            AttackType.DDoS: ["Block source IP", "Rate limit connections", "Enable DDoS protection"],
            AttackType.BRUTE_FORCE: ["Lock account", "Block IP temporarily", "Require MFA"],
            AttackType.MAN_IN_MIDDLE: ["Terminate connection", "Verify certificates", "Enable encryption"],
            AttackType.INJECTION: ["Sanitize inputs", "Update security patches", "Review logs"],
            AttackType.MALWARE: ["Quarantine file", "Run full scan", "Isolate system"],
            AttackType.INSIDER_THREAT: ["Review access logs", "Suspend account", "Investigate activity"],
            AttackType.PROTOCOL_ANOMALY: ["Validate protocol compliance", "Update firmware", "Monitor traffic"],
            AttackType.DATA_EXFILTRATION: ["Block data transfer", "Investigate access", "Audit permissions"]
        }
        return actions.get(attack_type, ["Manual investigation required"])
    
    def _cleanup_expired_sessions(self):
        """Clean up expired authentication sessions"""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.auth_manager.active_sessions.items():
            if (now - session['last_activity']).seconds > self.auth_manager.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.auth_manager.active_sessions[session_id]
    
    def process_network_event(self, source_ip: str, port: int, protocol: str, data: bytes = b''):
        """Process network event for security analysis"""
        # Log the connection
        self.network_monitor.log_connection(source_ip, port, protocol)
        
        # Check if IP is blocked
        if source_ip in self.network_monitor.blocked_ips:
            return {"blocked": True, "reason": "IP blacklisted"}
        
        # Analyze protocol if data provided
        if data:
            protocol_anomaly = self.protocol_analyzer.analyze_message(data, source_ip, port)
            if protocol_anomaly:
                self._create_security_event(
                    source_ip, "protocol", AttackType.PROTOCOL_ANOMALY, ThreatLevel.MEDIUM,
                    f"Protocol anomaly detected: {protocol_anomaly['anomalies']}",
                    protocol_anomaly
                )
            
            # Scan for malware
            malware_result = self.malware_detector.scan_data(data, source_ip)
            if not malware_result['clean']:
                self._create_security_event(
                    source_ip, "system", AttackType.MALWARE, ThreatLevel.CRITICAL,
                    f"Malware detected: {malware_result['threats_found']} threats",
                    malware_result
                )
                return {"blocked": True, "reason": "Malware detected"}
        
        return {"allowed": True}
    
    def get_security_status(self) -> Dict:
        """Get current security status"""
        recent_events = [e for e in self.security_events if 
                        (datetime.now() - e.timestamp).seconds < 3600]  # Last hour
        
        threat_counts = defaultdict(int)
        for event in recent_events:
            threat_counts[event.threat_level] += 1
        
        return {
            "monitoring_active": self.running,
            "blocked_ips": len(self.network_monitor.blocked_ips),
            "active_sessions": len(self.auth_manager.active_sessions),
            "locked_accounts": len(self.auth_manager.locked_accounts),
            "recent_threats": {
                "critical": threat_counts[ThreatLevel.CRITICAL],
                "high": threat_counts[ThreatLevel.HIGH],
                "medium": threat_counts[ThreatLevel.MEDIUM],
                "low": threat_counts[ThreatLevel.LOW]
            },
            "total_events": len(self.security_events)
        }
    
    def get_recent_events(self, limit: int = 10) -> List[SecurityEvent]:
        """Get recent security events"""
        return list(self.security_events)[-limit:]
    
    def block_ip(self, ip: str, reason: str = "Manual block"):
        """Manually block IP address"""
        self.network_monitor.blocked_ips.add(ip)
        self._create_security_event(
            ip, "manual", AttackType.INSIDER_THREAT, ThreatLevel.HIGH,
            f"IP manually blocked: {reason}",
            {"manual_action": True}
        )
    
    def unblock_ip(self, ip: str):
        """Unblock IP address"""
        self.network_monitor.blocked_ips.discard(ip)
        print(f"[Security] IP {ip} unblocked")

# Global security orchestrator
security_system = None

def init_security_system():
    """Initialize cybersecurity system"""
    global security_system
    if security_system is None:
        security_system = SecurityOrchestrator()
        security_system.start_monitoring()
        print("[Security] Cybersecurity system initialized")

def stop_security_system():
    """Stop cybersecurity system"""
    global security_system
    if security_system:
        security_system.stop_monitoring()
        security_system = None

def get_security_status():
    """Get security system status"""
    global security_system
    if security_system:
        return security_system.get_security_status()
    return {"monitoring_active": False}

if __name__ == "__main__":
    print("AutoGrid Cybersecurity Engine Test")
    print("==================================")
    
    init_security_system()
    
    try:
        # Simulate some network events
        for i in range(10):
            result = security_system.process_network_event(
                f"192.168.1.{100 + i}", 502, "modbus", 
                b'\x00\x01\x00\x00\x00\x06\x01\x03\x00\x00\x00\x02'
            )
            print(f"Network event {i}: {result}")
            time.sleep(1)
        
        # Simulate DDoS attack
        for i in range(60):
            security_system.process_network_event("192.168.1.200", 80, "http")
        
        time.sleep(15)
        
        status = get_security_status()
        print(f"Security status: {status}")
        
        events = security_system.get_recent_events(5)
        for event in events:
            print(f"Event: {event.attack_type.value} - {event.description}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_security_system()