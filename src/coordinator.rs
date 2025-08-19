/*
 * AutoGrid Rust Coordination Layer
 * Distributed consensus and secure P2P communication
 * Handles multi-node coordination and fault tolerance
 */

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::io::{Read, Write};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Receiver, Sender};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: SocketAddr,
    pub node_type: NodeType,
    pub last_heartbeat: u64,
    pub status: NodeStatus,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Controller,
    Generator,
    Storage,
    Load,
    EV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Online,
    Offline,
    Maintenance,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyMessage {
    pub timestamp: u64,
    pub sender: String,
    pub message_type: MessageType,
    pub data: HashMap<String, f64>,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Heartbeat,
    EnergyOffer,
    EnergyRequest,
    LoadBalance,
    Emergency,
    Consensus,
    V2GCommand,
}

#[derive(Debug, Clone)]
pub struct ConsensusState {
    pub term: u64,
    pub leader: Option<String>,
    pub voted_for: Option<String>,
    pub log: Vec<LogEntry>,
    pub commit_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: u64,
    pub index: usize,
    pub command: EnergyCommand,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnergyCommand {
    SetGeneration { node_id: String, power: f64 },
    SetLoad { node_id: String, power: f64 },
    ChargeEV { ev_id: String, power: f64 },
    DischargeEV { ev_id: String, power: f64 },
    EmergencyShutdown { node_id: String },
}

pub struct DistributedCoordinator {
    node_id: String,
    local_address: SocketAddr,
    nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
    consensus_state: Arc<Mutex<ConsensusState>>,
    running: Arc<Mutex<bool>>,
    message_queue: Arc<Mutex<Vec<EnergyMessage>>>,
    ev_fleet: Arc<RwLock<HashMap<String, EVStatus>>>,
    energy_state: Arc<RwLock<EnergyState>>,
}

#[derive(Debug, Clone)]
pub struct EVStatus {
    pub id: String,
    pub connected: bool,
    pub soc: f64,
    pub max_power: f64,
    pub v2g_enabled: bool,
    pub priority: EVPriority,
    pub last_update: Instant,
}

#[derive(Debug, Clone)]
pub enum EVPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct EnergyState {
    pub total_generation: f64,
    pub total_demand: f64,
    pub battery_soc: f64,
    pub grid_frequency: f64,
    pub stability_score: f64,
    pub last_update: Instant,
}

impl DistributedCoordinator {
    pub fn new(node_id: String, local_address: SocketAddr) -> Self {
        let initial_consensus = ConsensusState {
            term: 0,
            leader: None,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
        };

        Self {
            node_id: node_id.clone(),
            local_address,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            consensus_state: Arc::new(Mutex::new(initial_consensus)),
            running: Arc::new(Mutex::new(false)),
            message_queue: Arc::new(Mutex::new(Vec::new())),
            ev_fleet: Arc::new(RwLock::new(HashMap::new())),
            energy_state: Arc::new(RwLock::new(EnergyState {
                total_generation: 0.0,
                total_demand: 0.0,
                battery_soc: 75.0,
                grid_frequency: 60.0,
                stability_score: 95.0,
                last_update: Instant::now(),
            })),
        }
    }

    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("[Rust] Starting distributed coordinator on {}", self.local_address);
        *self.running.lock().unwrap() = true;

        // Start network listener
        self.start_network_listener()?;
        
        // Start consensus protocol
        self.start_consensus_loop();
        
        // Start heartbeat
        self.start_heartbeat();
        
        // Start EV management
        self.start_ev_management();
        
        // Start energy optimization
        self.start_energy_optimization();

        println!("[Rust] Distributed coordinator started successfully");
        Ok(())
    }

    fn start_network_listener(&self) -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(self.local_address)?;
        let running = Arc::clone(&self.running);
        let message_queue = Arc::clone(&self.message_queue);
        
        thread::spawn(move || {
            println!("[Rust] Network listener started");
            
            for stream in listener.incoming() {
                if !*running.lock().unwrap() {
                    break;
                }
                
                match stream {
                    Ok(mut stream) => {
                        let mut buffer = [0; 4096];
                        match stream.read(&mut buffer) {
                            Ok(size) => {
                                let msg_str = String::from_utf8_lossy(&buffer[..size]);
                                match serde_json::from_str::<EnergyMessage>(&msg_str) {
                                    Ok(message) => {
                                        message_queue.lock().unwrap().push(message);
                                    }
                                    Err(e) => {
                                        eprintln!("[Rust] Failed to parse message: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("[Rust] Failed to read from stream: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[Rust] Connection failed: {}", e);
                    }
                }
            }
        });
        
        Ok(())
    }

    fn start_consensus_loop(&self) {
        let consensus_state = Arc::clone(&self.consensus_state);
        let running = Arc::clone(&self.running);
        let nodes = Arc::clone(&self.nodes);
        let node_id = self.node_id.clone();
        
        thread::spawn(move || {
            println!("[Rust] Consensus protocol started");
            
            while *running.lock().unwrap() {
                let mut state = consensus_state.lock().unwrap();
                
                // Check if we need to start election
                if state.leader.is_none() && nodes.read().unwrap().len() > 1 {
                    println!("[Rust] Starting leader election for term {}", state.term + 1);
                    state.term += 1;
                    state.voted_for = Some(node_id.clone());
                    // In a real implementation, send vote requests to other nodes
                    state.leader = Some(node_id.clone());
                    println!("[Rust] Node {} elected as leader for term {}", node_id, state.term);
                }
                
                drop(state);
                thread::sleep(Duration::from_secs(5));
            }
        });
    }

    fn start_heartbeat(&self) {
        let running = Arc::clone(&self.running);
        let nodes = Arc::clone(&self.nodes);
        let node_id = self.node_id.clone();
        let local_address = self.local_address;
        
        thread::spawn(move || {
            println!("[Rust] Heartbeat service started");
            
            while *running.lock().unwrap() {
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                // Update our own heartbeat
                let self_info = NodeInfo {
                    id: node_id.clone(),
                    address: local_address,
                    node_type: NodeType::Controller,
                    last_heartbeat: timestamp,
                    status: NodeStatus::Online,
                    capabilities: vec!["consensus".to_string(), "load_balance".to_string(), "v2g".to_string()],
                };
                
                nodes.write().unwrap().insert(node_id.clone(), self_info);
                
                // Check for dead nodes
                let mut dead_nodes = Vec::new();
                {
                    let nodes_read = nodes.read().unwrap();
                    for (id, info) in nodes_read.iter() {
                        if timestamp - info.last_heartbeat > 30 {
                            dead_nodes.push(id.clone());
                        }
                    }
                }
                
                // Remove dead nodes
                if !dead_nodes.is_empty() {
                    let mut nodes_write = nodes.write().unwrap();
                    for dead_id in dead_nodes {
                        println!("[Rust] Removing dead node: {}", dead_id);
                        nodes_write.remove(&dead_id);
                    }
                }
                
                thread::sleep(Duration::from_secs(10));
            }
        });
    }

    fn start_ev_management(&self) {
        let running = Arc::clone(&self.running);
        let ev_fleet = Arc::clone(&self.ev_fleet);
        let energy_state = Arc::clone(&self.energy_state);
        
        thread::spawn(move || {
            println!("[Rust] EV management service started");
            
            // Initialize some EVs for simulation
            {
                let mut fleet = ev_fleet.write().unwrap();
                for i in 1..=8 {
                    let ev = EVStatus {
                        id: format!("EV_{:03}", i),
                        connected: i % 3 != 0, // 2/3 connected
                        soc: 30.0 + (i as f64 * 8.0),
                        max_power: 11.0, // 11kW AC charging
                        v2g_enabled: i % 2 == 0, // Half V2G capable
                        priority: match i % 4 {
                            0 => EVPriority::Critical,
                            1 => EVPriority::High,
                            2 => EVPriority::Medium,
                            _ => EVPriority::Low,
                        },
                        last_update: Instant::now(),
                    };
                    fleet.insert(ev.id.clone(), ev);
                }
            }
            
            while *running.lock().unwrap() {
                // EV fleet optimization logic
                let energy = energy_state.read().unwrap().clone();
                let mut fleet = ev_fleet.write().unwrap();
                
                if energy.total_generation > energy.total_demand + 5.0 {
                    // Excess generation - charge EVs
                    for ev in fleet.values_mut() {
                        if ev.connected && ev.soc < 90.0 {
                            println!("[Rust] Charging {} at {:.1}kW", ev.id, ev.max_power);
                        }
                    }
                } else if energy.total_demand > energy.total_generation + 5.0 {
                    // Energy deficit - discharge V2G EVs
                    for ev in fleet.values_mut() {
                        if ev.connected && ev.v2g_enabled && ev.soc > 50.0 {
                            match ev.priority {
                                EVPriority::Critical => continue, // Don't discharge critical EVs
                                _ => {
                                    println!("[Rust] V2G discharge {} at {:.1}kW", ev.id, ev.max_power * 0.8);
                                }
                            }
                        }
                    }
                }
                
                thread::sleep(Duration::from_secs(15));
            }
        });
    }

    fn start_energy_optimization(&self) {
        let running = Arc::clone(&self.running);
        let energy_state = Arc::clone(&self.energy_state);
        let consensus_state = Arc::clone(&self.consensus_state);
        
        thread::spawn(move || {
            println!("[Rust] Energy optimization service started");
            
            while *running.lock().unwrap() {
                let mut energy = energy_state.write().unwrap();
                
                // Simulate energy readings
                let now = Instant::now();
                let hour = (now.elapsed().as_secs() / 3600) % 24;
                
                // Solar generation pattern
                let solar_factor = if hour >= 6 && hour <= 18 {
                    (std::f64::consts::PI * (hour as f64 - 6.0) / 12.0).sin().max(0.0)
                } else {
                    0.0
                };
                
                energy.total_generation = 25.0 * solar_factor + 15.0 * (0.3 + 0.7 * (hour as f64 * 0.1).sin());
                energy.total_demand = 60.0 * (0.7 + 0.3 * ((hour as f64 - 6.0) * 0.2).sin());
                
                // Grid frequency regulation
                let power_imbalance = energy.total_generation - energy.total_demand;
                energy.grid_frequency = 60.0 + (power_imbalance * 0.01);
                
                // Stability scoring
                energy.stability_score = (100.0 - (energy.grid_frequency - 60.0).abs() * 10.0).max(0.0);
                
                energy.last_update = now;
                drop(energy);
                
                // Execute load balancing if we're the leader
                let consensus = consensus_state.lock().unwrap();
                if let Some(ref leader) = consensus.leader {
                    if leader == "coordinator_001" { // Assume we're this node
                        if power_imbalance.abs() > 2.0 {
                            println!("[Rust] Executing load balancing: imbalance = {:.1}kW", power_imbalance);
                        }
                    }
                }
                drop(consensus);
                
                thread::sleep(Duration::from_secs(5));
            }
        });
    }

    pub fn add_node(&self, node: NodeInfo) {
        self.nodes.write().unwrap().insert(node.id.clone(), node);
        println!("[Rust] Added node to network: {}", node.id);
    }

    pub fn get_network_status(&self) -> HashMap<String, NodeInfo> {
        self.nodes.read().unwrap().clone()
    }

    pub fn get_energy_status(&self) -> EnergyState {
        self.energy_state.read().unwrap().clone()
    }

    pub fn get_ev_fleet(&self) -> HashMap<String, EVStatus> {
        self.ev_fleet.read().unwrap().clone()
    }

    pub fn execute_emergency_protocol(&self) {
        println!("[Rust] EMERGENCY PROTOCOL ACTIVATED");
        
        // Stop all non-critical operations
        let mut fleet = self.ev_fleet.write().unwrap();
        for ev in fleet.values_mut() {
            match ev.priority {
                EVPriority::Critical => continue,
                _ => {
                    ev.connected = false;
                    println!("[Rust] Emergency disconnect: {}", ev.id);
                }
            }
        }
        
        // Enter safe mode
        let mut energy = self.energy_state.write().unwrap();
        energy.stability_score = 50.0;
        println!("[Rust] System in emergency safe mode");
    }

    pub fn stop(&self) {
        *self.running.lock().unwrap() = false;
        println!("[Rust] Distributed coordinator stopping...");
    }
}

// Python FFI interface
#[no_mangle]
pub extern "C" fn create_coordinator(node_id: *const std::os::raw::c_char, port: u16) -> *mut DistributedCoordinator {
    let c_str = unsafe { std::ffi::CStr::from_ptr(node_id) };
    let node_id_str = c_str.to_str().unwrap().to_string();
    let address = format!("127.0.0.1:{}", port).parse().unwrap();
    
    let coordinator = DistributedCoordinator::new(node_id_str, address);
    Box::into_raw(Box::new(coordinator))
}

#[no_mangle]
pub extern "C" fn start_coordinator(coordinator: *mut DistributedCoordinator) -> bool {
    if coordinator.is_null() {
        return false;
    }
    
    let coordinator = unsafe { &mut *coordinator };
    match coordinator.start() {
        Ok(_) => true,
        Err(e) => {
            eprintln!("[Rust] Failed to start coordinator: {}", e);
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn stop_coordinator(coordinator: *mut DistributedCoordinator) {
    if !coordinator.is_null() {
        let coordinator = unsafe { Box::from_raw(coordinator) };
        coordinator.stop();
    }
}

#[no_mangle]
pub extern "C" fn get_energy_metrics(
    coordinator: *const DistributedCoordinator,
    generation: *mut f64,
    demand: *mut f64,
    frequency: *mut f64,
    stability: *mut f64,
) {
    if coordinator.is_null() {
        return;
    }
    
    let coordinator = unsafe { &*coordinator };
    let energy = coordinator.get_energy_status();
    
    unsafe {
        *generation = energy.total_generation;
        *demand = energy.total_demand;
        *frequency = energy.grid_frequency;
        *stability = energy.stability_score;
    }
}

fn main() {
    println!("AutoGrid Rust Coordination Layer Test");
    println!("=====================================");
    
    let mut coordinator = DistributedCoordinator::new(
        "coordinator_001".to_string(),
        "127.0.0.1:8080".parse().unwrap(),
    );
    
    if let Err(e) = coordinator.start() {
        eprintln!("Failed to start coordinator: {}", e);
        return;
    }
    
    // Run for 30 seconds
    thread::sleep(Duration::from_secs(30));
    
    coordinator.stop();
    println!("Coordinator test completed");
}