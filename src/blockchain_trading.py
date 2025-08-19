#!/usr/bin/env python3
"""
AutoGrid Blockchain Trading Engine
Distributed ledger for secure energy transactions
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import threading
import uuid

@dataclass
class EnergyTransaction:
    tx_id: str
    timestamp: datetime
    seller_id: str
    buyer_id: str
    energy_amount: float  # kWh
    price_per_kwh: float
    total_value: float
    contract_terms: Dict
    signature: str
    status: str  # pending, confirmed, executed, failed

@dataclass
class SmartContract:
    contract_id: str
    contract_type: str  # time_of_use, demand_response, v2g_service
    parties: List[str]
    terms: Dict
    auto_execute: bool
    execution_conditions: Dict
    created_at: datetime
    expires_at: datetime
    status: str

@dataclass
class Block:
    index: int
    timestamp: datetime
    transactions: List[EnergyTransaction]
    previous_hash: str
    nonce: int
    hash: str

class CryptoEngine:
    def __init__(self):
        self.key = self._generate_key()
        self.cipher = Fernet(self.key)
        
    def _generate_key(self) -> bytes:
        """Generate encryption key"""
        password = b"autogrid_secure_energy_trading"
        salt = b"energy_salt_2024"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def create_signature(self, data: str) -> str:
        """Create digital signature for transaction"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify transaction signature"""
        return self.create_signature(data) == signature

class BlockchainLedger:
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[EnergyTransaction] = []
        self.mining_difficulty = 4
        self.block_reward = 0.1  # Energy tokens
        self.crypto = CryptoEngine()
        self.create_genesis_block()
        
    def create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = Block(
            index=0,
            timestamp=datetime.now(),
            transactions=[],
            previous_hash="0",
            nonce=0,
            hash=""
        )
        genesis_block.hash = self._calculate_hash(genesis_block)
        self.chain.append(genesis_block)
        print("[Blockchain] Genesis block created")
        
    def _calculate_hash(self, block: Block) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            'index': block.index,
            'timestamp': block.timestamp.isoformat(),
            'transactions': [asdict(tx) for tx in block.transactions],
            'previous_hash': block.previous_hash,
            'nonce': block.nonce
        }, sort_keys=True, default=str)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def add_transaction(self, transaction: EnergyTransaction) -> bool:
        """Add transaction to pending pool"""
        try:
            # Validate transaction
            if self._validate_transaction(transaction):
                self.pending_transactions.append(transaction)
                print(f"[Blockchain] Transaction {transaction.tx_id} added to pool")
                return True
        except Exception as e:
            print(f"[Blockchain] Transaction validation failed: {e}")
        return False
    
    def _validate_transaction(self, tx: EnergyTransaction) -> bool:
        """Validate transaction integrity"""
        # Check signature
        tx_data = f"{tx.seller_id}{tx.buyer_id}{tx.energy_amount}{tx.price_per_kwh}{tx.timestamp}"
        if not self.crypto.verify_signature(tx_data, tx.signature):
            return False
        
        # Check amounts
        if tx.energy_amount <= 0 or tx.price_per_kwh <= 0:
            return False
            
        # Verify total value
        if abs(tx.total_value - (tx.energy_amount * tx.price_per_kwh)) > 0.01:
            return False
            
        return True
    
    def mine_block(self) -> Optional[Block]:
        """Mine a new block with pending transactions"""
        if not self.pending_transactions:
            return None
            
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now(),
            transactions=self.pending_transactions[:10],  # Max 10 transactions per block
            previous_hash=previous_block.hash,
            nonce=0,
            hash=""
        )
        
        # Proof of work
        target = "0" * self.mining_difficulty
        start_time = time.time()
        
        while not new_block.hash.startswith(target):
            new_block.nonce += 1
            new_block.hash = self._calculate_hash(new_block)
            
            # Prevent infinite mining
            if time.time() - start_time > 10:
                print("[Blockchain] Mining timeout - adjusting difficulty")
                self.mining_difficulty = max(2, self.mining_difficulty - 1)
                return None
        
        # Add block to chain
        self.chain.append(new_block)
        
        # Remove mined transactions from pending pool
        mined_count = len(new_block.transactions)
        self.pending_transactions = self.pending_transactions[mined_count:]
        
        mining_time = time.time() - start_time
        print(f"[Blockchain] Block {new_block.index} mined in {mining_time:.2f}s with {mined_count} transactions")
        
        return new_block
    
    def get_balance(self, participant_id: str) -> float:
        """Calculate energy token balance for participant"""
        balance = 0.0
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.seller_id == participant_id:
                    balance += tx.total_value
                elif tx.buyer_id == participant_id:
                    balance -= tx.total_value
                    
        return balance
    
    def get_transaction_history(self, participant_id: str) -> List[EnergyTransaction]:
        """Get transaction history for participant"""
        history = []
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.seller_id == participant_id or tx.buyer_id == participant_id:
                    history.append(tx)
                    
        return sorted(history, key=lambda x: x.timestamp, reverse=True)
    
    def validate_chain(self) -> bool:
        """Validate entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check hash integrity
            if current_block.hash != self._calculate_hash(current_block):
                return False
                
            # Check link to previous block
            if current_block.previous_hash != previous_block.hash:
                return False
                
            # Validate all transactions in block
            for tx in current_block.transactions:
                if not self._validate_transaction(tx):
                    return False
        
        return True

class SmartContractEngine:
    def __init__(self, blockchain: BlockchainLedger):
        self.blockchain = blockchain
        self.contracts: Dict[str, SmartContract] = {}
        self.running = False
        self.execution_thread = None
        
    def create_contract(self, contract_type: str, parties: List[str], terms: Dict, 
                       auto_execute: bool = True, duration_hours: int = 24) -> str:
        """Create a new smart contract"""
        contract_id = str(uuid.uuid4())[:8]
        
        contract = SmartContract(
            contract_id=contract_id,
            contract_type=contract_type,
            parties=parties,
            terms=terms,
            auto_execute=auto_execute,
            execution_conditions=self._create_execution_conditions(contract_type, terms),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=duration_hours),
            status="active"
        )
        
        self.contracts[contract_id] = contract
        print(f"[SmartContract] Created {contract_type} contract {contract_id}")
        return contract_id
    
    def _create_execution_conditions(self, contract_type: str, terms: Dict) -> Dict:
        """Create execution conditions based on contract type"""
        if contract_type == "time_of_use":
            return {
                "time_start": terms.get("peak_start", "17:00"),
                "time_end": terms.get("peak_end", "21:00"),
                "price_multiplier": terms.get("peak_multiplier", 1.5)
            }
        elif contract_type == "demand_response":
            return {
                "trigger_demand": terms.get("trigger_demand", 80.0),
                "reduction_target": terms.get("reduction_target", 10.0),
                "incentive_rate": terms.get("incentive_rate", 0.05)
            }
        elif contract_type == "v2g_service":
            return {
                "min_soc": terms.get("min_soc", 50.0),
                "max_discharge": terms.get("max_discharge", 10.0),
                "service_rate": terms.get("service_rate", 0.20)
            }
        return {}
    
    def start_execution_engine(self):
        """Start automated contract execution"""
        self.running = True
        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        print("[SmartContract] Execution engine started")
    
    def stop_execution_engine(self):
        """Stop contract execution"""
        self.running = False
        if self.execution_thread:
            self.execution_thread.join()
        print("[SmartContract] Execution engine stopped")
    
    def _execution_loop(self):
        """Main contract execution loop"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for contract_id, contract in self.contracts.items():
                    if contract.status != "active" or not contract.auto_execute:
                        continue
                        
                    # Check if contract expired
                    if current_time > contract.expires_at:
                        contract.status = "expired"
                        continue
                    
                    # Check execution conditions
                    if self._check_execution_conditions(contract):
                        self._execute_contract(contract)
                        
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"[SmartContract] Execution error: {e}")
                time.sleep(5)
    
    def _check_execution_conditions(self, contract: SmartContract) -> bool:
        """Check if contract conditions are met"""
        conditions = contract.execution_conditions
        current_time = datetime.now().time()
        
        if contract.contract_type == "time_of_use":
            start_time = datetime.strptime(conditions["time_start"], "%H:%M").time()
            end_time = datetime.strptime(conditions["time_end"], "%H:%M").time()
            return start_time <= current_time <= end_time
            
        elif contract.contract_type == "demand_response":
            # In real implementation, would check actual grid demand
            simulated_demand = 75.0  # Simulate current demand
            return simulated_demand > conditions["trigger_demand"]
            
        elif contract.contract_type == "v2g_service":
            # Check if V2G service is needed
            return True  # Simplified for demo
            
        return False
    
    def _execute_contract(self, contract: SmartContract):
        """Execute smart contract"""
        try:
            print(f"[SmartContract] Executing contract {contract.contract_id} ({contract.contract_type})")
            
            if contract.contract_type == "time_of_use":
                self._execute_time_of_use(contract)
            elif contract.contract_type == "demand_response":
                self._execute_demand_response(contract)
            elif contract.contract_type == "v2g_service":
                self._execute_v2g_service(contract)
                
            contract.status = "executed"
            
        except Exception as e:
            print(f"[SmartContract] Execution failed for {contract.contract_id}: {e}")
            contract.status = "failed"
    
    def _execute_time_of_use(self, contract: SmartContract):
        """Execute time-of-use contract"""
        multiplier = contract.execution_conditions["price_multiplier"]
        
        # Create automated transaction with adjusted pricing
        tx = EnergyTransaction(
            tx_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            seller_id=contract.parties[0],
            buyer_id=contract.parties[1],
            energy_amount=contract.terms.get("energy_amount", 10.0),
            price_per_kwh=contract.terms.get("base_price", 0.15) * multiplier,
            total_value=0.0,
            contract_terms=contract.terms,
            signature="",
            status="pending"
        )
        
        tx.total_value = tx.energy_amount * tx.price_per_kwh
        tx.signature = self.blockchain.crypto.create_signature(
            f"{tx.seller_id}{tx.buyer_id}{tx.energy_amount}{tx.price_per_kwh}"
        )
        
        self.blockchain.add_transaction(tx)
    
    def _execute_demand_response(self, contract: SmartContract):
        """Execute demand response contract"""
        reduction = contract.execution_conditions["reduction_target"]
        incentive = contract.execution_conditions["incentive_rate"]
        
        print(f"[SmartContract] Demand response: reduce {reduction}kW, incentive ${incentive}/kWh")
    
    def _execute_v2g_service(self, contract: SmartContract):
        """Execute V2G service contract"""
        max_discharge = contract.execution_conditions["max_discharge"]
        service_rate = contract.execution_conditions["service_rate"]
        
        print(f"[SmartContract] V2G service: discharge up to {max_discharge}kW at ${service_rate}/kWh")

class DistributedTradingEngine:
    def __init__(self):
        self.blockchain = BlockchainLedger()
        self.smart_contracts = SmartContractEngine(self.blockchain)
        self.market_makers = {}
        self.order_book = {"buy": [], "sell": []}
        self.running = False
        
    def start(self):
        """Start distributed trading engine"""
        self.running = True
        self.smart_contracts.start_execution_engine()
        
        # Start mining thread
        self.mining_thread = threading.Thread(target=self._mining_loop)
        self.mining_thread.daemon = True
        self.mining_thread.start()
        
        print("[Trading] Distributed trading engine started")
    
    def stop(self):
        """Stop trading engine"""
        self.running = False
        self.smart_contracts.stop_execution_engine()
        print("[Trading] Distributed trading engine stopped")
    
    def _mining_loop(self):
        """Continuous mining loop"""
        while self.running:
            try:
                block = self.blockchain.mine_block()
                if block:
                    print(f"[Trading] New block mined: {block.index} with {len(block.transactions)} transactions")
                time.sleep(5)
            except Exception as e:
                print(f"[Trading] Mining error: {e}")
                time.sleep(10)
    
    def create_energy_transaction(self, seller_id: str, buyer_id: str, 
                                energy_amount: float, price_per_kwh: float,
                                contract_terms: Dict = None) -> str:
        """Create and submit energy transaction"""
        tx_id = str(uuid.uuid4())[:8]
        
        transaction = EnergyTransaction(
            tx_id=tx_id,
            timestamp=datetime.now(),
            seller_id=seller_id,
            buyer_id=buyer_id,
            energy_amount=energy_amount,
            price_per_kwh=price_per_kwh,
            total_value=energy_amount * price_per_kwh,
            contract_terms=contract_terms or {},
            signature="",
            status="pending"
        )
        
        # Create signature
        tx_data = f"{seller_id}{buyer_id}{energy_amount}{price_per_kwh}{transaction.timestamp}"
        transaction.signature = self.blockchain.crypto.create_signature(tx_data)
        
        if self.blockchain.add_transaction(transaction):
            return tx_id
        return ""
    
    def create_smart_contract(self, contract_type: str, parties: List[str], 
                            terms: Dict, duration_hours: int = 24) -> str:
        """Create smart contract for automated trading"""
        return self.smart_contracts.create_contract(
            contract_type, parties, terms, True, duration_hours
        )
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        return {
            "blockchain_height": len(self.blockchain.chain),
            "pending_transactions": len(self.blockchain.pending_transactions),
            "active_contracts": len([c for c in self.smart_contracts.contracts.values() if c.status == "active"]),
            "total_volume_24h": self._calculate_volume_24h(),
            "chain_valid": self.blockchain.validate_chain()
        }
    
    def _calculate_volume_24h(self) -> float:
        """Calculate 24h trading volume"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        volume = 0.0
        
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if tx.timestamp > cutoff_time:
                    volume += tx.energy_amount
                    
        return volume
    
    def get_participant_status(self, participant_id: str) -> Dict:
        """Get participant trading status"""
        return {
            "balance": self.blockchain.get_balance(participant_id),
            "transaction_count": len(self.blockchain.get_transaction_history(participant_id)),
            "active_contracts": len([c for c in self.smart_contracts.contracts.values() 
                                   if participant_id in c.parties and c.status == "active"])
        }

# Global trading engine instance
trading_engine = None

def init_trading_engine():
    """Initialize distributed trading engine"""
    global trading_engine
    if trading_engine is None:
        trading_engine = DistributedTradingEngine()
        trading_engine.start()
        print("[Trading] Blockchain trading engine initialized")

def stop_trading_engine():
    """Stop trading engine"""
    global trading_engine
    if trading_engine:
        trading_engine.stop()
        trading_engine = None

def get_trading_status():
    """Get trading engine status"""
    global trading_engine
    if trading_engine:
        return trading_engine.get_market_status()
    return {"running": False}

if __name__ == "__main__":
    print("AutoGrid Blockchain Trading Engine Test")
    print("=======================================")
    
    init_trading_engine()
    
    try:
        # Create some test transactions
        for i in range(5):
            tx_id = trading_engine.create_energy_transaction(
                f"SOLAR_{i:02d}", f"HOME_{i:02d}", 
                10.0 + i * 2, 0.15 + i * 0.01
            )
            print(f"Created transaction: {tx_id}")
            time.sleep(1)
        
        # Create smart contracts
        contract_id = trading_engine.create_smart_contract(
            "time_of_use", ["SOLAR_01", "HOME_01"],
            {"energy_amount": 15.0, "base_price": 0.15, "peak_multiplier": 1.5}
        )
        print(f"Created smart contract: {contract_id}")
        
        # Run for 30 seconds
        time.sleep(30)
        
        status = get_trading_status()
        print(f"Final status: {status}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_trading_engine()