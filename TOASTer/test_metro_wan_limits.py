#!/usr/bin/env python3

"""
TOASTer Metro Area WAN - Ultra-Low Latency Stress Test

Tests transport sync at the PHYSICAL LIMITS for metro-scale networks:
- Multiple redundant transmission paths
- Packet deduplication algorithms  
- Sub-millisecond processing budgets
- Speed-of-light limited scenarios
- Metro area WAN conditions (50+ node networks)
- Ultra-aggressive stress testing

Target: 1ms total processing time per node
Requirement: Better than speed-of-light limitations
"""

import socket
import struct
import json
import time
import threading
import sys
import random
import statistics
import hashlib
from typing import Dict, List, Optional, Set
import argparse
from dataclasses import dataclass
from collections import deque, defaultdict
import concurrent.futures

# TOAST v2 Protocol Constants  
TOAST_MAGIC = 0x54534F54
TOAST_VERSION = 2

class TOASTFrameType:
    TRANSPORT = 0x05
    HEARTBEAT = 0x07
    REDUNDANT_TRANSPORT = 0x09  # New frame type for redundant transmission

@dataclass
class MetroWANCondition:
    """Metro area WAN stress conditions"""
    # Distance and speed-of-light limits
    metro_radius_km: float = 50.0           # Metro area radius in km
    speed_of_light_delay_us: float = 0.0    # Calculated from distance
    
    # Network conditions (much more aggressive)
    packet_loss_rate: float = 0.30          # 30% loss (metro WANs can be brutal)
    burst_loss_probability: float = 0.15    # 15% chance of burst losses
    burst_loss_duration_ms: float = 5.0     # Burst loss duration
    
    # Jitter and latency (microsecond precision)
    base_latency_us: float = 500.0          # Base latency in microseconds
    jitter_us: float = 200.0                # ±200μs jitter
    
    # Metro WAN specific issues
    routing_instability: float = 0.20       # 20% chance of route changes
    congestion_probability: float = 0.25    # 25% chance of congestion spikes
    congestion_multiplier: float = 5.0      # Latency multiplier during congestion
    
    # Physical infrastructure limits
    fiber_congestion_rate: float = 0.10     # 10% fiber link congestion
    switch_processing_delay_us: float = 50.0 # Switch processing delay
    
    # Redundancy configuration
    redundant_paths: int = 4                # Number of parallel transmission paths
    dedup_window_ms: float = 2.0            # Deduplication time window

class MetroWANStressTester:
    def __init__(self, multicast_group="239.255.77.77", port=7777, session_name="MetroWANTest"):
        self.multicast_group = multicast_group
        self.port = port
        self.session_name = session_name
        self.session_id = hash(session_name) & 0xFFFFFFFF
        self.sequence_number = random.randint(10000, 99999)
        self.running = False
        
        # Metro WAN conditions
        self.wan_condition = MetroWANCondition()
        
        # Ultra-precise timing
        self.timing_resolution_ns = 1000  # 1μs timing resolution
        self.processing_budget_us = 1000  # 1ms processing budget
        
        self.calculate_speed_of_light_limits()
        
        # Multiple transmission paths simulation
        self.transmission_paths = []
        self.setup_redundant_paths()
        
        # Deduplication system
        self.received_message_hashes: Set[str] = set()
        self.dedup_timestamps: Dict[str, float] = {}
        self.message_arrival_times: Dict[str, List[float]] = defaultdict(list)
        
        # Statistics tracking (microsecond precision)
        self.stats = {
            'sent_commands': 0,
            'received_responses': 0,
            'duplicate_responses': 0,
            'dropped_packets': 0,
            'sub_ms_responses': 0,
            'over_budget_responses': 0,
            'speed_of_light_violations': 0,
            'burst_losses': 0,
            'congestion_events': 0,
            'routing_changes': 0,
            'response_times_us': [],
            'processing_times_us': [],
            'dedup_efficiency': [],
            'path_success_rates': defaultdict(list)
        }
        
        # Network state simulation
        self.current_route = 0
        self.is_congested = False
        self.congestion_end_time = 0.0
        self.in_burst_loss = False
        self.burst_loss_end_time = 0.0
        
        # Socket setup
        self.sock = None
        self.setup_socket()
        
    def calculate_speed_of_light_limits(self):
        """Calculate theoretical speed-of-light limits"""
        # Speed of light in fiber optic cable: ~200,000 km/s (2/3 of vacuum speed)
        fiber_speed_km_per_s = 200000
        
        # Calculate minimum possible latency for metro area
        self.wan_condition.speed_of_light_delay_us = (
            (self.wan_condition.metro_radius_km * 2) / fiber_speed_km_per_s * 1000000
        )
        
        print(f"📡 Metro area radius: {self.wan_condition.metro_radius_km}km")
        print(f"⚡ Speed-of-light limit: {self.wan_condition.speed_of_light_delay_us:.1f}μs")
        print(f"🎯 Processing budget: {self.processing_budget_us}μs per node")
        
    def setup_redundant_paths(self):
        """Setup multiple redundant transmission paths"""
        for i in range(self.wan_condition.redundant_paths):
            path = {
                'id': i,
                'success_rate': random.uniform(0.6, 0.9),  # Each path has different reliability
                'latency_offset_us': random.uniform(-100, 100),  # Path-specific latency
                'congestion_probability': random.uniform(0.1, 0.4)  # Path-specific congestion
            }
            self.transmission_paths.append(path)
            
    def setup_socket(self):
        """Setup UDP multicast socket with high-precision timing"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Enable high-precision timestamps (if available)
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_TIMESTAMPING, 
                                   socket.SO_TIMESTAMPING_RX_HARDWARE | socket.SO_TIMESTAMPING_TX_HARDWARE)
            except:
                pass  # Hardware timestamping not available
                
            self.sock.bind(('0.0.0.0', self.port))
            
            # Join multicast group
            mreq = struct.pack('4sl', socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            # Enable outgoing multicast
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            
            print(f"✅ Metro WAN stress tester ready: {self.multicast_group}:{self.port}")
            
        except Exception as e:
            print(f"❌ Socket setup failed: {e}")
            sys.exit(1)
    
    def get_precise_time_us(self) -> float:
        """Get high-precision timestamp in microseconds"""
        return time.time_ns() / 1000.0  # Convert nanoseconds to microseconds
    
    def simulate_metro_wan_conditions(self, path_id: int = 0) -> tuple[bool, float]:
        """Simulate metro area WAN network conditions for a specific path"""
        current_time = time.time()
        path = self.transmission_paths[path_id]
        
        # Check for burst loss events
        if self.in_burst_loss:
            if current_time >= self.burst_loss_end_time:
                self.in_burst_loss = False
                print(f"📡 Burst loss ended on path {path_id}")
            else:
                self.stats['burst_losses'] += 1
                return False, 0.0  # Packet lost in burst
        else:
            # Check if burst loss should start
            if random.random() < self.wan_condition.burst_loss_probability:
                self.in_burst_loss = True
                self.burst_loss_end_time = current_time + (self.wan_condition.burst_loss_duration_ms / 1000)
                print(f"📡 Burst loss started on path {path_id} (duration: {self.wan_condition.burst_loss_duration_ms}ms)")
                return False, 0.0
        
        # Initialize congestion delay
        congestion_delay = 0.0
        
        # Check for congestion
        if self.is_congested:
            if current_time >= self.congestion_end_time:
                self.is_congested = False
                print(f"🚦 Congestion cleared on path {path_id}")
            else:
                # Add congestion delay
                congestion_delay = self.wan_condition.base_latency_us * self.wan_condition.congestion_multiplier
        else:
            # Check if congestion should start
            if random.random() < path['congestion_probability']:
                self.is_congested = True
                self.congestion_end_time = current_time + random.uniform(1.0, 5.0)  # 1-5 second congestion
                self.stats['congestion_events'] += 1
                print(f"🚦 Congestion detected on path {path_id}")
                congestion_delay = self.wan_condition.base_latency_us * self.wan_condition.congestion_multiplier
        
        # Check for routing instability
        if random.random() < self.wan_condition.routing_instability:
            self.current_route = (self.current_route + 1) % len(self.transmission_paths)
            self.stats['routing_changes'] += 1
            print(f"🔄 Route changed to path {self.current_route}")
        
        # Standard packet loss
        if random.random() < self.wan_condition.packet_loss_rate:
            self.stats['dropped_packets'] += 1
            return False, 0.0
        
        # Path-specific success rate
        if random.random() > path['success_rate']:
            self.stats['dropped_packets'] += 1
            return False, 0.0
        
        # Calculate total latency for this path
        base_latency = self.wan_condition.base_latency_us + path['latency_offset_us']
        jitter = random.uniform(-self.wan_condition.jitter_us, self.wan_condition.jitter_us)
        switch_delay = self.wan_condition.switch_processing_delay_us
        
        total_latency_us = base_latency + jitter + congestion_delay + switch_delay
        
        # Add speed-of-light baseline
        total_latency_us += self.wan_condition.speed_of_light_delay_us
        
        return True, total_latency_us
    
    def create_redundant_toast_frame(self, frame_type: int, payload: bytes, path_id: int) -> bytes:
        """Create TOAST v2 frame with redundancy markers"""
        timestamp_us = int(self.get_precise_time_us()) & 0xFFFFFFFF
        payload_size = len(payload)
        
        # Add path ID and redundancy info to payload
        redundancy_info = {
            'path_id': path_id,
            'total_paths': len(self.transmission_paths),
            'sequence': self.sequence_number,
            'timestamp_us': timestamp_us
        }
        
        # Create combined payload
        if isinstance(payload, bytes):
            try:
                original_data = json.loads(payload.decode('utf-8'))
            except:
                original_data = {'raw_data': payload.hex()}
        else:
            original_data = payload
            
        original_data['redundancy'] = redundancy_info
        combined_payload = json.dumps(original_data).encode('utf-8')
        
        # Calculate checksum
        checksum = sum(combined_payload) & 0xFFFF
        
        # Pack header (32 bytes)
        header = struct.pack('<I B B H I I I I B B H I',
            TOAST_MAGIC, TOAST_VERSION, frame_type, path_id,  # Use flags for path_id
            self.sequence_number, timestamp_us, len(combined_payload), 0,
            path_id, len(self.transmission_paths), checksum, self.session_id
        )
        
        self.sequence_number += 1
        return header + combined_payload
    
    def calculate_message_hash(self, message_data: dict) -> str:
        """Calculate hash for deduplication"""
        # Remove redundancy info for hash calculation
        core_data = {k: v for k, v in message_data.items() if k != 'redundancy'}
        core_json = json.dumps(core_data, sort_keys=True)
        return hashlib.md5(core_json.encode()).hexdigest()
    
    def send_redundant_transport_command(self, command: str, position: float = 0.0, bpm: float = 120.0):
        """Send transport command via multiple redundant paths"""
        send_time_us = self.get_precise_time_us()
        
        # Create transport message
        transport_msg = {
            "type": "transport",
            "command": command,
            "timestamp": int(send_time_us),
            "position": position,
            "bpm": bpm,
            "peer_id": "MetroWANTester"
        }
        
        successful_sends = 0
        
        # Send via all redundant paths
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.transmission_paths)) as executor:
            futures = []
            
            for path_id in range(len(self.transmission_paths)):
                future = executor.submit(self._send_via_path, transport_msg, path_id, send_time_us)
                futures.append(future)
            
            # Wait for all transmissions to complete
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    successful_sends += 1
        
        self.stats['sent_commands'] += 1
        
        redundancy_ratio = successful_sends / len(self.transmission_paths)
        print(f"📡 SENT: {command} via {successful_sends}/{len(self.transmission_paths)} paths ({redundancy_ratio*100:.1f}% redundancy)")
        
        return successful_sends > 0
    
    def _send_via_path(self, transport_msg: dict, path_id: int, send_time_us: float) -> bool:
        """Send message via specific path"""
        # Simulate metro WAN conditions for this path
        can_send, latency_us = self.simulate_metro_wan_conditions(path_id)
        
        if not can_send:
            print(f"📦 DROPPED: Path {path_id} (metro WAN conditions)")
            return False
        
        # Create frame for this path
        payload = json.dumps(transport_msg).encode('utf-8')
        frame = self.create_redundant_toast_frame(TOASTFrameType.REDUNDANT_TRANSPORT, payload, path_id)
        
        # Simulate transmission delay
        if latency_us > 0:
            time.sleep(latency_us / 1000000.0)  # Convert μs to seconds
        
        try:
            self.sock.sendto(frame, (self.multicast_group, self.port))
            
            # Track path success
            self.stats['path_success_rates'][path_id].append(1.0)
            
            # Check if this violates speed of light
            if latency_us < self.wan_condition.speed_of_light_delay_us:
                self.stats['speed_of_light_violations'] += 1
                print(f"⚡ WARNING: Path {path_id} faster than speed of light! ({latency_us:.1f}μs < {self.wan_condition.speed_of_light_delay_us:.1f}μs)")
            
            print(f"📤 Path {path_id}: {latency_us:.1f}μs latency")
            return True
            
        except Exception as e:
            print(f"❌ Path {path_id} send failed: {e}")
            self.stats['path_success_rates'][path_id].append(0.0)
            return False
    
    def listen_for_responses_with_deduplication(self):
        """Listen for responses with advanced deduplication"""
        print("👂 Listening with metro WAN deduplication...")
        
        while self.running:
            try:
                self.sock.settimeout(0.001)  # 1ms timeout for ultra-low latency
                data, addr = self.sock.recvfrom(4096)
                receive_time_us = self.get_precise_time_us()
                
                if len(data) < 32:
                    continue
                    
                # Parse header
                header = struct.unpack('<I B B H I I I I B B H I', data[:32])
                magic, version, frame_type = header[0], header[1], header[2]
                path_id = header[3]  # flags field used for path_id
                timestamp_us = header[5]
                session_id = header[11]
                payload_size = header[6]
                
                if magic != TOAST_MAGIC or version != TOAST_VERSION or session_id != self.session_id:
                    continue
                    
                payload = data[32:32+payload_size]
                
                if frame_type == TOASTFrameType.REDUNDANT_TRANSPORT:
                    try:
                        msg = json.loads(payload.decode('utf-8'))
                        
                        # Calculate message hash for deduplication
                        msg_hash = self.calculate_message_hash(msg)
                        
                        # Check if this is a duplicate
                        current_time = time.time()
                        if msg_hash in self.received_message_hashes:
                            # Check if still within dedup window
                            if current_time - self.dedup_timestamps[msg_hash] < (self.wan_condition.dedup_window_ms / 1000):
                                self.stats['duplicate_responses'] += 1
                                print(f"🔄 DEDUP: Message from path {path_id} (duplicate)")
                                continue
                            else:
                                # Outside dedup window, treat as new message
                                self.received_message_hashes.discard(msg_hash)
                                del self.dedup_timestamps[msg_hash]
                        
                        # Record this message
                        self.received_message_hashes.add(msg_hash)
                        self.dedup_timestamps[msg_hash] = current_time
                        self.message_arrival_times[msg_hash].append(receive_time_us)
                        
                        self.stats['received_responses'] += 1
                        
                        # Calculate response time
                        if 'timestamp' in msg:
                            response_time_us = receive_time_us - msg['timestamp']
                            self.stats['response_times_us'].append(response_time_us)
                            
                            # Check if under processing budget
                            if response_time_us <= self.processing_budget_us:
                                self.stats['sub_ms_responses'] += 1
                                status_icon = "⚡"
                            else:
                                self.stats['over_budget_responses'] += 1
                                status_icon = "⏰"
                            
                            # Extract command info
                            command = msg.get('command', 'UNKNOWN')
                            position = msg.get('position', 0.0)
                            bpm = msg.get('bpm', 120.0)
                            
                            print(f"{status_icon} RECV: {command} via path {path_id} | {response_time_us:.1f}μs | pos:{position:.3f} bpm:{bpm:.0f}")
                        
                    except json.JSONDecodeError as e:
                        print(f"📝 JSON decode error from path {path_id}: {e}")
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ Listen error: {e}")
    
    def run_metro_wan_stress_test(self, duration_seconds=60):
        """Run comprehensive metro WAN stress test"""
        print(f"\n🌐 Metro Area WAN Stress Test ({duration_seconds}s)")
        print("=" * 60)
        print(f"📊 Conditions:")
        print(f"  • Metro radius: {self.wan_condition.metro_radius_km}km")
        print(f"  • Speed-of-light limit: {self.wan_condition.speed_of_light_delay_us:.1f}μs")
        print(f"  • Processing budget: {self.processing_budget_us}μs per node")
        print(f"  • Packet Loss: {self.wan_condition.packet_loss_rate*100:.1f}%")
        print(f"  • Redundant Paths: {self.wan_condition.redundant_paths}")
        print(f"  • Dedup Window: {self.wan_condition.dedup_window_ms}ms")
        print("")
        
        self.running = True
        listen_thread = threading.Thread(target=self.listen_for_responses_with_deduplication, daemon=True)
        listen_thread.start()
        
        # Ultra-aggressive test commands
        start_time = time.time()
        test_commands = ["PLAY", "STOP", "PLAY", "PAUSE", "PLAY", "STOP", "PLAY", "STOP"]
        command_interval = duration_seconds / len(test_commands)
        
        try:
            for i, command in enumerate(test_commands):
                if not self.running:
                    break
                    
                position = random.uniform(0, 100)
                bpm = random.choice([120, 128, 140, 110, 96, 160])
                
                print(f"\n📍 Metro Test {i+1}/{len(test_commands)}: {command}")
                
                # Send with ultra-high precision timing
                self.send_redundant_transport_command(command, position, bpm)
                
                # Precise timing for next command
                elapsed = time.time() - start_time
                next_command_time = (i + 1) * command_interval
                wait_time = max(0, next_command_time - elapsed)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    
        except KeyboardInterrupt:
            print("\n🛑 Metro WAN stress test interrupted")
        finally:
            self.running = False
            time.sleep(2)  # Allow final responses
            
        self.print_metro_wan_results()
    
    def print_metro_wan_results(self):
        """Print comprehensive metro WAN test results"""
        print("\n" + "="*80)
        print("🌐 METRO AREA WAN STRESS TEST RESULTS")
        print("="*80)
        
        # Basic transmission stats
        print(f"\n📈 Transmission Statistics:")
        print(f"  • Commands Sent: {self.stats['sent_commands']}")
        print(f"  • Responses Received: {self.stats['received_responses']}")
        print(f"  • Duplicate Responses: {self.stats['duplicate_responses']}")
        print(f"  • Packets Dropped: {self.stats['dropped_packets']}")
        
        if self.stats['received_responses'] > 0:
            success_rate = (self.stats['received_responses'] / max(1, self.stats['sent_commands'])) * 100
            dedup_efficiency = (self.stats['duplicate_responses'] / max(1, self.stats['received_responses'])) * 100
            print(f"  • Success Rate: {success_rate:.1f}%")
            print(f"  • Deduplication Efficiency: {dedup_efficiency:.1f}%")
        
        # Ultra-low latency analysis
        if self.stats['response_times_us']:
            response_times = self.stats['response_times_us']
            print(f"\n⚡ Ultra-Low Latency Analysis:")
            print(f"  • Average Response: {statistics.mean(response_times):.1f}μs")
            print(f"  • Median Response: {statistics.median(response_times):.1f}μs")
            print(f"  • Min/Max: {min(response_times):.1f}μs / {max(response_times):.1f}μs")
            print(f"  • Std Dev: {statistics.stdev(response_times) if len(response_times) > 1 else 0:.1f}μs")
            
            # Processing budget analysis
            under_budget = self.stats['sub_ms_responses']
            over_budget = self.stats['over_budget_responses']
            budget_compliance = (under_budget / max(1, under_budget + over_budget)) * 100
            
            print(f"\n🎯 Processing Budget Analysis (1ms = 1000μs):")
            print(f"  • Under Budget: {under_budget} responses")
            print(f"  • Over Budget: {over_budget} responses")
            print(f"  • Budget Compliance: {budget_compliance:.1f}%")
        
        # Speed of light analysis
        print(f"\n⚡ Physical Limits Analysis:")
        print(f"  • Speed-of-Light Limit: {self.wan_condition.speed_of_light_delay_us:.1f}μs")
        print(f"  • Physics Violations: {self.stats['speed_of_light_violations']}")
        
        # Metro WAN specific stats
        print(f"\n🌐 Metro WAN Infrastructure:")
        print(f"  • Burst Loss Events: {self.stats['burst_losses']}")
        print(f"  • Congestion Events: {self.stats['congestion_events']}")
        print(f"  • Routing Changes: {self.stats['routing_changes']}")
        
        # Path performance analysis
        print(f"\n🛤️ Redundant Path Performance:")
        for path_id, success_rates in self.stats['path_success_rates'].items():
            if success_rates:
                avg_success = statistics.mean(success_rates) * 100
                print(f"  • Path {path_id}: {avg_success:.1f}% success rate")
        
        # Overall assessment for metro WAN
        print(f"\n🎯 Metro WAN Assessment:")
        
        if self.stats['response_times_us']:
            avg_response = statistics.mean(self.stats['response_times_us'])
            
            if avg_response <= 500:  # Sub-500μs
                grade = "PHYSICS-LIMITED EXCELLENCE 🌟⚡"
            elif avg_response <= 1000:  # Sub-1ms (within budget)
                grade = "EXCELLENT ✅"
            elif avg_response <= 2000:  # Sub-2ms
                grade = "GOOD ⚠️"
            else:
                grade = "NEEDS OPTIMIZATION ❌"
                
            budget_ok = budget_compliance >= 80 if 'budget_compliance' in locals() else False
            dedup_ok = (self.stats['duplicate_responses'] / max(1, self.stats['received_responses'])) >= 0.5
            
            print(f"  • Latency Grade: {grade}")
            print(f"  • Processing Budget: {'✅ PASS' if budget_ok else '❌ FAIL'}")
            print(f"  • Deduplication: {'✅ EFFECTIVE' if dedup_ok else '❌ INSUFFICIENT'}")
            
            if avg_response <= 1000 and budget_ok and dedup_ok:
                print(f"\n🏆 CONCLUSION: READY FOR METRO-SCALE DEPLOYMENT!")
                print(f"    Your transport sync meets the 1ms processing budget")
                print(f"    and handles metro WAN conditions with redundancy.")
            else:
                print(f"\n⚠️ CONCLUSION: Metro WAN performance needs optimization")
                print(f"    Consider additional redundancy or protocol optimization.")

def main():
    parser = argparse.ArgumentParser(description="Metro Area WAN Ultra-Low Latency Test")
    parser.add_argument("--group", default="239.255.77.77", help="Multicast group")
    parser.add_argument("--port", type=int, default=7777, help="UDP port")
    parser.add_argument("--duration", type=int, default=45, help="Test duration in seconds")
    parser.add_argument("--metro-radius", type=float, default=50.0, help="Metro area radius in km")
    parser.add_argument("--paths", type=int, default=4, help="Number of redundant paths")
    parser.add_argument("--budget", type=int, default=1000, help="Processing budget in microseconds")
    
    # Metro WAN scenarios
    parser.add_argument("--city-scale", action="store_true", help="City-scale metro WAN (100km)")
    parser.add_argument("--extreme-metro", action="store_true", help="Extreme metro conditions")
    parser.add_argument("--speed-of-light", action="store_true", help="Speed-of-light limited test")
    
    args = parser.parse_args()
    
    tester = MetroWANStressTester(args.group, args.port)
    
    # Configure based on scenario
    if args.city_scale:
        tester.wan_condition.metro_radius_km = 100.0
        tester.wan_condition.redundant_paths = 6
        tester.processing_budget_us = args.budget
        print("🏙️ City-Scale Metro WAN Scenario")
    elif args.extreme_metro:
        tester.wan_condition.packet_loss_rate = 0.40
        tester.wan_condition.jitter_us = 500.0
        tester.wan_condition.congestion_probability = 0.50
        tester.wan_condition.redundant_paths = 8
        print("⚡ EXTREME Metro WAN Scenario")
    elif args.speed_of_light:
        tester.wan_condition.metro_radius_km = 150.0  # Speed-of-light becomes significant
        tester.wan_condition.redundant_paths = 3
        print("⚡ Speed-of-Light Limited Scenario")
    else:
        # Custom configuration
        tester.wan_condition.metro_radius_km = args.metro_radius
        tester.wan_condition.redundant_paths = args.paths
        tester.processing_budget_us = args.budget
        print("🔧 Custom Metro WAN Scenario")
    
    # Recalculate speed of light limits
    tester.calculate_speed_of_light_limits()
    tester.setup_redundant_paths()
    
    try:
        tester.run_metro_wan_stress_test(args.duration)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    finally:
        if tester.sock:
            tester.sock.close()

if __name__ == "__main__":
    main() 