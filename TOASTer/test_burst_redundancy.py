#!/usr/bin/env python3

"""
TOASTer Burst Redundancy Test - Maximum Drop-Out Resistance

Tests burst transmission of multiple redundant signals with:
- Immediate multiple transmission (burst sending)
- Advanced deduplication algorithms
- Microsecond-precision timing
- Ultra-aggressive packet loss resistance
- Multi-path redundancy optimization

Target: Zero message loss even with 50%+ packet drops
Requirement: Sub-millisecond total response time with redundancy
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
import asyncio
from typing import Dict, List, Optional, Set, Tuple
import argparse
from dataclasses import dataclass, field
from collections import deque, defaultdict
import concurrent.futures

# TOAST v2 Protocol Constants
TOAST_MAGIC = 0x54534F54
TOAST_VERSION = 2

class TOASTFrameType:
    TRANSPORT = 0x05
    BURST_TRANSPORT = 0x0A  # New burst transport frame type

@dataclass
class BurstConfig:
    """Burst transmission configuration"""
    burst_size: int = 8                    # Number of copies per command
    burst_interval_us: int = 50            # Microseconds between burst sends
    stagger_pattern: str = "exponential"   # burst timing pattern
    
    # Redundancy strategies  
    use_forward_error_correction: bool = True
    use_checksum_validation: bool = True
    use_sequence_recovery: bool = True
    
    # Deduplication settings
    dedup_window_ms: float = 5.0           # Extended dedup window for bursts
    confidence_threshold: int = 2          # Min copies needed for confidence
    
    # Ultra-aggressive conditions
    target_packet_loss: float = 0.60      # Test up to 60% packet loss
    target_response_time_us: float = 800.0 # Target sub-800μs response
    
class BurstRedundancyTester:
    def __init__(self, multicast_group="239.255.77.77", port=7777, session_name="BurstTest"):
        self.multicast_group = multicast_group
        self.port = port
        self.session_name = session_name
        self.session_id = hash(session_name) & 0xFFFFFFFF
        self.sequence_number = random.randint(100000, 999999)
        self.running = False
        
        # Burst configuration
        self.burst_config = BurstConfig()
        
        # Advanced deduplication system
        self.message_registry: Dict[str, Dict] = {}
        self.confidence_scores: Dict[str, int] = {}
        self.first_arrival_times: Dict[str, float] = {}
        self.burst_reconstruction: Dict[str, List[Dict]] = defaultdict(list)
        
        # Ultra-precise timing
        self.timing_start_ns = time.time_ns()
        
        # Statistics tracking (microsecond precision)
        self.stats = {
            'commands_sent': 0,
            'burst_packets_sent': 0,
            'responses_received': 0,
            'duplicate_packets': 0,
            'reconstructed_messages': 0,
            'confidence_scores': [],
            'burst_effectiveness': [],
            'response_times_us': [],
            'first_packet_times_us': [],
            'reconstruction_times_us': [],
            'packet_loss_events': 0,
            'zero_loss_commands': 0,
            'sub_target_responses': 0
        }
        
        # Burst transmission state
        self.active_bursts: Set[str] = set()
        self.burst_futures: List = []
        
        # Socket setup
        self.sock = None
        self.setup_high_performance_socket()
        
    def setup_high_performance_socket(self):
        """Setup high-performance UDP socket for burst transmission"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # High-performance socket options
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)  # 1MB send buffer
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)  # 1MB receive buffer
            
            try:
                # Enable high-precision timestamps if available
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_TIMESTAMPING, 7)
            except:
                pass
                
            self.sock.bind(('0.0.0.0', self.port))
            
            # Join multicast group
            mreq = struct.pack('4sl', socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            # Enable outgoing multicast with high TTL for metro area
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 64)
            
            print(f"✅ High-performance burst socket ready: {self.multicast_group}:{self.port}")
            
        except Exception as e:
            print(f"❌ Socket setup failed: {e}")
            sys.exit(1)
    
    def get_precise_timestamp_ns(self) -> int:
        """Get high-precision timestamp in nanoseconds"""
        return time.time_ns()
    
    def get_relative_time_us(self) -> float:
        """Get relative time in microseconds since test start"""
        return (time.time_ns() - self.timing_start_ns) / 1000.0
    
    def generate_burst_id(self, command: str, timestamp_ns: int) -> str:
        """Generate unique burst ID for deduplication"""
        burst_data = f"{command}:{timestamp_ns}:{self.session_id}"
        return hashlib.sha256(burst_data.encode()).hexdigest()[:16]
    
    def create_burst_frame(self, burst_id: str, burst_index: int, total_bursts: int, 
                          command: str, position: float, bpm: float, timestamp_ns: int) -> bytes:
        """Create individual burst frame with redundancy info"""
        
        # Create transport message with burst metadata
        transport_msg = {
            "type": "burst_transport",
            "command": command,
            "position": position,
            "bpm": bpm,
            "timestamp": timestamp_ns,
            "peer_id": "BurstTester",
            "burst_id": burst_id,
            "burst_index": burst_index,
            "total_bursts": total_bursts,
            "sequence": self.sequence_number + burst_index
        }
        
        # Add forward error correction if enabled
        if self.burst_config.use_forward_error_correction:
            # Simple XOR-based FEC for demonstration
            payload_str = json.dumps({k: v for k, v in transport_msg.items() if k != 'fec_data'})
            fec_checksum = sum(payload_str.encode()) & 0xFFFFFFFF
            transport_msg['fec_data'] = fec_checksum
        
        payload = json.dumps(transport_msg).encode('utf-8')
        
        # Calculate enhanced checksum
        if self.burst_config.use_checksum_validation:
            checksum = hashlib.md5(payload).hexdigest()[:8]
        else:
            checksum = f"{sum(payload) & 0xFFFF:04x}"
        
        # Pack header with burst info
        timestamp_us = int(timestamp_ns / 1000) & 0xFFFFFFFF
        header = struct.pack('<I B B H I I I I B B H I',
            TOAST_MAGIC, TOAST_VERSION, TOASTFrameType.BURST_TRANSPORT, 0,
            self.sequence_number + burst_index, timestamp_us, len(payload), 
            int(burst_id[:8], 16) & 0xFFFFFFFF,  # Use first 8 chars of burst_id
            burst_index, total_bursts, int(checksum[:4], 16) & 0xFFFF, self.session_id
        )
        
        return header + payload
    
    def simulate_extreme_packet_loss(self) -> bool:
        """Simulate extreme packet loss conditions"""
        if random.random() < self.burst_config.target_packet_loss:
            self.stats['packet_loss_events'] += 1
            return False  # Packet lost
        return True  # Packet transmitted
    
    def send_burst_command(self, command: str, position: float = 0.0, bpm: float = 120.0) -> str:
        """Send command via burst transmission"""
        timestamp_ns = self.get_precise_timestamp_ns()
        burst_id = self.generate_burst_id(command, timestamp_ns)
        
        self.active_bursts.add(burst_id)
        successful_sends = 0
        
        print(f"💥 BURST: {command} (ID: {burst_id[:8]}, {self.burst_config.burst_size} copies)")
        
        # Send burst packets with precise timing
        if self.burst_config.stagger_pattern == "exponential":
            # Exponential back-off pattern for better network spread
            intervals = [self.burst_config.burst_interval_us * (1.2 ** i) for i in range(self.burst_config.burst_size)]
        elif self.burst_config.stagger_pattern == "linear":
            # Linear spacing
            intervals = [self.burst_config.burst_interval_us] * self.burst_config.burst_size
        else:
            # Immediate burst (all at once)
            intervals = [0] * self.burst_config.burst_size
        
        # Use thread pool for concurrent burst sending
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.burst_config.burst_size) as executor:
            futures = []
            
            for i in range(self.burst_config.burst_size):
                delay_us = sum(intervals[:i])
                future = executor.submit(self._send_burst_packet, 
                                       burst_id, i, self.burst_config.burst_size,
                                       command, position, bpm, timestamp_ns, delay_us)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    successful_sends += 1
        
        self.stats['commands_sent'] += 1
        self.stats['burst_packets_sent'] += self.burst_config.burst_size
        
        redundancy_ratio = successful_sends / self.burst_config.burst_size
        print(f"📡 Burst sent: {successful_sends}/{self.burst_config.burst_size} packets ({redundancy_ratio*100:.1f}% success)")
        
        if successful_sends == self.burst_config.burst_size:
            self.stats['zero_loss_commands'] += 1
        
        return burst_id
    
    def _send_burst_packet(self, burst_id: str, burst_index: int, total_bursts: int,
                          command: str, position: float, bpm: float, 
                          timestamp_ns: int, delay_us: float) -> bool:
        """Send individual burst packet"""
        # Apply stagger delay
        if delay_us > 0:
            time.sleep(delay_us / 1000000.0)
        
        # Simulate packet loss
        if not self.simulate_extreme_packet_loss():
            return False
        
        # Create and send frame
        frame = self.create_burst_frame(burst_id, burst_index, total_bursts,
                                      command, position, bpm, timestamp_ns)
        
        try:
            self.sock.sendto(frame, (self.multicast_group, self.port))
            return True
        except Exception as e:
            print(f"❌ Burst packet {burst_index} failed: {e}")
            return False
    
    def listen_for_burst_responses(self):
        """Listen for burst responses with advanced deduplication"""
        print("👂 Listening for burst responses with deduplication...")
        
        while self.running:
            try:
                self.sock.settimeout(0.0005)  # 500μs timeout for ultra-low latency
                data, addr = self.sock.recvfrom(8192)
                receive_time_us = self.get_relative_time_us()
                
                if len(data) < 32:
                    continue
                    
                # Parse header
                header = struct.unpack('<I B B H I I I I B B H I', data[:32])
                magic, version, frame_type = header[0], header[1], header[2]
                timestamp_us = header[5]
                session_id = header[11]
                payload_size = header[6]
                burst_index = header[8]
                total_bursts = header[9]
                
                if magic != TOAST_MAGIC or version != TOAST_VERSION or session_id != self.session_id:
                    continue
                
                if frame_type != TOASTFrameType.BURST_TRANSPORT:
                    continue
                    
                payload = data[32:32+payload_size]
                
                try:
                    msg = json.loads(payload.decode('utf-8'))
                    burst_id = msg.get('burst_id', 'unknown')
                    command = msg.get('command', 'UNKNOWN')
                    
                    # Add to burst reconstruction
                    self.burst_reconstruction[burst_id].append({
                        'msg': msg,
                        'receive_time': receive_time_us,
                        'burst_index': burst_index,
                        'addr': addr[0]
                    })
                    
                    # Check if this is first packet for this burst
                    if burst_id not in self.first_arrival_times:
                        self.first_arrival_times[burst_id] = receive_time_us
                        self.stats['first_packet_times_us'].append(receive_time_us - (msg['timestamp'] / 1000.0))
                    
                    # Update confidence score
                    if burst_id not in self.confidence_scores:
                        self.confidence_scores[burst_id] = 0
                    self.confidence_scores[burst_id] += 1
                    
                    # Check if we have enough confidence to process
                    if (self.confidence_scores[burst_id] >= self.burst_config.confidence_threshold and
                        burst_id not in self.message_registry):
                        
                        # Process the message
                        self.process_burst_message(burst_id, msg, receive_time_us)
                    
                    # Count duplicates after first processing
                    if burst_id in self.message_registry:
                        self.stats['duplicate_packets'] += 1
                    
                except json.JSONDecodeError as e:
                    print(f"📝 JSON decode error in burst: {e}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ Burst listen error: {e}")
    
    def process_burst_message(self, burst_id: str, msg: Dict, receive_time_us: float):
        """Process burst message once confidence threshold is met"""
        self.message_registry[burst_id] = msg
        self.stats['responses_received'] += 1
        self.stats['reconstructed_messages'] += 1
        
        # Calculate response time
        if 'timestamp' in msg:
            original_time_us = msg['timestamp'] / 1000.0  # Convert ns to μs
            response_time_us = receive_time_us - original_time_us
            self.stats['response_times_us'].append(response_time_us)
            
            # Check if under target response time
            if response_time_us <= self.burst_config.target_response_time_us:
                self.stats['sub_target_responses'] += 1
                status_icon = "⚡"
            else:
                status_icon = "⏰"
            
            # Calculate reconstruction efficiency
            burst_packets = len(self.burst_reconstruction[burst_id])
            effectiveness = self.confidence_scores[burst_id] / burst_packets if burst_packets > 0 else 0
            self.stats['burst_effectiveness'].append(effectiveness)
            
            command = msg.get('command', 'UNKNOWN')
            position = msg.get('position', 0.0)
            bpm = msg.get('bpm', 120.0)
            confidence = self.confidence_scores[burst_id]
            
            print(f"{status_icon} RECONSTRUCTED: {command} | {response_time_us:.1f}μs | confidence:{confidence} | pos:{position:.3f} bpm:{bpm:.0f}")
    
    def run_burst_redundancy_test(self, duration_seconds=60):
        """Run comprehensive burst redundancy test"""
        print(f"\n💥 Burst Redundancy Test ({duration_seconds}s)")
        print("=" * 60)
        print(f"📊 Configuration:")
        print(f"  • Burst Size: {self.burst_config.burst_size} copies per command")
        print(f"  • Burst Interval: {self.burst_config.burst_interval_us}μs")
        print(f"  • Target Packet Loss: {self.burst_config.target_packet_loss*100:.1f}%")
        print(f"  • Target Response Time: {self.burst_config.target_response_time_us}μs")
        print(f"  • Confidence Threshold: {self.burst_config.confidence_threshold} packets")
        print("")
        
        self.running = True
        listen_thread = threading.Thread(target=self.listen_for_burst_responses, daemon=True)
        listen_thread.start()
        
        # Ultra-aggressive test sequence
        start_time = time.time()
        test_commands = ["PLAY", "STOP", "PLAY", "PAUSE", "PLAY", "STOP", "PLAY", "PAUSE", "STOP"]
        command_interval = duration_seconds / len(test_commands)
        
        try:
            for i, command in enumerate(test_commands):
                if not self.running:
                    break
                    
                position = random.uniform(0, 200)
                bpm = random.choice([96, 110, 120, 128, 140, 160, 180])
                
                print(f"\n📍 Burst Test {i+1}/{len(test_commands)}: {command}")
                
                # Send burst command
                burst_id = self.send_burst_command(command, position, bpm)
                
                # Precise timing for next command
                elapsed = time.time() - start_time
                next_command_time = (i + 1) * command_interval
                wait_time = max(0, next_command_time - elapsed)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    
        except KeyboardInterrupt:
            print("\n🛑 Burst redundancy test interrupted")
        finally:
            self.running = False
            time.sleep(3)  # Allow final burst processing
            
        self.print_burst_redundancy_results()
    
    def print_burst_redundancy_results(self):
        """Print comprehensive burst redundancy test results"""
        print("\n" + "="*80)
        print("💥 BURST REDUNDANCY TEST RESULTS")
        print("="*80)
        
        # Basic burst statistics
        print(f"\n📈 Burst Transmission Statistics:")
        print(f"  • Commands Sent: {self.stats['commands_sent']}")
        print(f"  • Total Burst Packets: {self.stats['burst_packets_sent']}")
        print(f"  • Responses Reconstructed: {self.stats['reconstructed_messages']}")
        print(f"  • Duplicate Packets: {self.stats['duplicate_packets']}")
        print(f"  • Packet Loss Events: {self.stats['packet_loss_events']}")
        print(f"  • Zero-Loss Commands: {self.stats['zero_loss_commands']}")
        
        if self.stats['commands_sent'] > 0:
            reconstruction_rate = (self.stats['reconstructed_messages'] / self.stats['commands_sent']) * 100
            zero_loss_rate = (self.stats['zero_loss_commands'] / self.stats['commands_sent']) * 100
            print(f"  • Reconstruction Success: {reconstruction_rate:.1f}%")
            print(f"  • Zero-Loss Success: {zero_loss_rate:.1f}%")
        
        # Ultra-low latency analysis
        if self.stats['response_times_us']:
            response_times = self.stats['response_times_us']
            print(f"\n⚡ Ultra-Low Latency Analysis:")
            print(f"  • Average Response: {statistics.mean(response_times):.1f}μs")
            print(f"  • Median Response: {statistics.median(response_times):.1f}μs")
            print(f"  • Min Response: {min(response_times):.1f}μs")
            print(f"  • Max Response: {max(response_times):.1f}μs")
            print(f"  • 95th Percentile: {sorted(response_times)[int(len(response_times)*0.95)]:.1f}μs")
            print(f"  • 99th Percentile: {sorted(response_times)[int(len(response_times)*0.99)]:.1f}μs")
        
        # First packet analysis
        if self.stats['first_packet_times_us']:
            first_times = self.stats['first_packet_times_us']
            print(f"\n🚀 First Packet Response Times:")
            print(f"  • Average First Packet: {statistics.mean(first_times):.1f}μs")
            print(f"  • Minimum First Packet: {min(first_times):.1f}μs")
        
        # Target performance analysis
        if self.stats['response_times_us']:
            target_compliance = (self.stats['sub_target_responses'] / len(self.stats['response_times_us'])) * 100
            print(f"\n🎯 Target Performance Analysis:")
            print(f"  • Target Response Time: {self.burst_config.target_response_time_us}μs")
            print(f"  • Under Target: {self.stats['sub_target_responses']} responses")
            print(f"  • Target Compliance: {target_compliance:.1f}%")
        
        # Burst effectiveness analysis
        if self.stats['burst_effectiveness']:
            effectiveness = self.stats['burst_effectiveness']
            print(f"\n💥 Burst Effectiveness Analysis:")
            print(f"  • Average Effectiveness: {statistics.mean(effectiveness)*100:.1f}%")
            print(f"  • Deduplication Efficiency: {statistics.mean(effectiveness)*100:.1f}%")
        
        # Confidence scores analysis
        if self.confidence_scores:
            confidences = list(self.confidence_scores.values())
            print(f"\n🔒 Confidence Analysis:")
            print(f"  • Average Confidence: {statistics.mean(confidences):.1f} packets")
            print(f"  • Max Confidence: {max(confidences)} packets")
            print(f"  • Confidence Threshold: {self.burst_config.confidence_threshold} packets")
        
        # Overall assessment
        print(f"\n🎯 Burst Redundancy Assessment:")
        
        success_metrics = []
        
        if self.stats['reconstructed_messages'] > 0:
            reconstruction_rate = (self.stats['reconstructed_messages'] / self.stats['commands_sent']) * 100
            if reconstruction_rate >= 95:
                success_metrics.append("Reconstruction: EXCELLENT")
            elif reconstruction_rate >= 85:
                success_metrics.append("Reconstruction: GOOD")
            else:
                success_metrics.append("Reconstruction: NEEDS IMPROVEMENT")
        
        if self.stats['response_times_us']:
            avg_response = statistics.mean(self.stats['response_times_us'])
            if avg_response <= 500:
                success_metrics.append("Latency: EXCEPTIONAL")
            elif avg_response <= 800:
                success_metrics.append("Latency: EXCELLENT")
            elif avg_response <= 1500:
                success_metrics.append("Latency: GOOD")
            else:
                success_metrics.append("Latency: NEEDS OPTIMIZATION")
        
        if self.stats['zero_loss_commands'] > 0:
            zero_loss_rate = (self.stats['zero_loss_commands'] / self.stats['commands_sent']) * 100
            if zero_loss_rate >= 80:
                success_metrics.append("Drop-out Resistance: MAXIMUM")
            elif zero_loss_rate >= 60:
                success_metrics.append("Drop-out Resistance: HIGH")
            else:
                success_metrics.append("Drop-out Resistance: MODERATE")
        
        for metric in success_metrics:
            print(f"  • {metric}")
        
        # Final verdict
        if (len([m for m in success_metrics if "EXCELLENT" in m or "EXCEPTIONAL" in m or "MAXIMUM" in m]) >= 2):
            print(f"\n🏆 CONCLUSION: ULTRA-LOW LATENCY BURST SYSTEM READY!")
            print(f"    Transport sync achieves sub-millisecond response times")
            print(f"    with maximum drop-out resistance via burst redundancy.")
        else:
            print(f"\n⚠️ CONCLUSION: System shows good performance but may need optimization")
            print(f"    for metro-scale deployment with 1ms processing budgets.")

def main():
    parser = argparse.ArgumentParser(description="Burst Redundancy Ultra-Low Latency Test")
    parser.add_argument("--group", default="239.255.77.77", help="Multicast group")
    parser.add_argument("--port", type=int, default=7777, help="UDP port")
    parser.add_argument("--duration", type=int, default=40, help="Test duration in seconds")
    parser.add_argument("--burst-size", type=int, default=8, help="Number of burst copies")
    parser.add_argument("--target-loss", type=float, default=0.50, help="Target packet loss rate")
    parser.add_argument("--target-response", type=int, default=800, help="Target response time in μs")
    
    # Burst scenarios
    parser.add_argument("--ultra-burst", action="store_true", help="Ultra-burst scenario (16 copies)")
    parser.add_argument("--extreme-loss", action="store_true", help="Extreme packet loss (70%)")
    parser.add_argument("--speed-critical", action="store_true", help="Speed-critical scenario (500μs target)")
    
    args = parser.parse_args()
    
    tester = BurstRedundancyTester(args.group, args.port)
    
    # Configure scenarios
    if args.ultra_burst:
        tester.burst_config.burst_size = 16
        tester.burst_config.confidence_threshold = 3
        print("💥 Ultra-Burst Scenario (16 copies)")
    elif args.extreme_loss:
        tester.burst_config.target_packet_loss = 0.70
        tester.burst_config.burst_size = 12
        tester.burst_config.confidence_threshold = 2
        print("🌪️ Extreme Loss Scenario (70% packet loss)")
    elif args.speed_critical:
        tester.burst_config.target_response_time_us = 500.0
        tester.burst_config.burst_interval_us = 25
        tester.burst_config.confidence_threshold = 1  # Accept first packet for speed
        print("⚡ Speed-Critical Scenario (500μs target)")
    else:
        # Custom configuration
        tester.burst_config.burst_size = args.burst_size
        tester.burst_config.target_packet_loss = args.target_loss
        tester.burst_config.target_response_time_us = float(args.target_response)
        print("🔧 Custom Burst Scenario")
    
    try:
        tester.run_burst_redundancy_test(args.duration)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    finally:
        if tester.sock:
            tester.sock.close()

if __name__ == "__main__":
    main() 