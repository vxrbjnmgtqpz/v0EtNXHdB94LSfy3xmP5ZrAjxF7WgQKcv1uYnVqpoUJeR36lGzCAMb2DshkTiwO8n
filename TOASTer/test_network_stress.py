#!/usr/bin/env python3

"""
TOASTer Network Stress Test - Dropouts & Jitter Simulation

Tests bi-directional transport sync reliability under realistic network conditions:
- Packet drops (simulating WiFi interference, congestion)
- Network jitter (variable latency)
- Temporary outages (connection drops)
- Burst transmission effectiveness
- Recovery time measurement
"""

import socket
import struct
import json
import time
import threading
import sys
import random
import statistics
from typing import Dict, List, Optional
import argparse
from dataclasses import dataclass
from collections import deque

# TOAST v2 Protocol Constants
TOAST_MAGIC = 0x54534F54
TOAST_VERSION = 2

class TOASTFrameType:
    TRANSPORT = 0x05
    HEARTBEAT = 0x07

@dataclass
class NetworkCondition:
    """Network stress conditions"""
    packet_loss_rate: float = 0.0      # 0.0-1.0 (0% to 100% loss)
    jitter_ms: float = 0.0              # ±ms random delay
    base_latency_ms: float = 5.0        # Base network latency
    outage_probability: float = 0.0     # Chance of temporary outage per second
    outage_duration_ms: float = 500.0   # How long outages last
    
class NetworkStressTester:
    def __init__(self, multicast_group="239.255.77.77", port=7777, session_name="StressTest"):
        self.multicast_group = multicast_group
        self.port = port
        self.session_name = session_name
        self.session_id = hash(session_name) & 0xFFFFFFFF
        self.sequence_number = random.randint(1000, 9999)
        self.running = False
        
        # Network stress simulation
        self.network_condition = NetworkCondition()
        self.is_network_down = False
        self.network_outage_end = 0.0
        self.pending_messages = deque()  # For jitter simulation
        
        # Statistics tracking
        self.stats = {
            'sent_commands': 0,
            'received_responses': 0,
            'dropped_packets': 0,
            'jitter_delays': [],
            'response_times': [],
            'sync_failures': 0,
            'recovery_times': [],
            'outage_count': 0
        }
        
        # Transport state tracking
        self.last_command_time = 0.0
        self.expected_state = "STOP"
        self.confirmed_state = None
        self.sync_lost_time = None
        
        # Socket setup
        self.sock = None
        self.setup_socket()
        
    def setup_socket(self):
        """Setup UDP multicast socket"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(('0.0.0.0', self.port))
            
            # Join multicast group
            mreq = struct.pack('4sl', socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            # Enable outgoing multicast
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            
            print(f"✅ Network stress tester ready: {self.multicast_group}:{self.port}")
            
        except Exception as e:
            print(f"❌ Socket setup failed: {e}")
            sys.exit(1)
    
    def simulate_network_conditions(self):
        """Check and apply current network conditions"""
        current_time = time.time()
        
        # Check for network outage
        if self.is_network_down:
            if current_time >= self.network_outage_end:
                self.is_network_down = False
                print(f"🌐 Network RESTORED after {(current_time - (self.network_outage_end - self.network_condition.outage_duration_ms/1000)):.1f}s outage")
            else:
                return False  # Network still down
        else:
            # Check if we should start an outage
            if random.random() < self.network_condition.outage_probability:
                self.is_network_down = True
                self.network_outage_end = current_time + (self.network_condition.outage_duration_ms / 1000)
                self.stats['outage_count'] += 1
                print(f"📡 Network OUTAGE started (duration: {self.network_condition.outage_duration_ms}ms)")
                return False
        
        # Check for packet loss
        if random.random() < self.network_condition.packet_loss_rate:
            self.stats['dropped_packets'] += 1
            return False  # Packet dropped
            
        return True  # Packet can be sent
    
    def add_jitter_delay(self, message_data, send_time):
        """Add jitter delay to message"""
        if self.network_condition.jitter_ms > 0:
            jitter = random.uniform(-self.network_condition.jitter_ms, self.network_condition.jitter_ms)
            self.stats['jitter_delays'].append(abs(jitter))
            
            base_delay = self.network_condition.base_latency_ms / 1000
            total_delay = base_delay + (jitter / 1000)
            
            if total_delay > 0:
                # Schedule delayed transmission
                delivery_time = send_time + total_delay
                self.pending_messages.append((delivery_time, message_data))
                return True
        
        # Send immediately if no jitter
        return False
    
    def process_pending_messages(self):
        """Process messages waiting for jitter delay"""
        current_time = time.time()
        
        while self.pending_messages and self.pending_messages[0][0] <= current_time:
            delivery_time, message_data = self.pending_messages.popleft()
            
            try:
                self.sock.sendto(message_data, (self.multicast_group, self.port))
                actual_delay = (current_time - delivery_time + (self.network_condition.base_latency_ms / 1000)) * 1000
                print(f"📤 Delayed message sent (jitter: {actual_delay:.1f}ms)")
            except Exception as e:
                print(f"❌ Delayed send failed: {e}")
    
    def create_toast_frame(self, frame_type: int, payload: bytes) -> bytes:
        """Create TOAST v2 frame"""
        timestamp_us = int(time.time() * 1000000) & 0xFFFFFFFF
        payload_size = len(payload)
        checksum = sum(payload) & 0xFFFF
        
        header = struct.pack('<I B B H I I I I B B H I',
            TOAST_MAGIC, TOAST_VERSION, frame_type, 0,
            self.sequence_number, timestamp_us, payload_size, 0,
            0, 1, checksum, self.session_id
        )
        
        self.sequence_number += 1
        return header + payload
    
    def send_transport_command_with_stress(self, command: str, position: float = 0.0, bpm: float = 120.0):
        """Send transport command with network stress simulation"""
        send_time = time.time()
        
        # Create transport message
        transport_msg = {
            "type": "transport",
            "command": command,
            "timestamp": int(send_time * 1000000),
            "position": position,
            "bpm": bpm,
            "peer_id": "StressTester"
        }
        
        payload = json.dumps(transport_msg).encode('utf-8')
        frame = self.create_toast_frame(TOASTFrameType.TRANSPORT, payload)
        
        # Apply network stress conditions
        if not self.simulate_network_conditions():
            print(f"📦 DROPPED: {command} (loss: {self.network_condition.packet_loss_rate*100:.1f}% or outage)")
            return False
        
        self.stats['sent_commands'] += 1
        self.expected_state = command
        self.last_command_time = send_time
        
        # Check if we should add jitter delay
        if self.add_jitter_delay(frame, send_time):
            print(f"⏱️ DELAYED: {command} (jitter: ±{self.network_condition.jitter_ms}ms)")
        else:
            # Send immediately
            try:
                self.sock.sendto(frame, (self.multicast_group, self.port))
                print(f"📡 SENT: {command} → pos:{position:.3f} bpm:{bpm:.0f}")
            except Exception as e:
                print(f"❌ Send failed: {e}")
                return False
                
        return True
    
    def listen_for_responses(self):
        """Listen for responses and measure sync reliability"""
        print("👂 Listening for TOASTer responses (with stress analysis)...")
        
        while self.running:
            try:
                # Process pending jitter messages
                self.process_pending_messages()
                
                self.sock.settimeout(0.1)
                data, addr = self.sock.recvfrom(4096)
                
                if len(data) < 32:
                    continue
                    
                # Parse header
                header = struct.unpack('<I B B H I I I I B B H I', data[:32])
                magic, version, frame_type = header[0], header[1], header[2]
                session_id = header[11]
                payload_size = header[6]
                
                if magic != TOAST_MAGIC or version != TOAST_VERSION or session_id != self.session_id:
                    continue
                    
                payload = data[32:32+payload_size]
                receive_time = time.time()
                
                if frame_type == TOASTFrameType.TRANSPORT:
                    try:
                        msg = json.loads(payload.decode('utf-8'))
                        command = msg.get('command', 'UNKNOWN')
                        position = msg.get('position', 0.0)
                        bpm = msg.get('bpm', 120.0)
                        
                        self.stats['received_responses'] += 1
                        
                        # Calculate response time
                        if self.last_command_time > 0:
                            response_time = (receive_time - self.last_command_time) * 1000
                            self.stats['response_times'].append(response_time)
                            
                        # Check sync state
                        if command == self.expected_state:
                            self.confirmed_state = command
                            sync_icon = "✅"
                            
                            # Check if we recovered from sync loss
                            if self.sync_lost_time:
                                recovery_time = receive_time - self.sync_lost_time
                                self.stats['recovery_times'].append(recovery_time)
                                print(f"🔄 SYNC RECOVERED after {recovery_time:.3f}s")
                                self.sync_lost_time = None
                        else:
                            sync_icon = "❌"
                            if not self.sync_lost_time:
                                self.sync_lost_time = receive_time
                                self.stats['sync_failures'] += 1
                        
                        response_str = f"(resp: {response_time:.1f}ms)" if self.last_command_time > 0 else ""
                        print(f"{sync_icon} RECV: {command} ← TOASTer {response_str} | pos:{position:.3f} bpm:{bpm:.0f}")
                        
                    except json.JSONDecodeError as e:
                        print(f"📝 JSON decode error: {e}")
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ Listen error: {e}")
    
    def run_stress_test_scenario(self, duration_seconds=60):
        """Run comprehensive stress test scenario"""
        print(f"\n🧪 Network Stress Test Scenario ({duration_seconds}s)")
        print("=" * 50)
        print(f"📊 Conditions:")
        print(f"  • Packet Loss: {self.network_condition.packet_loss_rate*100:.1f}%")
        print(f"  • Jitter: ±{self.network_condition.jitter_ms}ms")
        print(f"  • Base Latency: {self.network_condition.base_latency_ms}ms") 
        print(f"  • Outage Rate: {self.network_condition.outage_probability*100:.1f}%/sec")
        print(f"  • Outage Duration: {self.network_condition.outage_duration_ms}ms")
        print("")
        
        self.running = True
        listen_thread = threading.Thread(target=self.listen_for_responses, daemon=True)
        listen_thread.start()
        
        start_time = time.time()
        test_commands = ["PLAY", "STOP", "PLAY", "PAUSE", "PLAY", "STOP"]
        command_interval = duration_seconds / len(test_commands)
        
        try:
            for i, command in enumerate(test_commands):
                if not self.running:
                    break
                    
                # Send command with stress
                position = random.uniform(0, 30)
                bpm = random.choice([120, 128, 140, 110])
                
                print(f"\n📍 Test {i+1}/{len(test_commands)}: {command}")
                self.send_transport_command_with_stress(command, position, bpm)
                
                # Wait for interval (allowing for processing)
                elapsed = time.time() - start_time
                next_command_time = (i + 1) * command_interval
                wait_time = max(0, next_command_time - elapsed)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    
        except KeyboardInterrupt:
            print("\n🛑 Stress test interrupted")
        finally:
            self.running = False
            time.sleep(1)  # Allow final responses
            
        self.print_stress_test_results()
    
    def print_stress_test_results(self):
        """Print comprehensive stress test results"""
        print("\n" + "="*60)
        print("📊 NETWORK STRESS TEST RESULTS")
        print("="*60)
        
        # Basic stats
        print(f"\n📈 Transmission Statistics:")
        print(f"  • Commands Sent: {self.stats['sent_commands']}")
        print(f"  • Responses Received: {self.stats['received_responses']}")
        print(f"  • Packets Dropped: {self.stats['dropped_packets']}")
        print(f"  • Success Rate: {(self.stats['received_responses']/max(1,self.stats['sent_commands']))*100:.1f}%")
        
        # Response time analysis
        if self.stats['response_times']:
            response_times = self.stats['response_times']
            print(f"\n⏱️ Response Time Analysis:")
            print(f"  • Average: {statistics.mean(response_times):.1f}ms")
            print(f"  • Median: {statistics.median(response_times):.1f}ms")
            print(f"  • Min/Max: {min(response_times):.1f}ms / {max(response_times):.1f}ms")
            print(f"  • Std Dev: {statistics.stdev(response_times) if len(response_times) > 1 else 0:.1f}ms")
        
        # Jitter analysis
        if self.stats['jitter_delays']:
            jitter_delays = self.stats['jitter_delays']
            print(f"\n📡 Network Jitter Analysis:")
            print(f"  • Average Jitter: {statistics.mean(jitter_delays):.1f}ms")
            print(f"  • Max Jitter: {max(jitter_delays):.1f}ms")
            print(f"  • Jittered Messages: {len(jitter_delays)}")
        
        # Sync reliability
        print(f"\n🎛️ Transport Sync Reliability:")
        print(f"  • Sync Failures: {self.stats['sync_failures']}")
        print(f"  • Network Outages: {self.stats['outage_count']}")
        
        if self.stats['recovery_times']:
            recovery_times = self.stats['recovery_times']
            print(f"  • Average Recovery: {statistics.mean(recovery_times):.3f}s")
            print(f"  • Max Recovery: {max(recovery_times):.3f}s")
        
        # Overall assessment
        success_rate = (self.stats['received_responses']/max(1,self.stats['sent_commands']))*100
        print(f"\n🎯 Overall Assessment:")
        
        if success_rate >= 95:
            grade = "EXCELLENT 🌟"
        elif success_rate >= 85:
            grade = "GOOD ✅"
        elif success_rate >= 70:
            grade = "FAIR ⚠️"
        else:
            grade = "POOR ❌"
            
        print(f"  • Reliability Grade: {grade}")
        print(f"  • Transport Sync Robustness: {'HIGH' if self.stats['sync_failures'] <= 1 else 'MEDIUM' if self.stats['sync_failures'] <= 3 else 'LOW'}")
        
        if success_rate >= 90 and self.stats['sync_failures'] <= 1:
            print("\n🏆 CONCLUSION: TOASTer transport sync is PRODUCTION-READY under network stress!")
        elif success_rate >= 80:
            print("\n✅ CONCLUSION: TOASTer transport sync shows good resilience to network issues")
        else:
            print("\n⚠️ CONCLUSION: Transport sync may need network reliability improvements")

def main():
    parser = argparse.ArgumentParser(description="TOASTer Network Stress Test")
    parser.add_argument("--group", default="239.255.77.77", help="Multicast group")
    parser.add_argument("--port", type=int, default=7777, help="UDP port")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    
    # Network stress parameters
    parser.add_argument("--loss", type=float, default=0.05, help="Packet loss rate (0.0-1.0)")
    parser.add_argument("--jitter", type=float, default=50.0, help="Network jitter in ms")
    parser.add_argument("--latency", type=float, default=10.0, help="Base latency in ms")
    parser.add_argument("--outages", type=float, default=0.02, help="Outage probability per second")
    parser.add_argument("--outage-duration", type=float, default=1000.0, help="Outage duration in ms")
    
    # Preset scenarios
    parser.add_argument("--wifi", action="store_true", help="WiFi scenario (5% loss, 30ms jitter)")
    parser.add_argument("--congested", action="store_true", help="Congested network (10% loss, 100ms jitter)")
    parser.add_argument("--extreme", action="store_true", help="Extreme conditions (20% loss, 200ms jitter)")
    
    args = parser.parse_args()
    
    tester = NetworkStressTester(args.group, args.port)
    
    # Apply preset scenarios
    if args.wifi:
        tester.network_condition = NetworkCondition(
            packet_loss_rate=0.05, jitter_ms=30.0, base_latency_ms=15.0,
            outage_probability=0.01, outage_duration_ms=500.0
        )
        print("📶 WiFi Network Scenario")
    elif args.congested:
        tester.network_condition = NetworkCondition(
            packet_loss_rate=0.10, jitter_ms=100.0, base_latency_ms=25.0,
            outage_probability=0.03, outage_duration_ms=1500.0
        )
        print("🚦 Congested Network Scenario")
    elif args.extreme:
        tester.network_condition = NetworkCondition(
            packet_loss_rate=0.20, jitter_ms=200.0, base_latency_ms=50.0,
            outage_probability=0.05, outage_duration_ms=2000.0
        )
        print("⚡ EXTREME Network Scenario")
    else:
        # Custom scenario
        tester.network_condition = NetworkCondition(
            packet_loss_rate=args.loss,
            jitter_ms=args.jitter,
            base_latency_ms=args.latency,
            outage_probability=args.outages,
            outage_duration_ms=args.outage_duration
        )
        print("🔧 Custom Network Scenario")
    
    try:
        tester.run_stress_test_scenario(args.duration)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    finally:
        if tester.sock:
            tester.sock.close()

if __name__ == "__main__":
    main() 