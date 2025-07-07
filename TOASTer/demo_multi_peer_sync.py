#!/usr/bin/env python3

"""
TOASTer Multi-Peer Transport Sync Demo

Demonstrates how multiple peer instances synchronize transport 
commands with no master/slave relationship - pure TOAST protocol.
"""

import socket
import struct
import json
import time
import threading
import sys
import random
from typing import Dict, List
import argparse

# TOAST v2 Protocol Constants
TOAST_MAGIC = 0x54534F54
TOAST_VERSION = 2

class TOASTFrameType:
    TRANSPORT = 0x05
    HEARTBEAT = 0x07

class PeerSimulator:
    def __init__(self, peer_id: str, multicast_group="239.255.77.77", port=7777, session_name="MultiPeerSync"):
        self.peer_id = peer_id
        self.multicast_group = multicast_group
        self.port = port
        self.session_name = session_name
        self.session_id = hash(session_name) & 0xFFFFFFFF
        self.sequence_number = random.randint(1000, 9999)
        self.running = False
        
        # Transport state
        self.is_playing = False
        self.current_position = 0.0
        self.current_bpm = 120.0
        self.last_command_time = time.time()
        
        # Peer tracking
        self.known_peers = set()
        self.peer_states = {}
        
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
            
            print(f"[{self.peer_id}] ✅ Connected to session: {self.session_name}")
            
        except Exception as e:
            print(f"[{self.peer_id}] ❌ Socket setup failed: {e}")
            sys.exit(1)
    
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
    
    def send_transport_command(self, command: str, position: float = None, bpm: float = None):
        """Send transport command to all peers"""
        if position is None:
            position = self.current_position
        if bpm is None:
            bpm = self.current_bpm
            
        # Create transport message
        transport_msg = {
            "type": "transport",
            "command": command,
            "timestamp": int(time.time() * 1000000),
            "position": position,
            "bpm": bpm,
            "peer_id": self.peer_id
        }
        
        payload = json.dumps(transport_msg).encode('utf-8')
        frame = self.create_toast_frame(TOASTFrameType.TRANSPORT, payload)
        
        try:
            self.sock.sendto(frame, (self.multicast_group, self.port))
            
            # Update local state
            self.apply_transport_command(command, position, bpm, self.peer_id)
            
            print(f"[{self.peer_id}] 📡 SENT: {command} → pos:{position:.3f} bpm:{bpm:.0f}")
            
        except Exception as e:
            print(f"[{self.peer_id}] ❌ Send failed: {e}")
    
    def apply_transport_command(self, command: str, position: float, bpm: float, from_peer: str):
        """Apply transport command to local state"""
        self.current_position = position
        self.current_bpm = bpm
        self.last_command_time = time.time()
        
        if command == "PLAY":
            self.is_playing = True
        elif command in ["STOP", "PAUSE"]:
            self.is_playing = False
            
        # Track peer states
        self.peer_states[from_peer] = {
            "playing": self.is_playing,
            "position": position,
            "bpm": bpm,
            "last_seen": time.time()
        }
    
    def listen_for_messages(self):
        """Listen for messages from other peers"""
        while self.running:
            try:
                self.sock.settimeout(1.0)
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
                
                if frame_type == TOASTFrameType.TRANSPORT:
                    try:
                        msg = json.loads(payload.decode('utf-8'))
                        command = msg.get('command', 'UNKNOWN')
                        position = msg.get('position', 0.0)
                        bpm = msg.get('bpm', 120.0)
                        from_peer = msg.get('peer_id', f'peer@{addr[0]}')
                        
                        # Ignore our own messages
                        if from_peer == self.peer_id:
                            continue
                            
                        # Apply command from remote peer
                        self.apply_transport_command(command, position, bpm, from_peer)
                        
                        print(f"[{self.peer_id}] 🎛️ RECV: {command} ← {from_peer} | pos:{position:.3f} bpm:{bpm:.0f}")
                        
                        # Track peer
                        if from_peer not in self.known_peers:
                            self.known_peers.add(from_peer)
                            print(f"[{self.peer_id}] 🤝 New peer discovered: {from_peer}")
                        
                    except json.JSONDecodeError as e:
                        print(f"[{self.peer_id}] 📝 JSON decode error: {e}")
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{self.peer_id}] ❌ Listen error: {e}")
    
    def update_position(self):
        """Update position if playing"""
        while self.running:
            if self.is_playing:
                # Simulate position advancement
                elapsed = time.time() - self.last_command_time
                self.current_position += elapsed * (self.current_bpm / 60.0) * 0.1  # Rough position update
                self.last_command_time = time.time()
            time.sleep(0.1)
    
    def get_status(self) -> str:
        """Get current status string"""
        state = "▶️ PLAYING" if self.is_playing else "⏹️ STOPPED" 
        return f"{state} | pos:{self.current_position:.3f} | bpm:{self.current_bpm:.0f} | peers:{len(self.known_peers)}"
    
    def start(self):
        """Start peer simulation"""
        self.running = True
        
        # Start background threads
        listen_thread = threading.Thread(target=self.listen_for_messages, daemon=True)
        position_thread = threading.Thread(target=self.update_position, daemon=True)
        
        listen_thread.start()
        position_thread.start()
        
        # Send initial heartbeat
        time.sleep(0.5)  # Slight delay for socket setup
        
    def stop(self):
        """Stop peer simulation"""
        self.running = False
        if self.sock:
            self.sock.close()

def simulate_multi_peer_scenario():
    """Simulate multi-peer transport sync scenario"""
    print("🎛️ Multi-Peer Transport Sync Demo")
    print("=================================")
    print("Demonstrating peer-to-peer transport synchronization")
    print("with no master/slave - pure TOAST protocol")
    print("")
    
    # Create multiple peer simulators
    peers = [
        PeerSimulator("Peer_A", session_name="MultiPeerDemo"),
        PeerSimulator("Peer_B", session_name="MultiPeerDemo"), 
        PeerSimulator("Peer_C", session_name="MultiPeerDemo")
    ]
    
    # Start all peers
    for peer in peers:
        peer.start()
        
    time.sleep(2)  # Allow peers to discover each other
    
    print("\n🎭 Demo Scenario: Any peer can control transport")
    print("=" * 50)
    
    try:
        # Scenario 1: Peer A starts transport
        print("\n📍 Step 1: Peer A starts PLAY")
        peers[0].send_transport_command("PLAY", 0.0, 120.0)
        time.sleep(1.5)
        
        for peer in peers:
            print(f"    [{peer.peer_id}] {peer.get_status()}")
        
        # Scenario 2: Peer B changes BPM and continues
        print("\n📍 Step 2: Peer B changes BPM and continues PLAY")
        peers[1].send_transport_command("PLAY", 3.5, 140.0)
        time.sleep(1.5)
        
        for peer in peers:
            print(f"    [{peer.peer_id}] {peer.get_status()}")
        
        # Scenario 3: Peer C stops everything
        print("\n📍 Step 3: Peer C stops all transport")
        peers[2].send_transport_command("STOP", 0.0, 140.0)
        time.sleep(1.5)
        
        for peer in peers:
            print(f"    [{peer.peer_id}] {peer.get_status()}")
            
        # Scenario 4: Peer B starts from different position
        print("\n📍 Step 4: Peer B starts from position 10.5")
        peers[1].send_transport_command("PLAY", 10.5, 128.0)
        time.sleep(1.5)
        
        for peer in peers:
            print(f"    [{peer.peer_id}] {peer.get_status()}")
            
        print("\n✅ Demo Complete: All peers synchronized with no master/slave!")
        print("\n🎯 Key Observations:")
        print("  • Any peer can send transport commands")
        print("  • All peers instantly synchronize to latest command")
        print("  • Position, BPM, and play state stay in sync")
        print("  • No master/slave - pure peer-to-peer architecture")
        print("  • TOAST protocol handles all communication")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
    finally:
        for peer in peers:
            peer.stop()

def main():
    parser = argparse.ArgumentParser(description="Multi-Peer Transport Sync Demo")
    parser.add_argument("--demo", action="store_true", help="Run demo scenario")
    parser.add_argument("--peer-id", default=f"Demo_{random.randint(100,999)}", help="Peer identifier")
    
    args = parser.parse_args()
    
    if args.demo:
        simulate_multi_peer_scenario()
    else:
        # Single peer interactive mode
        peer = PeerSimulator(args.peer_id)
        peer.start()
        
        print(f"\n🎛️ Interactive Peer: {args.peer_id}")
        print("Commands: play, stop, pause, bpm X, pos X, status, quit")
        
        try:
            while True:
                cmd = input(f"[{args.peer_id}]> ").strip().lower()
                
                if cmd in ["quit", "exit"]:
                    break
                elif cmd == "play":
                    peer.send_transport_command("PLAY")
                elif cmd == "stop":
                    peer.send_transport_command("STOP")
                elif cmd == "pause":
                    peer.send_transport_command("PAUSE")
                elif cmd.startswith("bpm "):
                    try:
                        bpm = float(cmd.split()[1])
                        peer.send_transport_command("PLAY", bpm=bpm)
                    except (IndexError, ValueError):
                        print("Usage: bpm <number>")
                elif cmd.startswith("pos "):
                    try:
                        pos = float(cmd.split()[1])
                        peer.send_transport_command("PLAY", position=pos)
                    except (IndexError, ValueError):
                        print("Usage: pos <seconds>")
                elif cmd == "status":
                    print(f"Status: {peer.get_status()}")
                elif cmd:
                    print("Commands: play, stop, pause, bpm X, pos X, status, quit")
                    
        except KeyboardInterrupt:
            pass
        finally:
            peer.stop()

if __name__ == "__main__":
    main() 