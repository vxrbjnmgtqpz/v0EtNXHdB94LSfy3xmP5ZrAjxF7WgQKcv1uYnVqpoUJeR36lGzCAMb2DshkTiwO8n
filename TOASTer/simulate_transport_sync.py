#!/usr/bin/env python3

"""
TOASTer Transport Sync Simulator

Simulates peer instances sending transport commands via UDP multicast
to test if your TOASTer app responds correctly to bi-directional sync.
"""

import socket
import struct
import json
import time
import threading
import sys
import random
from typing import Dict, Any
import argparse

# TOAST v2 Protocol Constants
TOAST_MAGIC = 0x54534F54  # "TOST"
TOAST_VERSION = 2

class TOASTFrameType:
    MIDI = 0x01
    AUDIO = 0x02  
    VIDEO = 0x03
    SYNC = 0x04
    TRANSPORT = 0x05
    DISCOVERY = 0x06
    HEARTBEAT = 0x07
    BURST_HEADER = 0x08

class TransportSimulator:
    def __init__(self, multicast_group="239.255.77.77", port=7777, session_name="TransportSyncTest"):
        self.multicast_group = multicast_group
        self.port = port
        self.session_name = session_name
        self.session_id = hash(session_name) & 0xFFFFFFFF
        self.sequence_number = 0
        self.running = False
        
        # Socket setup
        self.sock = None
        self.setup_socket()
        
        # State
        self.is_playing = False
        self.current_position = 0.0
        self.current_bpm = 120.0
        self.peer_count = 0
        
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
            
            print(f"✅ UDP multicast socket ready: {self.multicast_group}:{self.port}")
            
        except Exception as e:
            print(f"❌ Socket setup failed: {e}")
            sys.exit(1)
    
    def create_toast_frame(self, frame_type: int, payload: bytes) -> bytes:
        """Create a TOAST v2 frame with proper header"""
        timestamp_us = int(time.time() * 1000000) & 0xFFFFFFFF
        payload_size = len(payload)
        
        # Calculate simple checksum
        checksum = sum(payload) & 0xFFFF
        
        # Pack header (32 bytes)
        header = struct.pack('<I B B H I I I I B B H I',
            TOAST_MAGIC,        # magic (4 bytes)
            TOAST_VERSION,      # version (1 byte)  
            frame_type,         # frame_type (1 byte)
            0,                  # flags (2 bytes)
            self.sequence_number, # sequence_number (4 bytes)
            timestamp_us,       # timestamp_us (4 bytes)
            payload_size,       # payload_size (4 bytes)
            0,                  # burst_id (4 bytes)
            0,                  # burst_index (1 byte)
            1,                  # burst_total (1 byte)
            checksum,           # checksum (2 bytes)
            self.session_id     # session_id (4 bytes)
        )
        
        self.sequence_number += 1
        return header + payload
    
    def send_transport_command(self, command: str, position: float = None, bpm: float = None):
        """Send transport command to all peers"""
        if position is None:
            position = self.current_position
        if bpm is None:
            bpm = self.current_bpm
            
        timestamp = int(time.time() * 1000000)
        
        # Create JSON transport message (matches TOASTer format)
        transport_msg = {
            "type": "transport",
            "command": command,
            "timestamp": timestamp,
            "position": position,
            "bpm": bpm
        }
        
        payload = json.dumps(transport_msg).encode('utf-8')
        frame = self.create_toast_frame(TOASTFrameType.TRANSPORT, payload)
        
        try:
            self.sock.sendto(frame, (self.multicast_group, self.port))
            print(f"📡 Sent: {command} (pos: {position:.6f}, bpm: {bpm:.1f})")
            
            # Update local state
            if command == "PLAY":
                self.is_playing = True
            elif command == "STOP":
                self.is_playing = False
                self.current_position = 0.0
            elif command == "PAUSE":
                self.is_playing = False
            
            self.current_position = position
            self.current_bpm = bpm
            
        except Exception as e:
            print(f"❌ Send failed: {e}")
    
    def send_heartbeat(self):
        """Send heartbeat to maintain session"""
        heartbeat_msg = {
            "type": "heartbeat",
            "timestamp": int(time.time() * 1000000),
            "session": self.session_name,
            "peer_id": f"simulator_{random.randint(1000, 9999)}"
        }
        
        payload = json.dumps(heartbeat_msg).encode('utf-8')
        frame = self.create_toast_frame(TOASTFrameType.HEARTBEAT, payload)
        
        try:
            self.sock.sendto(frame, (self.multicast_group, self.port))
            print(f"💓 Heartbeat sent")
        except Exception as e:
            print(f"❌ Heartbeat failed: {e}")
    
    def listen_for_responses(self):
        """Listen for responses from TOASTer instances"""
        print("👂 Listening for TOASTer responses...")
        
        while self.running:
            try:
                self.sock.settimeout(1.0)
                data, addr = self.sock.recvfrom(4096)
                
                if len(data) < 32:  # Minimum header size
                    continue
                    
                # Parse header
                header = struct.unpack('<I B B H I I I I B B H I', data[:32])
                magic, version, frame_type, flags = header[:4]
                sequence_num, timestamp_us, payload_size = header[4:7]
                session_id = header[11]
                
                if magic != TOAST_MAGIC or version != TOAST_VERSION:
                    continue
                    
                if session_id != self.session_id:
                    continue
                    
                payload = data[32:32+payload_size]
                
                if frame_type == TOASTFrameType.TRANSPORT:
                    try:
                        msg = json.loads(payload.decode('utf-8'))
                        command = msg.get('command', 'UNKNOWN')
                        position = msg.get('position', 0.0)
                        bpm = msg.get('bpm', 120.0)
                        
                        print(f"🎛️ Received from TOASTer: {command} (pos: {position:.6f}, bpm: {bpm:.1f}) from {addr[0]}")
                        
                    except json.JSONDecodeError:
                        print(f"📝 Received non-JSON transport from {addr[0]}: {payload}")
                
                elif frame_type == TOASTFrameType.HEARTBEAT:
                    print(f"💓 Heartbeat from TOASTer at {addr[0]}")
                    
                elif frame_type == TOASTFrameType.DISCOVERY:
                    print(f"🔍 Discovery from TOASTer at {addr[0]}")
                    
                elif frame_type == TOASTFrameType.SYNC:
                    print(f"🎵 Sync message from TOASTer at {addr[0]}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"❌ Listen error: {e}")
    
    def run_interactive_test(self):
        """Run interactive transport sync test"""
        print("\n🎛️ TOASTer Transport Sync Interactive Test")
        print("==========================================")
        print(f"Session: {self.session_name}")
        print(f"Multicast: {self.multicast_group}:{self.port}")
        print("")
        print("Commands:")
        print("  play    - Send PLAY command")
        print("  stop    - Send STOP command") 
        print("  pause   - Send PAUSE command")
        print("  bpm X   - Change BPM to X")
        print("  pos X   - Set position to X seconds")
        print("  auto    - Auto test sequence")
        print("  quit    - Exit")
        print("")
        
        self.running = True
        listen_thread = threading.Thread(target=self.listen_for_responses, daemon=True)
        listen_thread.start()
        
        # Send initial heartbeat
        self.send_heartbeat()
        
        while self.running:
            try:
                cmd = input("Transport> ").strip().lower()
                
                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "play":
                    self.send_transport_command("PLAY")
                elif cmd == "stop":
                    self.send_transport_command("STOP")  
                elif cmd == "pause":
                    self.send_transport_command("PAUSE")
                elif cmd.startswith("bpm "):
                    try:
                        bpm = float(cmd.split()[1])
                        self.send_transport_command("BPM", bpm=bpm)
                    except (IndexError, ValueError):
                        print("Usage: bpm <number>")
                elif cmd.startswith("pos "):
                    try:
                        pos = float(cmd.split()[1])
                        self.send_transport_command("POSITION", position=pos)
                    except (IndexError, ValueError):
                        print("Usage: pos <seconds>")
                elif cmd == "auto":
                    self.run_auto_test()
                elif cmd == "heartbeat":
                    self.send_heartbeat()
                elif cmd == "help":
                    print("Commands: play, stop, pause, bpm X, pos X, auto, quit")
                elif cmd:
                    print("Unknown command. Type 'help' for commands.")
                    
            except KeyboardInterrupt:
                break
                
        self.running = False
        print("\n👋 Transport simulator stopped")
    
    def run_auto_test(self):
        """Run automated test sequence"""
        print("\n🤖 Running automated transport sync test...")
        
        test_sequence = [
            ("PLAY", 0.0, 120.0),
            ("STOP", 5.5, 120.0), 
            ("PLAY", 0.0, 128.0),
            ("PAUSE", 10.25, 128.0),
            ("PLAY", 10.25, 140.0),
            ("STOP", 0.0, 120.0)
        ]
        
        for i, (command, position, bpm) in enumerate(test_sequence):
            print(f"\n🧪 Test {i+1}/6: {command}")
            self.send_transport_command(command, position, bpm)
            time.sleep(2)  # Wait 2 seconds between commands
            
        print("\n✅ Automated test sequence complete")
    
    def cleanup(self):
        """Cleanup socket"""
        if self.sock:
            self.sock.close()

def main():
    parser = argparse.ArgumentParser(description="TOASTer Transport Sync Simulator")
    parser.add_argument("--group", default="239.255.77.77", help="Multicast group")
    parser.add_argument("--port", type=int, default=7777, help="UDP port")
    parser.add_argument("--session", default="TransportSyncTest", help="Session name")
    parser.add_argument("--auto", action="store_true", help="Run automated test")
    
    args = parser.parse_args()
    
    print("🎛️ TOASTer Transport Sync Simulator")
    print("===================================")
    print(f"📡 Multicast Group: {args.group}")
    print(f"🔢 Port: {args.port}")
    print(f"🎵 Session: {args.session}")
    print("")
    
    simulator = TransportSimulator(args.group, args.port, args.session)
    
    try:
        if args.auto:
            simulator.running = True
            listen_thread = threading.Thread(target=simulator.listen_for_responses, daemon=True)
            listen_thread.start()
            simulator.send_heartbeat()
            time.sleep(1)
            simulator.run_auto_test()
            time.sleep(2)
        else:
            simulator.run_interactive_test()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        simulator.cleanup()

if __name__ == "__main__":
    main() 