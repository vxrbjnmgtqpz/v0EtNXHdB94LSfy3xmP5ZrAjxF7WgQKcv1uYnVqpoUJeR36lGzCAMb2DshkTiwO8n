# Computer-to-Computer TOAST Network Setup Guide 🌐

## Current Network Configuration
**Server Computer (this machine):** `192.168.1.188`
- **TCP Server:** Port 8080 (reliable connections)
- **UDP Server:** Port 8081 (low-latency connections)

## 🖥️ Setting Up Server Computer

1. **Start the TOAST servers:**
   ```bash
   cd /Users/timothydowler/Projects/MIDIp2p/JMID_Framework
   ./toast_server           # TCP on port 8080
   ```
   
   ```bash
   cd /Users/timothydowler/Projects/MIDIp2p/TOASTer
   ./build/jmid_framework/toast_udp_server 8081  # UDP on port 8081
   ```

2. **Open TOASTer app** (optional - server computer can also be a client):
   ```bash
   cd /Users/timothydowler/Projects/MIDIp2p/TOASTer
   open build/TOASTer_artefacts/Release/TOASTer.app
   ```

## 💻 Setting Up Client Computer(s)

1. **Copy TOASTer.app** to the client computer
2. **Launch TOASTer.app**
3. **Configure Network Connection:**
   - **Protocol:** Choose TCP or UDP
   - **IP Address:** Enter `192.168.1.188` (server computer's IP)
   - **Port:** 
     - `8080` for TCP connections
     - `8081` for UDP connections
   - **Session Name:** Choose any name (e.g., "JamSession")

4. **Connect:** Click "Connect" button
5. **Join/Create Session:** Click "Create Session" or "Join Session"

## 🔧 Network Requirements

### Firewall Settings
Make sure the server computer allows incoming connections on:
- **Port 8080** (TCP)
- **Port 8081** (UDP)

### Network Connectivity
- All computers must be on the same network (Wi-Fi/Ethernet)
- Or use port forwarding for internet connections

## 🎵 Testing the Connection

### TCP Connection Test
1. Set protocol to **TCP**
2. Enter IP: `192.168.1.188`
3. Port: `8080`
4. Click **Connect**
5. Expected: "TCP Connected to 192.168.1.188:8080"

### UDP Connection Test  
1. Set protocol to **UDP**
2. Enter IP: `192.168.1.188`
3. Port: `8081`
4. Click **Connect**
5. Expected: "UDP connection ready (connection-less protocol)"

## 🚀 Multi-Computer Jamming

Once connected:
1. **Create Session** on one computer (becomes session host)
2. **Join Session** from other computers using same session name
3. **Start Transport** - all computers sync to microsecond precision
4. **Send MIDI** - notes and timing sync across all connected devices

## 📊 Performance Monitoring

The TOASTer app displays:
- **Framework Latency** (network round-trip time)
- **Clock Accuracy** (sync precision between devices)
- **Connection Status** (active clients, messages sent/received)
- **Protocol Type** (TCP reliable vs UDP low-latency)

## 🔍 Troubleshooting

### Can't Connect
- ✅ Check server computer IP: `ifconfig | grep "inet "`
- ✅ Verify servers are running on server computer
- ✅ Check firewall allows ports 8080, 8081
- ✅ Ensure all computers on same network

### High Latency
- 🔄 Try UDP instead of TCP for lower latency
- 📡 Check Wi-Fi signal strength
- 🌐 Use wired Ethernet for best performance

### Sync Issues
- ⏱️ Enable "Force Master" on one computer only
- 🔧 Use "Calibrate" to adjust clock sync
- 📊 Monitor "Sync Quality" percentage

---

## ✅ Ready for Testing!

**Server Status:** 
- TCP Server: ✅ Running on 192.168.1.188:8080
- UDP Server: ✅ Running on 192.168.1.188:8081  
- TOASTer App: ✅ Updated with network configuration

**Next:** Connect from other computers using the IP `192.168.1.188` and start jamming! 🎵
