# Wi-Fi Network Discovery Configuration for TOASTer

## Quick Wi-Fi Testing Setup

### 1. **Same Wi-Fi Network Testing** (Easiest)
Both devices connected to the same router:
- Network: `192.168.1.x` (your current network)
- Port: `7777` (default TOASTer port)
- Scan range: `192.168.1.1-254`

### 2. **Wi-Fi Direct/Ad-Hoc Testing** (Point-to-Point)
Create a direct Wi-Fi network between devices:

**On macOS:**
1. Hold Option + Click Wi-Fi icon in menu bar
2. Select "Create Network..."
3. Name: `TOASTer-Test`
4. Channel: 6 or 11
5. Security: WPA2 Personal (optional)

**Both devices will auto-assign:**
- IP range: `169.254.x.x/16` (like Thunderbolt)
- Subnet: `255.255.0.0`
- No router needed

### 3. **Drop-in Replacement Instructions**

**Option A: Switch Network Mode in UI**
1. Launch TOASTer
2. In JAM Network Panel, change dropdown from "🔗 Thunderbolt Bridge" to "📶 Wi-Fi (Recommended)"
3. Click "🔍 Scan Wi-Fi" 
4. Wait for peer discovery
5. Select discovered device and click "🚀 Connect"

**Option B: Test with Simple UDP Tools**
```bash
# On Device 1 (Listener):
nc -lu 7777

# On Device 2 (Sender):
echo "TOAST_DISCOVERY" | nc -u 192.168.1.188 7777
```

### 4. **Expected Behavior**

**Wi-Fi Discovery Process:**
1. Auto-detects current network base (`192.168.1`)
2. Scans priority IPs first (1, 2, 100-105, etc.)
3. Attempts TCP connection to port 7777 on each IP
4. Displays found TOASTer instances in dropdown
5. Enables direct connection to selected peer

**Discovery Speed:**
- Priority IPs (30 addresses): ~3 seconds
- Full subnet scan (254 addresses): ~25 seconds
- Wi-Fi timeout: 0.5 seconds per IP (faster than Thunderbolt)

### 5. **Troubleshooting**

**No devices found:**
- Ensure both devices on same Wi-Fi network
- Check firewall settings (System Preferences → Security → Firewall)
- Verify TOASTer is listening on port 7777: `lsof -i :7777`
- Try manual IP entry in custom field

**Connection fails:**
- Check IP address with: `ifconfig en0 | grep inet`
- Ping test: `ping 192.168.1.XXX`
- Verify router allows inter-device communication
- Try different port (8888, 8889)

**Network detection wrong:**
- Wi-Fi discovery auto-detects `en0` interface
- For manual override, edit `getCurrentNetworkBase()` function
- Force specific subnet in `generateScanIPs()`

### 6. **Performance Expectations**

**Wi-Fi vs Thunderbolt:**
- ✅ **Latency:** ~2-5ms (vs 1-2ms TB)
- ✅ **Throughput:** 100+ Mbps (sufficient for MIDI/transport)
- ✅ **Reliability:** Very high on local network
- ✅ **Setup:** No cables, no drivers
- ⚠️ **Jitter:** Slightly higher than wired

**Optimal for:**
- MIDI synchronization
- Transport commands
- Device discovery
- Multi-room setups
- Quick testing/demos

### 7. **Code Integration**

The Wi-Fi discovery is now integrated into `JAMNetworkPanel` as a selectable mode:

```cpp
// Switch to Wi-Fi mode programmatically:
networkModeCombo.setSelectedId(1); // Wi-Fi mode
networkModeChanged(); // Apply change

// Get discovered devices:
auto devices = wifiDiscovery->getDiscoveredDevices();

// Connect to specific IP:
jamFramework->startNetworkDirect("192.168.1.100", 7777);
```

### 8. **Next Steps**

1. **Build and test** Wi-Fi discovery
2. **Compare results** with Thunderbolt approach
3. **Measure latency/jitter** on Wi-Fi vs Thunderbolt
4. **Confirm UDP/TCP** communication works over Wi-Fi
5. **Document findings** for Phase 4 decision

---

**This Wi-Fi solution should provide immediate testing capability without USB4/Thunderbolt dependencies and potentially better compatibility across different Mac models.**
