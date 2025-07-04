# Phase 2 Testing Instructions

## Two-Computer Network Testing for JMID Framework

### Prerequisites

- **Two computers on the same network** (both macOS)
- **MIDI keyboard/controller** (optional, for real MIDI testing)
- **Network connectivity** between computers

---

## 📦 **Step 1: Download the MIDILink App on Both Computers**

### Computer A (Master/Server)

1. **Copy the built app:**

   ```bash
   # From your development machine, copy the built app
   scp -r MIDILink/build/MIDILink.app [username]@[computer-a-ip]:~/Desktop/
   ```

2. **Or use a USB drive/shared folder:**
   - Copy `MIDILink/build/MIDILink.app` to USB drive
   - Transfer to Computer A's Desktop

### Computer B (Client/Slave)

1. **Copy the same app:**

   ```bash
   # Copy to second computer
   scp -r MIDILink/build/MIDILink.app [username]@[computer-b-ip]:~/Desktop/
   ```

2. **Or use the same USB drive/shared folder method**

### Verify Installation

**On Both Computers:**

```bash
# Make sure the app is executable
chmod +x ~/Desktop/MIDILink.app/Contents/MacOS/MIDILink

# Test launch (should open the app)
open ~/Desktop/MIDILink.app
```

---

## 🌐 **Step 2: Find Network IP Addresses**

### Computer A

```bash
# Find your IP address
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Write down Computer A's IP:** `___.___.___.___`

### Computer B

```bash
# Test connectivity to Computer A
ping [computer-a-ip]
```

**Verify:** Should see ping responses (Ctrl+C to stop)

---

## 🎯 **Step 3: Start the Master Session (Computer A)**

1. **Launch MIDILink:**

   ```bash
   open ~/Desktop/MIDILink.app
   ```

2. **In the Network Connection Panel:**

   - Leave IP field as: `127.0.0.1` (or enter Computer A's own IP)
   - Set Port: `8080`
   - Click **"Create Session"**
   - Session Name: `TestSession`
   - Click **"Start Server"**

3. **Verify Master Status:**
   - Status should show: `🟢 Server Active on port 8080`
   - Clock Sync Panel should show: `Role: Master`
   - Note: Server is now waiting for connections

---

## 🔗 **Step 4: Connect the Client (Computer B)**

1. **Launch MIDILink:**

   ```bash
   open ~/Desktop/MIDILink.app
   ```

2. **In the Network Connection Panel:**

   - Set IP: `[Computer A's IP address]`
   - Set Port: `8080`
   - Click **"Connect"**
   - Wait for connection (should show `🟡 Connected`)

3. **Join the Session:**

   - Click **"Join Session"**
   - Session Name: `TestSession`
   - Click **"Join"**

4. **Verify Client Status:**
   - Status should show: `🟡 Connected to [Computer A IP]`
   - Clock Sync Panel should show: `Role: Slave`
   - Session info should show: `Session: TestSession`

---

## ⏰ **Step 5: Verify Clock Synchronization**

### On Computer A (Master)

1. **Check Clock Sync Panel:**
   - Role: `Master`
   - Sync Quality: Should be `1.0` (perfect)
   - Connected Peers: Should show `1` (Computer B)

### On Computer B (Slave)

1. **Check Clock Sync Panel:**
   - Role: `Slave`
   - Master: Should show Computer A's info
   - Sync Quality: Should be `> 0.8` (good)
   - Network Latency: Should be `< 10ms` on local network

### Wait for Synchronization

- **Allow 10-15 seconds** for initial sync
- Both computers should show stable sync quality
- If sync quality is low, check network connection

---

## 🎵 **Step 6: Test MIDI Transmission**

### Setup MIDI (Optional)

**If you have a MIDI keyboard:**

1. **Connect MIDI keyboard to Computer A**
2. **In MIDI Testing Panel on Computer A:**
   - Select your MIDI input device
   - Select MIDI output (optional)

### Send Test Notes

**On Computer A:**

1. **Go to MIDI Testing Panel**
2. **Set:**
   - Channel: `1`
   - Note: `60` (Middle C)
   - Velocity: `100`
3. **Click "Send Test Note"**

### Verify Reception

**On Computer B:**

1. **Check MIDI Testing Panel log**
2. **Should see:** `🎵 Received Note On: Ch=1, Note=60, Vel=100`
3. **Should also see:** `🎵 Received Note Off: Ch=1, Note=60`

### Test Real-Time MIDI (If keyboard connected)

**On Computer A:**

1. **Play notes on MIDI keyboard**
2. **Computer B should show** all notes in real-time in the log
3. **Check latency:** Notes should appear on Computer B within 50ms

---

## 📊 **Step 7: Performance Verification**

### Check Network Statistics

**On Both Computers - Performance Monitor Panel:**

1. **Messages Sent/Received:** Should be increasing
2. **Network Latency:** Should be `< 20ms`
3. **Clock Drift:** Should be `< 1ms`
4. **Dropped Messages:** Should be `0`

### Stress Test

**On Computer A:**

1. **Rapidly click "Send Test Note"** 20+ times
2. **All notes should appear on Computer B**
3. **No "dropped messages" errors**

---

## 🔧 **Step 8: Advanced Testing**

### Test Network Interruption

1. **Disconnect Computer B's WiFi for 10 seconds**
2. **Reconnect WiFi**
3. **Computer B should automatically reconnect**
4. **Clock sync should restore within 30 seconds**

### Test Master Failover

1. **Close MIDILink on Computer A (Master)**
2. **Computer B should show "Connection Lost"**
3. **Restart MIDILink on Computer A**
4. **Computer B should reconnect automatically**

---

## ❌ **Troubleshooting**

### "Connection Failed"

```bash
# On Computer A - Check if server started
netstat -an | grep 8080
# Should show: *.8080 LISTEN

# On Computer B - Test basic connection
telnet [computer-a-ip] 8080
# Should connect (Ctrl+] then 'quit' to exit)
```

### "Poor Sync Quality"

- Check WiFi signal strength on both computers
- Ensure no heavy network usage (downloads, streaming)
- Try moving computers closer to router

### "No MIDI Received"

- Verify MIDI devices aren't used by other apps (GarageBand, etc.)
- Check MIDI permissions: **System Preferences → Security & Privacy → Microphone**
- Try the "Send Test Note" button first before real MIDI

### Firewall Issues

```bash
# On Computer A - Allow port 8080
sudo pfctl -f /etc/pf.conf
# Or temporarily disable firewall for testing
```

---

## ✅ **Success Criteria Checklist**

### Basic Connection

- [ ] Computer A shows "Server Active"
- [ ] Computer B shows "Connected"
- [ ] Both show joined session

### Clock Synchronization

- [ ] Computer A: Role = Master
- [ ] Computer B: Role = Slave
- [ ] Sync Quality > 0.8 on both
- [ ] Network latency < 20ms

### MIDI Transmission

- [ ] Test notes send from A to B
- [ ] Real MIDI (if available) transmits in real-time
- [ ] No dropped messages during normal use
- [ ] Latency feels responsive (< 50ms total)

### Stability

- [ ] Connection stable for 5+ minutes
- [ ] Survives network interruption
- [ ] Master failover works

---

## 🎉 **Phase 2 Complete!**

**If all checkboxes are ✅, you've successfully validated:**

- TOAST network transport protocol
- ClockDriftArbiter synchronization
- Real-time MIDI over JSON transmission
- MIDILink application integration

**🚀 Ready for Phase 2.3: Distributed Synchronization Engine!**
