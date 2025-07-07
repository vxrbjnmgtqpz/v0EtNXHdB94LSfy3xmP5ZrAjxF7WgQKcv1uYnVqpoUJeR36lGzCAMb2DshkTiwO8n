# 🎛️ TOASTer Transport Sync Test Instructions

## Test Setup (Manual - requires multiple instances)

### Step 1: Launch Multiple Instances

1. **Instance A (Master):**

   - Launch: `open "TOASTer/dist/TOASTer.app/Contents/MacOS/TOASTer"`
   - Go to JAM Network Panel
   - Set Session Name: "TransportSyncTest"
   - Set Multicast: "239.255.77.77:7777"
   - Click "Connect"

2. **Instance B (Peer):**

   - Launch: `open "TOASTer/dist/TOASTer.app/Contents/MacOS/TOASTer"` (new instance)
   - Go to JAM Network Panel
   - Set Session Name: "TransportSyncTest"
   - Set Multicast: "239.255.77.77:7777"
   - Click "Connect"

3. **Instance C (Optional):**
   - Repeat for third instance to test multi-peer sync

### Step 2: Verify Connection

- Check "Active Peers" count shows connected instances
- Look for "Network Status: Connected" in each instance
- Verify UDP status shows multicast group joined

### Step 3: Test Bi-Directional Transport Sync

#### Test 3A: Stop Propagation

1. ▶️ Press PLAY in Instance A
2. ⏹️ Press STOP in Instance B
3. **Expected:** All instances should STOP simultaneously
4. **Verify:** Transport buttons show stopped state on all instances

#### Test 3B: Play Propagation

1. ▶️ Press PLAY in Instance C (or any instance)
2. **Expected:** All instances should START playing simultaneously
3. **Verify:** Transport buttons show playing state on all instances
4. **Verify:** Position counters move in sync

#### Test 3C: Position Sync

1. ⏸️ Press PAUSE in any instance
2. 🎚️ Change position in any instance
3. ▶️ Press PLAY in different instance
4. **Expected:** All instances start from same position

#### Test 3D: BPM Sync

1. 🎵 Change BPM in Instance A
2. **Expected:** BPM updates on all instances
3. **Verify:** Tempo displays match across instances

### Step 4: Advanced Tests

#### Test 4A: Network Resilience

1. Disconnect one instance (close app)
2. Test transport commands still work between remaining instances
3. Reconnect instance - should sync to current state

#### Test 4B: Rapid Commands

1. Rapidly press PLAY/STOP in different instances
2. **Expected:** All instances follow the latest command
3. **Verify:** No stuck or inconsistent states

## Expected Transport Message Format

When working correctly, the console should show messages like:

```
📡 Sent transport command: PLAY (pos: 0.000000, bpm: 120.0)
🎛️ Received transport command: STOP (pos: 15.234567, bpm: 120.0)
🎵 Received transport sync: {"type":"transport","command":"PLAY","timestamp":1234567890,"position":0.0,"bpm":120.0}
```

## Success Criteria ✅

- [ ] Multiple instances connect to same session
- [ ] STOP in one instance stops all others immediately
- [ ] PLAY in any instance starts all others in unison
- [ ] Position and BPM changes propagate to all peers
- [ ] No master/slave - any instance can control transport
- [ ] Sub-100ms synchronization latency
- [ ] Reliable operation with network disconnections

## Troubleshooting

**No connection:**

- Check firewall allows UDP multicast
- Verify same session name and multicast address
- Try different port if 7777 is busy

**Transport not syncing:**

- Check console logs for TOAST transport messages
- Verify "Active Peers" count > 0
- Restart instances and reconnect

**Timing issues:**

- Check GPU transport is enabled
- Verify network latency is reasonable (<50ms)
- Look for GPU timebase initialization messages
