# TOASTer Testing Guide

**TOASTer is now ready for comprehensive testing!** 🎉

## 🚀 Current Status

### ✅ Built and Ready Components

1. **TOASTer Application**: Native macOS app with full JAMNet GUI
   - Location: `/TOASTer/build/TOASTer_artefacts/Release/TOASTer.app`
   - Status: ✅ Built successfully and running
   - Features: Network connection, MIDI testing, performance monitoring, clock sync

2. **TCP TOAST Protocol**: Complete server/client implementation
   - `toast_server`: Multi-client TCP server with session management
   - `toast_client`: Interactive TCP client with command interface
   - `test_tcp_communication`: Automated protocol validation tests
   - Status: ✅ All binaries built and tested

3. **JMID Framework**: Core protocol implementation
   - Message parsing, validation, and serialization
   - Performance profiling and benchmarking
   - Status: ✅ Complete with examples and tests

## 🧪 Testing Scenarios

### Scenario 1: Local TCP TOAST Protocol Testing

Test the core TCP TOAST protocol between two terminal sessions:

```bash
# Terminal 1 - Start the server
cd /Users/timothydowler/Projects/MIDIp2p/JMID_Framework
./toast_server

# Terminal 2 - Connect a client
./toast_client

# Test commands in client:
# - send_midi
# - join_session
# - send_heartbeat
# - quit
```

**Expected Results**:
- Server accepts client connections
- JMID messages transmitted successfully
- Session management works
- Heartbeat mechanism functional

### Scenario 2: TOASTer GUI Application Testing

Test the native macOS application interface:

```bash
# Launch TOASTer
cd /Users/timothydowler/Projects/MIDIp2p/TOASTer/build/TOASTer_artefacts/Release
open TOASTer.app
```

**Test Features**:
1. **Network Connection Panel**: Test connection setup and discovery
2. **MIDI Testing Panel**: Test MIDI I/O and message handling
3. **Performance Monitor**: Check latency measurements and throughput
4. **Clock Sync Panel**: Test clock synchronization features
5. **JMID Integration**: Test JSON message processing

### Scenario 3: Multi-Client Session Testing

Test distributed sessions with multiple clients:

```bash
# Terminal 1 - Server
./toast_server

# Terminal 2 - Client 1
./toast_client
# Commands: join_session test_session

# Terminal 3 - Client 2  
./toast_client
# Commands: join_session test_session

# Terminal 4 - TOASTer App
open TOASTer.app
# Connect to same session through GUI
```

**Expected Results**:
- Multiple clients can join the same session
- MIDI messages route correctly between clients
- Session state maintained across connections
- TOASTer GUI integrates with command-line clients

### Scenario 4: Performance Benchmarking

Test latency and throughput characteristics:

```bash
# Run automated performance tests
cd /Users/timothydowler/Projects/MIDIp2p/JMID_Framework
./test_tcp_communication

# Check specific benchmarks
cd build && make test
```

**Target Metrics**:
- **Message Latency**: < 5ms over LAN
- **Throughput**: 1000+ messages/second
- **Connection Setup**: < 100ms
- **Session Management**: < 50ms overhead

### Scenario 5: Cross-Machine LAN Testing

Test between two separate machines (MacBook Pro ↔ Mac Mini):

**Setup Requirements**:
1. Both machines on same LAN
2. TOASTer.app installed on both machines
3. Firewall configured to allow TCP connections

**Test Procedure**:
```bash
# Machine 1 (Server) - Find IP address
ifconfig | grep "inet "

# Machine 1 - Start server
./toast_server --port 8080

# Machine 2 - Connect client  
./toast_client --host <Machine1_IP> --port 8080

# Machine 2 - Launch TOASTer GUI
open TOASTer.app
# Connect to Machine1_IP:8080
```

## 🎯 Test Success Criteria

### Phase 1: Local Testing ✅
- [x] TOASTer app launches without crashes
- [x] TCP server/client communication works
- [x] JMID messages parse correctly
- [x] GUI panels display and respond

### Phase 2: Network Testing (Ready to Test)
- [ ] Multi-client sessions work reliably  
- [ ] Cross-machine LAN communication successful
- [ ] Latency meets <5ms target
- [ ] Session persistence and recovery

### Phase 3: Real-World Usage (Pending)
- [ ] MIDI hardware integration
- [ ] DAW bridge functionality  
- [ ] Extended multi-hour sessions
- [ ] Load testing with 16+ clients

## 🔧 Troubleshooting

### Common Issues

1. **TOASTer won't launch**:
   ```bash
   # Check for missing dependencies
   otool -L TOASTer.app/Contents/MacOS/TOASTer
   ```

2. **TCP connection fails**:
   ```bash
   # Check port availability
   lsof -i :8080
   
   # Test basic connectivity
   telnet localhost 8080
   ```

3. **Performance issues**:
   ```bash
   # Monitor CPU/memory usage
   top -pid $(pgrep TOASTer)
   
   # Check network traffic
   netstat -i
   ```

## 📊 Testing Progress

### Current Status: **Phase 2 Ready** 🟡

| Component | Status | Notes |
|-----------|--------|-------|
| **TOASTer App** | ✅ Ready | Built, launches, GUI functional |
| **TCP Protocol** | ✅ Ready | Server/client tested, working |
| **JMID Core** | ✅ Ready | Parsing/validation complete |
| **Multi-Client** | 🟡 Ready to Test | Infrastructure complete |
| **LAN Testing** | 🟡 Ready to Test | Need two machines |
| **Performance** | 🟡 Ready to Test | Benchmarks available |
| **Hardware MIDI** | 🔴 Pending | Needs Phase 3 |

### Next Steps

1. **Immediate Testing** (Today):
   - Run multi-client scenarios locally
   - Validate session management
   - Test TOASTer GUI integration
   - Measure baseline performance

2. **LAN Testing** (This Week):
   - Set up MacBook Pro ↔ Mac Mini testing
   - Validate cross-machine communication
   - Measure real-world latency
   - Test network interruption recovery

3. **Integration Testing** (Next Phase):
   - MIDI hardware integration
   - DAW bridge development
   - Extended stability testing
   - Performance optimization

## 🎉 Conclusion

**TOASTer is ready for comprehensive testing!** All core components are built, functional, and tested at the basic level. The framework provides:

- ✅ **Complete TCP TOAST implementation**
- ✅ **Native macOS application** 
- ✅ **JMID protocol stack**
- ✅ **Multi-client session management**
- ✅ **Performance monitoring tools**

The next phase involves validating these components under real-world network conditions and beginning the transition to UDP + PNTBTR for ultra-low latency performance.

**Ready to jam! 🎵**
