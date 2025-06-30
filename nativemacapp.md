# Native Mac App for MIDIp2p Network Testing
## TOAST Network Testing Application - "MIDILink"

*VST3-Bridge Compatible Native macOS Application for LAN-based MIDI Streaming Tests*

---

## Project Overview

**MIDILink** is a native macOS application designed to test and validate the MIDIp2p JSONMIDI framework over TOAST (Transport Oriented Audio Synchronization Tunnel) network connections. This app will serve as the primary testing platform for Phase 2 and Phase 3 of the MIDIp2p roadmap, enabling real-world validation of distributed MIDI synchronization between multiple Mac devices.

### Key Requirements
- **Native macOS Application** (replacing Electron-based approach)
- **VST3 Bridge Integration** for cross-DAW compatibility
- **TOAST Network Protocol** implementation
- **ClockDriftArbiter** testing and validation
- **Dual Installation** support (MacBook + Mac Mini testing)
- **LAN TCP Connection** management

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ TRANSPORT BAR ▶️⏹⏺ SESSION TIME / BARS and beats               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   JUCE GUI      │  │   VST3 Bridge    │  │  MIDI I/O Mgr   │ │
│  │  (Native Mac)   │  │  (Cross-DAW)     │  │ (Core Audio)    │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ JSONMIDI Engine │  │  Bassoon.js Core │  │ Performance Mon │ │
│  │  (Protocol)     │  │  (Ultra-Low Lat) │  │  (Latency)      │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ TOAST Network   │  │ ClockDriftArbiter│  │ Connection Mgr  │ │
│  │  (TCP Tunnel)   │  │  (Time Sync)     │  │ (Discovery)     │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Leveraging Existing JUCE Components

### Current Assets Analysis

From the existing `JUCE_JSON_Messenger` project:

#### ✅ **PluginEditor.cpp Components**
```cpp
// Reusable UI Elements
- juce::TextEditor inputBox        // For JSON message input
- juce::TextButton sendButton      // For triggering actions
- JSON message creation logic      // Using juce::DynamicObject
- File-based message passing       // Via juce::File API
```

#### ✅ **Web Interface Pattern**
```html
// Polling-based message monitoring
- Real-time message display
- JSON data formatting
- Timestamp visualization
```

### Enhanced Components for MIDILink

#### 1. **Core JUCE Framework Extensions**
```cpp
class MIDILinkApplication : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override { return "MIDILink"; }
    const juce::String getApplicationVersion() override { return "1.0.0"; }
    
    void initialise(const juce::String&) override;
    void shutdown() override;
    
private:
    std::unique_ptr<MainWindow> mainWindow;
    std::unique_ptr<TOASTNetworkManager> networkManager;
    std::unique_ptr<MIDIDeviceManager> midiManager;
};
```

#### 2. **Enhanced UI Components** (Building on existing patterns)
```cpp
class MIDILinkMainWindow : public juce::DocumentWindow
{
public:
    MIDILinkMainWindow();
    
private:
    // Enhanced versions of existing components
    std::unique_ptr<NetworkConnectionPanel> networkPanel;
    std::unique_ptr<MIDITestingPanel> midiPanel;
    std::unique_ptr<PerformanceMonitorPanel> perfPanel;
    std::unique_ptr<ClockSyncPanel> clockPanel;
};

class NetworkConnectionPanel : public juce::Component
{
private:
    // Building on existing TextEditor pattern
    juce::TextEditor ipAddressField;     // Target IP input
    juce::TextButton connectButton;      // Enhanced send button
    juce::TextButton listenButton;       // Server mode toggle
    juce::Label connectionStatus;        // Status display
    
    // New TOAST-specific components
    juce::Slider latencySlider;          // Network simulation
    juce::ToggleButton masterModeToggle; // Clock master/slave
};
```

#### 3. **VST3 Bridge Integration**
```cpp
class VST3ClockBridge : public juce::AudioProcessor
{
public:
    // Clock synchronization via VST3 host
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    
    // Transport sync callbacks
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { true; }
    
private:
    std::unique_ptr<ClockDriftArbiter> clockArbiter;
    std::unique_ptr<TOASTConnection> networkConnection;
};
```

---

## Application Features & Testing Capabilities

### Core Testing Features

#### 1. **Network Connection Management**
- **Auto-Discovery**: Scan LAN for other MIDILink instances
- **Manual Connection**: Direct IP address connection
- **Connection Quality**: Real-time latency and stability monitoring
- **Failover Testing**: Simulate network interruptions

#### 2. **MIDI Testing Suite**
```cpp
class MIDITestSuite
{
public:
    // Based on existing JSON message creation
    void sendNoteOnTest();               // Single note timing test
    void sendControllerSweep();          // CC value streaming test
    void sendTimingStressTest();         // High-frequency event test
    void sendChannelMultiplexTest();     // Multi-channel simultaneous test
    
private:
    // Enhanced version of existing DynamicObject approach
    juce::var createMIDIMessage(const juce::String& type, 
                               int channel, int note, int velocity);
};
```

#### 3. **Clock Synchronization Validation**
- **Master/Slave Election**: Test automatic role assignment
- **Drift Measurement**: Real-time clock deviation tracking
- **Sync Recovery**: Test recovery from timing disruptions
- **VST3 Host Sync**: Validate DAW transport integration

#### 4. **Performance Monitoring Dashboard**
```cpp
class PerformanceMonitor : public juce::Component,
                          public juce::Timer
{
public:
    void timerCallback() override;       // Real-time updates
    
private:
    // Metrics display (building on existing HTML pattern)
    juce::Label latencyDisplay;         // Current latency
    juce::Label throughputDisplay;      // Messages/second
    juce::Label clockDriftDisplay;      // Timing deviation
    juce::Label packetLossDisplay;      // Network reliability
    
    // Performance graphs
    std::unique_ptr<LatencyGraph> latencyGraph;
    std::unique_ptr<ThroughputGraph> throughputGraph;
};
```

---

## Implementation Plan

### Phase 1: Foundation Development (Week 1-2)
**Building on Existing JUCE Components**

#### 1.1 **Project Setup**
- [ ] Create new JUCE macOS application project
- [ ] Import and refactor existing PluginEditor components
- [ ] Set up Xcode project with proper entitlements
- [ ] Configure Core Audio and Network framework access

#### 1.2 **Basic UI Framework**
```cpp
// Enhanced version of existing UI pattern
class MainComponent : public juce::Component
{
public:
    MainComponent()
    {
        // Upgrade existing simple UI
        addAndMakeVisible(networkPanel);
        addAndMakeVisible(midiTestPanel);
        addAndMakeVisible(performancePanel);
        
        setSize(800, 600);  // Expanded from 400x150
    }
    
private:
    NetworkConnectionPanel networkPanel;
    MIDITestingPanel midiTestPanel;
    PerformanceMonitorPanel performancePanel;
};
```

#### 1.3 **MIDI Device Integration**
- [ ] Implement Core MIDI device enumeration
- [ ] Create MIDI input/output routing
- [ ] Build on existing JSON message creation pattern
- [ ] Add MIDI message validation and formatting

### Phase 2: Network Layer Implementation (Week 3-4)
**TOAST Protocol Integration**

#### 2.1 **ClockDriftArbiter Integration**
```cpp
// Header implementation (ClockDriftArbiter.h)
class ClockDriftArbiter
{
public:
    // Network timing measurement
    struct NetworkTiming {
        std::chrono::nanoseconds roundTripTime;
        std::chrono::nanoseconds clockOffset;
        double driftRate;
        uint64_t lastSyncTimestamp;
    };
    
    // Master election (building on existing button logic)
    bool electAsMaster();
    void becomeSlave(const juce::String& masterIP);
    
    // Synchronization (enhanced JSON messaging)
    void sendSyncPulse();
    void receiveSyncPulse(const juce::var& syncMessage);
    
private:
    NetworkTiming currentTiming;
    juce::CriticalSection timingLock;
};
```

#### 2.2 **TOAST TCP Implementation**
- [ ] Implement TCP connection management
- [ ] Create JSONMIDI message framing
- [ ] Build on existing file-based messaging pattern
- [ ] Add network message queuing and buffering

#### 2.3 **Enhanced JSON Protocol**
```cpp
// Enhanced version of existing JSON creation
juce::var createTOASTMessage(const juce::String& messageType,
                           const juce::var& midiPayload)
{
    juce::DynamicObject::Ptr toast = new juce::DynamicObject();
    
    // TOAST envelope (extending existing timestamp approach)
    toast->setProperty("masterTime", getCurrentMasterTime());
    toast->setProperty("networkSeq", getNextSequenceNumber());
    toast->setProperty("syncQuality", getCurrentSyncQuality());
    toast->setProperty("role", isMaster() ? "master" : "slave");
    
    // MIDI payload (building on existing JSON structure)
    toast->setProperty("midi", midiPayload);
    
    return juce::var(toast.get());
}
```

### Phase 3: VST3 Bridge Development (Week 5-6)
**Cross-DAW Compatibility**

#### 3.1 **VST3 Plugin Component**
- [ ] Create VST3 wrapper for clock synchronization
- [ ] Implement transport state bridging
- [ ] Add parameter automation for network controls
- [ ] Create host communication interface

#### 3.2 **DAW Integration Testing**
- [ ] Test with Logic Pro X transport
- [ ] Validate with Ableton Live sync
- [ ] Verify Pro Tools compatibility
- [ ] Test with Reaper synchronization

### Phase 4: Testing & Optimization (Week 7-8)
**LAN Testing Between Devices**

#### 4.1 **Dual-Device Test Suite**
- [ ] MacBook ↔ Mac Mini connection tests
- [ ] Latency measurement validation
- [ ] Clock drift compensation verification
- [ ] Network interruption recovery testing

#### 4.2 **Performance Optimization**
- [ ] Profile critical path performance
- [ ] Optimize MIDI message processing
- [ ] Enhance network buffer management
- [ ] Validate <15ms end-to-end latency target

---

## Technical Specifications

### System Requirements
- **macOS**: 10.15+ (Catalina or later)
- **Architecture**: Universal Binary (Intel + Apple Silicon)
- **Frameworks**: 
  - JUCE 7.0+
  - Core Audio
  - Core MIDI
  - Network Framework
  - VST3 SDK

### Network Configuration
- **Protocol**: TCP over LAN
- **Port Range**: 7000-7010 (configurable)
- **Discovery**: Bonjour/mDNS service discovery
- **Security**: Local network trust model

### Performance Targets
- **UI Responsiveness**: 60fps native refresh
- **MIDI Latency**: <100μs local processing
- **Network Latency**: <15ms end-to-end over LAN
- **Clock Accuracy**: <1ms synchronization deviation
- **Throughput**: 1000+ MIDI events/second sustained

---

## Testing Scenarios

### Scenario 1: Basic Connection Test
1. Launch MIDILink on MacBook (Master mode)
2. Launch MIDILink on Mac Mini (Auto-discover and connect)
3. Verify clock synchronization establishment
4. Send test MIDI note and measure round-trip latency

### Scenario 2: VST3 DAW Integration
1. Load VST3 bridge in Logic Pro on MacBook
2. Connect to MIDILink instance on Mac Mini
3. Play MIDI from Logic Pro transport
4. Verify synchronized playback on Mac Mini

### Scenario 3: Network Stress Testing
1. Establish connection between devices
2. Send high-frequency MIDI data (500+ events/second)
3. Simulate network interruptions
4. Verify graceful recovery and data integrity

### Scenario 4: Clock Drift Validation
1. Run 30-minute continuous session
2. Monitor clock drift accumulation
3. Verify drift compensation effectiveness
4. Measure long-term synchronization stability

---

## Deployment Strategy

### Development Distribution
- **Development Builds**: Xcode direct install
- **Internal Testing**: Ad-hoc distribution profiles
- **Beta Testing**: TestFlight distribution

### Production Packaging
- **App Store**: Mac App Store distribution
- **Direct Distribution**: Notarized .dmg packages
- **Enterprise**: Custom installation packages

### Installation Requirements
- **Permissions**: Network access, MIDI device access
- **Entitlements**: Audio device access, network server
- **Certificates**: Developer ID application signing

---

## Integration with MIDIp2p Roadmap

### Roadmap Milestone Support

#### **Milestone 1: Local MIDI JSON Processing (Week 4)**
- ✅ **MIDILink provides**: Real-time MIDI → JSON → MIDI validation
- ✅ **Testing capability**: Latency measurement tools
- ✅ **Verification method**: Built-in oscilloscope-style timing display

#### **Milestone 2: Network MIDI Streaming (Week 8)**
- ✅ **MIDILink provides**: Dual-device LAN testing platform
- ✅ **Testing capability**: Piano → remote synthesizer simulation
- ✅ **Verification method**: User-perceivable timing validation tools

#### **Milestone 3: Multi-Client Synchronization (Week 12)**
- ✅ **MIDILink provides**: Foundation for multi-device testing
- ✅ **Testing capability**: Distributed ensemble simulation
- ✅ **Verification method**: Temporal analysis and recording tools

### Development Timeline Alignment
- **MIDILink Development**: Weeks 1-8 (parallel to MIDIp2p Phase 1-2)
- **Integration Testing**: Weeks 9-12 (supporting MIDIp2p Phase 3)
- **Production Readiness**: Weeks 13-16 (enabling MIDIp2p Phase 4)

---

## Success Criteria

### Technical Validation
- [ ] **Sub-15ms Latency**: Verified LAN connection performance
- [ ] **Clock Synchronization**: <1ms deviation between devices
- [ ] **Data Integrity**: Zero MIDI message corruption
- [ ] **Network Resilience**: Graceful recovery from interruptions
- [ ] **VST3 Compatibility**: Successful DAW integration

### User Experience Validation
- [ ] **Intuitive Setup**: <5 minute connection establishment
- [ ] **Stable Operation**: 8+ hour continuous session capability
- [ ] **Performance Monitoring**: Real-time latency and quality feedback
- [ ] **Cross-Device Compatibility**: MacBook ↔ Mac Mini seamless operation

### Development Impact
- [ ] **MIDIp2p Validation**: Proves core framework viability
- [ ] **TOAST Protocol**: Validates network layer design
- [ ] **ClockDriftArbiter**: Confirms synchronization approach
- [ ] **Production Readiness**: Demonstrates real-world applicability

---

*MIDILink serves as the critical validation platform for the MIDIp2p framework, providing hands-on testing capabilities that will prove the viability of JSONMIDI over TOAST for professional audio applications. Success with this native Mac app will demonstrate the readiness for broader JamNet deployment.*