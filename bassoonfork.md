# Bassoon.js Integration Roadmap for JAMNet Real-Time Network Streaming

Here's a focused integration roadmap to upgrade your existing bassoon.js system into a multicast .jsonl streamer—while keeping your current architecture intact.

🛠️ **Objective**
Integrate .jsonl streaming + multicast dispatch into your existing bassoon.js stack with minimal disruption and full backward compatibility.

## 📋 **Current System Analysis**

Based on your codebase, you have:

- ✅ **JSONMIDI Framework** with BassoonParser (C++ SIMD-optimized)
- ✅ **TOAST Transport Layer** (TCP/UDP) for real-time streaming
- ✅ **TOASTer JUCE Application** with network panels and MIDI testing
- ✅ **ClockDriftArbiter** for microsecond-precision timing
- ✅ **Lock-free queues** for real-time message handling
- ✅ **Multi-client network architecture** for JAMNet/MIDIp2p

## 🎯 **Adaptation for Your MIDIp2p Architecture**

### **Stage 0: Architecture Assessment & Prep**

✅ **Current Integration Points**

```cpp
// Your existing BassoonParser usage in TOASTer:
parser = std::make_unique<JSONMIDI::BassoonParser>();
auto message = parser->parseMessageWithValidation(json);
```

✅ **Network Flow Analysis**

```
MIDI Hardware → JSONMIDI → BassoonParser → TOAST Transport → Network
     ↓              ↓           ↓              ↓            ↓
 JUCE MIDI    JSON Schema   C++ Parser    TCP/UDP      Remote TOASTer
```

✅ **Performance Requirements**

- Current target: <50μs parse time (your Phase 1.2 goal)
- Network latency: <10ms via TOAST
- Multi-client: 16+ concurrent connections
- Real-time safety: Lock-free queues

---

### **Stage 1: Enhance BassoonParser for JSONL Streaming**

🔹 **Goal**: Add .jsonl support to your existing BassoonParser without breaking TOAST integration

**1.1: Extend BassoonParser for Streaming**

```cpp
// In JSONMIDI_Framework/include/JSONMIDIParser.h
class BassoonParser {
public:
    // Add JSONL streaming capabilities
    enum class ParseMode {
        SINGLE_JSON,    // Current mode
        JSONL_STREAM,   // New streaming mode
        COMPACT_JSONL   // Ultra-compact for <50μs target
    };

    void setParseMode(ParseMode mode);
    void enableMulticast(bool enabled);

    // Stream processing
    void feedJsonlLine(const std::string& line);
    bool hasStreamedMessage() const;
    std::unique_ptr<MIDIMessage> extractStreamedMessage();

    // Compact JSONL format support
    std::string compactifyMessage(const MIDIMessage& msg);
    std::unique_ptr<MIDIMessage> parseCompactJsonl(const std::string& line);
};
```

**1.2: Implement Compact JSONL Format**

```cpp
// Ultra-compact format for <50μs parsing
std::string BassoonParser::compactifyMessage(const MIDIMessage& msg) {
    switch(msg.getType()) {
        case MIDIMessageType::NOTE_ON:
            return fmt::format("{{\"t\":\"n+\",\"n\":{},\"v\":{},\"c\":{},\"ts\":{}}}",
                             msg.getNote(), msg.getVelocity(), msg.getChannel(),
                             msg.getTimestamp().time_since_epoch().count());
        case MIDIMessageType::NOTE_OFF:
            return fmt::format("{{\"t\":\"n-\",\"n\":{},\"v\":{},\"c\":{},\"ts\":{}}}",
                             msg.getNote(), msg.getVelocity(), msg.getChannel(),
                             msg.getTimestamp().time_since_epoch().count());
        case MIDIMessageType::CONTROL_CHANGE:
            return fmt::format("{{\"t\":\"cc\",\"n\":{},\"v\":{},\"c\":{},\"ts\":{}}}",
                             msg.getController(), msg.getValue(), msg.getChannel(),
                             msg.getTimestamp().time_since_epoch().count());
    }
}
```

---

### **Stage 2: Integrate Multicast with TOAST Transport**

🔹 **Goal**: Enable JSONL multicast through your existing TOAST network layer

**2.1: Add TOAST JSONL Message Type**

```cpp
// In JSONMIDI_Framework/include/TOASTTransport.h
enum class MessageType : uint8_t {
    MIDI = 0x01,
    JSONL_STREAM = 0x08,  // New message type
    JSONL_MULTICAST = 0x09,
    // ... existing types
};

class ProtocolHandler {
public:
    // Add JSONL streaming methods
    void startJsonlStream(const std::string& sessionId);
    void sendJsonlMessage(const std::string& jsonlLine);
    void subscribeToJsonlStream(std::function<void(const std::string&)> handler);
};
```

**2.2: Implement JSONL Multicast in ConnectionManager**

```cpp
// In JSONMIDI_Framework/src/TOASTTransport.cpp
class ConnectionManager::Impl {
public:
    // JSONL subscriber management
    std::unordered_map<std::string, std::vector<std::function<void(const std::string&)>>>
        jsonlSubscribers_;

    void multicastJsonlMessage(const std::string& sessionId, const std::string& jsonlLine) {
        auto it = jsonlSubscribers_.find(sessionId);
        if (it != jsonlSubscribers_.end()) {
            for (auto& subscriber : it->second) {
                subscriber(jsonlLine);
            }
        }

        // Also send over network to remote clients
        auto message = std::make_unique<TransportMessage>(
            MessageType::JSONL_MULTICAST, jsonlLine, getCurrentTimestamp(), getNextSequence());
        sendToAllClients(std::move(message));
    }
};
```

---

### **Stage 3: Enhance TOASTer Application for JSONL**

🔹 **Goal**: Add JSONL streaming UI to your existing TOASTer panels

**3.1: Add JSONL Panel to MainComponent**

```cpp
// In TOASTer/Source/MainComponent.h
class JsonlStreamingPanel : public juce::Component {
public:
    JsonlStreamingPanel(MIDIManager& midiManager);

    void startJsonlCapture();
    void stopJsonlCapture();
    void setStreamingMode(JSONMIDI::BassoonParser::ParseMode mode);

private:
    std::unique_ptr<JSONMIDI::BassoonParser> jsonlParser_;
    juce::TextEditor jsonlOutput_;
    juce::ToggleButton compactModeToggle_;
    juce::Label performanceLabel_;
};
```

**3.2: Integrate with Existing MIDI Flow**

```cpp
// In TOASTer/Source/MIDIManager.cpp
void MIDIManager::handleIncomingMidiMessage(juce::MidiInput* source,
                                          const juce::MidiMessage& message) {
    auto jsonMidiMessage = convertJuceMidiToJSONMIDI(message);

    // Existing queue processing
    if (incomingMessageQueue.tryPush(jsonMidiMessage)) {
        statistics.messagesReceived++;
    }

    // New JSONL streaming
    if (jsonlStreamingEnabled_ && jsonlParser_) {
        std::string jsonlLine = jsonlParser_->compactifyMessage(*jsonMidiMessage);
        multicastJsonlLine(jsonlLine);
    }
}
```

---

### **Stage 4: Network-Aware JSONL Distribution**

🔹 **Goal**: Distribute JSONL streams across your JAMWAN/MIDIp2p network

**4.1: Session-Based JSONL Streaming**

```cpp
// In JSONMIDI_Framework/include/TOASTTransport.h
class SessionManager {
public:
    struct JsonlSession {
        std::string sessionId;
        std::vector<std::string> subscribedClients;
        JSONMIDI::BassoonParser::ParseMode parseMode;
        bool isActive;
    };

    std::string createJsonlSession(const std::string& sessionName);
    bool joinJsonlSession(const std::string& sessionId, const std::string& clientId);
    void streamJsonlToSession(const std::string& sessionId, const std::string& jsonlLine);
    void setSessionParseMode(const std::string& sessionId,
                           JSONMIDI::BassoonParser::ParseMode mode);
};
```

**4.2: Add JSONL Controls to NetworkConnectionPanel**

```cpp
// In TOASTer/Source/NetworkConnectionPanel.h
class NetworkConnectionPanel : public juce::Component {
private:
    // Add JSONL streaming controls
    juce::ToggleButton jsonlStreamingToggle_;
    juce::ComboBox streamingModeSelector_;  // Single/JSONL/Compact
    juce::Slider compressionLevelSlider_;
    juce::Label jsonlStatusLabel_;

    // JSONL session management
    void startJsonlStreaming();
    void configureStreamingMode();
    void updateJsonlPerformance();
};
```

---

### **Stage 5: Real-Time Performance Optimization**

🔹 **Goal**: Achieve <50μs JSONL parsing while maintaining network streaming

**5.1: SIMD-Optimized JSONL Parser**

```cpp
// In JSONMIDI_Framework/src/BassoonParser.cpp
class BassoonParser::JsonlOptimizer {
public:
    // Pre-compiled compact format templates
    struct CompactTemplate {
        std::string_view prefix;
        std::array<size_t, 4> paramPositions;  // n, v, c, ts positions
        size_t totalLength;
    };

    static const CompactTemplate NOTE_ON_TEMPLATE;
    static const CompactTemplate NOTE_OFF_TEMPLATE;
    static const CompactTemplate CC_TEMPLATE;

    // SIMD-optimized compact parsing
    std::unique_ptr<MIDIMessage> parseCompactSIMD(const std::string_view& line);

private:
    // Use SIMD string operations for ultra-fast parsing
    bool matchTemplate(const std::string_view& line, const CompactTemplate& tmpl);
    void extractParamsSIMD(const std::string_view& line, const CompactTemplate& tmpl,
                          std::array<uint32_t, 4>& params);
};
```

**5.2: Lock-Free JSONL Queue Integration**

```cpp
// Integrate with existing LockFreeQueue system
template<size_t CAPACITY = 4096>
class JsonlStreamQueue {
public:
    struct JsonlEvent {
        std::string jsonlLine;
        std::chrono::high_resolution_clock::time_point timestamp;
        std::string sessionId;
    };

    bool tryPushJsonl(const JsonlEvent& event);
    bool tryPopJsonl(JsonlEvent& event);

private:
    JSONMIDI::LockFreeQueue<JsonlEvent, CAPACITY> queue_;
};
```

---

### **Stage 6: WebSocket Bridge for Browser Clients**

🔹 **Goal**: Enable browser-based JSONL streaming for web clients

**6.1: Add WebSocket Server to TOAST**

```cpp
// Optional: WebSocket bridge for browser integration
class WebSocketJsonlBridge {
public:
    void startWebSocketServer(uint16_t port);
    void bridgeToastToWebSocket(const std::string& sessionId);

    // Convert TOAST messages to WebSocket format
    void relayJsonlToWebClients(const std::string& jsonlLine);

private:
    std::unique_ptr<websocketpp::server<websocketpp::config::asio>> wsServer_;
    std::unordered_map<std::string, std::vector<websocketpp::connection_hdl>> wsClients_;
};
```

---

## 📈 **Performance Integration Targets**

| Component     | Current Target | JSONL Enhanced Target        |
| ------------- | -------------- | ---------------------------- |
| BassoonParser | <50μs          | <30μs (compact JSONL)        |
| TOAST Network | <10ms          | <8ms (reduced payload)       |
| Memory Usage  | Minimal        | +15% (streaming buffers)     |
| CPU Usage     | Real-time safe | +5% (JSONL processing)       |
| Throughput    | 16+ clients    | 32+ clients (compact format) |

## 🔄 **Migration Strategy**

### **Phase A: Backward Compatible** (Week 1-2)

- Add JSONL support alongside existing JSON
- Extend BassoonParser with streaming mode
- Update TOAST with new message types
- No breaking changes to current API

### **Phase B: Performance Optimization** (Week 3-4)

- Implement compact JSONL format
- Add SIMD optimizations
- Integrate lock-free JSONL queues
- Benchmark against <50μs target

### **Phase C: Network Enhancement** (Week 5-6)

- Deploy session-based JSONL streaming
- Add TOASTer UI controls
- Test multi-client JSONL distribution
- Validate real-world performance

### **Phase D: Advanced Features** (Week 7-8)

- WebSocket bridge for browsers
- Dynamic compression levels
- Advanced multicast routing
- JAMNet integration testing

## 🧪 **Testing Strategy**

```bash
# Test existing functionality
./test_tcp_communication

# Test new JSONL integration
./test_jsonl_streaming --mode compact --clients 16

# Benchmark performance
./test_phase12_performance --jsonl-enabled

# Network streaming test
./toast_server --jsonl-session "test-session"
./toast_client --join-jsonl "test-session"
```

## 🎯 **Expected Outcomes**

- ✅ **Full backward compatibility** with existing JSONMIDI/TOAST
- ✅ **<30μs JSONL parsing** (improved from 50μs JSON target)
- ✅ **2x network efficiency** via compact format
- ✅ **Real-time multicast streaming** for JAMNet
- ✅ **Enhanced TOASTer application** with JSONL controls
- ✅ **Future-proof architecture** for web integration

This roadmap leverages your existing ultra-low-latency infrastructure while adding modern streaming capabilities that align perfectly with your MIDIp2p network architecture and performance requirements.
