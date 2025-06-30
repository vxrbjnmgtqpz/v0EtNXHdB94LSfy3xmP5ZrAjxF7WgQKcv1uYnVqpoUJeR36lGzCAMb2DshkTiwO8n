# MIDIp2p: JSONMIDI Framework with TOAST Network Layer

**Real-time MIDI streaming over JSON with ultra-low-latency networking for distributed audio collaboration.**

## Overview

MIDIp2p is a comprehensive framework for streaming MIDI data over the internet using JSON, providing a 1:1 byte-level mapping of MIDI messages into a human-readable format. The project enables real-time MIDI collaboration over network connections using the TOAST (Transport Optimized Audio Synchronization Tunnel) protocol.

### Key Components

- **JSONMIDI Framework**: Complete MIDI 1.0/2.0 to JSON specification with lossless byte-level mapping
- **Bassoon.js**: Ultra-low-latency JSON parsing engine with signal-driven architecture
- **TOAST Protocol**: TCP-based network transport with distributed clock synchronization  
- **ClockDriftArbiter**: Network timing synchronization for sub-10ms distributed audio
- **MIDILink**: Native macOS testing application with VST3 bridge compatibility
- **JUCE Integration**: Cross-platform audio plugin framework integration

## Project Structure

```
MIDIp2p/
├── JSONMIDI_Framework/           # Core protocol specification
│   ├── Initialization.md         # Complete JSON-MIDI mapping spec
│   ├── memorymapandc++bridge.md  # Bassoon.js ultra-low-latency implementation
│   ├── ClockDriftArbiter.h       # Network synchronization module
│   └── Roadmap.md                # 16-week development timeline
├── MIDILink/                     # Main JUCE application
│   ├── Source/                   # Application source code
│   └── CMakeLists.txt           # Build configuration
├── nativemacapp.md               # MIDILink native app specification
└── README.md                     # This file
```

## Performance Targets

- **Local Processing**: <100μs latency (Bassoon.js)
- **Network Synchronization**: <10ms over LAN (TOAST)
- **Total End-to-End**: <15ms typical, <25ms worst-case
- **Throughput**: 10,000+ MIDI events/second
- **Clock Accuracy**: <1ms synchronization deviation
- **Concurrent Clients**: 16+ simultaneous connections

## Development Phases

### Phase 1: Core JSONMIDI Protocol (Weeks 1-4)
- JSON schema validation and refinement
- Bassoon.js SIMD-optimized parser implementation
- JUCE integration foundation

### Phase 2: TOAST Network Layer (Weeks 5-8)
- ClockDriftArbiter development
- TCP tunnel implementation  
- Distributed synchronization engine

### Phase 3: Integration & Testing (Weeks 9-12)
- End-to-end MIDI streaming
- Performance optimization
- Production readiness

### Phase 4: Platform & Ecosystem (Weeks 13-16)
- Cross-platform support
- Developer tools & SDK
- Open source preparation

## Technology Stack

- **Core Framework**: C++ with JUCE 7.0+
- **Networking**: TCP over LAN with Bonjour discovery
- **Platforms**: macOS (primary), Windows, Linux
- **Audio Integration**: VST3, Core Audio, ASIO
- **Protocol**: JSON over TOAST tunnel
- **Optimization**: SIMD, lock-free data structures

## Getting Started

### Prerequisites
- macOS 10.15+ (Catalina or later)
- Xcode with macOS development tools
- JUCE 7.0+ framework
- Core Audio/MIDI frameworks

### Quick Start
1. Clone the repository
2. Review the JSONMIDI specification in `JSONMIDI_Framework/Initialization.md`
3. Examine the MIDILink app plan in `nativemacapp.md`
4. Follow the roadmap in `JSONMIDI_Framework/Roadmap.md`

## Use Cases

- **Distributed DAW Sessions**: Musicians in different locations playing together
- **Cloud Audio Processing**: Remote DSP with maintained timing precision
- **Collaborative Music Creation**: Real-time shared virtual instruments
- **Multi-Room Audio**: Synchronized playback across network-connected spaces
- **Edge Audio Computing**: Distributed processing across IoT devices

## Future Vision: JamNet

MIDIp2p serves as the foundational MIDI layer for **JamNet**, a comprehensive distributed audio collaboration platform that will eventually include:

- Full audio streaming (post-MIDIp2p completion)
- Video synchronization
- Collaborative DAW features
- Cloud audio processing

## Contributing

This project is in active development. See `JSONMIDI_Framework/Roadmap.md` for current priorities and development timeline.

## License

[License information to be added]

---

*MIDIp2p represents a paradigm shift in audio networking, making JSON fast enough to replace binary MIDI protocols while retaining human-readable, web-compatible benefits. The framework enables professional-grade distributed audio applications with sub-15ms latency over standard network infrastructure.*
