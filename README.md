# MIDIp2p: JSONMIDI Framework with TOAST Network Layer

**Real-time MIDI streaming over JSON with ultra-low-latency networking for distributed audio collaboration.**

## Overview

MIDIp2p is a comprehensive framework for streaming MIDI data over the internet using JSON, providing a 1:1 byte-level mapping of MIDI messages into a human-readable format. The project enables real-time MIDI collaboration over network connections using the TOAST (Transport Oriented Audio Synchronization Tunnel) protocol.

## Why JSON? The Architecture Philosophy

### The Problem

**Transporting MIDI binary over an internet connection is sluggish.**

### The Solution

**The MIDI language does not have to be wrapped in binary in order to transmit a stream.** But it does need to be wrapped in something, so we use JSON. Here's exactly why JSON, paired with the JSONMIDI framework and streamed via Bassoon.js, IS the optimized cross-platform system, and not "verbose" at all.

**JavaScript's ubiquity + massive ecosystem = JSON's superpower.**

JSON is the most widely supported, best-documented, natively-parsed format in existence. There's no ambiguity. Nothing proprietary.

### Modular Architecture, Not Verbose

The JSONMIDI framework is a **broadcast language** — we are building a distributed, multi-device, multi-OS, multi-DAW framework with universal real-time interoperability.

**Our Development Strategy:**

- **Prototyping:** Native macOS, testing MacBook Pro to Mac Mini over USB4 TOAST link (most optimal conditions first)
- **Expansion:** Bootstrap Windows and Linux builds, perhaps even mobile apps, from the JSON layer first
- **Native Everything:** Each version built from scratch from the framework and documentation. No conversions, no wrappers, no Electron — all native apps built from the ground up
- **The Format is the Contract:** If all platforms speak the same JSONMIDI, they don't need to "see" each other — they just need to respect the schema and the stream

**That's not verbose. That's modular.**

### What We've Achieved Through JSONMIDI

🎯 **1. True Cross-Platform Sync Without Conversion Lag**

- If every system (macOS, Windows, Linux, mobile) is reading the same JSON stream, and decoding it in less than one millisecond...
- You've completely removed the need for any binary protocol or per-platform interpreter.

🔄 **2. Real-Time Collaborative Editing**

- Multiple musicians/devices can co-edit a MIDI session, and the stream can reflect changes in real time (note insertions, deletions, automation curves).
- This is DAW-agnostic, format-agnostic, and transport-flexible (TOAST, websockets, pipes, whatever).

🧠 **3. AI-Native Composition and Patch Interchange**

- You've now created a system where AI can generate, interpret, and manipulate live MIDI data in a readable format, without extra conversion steps.
- That means generative plugins, reactive performance assistants, adaptive sync layers — all on the same stream.

🧩 **4. Plug-and-Play Ecosystem**

- Any tool you build can tap into this stream by default. No wrapper needed.
- Third-party devs could build tools on top of your protocol without ever needing to reverse engineer anything.

### JSON = Real-Time Event Bus

This isn't just fast parsing. **It's foundational infrastructure for how audio software can be built going forward.**

We have a format that:

- **Anyone can read.**
- **Any AI can parse.**
- **Any OS can speak.**
- **Any stream can carry.**
- **And now, it's fast enough for real-time music.**

### Key Components

- **JSONMIDI Framework**: Complete MIDI 1.0/2.0 to JSON specification with lossless byte-level mapping
- **JELLIE (JAM Embedded Low Latency Instrument Encoding)**: Real-time audio streaming counterpart using JSONADAT format
- **JSONADAT**: JSON-based audio sample streaming with redundant mono encoding and 192kHz reconstruction
- **Bassoon.js**: Ultra-low-latency JSON parsing engine with signal-driven architecture
- **TOAST Protocol**: Transport Oriented Audio Synchronization Tunnel - UDP-based network transport with distributed clock synchronization
- **PNTBTR**: Predictive Network Temporal Buffered Transmission Recovery - musical continuity over packet loss
- **ClockDriftArbiter**: Network timing synchronization for sub-10ms distributed audio
- **MIDILink**: Native macOS testing application with VST3 bridge compatibility
- **JUCE Integration**: Cross-platform audio plugin framework integration

## The Complete Streaming Ecosystem

### Parallel Streaming Architecture

MIDIp2p is part of a **dual-stream architecture** that handles both MIDI and audio through parallel JSON-based protocols:

| **MIDI Stack**             | **Audio Stack**                 |
| -------------------------- | ------------------------------- |
| **MIDIp2p**                | **JELLIE**                      |
| → `JSONMIDI` format        | → `JSONADAT` format             |
| → Events & control data    | → PCM sample chunks             |
| → Sent over TOAST/UDP      | → Sent over TOAST/UDP           |
| → PNTBTR fills lost events | → PNTBTR predicts waveform gaps |

### JSONADAT: Audio Streaming Format

Each audio slice is transmitted as readable JSON:

```json
{
  "type": "audio",
  "id": "jsonadat",
  "seq": 142,
  "rate": 96000,
  "channel": 0,
  "redundancy": 1,
  "data": {
    "samples": [0.0012, 0.0034, -0.0005, ...]
  }
}
```

**Key Innovation - 192kHz Strategy:**

- Uses 4 ADAT channels for 1 mono stream
- 2 channels with offset sample timing = 192kHz reconstruction
- 2 additional channels for redundancy/parity
- No binary data - pure JSON throughout

### UDP + PNTBTR: Rethinking Network Reliability

**The Problem with TCP:**

- Handshakes and ACKs add unpredictable latency
- Retries kill musical timing
- "Reliable delivery" doesn't mean "musically relevant delivery"

**Our UDP + PNTBTR Solution:**

```
TCP Approach:     [ JSON ] → [ TCP ] → wait → retry → ACK → maybe late
Our Approach:     [ JSON ] → [ UDP ] → [ PNTBTR prediction ] → continuous music
```

**PNTBTR (Predictive Network Temporal Buffered Transmission Recovery):**

- **For MIDI**: Interpolates missing events, smooth CC curves
- **For Audio**: Predicts waveform continuation, buffer smoothing
- **Philosophy**: Prediction over retransmission, groove over perfection
- **Result**: Musical continuity maintained even with packet loss

## Network Protocol Evolution

### TOAST Protocol Layers

```
Application    →    MIDIp2p / JELLIE apps
Encoding      →    JSONMIDI / JSONADAT
Transport     →    TOAST (UDP-only)
Recovery      →    PNTBTR (musical prediction)
```

**Why UDP Won:**

- 🔥 No handshakes - immediate transmission
- ⚡ Sub-5ms latency achievable
- 🎯 Perfect for LAN and metro-area networks
- 🧱 PNTBTR handles gaps musically, not mechanically

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

_MIDIp2p represents a paradigm shift in audio networking, making JSON fast enough to replace binary MIDI protocols while retaining human-readable, web-compatible benefits. The framework enables professional-grade distributed audio applications with sub-15ms latency over standard network infrastructure._
