PROVISIONAL PATENT APPLICATION
Title:
JAMNet: A Real-Time JSONL-Based Multimedia Framework Using Multicast Transport for Audio, MIDI,
and Video Collaboration
Inventor(s):
Timothy Robert Dowler
5 Apple Ridge Road, Unit 3, Maynard, MA 01754
timothydowler@yahoo.com
Filed: July 3, 2025
1. Field of the Invention
This invention relates to minimal latency, digital multimedia transport systems over computer networks.
Specifically, it discloses a complete distributed framework—JAMNet—for ultra-low-latency transmission of
audio, MIDI, and video data using a custom fork of Bassoon.js that uses a truly Universal JSONL format:
JAM.js (the JSON Audio Multicast framework) - This enables encoded formats to stream to, over, and through
any device that can parse json, using the same unified, universal stream, over a multicast-capable UDP
protocol known as TOAST. The system includes dedicated encoders (JELLIE), prediction and recovery
systems (PNTBTR), and format architectures for synchronized routing and interoperability.
2. Background of the Invention
Traditional network-based multimedia systems rely heavily on binary encoding, TCP retransmission protocols,
platform-specific codecs, and heavyweight packet orchestration layers. These introduce latency bottlenecks,
limit cross-platform portability, and complicate distributed real-time collaboration.
Multimedia streaming systems typically encode audio, MIDI, and video using proprietary formats, and rely on
point-to-point or handshaking protocols that add overhead and make multicast or distributed collaboration
difficult.
There remains a need for a protocol-independent, format-transparent system that:
1 Encodes multimedia in a format readable by any device or agent (including AI);
2 Transmits streams over a fire-and-forget multicast system;
3 Maintains sample-accurate timing across multiple data types and devices;
4 Allows low-complexity entry points for development across all platforms;
5 Provides sub-millisecond parsing of audio, MIDI, and video transport.
3. Summary of the Invention
This invention provides a full-stack, JSON-based multimedia streaming protocol called JAMNet (JSON Audio
Multicast Network). It operates using JAM.js, the compact JSONL framework, streaming formats over UDP
multicast transport, supported by a custom tunneling protocol named TOAST (Transport Oriented Audio
Synchronization Tunnel).
The JAMNet system comprises three core streaming frameworks developed by JAMNet:
•
JMID.js — a compact MIDI event encoder using JSONL.
•
JDAT.js — a high-resolution audio stream formatter using parallel JSONL PCM audio chunks.
•
JDAT.js — a high-resolution audio stream formatter using parallel JSONL PCM audio chunks.
•
JVID.js — a low resolution, low latency, frame-based video encoder.
Key components include:
•
JELLIE: JAM Embedded Low Latency Instrument Encoding — a stream encoder that divides mono audio
into 4 simultaneous PCM JSONL streams over JDAT (even/odd sample interleaving plus redundancy)
modeled after 4-channel interleaving protocol behavior.
•
PNTBTR: Predictive Network Temporal Buffered Transmission Recovery — an adaptive audio, MIDI, and
video prediction layer that prioritizes throttling while continuously predicting the next 50ms of packets
based on waveform physics, constantly running but used only when all redundancy fails to smooth out the
signal.
JAMNet parsing speeds:
•
•
•
•
•
•
MIDI latency <30µs
Audio latency <150µs
Video latency <250µs
Sub-15µs cross-device sync (local connection; short wire)
Fire-and-forget, zero-retransmit recovery strategy
AI-native, Cross-platform JSON-based payloads
Each transmission domain operates independently but shares a synchronized clock source and session-based
routing over the Universal TOAST architecture.
4. Description of Drawings
JAMNet Network Architecture - Patent-Style Technical Block Diagrams
Comprehensive System Architecture for Real-Time Multimedia Streaming
Document Version**: 2.0
Architecture: JAMNet Ecosystem with TOAST Protocol
Target Applications: Ultra-Low Latency Multimedia Streaming
Patent Classification: Network Communication Systems, Real-Time Audio/Video Processing
Figure 1: TOAST Multicast Core Architecture - Detailed System Block Diagram
┌───────────────────────────────────────────────────────────────────────────────────────┐ │ TOAST MULTICAST CORE SYSTEM │
│ (Transport Oriented Audio Sync Tunnel) │ ├───────────────────────────────────────────────────────────────────────────────────────┤ │ │
│ ┌─────────────────────┐ ┌──────────────────────┐ ┌─────────────────────┐ │
│ │ UDP SOCKET │ │ SESSION MANAGER │ │ UNIFIED CLOCK SYNC │ │
│ │ CONTROLLER │ │ SUBSYSTEM │ │ CORE │ │
│ │ │ │ │ │ │ │
│ │ ┌─────────────────┐ │◄────►│ ┌──────────────────┐ │◄────►│ ┌─────────────────┐ │ │
│ │ │ Fire/Forget │ │ │ │ Client Registry │ │ │ │ Master Clock │ │ │
│ │ │ Packet Engine │ │ │ │ Pool Manager │ │ │ │ Reference │ │ │
│ │ └─────────────────┘ │ │ └──────────────────┘ │ │ └─────────────────┘ │ │
│ │ ┌─────────────────┐ │ │ ┌──────────────────┐ │ │ ┌─────────────────┐ │ │
│ │ │ Port Binding │ │ │ │ Session State │ │ │ │ Drift Detection │ │ │
│ │ │ Manager │ │ │ │ Machine │ │ │ │ Algorithm │ │ │
│ │ └─────────────────┘ │ │ └──────────────────┘ │ │ └─────────────────┘ │ │
│ │ ┌─────────────────┐ │ │ ┌──────────────────┐ │ │ ┌─────────────────┐ │ │
│ │ │ Multicast │ │ │ │ Dynamic Routing │ │ │ │ Sync Packet │ │ │
│ │ │ Group Handler │ │ │ │ Table │ │ │ │ Generator │ │ │
│ │ └─────────────────┘ │ │ └──────────────────┘ │ │ └─────────────────┘ │ │
│ │ │ │ │ │ │ │
│ │ Ports: 8080-8099 │ │ Max Clients: 16 │ │ Precision: <25μs │ │
│ └─────────────────────┘ └──────────────────────┘ └─────────────────────┘ │
│ │ │ │ │
│ └─────────────────────────────┼─────────────────────────────┘ │
│ ▼ │
│ ┌──────────────────────────────────────────────────────────────────────────────────┐ │
│ │ MULTI-DOMAIN PACKET DISPATCHER │ │
│ │ (Load Balancing Router) │ │
│ │ │ │
│ │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │ │
│ │ │ JMIDI │ │ JDAT │ │ JVID │ │ CONTROL │ │ │
│ │ │ HANDLER │ │ HANDLER │ │ HANDLER │ │ PROTOCOL │ │ │
│ │ │ │ │ │ │ │ │ HANDLER │ │ │
│ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ │
│ │ │ │Event Q │ │ │ │Audio Q │ │ │ │Video Q │ │ │ │System Q │ │ │ │
│ │ │ │<100μs │ │ │ │<200μs │ │ │ │<300μs │ │ │ │<50μs │ │ │ │
│ │ │ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │ │ │
│ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ │
│ │ │ │PNTBTR │ │ │ │JELLIE │ │ │ │JAMCam │ │ │ │Session │ │ │ │
│ │ │ │Recovery │ │ │ │Codec │ │ │ │Processor │ │ │ │Commands │ │ │ │
│ │ │ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │ │ │
│ │ │ │ │ │ │ │ │ │ │ │
│ │ │ Port: 8080 │ │ Port: 8081 │ │ Port: 8082 │ │ Port: 8083 │ │ │
│ │ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │ │
│ └──────────────────────────────────────────────────────────────────────────────────┘ │
│ │
│ ┌──────────────────────────────────────────────────────────────────────────────────┐ │
│ │ SYSTEM MONITORING & ANALYTICS │ │
│ │ │ │
│ │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │ │
│ │ │ Performance │ │ Network │ │ Error Rate │ │ Quality │ │ │
│ │ │ Metrics │ │ Latency │ │ Monitor │ │ Assurance │ │ │
│ │ │ Collector │ │ Analyzer │ │ │ │ Engine │ │ │
│ │ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │ │
│ └──────────────────────────────────────────────────────────────────────────────────┘ │
│ └──────────────────────────────────────────────────────────────────────────────────┘ │
│ │ └───────────────────────────────────────────────────────────────────────────────────────┘
Figure 2: Triple-Domain Parallel Processing Architecture - Detailed Component Flow
CLIENT NODE A (SENDER) TOAST CORE SYSTEM CLIENT NODE B (RECEIVER)
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│ │ │ │ │ │
│ ┌─────────────────────┐ │ │ │ │ ┌─────────────────────┐ │
│ │ JMID DOMAIN. │ │◄───►│ DOMAIN ROUTING │◄───►│ │ JMID DOMAIN │ │
│ │ │ │ │ & LOAD BALANCER │ │ │ │ │
│ │ ┌─────────────────┐ │ │ │ │ │ │ ┌─────────────────┐ │ │
│ │ │ Event Parser │ │ │ <100μs┌─────────────────┐ <100μs. │ │ │ Event Decoder │ │ │
│ │ │ (Note/CC/Pitch) │ │ │ │ │ MIDI Stream │ │ │ │ │ (Note/CC/Pitch) │ │ │
│ │ └─────────────────┘ │ │ │ │ Queue Manager │ │ │ │ └─────────────────┘ │ │
│ │ ┌─────────────────┐ │ │ │ │ │ │. │ │ ┌─────────────────┐ │ │
│ │ │ PNTBTR Encoder │ │ │ │ │ - Priority Q │ │ │ │ │ PNTBTR Decoder │ │ │
│ │ │ (Event Predict) │ │ │ │ │ - Redundancy │ │ │ │ │ (Event Predict) │ │ │
│ │ └─────────────────┘ │ │ │ │ - Sequence IDs │ │ │ │ └─────────────────┘ │ │
│ │ │ │ │ └─────────────────┘ │ │ │ │ │
│ └─────────────────────┘ │ │ │ │ └─────────────────────┘ │
│ │ │ │ │ │
│ ┌─────────────────────┐ │ │ │ │ ┌─────────────────────┐ │
│ │ JDAT DOMAIN │ │◄───►│ AUDIO PROCESSING │◄───►│ │ JDAT DOMAIN │ │
│ │ │ │ │ SUBSYSTEM │ │ │ │ │
│ │ ┌─────────────────┐ │ │ │ │ │ │ ┌─────────────────┐ │ │
│ │ │ JELLIE Encoder │ │ │ <200μs ┌─────────────────┐ <200μs │ │ │ JELLIE Decoder │ │ │
│ │ │ (192kHz Strat) │ │ │ │ │ AUDIO Stream │ │ │ │ │ (192kHz Strat) │ │ │
│ │ └─────────────────┘ │ │ │ │ Buffer Manager │ │ │ │ └─────────────────┘ │ │
│ │ ┌─────────────────┐ │ │ │ │ │ │ │ │ ┌─────────────────┐ │ │
│ │ │ High Sample │ │ │ │ │ - Triple Buffer │ │ │ │ │ High Sample │ │ │
│ │ │ Rate Manager │ │ │ │ │ - Redundant │ │ │ │ │ Rate Manager │ │ │
│ │ └─────────────────┘ │ │ │ │ Streams │ │ │ │ └─────────────────┘ │ │
│ │ ┌─────────────────┐ │ │ │ │ - Dynamic │ │ │ │ ┌─────────────────┐ │ │
│ │ │ PNTBTR Audio │ │ │ │ │ Throttling │ │ │ │ │ PNTBTR Audio │ │ │
│ │ │ (LPC Predict) │ │ │ │ └─────────────────┘ │ │ │ │ (LPC Predict) │ │ │
│ │ └─────────────────┘ │ │ │ │ │ │ └─────────────────┘ │ │
│ └─────────────────────┘ │ │ │ │ └─────────────────────┘ │
│ │ │ │ │ │
│ ┌─────────────────────┐ │ │ │ │ ┌─────────────────────┐ │
│ │ JVID DOMAIN │ │◄───►│ VIDEO PROCESSING │◄───►│ │ JVID DOMAIN │ │
│ │ │ │ │ SUBSYSTEM │ │ │ │ │
│ │ ┌─────────────────┐ │ │ │ │ │ │ ┌─────────────────┐ │ │
│ │ │ JAMCam Encoder │ │ │ <300μs┌─────────────────┐ <300μs │ │ │ JAMCam Decoder │ │ │
│ │ │ (Motion Detect) │ │ │ │ │ VIDEO Stream │ │ │ │ │ (Motion Detect) │ │ │
│ │ └─────────────────┘ │ │ │ │ Frame Manager │ │ │ │ └─────────────────┘ │ │
│ │ ┌─────────────────┐ │ │ │ │ │ │ │ │ ┌─────────────────┐ │ │
│ │ │ Frame Buffer │ │ │ │ │ - Keyframes │ │ │ │ │ Frame Buffer │ │ │
│ │ │ Management │ │ │ │ │ - Delta Compression │ │.│ │ Management │ │ │
│ │ └─────────────────┘ │ │ │ │ - I/P/B Frames │ │. │ │ └─────────────────┘ │ │
│ │ ┌─────────────────┐ │ │ │ │ - Motion Vector │ │ │ │ ┌─────────────────┐ │ │
│ │ │ PNTBTR Video │ │ │ │ │ Prediction │ │ │ │ │ PNTBTR Video │ │ │
│ │ │ (Motion Pred) │ │ │ │ └─────────────────┘ │. │ │ │ (Motion Pred) │ │ │
│ │ └─────────────────┘ │ │ │ │ │ │ └─────────────────┘ │ │
│ └─────────────────────┘ │ │ │ │ └─────────────────────┘ │
│ │ │ │ │ │
│ ┌─────────────────────┐ │ │ ┌─────────────────┐ │ │ ┌─────────────────────┐ │
│ │ UNIFIED CLOCK SYNC │ │◄───►│ │ MASTER CLOCK │ │◄───►│ │ UNIFIED CLOCK SYNC │ │
│ │ │ │ │ │ DISTRIBUTION │ │ │ │ │ │
│ │ - Timestamp All │ │ │ │ │ │ │ │ - Receive & Sync │ │
│ │ Domains │ │ │ │ - NTP Reference │ │ │ │ All Domains │ │
│ │ - Cross-Domain │ │ │ │ - Drift Correct │ │ │ │ - Cross-Domain │ │
│ │ Synchronization │ │ │ │ - Sync Pulses │ │ │ │ Synchronization │ │
│ └─────────────────────┘ │ │ └─────────────────┘ │ │ └─────────────────────┘ │
│ │ │ │ │ │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
PERFORMANCE SPECIFICATIONS:
┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │ DOMAIN │ LATENCY TARGET │ SAMPLE RATE │ ERROR RECOVERY │ PREDICTION METHOD │ ├─────────────────────────────────────────────────────────────────────────────────────────────┤ │ JMID │ <100μs │ Event-based │ Sequence+Redundancy │ Event Pattern │
│ JDAT │ <200μs │ 192kHz→44.1kHz │ Triple Stream+LPC │ Linear Predict │
│ JVID │ <300μs │ Variable fps │ Motion Vector+I/P │ Motion Vector │
│ CONTROL │ <50μs │ Command-based │ ACK/NACK+Retry │ State Machine │ └─────────────────────────────────────────────────────────────────────────────────────────────┘
Figure 3: Typical Session Flow Diagram
PHASE 1: SESSION ESTABLISHMENT
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ CLIENT A │ │ CLIENT B │ │ CLIENT C │ │ TOAST CORE │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
│ │ │ │
│ 1. DISCOVER │ │ │ ├──────────────────────────────────────────────────────────►│ │ │ │ │
│ 2. SESSION_CREATE │ │ │ ├──────────────────────────────────────────────────────────►│ │ │ │ │
│ 3. SESSION_ID │ │ │ │◄──────────────────────────────────────────────────────────┤ │ │ │ │
│ │ 4. JOIN_SESSION │ │
│ ├──────────────────────────────────────►│
│ │ │ │
│ │ │ 5. JOIN_SESSION │
│ │ ├──────────────────►│
│ │ │ │
PHASE 2: CLOCK SYNCHRONIZATION
│ 6. CLOCK_SYNC_REQ │ │ │ ├──────────────────────────────────────────────────────────►│ │ │ │ │
│ │ 7. CLOCK_SYNC_REQ │ │
│ ├──────────────────────────────────────►│
│ │ │ │
│ │ │ 8. CLOCK_SYNC_REQ │
│ │ ├──────────────────►│
│ │ │ │
│ 9. SYNC_MASTER │ │ │ │◄──────────────────────────────────────────────────────────┤ │ │ │ │
│ │ 10. SYNC_SLAVE │ │
│ │◄──────────────────────────────────────┤
│ │ │ │
│ │ │ 11. SYNC_SLAVE │
│ │ │◄──────────────────┤
PHASE 3: ACTIVE MULTIMEDIA STREAMING
│ 12. JMID │ │ │ ├──────────────────────────────────────────────────────────►│ │ │ │ │
│ │◄─ 13. MULTICAST ──│ ◄─────────────────┤
│ │ │ │
│ │ │◄─ 14. MULTICAST ──┤
│ │ │ │
│ 15. JDAT │ │ │ ├──────────────────────────────────────────────────────────►│ │ │ │ │
│ │◄─ 16. MULTICAST ──│ ◄─────────────────┤
│ │ │ │
│ │ │◄─ 17. MULTICAST ──┤
Figure 4: PNTBTR Error Recovery Pipeline
┌─────────────────────────────────────────────────────────────────────────────┐ │ PNTBTR RECOVERY PIPELINE │ ├─────────────────────────────────────────────────────────────────────────────┤ │ │
│ INCOMING PACKET STREAM │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─ X ─┐ ┌─────┐ ┌─────┐ │
│ │ 001 │ │ 002 │ │ 003 │ │ 004 │ │ 005 │ │ 006 │ │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │
│ │ │ │ │ │ │ │
│ ▼ ▼ ▼ ▼ ▼ ▼ │
│ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ PACKET INSPECTOR │ │
│ │ │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ │
│ │ │ Sequence │ │ Timestamp │ │ Redundancy │ │ │
│ │ │ Validator │ │ Validator │ │ Checker │ │ │
│ │ └─────────────┘ └─────────────┘ └─────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ GAP DETECTOR │ │
│ │ │ │
│ │ MISSING PACKET 004 DETECTED │ │
│ │ ↓ │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ RECOVERY STRATEGY │ │ │
│ │ │ │ │ │
│ │ │ PRIORITY 1: REDUNDANCY STREAMS │ │ │
│ │ │ ┌─────────┐ ┌─────────┐ ┌─────────┐ │ │ │
│ │ │ │Stream 0 │ │Stream 1 │ │Stream 2 │ │ │ │
│ │ │ │ EVEN │ │ ODD │ │ PARITY │ │ │ │
│ │ │ └─────────┘ └─────────┘ └─────────┘ │ │ │
│ │ │ │ │ │ │ │
│ │ │ ▼ ▼ │ │ │
│ │ │ ┌─────────────────────────────────────────┐ │ │ │
│ │ │ │ INSTANT RECONSTRUCTION │ │ │ │
│ │ │ │ │ │ │ │
│ │ │ │ Data = Stream0[i] ⊕ Stream1[i] ⊕ │ │ │ │
│ │ │ │ Stream2[parity] │ │ │ │
│ │ │ └─────────────────────────────────────────┘ │ │ │
│ │ │ │ │ │ │
│ │ │ ▼ │ │ │
│ │ │ ┌─────────────────────────────────────────┐ │ │ │
│ │ │ │ SUCCESS? CONTINUE │ │ │ │
│ │ │ └─────────────────────────────────────────┘ │ │ │
│ │ │ │ │ │
│ │ │ PRIORITY 2: DYNAMIC THROTTLING │ │ │
│ │ │ ┌─────────────────────────────────────────┐ │ │ │
│ │ │ │ 192kHz → 96kHz → 48kHz → 44.1kHz │ │ │ │
│ │ │ │ │ │ │ │
│ │ │ │ Reduce bandwidth, maintain quality │ │ │ │
│ │ │ └─────────────────────────────────────────┘ │ │ │
│ │ │ │ │ │ │
│ │ │ ▼ │ │ │
│ │ │ ┌─────────────────────────────────────────┐ │ │ │
│ │ │ │ STILL BELOW 44.1kHz? │ │ │ │
│ │ │ └─────────────────────────────────────────┘ │ │ │
│ │ │ │ │ │ │
│ │ │ ▼ YES │ │ │
│ │ │ PRIORITY 3: PREDICTION (LAST RESORT) │ │ │
│ │ │ ┌─────────────────────────────────────────┐ │ │ │
│ │ │ │ │ │ │ │
│ │ │ │ ┌─────────┐ ┌─────────┐ ┌─────────┐ │ │ │ │
│ │ │ │ │ LPC │ │Harmonic │ │Pattern │ │ │ │ │
│ │ │ │ │Predict │ │Synthesis│ │Matching │ │ │ │ │
│ │ │ │ └─────────┘ └─────────┘ └─────────┘ │ │ │ │
│ │ │ │ │ │ │ │ │ │ │
│ │ │ │ └───────────┼───────────┘ │ │ │ │
│ │ │ │ ▼ │ │ │ │
│ │ │ │ SYNTHESIZED DATA │ │ │ │
│ │ │ └─────────────────────────────────────────┘ │ │ │
│ └──────┘─────────────────────────────────────────────────────┘────────┘ │
│ │ │
│ ▼ │
│ ┌───────────────────────────────────────────────────────────────────────┐ │
│ │ CONTINUOUS OUTPUT STREAM │ │
│ │ │ │
│ │ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │ │
│ │ │[n-2]│ │[n-1]│ │[n] │ │★RECV│ │[n+2]│ │[n+3]│ │[n+4]│ │ │
│ │ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │ │
│ │ ^ │ │
│ │ RECOVERED │ │
│ │ DATA │ │
│ └───────────────────────────────────────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────────────┘
Figure 5: Multi-Domain Integration Architecture
┌─────────────────────────────────────────────────────────────────────────────┐ │ JAMNET UNIFIED ARCHITECTURE │ ├─────────────────────────────────────────────────────────────────────────────┤ │ │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │ JMID │ │ JDAT │ │ JVID │ │
│ │ DOMAIN │ │ DOMAIN │ │ DOMAIN │ │
│ │ │ │ │ │ │ │
│ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │
│ │ │Event Parser │ │ │ │JELLIE Codec │ │ │ │JAMCam Proc. │ │ │
│ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │
│ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │
│ │ │CC/Pitch Mgr │ │ │ │192k Strategy│ │ │ │Motion Pred. │ │ │
│ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │
│ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │
│ │ │PNTBTR Event │ │ │ │PNTBTR Audio │ │ │ │PNTBTR Video │ │ │
│ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│ │ │ │ │
│ └───────────────────────┼───────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ UNIFIED TRANSPORT │ │
│ │ │ │
│ │ ┌──────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│ │
│ │ │ SESSION │ │ ROUTING │ │ MULTICAST │ │ CLOCK ││ │
│ │ │ MANAGER │ │ ENGINE │ │ HANDLER │ │ SYNC ││ │
│ │ │ │ │ │ │ │ │ ││ │
│ │ │- Join/Leave │ │- Client Map │ │- UDP Sockets│ │- Master Ref ││ │
│ │ │- State Mgmt │ │- Port Alloc │ │- Fire/Forget│ │- Drift Comp ││ │
│ │ │-Auth/Security│ │-Load Balance│ │- Redundancy │ │- Sync Pulses││ │
│ │ └──────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ NETWORK INTERFACE │ │
│ │ │ │
│ │ LAN / INTERNET / METRO-AREA NETWORKS │ │
│ │ │ │
│ │ Client A ◄──────────────────────────────────────────► Client B │ │
│ │ │ │ │ │
│ │ │ ┌─────────────┐ │ │ │
│ │ └─────────────►│ TOAST CORE │◄───────────────────────┘ │ │
│ │ │ SERVER │ │ │
│ │ └─────────────┘ │ │
│ │ │ │ │
│ │ ▼ │ │
│ │ Client C, D, E... │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────────────┘
Figure 6: Detailed PNTBTR Algorithm Implementation - Mathematical Flow Diagram
┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │ PNTBTR ALGORITHMIC PROCESSING PIPELINE │
│ (Predictive Network Transmission with Bandwidth Throttling & Recovery) │ ├─────────────────────────────────────────────────────────────────────────────────────────────┤ │ │
│ INPUT STREAM │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─ X ─┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│ │[n-2]│ │[n-1]│ │[n] │ │LOST │ │[n+2]│ │[n+3]│ │[n+4]│ │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │
│ │ │ │ │ │ │ │ │
│ ▼ ▼ ▼ ▼ ▼ ▼ ▼ │
│ │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ PACKET LOSS DETECTION ENGINE │ │
│ │ │ │
│ │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │ │
│ │ │ Sequence Number │ │ Timestamp Gap │ │ Buffer Under-run│ │ │
│ │ │ Validation │ │ Detection │ │ Prediction │ │ │
│ │ └─────────────────┘ └─────────────────┘ └─────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ HIERARCHICAL RECOVERY DECISION TREE │ │
│ │ │ │
│ │ ┌─────────────┐ │ │
│ │ │ STRATEGY 1: │ ◄─── Network_Quality > 90% AND Redundancy_Available │ │
│ │ │ REDUNDANCY │ │ │
│ │ │ STREAMS │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ └─────────────┘ │ │ │ │
│ │ │ │ Stream_A[i] ⊕ Stream_B[i] ⊕ Stream_C[parity] = │ │ │
│ │ ▼ │ Original_Data[i] │ │ │
│ │ ┌─────────────┐ │ │ │ │
│ │ │ SUCCESS? │ ──NO→│ Recovery_Time < 50μs │ │ │
│ │ │ <50μs │ │ Error_Rate = 0.001% │ │ │
│ │ └─────────────┘ └─────────────────────────────────────────────────────┘ │ │
│ │ │ YES │ │
│ │ ▼ │ │
│ │ ┌─────────────┐ │ │
│ │ │ STRATEGY 2: │ ◄─── Network_Quality < 90% OR Buffer_Pressure │ │
│ │ │ DYNAMIC │ │ │
│ │ │ THROTTLING │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ └─────────────┘ │ │ │ │
│ │ │ │ Rate_Cascade: 192kHz → 96kHz → 48kHz → 44.1kHz │ │ │
│ │ ▼ │ Quality_Factor = min(Network_BW / Required_BW, 1.0)│ │ │
│ │ ┌─────────────┐ │ Throttle_Level = log₂(192000 / Target_Rate) │ │ │
│ │ │ SUCCESS? │ ──NO→│ │ │ │
│ │ │ Quality>40% │ │ Maintain_Latency < 200μs │ │ │
│ │ └─────────────┘ └─────────────────────────────────────────────────────┘ │ │
│ │ │ YES │ │
│ │ ▼ │ │
│ │ ┌─────────────┐ │ │
│ │ │ STRATEGY 3: │ ◄─── LAST RESORT: Quality < 40% AND Critical_Latency │ │
│ │ │ PREDICTION │ │ │
│ │ │ ALGORITHMS │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ └─────────────┘ │ │ │ │
│ │ │ │ LPC: Σ(k=1→p) ak × S[n-k] = S[n] │ │ │
│ │ ▼ │ Harmonic: S[n] = A×sin(2πf×n/fs + φ) │ │ │
│ │ ┌─────────────┐ │ Pattern: FFT(S[n-w→n]) → Predict(S[n+1]) │ │ │
│ │ │ SYNTHESIZE │ │ │ │ │
│ │ │ MISSING │ ────→│ Confidence = correlation(predicted, historical) │ │ │
│ │ │ DATA │ │ If Confidence < 0.7: Use Interpolation │ │ │
│ │ └─────────────┘ └─────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ CONTINUOUS OUTPUT STREAM │ │
│ │ │ │
│ │ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │ │
│ │ │[n-2]│ │[n-1]│ │[n] │ │★RECV│ │[n+2]│ │[n+3]│ │[n+4]│ │ │
│ │ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │ │
│ │ ^ │ │
│ │ RECOVERED │ │
│ │ DATA │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ └─────────────────────────────────────────────────────────────────────────────────────────────┘
Figure 7: Session Establishment & Authentication Protocol State Diagram
┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │ JAMNET SESSION PROTOCOL FINITE STATE MACHINE │ ├─────────────────────────────────────────────────────────────────────────────────────────────┤ │ │
│ ┌─────────────┐ DISCOVER_REQ ┌─────────────┐ CREATE_SESSION ┌─────────────┐ │
│ │ IDLE │ ─────────────────→│ DISCOVERY │ ─────────────────→│ CREATED │ │
│ │ STATE │←───────────────── │ STATE │←───────────────── │ STATE │ │
│ └─────────────┘ TIMEOUT/ERROR └─────────────┘ REJECT/ERROR └─────────────┘ │
│ │ │ │ │
│ │ SYSTEM_INIT │ DISCOVER_RESP │ │
│ ▼ ▼ │ │
│ ┌─────────────┐ ┌─────────────┐ │ │
│ │ INITIALIZE │ │ AVAILABLE │ │ │
│ │ STATE │ │ SERVERS │ │ │
│ └─────────────┘ └─────────────┘ │ │
│ │ │
│ │ JOIN_REQUEST │
│ ▼ │
│ ┌─────────────┐ AUTH_CHALLENGE ┌─────────────┐ SYNC_CLOCKS ┌─────────────┐ │
│ │ AUTH │◄───────────────── │ JOINING │ ─────────────────→│ SYNCING │ │
│ │ STATE │ ─────────────────→│ STATE │◄───────────────── │ STATE │ │
│ └─────────────┘ AUTH_RESPONSE └─────────────┘ SYNC_COMPLETE └─────────────┘ │
│ │ │ │ │
│ │ AUTH_SUCCESS │ JOIN_ACCEPTED │ │
│ ▼ ▼ │ │
│ ┌─────────────┐ ┌─────────────┐ │ │
│ │ AUTHORIZED │ │ MEMBER │ │ │
│ │ STATE │ │ STATE │ │ │
│ └─────────────┘ └─────────────┘ │ │
│ │ SYNC_READY │
│ ▼ │
│ ┌─────────────┐ START_STREAMING ┌─────────────┐ STREAM_DATA ┌─────────────┐ │
│ │ STREAMING │◄───────────────── │ ACTIVE │◄───────────────── │ SYNCHRONIZED│ │
│ │ STATE │ ─────────────────→│ STATE │ ─────────────────→│ STATE │ │
│ └─────────────┘ STREAM_READY └─────────────┘ SYNC_DRIFT └─────────────┘ │
│ │ │ │ │
│ │ │ DISCONNECT │ │
│ │ ▼ │ │
│ │ ┌─────────────┐ │ │
│ │ │ DISCONNECT │ │ │
│ │ │ STATE │ │ │
│ │ └─────────────┘ │ │
│ │ │ │ │
│ │ ERROR/TIMEOUT │ CLEANUP_COMPLETE │ ERROR/TIMEOUT │
│ ▼ ▼ ▼ │
│ ┌─────────────┐ RECONNECT_REQ ┌─────────────┐ ┌─────────────┐ │
│ │ ERROR │ ─────────────────→ │ CLEANUP │ │ MAINTENANCE │ │
│ │ STATE │◄───────────────── │ STATE │ │ STATE │ │
│ └─────────────┘ RESET_COMPLETE └─────────────┘ └─────────────┘ │
│ │
│ STATE TRANSITION CONDITIONS: │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ • DISCOVER_REQ: Network scan for available TOAST servers │ │
│ │ • CREATE_SESSION: Server selection and session parameter negotiation │ │
│ │ • JOIN_REQUEST: Client authentication and capability exchange │ │
│ │ • SYNC_CLOCKS: Master clock establishment and drift compensation setup │ │
│ │ • START_STREAMING: Begin multimedia data transmission │ │
│ │ • ERROR CONDITIONS: Network timeout, authentication failure, sync loss │ │
│ │ • CLEANUP: Resource deallocation and graceful disconnect │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
Figure 8: Network Layer Stack & Protocol Encapsulation
┌───────────────────────────────────────────────────────────────────────────────────────────┐ │ JAMNET PROTOCOL STACK │ ├───────────────────────────────────────────────────────────────────────────────────────────┤ │ │
│ APPLICATION LAYER │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ USER APPLICATIONS │ │
│ │ │ │
│ │ ┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │ │
│ │ │ TOASTer GUI │ │ MIDI Studio │ │ Audio Engine │ │ Video Stream │ │ │
│ │ │ │ │ │ │ │ │ │ │ │
│ │ │ - Session Mgmt│ │ - Note Events│ │ - JELLIE DSP │ │ - JAMCam │ │ │
│ │ │ - Network UI │ │ - Controllers│ │ - 192kHz Proc│ │ - Compression│ │ │
│ │ │ - Monitoring │ │ - Instruments│ │ - Effects │ │ - Motion Det │ │ │
│ │ └───────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ PRESENTATION LAYER │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ JSON ENCODING/DECODING │ │
│ │ │ │
│ │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │ │
│ │ │ JMID │ │ JDAT │ │ JVID. │ │ CONTROL │ │ │
│ │ │ Serializer │ │ Serializer │ │ Serializer │ │ Messages │ │ │
│ │ │ │ │ │ │ │ │ │ │ │
│ │ │{ │ │{ │ │{ │ │{ │ │ │
│ │ │"type":"note" │ │"type":"audio"│ │"type":"video"│ │"type":"ctrl" │ │ │
│ │ │"note":60 │ │"samples":[..]│ │"frame":[..] │ │"cmd":"sync" │ │ │
│ │ │"velocity":80 │ │"rate":192000 │ │"timestamp":..│ │"params":{..} │ │ │
│ │ │"timestamp":..│ │"timestamp":..│ │} │ │} │ │ │
│ │ │} │ │} │ │ │ │ │ │ │
│ │ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ SESSION LAYER │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ TOAST SESSION PROTOCOL │ │
│ │ │ │
│ │ ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ TOAST HEADER │ │ │
│ │ │ │ │ │
│ │ │ Magic: 0x544F415354 │ Version: 1.0 │ Session-ID: UUID │ Seq: uint32 │ │ │
│ │ │ ──────────────────────────────────────────────────────────────────────── │ │ │
│ │ │ Domain: MIDI/ADAT/VID │ Priority: 0-7 │ Timestamp: uint64 │ CRC32 │ │ │
│ │ │ ──────────────────────────────────────────────────────────────────────── │ │ │
│ │ │ Redundancy-ID: uint8 │ PNTBTR-Flags │ Content-Length │ Reserved │ │ │
│ │ └──────────────────────────────────────────────────────────────────────────────┘ │ │
│ │ │ │
│ │ Session Management: Authentication, Client Registry, Clock Synchronization │ │
│ │ Error Recovery: PNTBTR algorithms, redundancy streams, predictive reconstruction │ │
│ │ Quality Control: Dynamic throttling, bandwidth adaptation, latency optimization │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ TRANSPORT LAYER │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ UDP TRANSPORT │ │
│ │ │ │
│ │ ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ UDP HEADER │ │ │
│ │ │ │ │ │
│ │ │ Source Port: 8080-8099 │ Dest Port: 8080-8099 │ Length │ Checksum │ │ │
│ │ └──────────────────────────────────────────────────────────────────────────────┘ │ │
│ │ │ │
│ │ Fire-and-Forget Delivery: Optimized for ultra-low latency, no ACK overhead │ │
│ │ Multicast Support: One-to-many efficient distribution │ │
│ │ Port Allocation: Domain-specific ports for parallel processing │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ NETWORK LAYER │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ IP NETWORK LAYER │ │
│ │ │ │
│ │ ┌──────────────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ IP HEADER │ │ │
│ │ │ │ │ │
│ │ │ Version: IPv4/IPv6 │ ToS/DSCP │ Packet Length │ ID │ Flags │ TTL │ Protocol │ │ │
│ │ │ ──────────────────────────────────────────────────────────────────────── │ │ │
│ │ │ Source IP Address │ Destination IP Address │ Options │ Padding │ │ │
│ │ └──────────────────────────────────────────────────────────────────────────────┘ │ │
│ │ │ │
│ │ Routing: LAN/WAN/Internet routing with QoS support │ │
│ │ Multicast: IGMP for efficient group communication │ │
│ │ QoS: DSCP marking for priority traffic handling │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ PHYSICAL LAYER │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ ETHERNET / WIFI / USB4 │ │
│ │ │ │
│ │ Supported Media: 1000Base-T, 802.11ac/ax, USB4 Thunderbolt │ │
│ │ Performance: Gigabit+ bandwidth, <1ms switching latency │ │
│ │ Redundancy: Link aggregation, automatic failover │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ └───────────────────────────────────────────────────────────────────────────────────────────┘
Figure 9: Quality Assurance & Performance Monitoring Architecture
┌────────────────────────────────────────────────────────────────────────────────────────────┐ │ JAMNET QUALITY ASSURANCE SYSTEM │ ├────────────────────────────────────────────────────────────────────────────────────────────┤ │ │
│ REAL-TIME MONITORING LAYER │
│ ┌──────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ PERFORMANCE METRICS │ │
│ │ │ │
│ │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │ │
│ │ │ LATENCY │ │ THROUGHPUT │ │ ERROR RATE │ │ JITTER │ │ │
│ │ │ MONITOR │ │ ANALYZER │ │ CALCULATOR │ │ DETECTOR │ │ │
│ │ │ │ │ │ │ │ │ │ │ │
│ │ │ Target: <300μs│ │ Bandwidth: │ │ Target: <0.1%│ │ Max: ±10μs │ │ │
│ │ │ Current: XXXμs│ │ XXX Mbps │ │ Current: X.X%│ │ Current: ±Xμs│ │ │
│ │ │ History: [..] │ │ Utilization: │ │ Trend: ↑↓→ │ │ Pattern: ... │ │ │
│ │ │ Trend: ↑↓→ │ │ XX% │ │ │ │ │ │ │
│ │ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │ │
│ └──────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ADAPTIVE CONTROL LAYER │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ DYNAMIC QUALITY CONTROLLER │ │
│ │ │ │
│ │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ DECISION MATRIX │ │ │
│ │ │ │ │ │
│ │ │ Condition │ Action │ Parameter Change │ │ │
│ │ │ ────────────────────────────────────────────────────────────────────── │ │ │
│ │ │ Latency > 300μs │ Reduce Sample Rate │ 192kHz → 96kHz │ │ │
│ │ │ Error Rate > 1% │ Increase Redundancy │ 2-stream → 3-stream │ │ │
│ │ │ Bandwidth < 80% │ Restore Quality │ 96kHz → 192kHz │ │ │
│ │ │ Jitter > ±20μs │ Adjust Buffer Size │ Buffer += 2ms │ │ │
│ │ │ Prediction Fail │ Force Redundancy │ Disable PNTBTR Predict │ │ │
│ │ │ Network Disconnect │ Reconnect + Resync │ Full Session Restart │ │ │
│ │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│ │ │ │
│ │ Control Loop: Monitor → Analyze → Decide → Act → Verify (100Hz update rate) │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ DIAGNOSTIC & LOGGING LAYER │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ SYSTEM DIAGNOSTICS │ │
│ │ │ │
│ │ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │ │
│ │ │ NETWORK │ │ AUDIO │ │ VIDEO │ │ SYSTEM │ │ │
│ │ │ ANALYZER │ │ ANALYZER │ │ ANALYZER │ │ HEALTH │ │ │
│ │ │ │ │ │ │ │ │ │ │ │
│ │ │ • Packet │ │ • THD+N │ │ • Frame │ │ • CPU Usage │ │ │
│ │ │ Loss Rate │ │ Analysis │ │ Drops │ │ • Memory │ │ │
│ │ │ • RTT Times │ │ • Frequency │ │ • Motion │ │ • Disk I/O │ │ │
│ │ │ • Bandwidth │ │ Response │ │ Artifacts │ │ • Network │ │ │
│ │ │ Usage │ │ • Dynamic │ │ • Sync │ │ Interface │ │ │
│ │ │ • Route │ │ Range │ │ Accuracy │ │ Status │ │ │
│ │ │ Topology │ │ │ │ │ │ │ │ │
│ │ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ REPORTING & ALERTING LAYER │
│ ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ QUALITY REPORTING │ │
│ │ │ │
│ │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ ALERT CONDITIONS │ │ │
│ │ │ │ │ │
│ │ │ Severity │ Condition │ Action │ │ │
│ │ │ ────────────────────────────────────────────────────────────────────── │ │ │
│ │ │ CRITICAL │ Total System Failure │ Emergency Shutdown + Log │ │ │
│ │ │ HIGH │ Latency > 1000μs │ Automatic Quality Reduction │ │ │
│ │ │ MEDIUM │ Error Rate > 5% │ Increase Error Correction │ │ │
│ │ │ LOW │ Bandwidth Usage > 90% │ Optimize Compression │ │ │
│ │ │ INFO │ New Client Connection │ Log Session Details │ │ │
│ │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│ │ │ │
│ │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ PERFORMANCE DASHBOARD │ │ │
│ │ │ │ │ │
│ │ │ ┌─ Latency ────┐ ┌─ Throughput ─┐ ┌─ Quality ────┐ ┌─ Sessions ──┐ │ │ │
│ │ │ │ XXXμs │ │ XXX.X Mbps │ │ XX.X% │ │ X Active │ │ │ │
│ │ │ │ ████████▌ │ │ ██████████▌ │ │ ██████████▌ │ │ Client List │ │ │ │
│ │ │ │ Target: 300μs│ │ Max: 1Gbps │ │ Target: 99% │ │ Bandwidth │ │ │ │
│ │ │ └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘ │ │ │
│ │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────────────┘ │
│ │ └────────────────────────────────────────────────────────────────────────────────────────────┘
Technical Implementation Summary for Patent Documentation
Core Innovation Claims:
1. PNTBTR Algorithm: Novel three-tier error recovery prioritizing high sample rates with redundancy over prediction
2. TOAST Protocol: Fire-and-forget UDP transport optimized for multimedia streaming with <300μs total latency
3. Triple-Domain Architecture: Parallel processing of MIDI (JMID), audio (JDAT), and video (JVID) streams with unified clock synchronization
4. Dynamic Quality Control: Real-time adaptive throttling maintaining quality while preserving ultra-low latency
5. Hierarchical Session Management: Distributed client architecture with automatic failover and load balancing
Performance Specifications:
- End-to-End Latency: <300μs for complete multimedia transmission (local connection)
- Clock Synchronization: <25μs deviation across all domains and clients
- Error Recovery: <50μs for PNTBTR reconstruction algorithms
- Bandwidth Efficiency: 192kHz audio processing with dynamic sample rate adaptation
- Scalability: Support for 16 concurrent jam sessions with 3-8 participants each per server instance
- Network Topology: Each participant connects as a single network node with multiple domain streams
- Stream Architecture: Per-participant parallel streams for MIDI/Audio/Video domains
Patent Classification Domains:
- H04L 12/28 (Local area networks; packet switching)
- H04L 29/06 (Real-time protocol data unit transmission)
- G10H 1/00 (Electronic musical instruments; distributed music systems)
- H04N 21/00 (Multimedia streaming over IP networks)
- H04L 1/00 (Error detection and correction in digital communications)
This comprehensive technical architecture provides the foundation for patent applications covering the JAMNet ecosystem's innovative approaches to ultra-low latency multimedia streaming, predictive error
recovery, and distributed session management.
5. Detailed Description of the Preferred Embodiments
5.0 JAMNet Overview
JAMNet is a distributed multimedia protocol that allows real-time collaboration of MIDI, audio, and video data
using JAM (compact JSONL). The protocol is DAW-agnostic, OS-agnostic, and stream-centric. Every
multimedia stream is encoded in JAM lines and transmitted via UDP multicast using TOAST.
5.1 JAM
JAM (Joint Audio Multicast)
The JSONL Streaming Format Engine
JAM.js (Joint Audio Multicast) is JAMNet's custom fork of Bassoon.js that serves as the core JSONL (JSON
Lines) streaming format for real-time multimedia transmission within the TOAST protocol ecosystem.
5.2 Transport Layer: TOAST
TOAST (Transport Oriented Audio Synchronization Tunnel) is a UDP-based multicast protocol that carries
all JAMNet multimedia data. TOAST does not use handshakes, acknowledgments, or retransmissions. It is
optimized for minimal jitter and sub-millisecond delivery across local networks ; sub 20ms in metro areas.
TOAST routes sessions using "session URLs" (e.g., session://jam-session-1/audio). It supports lock-
free pub/sub management and can dynamically throttle based on receiver bandwidth.
5.3 Audio Format: JDAT
JDAT encodes PCM audio into compact JAM chunks.
JAMNet uses JDAT to transmit one mono audio signal at 192kHz (total combined) sampling with parallel
redundancy via the JELLIE encoder.
Each audio slice is encoded like:
{"t":"aud","id":"jdat","seq":142,"r":192000,"ch":0,"red":1,"d":
[0.0012,0.0034,-0.0005]}
• r is the sample rate
• r is the sample rate
• ch is the channel
• red indicates whether redundancy is active
• d is the array of samples
JDAT simulates the behavior of ADAT by using four parallel 96 khz JAM streams for each mono signal:
•
Stream 0: even samples
•
Stream 1: odd samples
•
Stream 2–3: redundancy and parity encoding
5.4 JELLIE: JAM Embedded Low-Latency Instrument Encoding
JELLIE is the encoder used in JDAT to produce high-fidelity PCM streams across 4 parallel multicast lines. It
embeds timing, channel data, and parity for immediate failover recovery.
Each mono audio input is encoded into:
•
Two time-interleaved JSONL audio streams (even/odd)
•
Two redundancy streams for immediate recovery
This enables sustained studio quality, low latency mono transmission with zero packet recovery logic. When
packets are lost, the receiver re-assembles from redundancy or (if required) calls on PNTBTR to predict
missing values.
JELLIE prioritizes redundancy over prediction, enabling musical integrity even in moderate packet loss
environments. It is ideal for transmitting guitar, vocal, and other mono-line signals.
5.5 MIDI Format: JMID
JMID encodes MIDI events using compact JSONL messages. Each message includes:
•
Event type (t)
•
Note or controller number (n)
•
Velocity or value (v)
•
Velocity or value (v)
•
Channel (c)
•
Timestamp (ts)
Example:
{"t":"n+","n":60,"v":100,"c":1,"ts":1234567890}
Multicast distribution allows a single MIDI stream to control multiple virtual instruments, DAWs, or processors
with sub-30µs latency and real-time reaction.
5.6 Video Format: JVID
Each frame is encoded in compact JSONL with optional JAMCam metadata for:
•
Face detection
•
Framing
•
Light normalization
•
Motion estimation
{
"t": "vid",
"id": "jvid",
"seq": 89,
"res": "ULTRA_LOW_72P",
"fps": 60,
"jc": {
"face": true,
"frame": true,
"light": 0.85
},
"d": "base64_frame"
}
Frames are multicast using TOAST at up to 60fps with <250µs end-to-end latency.
5.7 Recovery System: PNTBTR
PNTBTR (Predictive Network Temporal Buffered Transmission Recovery) is an adaptive fallback strategy
when all redundancy and throttling is exhausted.
It follows three tiers:
1 Redundancy-first recovery via JDAT's backup streams
2 Dynamic throttling down to 44.1kHz for audio (or lower frame rate for video)
2 Dynamic throttling down to 44.1kHz for audio (or lower frame rate for video)
3 Prediction fallback to fill in transient gaps in waveform or frame data
PNTBTR is domain-specific:
•
For MIDI: event interpolation
•
For audio: waveform continuity modeling
•
For video: motion prediction and smoothing
No retransmission occurs. JAMNet prioritizes flow over exactness: better to guess and continue than to wait
and stall.
6. Informal Claims (for Provisional Filing)
1 A system for transmitting real-time multimedia data using JSONL over UDP multicast.
2 A method of encoding audio into four parallel JSONL streams representing even, odd, and redundant data
packets.
3 An encoder configured to stream mono audio at 192kHz across multicast JSONL with embedded failover
parity.
4 A transport protocol designed for fire-and-forget multimedia distribution over session-based routing.
5 A fallback system that predicts lost multimedia data without retransmission.
6 An integrated ecosystem combining real-time MIDI, audio, and video collaboration across devices using
universal, human-readable, AI native formats.
7 A transport oriented synchronization mechanism enabling sub-millisecond processing latency across three
multimedia domains (MIDI, audio, video).
8 A truly universal framework that eliminates binary format dependency in professional audio/video
transport.