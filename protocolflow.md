         [JAMWAN Node] <---> [Other JAMWAN Nodes]
              ↑      ↑
              |      |
        [TOAST UDP Transport Layer]
              ↑      ↑
      [JELLIE]     [MIDIp2p]
         (Audio)        (MIDI)
              ↑      ↑
             /        \
            /          \

[User A] [User B] [User C] [User D]
\ | | /
\ | | /
\------[User Sync Hub]------/
↓
[PNTBTR Auto-Throttle + Recovery]
↓
[Waveform + MIDI Prediction Layer]

## Protocol Flow Overview (JAMNet v0.9)

This document outlines the core architecture of the JAMNet protocol stack, preparing the system for Fly.io deployment with edge-routed, UDP-based real-time streaming infrastructure.

### [JAMWAN Node] <---> [Other JAMWAN Nodes]

JAMWAN Nodes are globally distributed edge instances deployed via Fly.io. Each node handles localized traffic for its connected users and synchronizes with sibling nodes across the network for redundancy and roaming. Nodes communicate using UDP over the TOAST transport layer.

### TOAST (Transport Over Asynchronous Streaming Transport)

TOAST is the custom UDP packet layer responsible for delivering time-critical streaming packets. It includes connection handshakes, clock drift correction, and packet scheduling. TOAST replaces TCP entirely to prioritize speed over guaranteed delivery, relying on higher layers (PNTBTR) to handle recovery.

### JELLIE and MIDIp2p

- **JELLIE (JAM Embedded Low-Latency Instrument Encoding)** is the audio transport protocol built on JDAT. It streams mono audio over 4-channel interleaving style channel representations using JSON formatting, enabling platform-independent encoding without raw binary.
- **MIDIp2p** is the MIDI transport layer built on JMID. It handles channel-separated MIDI events as JSON packets, supporting transport sync, CC, and note events in real time.

### User Sync Hub

All connected users synchronize through their regional JAMWAN Node. The User Sync Hub manages timing alignment and jitter compensation across **multi-user peer groups**. With 4 users (e.g., A, B, C, D), the hub ensures everyone is locally synced to the node’s clock and cross-synced to each other via broadcast deltas and corrective nudging.

- Each user maintains a local clock offset table for the other participants.
- Packet arrival deltas are used to estimate individual jitter.
- Corrections are distributed by the JAMWAN Node every few milliseconds.
- Late joiners receive a time-aligned snapshot of the current jam state.

This system keeps all participants musically synced in real time, regardless of individual device or network variability.

### PNTBTR (Predictive Network Temporal Buffered Transmission Recovery)

PNTBTR is the intelligent fallback mechanism. It automatically throttles between 192k, 96k, 48k, and 44.1k sample rates based on real-time network throughput. In case of data loss, it performs waveform prediction using tokenized JDAT to synthesize audio for up to 50ms, making dropouts inaudible.

### Waveform + MIDI Prediction Layer

Final stage fallback used only when multiple packet losses are detected. Predicts the next few frames of both waveform and MIDI data to maintain groove and timing even across brief network interruptions.

---

Next step: Configure your Fly.io `fly.toml`, set up UDP port forwarding, and deploy the first JAMWAN instance.
