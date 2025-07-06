# JAM Framework

**UDP GPU JSONL Native TOAST-Optimized Fork of Bassoon.js**

JAM (JSON Audio Multicast) Framework is the core parsing and processing engine for JAMNet. It's a specialized fork of Bassoon.js optimized for:

- **UDP Multicast**: Fire-and-forget networking with TOAST protocol
- **GPU Acceleration**: Vulkan/Metal compute shaders for JSONL parsing
- **Real-time Processing**: Sub-3ms latency with 67% size reduction
- **Native Performance**: C++ implementation with modern standards

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   UDP Multicast │───▶│  JAM Framework   │───▶│  GPU Pipeline   │
│   JSONL Stream  │    │  (Bassoon Fork)  │    │  (Audio/MIDI)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  TOAST Protocol  │
                       │  Optimization    │
                       └──────────────────┘
```

## Core Components

### 1. UDP Parser (`jam_parser.cpp`)
- High-performance JSONL stream parsing
- GPU-accelerated via compute shaders
- TOAST protocol integration
- 67% compression efficiency

### 2. Multicast Manager (`jam_multicast.cpp`)
- UDP multicast group management
- Session discovery and joining
- Fire-and-forget reliability
- Network health monitoring

### 3. GPU Interface (`jam_gpu.cpp`)
- Vulkan/Metal compute shader integration
- Direct GPU memory mapping
- Real-time audio/MIDI processing
- Zero-copy data transfer

### 4. TOAST Optimizer (`jam_toast.cpp`)
- Protocol-level optimizations
- Adaptive compression
- Latency minimization
- Burst logic integration

## Quick Start

```cpp
#include "jam_framework.h"

// Initialize JAM Framework
JAMFramework jam;
jam.initialize({
    .multicast_group = "239.255.77.77",
    .port = 7777,
    .gpu_backend = JAM_GPU_VULKAN,
    .compression_level = JAM_TOAST_OPTIMIZED
});

// Start processing
jam.start_session("session_uuid");

// Process incoming streams
jam.on_audio_stream([](const JAMAudioData& data) {
    // Handle audio data
});

jam.on_midi_stream([](const JAMMIDIData& data) {
    // Handle MIDI data
});
```

## Build Requirements

- **C++17/20** compiler
- **Vulkan SDK** (for GPU acceleration)
- **Metal** (macOS/iOS)
- **CMake 3.20+**
- **libUV** (for networking)

## Performance Targets

- **Latency**: <3ms end-to-end
- **Throughput**: 1000+ concurrent streams
- **Compression**: 67% size reduction
- **GPU Utilization**: >80% efficiency

## Integration with JAMNet

JAM Framework serves as the foundation for:

- **JDAT Framework**: Audio data processing
- **JMID Framework**: MIDI event handling  
- **JVID Framework**: Video streaming
- **PNBTR**: Physics-based audio reconstruction

---

*JAM Framework: The high-performance heart of JAMNet's revolutionary architecture.*
