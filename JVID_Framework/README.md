# JVID Framework

Ultra-low latency JSON video streaming framework for real-time applications.

## Overview

JVID is the video counterpart to JDAT, designed for streaming compressed visual data as **stateless JSON objects** with frame-by-frame precision over **fire-and-forget UDP multicast**. It prioritizes ultra-low latency over quality, targeting <300μs end-to-end video transmission latency for real-time collaborative applications.

### Core UDP GPU Fundamentals

**JVID embodies JAMNet's stateless, optimized, fire-and-forget architecture:**

#### **Stateless Frame Design**
- **Self-Contained Messages**: Every frame JSON is complete and independent - no dependencies on previous frames
- **Sequence-Based Recovery**: Lost frames don't break the stream - next frame can be processed immediately  
- **No Session State**: Zero connection state, handshake overhead, or acknowledgment tracking
- **Independent Decode**: Any frame can be decoded without prior frame data

#### **Fire-and-Forget UDP Multicast**
- **No Handshakes**: Eliminates TCP connection establishment (~3ms saved per stream)
- **No Acknowledgments**: Zero waiting for delivery confirmation or retransmission requests
- **No Retransmission**: Lost packets are never requested again - the show must go on
- **Multicast Efficiency**: Single video transmission reaches unlimited receivers simultaneously

#### **GPU-Accelerated Processing**
- **Memory-Mapped Buffers**: Zero-copy video frame processing from capture to GPU
- **Parallel Frame Processing**: GPU threads handle capture, compression, and JSON encoding simultaneously
- **Hardware Acceleration**: Leverages GPU for scaling, color conversion, and JPEG encoding
- **Lock-Free Pipelines**: Lockless producer-consumer patterns for maximum video throughput

## Key Features

### 🎥 JAMCam Video Processing

- **Ultra-low latency encoding**: Target <200μs encode time
- **Adaptive quality control**: Dynamic quality adjustment based on latency constraints
- **Face detection and auto-framing**: Intelligent camera positioning
- **Lighting normalization**: Automatic exposure and color correction
- **Multi-camera support**: Stream ID management for multiple sources

### 🚀 Performance Optimization

- **Lock-free video buffers**: Zero-copy frame processing where possible
- **SIMD acceleration**: AVX2/SSE4.2 optimized video operations
- **GPU encoding support**: Hardware acceleration for compatible devices
- **Predictive frame dropping**: Drop frames that would exceed latency budget

### 🔧 PNBTR Video Recovery

- **Frame prediction**: Motion-compensated frame reconstruction
- **Temporal interpolation**: Smooth playback despite packet loss
- **Adaptive prediction**: Quality-based prediction confidence
- **Reference frame management**: Optimal reference selection for prediction

### 🌐 TOAST Integration

- **Unified transport**: Shares TOAST protocol with JMID and JDAT
- **Message type routing**: VIDEO, KEYFRAME, CONTROL, SYNC message types
- **Clock synchronization**: Frame timing aligned with audio streams
- **JSON schema validation**: Structured video message format

## Latency Targets

| Component             | Target Latency | Description                  |
| --------------------- | -------------- | ---------------------------- |
| Video Encode          | <200μs         | Capture to JSON message      |
| Video Decode          | <100μs         | JSON message to display      |
| **Total Video**       | **<300μs**     | **End-to-end video latency** |
| Network Transit       | ~100μs         | LAN UDP transmission         |
| **Complete Pipeline** | **<400μs**     | **Including network**        |

_Compare to traditional video streaming: 500ms-3000ms typical latency_

## 📚 Detailed Implementation Guide

For a comprehensive, step-by-step guide to implementing JVID's stateless UDP GPU fundamentals, see **[adapt.md](adapt.md)** - it provides detailed instructions for:

- **Live Camera Streaming**: Complete implementation of 300×400 color feed over JVID
- **GPU Optimization**: Memory-mapped buffers and GPU-accelerated processing
- **Fire-and-Forget UDP**: Detailed explanation of stateless multicast transmission
- **JSON Frame Encoding**: Frame-by-frame JSON structure and packetization
- **Real-Time Performance**: Achieving <300μs latency with practical optimizations

**The `adapt.md` file is essential reading for understanding JVID's core architecture.**

## Resolution Presets

| Preset        | Resolution | Typical Frame Size | Encode Time | Use Case        |
| ------------- | ---------- | ------------------ | ----------- | --------------- |
| ULTRA_LOW_72P | 128×72     | ~1KB               | ~150μs      | Minimal latency |
| LOW_144P      | 256×144    | ~4KB               | ~300μs      | Balanced        |
| MEDIUM_240P   | 426×240    | ~8KB               | ~600μs      | Good quality    |
| HIGH_360P     | 640×360    | ~16KB              | ~1200μs     | High quality    |

## Quick Start

### Build Requirements

```bash
# Dependencies
- CMake 3.16+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- nlohmann/json 3.9.0+
- Threading support

# Optional
- CUDA toolkit (GPU acceleration)
- AVX2/SSE4.2 support (SIMD optimization)
- Platform video APIs (CoreVideo/V4L2)
```

### Build Instructions

```bash
# Clone repository
git clone <repository_url>
cd JVID_Framework

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DJVID_ENABLE_GPU=ON \
         -DJVID_ENABLE_SIMD=ON

# Build framework
make -j$(nproc)

# Run tests (optional)
make test

# Install (optional)
sudo make install
```

### Basic Usage Example

```cpp
#include "JVIDMessage.h"
#include "JAMCamEncoder.h"
#include "JAMCamDecoder.h"

using namespace jvid;

// Configure ultra-low latency encoder
JAMCamEncoder::Config encoder_config;
encoder_config.resolution = VideoResolution::LOW_144P;
encoder_config.quality = VideoQuality::FAST;
encoder_config.target_latency_us = 300;
encoder_config.enable_face_detection = true;

auto encoder = std::make_unique<JAMCamEncoder>(encoder_config);

// Set frame callback
encoder->setFrameCallback([](const JVIDMessage& frame) {
    // Transmit frame via TOAST transport
    toast_transport->send(frame.toCompactJSON());
});

// Start capturing
encoder->start();

// Configure decoder
JAMCamDecoder::Config decoder_config;
decoder_config.target_latency_us = 300;
decoder_config.enable_frame_prediction = true;

auto decoder = std::make_unique<JAMCamDecoder>(decoder_config);

// Set display callback
decoder->setFrameReadyCallback([](const VideoBufferManager::FrameBuffer* frame) {
    // Render frame to display
    renderFrame(frame);
});

// Start decoding
decoder->start();

// Process incoming video messages
decoder->processMessage(received_message);
```

## Configuration Options

### Encoder Configuration

```cpp
JAMCamEncoder::Config config;

// Basic settings
config.resolution = VideoResolution::LOW_144P;    // Video resolution preset
config.quality = VideoQuality::FAST;              // Encoding quality/speed trade-off
config.target_fps = 15;                           // Target framerate
config.target_latency_us = 300;                   // Target encode latency

// JAMCam features
config.enable_face_detection = true;              // Face detection
config.enable_auto_framing = true;                // Automatic framing
config.enable_lighting_norm = true;               // Lighting normalization

// Performance tuning
config.enable_gpu_encoding = true;                // GPU acceleration
config.enable_zero_copy = true;                   // Zero-copy processing
config.enable_frame_dropping = true;              // Drop late frames
config.max_encode_time_us = 500;                  // Max encoding time

// Quality settings
config.jpeg_quality = 60;                         // JPEG quality (0-100)
config.enable_chroma_subsampling = true;          // 4:2:0 subsampling
config.adaptive_quality = true;                   // Dynamic quality adjustment
```

### Decoder Configuration

```cpp
JAMCamDecoder::Config config;

// Basic settings
config.target_latency_us = 300;                   // Target decode latency
config.max_frame_age_us = 33333;                  // Drop frames older than 30fps
config.display_width = 256;                       // Target display width
config.display_height = 144;                      // Target display height

// PNBTR recovery
config.enable_frame_prediction = true;            // Frame prediction recovery
config.enable_interpolation = true;               // Frame interpolation
config.prediction_confidence_threshold = 128;     // Min prediction confidence

// Performance
config.enable_gpu_decoding = true;                // GPU acceleration
config.enable_adaptive_quality = true;            // Quality adaptation
config.max_decode_time_us = 500;                  // Max decode time

// Synchronization
config.enable_audio_sync = true;                  // A/V synchronization
config.sync_tolerance_us = 1000;                  // Sync tolerance window
```

## Architecture

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   JAMCam        │    │   JVID        │    │   JAMCam        │
│   Encoder       │───▶│   Message        │───▶│   Decoder       │
│                 │    │                  │    │                 │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Video Capture │    │ • Frame Data     │    │ • Frame Decode  │
│ • Face Detection│    │ • Motion Vectors │    │ • PNBTR Predict │
│ • Auto Framing  │    │ • Timing Info    │    │ • Display Render│
│ • Compression   │    │ • JAMCam Features│    │ • A/V Sync      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         └────────────────────────┼───────────────────────┘
                                  │
                    ┌──────────────▼──────────────┐
                    │      TOAST Transport        │
                    │   (Shared with JMID)    │
                    └─────────────────────────────┘
```

### Data Flow

1. **Capture**: Camera/screen capture via platform APIs
2. **Process**: JAMCam features (face detection, auto-framing, lighting)
3. **Encode**: Ultra-fast compression (JPEG/WebP/raw pixels)
4. **Packetize**: JSON message construction with timing metadata
5. **Transmit**: TOAST transport protocol over UDP
6. **Receive**: Message validation and parsing
7. **Recover**: PNBTR prediction for missing frames
8. **Decode**: Frame reconstruction and format conversion
9. **Display**: Render to screen/canvas with A/V sync

## Performance Tuning

### Ultra-Low Latency Mode

```cpp
// Minimal latency configuration
auto config = JAMCamEncoder::getOptimalConfig(200); // 200μs target
config.resolution = VideoResolution::ULTRA_LOW_72P;
config.quality = VideoQuality::ULTRA_FAST;
config.enable_frame_dropping = true;
config.max_encode_time_us = 300;
config.enable_gpu_encoding = true;
config.enable_zero_copy = true;
```

### Quality vs Latency Balance

```cpp
// Balanced configuration
auto config = JAMCamEncoder::getOptimalConfig(500); // 500μs target
config.resolution = VideoResolution::LOW_144P;
config.quality = VideoQuality::FAST;
config.adaptive_quality = true;
config.enable_face_detection = true;
config.enable_auto_framing = true;
```

### Network Optimization

```cpp
// Optimize for packet loss
JAMCamDecoder::Config decoder_config;
decoder_config.enable_frame_prediction = true;
decoder_config.prediction_confidence_threshold = 64; // Lower threshold
decoder_config.enable_interpolation = true;
decoder_config.max_frame_age_us = 50000; // 20fps tolerance

FramePredictor::Config predictor_config;
predictor_config.max_reference_frames = 6;
predictor_config.enable_motion_estimation = true;
predictor_config.enable_parallel_prediction = true;
```

## Integration with JDAT/MIDI

### Unified Clock Synchronization

```cpp
// Synchronize video with audio timestamps
uint64_t audio_timestamp = adat_message.timing_info.sample_timestamp_us;
decoder->setAudioTimestamp(audio_timestamp);

// Align video frames with MIDI events
uint64_t midi_timestamp = midi_message.timestamp_us;
video_frame.timing_info.audio_sync_timestamp = midi_timestamp;
```

### Shared TOAST Transport

```cpp
// Route messages by type
switch (message_type) {
    case JMID_MESSAGE:
        midi_processor->process(message);
        break;
    case JDAT_MESSAGE:
        audio_processor->process(message);
        break;
    case JVID_MESSAGE:
        video_processor->process(message);
        break;
}
```

## Examples and Demos

### Basic Demo

```bash
# Run basic JAMCam demo
./build/examples/jamcam_basic_demo --resolution 144p --quality fast --duration 30

# With face detection and auto-framing
./build/examples/jamcam_basic_demo --resolution 240p --fps 20 --face-detection --auto-framing

# Ultra-low latency mode
./build/examples/jamcam_basic_demo --resolution 72p --quality ultra_fast --fps 30
```

### Performance Benchmark

```bash
# Benchmark encoding performance
./build/benchmarks/encode_benchmark --resolution all --quality all --iterations 1000

# Benchmark with packet loss simulation
./build/benchmarks/recovery_benchmark --packet-loss 0.1 --prediction-enabled
```

## Troubleshooting

### Common Issues

1. **High Encoding Latency**

   - Reduce resolution or quality
   - Enable GPU acceleration
   - Disable face detection for minimal latency

2. **Frame Drops**

   - Increase max_encode_time_us threshold
   - Reduce target FPS
   - Check CPU/GPU utilization

3. **Poor Prediction Quality**

   - Increase reference frame count
   - Lower motion threshold
   - Enable temporal smoothing

4. **Audio/Video Sync Issues**
   - Check clock synchronization
   - Adjust sync tolerance
   - Verify timestamp alignment

### Debug Options

```cpp
// Enable verbose logging
JAMCamEncoder::Config config;
config.enable_debug_logging = true;
config.log_frame_timing = true;

// Performance monitoring
encoder->setStatsCallback([](const VideoStreamStats& stats) {
    std::cout << "Encode latency: " << stats.average_encode_time_us << "μs" << std::endl;
    std::cout << "Frame rate: " << stats.current_fps << " fps" << std::endl;
    std::cout << "Bitrate: " << stats.bandwidth_kbps << " kbps" << std::endl;
});
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built on the TOAST transport protocol foundation
- Inspired by ultra-low latency gaming and VR applications
- Leverages modern C++17 performance optimizations
- Designed for real-time collaborative music and multimedia
