# JDAT Framework

**JAM Embedded Low Latency Instrument Encoding (JELLIE) for Audio Streaming**

JDAT is the audio streaming counterpart to JMID, implementing a revolutionary approach to real-time audio transmission using JSON-based messaging over UDP with TOAST protocol and PNTBTR recovery.

## 🎯 Overview

The JDAT Framework enables:

- **Ultra-low latency audio streaming** (sub-20ms total latency)
- **192kHz simulation** using innovative ADAT channel interleaving
- **PNTBTR recovery** for intelligent gap filling during packet loss
- **Pure JSON protocol** - human-readable and platform-agnostic
- **TOAST transport** - optimized UDP-based delivery
- **Real-time redundancy** for mission-critical audio applications

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│ JELLIE Encoder  │───▶│ TOAST Transport │
│   (Mono PCM)    │    │                 │    │   (UDP/JSON)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▼
                       ┌─────────────────┐
                       │ JDAT Format │
                       │ {               │
                       │   "samples": [] │
                       │   "rate": 96000 │
                       │   "channel": 0  │
                       │ }               │
                       └─────────────────┘
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Audio Output   │◀───│ JELLIE Decoder  │◀───│ TOAST Transport │
│   (Recovered)   │    │   + PNTBTR      │    │   (UDP/JSON)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Key Features

### 192kHz Strategy (x2 Hijack)

- Uses ADAT's 4-channel capability to achieve 192kHz effective sampling
- Stream 1: samples 0, 4, 8, ... (even samples)
- Stream 2: samples 2, 6, 10, ... (odd samples, offset)
- Streams 3-4: Redundancy and parity data
- Reconstructs full 192kHz stream at receiver

### PNTBTR Recovery

- **Predictive**: Analyzes audio patterns for intelligent gap filling
- **Network**: Handles UDP packet loss gracefully
- **Temporal**: Maintains timing synchronization
- **Buffered**: Smart buffering strategies
- **Transmission**: Fire-and-forget UDP with recovery
- **Recovery**: Multiple prediction algorithms (LPC, harmonic synthesis, pattern matching)

### JSON Format

```json
{
  "type": "audio",
  "id": "jdat",
  "seq": 142,
  "timestamp": 1640995200000000,
  "session_id": "uuid-session-id",
  "message_id": "uuid-message-id",
  "data": {
    "samples": [0.0012, 0.0034, -0.0005, ...],
    "sample_rate": 96000,
    "channel": 0,
    "frame_size": 480,
    "redundancy_level": 1,
    "is_interleaved": false,
    "offset_samples": 0
  }
}
```

## 📁 Project Structure

```
JDAT_Framework/
├── include/                    # Header files
│   ├── JDATMessage.h      # Core message format
│   ├── JELLIEEncoder.h        # Audio encoder
│   ├── JELLIEDecoder.h        # Audio decoder
│   ├── ADATSimulator.h        # 192k ADAT strategy
│   ├── WaveformPredictor.h    # PNTBTR implementation
│   ├── AudioBufferManager.h   # Lock-free audio buffering
│   └── LockFreeQueue.h        # High-performance queues
├── src/                       # Implementation files
├── schemas/                   # JSON schemas
│   └── jdat-message.schema.json
├── examples/                  # Example applications
│   ├── basic_jellie_demo.cpp  # Basic encoding/decoding
│   └── adat_192k_demo.cpp     # 192k strategy demo
├── tests/                     # Unit tests
├── benchmarks/               # Performance tests
└── CMakeLists.txt           # Build configuration
```

## 🛠️ Building

### Prerequisites

- CMake 3.20+
- C++20 compatible compiler
- nlohmann/json library
- simdjson library (for performance)

### Build Steps

```bash
# Clone and build
git clone <repository>
cd JDAT_Framework

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Run tests
make test

# Run examples
./examples/basic_jellie_demo
./examples/adat_192k_demo
```

## 📖 Usage Examples

### Basic Audio Streaming

```cpp
#include "JELLIEEncoder.h"
#include "JELLIEDecoder.h"

// Create encoder
JELLIEEncoder::Config config;
config.sample_rate = SampleRate::SR_96000;
config.quality = AudioQuality::HIGH_PRECISION;
auto encoder = std::make_unique<JELLIEEncoder>(config);

// Set message callback
encoder->setMessageCallback([](const JDATMessage& msg) {
    // Send over network using TOAST
    sendOverNetwork(msg.toString());
});

// Process audio
std::vector<float> audio_samples = getAudioInput();
encoder->processAudio(audio_samples);
```

### 192kHz Mode

```cpp
// Enable 192k mode
config.enable_192k_mode = true;
config.enable_adat_mapping = true;
config.redundancy_level = 2;

auto encoder = create192kEncoder("session-id");
auto decoder = create192kDecoder();

// The encoder automatically creates interleaved streams
// The decoder reconstructs the full 192k signal
```

### PNTBTR Recovery

```cpp
// Decoder with PNTBTR enabled
JELLIEDecoder::Config decoder_config;
decoder_config.enable_pntbtr = true;
decoder_config.max_recovery_gap_ms = 20;  // Recover up to 20ms gaps

auto decoder = std::make_unique<JELLIEDecoder>(decoder_config);

// Set recovery callback to monitor gap filling
decoder->setRecoveryCallback([](uint64_t gap_start, uint64_t gap_end, uint32_t samples) {
    std::cout << "Recovered " << samples << " samples" << std::endl;
});
```

## 🎛️ Configuration Options

### Sample Rates

- **48kHz**: Standard audio rate
- **96kHz**: High-quality audio (recommended)
- **192kHz**: Ultra-high quality (via interleaving strategy)

### Quality Levels

- **HIGH_PRECISION**: 32-bit float samples (recommended)
- **STANDARD**: 24-bit samples
- **COMPRESSED**: 16-bit samples with compression

### Redundancy Levels

- **Level 1**: No redundancy (minimum latency)
- **Level 2**: Single redundant stream
- **Level 3**: Dual redundancy
- **Level 4**: Maximum redundancy (ADAT full utilization)

## 📊 Performance Characteristics

### Latency Targets

- **Encoding latency**: < 2ms
- **Network transmission**: < 5ms (LAN)
- **Decoding latency**: < 3ms
- **Total end-to-end**: < 15ms (LAN)

### Throughput

- **96kHz mono**: ~12 Mbps (uncompressed JSON)
- **192kHz effective**: ~24 Mbps (via interleaving)
- **With redundancy**: Scales linearly with redundancy level

### Recovery Performance

- **Gap detection**: < 1ms
- **Prediction latency**: < 500μs
- **Recovery accuracy**: > 95% for gaps < 10ms

## 🔧 Advanced Features

### ADAT Channel Mapping

Maps mono audio to 4-channel ADAT structure:

- **Channel 0**: Primary stream (even samples for 192k)
- **Channel 1**: Interleaved stream (odd samples for 192k)
- **Channel 2**: Parity/redundancy stream
- **Channel 3**: Additional redundancy/prediction

### Waveform Prediction Methods

- **Linear Prediction Coding (LPC)**: Mathematical extrapolation
- **Harmonic Synthesis**: Frequency-domain reconstruction
- **Pattern Matching**: Repetitive pattern detection
- **Spectral Matching**: Spectral envelope continuation
- **Zero Crossing Optimization**: Phase-coherent gap filling

### Lock-Free Architecture

- **Single Producer Single Consumer (SPSC)** queues for encoder
- **Multi Producer Single Consumer (MPSC)** queues for decoder
- **Cache-aligned** data structures
- **Memory order optimized** atomic operations

## 🧪 Testing

### Unit Tests

```bash
cd build
make test
```

### Performance Benchmarks

```bash
./benchmarks/audio_encoding_benchmark
./benchmarks/pntbtr_recovery_benchmark
./benchmarks/network_simulation_benchmark
```

### Real-time Testing

```bash
./tests/realtime_latency_test
./tests/packet_loss_simulation
./tests/192k_reconstruction_test
```

## 📈 Roadmap

### Phase 1 (Current)

- [x] Core JDAT message format
- [x] Basic JELLIE encoder/decoder
- [x] ADAT simulator framework
- [x] PNTBTR predictor structure
- [x] Lock-free audio buffering

### Phase 2

- [ ] Complete encoder/decoder implementation
- [ ] PNTBTR algorithm implementation
- [ ] Network integration with TOAST
- [ ] Comprehensive testing suite

### Phase 3

- [ ] Advanced prediction algorithms
- [ ] Machine learning integration
- [ ] Multi-channel support
- [ ] Hardware acceleration

## 🤝 Integration with JAMNet

JDAT is a core component of the JAMNet multimedia streaming ecosystem:

| Component       | MIDI Stack      | Audio Stack        |
| --------------- | --------------- | ------------------ |
| **Protocol**    | JMID        | JDAT           |
| **Transport**   | TOAST/UDP       | TOAST/UDP          |
| **Recovery**    | PNTBTR (events) | PNTBTR (waveforms) |
| **Application** | TOASTer         | JELLIE             |

Both systems run simultaneously in JAMNet for complete audio+MIDI streaming.

## 📄 License

[License information to be added]

## 🆘 Support

For questions, issues, or contributions:

- Create an issue in the repository
- Refer to the examples for usage patterns
- Check the test suite for expected behavior

---

**JDAT Framework** - Revolutionizing real-time audio streaming with JSON clarity and UDP performance.
