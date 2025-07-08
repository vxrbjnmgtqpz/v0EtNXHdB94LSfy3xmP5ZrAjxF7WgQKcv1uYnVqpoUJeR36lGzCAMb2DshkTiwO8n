# JAMCam - JVID Video Streaming Test Application

JAMCam is a standalone Electron application designed to test and validate the JVID framework's video streaming capabilities over the TOAST protocol. It provides a practical implementation for testing video throughput, latency, and protocol performance.

## Features

### 🎥 **Dual Mode Operation**
- **Transmit Mode**: Captures and encodes video frames, streams via JVID protocol
- **Receive Mode**: Receives and decodes JVID video streams from remote transmitters

### 📊 **Real-time Performance Monitoring**
- Frame rate (FPS) tracking
- Encode/decode latency measurement  
- End-to-end latency calculation
- Packet loss detection
- Bitrate monitoring
- Frame sequence validation

### ⚡ **JVID Protocol Implementation**
- Complete JVID message structure
- Base64 JPEG frame encoding
- Microsecond timestamp precision
- Session-based streaming
- WebSocket transport (TOAST protocol simulation)

### 🎛️ **Configurable Settings**
- Adjustable server port and target FPS
- Custom server address for receiver mode
- Session ID management
- Connection timeout configuration

## Quick Start

### Installation

```bash
cd JAMCam
npm install
```

### Development Mode

```bash
npm run dev
```

### Production Build

```bash
npm run build
```

## Usage

### Transmitter Setup
1. Select **Transmit** mode
2. Configure server port (default: 8080)
3. Set target FPS (default: 15)
4. Click **Start Transmitting**
5. JAMCam will generate test frames with sequence numbers and timestamps

### Receiver Setup
1. Select **Receive** mode  
2. Enter transmitter address (e.g., `ws://localhost:8080`)
3. Click **Start Receiving**
4. Incoming video frames will be displayed with latency information

### Testing Workflow
1. Launch two JAMCam instances (or run on separate machines)
2. Configure one as transmitter, one as receiver
3. Start transmitting first, then connect receiver
4. Monitor real-time performance statistics
5. Test different FPS settings and network conditions

## JVID Message Format

JAMCam implements the complete JVID protocol structure:

```json
{
  "t": "vid",
  "id": "jvid", 
  "seq": 12345,
  "sid": "jamcam_session_123",
  "ts": 1641234567890123,
  "vi": {
    "resolution": "LOW_144P",
    "quality": "FAST",
    "format": "BASE64_JPEG",
    "frame_width": 256,
    "frame_height": 144,
    "fps_target": 15,
    "is_keyframe": false,
    "stream_id": 0
  },
  "fd": {
    "frame_base64": "...",
    "compressed_size": 4096,
    "original_size": 110592,
    "compression_ratio": 27.0
  },
  "ti": {
    "capture_timestamp_us": 1641234567890123,
    "encode_timestamp_us": 1641234567890200,
    "send_timestamp_us": 1641234567890250,
    "encode_duration_us": 150,
    "expected_decode_us": 200
  },
  "in": {
    "checksum": 0,
    "is_predicted": false,
    "prediction_confidence": 0
  }
}
```

## Performance Targets

### Latency Goals
- **Encode Time**: < 500μs (Ultra-fast mode)
- **Decode Time**: < 200μs  
- **End-to-End**: < 300μs (local network)
- **Network Transport**: < 50μs (TOAST over UDP)

### Quality Settings
- **ULTRA_LOW_72P**: 128x72 (~1KB frames) - Maximum speed
- **LOW_144P**: 256x144 (~4KB frames) - Balanced performance  
- **MEDIUM_240P**: 426x240 (~8KB frames) - Better quality
- **HIGH_360P**: 640x360 (~16KB frames) - High quality

## Integration with JAMNet

JAMCam serves as a testbed for:

### TOAST Protocol Testing
- Validates JVID message serialization/deserialization
- Tests transport reliability and performance
- Measures protocol overhead

### TOASTer Integration
- Provides video streaming foundation for TOASTer app
- Validates multi-stream synchronization (MIDI + Audio + Video)
- Tests cross-framework compatibility

### PNBTR Development
- Framework for testing frame prediction algorithms
- Packet loss simulation and recovery testing
- Neural network integration for video prediction

## Development Roadmap

### Phase 1: Basic Streaming ✅
- [x] JVID message implementation
- [x] WebSocket transport
- [x] Test frame generation
- [x] Performance monitoring

### Phase 2: Real Video Capture 🔄
- [ ] Camera/screen capture integration
- [ ] Hardware acceleration (where available)  
- [ ] Multiple video sources
- [ ] Real-time compression optimization

### Phase 3: UDP Migration ⏳
- [ ] Replace WebSocket with UDP sockets
- [ ] Implement actual TOAST protocol
- [ ] Multicast support
- [ ] Burst transmission

### Phase 4: Advanced Features ⏳
- [ ] PNBTR frame prediction
- [ ] Multi-stream synchronization
- [ ] TOASTer integration
- [ ] Cross-platform optimization

## Architecture

```
JAMCam Electron App
├── Main Process (Node.js)
│   ├── JAMCamTransmitter
│   │   ├── Frame capture/generation
│   │   ├── JVID encoding
│   │   └── WebSocket server
│   └── JAMCamReceiver  
│       ├── WebSocket client
│       ├── JVID decoding
│       └── Frame display
└── Renderer Process (HTML/JS)
    ├── UI controls and configuration
    ├── Real-time statistics display
    └── Video frame rendering
```

## Testing and Validation

### Automated Testing
```bash
# Run unit tests
npm test

# Performance benchmarks  
npm run benchmark

# Network stress testing
npm run stress-test
```

### Manual Testing Scenarios
1. **Latency Testing**: Measure end-to-end delays
2. **Packet Loss Simulation**: Test with simulated network issues
3. **Load Testing**: High FPS and multiple streams
4. **Cross-Platform**: Windows, macOS, Linux compatibility

## Contributing

JAMCam is part of the JAMNet ecosystem. Contributions should focus on:

- Performance optimization and latency reduction
- Real video capture implementation  
- TOAST protocol integration
- Cross-platform compatibility
- Documentation and testing

## License

MIT License - Part of the JAMNet open source framework.

---

**JAMCam: Making video streaming testable, measurable, and fast.** 🚀
