# JSONVID Framework Initialization Guide

This guide walks through setting up the JSONVID framework for ultra-low latency video streaming.

## Quick Start Setup

### 1. System Requirements

**Minimum Requirements:**

- CPU: Dual-core 2.0GHz+ (quad-core recommended)
- RAM: 4GB+ (8GB recommended for multi-stream)
- Network: Gigabit Ethernet for LAN streaming
- OS: macOS 10.15+, Ubuntu 20.04+, Windows 10+

**Optimal Performance:**

- CPU: 8+ cores with AVX2 support (Intel i7/i9, AMD Ryzen)
- GPU: CUDA-compatible NVIDIA card (optional)
- Network: 10GbE for multiple high-quality streams
- Camera: USB 3.0+ or integrated camera with 720p+ support

### 2. Dependencies Installation

#### macOS

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake ninja nlohmann-json pkg-config

# Optional: CUDA for GPU acceleration
# Download from NVIDIA website and install CUDA toolkit

# Camera support (built-in via AVFoundation)
```

#### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install -y build-essential cmake ninja-build pkg-config

# Install JSON library
sudo apt install -y nlohmann-json3-dev

# Camera support
sudo apt install -y libv4l-dev v4l-utils

# Optional: CUDA support
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
# sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get -y install cuda
```

### 3. Framework Compilation

```bash
# Clone repository
git clone <repository_url>
cd JSONVID_Framework

# Create and enter build directory
mkdir build && cd build

# Configure build (Release mode for production)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DJSONVID_ENABLE_EXAMPLES=ON \
    -DJSONVID_ENABLE_TESTS=ON \
    -DJSONVID_ENABLE_BENCHMARKS=ON \
    -DJSONVID_ENABLE_GPU=ON \
    -DJSONVID_ENABLE_SIMD=ON \
    -GNinja

# Build framework
ninja

# Run tests to verify installation
ninja test

# Optional: Install system-wide
sudo ninja install
```

### 4. Camera Setup and Testing

#### Test Camera Access

```bash
# List available cameras (Linux)
v4l2-ctl --list-devices

# Test camera capture (Linux)
ffplay /dev/video0

# macOS camera test
system_profiler SPCameraDataType

# Test basic video capture
./build/examples/jamcam_basic_demo --resolution 144p --duration 5 --no-packet-loss
```

#### Camera Permissions

**macOS:**

```bash
# Grant camera access to terminal/IDE
# System Preferences > Security & Privacy > Privacy > Camera
# Add Terminal.app or your development environment
```

**Linux:**

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Set camera permissions (if needed)
sudo chmod 666 /dev/video0
```

## Configuration Profiles

### Profile 1: Ultra-Low Latency (Gaming/VR)

**Target: <200μs encoding, <300μs total**

```cpp
// Configuration for minimal latency
JAMCamEncoder::Config ultra_low_config;
ultra_low_config.resolution = VideoResolution::ULTRA_LOW_72P; // 128x72
ultra_low_config.quality = VideoQuality::ULTRA_FAST;
ultra_low_config.target_fps = 30;
ultra_low_config.target_latency_us = 200;
ultra_low_config.max_encode_time_us = 300;

// Disable CPU-intensive features
ultra_low_config.enable_face_detection = false;
ultra_low_config.enable_auto_framing = false;
ultra_low_config.enable_lighting_norm = false;

// Enable performance optimizations
ultra_low_config.enable_gpu_encoding = true;
ultra_low_config.enable_zero_copy = true;
ultra_low_config.enable_frame_dropping = true;
ultra_low_config.adaptive_quality = true;

// Decoder settings
JAMCamDecoder::Config ultra_decoder_config;
ultra_decoder_config.target_latency_us = 100;
ultra_decoder_config.max_decode_time_us = 200;
ultra_decoder_config.enable_frame_prediction = true;
ultra_decoder_config.enable_gpu_decoding = true;
```

### Profile 2: Balanced Quality (Music/Collaboration)

**Target: <500μs encoding, balanced quality**

```cpp
// Balanced configuration
JAMCamEncoder::Config balanced_config;
balanced_config.resolution = VideoResolution::LOW_144P; // 256x144
balanced_config.quality = VideoQuality::FAST;
balanced_config.target_fps = 20;
balanced_config.target_latency_us = 500;
balanced_config.max_encode_time_us = 700;

// Enable JAMCam features
balanced_config.enable_face_detection = true;
balanced_config.enable_auto_framing = true;
balanced_config.enable_lighting_norm = true;

// Quality settings
balanced_config.jpeg_quality = 70;
balanced_config.adaptive_quality = true;
balanced_config.enable_chroma_subsampling = true;

// Decoder settings
JAMCamDecoder::Config balanced_decoder_config;
balanced_decoder_config.target_latency_us = 300;
balanced_decoder_config.enable_frame_prediction = true;
balanced_decoder_config.enable_interpolation = true;
balanced_decoder_config.enable_adaptive_quality = true;
```

### Profile 3: High Quality (Streaming/Recording)

**Target: <1200μs encoding, maximum quality**

```cpp
// High quality configuration
JAMCamEncoder::Config hq_config;
hq_config.resolution = VideoResolution::HIGH_360P; // 640x360
hq_config.quality = VideoQuality::HIGH_QUALITY;
hq_config.target_fps = 15;
hq_config.target_latency_us = 1200;
hq_config.max_encode_time_us = 1500;

// Enable all features
hq_config.enable_face_detection = true;
hq_config.enable_auto_framing = true;
hq_config.enable_lighting_norm = true;
hq_config.enable_motion_estimation = true;

// High quality settings
hq_config.jpeg_quality = 85;
hq_config.enable_chroma_subsampling = false; // Full color
hq_config.adaptive_quality = false; // Consistent quality

// Decoder settings
JAMCamDecoder::Config hq_decoder_config;
hq_decoder_config.target_latency_us = 600;
hq_decoder_config.enable_frame_prediction = true;
hq_decoder_config.enable_interpolation = true;
hq_decoder_config.maintain_aspect_ratio = true;
hq_decoder_config.enable_upscaling = true;
```

## Network Configuration

### LAN Setup (Recommended)

```cpp
// TOAST transport configuration for LAN
struct TOASTConfig {
    std::string local_ip = "192.168.1.100";
    uint16_t local_port = 8080;
    std::string remote_ip = "192.168.1.101";
    uint16_t remote_port = 8080;

    // Ultra-low latency UDP settings
    bool enable_nodelay = true;
    uint32_t send_buffer_size = 1024 * 1024;    // 1MB
    uint32_t recv_buffer_size = 1024 * 1024;    // 1MB

    // Video-specific settings
    uint8_t video_message_type = 0x10;          // VIDEO_FRAME
    uint8_t keyframe_message_type = 0x11;       // VIDEO_KEYFRAME
    uint32_t max_packet_size = 1400;            // MTU-safe
};
```

### Network Optimization

```bash
# Linux: Optimize network stack for low latency
echo 'net.core.rmem_default = 1048576' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_default = 1048576' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.netdev_max_backlog = 5000' | sudo tee -a /etc/sysctl.conf

# Apply settings
sudo sysctl -p

# Disable CPU power management for consistent performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## TOAST Integration Setup

### 1. Configure Unified Transport

```cpp
// Initialize TOAST transport for all protocols
#include "../JSONMIDI_Framework/include/TOASTTransport.h"

class UnifiedTransport {
public:
    bool initialize() {
        // Initialize TOAST transport
        toast_config_.local_port = 8080;
        toast_config_.enable_video = true;
        toast_config_.enable_audio = true;
        toast_config_.enable_midi = true;

        toast_transport_ = std::make_unique<TOASTTransport>(toast_config_);
        return toast_transport_->initialize();
    }

    void routeMessage(const std::string& json_message) {
        auto message_type = detectMessageType(json_message);

        switch (message_type) {
            case MessageType::VIDEO:
                video_decoder_->processMessage(
                    JSONVIDMessage::fromJSON(json_message)
                );
                break;
            case MessageType::AUDIO:
                // Route to JSONADAT
                break;
            case MessageType::MIDI:
                // Route to JSONMIDI
                break;
        }
    }

private:
    std::unique_ptr<TOASTTransport> toast_transport_;
    TOASTTransport::Config toast_config_;
};
```

### 2. Clock Synchronization

```cpp
// Unified timestamp management
class TimestampManager {
public:
    static uint64_t getMasterTimestamp() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }

    static void synchronizeStreams(
        uint64_t video_timestamp,
        uint64_t audio_timestamp,
        uint64_t midi_timestamp
    ) {
        // Calculate offsets and adjust for synchronization
        video_audio_offset_ = static_cast<int64_t>(video_timestamp) -
                             static_cast<int64_t>(audio_timestamp);

        // Apply compensation
        compensated_video_time_ = video_timestamp - video_audio_offset_;
    }

private:
    static int64_t video_audio_offset_;
    static uint64_t compensated_video_time_;
};
```

## Performance Monitoring

### 1. Built-in Metrics

```cpp
// Enable comprehensive monitoring
void setupPerformanceMonitoring() {
    encoder_->setStatsCallback([](const VideoStreamStats& stats) {
        std::cout << "=== Video Encoder Stats ===" << std::endl;
        std::cout << "Encode latency: " << stats.average_encode_time_us << "μs" << std::endl;
        std::cout << "Frame rate: " << stats.current_fps << " fps" << std::endl;
        std::cout << "Bitrate: " << stats.bandwidth_kbps << " kbps" << std::endl;
        std::cout << "Dropped frames: " << stats.frames_dropped << std::endl;
    });

    decoder_->setStatsCallback([](const VideoStreamStats& stats) {
        std::cout << "=== Video Decoder Stats ===" << std::endl;
        std::cout << "Decode latency: " << stats.average_decode_time_us << "μs" << std::endl;
        std::cout << "Predicted frames: " << stats.frames_predicted << std::endl;
        std::cout << "End-to-end latency: " << stats.average_end_to_end_latency_us << "μs" << std::endl;
    });
}
```

### 2. System Resource Monitoring

```bash
# Monitor system performance during streaming
# CPU usage
top -p $(pgrep jamcam_demo)

# Memory usage
ps aux | grep jamcam_demo

# Network traffic
sudo iftop -i eth0

# GPU usage (NVIDIA)
nvidia-smi -l 1
```

## Troubleshooting Common Issues

### Issue 1: High Latency

**Symptoms:** Video latency >1ms consistently

**Solutions:**

```cpp
// Reduce quality settings
config.resolution = VideoResolution::ULTRA_LOW_72P;
config.quality = VideoQuality::ULTRA_FAST;
config.enable_face_detection = false;

// Enable hardware acceleration
config.enable_gpu_encoding = true;
config.enable_zero_copy = true;

// Aggressive frame dropping
config.enable_frame_dropping = true;
config.max_encode_time_us = 200;
```

### Issue 2: Frame Drops

**Symptoms:** Frequent "Frame dropped" messages

**Solutions:**

```cpp
// Increase timing thresholds
config.max_encode_time_us = 1000;
config.max_decode_time_us = 500;

// Reduce target FPS
config.target_fps = 10;

// Enable adaptive quality
config.adaptive_quality = true;
```

### Issue 3: Poor Video Quality

**Symptoms:** Blocky or artifact-heavy video

**Solutions:**

```cpp
// Increase quality settings
config.quality = VideoQuality::BALANCED;
config.jpeg_quality = 80;

// Disable aggressive compression
config.enable_chroma_subsampling = false;
config.adaptive_quality = false;

// Enable face detection for better framing
config.enable_face_detection = true;
config.enable_auto_framing = true;
```

### Issue 4: Camera Not Detected

**Linux:**

```bash
# Check camera devices
ls -la /dev/video*

# Test camera access
sudo chmod 666 /dev/video0
ffplay /dev/video0
```

**macOS:**

```bash
# Check camera permissions
system_profiler SPCameraDataType

# Reset camera permissions if needed
sudo tccutil reset Camera
```

## Advanced Configuration

### Multi-Camera Setup

```cpp
// Configure multiple camera streams
struct MultiCamConfig {
    std::vector<JAMCamEncoder::Config> encoder_configs;
    uint8_t primary_stream_id = 0;
    bool enable_stream_switching = true;
    uint32_t switch_threshold_quality = 50; // Switch on quality drop
};

// Setup encoders for each camera
for (size_t i = 0; i < available_cameras.size(); ++i) {
    JAMCamEncoder::Config config;
    config.stream_id = static_cast<uint8_t>(i);
    config.device_id = available_cameras[i].id;

    auto encoder = std::make_unique<JAMCamEncoder>(config);
    encoders_.push_back(std::move(encoder));
}
```

### GPU Acceleration Setup

```cpp
// Verify GPU capabilities
bool checkGPUSupport() {
    #ifdef JSONVID_GPU_ENABLED
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

        return prop.major >= 6; // Require compute capability 6.0+
    }
    #endif

    return false;
}
```

## Integration Testing

### End-to-End Test

```bash
# Terminal 1: Start video receiver
./build/examples/jamcam_basic_demo --mode receiver --port 8080

# Terminal 2: Start video sender
./build/examples/jamcam_basic_demo --mode sender --target 127.0.0.1:8080 --duration 30

# Monitor performance
watch -n 1 'echo "=== Performance ===" && ps aux | grep jamcam | head -2'
```

### Latency Measurement

```cpp
// Built-in latency measurement
class LatencyMeasurement {
public:
    void measureEndToEnd() {
        // Encoder side
        auto encode_start = std::chrono::high_resolution_clock::now();
        encoder_->encodeFrame(test_frame);

        // Network transmission (measured separately)

        // Decoder side
        auto decode_start = std::chrono::high_resolution_clock::now();
        decoder_->processMessage(received_message);
        auto decode_end = std::chrono::high_resolution_clock::now();

        auto total_latency = std::chrono::duration_cast<std::chrono::microseconds>(
            decode_end - encode_start
        ).count();

        std::cout << "End-to-end latency: " << total_latency << "μs" << std::endl;
    }
};
```

## Production Deployment

### Performance Optimization Checklist

- [ ] CPU governor set to 'performance'
- [ ] Network buffers optimized
- [ ] Camera permissions configured
- [ ] GPU acceleration enabled (if available)
- [ ] Frame dropping thresholds tuned
- [ ] Quality vs latency profile selected
- [ ] Clock synchronization verified
- [ ] Error recovery tested
- [ ] Multi-stream capability tested
- [ ] Resource monitoring enabled

### Recommended System Tuning

```bash
# Create tuning script
cat > tune_system.sh << 'EOF'
#!/bin/bash

# CPU performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Network optimization
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.netdev_max_backlog=5000

# Disable power management
sudo systemctl disable cpufreqd 2>/dev/null || true

# Set process priority (run before starting application)
echo "System tuned for ultra-low latency video streaming"
EOF

chmod +x tune_system.sh
./tune_system.sh
```

This completes the JSONVID framework initialization. The system is now ready for ultra-low latency video streaming with sub-400μs total latency targets.
