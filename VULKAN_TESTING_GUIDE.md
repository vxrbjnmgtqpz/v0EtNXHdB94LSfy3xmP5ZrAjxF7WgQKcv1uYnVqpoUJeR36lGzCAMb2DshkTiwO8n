/**
 * Vulkan GPU Backend Testing Guide for JAMNet
 * 
 * This guide outlines the steps needed to test and validate the Vulkan implementation
 * on Linux systems with proper GPU support.
 */

## Prerequisites for Linux Testing

### 1. System Requirements
- Linux distribution with modern kernel (Ubuntu 20.04+, Fedora 35+, etc.)
- GPU with Vulkan support (NVIDIA, AMD, or Intel)
- Vulkan SDK installed
- JACK Audio Connection Kit
- CMake 3.16+
- C++17 compatible compiler

### 2. Vulkan SDK Installation

#### Ubuntu/Debian:
```bash
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list https://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
sudo apt update
sudo apt install vulkan-sdk vulkan-tools vulkan-validationlayers-dev
```

#### Arch Linux:
```bash
sudo pacman -S vulkan-devel vulkan-tools vulkan-validation-layers
```

#### Fedora/RHEL:
```bash
sudo dnf install vulkan-devel vulkan-tools vulkan-validation-layers
```

### 3. Verify Vulkan Installation
```bash
vulkaninfo
vkcube  # Should display a spinning cube
glslc --version  # Should show GLSL compiler version
```

## Build and Test Process

### 1. Compile Vulkan Shaders
```bash
cd /path/to/MIDIp2p/JAM_Framework_v2
./compile_vulkan_shaders.sh
```

Expected output:
```
Compiling Vulkan shaders to SPIR-V...
Compiling audio_processing.comp...
Compiling pnbtr_predict.comp...
Shader compilation complete!
✓ audio_processing.spv valid
✓ pnbtr_predict.spv valid
All shaders validated successfully!
```

### 2. Build JAM Framework with Vulkan
```bash
mkdir -p build
cd build
cmake .. -DJAM_GPU_BACKEND=vulkan -DJAM_ENABLE_GPU=ON -DJAM_ENABLE_JACK=ON
make -j$(nproc)
```

### 3. Validation Tests

#### A. Vulkan Device Detection Test
```bash
./vulkan_device_test
```
Expected: Lists available Vulkan devices and compute queues

#### B. GPU Buffer Test
```bash
./vulkan_buffer_test
```
Expected: Creates and maps GPU buffers successfully

#### C. Compute Pipeline Test
```bash
./vulkan_pipeline_test
```
Expected: Loads and creates compute pipelines from SPIR-V

#### D. Audio Processing Test
```bash
./vulkan_audio_test
```
Expected: Processes audio samples on GPU and validates output

### 4. Cross-Platform Timing Validation

#### A. Timestamp Accuracy Test
```bash
./timing_accuracy_test
```
Expected: GPU timestamps correlate with system time within 50µs

#### B. Latency Measurement Test
```bash
./latency_test
```
Expected: End-to-end latency < 5ms, jitter < 50µs

#### C. Drift Correction Test
```bash
./drift_test --duration=600  # 10 minutes
```
Expected: Timing drift < 100µs over 10 minutes

### 5. JACK Integration Testing

#### A. JACK Basic Integration
```bash
# Start JACK with custom clock injection
jackd -d alsa -r 48000 -p 64 &
./jack_vulkan_integration_test
```

#### B. Multi-Client Test
```bash
# Test coexistence with other JACK clients
./jack_multi_client_test
```

#### C. XRUNs Stress Test
```bash
./jack_stress_test --duration=300 --load=high
```
Expected: Zero XRUNs under normal load

### 6. Cross-Platform Parity Validation

#### A. macOS vs Linux Audio Comparison
```bash
# On macOS:
./metal_audio_reference_test > macos_output.log

# On Linux:
./vulkan_audio_reference_test > linux_output.log

# Compare outputs:
./compare_audio_outputs.py macos_output.log linux_output.log
```
Expected: < 0.1% difference in audio output, < 50µs timing variance

#### B. Synchronization Test (Multi-Node)
```bash
# Run on both macOS and Linux machines simultaneously
./cross_platform_sync_test --role=master --ip=192.168.1.100
./cross_platform_sync_test --role=slave --master=192.168.1.100
```
Expected: Audio sync within 50µs between nodes

## Performance Benchmarks

### 1. GPU Dispatch Performance
- Target: 750+ dispatches/second at 48kHz
- Measure: CPU overhead < 5% on modern systems

### 2. Memory Transfer Performance
- Target: Zero-copy buffer access
- Measure: No memcpy in critical path

### 3. Real-Time Consistency
- Target: Consistent frame delivery every 1.33ms (64 samples @ 48kHz)
- Measure: Jitter < 50µs standard deviation

## Known Issues and Workarounds

### 1. GPU Driver Compatibility
- Issue: Some older drivers may not support compute timestamps
- Workaround: Fallback to system time with warning

### 2. JACK Custom Build
- Issue: Custom JACK modifications required
- Workaround: Bundle modified JACK or provide build scripts

### 3. GPU Resource Contention
- Issue: Graphics workload may affect audio processing
- Workaround: Use dedicated compute queue and priority

## Debugging Tools

### 1. Vulkan Validation Layers
```bash
export VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
```

### 2. GPU Profiling
```bash
# NVIDIA:
nsight-compute ./vulkan_audio_test

# AMD:
rocprof ./vulkan_audio_test
```

### 3. JACK Debugging
```bash
export JACK_DEBUG=1
./jack_vulkan_integration_test
```

## Success Criteria Summary

✅ **Phase 1 Complete**: Vulkan backend renders audio equivalent to Metal
✅ **Phase 2 Complete**: Cross-platform timing variance < 50µs  
✅ **Phase 3 Complete**: JACK integration robust, zero XRUNs
✅ **Phase 4 Complete**: End-to-end latency < 5ms on both platforms

This testing framework ensures that the Vulkan implementation achieves full
parity with the Metal backend and provides reliable, low-latency audio
performance on Linux systems.
