# 🔧 GPU-Native Audio Engine Specification
## Technical Implementation Guide for Cross-Platform JAMNet

### 📋 Overview

This document provides detailed technical specifications for implementing the GPU-native, cross-platform audio architecture outlined in the Integration Plan. It serves as the definitive reference for developers working on the JAMNet audio engine transformation.

---

## 🧱 Core Architecture Components

### 1. GPURenderEngine Interface

```cpp
class GPURenderEngine {
public:
    struct RenderConfig {
        uint32_t sampleRate = 48000;      // Target sample rate
        uint32_t bufferSize = 64;         // Frames per buffer
        uint32_t channels = 2;            // Audio channels
        bool enablePNBTR = true;          // Prediction/correction
        float maxLatencyMs = 5.0f;        // Latency budget
    };
    
    struct GPUTimestamp {
        uint64_t gpu_time_ns;             // GPU clock nanoseconds
        uint64_t system_time_ns;          // System clock correlation
        float calibration_offset_ms;      // Drift correction
    };
    
    virtual ~GPURenderEngine() = default;
    
    // Core rendering interface
    virtual bool initialize(const RenderConfig& config) = 0;
    virtual void shutdown() = 0;
    virtual void renderToBuffer(float* outputBuffer, 
                               uint32_t numFrames,
                               const GPUTimestamp& timestamp) = 0;
    
    // Timing and synchronization
    virtual GPUTimestamp getCurrentTimestamp() = 0;
    virtual void updateSyncCalibration(float offsetMs) = 0;
    virtual bool isGPUAvailable() const = 0;
    
    // Buffer management
    virtual void* getSharedBuffer() = 0;
    virtual size_t getSharedBufferSize() const = 0;
    virtual void flushBufferToGPU() = 0;
};
```

### 2. Platform-Specific Engine Implementations

#### MetalRenderEngine (macOS)
```cpp
class MetalRenderEngine : public GPURenderEngine {
private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> commandQueue_;
    id<MTLBuffer> audioBuffer_;
    id<MTLBuffer> timestampBuffer_;
    id<MTLComputePipelineState> renderPipeline_;
    id<MTLComputePipelineState> pnbtrPipeline_;
    
    struct SyncCalibrationBlock {
        uint64_t metal_timestamp;
        uint64_t host_timestamp;
        float calibration_offset;
    } syncBlock_;
    
public:
    bool initialize(const RenderConfig& config) override;
    void renderToBuffer(float* outputBuffer, uint32_t numFrames, 
                       const GPUTimestamp& timestamp) override;
    GPUTimestamp getCurrentTimestamp() override;
    void updateSyncCalibration(float offsetMs) override;
};
```

#### VulkanRenderEngine (Linux)
```cpp
class VulkanRenderEngine : public GPURenderEngine {
private:
    VkInstance instance_;
    VkDevice device_;
    VkQueue computeQueue_;
    VkBuffer audioBuffer_;
    VkDeviceMemory audioBufferMemory_;
    VkCommandPool commandPool_;
    VkPipeline renderPipeline_;
    VkPipeline pnbtrPipeline_;
    
    struct SyncCalibrationBlock {
        uint64_t vulkan_timestamp;
        uint64_t clock_timestamp;
        float calibration_offset;
    } syncBlock_;
    
public:
    bool initialize(const RenderConfig& config) override;
    void renderToBuffer(float* outputBuffer, uint32_t numFrames,
                       const GPUTimestamp& timestamp) override;
    GPUTimestamp getCurrentTimestamp() override;
    void updateSyncCalibration(float offsetMs) override;
};
```

### 3. AudioOutputBackend Interface

```cpp
class AudioOutputBackend {
public:
    struct AudioConfig {
        uint32_t sampleRate = 48000;
        uint32_t bufferSize = 64;
        uint32_t channels = 2;
        std::string deviceName = "default";
        bool enableLowLatency = true;
    };
    
    enum class BackendType {
        JACK,
        CORE_AUDIO,
        ALSA,
        DUMMY
    };
    
    virtual ~AudioOutputBackend() = default;
    
    // Lifecycle management
    virtual bool initialize(const AudioConfig& config) = 0;
    virtual void shutdown() = 0;
    virtual BackendType getType() const = 0;
    
    // Audio processing
    virtual void pushAudio(const float* data, uint32_t numFrames, 
                          uint64_t timestampNs) = 0;
    virtual void setProcessCallback(std::function<void(float*, uint32_t, uint64_t)> callback) = 0;
    
    // Timing and sync
    virtual uint64_t getCurrentTimeNs() = 0;
    virtual float getActualLatencyMs() const = 0;
    virtual bool supportsGPUMemory() const = 0;
    
    // Factory method
    static std::unique_ptr<AudioOutputBackend> create(BackendType preferredType = BackendType::JACK);
};
```

### 4. JACK Backend Implementation

```cpp
class JackAudioBackend : public AudioOutputBackend {
private:
    jack_client_t* client_;
    jack_port_t* outputPorts_[8];  // Support up to 8 channels
    std::function<void(float*, uint32_t, uint64_t)> processCallback_;
    
    // GPU integration
    void* gpuSharedMemory_;
    size_t gpuBufferSize_;
    bool useGPUMemory_;
    
    // Timing
    uint64_t lastGPUTimestamp_;
    float calibrationOffset_;
    
    // JACK callbacks
    static int jackProcessCallback(jack_nframes_t nframes, void* arg);
    static void jackShutdownCallback(void* arg);
    
public:
    bool initialize(const AudioConfig& config) override;
    void shutdown() override;
    BackendType getType() const override { return BackendType::JACK; }
    
    void pushAudio(const float* data, uint32_t numFrames, uint64_t timestampNs) override;
    void setProcessCallback(std::function<void(float*, uint32_t, uint64_t)> callback) override;
    
    uint64_t getCurrentTimeNs() override;
    float getActualLatencyMs() const override;
    bool supportsGPUMemory() const override { return useGPUMemory_; }
    
    // JACK-specific GPU integration
    bool enableGPUMemoryMode(void* sharedBuffer, size_t bufferSize);
    void setExternalClock(std::function<uint64_t()> clockCallback);
};
```

---

## ⚙️ JACK Transformation Specifications

### 1. Clock Injection System

#### Modified `jack_time.c`:
```c
// Global GPU clock callback
static uint64_t (*gpu_clock_callback)(void) = NULL;
static void* gpu_clock_userdata = NULL;

// New API function
void jack_set_external_clock(uint64_t (*callback)(void), void* userdata) {
    gpu_clock_callback = callback;
    gpu_clock_userdata = userdata;
}

// Modified time functions
jack_time_t jack_get_time(void) {
    if (gpu_clock_callback) {
        return gpu_clock_callback();
    }
    // Fall back to system clock
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (jack_time_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}
```

### 2. GPU Memory Integration

#### Modified `jack_port.c`:
```c
// New port flags for GPU memory
#define JackPortIsGPUBacked     0x20
#define JackPortUsesSharedMem   0x40

// GPU memory registration
int jack_port_register_gpu_buffer(jack_client_t* client,
                                  const char* port_name,
                                  const char* port_type,
                                  void* gpu_buffer,
                                  size_t buffer_size) {
    // Implementation for GPU-backed port creation
}

// Modified buffer access
void* jack_port_get_buffer(jack_port_t* port, jack_nframes_t nframes) {
    if (port->flags & JackPortIsGPUBacked) {
        return port->gpu_buffer + (port->buffer_offset * sizeof(float));
    }
    // Standard buffer access
    return port->buffer;
}
```

### 3. Transport Layer Enhancement

#### Modified `transport.c`:
```c
// GPU-driven transport state
typedef struct {
    jack_position_t position;
    uint64_t gpu_timestamp;
    float calibration_offset;
    bool gpu_sync_enabled;
} jack_gpu_transport_state_t;

// New transport functions
int jack_transport_set_gpu_sync(jack_client_t* client, bool enable);
int jack_transport_get_gpu_position(jack_client_t* client, 
                                   jack_gpu_transport_state_t* state);
```

---

## 🔄 Shared Audio Frame Format

### Universal Frame Structure
```cpp
struct JamAudioFrame {
    // Audio data
    alignas(16) float samples[JAM_MAX_BUFFER_SIZE];  // SIMD-aligned
    uint32_t numSamples;
    uint32_t sampleRate;
    uint8_t channels;
    
    // Timing information
    uint64_t timestamp_gpu_ns;      // GPU clock nanoseconds
    uint64_t timestamp_system_ns;   // Correlated system time
    float calibration_offset_ms;    // Current drift correction
    
    // Quality and prediction flags
    uint32_t flags;
    #define JAM_FRAME_CLEAN         0x00
    #define JAM_FRAME_PREDICTED     0x01  // PNBTR filled
    #define JAM_FRAME_INTERPOLATED  0x02  // Gap filled
    #define JAM_FRAME_SILENCE       0x04  // Intentional silence
    #define JAM_FRAME_DROPOUT       0x08  // Network loss
    #define JAM_FRAME_OVERRUN       0x10  // Buffer overrun
    
    // Prediction data (PNBTR)
    float prediction_confidence;    // 0.0-1.0
    uint32_t prediction_samples;    // How many samples are predicted
    
    // Network metadata (for TOAST)
    uint32_t sequence_number;
    uint32_t source_node_id;
    uint16_t checksum;
};
```

### Buffer Management
```cpp
class SharedAudioBuffer {
private:
    static constexpr size_t RING_BUFFER_SIZE = 8192;  // Power of 2
    JamAudioFrame frames_[RING_BUFFER_SIZE];
    std::atomic<uint32_t> writeIndex_{0};
    std::atomic<uint32_t> readIndex_{0};
    
public:
    bool pushFrame(const JamAudioFrame& frame);
    bool popFrame(JamAudioFrame& frame);
    bool isEmpty() const;
    bool isFull() const;
    uint32_t getAvailableFrames() const;
    void flush();
};
```

---

## 🎯 Latency Doctrine Implementation

### Timing Precision Requirements
```cpp
class LatencyController {
private:
    static constexpr uint64_t MAX_DRIFT_NS = 100000;      // 100µs max drift
    static constexpr uint64_t CORRECTION_INTERVAL_NS = 2000000;  // 2ms correction window
    
    struct LatencyBudget {
        uint64_t target_roundtrip_ns = 5000000;  // 5ms default
        uint64_t current_measured_ns = 0;
        uint64_t max_allowed_ns = 0;
        bool budget_exceeded = false;
    };
    
public:
    void updateLatencyMeasurement(uint64_t measured_ns);
    bool isBudgetExceeded() const;
    void enforceLatencyDiscipline();
    uint64_t getTimeToNextCorrection() const;
};
```

### PNBTR Integration Specifications
```cpp
class PNBTRProcessor {
public:
    struct PredictionResult {
        float* predictedSamples;
        uint32_t numSamples;
        float confidence;
        bool useGPUPrediction;
    };
    
    virtual PredictionResult predictMissingSamples(
        const JamAudioFrame* historyFrames,
        uint32_t historyCount,
        uint32_t missingSamples) = 0;
        
    virtual void updatePredictionModel(
        const JamAudioFrame& actualFrame,
        const PredictionResult& previousPrediction) = 0;
};
```

---

## 🔧 Build System Configuration

### CMake Platform Detection
```cmake
# Platform-specific GPU backend selection
if(APPLE)
    set(GPU_BACKEND "Metal")
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    set(GPU_LIBRARIES ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK})
    set(GPU_SOURCES src/gpu/MetalRenderEngine.cpp)
elseif(UNIX AND NOT APPLE)
    set(GPU_BACKEND "Vulkan")
    find_package(Vulkan REQUIRED)
    set(GPU_LIBRARIES ${Vulkan_LIBRARIES})
    set(GPU_SOURCES src/gpu/VulkanRenderEngine.cpp)
endif()

# Audio backend configuration
if(ENABLE_JACK)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(JACK REQUIRED jack)
    set(AUDIO_LIBRARIES ${AUDIO_LIBRARIES} ${JACK_LIBRARIES})
    set(AUDIO_SOURCES ${AUDIO_SOURCES} src/audio/JackAudioBackend.cpp)
endif()

if(APPLE AND ENABLE_COREAUDIO)
    find_library(COREAUDIO_FRAMEWORK CoreAudio)
    find_library(AUDIOTOOLBOX_FRAMEWORK AudioToolbox)
    set(AUDIO_LIBRARIES ${AUDIO_LIBRARIES} ${COREAUDIO_FRAMEWORK} ${AUDIOTOOLBOX_FRAMEWORK})
    set(AUDIO_SOURCES ${AUDIO_SOURCES} src/audio/CoreAudioBackend.cpp)
endif()
```

### Preprocessor Configuration
```cpp
// Platform-specific includes and definitions
#ifdef __APPLE__
    #define JAM_GPU_BACKEND_METAL 1
    #import <Metal/Metal.h>
    #import <MetalKit/MetalKit.h>
    #ifdef JAM_ENABLE_COREAUDIO
        #define JAM_AUDIO_BACKEND_COREAUDIO 1
        #include <CoreAudio/CoreAudio.h>
    #endif
#elif defined(__linux__)
    #define JAM_GPU_BACKEND_VULKAN 1
    #include <vulkan/vulkan.h>
#endif

#ifdef JAM_ENABLE_JACK
    #define JAM_AUDIO_BACKEND_JACK 1
    #include <jack/jack.h>
#endif
```

---

## 🧪 Testing and Validation Framework

### Performance Benchmarks
```cpp
class PerformanceBenchmark {
public:
    struct BenchmarkResult {
        uint64_t min_latency_ns;
        uint64_t max_latency_ns;
        uint64_t avg_latency_ns;
        uint64_t jitter_ns;
        float cpu_usage_percent;
        float gpu_usage_percent;
        uint32_t dropped_frames;
        bool meets_latency_doctrine;
    };
    
    BenchmarkResult runLatencyTest(uint32_t durationSeconds = 60);
    BenchmarkResult runStabilityTest(uint32_t durationSeconds = 600);
    BenchmarkResult runCrossPlatformSyncTest();
};
```

### Validation Tests
- [ ] GPU timer accuracy vs system clock
- [ ] Cross-platform timing consistency
- [ ] JACK clock injection functionality
- [ ] Memory-mapped GPU buffer access
- [ ] PNBTR prediction accuracy
- [ ] Network round-trip latency
- [ ] Multi-platform audio rendering identical output
- [ ] Real-time performance under load

This specification provides the technical foundation for implementing the cross-platform, GPU-native, latency-doctrine-compliant JAMNet audio architecture. Each component is designed to maintain the highest performance standards while enabling seamless operation across macOS and Linux platforms.
