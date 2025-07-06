#pragma once

#include <cstdint>
#include <memory>
#include <functional>

namespace JAMNet {

/**
 * @brief Abstract interface for GPU-native audio rendering engines
 * 
 * This interface provides platform-agnostic GPU audio rendering capabilities,
 * with implementations for Metal (macOS) and Vulkan (Linux).
 */
class GPURenderEngine {
public:
    struct RenderConfig {
        uint32_t sampleRate = 48000;      ///< Target sample rate
        uint32_t bufferSize = 64;         ///< Frames per buffer
        uint32_t channels = 2;            ///< Audio channels
        bool enablePNBTR = true;          ///< Prediction/correction
        float maxLatencyMs = 5.0f;        ///< Latency budget
    };
    
    struct GPUTimestamp {
        uint64_t gpu_time_ns;             ///< GPU clock nanoseconds
        uint64_t system_time_ns;          ///< System clock correlation
        float calibration_offset_ms;      ///< Drift correction
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
    
    // Factory method
    static std::unique_ptr<GPURenderEngine> create();
};

} // namespace JAMNet
