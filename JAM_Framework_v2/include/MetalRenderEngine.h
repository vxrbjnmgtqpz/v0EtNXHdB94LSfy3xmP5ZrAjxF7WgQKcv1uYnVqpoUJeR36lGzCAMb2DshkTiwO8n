#pragma once

#include "GPURenderEngine.h"

#ifdef __APPLE__

// Forward declarations for Metal types to avoid including Metal headers in C++
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#else
// Forward declarations for C++ compilation
typedef struct objc_object* id;
#endif

namespace JAMNet {

/**
 * @brief Metal-based GPU render engine for macOS
 * 
 * Implements GPU-native audio rendering using Metal shaders with
 * precise timing integration and Core Audio compatibility.
 */
class MetalRenderEngine : public GPURenderEngine {
private:
    // Use void* for C++ compilation, cast to proper types in .mm file
    void* device_;
    void* commandQueue_;
    void* audioBuffer_;
    void* timestampBuffer_;
    void* renderPipeline_;
    void* pnbtrPipeline_;
    
    // Sync calibration block for GPU ↔ CPU timing
    struct SyncCalibrationBlock {
        uint64_t metal_timestamp;
        uint64_t host_timestamp;
        float calibration_offset;
        bool valid;
    } syncBlock_;
    
    RenderConfig config_;
    void* sharedBuffer_;
    size_t sharedBufferSize_;
    bool initialized_;
    
    // Metal shader compilation and setup
    bool setupMetalPipelines();
    bool setupBuffers();
    void updateSyncBlock();
    
public:
    MetalRenderEngine();
    virtual ~MetalRenderEngine();
    
    // GPURenderEngine interface implementation
    bool initialize(const RenderConfig& config) override;
    void shutdown() override;
    void renderToBuffer(float* outputBuffer, uint32_t numFrames,
                       const GPUTimestamp& timestamp) override;
    
    GPUTimestamp getCurrentTimestamp() override;
    void updateSyncCalibration(float offsetMs) override;
    bool isGPUAvailable() const override;
    
    void* getSharedBuffer() override { return sharedBuffer_; }
    size_t getSharedBufferSize() const override { return sharedBufferSize_; }
    void flushBufferToGPU() override;
    
    // Metal-specific methods (only available in Objective-C++ context)
#ifdef __OBJC__
    id<MTLDevice> getMetalDevice() const;
#endif
    const SyncCalibrationBlock& getSyncBlock() const { return syncBlock_; }
};

} // namespace JAMNet

#endif // __APPLE__
