#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h>
#include "../Core/FrameStateTracker.h"
#endif

#include <string>
#include <memory>

class GPUDebugHelper {
public:
    static GPUDebugHelper& getInstance();
    
    struct DummyParams {
        uint32_t frameOffset;
        uint32_t numSamples;
        float testGain;
        uint32_t frameIndex;
    };
    
    enum class DebugMode {
        PassThrough,    // Simple pass-through with gain
        TestPattern,    // Generate test pattern
        Validation      // Validate input and pass through
    };
    
#ifdef __OBJC__
    bool initializeWithDevice(id<MTLDevice> device);
    
    // Process audio using dummy kernels to test pipeline
    bool processDummyAudio(
        id<MTLCommandBuffer> commandBuffer,
        id<MTLBuffer> inputBuffer,
        id<MTLBuffer> outputBuffer,
        size_t numSamples,
        int frameIndex,
        DebugMode mode = DebugMode::PassThrough
    );
    
    // Enhanced logging version with frame state tracking
    bool processDummyAudioWithLogging(
        id<MTLCommandBuffer> commandBuffer,
        id<MTLBuffer> inputBuffer,
        id<MTLBuffer> outputBuffer,
        size_t numSamples,
        int frameIndex,
        FrameStateTracker& frameTracker,
        DebugMode mode = DebugMode::PassThrough
    );
    
    // Get validation results
    uint32_t getValidationNonZeroCount() const;
    void resetValidationCounter();
    
private:
    id<MTLDevice> metalDevice;
    id<MTLLibrary> defaultLibrary;
    id<MTLComputePipelineState> passThroughPipeline;
    id<MTLComputePipelineState> testPatternPipeline;
    id<MTLComputePipelineState> validationPipeline;
    id<MTLBuffer> validationCounterBuffer;
    
    bool initialized = false;
    
    bool createPipelines();
#endif
    
    GPUDebugHelper() = default;
    ~GPUDebugHelper() = default;
    GPUDebugHelper(const GPUDebugHelper&) = delete;
    GPUDebugHelper& operator=(const GPUDebugHelper&) = delete;
};
