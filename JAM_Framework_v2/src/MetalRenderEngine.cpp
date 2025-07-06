#include "MetalRenderEngine.h"

#ifdef __APPLE__
#import <CoreServices/CoreServices.h>
#include <mach/mach_time.h>
#include <iostream>

namespace JAMNet {

MetalRenderEngine::MetalRenderEngine() 
    : device_(nil), commandQueue_(nil), audioBuffer_(nil), timestampBuffer_(nil),
      renderPipeline_(nil), pnbtrPipeline_(nil), sharedBuffer_(nullptr), 
      sharedBufferSize_(0), initialized_(false) {
    syncBlock_ = {0, 0, 0.0f, false};
}

MetalRenderEngine::~MetalRenderEngine() {
    shutdown();
}

bool MetalRenderEngine::initialize(const RenderConfig& config) {
    if (initialized_) {
        return true;
    }
    
    config_ = config;
    
    // Get the default Metal device
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        std::cerr << "MetalRenderEngine: Failed to create Metal device" << std::endl;
        return false;
    }
    
    // Create command queue
    commandQueue_ = [device_ newCommandQueue];
    if (!commandQueue_) {
        std::cerr << "MetalRenderEngine: Failed to create command queue" << std::endl;
        return false;
    }
    
    // Setup buffers and pipelines
    if (!setupBuffers() || !setupMetalPipelines()) {
        return false;
    }
    
    // Initialize sync calibration block
    updateSyncBlock();
    
    initialized_ = true;
    std::cout << "MetalRenderEngine: Initialized successfully" << std::endl;
    std::cout << "  Device: " << [device_.name UTF8String] << std::endl;
    std::cout << "  Sample Rate: " << config_.sampleRate << "Hz" << std::endl;
    std::cout << "  Buffer Size: " << config_.bufferSize << " frames" << std::endl;
    std::cout << "  Channels: " << config_.channels << std::endl;
    
    return true;
}

void MetalRenderEngine::shutdown() {
    if (!initialized_) {
        return;
    }
    
    // Release Metal resources
    if (audioBuffer_) {
        [audioBuffer_ release];
        audioBuffer_ = nil;
    }
    if (timestampBuffer_) {
        [timestampBuffer_ release];
        timestampBuffer_ = nil;
    }
    if (renderPipeline_) {
        [renderPipeline_ release];
        renderPipeline_ = nil;
    }
    if (pnbtrPipeline_) {
        [pnbtrPipeline_ release];
        pnbtrPipeline_ = nil;
    }
    if (commandQueue_) {
        [commandQueue_ release];
        commandQueue_ = nil;
    }
    if (device_) {
        [device_ release];
        device_ = nil;
    }
    
    sharedBuffer_ = nullptr;
    sharedBufferSize_ = 0;
    initialized_ = false;
    
    std::cout << "MetalRenderEngine: Shutdown complete" << std::endl;
}

bool MetalRenderEngine::setupBuffers() {
    // Calculate buffer sizes
    size_t audioBufferSize = config_.bufferSize * config_.channels * sizeof(float);
    size_t timestampBufferSize = sizeof(uint64_t) * 2; // GPU and host timestamps
    
    // Create audio buffer
    audioBuffer_ = [device_ newBufferWithLength:audioBufferSize 
                                        options:MTLResourceStorageModeShared];
    if (!audioBuffer_) {
        std::cerr << "MetalRenderEngine: Failed to create audio buffer" << std::endl;
        return false;
    }
    
    // Create timestamp buffer
    timestampBuffer_ = [device_ newBufferWithLength:timestampBufferSize
                                            options:MTLResourceStorageModeShared];
    if (!timestampBuffer_) {
        std::cerr << "MetalRenderEngine: Failed to create timestamp buffer" << std::endl;
        return false;
    }
    
    // Setup shared buffer for external access
    sharedBuffer_ = [audioBuffer_ contents];
    sharedBufferSize_ = audioBufferSize;
    
    return true;
}

bool MetalRenderEngine::setupMetalPipelines() {
    NSError* error = nil;
    
    // Simple Metal shader source for audio rendering
    NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void audioRenderKernel(device float* audioBuffer [[buffer(0)]],
                             device uint64_t* timestamps [[buffer(1)]],
                             uint index [[thread_position_in_grid]]) {
    // Simple sine wave generator for testing
    // In production, this would be replaced with sophisticated audio DSP
    float sampleRate = 48000.0;
    float frequency = 440.0; // A4 note
    uint64_t sampleIndex = timestamps[0] + index;
    float phase = (float(sampleIndex) / sampleRate) * frequency * 2.0 * M_PI_F;
    audioBuffer[index] = sin(phase) * 0.1; // Low volume for safety
}

kernel void pnbtrPredictionKernel(device float* audioBuffer [[buffer(0)]],
                                 device float* historyBuffer [[buffer(1)]],
                                 device uint* predictionParams [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
    // PNBTR prediction algorithm placeholder
    // This would implement the actual prediction logic
    uint historyLength = predictionParams[0];
    uint predictionLength = predictionParams[1];
    
    if (index < predictionLength) {
        // Simple linear extrapolation for now
        if (historyLength >= 2) {
            float prev1 = historyBuffer[historyLength - 1];
            float prev2 = historyBuffer[historyLength - 2];
            audioBuffer[index] = prev1 + (prev1 - prev2);
        } else {
            audioBuffer[index] = 0.0;
        }
    }
}
)";
    
    // Compile shader library
    id<MTLLibrary> library = [device_ newLibraryWithSource:shaderSource
                                                   options:nil
                                                     error:&error];
    if (!library) {
        std::cerr << "MetalRenderEngine: Failed to compile shaders: " 
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    // Create render pipeline
    id<MTLFunction> renderFunction = [library newFunctionWithName:@"audioRenderKernel"];
    renderPipeline_ = [device_ newComputePipelineStateWithFunction:renderFunction
                                                             error:&error];
    if (!renderPipeline_) {
        std::cerr << "MetalRenderEngine: Failed to create render pipeline: "
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    // Create PNBTR pipeline
    id<MTLFunction> pnbtrFunction = [library newFunctionWithName:@"pnbtrPredictionKernel"];
    pnbtrPipeline_ = [device_ newComputePipelineStateWithFunction:pnbtrFunction
                                                            error:&error];
    if (!pnbtrPipeline_) {
        std::cerr << "MetalRenderEngine: Failed to create PNBTR pipeline: "
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    [library release];
    [renderFunction release];
    [pnbtrFunction release];
    
    return true;
}

void MetalRenderEngine::renderToBuffer(float* outputBuffer, uint32_t numFrames,
                                      const GPUTimestamp& timestamp) {
    if (!initialized_ || !outputBuffer) {
        return;
    }
    
    // Update timestamp buffer
    uint64_t* timestampData = (uint64_t*)[timestampBuffer_ contents];
    timestampData[0] = timestamp.gpu_time_ns;
    timestampData[1] = timestamp.system_time_ns;
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Setup render compute pass
    [encoder setComputePipelineState:renderPipeline_];
    [encoder setBuffer:audioBuffer_ offset:0 atIndex:0];
    [encoder setBuffer:timestampBuffer_ offset:0 atIndex:1];
    
    // Calculate thread group sizes
    NSUInteger threadsPerGroup = renderPipeline_.threadExecutionWidth;
    NSUInteger numGroups = (numFrames + threadsPerGroup - 1) / threadsPerGroup;
    
    [encoder dispatchThreadgroups:MTLSizeMake(numGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    
    [encoder endEncoding];
    
    // Commit and wait for completion
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results to output buffer
    float* audioData = (float*)[audioBuffer_ contents];
    for (uint32_t i = 0; i < numFrames * config_.channels; ++i) {
        outputBuffer[i] = audioData[i];
    }
    
    // Update sync calibration
    updateSyncBlock();
}

GPURenderEngine::GPUTimestamp MetalRenderEngine::getCurrentTimestamp() {
    updateSyncBlock();
    
    GPUTimestamp timestamp;
    timestamp.gpu_time_ns = syncBlock_.metal_timestamp;
    timestamp.system_time_ns = syncBlock_.host_timestamp;
    timestamp.calibration_offset_ms = syncBlock_.calibration_offset;
    
    return timestamp;
}

void MetalRenderEngine::updateSyncCalibration(float offsetMs) {
    syncBlock_.calibration_offset = offsetMs;
}

bool MetalRenderEngine::isGPUAvailable() const {
    return device_ != nil && initialized_;
}

void MetalRenderEngine::flushBufferToGPU() {
    // For Metal on macOS, shared buffers are automatically coherent
    // No explicit flush needed, but we could add memory barriers here if required
}

void MetalRenderEngine::updateSyncBlock() {
    // Get current Metal timestamp
    syncBlock_.metal_timestamp = mach_absolute_time();
    
    // Get correlated host timestamp
    syncBlock_.host_timestamp = mach_absolute_time();
    
    // Mark as valid
    syncBlock_.valid = true;
}

} // namespace JAMNet

#endif // __APPLE__
