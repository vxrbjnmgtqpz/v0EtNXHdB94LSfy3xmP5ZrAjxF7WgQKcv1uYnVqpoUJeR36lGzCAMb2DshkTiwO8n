#pragma once

#include "GPUDebugHelper.h"
#include "../Core/FrameStateTracker.h"

class MetalBridgeDebugIntegration {
public:
    static MetalBridgeDebugIntegration& getInstance();
    
    // Debug mode flags
    struct DebugConfig {
        bool enableFrameLogging = true;
        bool enableGPUValidation = true;
        bool useDummyProcessing = false;  // If true, replace all DSP with dummy kernels
        bool enableDetailedPipelineLogging = true;
        float debugTestGain = 1.0f;
        
        // Performance monitoring
        bool enableLatencyTracking = true;
        int logEveryNFrames = 100;  // Log summary every N frames
    };
    
    void setDebugConfig(const DebugConfig& config) { debugConfig = config; }
    const DebugConfig& getDebugConfig() const { return debugConfig; }
    
#ifdef __OBJC__
    // Initialize debug system
    bool initialize(id<MTLDevice> device);
    
    // Main debug processing function that can replace normal GPU processing
    bool processAudioFrameWithFullDebugging(
        id<MTLCommandBuffer> commandBuffer,
        id<MTLBuffer> inputBuffer,
        id<MTLBuffer> outputBuffer,
        size_t numSamples,
        int frameIndex
    );
    
    // Enhanced checkpoint logging system
    void logCheckpoint(int checkpointNumber, const std::string& message, 
                      int frameIndex, float peakValue = 0.0f);
    
    // Pipeline validation
    bool validateGPUPipeline(int frameIndex);
    
    // Performance tracking
    void startFrameTimer(int frameIndex);
    void endFrameTimer(int frameIndex);
    void logPerformanceStats();
    
private:
    DebugConfig debugConfig;
    FrameStateTracker frameTracker;
    GPUDebugHelper& gpuDebugHelper;
    
    // Timing
    std::map<int, std::chrono::high_resolution_clock::time_point> frameStartTimes;
    std::vector<float> frameLatencies;
    
    // Statistics
    int totalFramesProcessed = 0;
    int successfulFrames = 0;
    int failedFrames = 0;
    
    bool debugInitialized = false;
#endif
    
    MetalBridgeDebugIntegration();
    ~MetalBridgeDebugIntegration() = default;
    MetalBridgeDebugIntegration(const MetalBridgeDebugIntegration&) = delete;
    MetalBridgeDebugIntegration& operator=(const MetalBridgeDebugIntegration&) = delete;
};
