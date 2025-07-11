#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h> // This single import is sufficient
#include <string>
#include <vector>
#endif

// The number of command buffers in flight, allowing the CPU to get ahead of the GPU.
#define MAX_FRAMES_IN_FLIGHT 3

class MetalBridge {
public:
    static MetalBridge& getInstance();

    MetalBridge(const MetalBridge&) = delete;
    MetalBridge& operator=(const MetalBridge&) = delete;

    bool initialize();
    void prepareBuffers(size_t numSamples, double sampleRate);
    void processAudioBlock(const float* inputData, float* outputData, size_t numSamples);
    void setRecordArmStates(bool jellieArmed, bool pnbtrArmed);
    bool isInitialized() const { return initialized; }
    
    // Async visualization loop
    void startVisualizationLoop();
    void stopVisualizationLoop();
    void updateVisualizationBuffer(const float* audioData, size_t numSamples);
    const float* getVisualizationBuffer(size_t& bufferSize) const;
    
    // ADDED: Thread-safe access to GPU processing buffers for GUI visualization
    const float* getLatestInputBuffer() const;
    const float* getLatestReconstructedBuffer() const;
    const float* getLatestSpectralBuffer() const;
    const float* getLatestNetworkBuffer() const;
    
    // ADDED: Performance metrics for dashboard
    struct PerformanceMetrics {
        float totalLatency_us = 0.0f;
        float gpuLatency_us = 0.0f;
        float averageLatency_us = 0.0f;
        float peakLatency_us = 0.0f;
        float qualityLevel = 1.0f;
        uint32_t samplesProcessed = 0;
        uint32_t fftSize = 1024;
        bool spectralProcessingEnabled = true;
        bool neuralProcessingEnabled = true;
    };
    
    PerformanceMetrics getPerformanceMetrics() const;

private:
    MetalBridge();
    ~MetalBridge();

    void loadShaders();
    void createComputePipelines();
    
    // Common member variables accessible from both C++ and Objective-C
    bool initialized = false;
    bool jellieRecordArmed = false;
    bool pnbtrRecordArmed = false;

#ifdef __OBJC__
    void dispatchThreadsForEncoder(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pso, size_t numSamples);
    
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> metalLibrary;
    
    dispatch_semaphore_t frameBoundarySemaphore;
    int currentFrameIndex;

    // --- Pipeline States ---
    id<MTLComputePipelineState> audioInputGatePSO;
    id<MTLComputePipelineState> djAnalysisPSO;
    id<MTLComputePipelineState> recordArmVisualPSO;
    id<MTLComputePipelineState> jelliePreprocessPSO;
    id<MTLComputePipelineState> networkSimPSO;
    id<MTLComputePipelineState> pnbtrReconstructionPSO;

    // --- Metal Buffers (arrays for multi-buffering) ---
    id<MTLBuffer> audioInputBuffer[MAX_FRAMES_IN_FLIGHT];
    id<MTLBuffer> reconstructedBuffer[MAX_FRAMES_IN_FLIGHT];
    
    id<MTLBuffer> gateParamsBuffer[MAX_FRAMES_IN_FLIGHT];
    id<MTLBuffer> stage1Buffer[MAX_FRAMES_IN_FLIGHT];
    
    id<MTLBuffer> djAnalysisParamsBuffer[MAX_FRAMES_IN_FLIGHT];
    id<MTLBuffer> djTransformedBuffer[MAX_FRAMES_IN_FLIGHT];

    id<MTLBuffer> recordArmVisualParamsBuffer[MAX_FRAMES_IN_FLIGHT];
    id<MTLBuffer> stage2Buffer[MAX_FRAMES_IN_FLIGHT];

    id<MTLBuffer> jelliePreprocessParamsBuffer[MAX_FRAMES_IN_FLIGHT];
    id<MTLBuffer> stage3Buffer[MAX_FRAMES_IN_FLIGHT];

    id<MTLBuffer> networkSimParamsBuffer[MAX_FRAMES_IN_FLIGHT];
    id<MTLBuffer> stage4Buffer[MAX_FRAMES_IN_FLIGHT];

    id<MTLBuffer> pnbtrReconstructionParamsBuffer[MAX_FRAMES_IN_FLIGHT];
#endif
};