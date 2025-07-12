#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h> // This single import is sufficient
#include <string>
#include <vector>
#endif

// Forward declarations to avoid header conflicts in C++ files
#ifndef __OBJC__
class FrameSyncCoordinator;
class GPUCommandFencePool;
class DeferredSignalQueue;
#else
#include "../Core/FrameSyncCoordinator.h"
#include "GPUCommandFencePool.h"
#endif
#include <cmath>

// The number of command buffers in flight, allowing the CPU to get ahead of the GPU.
#define MAX_FRAMES_IN_FLIGHT 3

// ADDED: Biquad parameter structure for anti-aliasing
struct BiquadParams {
    float b0, b1, b2;
    float a1, a2;
    uint  frameOffset;
    uint  numSamples;
    
    // Ensure proper initialization
    BiquadParams() : b0(0), b1(0), b2(0), a1(0), a2(0), frameOffset(0), numSamples(0) {}
};

// ADDED: Biquad coefficient generators
class BiquadCoefficients {
public:
    static BiquadParams makeLowPassBiquad(float Fs, float Fc, float Q);
    static BiquadParams makeHighPassBiquad(float Fs, float Fc, float Q);
    static BiquadParams makeBandPassBiquad(float Fs, float Fc, float Q);
    static BiquadParams makeNotchBiquad(float Fs, float Fc, float Q);
    static BiquadParams makePeakingEQ(float Fs, float Fc, float Q, float gainDB);
    static BiquadParams makeLowShelf(float Fs, float Fc, float Q, float gainDB);
    static BiquadParams makeHighShelf(float Fs, float Fc, float Q, float gainDB);
    
private:
    static float dbToLinear(float db) { return powf(10.0f, db / 20.0f); }
    static constexpr float PI = 3.14159265359f;
};

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
    
    // ADDED: Anti-aliasing filter control
    void setAntiAliasingParams(float cutoffFreq, float Q = 0.707f);
    
    // ADDED: Frame synchronization
    FrameSyncCoordinator* getFrameSyncCoordinator() { return frameSyncCoordinator.get(); }
    
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
        uint64_t currentFrameIndex = 0;  // ADDED: Frame tracking
        float antiAliasingCutoff = 20000.0f;  // ADDED: Anti-aliasing cutoff frequency
    };
    
    PerformanceMetrics getPerformanceMetrics() const;

private:
    MetalBridge();
    ~MetalBridge();

    void loadShaders();
    void createComputePipelines();
    
    // ADDED: Anti-aliasing methods
    void executeStage1_5_AntiAliasing(size_t numSamples);
    void updateAntiAliasingCoefficients();
    
    // ADDED: Frame synchronization methods
    void beginNewAudioFrame();
    void runSevenStageProcessingPipelineWithSync(size_t numSamples);
    void onGPUProcessingComplete(uint64_t frameIndex);
    void onGPUFrameComplete(uint64_t frameIndex, size_t numSamples);
    void onAntiAliasingComplete(uint64_t frameIndex);  // ADDED: Anti-aliasing completion handler
    
    // ADDED: Enhanced async completion handlers with error handling
    void onAntiAliasingCompleteWithValidation(uint64_t frameIndex, bool success, float processingTime_us);
    void onAntiAliasingError(uint64_t frameIndex, const std::string& error);
    
    // ADDED: Frame state validation for buffer safety
    bool validateFrameState(uint64_t frameIndex, const std::string& stageName);
    void markFrameStageComplete(uint64_t frameIndex, const std::string& stageName);
    bool isFrameStageComplete(uint64_t frameIndex, const std::string& stageName);
    
    // Common member variables accessible from both C++ and Objective-C
    bool initialized = false;
    bool jellieRecordArmed = false;
    bool pnbtrRecordArmed = false;
    
    // ADDED: Anti-aliasing parameters
    float sampleRate = 48000.0f;
    BiquadParams currentBiquadParams;
    
    // ADDED: Frame state validation flags (atomic for thread safety)
    std::atomic<uint64_t> lastAntiAliasingFrame{0};
    std::atomic<bool> antiAliasingInProgress{false};
    std::atomic<float> lastAntiAliasingLatency_us{0.0f};
    std::atomic<uint32_t> antiAliasingErrorCount{0};
    
    // ADDED: Frame synchronization components (using unique_ptr to avoid atomic copy issues)
    std::unique_ptr<FrameSyncCoordinator> frameSyncCoordinator;
    std::unique_ptr<DeferredSignalQueue> deferredSignalQueue;

#ifdef __OBJC__
    void dispatchThreadsForEncoder(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pso, size_t numSamples);
    void executeSevenStageProcessingPipeline(id<MTLCommandBuffer> commandBuffer, size_t numSamples, uint64_t currentFrame);
    
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> metalLibrary;
    
    dispatch_semaphore_t frameBoundarySemaphore;
    int currentFrameIndex;
    
    // ADDED: GPU fence pool for frame synchronization
    std::unique_ptr<GPUCommandFencePool> fencePool;
    
    // ADDED: MTLSharedEvent for low-latency completion signaling
    id<MTLSharedEvent> antiAliasingCompletionEvent;
    uint64_t antiAliasingEventValue;
    
    // ADDED: Performance monitoring for async operations
    std::chrono::high_resolution_clock::time_point antiAliasingStartTime;
    std::atomic<uint32_t> totalAntiAliasingFrames{0};
    std::atomic<float> averageAntiAliasingLatency_us{0.0f};

    // --- Pipeline States ---
    id<MTLComputePipelineState> audioInputCapturePSO;
    id<MTLComputePipelineState> audioAntiAliasPSO;  // ADDED: Anti-aliasing pipeline state
    id<MTLComputePipelineState> audioInputGatePSO;
    id<MTLComputePipelineState> djAnalysisPSO;
    id<MTLComputePipelineState> recordArmVisualPSO;
    id<MTLComputePipelineState> jelliePreprocessPSO;
    id<MTLComputePipelineState> networkSimPSO;
    id<MTLComputePipelineState> pnbtrReconstructionPSO;

    // --- Metal Buffers (arrays for multi-buffering) ---
    id<MTLBuffer> audioInputBuffer[MAX_FRAMES_IN_FLIGHT];
    id<MTLBuffer> antiAliasedBuffer[MAX_FRAMES_IN_FLIGHT];  // ADDED: Anti-aliased buffer
    id<MTLBuffer> reconstructedBuffer[MAX_FRAMES_IN_FLIGHT];
    
    // ADDED: Anti-aliasing state buffers
    id<MTLBuffer> antiAliasStateXBuffer;
    id<MTLBuffer> antiAliasStateYBuffer;
    id<MTLBuffer> biquadParamsBuffer;
    id<MTLBuffer> antiAliasDebugBuffer;  // ADDED: Debug buffer for anti-aliasing inspection
    
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
    
    // ADDED: Frame index buffer for GPU shaders
    id<MTLBuffer> frameIndexBuffer[MAX_FRAMES_IN_FLIGHT];
#endif
};