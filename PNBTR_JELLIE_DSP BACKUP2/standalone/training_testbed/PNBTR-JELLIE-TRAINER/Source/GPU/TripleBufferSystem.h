/*
  ==============================================================================

    TripleBufferSystem.h
    Created: Enhanced Triple-Buffering for GPU↔CPU Synchronization

    Advanced triple-buffering system for real-time audio processing:
    - Lock-free triple-buffered GPU↔CPU data exchange
    - Zero-copy shared memory buffers (Metal MTLResourceStorageModeShared)
    - Atomic synchronization with minimal latency
    - Real-time performance guarantees for audio thread

    Features:
    - Sub-microsecond buffer swapping
    - Automatic GPU↔CPU coherency management
    - Memory bandwidth optimization
    - Lock-free reader/writer pattern

    Based on Unity/Unreal triple-buffering patterns for real-time graphics/audio.

  ==============================================================================
*/

#pragma once

#include "GPUComputeSystem.h"
#include "../DSP/DSPEntitySystem.h"
#include <atomic>
#include <memory>
#include <array>
#include <functional>

//==============================================================================
// Triple Buffer Constants
static constexpr size_t TRIPLE_BUFFER_COUNT = 3;
static constexpr size_t BUFFER_READ = 0;     // CPU reading (completed GPU processing)
static constexpr size_t BUFFER_WRITE = 1;    // CPU writing (preparing for GPU)
static constexpr size_t BUFFER_GPU = 2;      // GPU processing (active computation)

//==============================================================================
/**
 * Triple Buffer Slot - Single buffer in the rotation
 * Manages one buffer in the triple-buffering system
 */
struct TripleBufferSlot {
    GPUBufferID bufferID = INVALID_GPU_BUFFER;
    
    // Buffer state management
    std::atomic<bool> cpuReady{false};        // CPU finished writing
    std::atomic<bool> gpuProcessing{false};   // GPU actively processing
    std::atomic<bool> gpuComplete{false};     // GPU finished processing
    std::atomic<bool> cpuReading{false};      // CPU actively reading
    
    // Synchronization
    uint64_t frameNumber = 0;
    uint64_t gpuSubmissionTime_ns = 0;
    uint64_t gpuCompletionTime_ns = 0;
    
    // Performance tracking
    float processingTime_us = 0.0f;
    size_t dataSize_bytes = 0;
    
    std::string debugName;
    
    void reset() {
        cpuReady = false;
        gpuProcessing = false;
        gpuComplete = false;
        cpuReading = false;
        frameNumber = 0;
        gpuSubmissionTime_ns = 0;
        gpuCompletionTime_ns = 0;
        processingTime_us = 0.0f;
    }
};

//==============================================================================
/**
 * Triple Buffer Manager - Lock-free triple-buffering for GPU↔CPU
 * Provides Unity/Unreal-style triple-buffering for real-time processing
 */
class TripleBufferManager {
public:
    TripleBufferManager();
    ~TripleBufferManager();
    
    //==============================================================================
    // System lifecycle
    bool initialize(GPUComputeSystem* gpu, size_t bufferSize, size_t numChannels,
                   const std::string& bufferName = "TripleBuffer");
    void shutdown();
    
    //==============================================================================
    // Triple-buffered processing (lock-free)
    
    // CPU Producer API (audio input thread)
    TripleBufferSlot* beginCPUWrite(uint64_t frameNumber);
    bool uploadAudioData(TripleBufferSlot* slot, const AudioBlock& audioData);
    void endCPUWrite(TripleBufferSlot* slot);
    
    // GPU Processing API (async)
    TripleBufferSlot* beginGPUProcessing();
    bool processOnGPU(TripleBufferSlot* slot, std::shared_ptr<GPUComputeKernel> kernel,
                     const GPUComputeKernel::DispatchParams& params);
    void endGPUProcessing(TripleBufferSlot* slot, bool success);
    
    // CPU Consumer API (audio output thread)
    TripleBufferSlot* beginCPURead();
    bool downloadAudioData(TripleBufferSlot* slot, AudioBlock& audioData);
    void endCPURead(TripleBufferSlot* slot);
    
    //==============================================================================
    // Buffer rotation management
    void rotateBuffers();
    bool isDataReady() const;
    
    //==============================================================================
    // Performance monitoring
    struct TripleBufferStats {
        float averageGPUTime_us = 0.0f;
        float peakGPUTime_us = 0.0f;
        float averageUploadTime_us = 0.0f;
        float averageDownloadTime_us = 0.0f;
        
        uint64_t totalFramesProcessed = 0;
        uint64_t droppedFrames = 0;
        uint64_t bufferUnderruns = 0;
        
        float memoryBandwidth_mbps = 0.0f;
        float bufferUtilization = 0.0f;
        
        // Real-time safety metrics
        float maxBufferSwapTime_us = 0.0f;
        bool realTimeSafe = true;
    };
    
    TripleBufferStats getStats() const { return stats; }
    void resetStats();
    
    //==============================================================================
    // Advanced synchronization
    void waitForGPUCompletion(uint32_t timeout_ms = 10);
    bool isGPUBusy() const;
    
    // Frame timing
    uint64_t getCurrentFrame() const { return currentFrame.load(); }
    void setFrameNumber(uint64_t frame) { currentFrame = frame; }

private:
    GPUComputeSystem* gpuSystem = nullptr;
    
    //==============================================================================
    // Triple buffer storage
    std::array<std::unique_ptr<TripleBufferSlot>, TRIPLE_BUFFER_COUNT> buffers;
    
    // Atomic buffer indices (lock-free rotation)
    std::atomic<size_t> readIndex{0};     // CPU reading completed GPU data
    std::atomic<size_t> writeIndex{1};    // CPU writing new data for GPU
    std::atomic<size_t> gpuIndex{2};      // GPU processing data
    
    //==============================================================================
    // Buffer configuration
    size_t bufferSize = 512;
    size_t numChannels = 2;
    size_t bufferSizeBytes = 0;
    std::string bufferName;
    
    //==============================================================================
    // Frame synchronization
    std::atomic<uint64_t> currentFrame{0};
    std::atomic<uint64_t> lastCompletedFrame{0};
    
    //==============================================================================
    // Performance monitoring
    mutable TripleBufferStats stats;
    
    //==============================================================================
    // Buffer management helpers
    bool createBufferSlot(size_t index);
    void destroyBufferSlot(size_t index);
    
    // Atomic buffer rotation (lock-free)
    void atomicRotateIndices();
    
    // Performance tracking
    void updatePerformanceStats(TripleBufferSlot* slot);
    
    // Real-time safety validation
    bool validateRealTimeSafety() const;
    
    // Non-copyable
    TripleBufferManager(const TripleBufferManager&) = delete;
    TripleBufferManager& operator=(const TripleBufferManager&) = delete;
};

//==============================================================================
/**
 * GPU Visualization System - Real-time waveform and spectrum rendering
 * Implements Unity/Unreal-style GPU-based audio visualization
 */
class GPUVisualizationSystem {
public:
    GPUVisualizationSystem();
    ~GPUVisualizationSystem();
    
    //==============================================================================
    // Visualization types
    enum VisualizationType {
        WAVEFORM = 0,           // Time-domain waveform
        SPECTRUM = 1,           // Frequency spectrum (FFT)
        SPECTROGRAM = 2,        // Time-frequency spectrogram
        VECTORSCOPE = 3,        // Stereo vectorscope
        PHASE_CORRELATION = 4,  // Phase correlation display
        LEVEL_METERS = 5        // Peak/RMS level meters
    };
    
    //==============================================================================
    // GPU visualization buffers
    struct VisualizationBuffer {
        GPUBufferID pixelBuffer = INVALID_GPU_BUFFER;
        size_t width = 0;
        size_t height = 0;
        size_t channels = 4; // RGBA
        
        // Buffer management
        bool isValid() const { return pixelBuffer != INVALID_GPU_BUFFER; }
        size_t sizeBytes() const { return width * height * channels * sizeof(float); }
    };
    
    //==============================================================================
    // System lifecycle
    bool initialize(GPUComputeSystem* gpu, size_t maxDisplayWidth = 1920, 
                   size_t maxDisplayHeight = 1080);
    void shutdown();
    
    //==============================================================================
    // Visualization rendering (GPU-accelerated)
    bool renderWaveform(const AudioBlock& audioData, VisualizationBuffer& output,
                       size_t displayWidth, size_t displayHeight);
    
    bool renderSpectrum(const AudioBlock& audioData, VisualizationBuffer& output,
                       size_t displayWidth, size_t displayHeight, size_t fftSize = 1024);
    
    bool renderSpectrogram(const AudioBlock& audioData, VisualizationBuffer& output,
                          size_t displayWidth, size_t displayHeight);
    
    bool renderVectorscope(const AudioBlock& audioData, VisualizationBuffer& output,
                          size_t displayWidth, size_t displayHeight);
    
    //==============================================================================
    // Real-time visualization parameters
    struct VisualizationParams {
        float gainScale = 1.0f;
        float timeScale = 1.0f;
        float frequencyScale = 1.0f;
        
        bool logScale = true;
        bool smoothing = true;
        float smoothingFactor = 0.8f;
        
        // Color mapping
        float hueShift = 0.0f;
        float saturation = 1.0f;
        float brightness = 1.0f;
        
        // Display options
        bool showGrid = true;
        bool showLabels = true;
        size_t maxDisplayFPS = 60;
    };
    
    void setVisualizationParams(const VisualizationParams& params) { vizParams = params; }
    VisualizationParams getVisualizationParams() const { return vizParams; }
    
    //==============================================================================
    // Performance optimization
    void setMaxRenderFPS(size_t fps) { maxRenderFPS = fps; }
    bool shouldRender() const; // Frame rate limiting
    
    VisualizationBuffer createVisualizationBuffer(size_t width, size_t height);
    void destroyVisualizationBuffer(VisualizationBuffer& buffer);
    
    //==============================================================================
    // Advanced visualization features
    bool renderMultiChannelWaveform(const std::vector<AudioBlock>& channelData,
                                   VisualizationBuffer& output);
    
    bool renderFrequencyResponse(const std::vector<float>& frequencyResponse,
                                VisualizationBuffer& output);
    
    // Real-time spectrum analysis
    bool updateSpectrumAnalysis(const AudioBlock& audioData);
    const std::vector<float>& getCurrentSpectrum() const { return currentSpectrum; }

private:
    GPUComputeSystem* gpuSystem = nullptr;
    
    //==============================================================================
    // GPU visualization kernels
    std::shared_ptr<GPUComputeKernel> waveformKernel;
    std::shared_ptr<GPUComputeKernel> spectrumKernel;
    std::shared_ptr<GPUComputeKernel> spectrogramKernel;
    std::shared_ptr<GPUComputeKernel> vectorscopeKernel;
    std::shared_ptr<GPUComputeKernel> fftKernel;
    
    //==============================================================================
    // Visualization parameters
    VisualizationParams vizParams;
    size_t maxDisplayWidth = 1920;
    size_t maxDisplayHeight = 1080;
    
    //==============================================================================
    // Frame rate management
    size_t maxRenderFPS = 60;
    uint64_t lastRenderTime_ns = 0;
    uint64_t renderInterval_ns = 16666667; // 60 FPS = 16.67ms
    
    //==============================================================================
    // Spectrum analysis state
    std::vector<float> currentSpectrum;
    std::vector<float> previousSpectrum;
    GPUBufferID fftBuffer = INVALID_GPU_BUFFER;
    GPUBufferID windowBuffer = INVALID_GPU_BUFFER;
    
    //==============================================================================
    // GPU buffer management
    std::vector<VisualizationBuffer> activeBuffers;
    
    //==============================================================================
    // Kernel loading and initialization
    bool loadVisualizationKernels();
    bool initializeFFTResources(size_t maxFFTSize = 2048);
    
    // Rendering helpers
    bool renderVisualization(VisualizationType type, const AudioBlock& audioData,
                           VisualizationBuffer& output, size_t width, size_t height);
    
    // Performance optimization
    void updateFrameRateLimit();
    
    // Non-copyable
    GPUVisualizationSystem(const GPUVisualizationSystem&) = delete;
    GPUVisualizationSystem& operator=(const GPUVisualizationSystem&) = delete;
}; 