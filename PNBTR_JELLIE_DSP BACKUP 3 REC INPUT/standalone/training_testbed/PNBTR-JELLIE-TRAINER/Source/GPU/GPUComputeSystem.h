/*
  ==============================================================================

    GPUComputeSystem.h
    Created: GPU Async Compute System for Audio Processing

    Implements Unity/Unreal-style GPU compute patterns:
    - Async compute dispatch (like Unity ComputeShader.Dispatch)
    - Triple-buffered GPU↔CPU data exchange
    - Metal compute kernels for audio processing
    - Zero-copy shared memory buffers
    - Integration with ECS DSP components

    Features:
    - Non-blocking GPU compute operations
    - Automatic GPU↔CPU synchronization
    - Metal Performance Shaders optimization
    - Real-time audio processing on GPU

  ==============================================================================
*/

#pragma once

#include "../DSP/DSPEntitySystem.h"
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <functional>
#include <unordered_map>
#include <string>

// Forward declarations for Metal objects (to avoid Metal headers in header)
#ifdef __OBJC__
@class MTLDevice;
@class MTLCommandQueue;
@class MTLComputeCommandEncoder;
@class MTLCommandBuffer;
@class MTLBuffer;
@class MTLComputePipelineState;
@class MTLLibrary;
#else
typedef void* MTLDevice;
typedef void* MTLCommandQueue;
typedef void* MTLComputeCommandEncoder;
typedef void* MTLCommandBuffer;
typedef void* MTLBuffer;
typedef void* MTLComputePipelineState;
typedef void* MTLLibrary;
#endif

//==============================================================================
// GPU Buffer Types and Constants

using GPUBufferID = uint32_t;
static constexpr GPUBufferID INVALID_GPU_BUFFER = 0;
static constexpr size_t MAX_GPU_BUFFERS = 256;
static constexpr size_t MAX_AUDIO_FRAMES_GPU = 2048;
static constexpr size_t MAX_GPU_THREADS_PER_GROUP = 256;

//==============================================================================
/**
 * GPU Audio Buffer - Triple-buffered for async processing
 * Like Unity's ComputeBuffer or Unreal's Structured Buffer
 */
struct GPUAudioBuffer {
    GPUBufferID bufferID = INVALID_GPU_BUFFER;
    
    // Triple buffering for async GPU operations
    MTLBuffer* cpuBuffer = nullptr;    // CPU-accessible staging buffer
    MTLBuffer* gpuBuffer = nullptr;    // GPU-only processing buffer  
    MTLBuffer* resultBuffer = nullptr; // GPU→CPU result buffer
    
    size_t sizeBytes = 0;
    size_t numChannels = 2;
    size_t numFrames = 512;
    
    // Synchronization
    std::atomic<bool> gpuProcessing{false};
    std::atomic<bool> resultReady{false};
    uint64_t submissionFrame = 0;
    
    // Buffer state
    bool isAllocated = false;
    std::string debugName;
};

//==============================================================================
/**
 * GPU Compute Kernel - Metal shader wrapper
 * Like Unity's ComputeShader or Unreal's Compute Shader
 */
class GPUComputeKernel {
public:
    GPUComputeKernel(const std::string& kernelName);
    ~GPUComputeKernel();
    
    //==============================================================================
    // Kernel management
    bool compileFromSource(const std::string& metalSource);
    bool loadFromLibrary(const std::string& functionName);
    bool isValid() const { return pipelineState != nullptr; }
    
    //==============================================================================
    // Parameter binding (Unity ComputeShader.SetFloat pattern)
    void setFloat(const std::string& name, float value);
    void setVector(const std::string& name, const float* values, size_t count);
    void setBuffer(const std::string& name, GPUBufferID bufferID);
    
    //==============================================================================
    // Dispatch configuration
    struct DispatchParams {
        size_t threadsPerGroup = 64;
        size_t numGroups = 1;
        size_t totalThreads = 64;
        
        // Audio-specific parameters
        size_t samplesPerThread = 1;
        size_t channelCount = 2;
    };
    
    //==============================================================================
    // Async dispatch (Unity ComputeShader.Dispatch pattern)
    bool dispatchAsync(const DispatchParams& params, 
                      std::function<void(bool success)> completionCallback = nullptr);
    
    //==============================================================================
    // Synchronous dispatch (for debugging)
    bool dispatchSync(const DispatchParams& params);

private:
    std::string kernelName;
    MTLComputePipelineState* pipelineState = nullptr;
    
    // Parameter storage
    std::unordered_map<std::string, float> floatParams;
    std::unordered_map<std::string, std::vector<float>> vectorParams;
    std::unordered_map<std::string, GPUBufferID> bufferParams;
    
    // Metal state
    MTLDevice* device = nullptr;
    MTLCommandQueue* commandQueue = nullptr;
    
    friend class GPUComputeSystem;
};

//==============================================================================
/**
 * GPU Compute System - Main GPU processing manager
 * Like Unity's Graphics class or Unreal's RHI system
 */
class GPUComputeSystem {
public:
    GPUComputeSystem();
    ~GPUComputeSystem();
    
    //==============================================================================
    // System lifecycle
    bool initialize();
    void shutdown();
    bool isInitialized() const { return initialized.load(); }
    
    //==============================================================================
    // Buffer management (Unity ComputeBuffer pattern)
    GPUBufferID createAudioBuffer(size_t numFrames, size_t numChannels, 
                                 const std::string& debugName = "");
    void destroyBuffer(GPUBufferID bufferID);
    
    // Buffer data access
    bool uploadAudioData(GPUBufferID bufferID, const AudioBlock& audioData);
    bool downloadAudioData(GPUBufferID bufferID, AudioBlock& audioData);
    
    // Async buffer operations
    bool uploadAudioDataAsync(GPUBufferID bufferID, const AudioBlock& audioData,
                             std::function<void(bool)> callback = nullptr);
    bool downloadAudioDataAsync(GPUBufferID bufferID, AudioBlock& audioData,
                               std::function<void(bool)> callback = nullptr);
    
    //==============================================================================
    // Kernel management
    std::shared_ptr<GPUComputeKernel> createKernel(const std::string& kernelName);
    std::shared_ptr<GPUComputeKernel> getKernel(const std::string& kernelName);
    
    // Load built-in audio processing kernels
    bool loadAudioKernels();
    
    //==============================================================================
    // Async compute dispatch system
    struct ComputeJob {
        std::shared_ptr<GPUComputeKernel> kernel;
        GPUComputeKernel::DispatchParams params;
        std::function<void(bool)> completionCallback;
        uint64_t submissionFrame;
        std::string jobName;
    };
    
    void submitComputeJob(const ComputeJob& job);
    void waitForCompletion(); // Block until all jobs complete
    void flushCompletedJobs(); // Process completed job callbacks
    
    //==============================================================================
    // ECS Integration - GPU-accelerated DSP components
    void processEntityOnGPU(EntityID entityID, DSPEntitySystem* ecs,
                           const AudioBlock& input, AudioBlock& output);
    
    //==============================================================================
    // Performance monitoring
    struct GPUStats {
        uint64_t totalJobsSubmitted = 0;
        uint64_t totalJobsCompleted = 0;
        uint64_t activeJobs = 0;
        float averageJobTime_ms = 0.0f;
        float peakJobTime_ms = 0.0f;
        
        // GPU memory usage
        size_t totalGPUMemory_bytes = 0;
        size_t usedGPUMemory_bytes = 0;
        size_t allocatedBuffers = 0;
        
        // Performance metrics
        float gpuUtilization = 0.0f;
        float memoryBandwidth_gbps = 0.0f;
    };
    
    GPUStats getStats() const { return stats; }
    
    //==============================================================================
    // Synchronization and frame management
    void beginFrame(uint64_t frameNumber);
    void endFrame();
    uint64_t getCurrentFrame() const { return currentFrame.load(); }

private:
    //==============================================================================
    // Metal device and command infrastructure
    MTLDevice* device = nullptr;
    MTLCommandQueue* commandQueue = nullptr;
    MTLLibrary* shaderLibrary = nullptr;
    
    //==============================================================================
    // Buffer management
    std::unordered_map<GPUBufferID, std::unique_ptr<GPUAudioBuffer>> audioBuffers;
    GPUBufferID nextBufferID = 1;
    std::mutex bufferMutex;
    
    //==============================================================================
    // Kernel management
    std::unordered_map<std::string, std::shared_ptr<GPUComputeKernel>> kernels;
    std::mutex kernelMutex;
    
    //==============================================================================
    // Async compute job system
    std::vector<ComputeJob> pendingJobs;
    std::vector<ComputeJob> completedJobs;
    std::mutex jobMutex;
    
    std::thread computeThread;
    std::atomic<bool> computeThreadRunning{false};
    void computeThreadProc();
    
    //==============================================================================
    // Frame synchronization
    std::atomic<uint64_t> currentFrame{0};
    std::atomic<bool> initialized{false};
    
    //==============================================================================
    // Performance monitoring
    mutable GPUStats stats;
    
    //==============================================================================
    // Built-in audio processing kernels
    bool loadJELLIEKernels();
    bool loadPNBTRKernels();
    bool loadAudioEffectKernels();
    
    //==============================================================================
    // Metal utility functions
    MTLBuffer* createMetalBuffer(size_t sizeBytes, const std::string& debugName);
    bool copyBufferData(MTLBuffer* source, MTLBuffer* destination, size_t sizeBytes);
    
    // Platform-specific GPU device initialization
    bool initializeMetalDevice();
    void shutdownMetalDevice();
    
    // Non-copyable
    GPUComputeSystem(const GPUComputeSystem&) = delete;
    GPUComputeSystem& operator=(const GPUComputeSystem&) = delete;
}; 