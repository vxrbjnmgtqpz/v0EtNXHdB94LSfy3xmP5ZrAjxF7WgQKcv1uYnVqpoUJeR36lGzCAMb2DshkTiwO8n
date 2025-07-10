/*
  ==============================================================================

    MultiGPUSystem.h
    Created: Phase 4C - Advanced Multi-GPU Processing System

    Professional multi-GPU compute system for maximum performance:
    - Automatic GPU discovery and enumeration
    - Intelligent load balancing across multiple GPUs
    - Unified memory management with peer-to-peer transfers
    - SIMD-optimized Metal kernels with advanced scheduling
    - Real-time performance profiling and optimization

    Features:
    - Multi-GPU workload distribution (up to 4 GPUs)
    - Automatic load balancing based on GPU capabilities
    - Zero-copy peer-to-peer memory transfers
    - Advanced SIMD instruction optimization
    - Unity/Unreal-style GPU resource management

    Based on modern game engine multi-GPU architectures.

  ==============================================================================
*/

#pragma once

#include "GPUComputeSystem.h"
#include "TripleBufferSystem.h"
#include <vector>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>

//==============================================================================
// Multi-GPU Constants
static constexpr size_t MAX_GPU_COUNT = 4;
static constexpr size_t SIMD_VECTOR_SIZE = 8;  // Metal SIMD8 optimization
static constexpr size_t GPU_MEMORY_POOL_SIZE = 512 * 1024 * 1024; // 512MB per GPU

//==============================================================================
// GPU Device Information
struct GPUDeviceInfo {
    std::string deviceName;
    size_t maxMemory_bytes = 0;
    size_t availableMemory_bytes = 0;
    bool supportsUnifiedMemory = false;
    bool supportsSIMD = false;
    bool supportsAsync = false;
    
    // Performance characteristics
    float computePerformance = 1.0f;    // Relative performance rating
    float memoryBandwidth_gbps = 0.0f;  // Memory bandwidth in GB/s
    size_t maxThreadsPerGroup = 0;
    size_t maxThreadGroups = 0;
    
    // Utilization tracking
    std::atomic<float> currentUtilization{0.0f};
    std::atomic<size_t> activeJobs{0};
    std::atomic<uint64_t> totalJobsProcessed{0};
    
    bool isValid() const { return maxMemory_bytes > 0; }
};

//==============================================================================
// Multi-GPU Job System
struct MultiGPUJob {
    uint32_t jobID = 0;
    std::string jobName;
    
    // Job data and parameters
    std::shared_ptr<GPUComputeKernel> kernel;
    GPUComputeKernel::DispatchParams params;
    
    // Memory requirements
    std::vector<GPUBufferID> inputBuffers;
    std::vector<GPUBufferID> outputBuffers;
    size_t estimatedMemoryUsage_bytes = 0;
    
    // Scheduling parameters
    uint32_t priority = 0;              // Higher = more important
    uint32_t targetGPU = UINT32_MAX;    // Specific GPU or auto-select
    bool allowMultiGPU = true;          // Can be split across GPUs
    
    // Performance tracking
    uint64_t submissionTime_ns = 0;
    uint64_t startTime_ns = 0;
    uint64_t completionTime_ns = 0;
    uint32_t assignedGPU = UINT32_MAX;
    
    // Completion callback
    std::function<void(bool success, const MultiGPUJob& job)> completionCallback;
    
    bool isValid() const { return kernel != nullptr; }
};

//==============================================================================
// Load Balancing Strategy
enum class LoadBalancingStrategy {
    ROUND_ROBIN = 0,        // Simple round-robin assignment
    PERFORMANCE_BASED = 1,  // Assign based on GPU performance
    MEMORY_AWARE = 2,       // Consider memory usage
    ADAPTIVE = 3,           // Dynamic based on current load
    LATENCY_OPTIMIZED = 4   // Minimize latency
};

//==============================================================================
/**
 * Multi-GPU Compute System - Professional multi-GPU processing
 * Provides Unity/Unreal-style multi-GPU compute with automatic load balancing
 */
class MultiGPUSystem {
public:
    MultiGPUSystem();
    ~MultiGPUSystem();
    
    //==============================================================================
    // System lifecycle
    bool initialize();
    void shutdown();
    
    //==============================================================================
    // GPU device management
    size_t getGPUCount() const { return gpuDevices.size(); }
    const GPUDeviceInfo& getGPUInfo(size_t gpuIndex) const;
    std::vector<GPUDeviceInfo> getAllGPUInfo() const;
    
    // GPU selection and load balancing
    void setLoadBalancingStrategy(LoadBalancingStrategy strategy) { loadBalancingStrategy = strategy; }
    LoadBalancingStrategy getLoadBalancingStrategy() const { return loadBalancingStrategy; }
    
    uint32_t selectOptimalGPU(const MultiGPUJob& job) const;
    bool canJobFitOnGPU(const MultiGPUJob& job, uint32_t gpuIndex) const;
    
    //==============================================================================
    // Job submission and execution
    uint32_t submitJob(const MultiGPUJob& job);
    bool submitJobToGPU(const MultiGPUJob& job, uint32_t gpuIndex);
    
    // Multi-GPU job splitting
    std::vector<MultiGPUJob> splitJobAcrossGPUs(const MultiGPUJob& job, size_t numGPUs) const;
    bool submitMultiGPUJob(const MultiGPUJob& job);
    
    // Job management
    bool cancelJob(uint32_t jobID);
    bool isJobComplete(uint32_t jobID) const;
    void waitForJob(uint32_t jobID, uint32_t timeout_ms = 1000);
    void waitForAllJobs(uint32_t timeout_ms = 5000);
    
    //==============================================================================
    // Memory management
    GPUBufferID createUnifiedBuffer(size_t size, const std::string& name = "MultiGPU_Buffer");
    bool copyBufferBetweenGPUs(GPUBufferID buffer, uint32_t srcGPU, uint32_t dstGPU);
    bool synchronizeBufferAcrossGPUs(GPUBufferID buffer);
    
    // Memory pool management
    void preallocateMemoryPools();
    GPUBufferID allocateFromPool(size_t size, uint32_t gpuIndex);
    void returnToPool(GPUBufferID buffer, uint32_t gpuIndex);
    
    //==============================================================================
    // Performance monitoring and optimization
    struct MultiGPUStats {
        size_t totalGPUs = 0;
        size_t activeGPUs = 0;
        
        // Job statistics
        uint64_t totalJobsSubmitted = 0;
        uint64_t totalJobsCompleted = 0;
        uint64_t totalJobsFailed = 0;
        
        // Performance metrics
        float averageJobTime_ms = 0.0f;
        float peakJobTime_ms = 0.0f;
        float totalThroughput_jobsPerSecond = 0.0f;
        
        // Load balancing effectiveness
        float loadBalanceEfficiency = 1.0f;  // 1.0 = perfect balance
        std::vector<float> gpuUtilizations;
        
        // Memory usage
        size_t totalMemoryAllocated_mb = 0;
        size_t totalMemoryUsed_mb = 0;
        float memoryFragmentation = 0.0f;
        
        // Real-time constraints
        bool realTimeConstraintsMet = true;
        float worstCaseLatency_ms = 0.0f;
    };
    
    MultiGPUStats getStats() const { return stats; }
    void resetStats();
    
    // Performance optimization
    void optimizeLoadBalancing();
    void defragmentMemory();
    
    //==============================================================================
    // Advanced features
    
    // Streaming computation for large datasets
    bool submitStreamingJob(const MultiGPUJob& baseJob, size_t streamChunkSize, 
                           const std::vector<GPUBufferID>& inputStreams);
    
    // AI/Neural network support
    bool loadNeuralNetwork(const std::string& modelPath, const std::string& networkName);
    bool runInference(const std::string& networkName, const std::vector<GPUBufferID>& inputs,
                     std::vector<GPUBufferID>& outputs);
    
    // Advanced synchronization
    void synchronizeAllGPUs();
    bool areAllGPUsIdle() const;
    
    //==============================================================================
    // SIMD optimization support
    bool enableSIMDOptimization(bool enable) { simdOptimizationEnabled = enable; return true; }
    bool isSIMDOptimizationEnabled() const { return simdOptimizationEnabled; }
    
    // Advanced kernel compilation with SIMD
    std::shared_ptr<GPUComputeKernel> createOptimizedKernel(const std::string& kernelName,
                                                          const std::vector<std::string>& optimizationFlags = {});

private:
    //==============================================================================
    // GPU device management
    std::vector<std::unique_ptr<GPUComputeSystem>> gpuSystems;
    std::vector<GPUDeviceInfo> gpuDevices;
    
    //==============================================================================
    // Job scheduling and execution
    std::atomic<uint32_t> nextJobID{1};
    std::unordered_map<uint32_t, MultiGPUJob> activeJobs;
    std::unordered_map<uint32_t, MultiGPUJob> completedJobs;
    
    // Job queues per GPU
    std::vector<std::queue<MultiGPUJob>> gpuJobQueues;
    std::vector<std::mutex> gpuQueueMutexes;
    std::vector<std::condition_variable> gpuQueueConditions;
    
    // Worker threads
    std::vector<std::thread> workerThreads;
    std::atomic<bool> shutdownRequested{false};
    
    //==============================================================================
    // Load balancing
    LoadBalancingStrategy loadBalancingStrategy = LoadBalancingStrategy::ADAPTIVE;
    
    //==============================================================================
    // Memory management
    std::vector<std::vector<GPUBufferID>> memoryPools; // Per GPU memory pools
    std::vector<std::mutex> memoryPoolMutexes;
    
    //==============================================================================
    // Performance monitoring
    mutable MultiGPUStats stats;
    mutable std::mutex statsMutex;
    
    //==============================================================================
    // Advanced features
    bool simdOptimizationEnabled = true;
    std::unordered_map<std::string, std::vector<float>> neuralNetworks; // Simplified storage
    
    //==============================================================================
    // Initialization helpers
    bool discoverGPUDevices();
    bool initializeGPUSystem(size_t gpuIndex);
    void startWorkerThreads();
    void stopWorkerThreads();
    
    // Job execution
    void workerThreadFunction(uint32_t gpuIndex);
    bool executeJobOnGPU(const MultiGPUJob& job, uint32_t gpuIndex);
    
    // Load balancing implementation
    uint32_t selectGPU_RoundRobin() const;
    uint32_t selectGPU_PerformanceBased(const MultiGPUJob& job) const;
    uint32_t selectGPU_MemoryAware(const MultiGPUJob& job) const;
    uint32_t selectGPU_Adaptive(const MultiGPUJob& job) const;
    uint32_t selectGPU_LatencyOptimized(const MultiGPUJob& job) const;
    
    // Performance tracking
    void updateJobStats(const MultiGPUJob& job);
    void updateGPUUtilization();
    void updateLoadBalanceEfficiency();
    
    // Memory management helpers
    bool initializeMemoryPool(uint32_t gpuIndex);
    void cleanupMemoryPool(uint32_t gpuIndex);
    
    // Non-copyable
    MultiGPUSystem(const MultiGPUSystem&) = delete;
    MultiGPUSystem& operator=(const MultiGPUSystem&) = delete;
};

//==============================================================================
/**
 * SIMD-Optimized Audio Processing - Metal SIMD8/SIMD16 kernels
 * Advanced vectorized audio processing for maximum performance
 */
class SIMDAudioProcessor {
public:
    SIMDAudioProcessor();
    ~SIMDAudioProcessor();
    
    bool initialize(MultiGPUSystem* multiGPU);
    void shutdown();
    
    //==============================================================================
    // SIMD-optimized audio operations
    bool processAudioSIMD8(const AudioBlock& input, AudioBlock& output, 
                          const std::string& operation);
    bool processAudioSIMD16(const AudioBlock& input, AudioBlock& output,
                           const std::string& operation);
    
    // Vectorized DSP operations
    bool vectorizedFFT(const AudioBlock& input, std::vector<float>& magnitude, 
                      std::vector<float>& phase);
    bool vectorizedConvolution(const AudioBlock& input, const std::vector<float>& impulse,
                              AudioBlock& output);
    bool vectorizedBiquadFilter(const AudioBlock& input, AudioBlock& output,
                               const std::vector<float>& coefficients);
    
    //==============================================================================
    // Performance optimization
    void optimizeForCurrentHardware();
    bool isSIMD8Supported() const { return simd8Supported; }
    bool isSIMD16Supported() const { return simd16Supported; }

private:
    MultiGPUSystem* multiGPUSystem = nullptr;
    
    // SIMD capability detection
    bool simd8Supported = false;
    bool simd16Supported = false;
    
    // Optimized kernels
    std::shared_ptr<GPUComputeKernel> simd8AudioKernel;
    std::shared_ptr<GPUComputeKernel> simd16AudioKernel;
    std::shared_ptr<GPUComputeKernel> simdFFTKernel;
    std::shared_ptr<GPUComputeKernel> simdConvolutionKernel;
    
    bool loadSIMDKernels();
    bool detectSIMDCapabilities();
};

//==============================================================================
/**
 * Streaming Buffer System - Large file processing with memory mapping
 * Professional streaming system for processing large audio files
 */
class StreamingBufferSystem {
public:
    StreamingBufferSystem();
    ~StreamingBufferSystem();
    
    bool initialize(MultiGPUSystem* multiGPU, size_t maxStreamSize_mb = 1024);
    void shutdown();
    
    //==============================================================================
    // Streaming operations
    bool openAudioStream(const std::string& filePath, const std::string& streamName);
    bool processAudioStream(const std::string& streamName, 
                           std::shared_ptr<GPUComputeKernel> processor,
                           const std::string& outputPath);
    
    // Memory-mapped streaming
    bool createMemoryMappedStream(const std::string& streamName, size_t size_bytes);
    bool streamChunk(const std::string& streamName, size_t chunkIndex, 
                    const MultiGPUJob& processingJob);
    
    //==============================================================================
    // Stream management
    void closeStream(const std::string& streamName);
    bool isStreamActive(const std::string& streamName) const;
    size_t getStreamProgress(const std::string& streamName) const; // Returns percentage

private:
    MultiGPUSystem* multiGPUSystem = nullptr;
    size_t maxStreamSize_bytes = 0;
    
    struct StreamInfo {
        std::string filePath;
        size_t totalSize_bytes = 0;
        size_t currentPosition = 0;
        size_t chunkSize_bytes = 0;
        bool isActive = false;
        
        // Memory mapping
        void* mappedMemory = nullptr;
        std::vector<GPUBufferID> gpuBuffers;
    };
    
    std::unordered_map<std::string, StreamInfo> activeStreams;
    std::mutex streamsMutex;
}; 