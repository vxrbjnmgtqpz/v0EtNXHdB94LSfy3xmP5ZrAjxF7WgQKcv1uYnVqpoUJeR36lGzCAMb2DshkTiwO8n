/*
  ==============================================================================

    GPUDSPComponents.h
    Created: GPU-Accelerated DSP Components for ECS

    Extends the ECS DSP system with GPU-accelerated components:
    - GPU-accelerated JELLIE encoding/decoding
    - GPU-accelerated PNBTR enhancement/reconstruction
    - GPU-accelerated audio effects (filters, delays, etc.)
    - Seamless CPU↔GPU data exchange

    Features:
    - Automatic GPU fallback to CPU processing
    - Real-time performance monitoring
    - Hot-swappable GPU/CPU processing modes
    - Unity/Unreal-style GPU component patterns

  ==============================================================================
*/

#pragma once

#include "DSPEntitySystem.h"
#include "../GPU/GPUComputeSystem.h"
#include <memory>
#include <atomic>

//==============================================================================
/**
 * Base GPU DSP Component
 * Extends DSPComponent with GPU processing capabilities
 */
class GPUDSPComponent : public DSPComponent {
public:
    GPUDSPComponent(const std::string& name);
    virtual ~GPUDSPComponent();
    
    //==============================================================================
    // GPU Processing Control
    bool isGPUEnabled() const { return gpuEnabled.load(); }
    void setGPUEnabled(bool enabled);
    
    bool isGPUAvailable() const;
    void setGPUSystem(GPUComputeSystem* gpu) { gpuSystem = gpu; }
    
    //==============================================================================
    // Processing Mode Control
    enum ProcessingMode {
        CPU_ONLY = 0,       // Force CPU processing
        GPU_PREFERRED = 1,  // Try GPU, fallback to CPU
        GPU_ONLY = 2,       // Force GPU processing
        AUTO_SELECT = 3     // Automatic based on performance
    };
    
    void setProcessingMode(ProcessingMode mode) { processingMode = mode; }
    ProcessingMode getProcessingMode() const { return processingMode; }
    
    //==============================================================================
    // Performance Monitoring
    struct GPUPerformanceStats {
        float gpuProcessingTime_us = 0.0f;
        float cpuProcessingTime_us = 0.0f;
        float gpuUploadTime_us = 0.0f;
        float gpuDownloadTime_us = 0.0f;
        
        uint64_t totalGPUProcessed = 0;
        uint64_t totalCPUProcessed = 0;
        uint64_t gpuFailures = 0;
        
        float gpuEfficiency = 1.0f; // GPU time / CPU time
        bool usingGPU = false;
    };
    
    GPUPerformanceStats getGPUStats() const { return gpuStats; }
    
    //==============================================================================
    // DSPComponent interface (GPU-aware)
    void processAudio(AudioBlock& input, AudioBlock& output) override;

protected:
    //==============================================================================
    // GPU Processing Interface (implemented by subclasses)
    virtual bool processAudioGPU(const AudioBlock& input, AudioBlock& output) = 0;
    virtual bool processAudioCPU(const AudioBlock& input, AudioBlock& output) = 0;
    
    // GPU resource management
    virtual bool initializeGPUResources() = 0;
    virtual void cleanupGPUResources() = 0;
    
    //==============================================================================
    // GPU Buffer Management
    GPUBufferID createGPUBuffer(size_t numFrames, size_t numChannels, const std::string& name);
    void destroyGPUBuffer(GPUBufferID bufferID);
    
    bool uploadToGPU(GPUBufferID bufferID, const AudioBlock& audioData);
    bool downloadFromGPU(GPUBufferID bufferID, AudioBlock& audioData);
    
    //==============================================================================
    // Kernel Management
    std::shared_ptr<GPUComputeKernel> getKernel(const std::string& kernelName);
    
    //==============================================================================
    // Performance monitoring
    void updatePerformanceStats(bool usedGPU, float processingTime_us, 
                               float uploadTime_us = 0.0f, float downloadTime_us = 0.0f);

private:
    GPUComputeSystem* gpuSystem = nullptr;
    std::atomic<bool> gpuEnabled{true};
    ProcessingMode processingMode = GPU_PREFERRED;
    
    mutable GPUPerformanceStats gpuStats;
    
    // Adaptive processing mode selection
    void updateProcessingModeSelection();
    float cpuPerformanceBaseline = 100.0f; // Baseline CPU processing time in microseconds
};

//==============================================================================
/**
 * GPU-Accelerated JELLIE Encoder Component
 * Performs audio compression using GPU compute shaders
 */
class GPUJELLIEEncoderComponent : public GPUDSPComponent {
public:
    GPUJELLIEEncoderComponent();
    virtual ~GPUJELLIEEncoderComponent();
    
    //==============================================================================
    // JELLIE Parameters (GPU-optimized)
    void setCompressionRatio(float ratio);
    void setQualityLevel(float quality);
    void setAdaptiveMode(bool enabled);
    void setNetworkLatency(float latency_ms);
    
    float getCompressionRatio() const { return compressionRatio.load(); }
    float getQualityLevel() const { return qualityLevel.load(); }
    bool getAdaptiveMode() const { return adaptiveMode.load(); }
    float getNetworkLatency() const { return networkLatency.load(); }
    
    //==============================================================================
    // Compression Statistics
    struct CompressionStats {
        float actualCompressionRatio = 1.0f;
        float signalToNoiseRatio_dB = 60.0f;
        float bitrateReduction_percent = 0.0f;
        uint64_t totalSamplesProcessed = 0;
    };
    
    CompressionStats getCompressionStats() const { return compressionStats; }

protected:
    //==============================================================================
    // GPUDSPComponent interface
    bool processAudioGPU(const AudioBlock& input, AudioBlock& output) override;
    bool processAudioCPU(const AudioBlock& input, AudioBlock& output) override;
    
    bool initializeGPUResources() override;
    void cleanupGPUResources() override;

private:
    // JELLIE parameters (atomic for real-time updates)
    std::atomic<float> compressionRatio{4.0f};
    std::atomic<float> qualityLevel{0.8f};
    std::atomic<bool> adaptiveMode{true};
    std::atomic<float> networkLatency{5.0f};
    
    // GPU resources
    GPUBufferID inputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID outputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID compressionStateBuffer = INVALID_GPU_BUFFER;
    
    std::shared_ptr<GPUComputeKernel> encoderKernel;
    
    // Compression statistics
    mutable CompressionStats compressionStats;
    
    // CPU fallback implementation
    bool processJELLIEEncoder_CPU(const AudioBlock& input, AudioBlock& output);
};

//==============================================================================
/**
 * GPU-Accelerated JELLIE Decoder Component
 * Performs audio decompression and reconstruction using GPU
 */
class GPUJELLIEDecoderComponent : public GPUDSPComponent {
public:
    GPUJELLIEDecoderComponent();
    virtual ~GPUJELLIEDecoderComponent();
    
    //==============================================================================
    // JELLIE Decoder Parameters
    void setQualityLevel(float quality);
    void setReconstructionMode(bool enabled);
    
    float getQualityLevel() const { return qualityLevel.load(); }
    bool getReconstructionMode() const { return reconstructionMode.load(); }

protected:
    bool processAudioGPU(const AudioBlock& input, AudioBlock& output) override;
    bool processAudioCPU(const AudioBlock& input, AudioBlock& output) override;
    
    bool initializeGPUResources() override;
    void cleanupGPUResources() override;

private:
    std::atomic<float> qualityLevel{0.8f};
    std::atomic<bool> reconstructionMode{true};
    
    GPUBufferID inputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID outputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID reconstructionStateBuffer = INVALID_GPU_BUFFER;
    
    std::shared_ptr<GPUComputeKernel> decoderKernel;
    
    bool processJELLIEDecoder_CPU(const AudioBlock& input, AudioBlock& output);
};

//==============================================================================
/**
 * GPU-Accelerated PNBTR Enhancer Component
 * Performs neural audio enhancement using GPU compute
 */
class GPUPNBTREnhancerComponent : public GPUDSPComponent {
public:
    GPUPNBTREnhancerComponent();
    virtual ~GPUPNBTREnhancerComponent();
    
    //==============================================================================
    // PNBTR Parameters
    void setEnhancementLevel(float level);
    void setModelStrength(float strength);
    void setNeuralMode(int mode); // 0=Tanh, 1=ReLU, 2=Sigmoid
    
    float getEnhancementLevel() const { return enhancementLevel.load(); }
    float getModelStrength() const { return modelStrength.load(); }
    int getNeuralMode() const { return neuralMode.load(); }
    
    //==============================================================================
    // Neural Model Management
    bool loadNeuralWeights(const std::vector<float>& weights);
    void resetNeuralState();

protected:
    bool processAudioGPU(const AudioBlock& input, AudioBlock& output) override;
    bool processAudioCPU(const AudioBlock& input, AudioBlock& output) override;
    
    bool initializeGPUResources() override;
    void cleanupGPUResources() override;

private:
    std::atomic<float> enhancementLevel{0.7f};
    std::atomic<float> modelStrength{0.8f};
    std::atomic<int> neuralMode{0};
    
    GPUBufferID inputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID outputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID neuralWeightsBuffer = INVALID_GPU_BUFFER;
    GPUBufferID neuralStateBuffer = INVALID_GPU_BUFFER;
    
    std::shared_ptr<GPUComputeKernel> enhancerKernel;
    
    // Neural network weights
    std::vector<float> neuralWeights;
    
    bool processPNBTREnhancer_CPU(const AudioBlock& input, AudioBlock& output);
};

//==============================================================================
/**
 * GPU-Accelerated PNBTR Reconstructor Component
 * Performs spectral reconstruction using GPU neural processing
 */
class GPUPNBTRReconstructorComponent : public GPUDSPComponent {
public:
    GPUPNBTRReconstructorComponent();
    virtual ~GPUPNBTRReconstructorComponent();
    
    //==============================================================================
    // Reconstruction Parameters
    void setReconstructionDepth(float depth);
    void setSpectralMode(bool enabled);
    
    float getReconstructionDepth() const { return reconstructionDepth.load(); }
    bool getSpectralMode() const { return spectralMode.load(); }

protected:
    bool processAudioGPU(const AudioBlock& input, AudioBlock& output) override;
    bool processAudioCPU(const AudioBlock& input, AudioBlock& output) override;
    
    bool initializeGPUResources() override;
    void cleanupGPUResources() override;

private:
    std::atomic<float> reconstructionDepth{0.6f};
    std::atomic<bool> spectralMode{true};
    
    GPUBufferID inputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID outputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID spectralBuffer = INVALID_GPU_BUFFER;
    
    std::shared_ptr<GPUComputeKernel> reconstructorKernel;
    
    bool processPNBTRReconstructor_CPU(const AudioBlock& input, AudioBlock& output);
};

//==============================================================================
/**
 * GPU-Accelerated Biquad Filter Component
 * High-performance GPU filtering with multiple filter types
 */
class GPUBiquadFilterComponent : public GPUDSPComponent {
public:
    GPUBiquadFilterComponent();
    virtual ~GPUBiquadFilterComponent();
    
    //==============================================================================
    // Filter Parameters
    enum FilterType {
        LOWPASS = 0,
        HIGHPASS = 1,
        BANDPASS = 2,
        NOTCH = 3
    };
    
    void setCutoffFrequency(float cutoff_hz);
    void setResonance(float resonance);
    void setFilterType(FilterType type);
    
    float getCutoffFrequency() const { return cutoffFrequency.load(); }
    float getResonance() const { return resonance.load(); }
    FilterType getFilterType() const { return static_cast<FilterType>(filterType.load()); }

protected:
    bool processAudioGPU(const AudioBlock& input, AudioBlock& output) override;
    bool processAudioCPU(const AudioBlock& input, AudioBlock& output) override;
    
    bool initializeGPUResources() override;
    void cleanupGPUResources() override;

private:
    std::atomic<float> cutoffFrequency{1000.0f};
    std::atomic<float> resonance{0.707f};
    std::atomic<int> filterType{LOWPASS};
    
    GPUBufferID inputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID outputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID filterStateBuffer = INVALID_GPU_BUFFER;
    
    std::shared_ptr<GPUComputeKernel> filterKernel;
    
    // CPU filter state (per channel)
    struct FilterState {
        float x1 = 0.0f, x2 = 0.0f; // Input history
        float y1 = 0.0f, y2 = 0.0f; // Output history
    };
    std::vector<FilterState> cpuFilterStates;
    
    bool processBiquadFilter_CPU(const AudioBlock& input, AudioBlock& output);
};

//==============================================================================
/**
 * GPU-Accelerated Gain Component
 * Simple but optimized GPU gain processing with smoothing
 */
class GPUGainComponent : public GPUDSPComponent {
public:
    GPUGainComponent();
    virtual ~GPUGainComponent();
    
    //==============================================================================
    // Gain Parameters
    void setGain(float gain_linear);
    void setGain_dB(float gain_dB);
    void setSmoothingTime(float smoothing_ms);
    
    float getGain() const { return gain.load(); }
    float getGain_dB() const;
    float getSmoothingTime() const { return smoothingTime.load(); }

protected:
    bool processAudioGPU(const AudioBlock& input, AudioBlock& output) override;
    bool processAudioCPU(const AudioBlock& input, AudioBlock& output) override;
    
    bool initializeGPUResources() override;
    void cleanupGPUResources() override;

private:
    std::atomic<float> gain{1.0f};
    std::atomic<float> smoothingTime{5.0f}; // milliseconds
    
    GPUBufferID inputBuffer = INVALID_GPU_BUFFER;
    GPUBufferID outputBuffer = INVALID_GPU_BUFFER;
    
    std::shared_ptr<GPUComputeKernel> gainKernel;
    
    // CPU smoothing state
    float currentGain = 1.0f;
    float targetGain = 1.0f;
    
    bool processGain_CPU(const AudioBlock& input, AudioBlock& output);
}; 