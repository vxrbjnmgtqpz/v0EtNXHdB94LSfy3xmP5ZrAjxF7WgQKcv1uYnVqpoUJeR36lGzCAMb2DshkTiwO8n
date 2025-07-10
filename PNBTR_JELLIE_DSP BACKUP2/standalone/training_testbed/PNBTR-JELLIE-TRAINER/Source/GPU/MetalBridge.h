/*
  ==============================================================================

    MetalBridge.h
    Created: GPU-native Metal bridge for zero-copy processing

    UPDATED: Manages 7-stage Metal compute pipeline for:
    - Stage 1: Input Capture (record-armed audio with gain control)
    - Stage 2: Input Gating (noise suppression and signal detection) 
    - Stage 3: DJ-Style Spectral Analysis (real-time FFT with color mapping)
    - Stage 4: Record Arm Visual (animated record-arm feedback)
    - Stage 5: JELLIE Preprocessing (prepare audio for neural processing)
    - Stage 6: Network Simulation (packet loss and jitter simulation)
    - Stage 7: PNBTR Reconstruction (neural prediction and audio restoration)

  ==============================================================================
*/

#pragma once

#ifdef __OBJC__
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#endif

#include "MetalBridgeInterface.h"
#include <string>
#include <memory>


class MetalBridge : public MetalBridgeInterface {
public:
    // Singleton access
    static MetalBridge& getInstance();

    MetalBridge(const MetalBridge&) = delete;
    MetalBridge& operator=(const MetalBridge&) = delete;

    // Public API (add your methods here)
    bool initialize();
    void cleanup();
    void shutdown() { cleanup(); }
    bool isInitialized() const;
    void setProcessingParameters(double sampleRate, int samplesPerBlock);
    void prepareBuffers(int samplesPerBlock, double sampleRate);

#ifdef __OBJC__
    id<MTLBuffer> createSharedBuffer(size_t size);
    void updateAudioBuffers(size_t bufferSize, size_t numChannels);
#else
    void* createSharedBuffer(size_t) { return nullptr; }
    void updateAudioBuffers(size_t, size_t) {}
#endif

    // MetalBridgeInterface implementation
    const float* getAudioInputBuffer(size_t& bufferSize) override;
    const float* getJellieBuffer(size_t& bufferSize) override;
    const float* getNetworkBuffer(size_t& bufferSize) override;
    const float* getReconstructedBuffer(size_t& bufferSize) override;
    AudioMetrics getLatestMetrics() override;
    bool isSessionActive() override;
    void startSession() override;
    void stopSession() override;

#ifdef __OBJC__
    id<MTLBuffer> getInputBuffer() const;
    id<MTLBuffer> getJellieBuffer() const;
    id<MTLBuffer> getNetworkBuffer() const;
    id<MTLBuffer> getReconstructedBuffer() const;
    void dispatchKernel(const std::string& kernelName, id<MTLBuffer> inputBuffer, id<MTLBuffer> outputBuffer, size_t threadCount);
    void updateWaveformTexture(id<MTLTexture> texture, size_t width, size_t height);
    id<MTLDevice> getDevice() const;
    id<MTLCommandQueue> getCommandQueue() const;
    
    // NEW: 7-stage processing pipeline methods
    void uploadInputToGPU(const float* input, size_t numSamples);
    void runSevenStageProcessingPipeline(size_t numSamples);
    void downloadOutputFromGPU(float* output, size_t numSamples);
    void dispatchThreadsForEncoder(id<MTLComputeCommandEncoder> encoder, 
                                   id<MTLComputePipelineState> pipeline, 
                                   size_t numSamples);
#endif

    void processAudioBlock(const float* input, float* output, size_t numSamples);
    bool runJellieEncode();
    bool runNetworkSimulation(float packetLoss, float jitter);
    bool runPNBTRReconstruction();
    void updateMetrics();
    void getMetricsData(float* snr, float* thd, float* latency);
    void getPacketLossStats(int* totalPackets, int* lostPackets);
    void copyInputToGPU(const float* data, int numSamples, int numChannels);
    void copyOutputFromGPU(float* data, int numSamples, int numChannels);
    const float* getInputBufferPtr() const;
    const float* getOutputBufferPtr() const;
    void setNetworkParameters(float packetLoss, float jitter);

private:
    MetalBridge();
    ~MetalBridge() override;

    // Buffer Properties
    size_t currentBufferSize = 0;
    size_t currentNumChannels = 0;
    bool sessionActive = false;
    AudioMetrics latestMetrics;

#ifdef __OBJC__
    // Metal Resources
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    
    // NEW: 7-stage compute pipeline states (corrected shader names)
    id<MTLComputePipelineState> inputCapturePipeline;      // Stage 1: AudioInputCaptureShader
    id<MTLComputePipelineState> inputGatePipeline;         // Stage 2: AudioInputGateShader  
    id<MTLComputePipelineState> spectralAnalysisPipeline;  // Stage 3: DJSpectralAnalysisShader
    id<MTLComputePipelineState> recordArmPipeline;         // Stage 4: RecordArmVisualShader
    id<MTLComputePipelineState> jellieEncodePipeline;      // Stage 5: JELLIEPreprocessShader
    id<MTLComputePipelineState> networkSimPipeline;       // Stage 6: NetworkSimulationShader
    id<MTLComputePipelineState> pnbtrReconstructPipeline; // Stage 7: PNBTRReconstructionShader
    id<MTLComputePipelineState> metricsPipeline;          // Metrics: MetricsComputeShader
    id<MTLComputePipelineState> waveformPipeline;         // Legacy: waveformRenderer
    
    // Audio buffers (main pipeline)
    id<MTLBuffer> audioInputBuffer;
    id<MTLBuffer> jellieBuffer;
    id<MTLBuffer> networkBuffer;
    id<MTLBuffer> reconstructedBuffer;
    id<MTLBuffer> metricsBuffer;
    
    // NEW: Stage buffers for 7-stage pipeline
    id<MTLBuffer> stage1Buffer;  // After input capture
    id<MTLBuffer> stage2Buffer;  // After input gating  
    id<MTLBuffer> stage3Buffer;  // After spectral analysis
    id<MTLBuffer> stage4Buffer;  // After record arm visual
#endif

    // Internal Methods
    bool createComputePipelines();
    void dispatchAudioPipeline(size_t numSamples);
    void calculateMetrics(size_t numSamples);
};