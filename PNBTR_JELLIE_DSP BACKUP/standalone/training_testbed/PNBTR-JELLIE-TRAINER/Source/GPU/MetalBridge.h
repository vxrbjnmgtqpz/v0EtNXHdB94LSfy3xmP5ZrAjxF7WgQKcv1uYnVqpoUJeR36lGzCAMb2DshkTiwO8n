/*
  ==============================================================================

    MetalBridge.h
    Created: GPU-native Metal bridge for zero-copy processing

    Manages Metal buffers and dispatches kernels for:
    - JELLIE encoding (48kHz→192kHz, 8-channel distribution)
    - Network simulation (packet loss, jitter)
    - PNBTR reconstruction (neural gap filling)

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
    id<MTLComputePipelineState> jellieEncodePipeline;
    id<MTLComputePipelineState> networkSimPipeline;
    id<MTLComputePipelineState> pnbtrReconstructPipeline;
    id<MTLComputePipelineState> metricsPipeline;
    id<MTLComputePipelineState> waveformPipeline;
    id<MTLBuffer> audioInputBuffer;
    id<MTLBuffer> jellieBuffer;
    id<MTLBuffer> networkBuffer;
    id<MTLBuffer> reconstructedBuffer;
    id<MTLBuffer> metricsBuffer;
#endif

    // Internal Methods
    bool createComputePipelines();
    void dispatchAudioPipeline(size_t numSamples);
    void calculateMetrics(size_t numSamples);
};