/*
  ==============================================================================

    MetalSpectralBridge.h
    Created: GPU-native Metal bridge for spectral analysis

    Manages Metal compute shaders for:
    - Real-time FFT analysis (1024-point Cooley-Tukey)
    - DJ-style spectrum visualization with frequency coloring
    - Peak hold and smoothing for professional display
    - Waveform overlay rendering

  ==============================================================================
*/

#pragma once

#ifdef __OBJC__
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#endif

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <memory>

// Forward declarations
class MetalBridge;
class PNBTRTrainer;

struct SpectralAnalysisConfig {
    static constexpr size_t FFT_SIZE = 1024;
    static constexpr size_t SPECTRUM_BINS = FFT_SIZE / 2;
    static constexpr size_t TEXTURE_WIDTH = 512;
    static constexpr size_t TEXTURE_HEIGHT = 256;
    
    // DJ-style color configuration
    juce::Colour lowColor = juce::Colour::fromRGB(255, 0, 0);      // Red for bass
    juce::Colour midColor = juce::Colour::fromRGB(0, 255, 0);      // Green for mids
    juce::Colour highColor = juce::Colour::fromRGB(0, 255, 255);   // Cyan for highs
    
    float maxMagnitude = 1.0f;
    float logScale = 1.2f;
    float smoothingFactor = 0.8f;
    bool armed = true;
    float pulse = 0.0f;
};

class MetalSpectralBridge {
public:
    MetalSpectralBridge(MetalBridge& bridge);
    ~MetalSpectralBridge();

    // Initialization
    bool initialize();
    void cleanup();
    bool isInitialized() const { return initialized; }

    // Configuration
    void setConfig(const SpectralAnalysisConfig& config);
    const SpectralAnalysisConfig& getConfig() const { return config; }

    // Processing
    void processAudioBuffer(const float* audioData, size_t numSamples, bool isOriginal);
    void updateSpectralTexture(bool isOriginal);
    void renderToJUCEImage(juce::Image& image, bool isOriginal);

    // Access to results
    const float* getSpectralBins(bool isOriginal) const;
    size_t getSpectrumBinCount() const { return SpectralAnalysisConfig::SPECTRUM_BINS; }

#ifdef __OBJC__
    id<MTLTexture> getSpectralTexture(bool isOriginal) const;
    void updatePulseAnimation(float time);
#endif

private:
    MetalBridge& metalBridge;
    SpectralAnalysisConfig config;
    bool initialized = false;

#ifdef __OBJC__
    // Metal Resources
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    
    // Compute Pipeline States
    id<MTLComputePipelineState> fftComputePipeline;
    id<MTLComputePipelineState> applyWindowPipeline;
    id<MTLComputePipelineState> spectrumVisualPipeline;
    id<MTLComputePipelineState> peakHoldPipeline;
    id<MTLComputePipelineState> smoothingPipeline;
    id<MTLComputePipelineState> waveformOverlayPipeline;
    
    // Buffers for Original Audio
    id<MTLBuffer> originalAudioBuffer;
    id<MTLBuffer> originalRealBuffer;
    id<MTLBuffer> originalImagBuffer;
    id<MTLBuffer> originalSpectrumBuffer;
    id<MTLBuffer> originalPeakBuffer;
    id<MTLBuffer> originalPeakDecayBuffer;
    id<MTLBuffer> originalSmoothingBuffer;
    
    // Buffers for Reconstructed Audio
    id<MTLBuffer> reconstructedAudioBuffer;
    id<MTLBuffer> reconstructedRealBuffer;
    id<MTLBuffer> reconstructedImagBuffer;
    id<MTLBuffer> reconstructedSpectrumBuffer;
    id<MTLBuffer> reconstructedPeakBuffer;
    id<MTLBuffer> reconstructedPeakDecayBuffer;
    id<MTLBuffer> reconstructedSmoothingBuffer;
    
    // Uniform Buffers
    id<MTLBuffer> fftUniformsBuffer;
    id<MTLBuffer> visualUniformsBuffer;
    id<MTLBuffer> smoothingFactorBuffer;
    
    // Textures
    id<MTLTexture> originalSpectralTexture;
    id<MTLTexture> reconstructedSpectralTexture;
    
    // Texture Descriptors
    MTLTextureDescriptor* spectralTextureDescriptor;
#endif

    // Internal Methods
    bool createComputePipelines();
    bool createBuffers();
    bool createTextures();
    void updateUniforms();
    void dispatchFFTKernel(bool isOriginal);
    void dispatchVisualKernel(bool isOriginal);
    void dispatchSmoothingKernel(bool isOriginal);
    void copySpectrumToTexture(bool isOriginal);
    
    // Helper methods
    void setupFFTUniforms();
    void setupVisualUniforms();
    void animatePulse(float deltaTime);
    
    // Buffer management
    void updateAudioBuffer(const float* audioData, size_t numSamples, bool isOriginal);
    void zeroBuffers(bool isOriginal);
    
    // CPU-side data for JUCE integration
    std::vector<float> originalSpectralData;
    std::vector<float> reconstructedSpectralData;
    
    // Timing
    float lastUpdateTime = 0.0f;
    float pulsePhase = 0.0f;
}; 