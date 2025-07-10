/*
  ==============================================================================

    MetalAudioTrack.h
    Created: GPU-backed audio track for real-time waveform visualization

    Stateless, zero-copy track interface that mimics JUCE's playback and 
    visualization APIs but runs over Metal buffers for:
    - Live waveform visualizers
    - PNBTR training feedback (gap fill, error correlation)
    - Metrics (SNR, latency, gap rate)
    - Export functionality

  ==============================================================================
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <memory>
#include <atomic>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef struct objc_object* id;
#endif

//==============================================================================
class MetalAudioTrack
{
public:
    MetalAudioTrack(id sourceBuffer, int sampleRate, int bufferLength);
    ~MetalAudioTrack();

    //==============================================================================
    // GPU buffer management
    void updateFromGPU();  // Copy in newest samples from GPU if needed
    
    //==============================================================================
    // Sample access (read-only for GUI or export)
    float getSampleAt(int frame) const;
    void getSamples(float* destination, int startFrame, int numFrames) const;
    
    //==============================================================================
    // Waveform visualization
    void drawWaveform(juce::Graphics& g, juce::Rectangle<float> bounds);
    void drawWaveformGPU(juce::Graphics& g, juce::Rectangle<float> bounds); // GPU-accelerated
    
    //==============================================================================
    // Metrics analysis (called from processBlock)
    void analyzeForMetrics();
    float calculateSNR(const MetalAudioTrack& reference) const;
    float calculateTHD() const;
    float calculateRMS() const;
    
    //==============================================================================
    // Export functionality
    bool exportToWav(const juce::File& outputFile) const;
    bool exportToWav(const juce::File& outputFile, int startFrame, int numFrames) const;
    
    //==============================================================================
    // Configuration
    void setSampleRate(int newSampleRate) { sampleRate = newSampleRate; }
    void setBufferLength(int newLength);
    
    //==============================================================================
    // Status
    int getSampleRate() const { return sampleRate; }
    int getBufferLength() const { return bufferLength; }
    int getAvailableSamples() const { return availableSamples.load(); }
    bool isReady() const { return metalBuffer != nullptr; }
    
    //==============================================================================
    // GPU buffer access (for MetalDrawableTrack)
    id getMetalBuffer() const { return metalBuffer; }
    const float* getCPUBufferPtr() const { return cpuRingBuffer.data(); }

private:
    //==============================================================================
    // Metal integration
    id metalBuffer;          // MTLBuffer (shared with GPU processing)
    id metalDevice;          // MTLDevice
    
    //==============================================================================
    // CPU ring buffer (for export and fallback)
    std::vector<float> cpuRingBuffer;
    std::atomic<int> writePosition{0};
    std::atomic<int> availableSamples{0};
    
    //==============================================================================
    // Configuration
    int sampleRate = 48000;
    int bufferLength = 0;
    
    //==============================================================================
    // Metrics cache
    mutable std::atomic<float> cachedRMS{0.0f};
    mutable std::atomic<float> cachedTHD{0.0f};
    mutable std::atomic<bool> metricsValid{false};
    
    //==============================================================================
    // Internal methods
    void initializeBuffers();
    void syncFromGPU();
    void updateMetricsCache() const;
    float calculatePeakInRange(int startFrame, int numFrames) const;
    
    // No copy constructor or assignment
    MetalAudioTrack(const MetalAudioTrack&) = delete;
    MetalAudioTrack& operator=(const MetalAudioTrack&) = delete;
}; 