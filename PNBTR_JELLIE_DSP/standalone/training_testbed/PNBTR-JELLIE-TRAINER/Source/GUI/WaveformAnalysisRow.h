#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "../DSP/PNBTRTrainer.h"
#include "../GPU/MetalSpectralBridge.h"
#include <vector>
#include <memory>

/**
 * GPU-NATIVE SPECTRAL WAVEFORM DISPLAY ROW
 * Shows DJ-style spectral waveforms with real-time Metal FFT analysis
 * 
 * LEFT: Original recorded waveform with GPU-accelerated spectral colors
 * RIGHT: Reconstructed waveform with GPU-accelerated spectral colors
 */
class WaveformAnalysisRow : public juce::Component, private juce::Timer
{
public:
    void setTrainer(PNBTRTrainer* trainerPtr);
    WaveformAnalysisRow();
    ~WaveformAnalysisRow() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void updateGPUSpectralData();
    
    // GPU-native spectral visualization methods
    void drawGPUSpectralWaveform(juce::Graphics& g, juce::Rectangle<int> bounds, bool isOriginal);
    void renderMetalTextureToGraphics(juce::Graphics& g, juce::Rectangle<int> bounds, bool isOriginal);
    
    PNBTRTrainer* trainer = nullptr;
    std::unique_ptr<MetalSpectralBridge> spectralBridge;
    SpectralAnalysisConfig spectralConfig;
    
    // JUCE images for Metal texture conversion
    juce::Image originalSpectralImage;
    juce::Image reconstructedSpectralImage;
    
    // Animation state
    float pulseTime = 0.0f;
    
    static constexpr double sampleRate = 48000.0;
    static constexpr int spectralBufferSize = 1024; // FFT size

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WaveformAnalysisRow)
};
