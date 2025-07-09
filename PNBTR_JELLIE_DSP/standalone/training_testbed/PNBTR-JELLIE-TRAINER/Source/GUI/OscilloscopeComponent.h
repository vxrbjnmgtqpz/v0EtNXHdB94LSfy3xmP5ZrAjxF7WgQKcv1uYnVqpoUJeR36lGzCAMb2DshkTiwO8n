#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include <vector>
#include <memory>
#include "../DSP/PNBTRTrainer.h"

// Forward declarations
class MetalBridgeInterface;

/**
 * Real-time oscilloscope component for waveform visualization
 * Connects directly to MetalBridge shared buffers for zero-copy rendering
 * Part of the schematic-based GUI implementation
 */
class OscilloscopeComponent : public juce::Component, public juce::Timer
{
public:
    void setTrainer(PNBTRTrainer* trainerPtr);
    enum class BufferType {
        AudioInput,      // audioInputBuffer (48kHz mono input)
        JellieEncoded,   // jellieBuffer (192kHz 8-channel)
        NetworkProcessed,// networkBuffer (with packet loss/jitter)
        Reconstructed    // reconstructedBuffer (48kHz mono output)
    };
    
    OscilloscopeComponent(BufferType bufferType, const juce::String& title);
    ~OscilloscopeComponent() override;
    
private:
    PNBTRTrainer* trainer = nullptr;
    //==============================================================================
    // Component interface
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // Timer callback for real-time updates
    void timerCallback() override;
    
    //==============================================================================
    // Configuration
    void setRefreshRate(int hz) { refreshRateHz = hz; startTimer(1000 / hz); }
    void setAmplitudeScale(float scale) { amplitudeScale = scale; repaint(); }
    void setTimeWindow(float seconds) { timeWindowSeconds = seconds; updateDisplayBuffer(); }
    void setGridVisible(bool visible) { showGrid = visible; repaint(); }
    
    //==============================================================================

    // Data access
    void updateFromMetalBuffer(); // Called by timer to read from Metal buffers
    const std::vector<float>& getDisplayBuffer() const { return displayBuffer; }

    // DSP integration
    // ...existing code...
    //==============================================================================
    // Buffer connection
    BufferType bufferType;
    MetalBridgeInterface* metalBridge;
    juce::String oscilloscopeTitle;
    
    //==============================================================================
    // Display parameters
    int refreshRateHz = 60;
    float amplitudeScale = 1.0f;
    float timeWindowSeconds = 0.1f; // 100ms window
    bool showGrid = true;
    bool isActive = false;
    
    //==============================================================================
    // Display data
    std::vector<float> displayBuffer;
    std::vector<float> previousBuffer; // For detecting changes
    size_t bufferSize = 1024;
    int displayWidth = 800;
    
    //==============================================================================
    // Visual styling
    juce::Colour backgroundColour{0xff1a1a1a};
    juce::Colour waveformColour{0xff00ff00};
    juce::Colour gridColour{0xff333333};
    juce::Colour textColour{0xffcccccc};
    
    //==============================================================================
    // Internal methods
    void updateDisplayBuffer();
    void drawWaveform(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawGrid(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawTitle(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawScale(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    
    // Buffer type specific methods
    void readAudioInputBuffer();
    void readJellieBuffer();
    void readNetworkBuffer();
    void readReconstructedBuffer();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OscilloscopeComponent)
}; 