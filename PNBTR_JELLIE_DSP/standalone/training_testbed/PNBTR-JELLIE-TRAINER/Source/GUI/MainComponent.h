/*
  ==============================================================================

    MainComponent.h
    Created: Main GUI component for PNBTR+JELLIE Training Testbed

    Implements the exact schematic layout:
    Row 1: 4 Oscilloscopes (Input, Network Sim, Log/Status, Output)
    Row 2: 2 Waveform Analysis (Original vs Reconstructed)
    Row 3: 2 Audio Track Placeholders (JELLIE & PNBTR)
    Row 4: Metrics Dashboard (6 metrics horizontal)
    Row 5: Controls (Start/Stop/Export + sliders)

  ==============================================================================
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <memory>
#include "../Core/SessionManager.h"
#include "OscilloscopeComponent.h"
#include "MetricsDashboard.h"

// Forward declarations to avoid include issues
namespace juce {
    class Component;
    class Timer;
    class Graphics;
    template<typename T> class Rectangle;
    class TextButton;
    class Slider;
    class Label;
    class Colour;
}

//==============================================================================
class MainComponent : public juce::Component, public juce::Timer
{
public:
    MainComponent();
    ~MainComponent() override;

    //==============================================================================
    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    //==============================================================================
    // Core system
    std::unique_ptr<SessionManager> sessionManager;
    
    //==============================================================================
    // Row 1: Four Oscilloscopes
    std::unique_ptr<OscilloscopeComponent> inputOscilloscope;
    std::unique_ptr<OscilloscopeComponent> networkOscilloscope;
    std::unique_ptr<juce::Component> logStatusWindow;
    std::unique_ptr<OscilloscopeComponent> outputOscilloscope;
    
    //==============================================================================
    // Row 2: Waveform Analysis
    std::unique_ptr<OscilloscopeComponent> originalWaveform;
    std::unique_ptr<OscilloscopeComponent> reconstructedWaveform;
    
    //==============================================================================
    // Row 3: Audio Track Placeholders
    std::unique_ptr<juce::Component> jellieTrack;
    std::unique_ptr<juce::Component> pnbtrTrack;
    
    //==============================================================================
    // Row 4: Metrics Dashboard
    std::unique_ptr<MetricsDashboard> metricsDashboard;
    
    //==============================================================================
    // Row 5: Controls
    std::unique_ptr<juce::TextButton> startButton;
    std::unique_ptr<juce::TextButton> stopButton;
    std::unique_ptr<juce::TextButton> exportButton;
    std::unique_ptr<juce::Slider> packetLossSlider;
    std::unique_ptr<juce::Slider> jitterSlider;
    std::unique_ptr<juce::Slider> gainSlider;
    std::unique_ptr<juce::Label> packetLossLabel;
    std::unique_ptr<juce::Label> jitterLabel;
    std::unique_ptr<juce::Label> gainLabel;
    
    //==============================================================================
    // Layout configuration
    struct LayoutConfig {
        int titleHeight = 40;
        int oscilloscopeRowHeight = 180;
        int waveformRowHeight = 150;
        int audioTrackRowHeight = 100;
        int metricsRowHeight = 120;
        int controlsRowHeight = 80;
        int margin = 10;
        int componentSpacing = 5;
    } layout;
    
    //==============================================================================
    // Processing state
    bool isProcessing = false;
    
    //==============================================================================
    // Layout methods
    void layoutOscilloscopeRow(juce::Rectangle<int>& bounds);
    void layoutWaveformRow(juce::Rectangle<int>& bounds);
    void layoutAudioTrackRow(juce::Rectangle<int>& bounds);
    void layoutMetricsRow(juce::Rectangle<int>& bounds);
    void layoutControlsRow(juce::Rectangle<int>& bounds);
    
    //==============================================================================
    // Drawing methods
    void drawTitle(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawRowSeparator(juce::Graphics& g, int y, int width);
    
    //==============================================================================
    // Component creation
    void createOscilloscopes();
    void createWaveformAnalysis();
    void createAudioTracks();
    void createMetricsDashboard();
    void createControls();
    void createLogStatusWindow();
    
    //==============================================================================
    // Control methods
    void startProcessing();
    void stopProcessing();
    void exportSession();
    void updateNetworkParameters();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
