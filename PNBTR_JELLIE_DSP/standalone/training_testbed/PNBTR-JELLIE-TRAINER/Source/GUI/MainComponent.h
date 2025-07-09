/*
  ==============================================================================

    MainComponent.h
    Created: Main GUI component for PNBTR+JELLIE Training Testbed

    Implements the exact schematic layout with fixed row heights:
    - Title Bar: 40px
    - Row 1 (Oscilloscopes): 200px  
    - Row 2 (Waveform Analysis): 120px
    - Row 3 (Audio Tracks): 80px
    - Row 4 (Metrics Dashboard): 100px
    - Row 5 (Controls): 60px

  ==============================================================================
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>

// Forward declarations for your row components
class ProfessionalTransportController;
class TitleComponent;
class OscilloscopeRow;
class WaveformAnalysisRow;
class AudioTracksRow;
class MetricsDashboard;
class ControlsRow;

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

class MainComponent : public juce::Component, public juce::AudioIODeviceCallback
{
public:
    MainComponent();
    ~MainComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    // Audio callback
    void audioDeviceIOCallback(const float** inputChannelData, int numInputChannels,
                               float** outputChannelData, int numOutputChannels, int numSamples) override;
    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceStopped() override;

    // Transport control wiring
    void handleTransportPlay();
    void handleTransportStop();
    void handleTransportRecord();

private:
    std::unique_ptr<ProfessionalTransportController> transportBar;
    std::unique_ptr<TitleComponent> title;
    std::unique_ptr<OscilloscopeRow> oscilloscopeRow;
    std::unique_ptr<WaveformAnalysisRow> waveformAnalysisRow;
    std::unique_ptr<AudioTracksRow> audioTracksRow;
    std::unique_ptr<MetricsDashboard> metricsDashboard;
    std::unique_ptr<ControlsRow> controlsRow;

    // DSP pipeline
    std::unique_ptr<class PNBTRTrainer> pnbtrTrainer;

    // Audio device manager and device selectors
    juce::AudioDeviceManager deviceManager;
    std::unique_ptr<juce::ComboBox> inputDeviceBox;
    std::unique_ptr<juce::ComboBox> outputDeviceBox;
    void updateDeviceLists();
    void inputDeviceChanged();
    void outputDeviceChanged();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
