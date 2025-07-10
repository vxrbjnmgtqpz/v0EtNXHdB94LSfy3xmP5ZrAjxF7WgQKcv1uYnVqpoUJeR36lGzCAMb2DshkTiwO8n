/*
  ==============================================================================

    MainComponent.h
    Created: Main GUI component for PNBTR+JELLIE Training Testbed

    Implements the corrected layout with fixed row heights (no title row):
    - Row 1 (Transport): 48px - Play/Pause/Stop/Record, session time, BPM, packet loss/jitter
    - Row 2 (Oscilloscopes): 200px - Input, TOAST Network, Log/Status, Output
    - Row 3 (Waveform Analysis): 240px
    - Row 4 (Audio Tracks): 160px
    - Row 5 (Metrics Dashboard): 100px
    - Row 6 (Controls): 60px

  ==============================================================================
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <memory>

// Include full headers for all row components
#include "ProfessionalTransportController.h"
#include "OscilloscopeRow.h"
#include "WaveformAnalysisRow.h"
#include "SpectralAudioTrack.h"
#include "MetricsDashboard.h"
#include "ControlsRow.h"

// Explicitly include ComboBox header (do not include directly, already included by juce_gui_basics.h)

// No forward declarations for JUCE types; always include the real JUCE headers

//==============================================================================

// Ensure JUCE Component and AudioIODeviceCallback are included
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>

class MainComponent : public juce::Component, public juce::AudioIODeviceCallback, public juce::Timer
{
public:
    MainComponent();
    ~MainComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    // Timer callback for progressive loading (video game engine style)
    void timerCallback() override;

    // Audio callback
    void audioDeviceIOCallback(const float** inputChannelData, int numInputChannels,
                               float** outputChannelData, int numOutputChannels, int numSamples);
    void audioDeviceAboutToStart(juce::AudioIODevice* device);
    void audioDeviceStopped();

    // Transport control wiring
    void handleTransportPlay();
    void handleTransportStop();
    void handleTransportRecord();

private:
    // Progressive loading (video game engine style)
    std::unique_ptr<juce::Label> loadingLabel;
    int initializationStep = 0;
    bool isFullyLoaded = false;
    
    // GUI components (lazy loaded) - REMOVED TitleComponent per user request
    std::unique_ptr<ProfessionalTransportController> transportBar;
    std::unique_ptr<OscilloscopeRow> oscilloscopeRow;
    std::unique_ptr<WaveformAnalysisRow> waveformAnalysisRow;
    std::unique_ptr<SpectralAudioTrack> jellieTrack;
    std::unique_ptr<SpectralAudioTrack> pnbtrTrack;
    std::unique_ptr<MetricsDashboard> metricsDashboard;
    std::unique_ptr<ControlsRow> controlsRow;

    // DSP pipeline (loaded in background)
    std::unique_ptr<class PNBTRTrainer> pnbtrTrainer;

    // Audio device manager and device selectors
    juce::AudioDeviceManager deviceManager;
    std::unique_ptr<juce::ComboBox> inputDeviceBox;
    std::unique_ptr<juce::ComboBox> outputDeviceBox;
    void updateDeviceLists();
    void inputDeviceChanged();
    void outputDeviceChanged();
    
    // Audio engine integration
    void initializeAudioEngine();
    void shutdownAudioEngine();
    
    // ADDED: Record arm state management for connecting UI to audio pipeline
    bool isJellieTrackRecordArmed() const;
    bool isPNBTRTrackRecordArmed() const;


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
