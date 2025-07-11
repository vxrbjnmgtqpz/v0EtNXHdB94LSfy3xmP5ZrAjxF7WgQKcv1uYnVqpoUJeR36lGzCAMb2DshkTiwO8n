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

// Forward declarations
class ProfessionalTransportController;
class OscilloscopeRow;
class WaveformAnalysisRow;
class SpectralAudioTrack;
class MetricsDashboard;
class ControlsRow;

// Core Audio → Metal Bridge function declarations
extern "C" {
    void* createCoreAudioGPUBridge();
    void destroyCoreAudioGPUBridge();
    int getCoreAudioInputDeviceCount();
    int getCoreAudioOutputDeviceCount();
    const char* getCoreAudioInputDeviceName(int index);
    const char* getCoreAudioOutputDeviceName(int index);
    void setCoreAudioInputDevice(int deviceIndex);
    void setCoreAudioOutputDevice(int deviceIndex);
    void setCoreAudioRecordArmStates(bool jellieArmed, bool pnbtrArmed);
    void startCoreAudioCapture();
    void stopCoreAudioCapture();
    
    // Debugging functions
    void enableCoreAudioSineTest(bool enable);
    void checkMetalBridgeStatus();
    void forceCoreAudioCallback();
    void useDefaultInputDevice();
}

/**
 * PNBTR+JELLIE Training Testbed - Main Component
 * 
 * ARCHITECTURE: Core Audio → Metal GPU Pipeline (bypasses JUCE audio processing)
 * 
 * Real-time DAW simulating network audio transmission with AI reconstruction.
 * Progressive loading system displays components over time (video game engine style).
 * 
 * 7-row layout:
 * - Row 1: Professional Transport Controller (48px)
 * - Row 2: Device Selection (32px)
 * - Row 3: Oscilloscope Arrays (200px)
 * - Row 4: JELLIE Track (160px)
 * - Row 5: PNBTR Track (160px)
 * - Row 6: Metrics Dashboard (100px)
 * - Row 7: Controls Row (60px)
 */
class MainComponent : public juce::Component, public juce::Timer, public juce::ComboBox::Listener
{
public:
    MainComponent();
    ~MainComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    // Timer callback for progressive loading (video game engine style)
    void timerCallback() override;

    // ComboBox listener for device selection
    void comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged) override;

    // Transport control wiring
    void handleTransportPlay();
    void handleTransportStop();
    void handleTransportRecord();

private:
    // Progressive loading (video game engine style)
    std::unique_ptr<juce::Label> loadingLabel;
    int initializationStep = 0;
    bool isFullyLoaded = false;
    
    // GUI components (lazy loaded)
    std::unique_ptr<ProfessionalTransportController> transportBar;
    std::unique_ptr<juce::ComboBox> inputDeviceBox;
    std::unique_ptr<juce::ComboBox> outputDeviceBox;
    std::unique_ptr<juce::Label> inputDeviceLabel;
    std::unique_ptr<juce::Label> outputDeviceLabel;
    std::unique_ptr<OscilloscopeRow> oscilloscopeRow;
    std::unique_ptr<WaveformAnalysisRow> waveformAnalysisRow;
    std::unique_ptr<SpectralAudioTrack> jellieTrack;
    std::unique_ptr<SpectralAudioTrack> pnbtrTrack;
    std::unique_ptr<MetricsDashboard> metricsDashboard;
    std::unique_ptr<ControlsRow> controlsRow;

    // Core Audio → Metal Bridge (replaces JUCE AudioDeviceManager)
    void* coreAudioBridge = nullptr;
    
    // Audio engine integration
    void initializeCoreAudioBridge();
    void shutdownCoreAudioBridge();
    void updateDeviceLists();
    void updateRecordArmStates();
    
    // ADDED: Record arm state management for connecting UI to Core Audio → Metal pipeline
    bool isJellieTrackRecordArmed() const;
    bool isPNBTRTrackRecordArmed() const;
    
    // DEBUGGING: GUI buttons for testing audio pipeline
    std::unique_ptr<juce::TextButton> useDefaultInputButton;
    std::unique_ptr<juce::TextButton> enableSineTestButton;
    std::unique_ptr<juce::TextButton> checkMetalBridgeButton;
    std::unique_ptr<juce::TextButton> forceCallbackButton;
    void setupDebuggingButtons();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
