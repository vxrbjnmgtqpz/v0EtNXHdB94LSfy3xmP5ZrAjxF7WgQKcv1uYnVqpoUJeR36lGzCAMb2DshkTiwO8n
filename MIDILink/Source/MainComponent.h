#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

// Forward declarations
class TransportController;
class NetworkConnectionPanel;
class MIDITestingPanel;
class PerformanceMonitorPanel;
class ClockSyncPanel;
class JSONMIDIIntegrationPanel;
class MIDIManager;

//==============================================================================
class MainComponent : public juce::Component
{
public:
    MainComponent();
    ~MainComponent() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    
    // Provide access to MIDI manager for child components
    MIDIManager& getMIDIManager() { return *midiManager; }

private:
    // MIDI I/O System
    std::unique_ptr<MIDIManager> midiManager;
    
    // UI Components
    std::unique_ptr<TransportController> transportController;
    std::unique_ptr<NetworkConnectionPanel> networkPanel;
    std::unique_ptr<MIDITestingPanel> midiPanel;
    std::unique_ptr<PerformanceMonitorPanel> performancePanel;
    std::unique_ptr<ClockSyncPanel> clockSyncPanel;
    std::unique_ptr<JSONMIDIIntegrationPanel> jsonmidiPanel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};