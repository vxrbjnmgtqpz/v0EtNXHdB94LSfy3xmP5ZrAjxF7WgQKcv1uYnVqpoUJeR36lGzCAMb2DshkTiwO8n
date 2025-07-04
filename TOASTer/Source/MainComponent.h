#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <string>
#include <chrono>
#include <cstdint>

// Forward declarations
class TransportController;
class JAMNetworkPanel;  // Using JAM Framework v2 panel
class MIDITestingPanel;
class PerformanceMonitorPanel;
class ClockSyncPanel;
class JMIDIntegrationPanel;
class MIDIManager;

// Shared application state
struct AppState {
    // Network state
    bool isNetworkConnected = false;
    int activeConnections = 0;
    double networkLatency = 0.0;
    std::string connectedIP = "";
    int connectedPort = 0;
    
    // Clock sync state
    bool isClockSyncEnabled = false;
    double clockAccuracy = 0.0;
    double clockOffset = 0.0;
    uint64_t roundTripTime = 0;
    
    // Performance state
    int messageProcessingRate = 0;
    int midiThroughput = 0;
    
    // Update timestamp
    std::chrono::high_resolution_clock::time_point lastUpdate;
    
    AppState() : lastUpdate(std::chrono::high_resolution_clock::now()) {}
};

//==============================================================================
class MainComponent : public juce::Component, public juce::Timer
{
public:
    MainComponent();
    ~MainComponent() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;
    
    // Provide access to MIDI manager for child components
    MIDIManager& getMIDIManager() { return *midiManager; }
    
    // Shared state management
    AppState& getAppState() { return appState; }
    void updateNetworkState(bool connected, int connections, const std::string& ip, int port);
    void updateNetworkLatency(double latencyMs);
    void updateClockSync(bool enabled, double accuracy, double offset, uint64_t rtt);
    void updatePerformanceMetrics(int msgRate, int midiRate);
    
    // JAM Framework v2 integration
    void sendMIDIEventViaJAM(uint8_t status, uint8_t data1, uint8_t data2);
    bool isJAMFrameworkConnected() const;

private:
    // Shared application state
    AppState appState;
    
    // MIDI I/O System
    std::unique_ptr<MIDIManager> midiManager;
    
    // UI Components
    std::unique_ptr<TransportController> transportController;
    std::unique_ptr<JAMNetworkPanel> jamNetworkPanel;  // Using JAM Framework v2 panel
    std::unique_ptr<MIDITestingPanel> midiPanel;
    std::unique_ptr<PerformanceMonitorPanel> performancePanel;
    std::unique_ptr<ClockSyncPanel> clockSyncPanel;
    std::unique_ptr<JMIDIntegrationPanel> jmidPanel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};