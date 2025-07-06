#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <string>
#include <chrono>
#include <cstdint>

// GPU-Native JAM Framework v2 includes
#include "jam_core.h"
#include "jam_gpu.h"
#include "gpu_native/gpu_timebase.h"
#include "gpu_native/gpu_shared_timeline.h"
#include "jmid_gpu/gpu_jmid_framework.h"
#include "jdat_gpu/gpu_jdat_framework.h"
#include "jvid_gpu/gpu_jvid_framework.h"

// Forward declarations
class GPUTransportController;   // GPU-native transport controller
class JAMNetworkPanel;          // Using JAM Framework v2 panel
class MIDITestingPanel;
class PerformanceMonitorPanel;
class ClockSyncPanel;
class JMIDIntegrationPanel;
class GPUMIDIManager;           // GPU-native MIDI manager
class WiFiNetworkDiscovery;     // WiFi peer discovery component

// GPU-Native Application State
struct GPUAppState {
    // Network state (GPU timeline synchronized)
    bool isNetworkConnected = false;
    int activeConnections = 0;
    double networkLatency = 0.0;
    std::string connectedIP = "";
    int connectedPort = 0;
    
    // GPU clock sync state
    bool isClockSyncEnabled = false;
    double clockAccuracy = 0.0;
    double clockOffset = 0.0;
    uint64_t roundTripTime = 0;
    
    // GPU performance state
    int messageProcessingRate = 0;
    int midiThroughput = 0;
    
    // GPU timeline timestamp (not CPU clock)
    uint64_t lastGPUTimestamp = 0;
    
    GPUAppState() {
        // Initialize with GPU timebase if available
        if (jam::gpu_native::GPUTimebase::is_initialized()) {
            lastGPUTimestamp = jam::gpu_native::GPUTimebase::get_current_time_ns();
        }
    }
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
    
    // Provide access to GPU MIDI manager for child components
    GPUMIDIManager& getGPUMIDIManager() { return *midiManager; }
    
    // GPU-native shared state management
    void updateNetworkState(bool connected, int connections, const std::string& ip, int port);
    void updateNetworkLatency(double latencyMs);
    void updateClockSync(bool enabled, double accuracy, double offset, uint64_t rtt);
    void updatePerformanceMetrics(int msgRate, int midiRate);
    
    // GPU-native JAM Framework v2 integration
    void sendMIDIEventViaJAM(uint8_t status, uint8_t data1, uint8_t data2);
    bool isJAMFrameworkConnected() const;
    
    // GPU-native state management
    void updateGPUPerformance();
    GPUAppState& getGPUAppState() { return gpuAppState; }

private:
    // GPU-native application state
    GPUAppState gpuAppState;
    
    // GPU-native infrastructure (using static singleton)
    // GPU-native frameworks (now static, no instances needed)
    std::unique_ptr<jam::jmid_gpu::GPUJMIDFramework> jmidFramework;
    std::unique_ptr<jam::jdat::GPUJDATFramework> jdatFramework;
    std::unique_ptr<jam::jvid::GPUJVIDFramework> jvidFramework;
    
    // GPU-native MIDI I/O System
    std::unique_ptr<GPUMIDIManager> midiManager;
    
    // UI Components (using GPU-native backends)
    std::unique_ptr<GPUTransportController> transportController;
    std::unique_ptr<JAMNetworkPanel> jamNetworkPanel;  // Using JAM Framework v2 panel
    std::unique_ptr<WiFiNetworkDiscovery> wifiDiscovery;  // WiFi peer discovery panel
    std::unique_ptr<MIDITestingPanel> midiPanel;
    std::unique_ptr<PerformanceMonitorPanel> performancePanel;
    std::unique_ptr<ClockSyncPanel> clockSyncPanel;
    std::unique_ptr<JMIDIntegrationPanel> jmidPanel;
    
    // GPU timeline synchronization
    uint64_t lastGPUFrame = 0;
    double gpuFrameRate = 48000.0; // Default sample rate

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};