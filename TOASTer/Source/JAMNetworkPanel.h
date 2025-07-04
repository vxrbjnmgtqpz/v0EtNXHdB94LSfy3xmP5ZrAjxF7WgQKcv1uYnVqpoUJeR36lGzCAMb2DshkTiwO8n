#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "JAMFrameworkIntegration.h"
#include "BonjourDiscovery.h"

/**
 * JAM Framework v2 Network Panel for TOASTer
 * 
 * Replaces the old TCP-based NetworkConnectionPanel with UDP multicast
 * using JAM Framework v2, including PNBTR prediction and GPU acceleration.
 */
class JAMNetworkPanel : public juce::Component, 
                       public juce::Timer,
                       public BonjourDiscovery::Listener {
public:
    JAMNetworkPanel();
    ~JAMNetworkPanel() override;
    
    // Component overrides
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // Network state
    bool isConnected() const;
    int getActivePeers() const;
    double getCurrentLatency() const;
    double getCurrentThroughput() const;
    
    // Send MIDI events through JAM Framework v2
    void sendMIDIEvent(uint8_t status, uint8_t data1, uint8_t data2);
    void sendMIDIData(const uint8_t* data, size_t size);
    
    // Configuration
    void setSessionName(const juce::String& sessionName);
    void setMulticastAddress(const juce::String& address);
    void setUDPPort(int port);
    void enablePNBTRPrediction(bool audio, bool video);
    void enableGPUAcceleration(bool enable);
    
private:
    // JAM Framework v2 integration
    std::unique_ptr<JAMFrameworkIntegration> jamFramework;
    
    // UI Components
    juce::Label titleLabel;
    juce::Label statusLabel;
    juce::Label performanceLabel;
    juce::Label pnbtrStatusLabel;
    
    // Configuration controls
    juce::Label sessionLabel;
    juce::TextEditor sessionNameEditor;
    juce::Label multicastLabel;
    juce::TextEditor multicastAddressEditor;
    juce::Label portLabel;
    juce::TextEditor portEditor;
    
    // Connection controls
    juce::TextButton connectButton;
    juce::TextButton disconnectButton;
    juce::TextButton gpuInitButton;
    
    // Feature toggles
    juce::ToggleButton pnbtrAudioToggle;
    juce::ToggleButton pnbtrVideoToggle;
    juce::ToggleButton burstTransmissionToggle;
    juce::ToggleButton gpuAccelerationToggle;
    
    // Discovery
    std::unique_ptr<BonjourDiscovery> bonjourDiscovery;
    
    // Performance metrics display
    juce::Label latencyLabel;
    juce::Label throughputLabel;
    juce::Label peersLabel;
    juce::Label predictionLabel;
    
    // State
    bool networkConnected = false;
    int activePeers = 0;
    double currentLatency = 0.0;
    double currentThroughput = 0.0;
    double predictionAccuracy = 0.0;
    bool gpuInitialized = false;
    
    // Session configuration
    juce::String currentSessionName = "TOASTer_Session";
    juce::String currentMulticastAddr = "239.255.77.77";
    int currentUDPPort = 7777;
    
    // Callbacks
    void connectButtonClicked();
    void disconnectButtonClicked();
    void gpuInitButtonClicked();
    void sessionConfigurationChanged();
    
    // JAM Framework callbacks
    void onJAMStatusChanged(const std::string& status, bool connected);
    void onJAMPerformanceUpdate(double latency_us, double throughput_mbps, int active_peers);
    void onJAMMIDIReceived(uint8_t status, uint8_t data1, uint8_t data2, uint32_t timestamp);
    
    // BonjourDiscovery::Listener implementation
    void deviceFound(const std::string& name, const std::string& ip, int port) override;
    void deviceLost(const std::string& name) override;
    
    // Timer callback
    void timerCallback() override;
    
    // Helper methods
    void updateUI();
    void updatePerformanceDisplay();
    juce::Font getEmojiCompatibleFont(float size = 12.0f);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(JAMNetworkPanel)
};
