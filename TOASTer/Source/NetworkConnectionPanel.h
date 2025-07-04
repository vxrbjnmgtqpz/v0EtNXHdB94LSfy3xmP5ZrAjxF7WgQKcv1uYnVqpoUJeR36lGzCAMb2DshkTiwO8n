#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "TOASTTransport.h"
#include "ClockDriftArbiter.h"
#include "BonjourDiscovery.h"
#include <memory>

//==============================================================================
class NetworkConnectionPanel : public juce::Component, public BonjourDiscovery::Listener
{
public:
    NetworkConnectionPanel();
    ~NetworkConnectionPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    
    // BonjourDiscovery::Listener implementation
    void deviceDiscovered(const BonjourDiscovery::DiscoveredDevice& device) override;
    void deviceLost(const std::string& deviceName) override;
    void deviceConnected(const BonjourDiscovery::DiscoveredDevice& device) override;

private:
    void connectButtonClicked();
    void disconnectButtonClicked();
    void createSessionClicked();
    void joinSessionClicked();
    void updateConnectionStatus();
    void autoConnectToDHCPDevice();
    void startSimulationMode();
    void establishConnection(const std::string& ip, const std::string& port, const std::string& connectionType);
    
    // UI Components
    juce::Label titleLabel;
    juce::Label networkInfoLabel;
    std::unique_ptr<BonjourDiscovery> bonjourDiscovery;
    juce::Label protocolLabel;
    juce::ComboBox protocolSelector;
    juce::TextEditor ipAddressEditor;
    juce::TextEditor portEditor;
    juce::TextEditor sessionNameEditor;
    juce::TextButton connectButton;
    juce::TextButton disconnectButton;
    juce::TextButton createSessionButton;
    juce::TextButton joinSessionButton;
    juce::Label statusLabel;
    juce::Label sessionInfoLabel;
    juce::Label performanceLabel;
    
    // TOAST Framework Integration
    std::unique_ptr<TOAST::ClockDriftArbiter> clockArbiter;
    std::unique_ptr<TOAST::ConnectionManager> connectionManager;
    std::unique_ptr<TOAST::ProtocolHandler> toastHandler;
    std::unique_ptr<TOAST::SessionManager> sessionManager;
    
    bool isConnected = false;
    std::string currentSessionId;
    std::string clientId;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NetworkConnectionPanel)
};