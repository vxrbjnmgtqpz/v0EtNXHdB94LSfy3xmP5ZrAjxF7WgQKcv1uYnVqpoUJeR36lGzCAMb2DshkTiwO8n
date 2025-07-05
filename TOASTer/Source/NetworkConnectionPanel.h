#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "TOASTTransport.h"
#include "ClockDriftArbiter.h"
#include "BonjourDiscovery.h"
#include "../../JAM_Framework_v2/include/jam_transport.h"
#include <memory>

//==============================================================================
class NetworkConnectionPanel : public juce::Component, public BonjourDiscovery::Listener
{
public:
    NetworkConnectionPanel();
    ~NetworkConnectionPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    
    // Public interface for testing transport and MIDI
    bool isConnectedToRemote() const { return isConnected && handshakeVerified; }
    void testTransportStart() { if (isConnectedToRemote()) sendTransportCommand("TRANSPORT_START"); }
    void testTransportStop() { if (isConnectedToRemote()) sendTransportCommand("TRANSPORT_STOP"); }
    void testTransportPause() { if (isConnectedToRemote()) sendTransportCommand("TRANSPORT_PAUSE"); }
    void testMIDINote(uint8_t note, uint8_t velocity = 127) { 
        if (isConnectedToRemote()) { 
            sendMIDINote(note, velocity, true);  // Note on
            // Note off will be sent after a short delay in a real implementation
        } 
    }
    
    // BonjourDiscovery::Listener implementation
    void deviceDiscovered(const BonjourDiscovery::DiscoveredDevice& device) override;
    void deviceLost(const std::string& deviceName) override;
    void deviceConnected(const BonjourDiscovery::DiscoveredDevice& device) override;

    // Transport state provider interface for getting current position and BPM
    struct TransportStateProvider {
        virtual ~TransportStateProvider() = default;
        virtual uint64_t getCurrentPosition() const = 0;
        virtual double getCurrentBPM() const = 0;
        virtual bool isPlaying() const = 0;
    };
    
    void setTransportStateProvider(TransportStateProvider* provider) { transportStateProvider = provider; }

private:
    void connectButtonClicked();
    void disconnectButtonClicked();
    void createSessionClicked();
    void joinSessionClicked();
    void updateConnectionStatus();
    void autoConnectToDHCPDevice();
    void startSimulationMode();
    void establishConnection(const std::string& ip, const std::string& port, const std::string& connectionType);
    void connectViaUDP(const std::string& multicastGroup, int port);
    
    // UDP message handling
    void handleUDPMessage(const uint8_t* data, size_t size);
    
    // Message handling
    void handleIncomingMessage(std::unique_ptr<TOAST::TransportMessage> message);
    void sendTransportCommand(const std::string& command);
    void sendMIDINote(uint8_t note, uint8_t velocity, bool isOn);
    void sendConnectionVerificationHandshake();
    void confirmConnectionEstablished();
    
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
    
    // UDP Transport for multicast
    std::unique_ptr<jam::UDPTransport> udpTransport;
    
    // Transport state provider for getting current position/BPM
    TransportStateProvider* transportStateProvider = nullptr;
    
    // Connection state
    bool isConnected = false;
    bool isServer = false;
    bool handshakeVerified = false;
    std::string currentSessionId;
    std::string clientId;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NetworkConnectionPanel)
};