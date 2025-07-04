#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

//==============================================================================
/**
 * Auto-discovery service for TOAST connections
 * Scans network for other TOAST instances and USB connections
 */
class ConnectionDiscovery : public juce::Component, public juce::Timer
{
public:
    struct DiscoveredDevice {
        std::string name;
        std::string ipAddress;
        int port;
        std::string connectionType; // "WiFi", "Ethernet", "USB4", etc.
        bool isAvailable;
        uint64_t lastSeen;
    };
    
    class Listener {
    public:
        virtual ~Listener() = default;
        virtual void deviceDiscovered(const DiscoveredDevice& device) = 0;
        virtual void deviceLost(const std::string& deviceId) = 0;
    };
    
    ConnectionDiscovery();
    ~ConnectionDiscovery() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    
    // Discovery control
    void startDiscovery();
    void stopDiscovery();
    void refreshDiscovery();
    
    // Device management
    std::vector<DiscoveredDevice> getDiscoveredDevices() const;
    bool connectToDevice(const std::string& deviceId);
    
    // Listener management
    void addListener(Listener* listener);
    void removeListener(Listener* listener);
    
private:
    void scanNetwork();
    void scanUSBConnections();
    void broadcastPresence();
    void updateDeviceList();
    
    // UI Components
    juce::Label titleLabel;
    juce::Label statusLabel;
    juce::ListBox deviceListBox;
    juce::TextButton refreshButton;
    juce::TextButton autoConnectButton;
    
    // Discovery state
    std::atomic<bool> isDiscovering{false};
    std::vector<DiscoveredDevice> discoveredDevices;
    std::vector<Listener*> listeners;
    std::thread discoveryThread;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConnectionDiscovery)
};
