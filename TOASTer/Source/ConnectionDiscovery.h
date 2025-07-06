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
    struct NetworkInterface {
        std::string name;
        std::string ipAddress;
        std::string connectionType;
        bool isDHCP;
        bool isLinkLocal;
    };
    
    struct DiscoveredDevice {
        std::string name;
        std::string ipAddress;
        int port;
        std::string connectionType; // "WiFi-DHCP", "Ethernet-DHCP", "USB4-LinkLocal", "USB4-DHCP", "Static", etc.
        std::string networkInterface; // "en0", "bridge100", "utun0", etc.
        bool isAvailable;
        uint64_t lastSeen;
        bool isDHCP;
        bool isLinkLocal;
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
    std::vector<std::string> getDiscoveredPeers() const; // Simplified peer list for BasicNetworkPanel
    bool connectToDevice(const std::string& deviceId);
    bool isRunning() const { return isDiscovering.load(); }
    
    // Listener management
    void addListener(Listener* listener);
    void removeListener(Listener* listener);
    
private:
    void scanNetwork();
    void scanNetworkInterface(const NetworkInterface& iface);
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
