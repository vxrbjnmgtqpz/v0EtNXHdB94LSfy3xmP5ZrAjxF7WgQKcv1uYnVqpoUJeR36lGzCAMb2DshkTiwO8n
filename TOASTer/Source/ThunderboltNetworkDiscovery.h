#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <string>
#include <vector>
#include <functional>

/**
 * Simplified network discovery for Thunderbolt/USB4 direct connections
 * Bypasses complex multicast setup for direct peer-to-peer discovery
 */
class ThunderboltNetworkDiscovery : public juce::Component, public juce::Timer
{
public:
    struct PeerDevice {
        std::string name;
        std::string ip_address;
        int port;
        bool is_thunderbolt;
        bool is_responsive;
        double latency_ms;
    };
    
    class Listener {
    public:
        virtual ~Listener() = default;
        virtual void peerDiscovered(const PeerDevice& device) = 0;
        virtual void peerLost(const std::string& device_name) = 0;
        virtual void connectionEstablished(const PeerDevice& device) = 0;
    };
    
    ThunderboltNetworkDiscovery();
    ~ThunderboltNetworkDiscovery() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    
    // Discovery control
    void startDiscovery();
    void stopDiscovery();
    bool isDiscovering() const { return discovering_; }
    
    // Direct connection methods
    bool connectToIP(const std::string& ip_address, int port = 7777);
    bool testDirectConnection(const std::string& ip_address, int port);
    
    // Get discovered devices
    std::vector<PeerDevice> getDiscoveredDevices() const { return discovered_devices_; }
    
    // Listener management
    void addListener(Listener* listener);
    void removeListener(Listener* listener);
    
private:
    void scanThunderboltNetwork();
    void scanPredefinedIPs();
    void pingDevice(const std::string& ip_address, int port);
    bool isThunderboltIP(const std::string& ip_address);
    void updateDeviceUI();
    
    // UI Components
    juce::Label titleLabel;
    juce::ComboBox deviceDropdown;
    juce::TextButton scanButton;
    juce::TextButton connectButton;
    juce::Label statusLabel;
    juce::TextEditor customIPEditor;
    
    // Discovery state
    std::vector<PeerDevice> discovered_devices_;
    std::vector<Listener*> listeners_;
    bool discovering_ = false;
    int scan_cycle_ = 0;
    
    // Predefined IPs to try for Thunderbolt Bridge
    static constexpr const char* THUNDERBOLT_IPS[] = {
        "169.254.212.92",  // From your screenshot
        "169.254.1.1",
        "169.254.1.2", 
        "169.254.2.1",
        "169.254.2.2",
        "169.254.100.1",
        "169.254.100.2"
    };
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ThunderboltNetworkDiscovery)
};
