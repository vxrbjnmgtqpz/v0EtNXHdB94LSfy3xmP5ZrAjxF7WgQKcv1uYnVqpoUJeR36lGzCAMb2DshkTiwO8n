#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <string>

struct WiFiPeer {
    std::string ip_address;
    std::string device_name;
    int port;
    bool responded;
    
    WiFiPeer(const std::string& ip, int p = 7777) 
        : ip_address(ip), port(p), responded(false) {}
};

class WiFiNetworkDiscovery : public juce::Component, 
                             public juce::Timer,
                             public juce::TextEditor::Listener
{
public:
    WiFiNetworkDiscovery();
    ~WiFiNetworkDiscovery() override;
    
    // Component overrides
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // Timer callback for incremental scanning
    void timerCallback() override;
    
    // Discovery control
    void startDiscovery();
    void stopDiscovery();
    bool isDiscovering() const { return discovering_; }
    
    // Get discovered devices
    const std::vector<WiFiPeer>& getDiscoveredDevices() const { return discovered_devices_; }
    std::string getSelectedDeviceIP() const;
    
    // Listener pattern for notifications
    class Listener {
    public:
        virtual ~Listener() = default;
        virtual void deviceDiscovered(const WiFiPeer& device) {}
        virtual void discoveryCompleted() {}
    };
    
    void addListener(Listener* listener) { listeners_.push_back(listener); }
    void removeListener(Listener* listener) { 
        listeners_.erase(std::remove(listeners_.begin(), listeners_.end(), listener), listeners_.end()); 
    }

private:
    // UI Components
    juce::Label titleLabel;
    juce::ComboBox deviceDropdown;
    juce::TextEditor customIPEditor;
    juce::TextButton scanButton;
    juce::TextButton connectButton;
    juce::Label statusLabel;
    juce::Label currentIPLabel;
    
    // Discovery state
    bool discovering_;
    int scan_index_;  // Current IP being scanned
    std::vector<WiFiPeer> discovered_devices_;
    std::vector<Listener*> listeners_;
    
    // Network utilities
    bool pingDevice(const std::string& ip_address, int port);
    void updateDeviceUI();
    void notifyListeners();
    std::string getCurrentNetworkBase();
    std::vector<std::string> generateScanIPs();
    
    // Current network info
    std::string current_network_base_;
    std::vector<std::string> scan_ips_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WiFiNetworkDiscovery)
};
