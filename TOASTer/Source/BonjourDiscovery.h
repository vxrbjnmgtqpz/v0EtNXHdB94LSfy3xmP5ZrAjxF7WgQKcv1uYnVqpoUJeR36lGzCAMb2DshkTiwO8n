#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <string>

#ifdef __OBJC__
    @class NSNetService;
    @class NSNetServiceBrowser;
    @class TOASTBonjourDelegate;
#else
    typedef struct objc_object NSNetService;
    typedef struct objc_object NSNetServiceBrowser;
    typedef struct objc_object TOASTBonjourDelegate;
#endif

//==============================================================================
/**
 * Native Apple Bonjour service discovery for TOAST devices
 * Uses NSNetServiceBrowser for seamless network device detection
 */
class BonjourDiscovery : public juce::Component, public juce::ComboBox::Listener
{
public:
    struct DiscoveredDevice {
        std::string name;
        std::string hostname;
        int port;
        std::string serviceType;
        bool isAvailable;
    };
    
    class Listener {
    public:
        virtual ~Listener() = default;
        virtual void deviceDiscovered(const DiscoveredDevice& device) = 0;
        virtual void deviceLost(const std::string& deviceName) = 0;
        virtual void deviceConnected(const DiscoveredDevice& device) = 0;
    };
    
    BonjourDiscovery();
    ~BonjourDiscovery() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged) override;
    
    // Discovery control
    void startDiscovery();
    void stopDiscovery();
    
    // Device management
    std::vector<DiscoveredDevice> getDiscoveredDevices() const;
    bool connectToSelectedDevice();
    
    // Listener management
    void addListener(Listener* listener);
    void removeListener(Listener* listener);
    
    // Called from Objective-C delegate
    void onServiceFound(const std::string& name, const std::string& hostname, int port);
    void onServiceLost(const std::string& name);
    
private:
    void updateDeviceDropdown();
    void publishOurService();
    void unpublishOurService();
    
    // UI Components
    juce::Label titleLabel;
    juce::ComboBox deviceDropdown;
    juce::TextButton connectButton;
    juce::Label statusLabel;
    
    // Bonjour/mDNS objects
    NSNetServiceBrowser* serviceBrowser;
    NSNetService* publishedService;
    TOASTBonjourDelegate* bonjourDelegate;
    
    // Discovery state
    std::vector<DiscoveredDevice> discoveredDevices;
    std::vector<Listener*> listeners;
    bool isDiscovering = false;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BonjourDiscovery)
};
