#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <string>

#ifdef __OBJC__
    @class IOBluetoothDevice;
#else
    typedef struct objc_object IOBluetoothDevice;
#endif

//==============================================================================
/**
 * USB4/Thunderbolt direct connection detection and management
 * Handles peer-to-peer connections without relying on DHCP/network infrastructure
 */
class USB4ConnectionManager : public juce::Component
{
public:
    struct USB4Device {
        std::string deviceName;
        std::string serialNumber;
        std::string connectionType; // "USB4", "Thunderbolt 3", "Thunderbolt 4" 
        bool isDirectConnection;
        bool isTOASTCapable;
        std::string directIP; // For direct P2P communication
    };
    
    class Listener {
    public:
        virtual ~Listener() = default;
        virtual void usb4DeviceConnected(const USB4Device& device) = 0;
        virtual void usb4DeviceDisconnected(const std::string& deviceName) = 0;
        virtual void directConnectionEstablished(const USB4Device& device) = 0;
    };
    
    USB4ConnectionManager();
    ~USB4ConnectionManager() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
    // USB4/Thunderbolt detection
    void startUSB4Detection();
    void stopUSB4Detection();
    void scanForUSB4Devices();
    
    // Direct connection management
    bool establishDirectConnection(const USB4Device& device);
    bool createDirectP2PLink(const std::string& targetDevice);
    void configureDirectNetworking();
    
    // Device management
    std::vector<USB4Device> getConnectedUSB4Devices() const;
    bool isUSB4ConnectionAvailable() const;
    
    // Listener management
    void addListener(Listener* listener);
    void removeListener(Listener* listener);
    
private:
    void detectThunderboltDevices();
    void setupDirectP2PNetworking();
    void assignDirectIPs();
    std::string generateDirectIP();
    bool isThunderboltBridgeActive();
    
    // UI Components
    juce::Label titleLabel;
    juce::Label statusLabel;
    juce::ComboBox usb4DeviceDropdown;
    juce::TextButton connectDirectButton;
    juce::Label connectionInfoLabel;
    
    // USB4 state
    std::vector<USB4Device> connectedDevices;
    std::vector<Listener*> listeners;
    bool isDetecting = false;
    bool hasDirectConnection = false;
    std::string ourDirectIP;
    std::string peerDirectIP;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(USB4ConnectionManager)
};
