#include "BonjourDiscovery.h"
#import <Foundation/Foundation.h>

// Objective-C delegate to handle Bonjour callbacks
@interface TOASTBonjourDelegate : NSObject <NSNetServiceBrowserDelegate, NSNetServiceDelegate>
{
    BonjourDiscovery* cppInstance;
}
- (instancetype)initWithCppInstance:(BonjourDiscovery*)instance;
@end

@implementation TOASTBonjourDelegate

- (instancetype)initWithCppInstance:(BonjourDiscovery*)instance {
    if (self = [super init]) {
        cppInstance = instance;
    }
    return self;
}

// NSNetServiceBrowserDelegate methods
- (void)netServiceBrowser:(NSNetServiceBrowser *)browser 
           didFindService:(NSNetService *)service 
               moreComing:(BOOL)moreComing {
    // Resolve the service to get hostname and port
    service.delegate = self;
    [service resolveWithTimeout:10.0];
}

- (void)netServiceBrowser:(NSNetServiceBrowser *)browser 
         didRemoveService:(NSNetService *)service 
               moreComing:(BOOL)moreComing {
    if (cppInstance) {
        cppInstance->onServiceLost([service.name UTF8String]);
    }
}

// NSNetServiceDelegate methods for resolution
- (void)netServiceDidResolveAddress:(NSNetService *)service {
    if (cppInstance) {
        std::string hostname = service.hostName ? [service.hostName UTF8String] : "";
        cppInstance->onServiceFound([service.name UTF8String], hostname, (int)service.port);
    }
}

- (void)netService:(NSNetService *)service didNotResolve:(NSDictionary<NSString *, NSNumber *> *)errorDict {
    NSLog(@"Failed to resolve service: %@", service.name);
}

@end

//==============================================================================
// Helper function for emoji-compatible font setup
static juce::Font getEmojiCompatibleFont(float size = 12.0f)
{
    #if JUCE_MAC
        return juce::Font(juce::FontOptions().withName("SF Pro Text").withHeight(size));
    #elif JUCE_WINDOWS
        return juce::Font(juce::FontOptions().withName("Segoe UI Emoji").withHeight(size));
    #else
        return juce::Font(juce::FontOptions().withName("Noto Color Emoji").withHeight(size));
    #endif
}

BonjourDiscovery::BonjourDiscovery()
    : serviceBrowser(nullptr), publishedService(nullptr), bonjourDelegate(nullptr)
{
    // Initialize UI components
    titleLabel.setText("📡 Discover TOAST Devices", juce::dontSendNotification);
    titleLabel.setFont(getEmojiCompatibleFont(14.0f));
    addAndMakeVisible(titleLabel);
    
    deviceDropdown.setTextWhenNothingSelected("🔍 Searching for devices...");
    deviceDropdown.addListener(this);
    addAndMakeVisible(deviceDropdown);
    
    connectButton.setButtonText("🔗 Auto Connect");
    connectButton.onClick = [this] { connectToSelectedDevice(); };
    connectButton.setEnabled(false);
    addAndMakeVisible(connectButton);
    
    statusLabel.setText("Ready to discover devices", juce::dontSendNotification);
    statusLabel.setFont(getEmojiCompatibleFont(11.0f));
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(statusLabel);
    
    // Create Objective-C delegate
    bonjourDelegate = [[TOASTBonjourDelegate alloc] initWithCppInstance:this];
    
    // Start discovery automatically
    startDiscovery();
    
    // Publish our own service so others can find us
    publishOurService();
}

BonjourDiscovery::~BonjourDiscovery()
{
    stopDiscovery();
    unpublishOurService();
    
    if (bonjourDelegate) {
        [bonjourDelegate release];
        bonjourDelegate = nullptr;
    }
}

void BonjourDiscovery::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::darkgrey);
    g.drawRect(getLocalBounds(), 1);
}

void BonjourDiscovery::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    titleLabel.setBounds(bounds.removeFromTop(20));
    bounds.removeFromTop(5);
    
    auto row = bounds.removeFromTop(25);
    deviceDropdown.setBounds(row.removeFromLeft(200));
    row.removeFromLeft(10);
    connectButton.setBounds(row.removeFromLeft(100));
    
    bounds.removeFromTop(5);
    statusLabel.setBounds(bounds.removeFromTop(15));
}

void BonjourDiscovery::comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged)
{
    if (comboBoxThatHasChanged == &deviceDropdown) {
        connectButton.setEnabled(deviceDropdown.getSelectedId() > 0);
    }
}

void BonjourDiscovery::startDiscovery()
{
    if (isDiscovering) return;
    
    // Create and configure NSNetServiceBrowser
    serviceBrowser = [[NSNetServiceBrowser alloc] init];
    serviceBrowser.delegate = bonjourDelegate;
    
    // Search for TOAST services
    [serviceBrowser searchForServicesOfType:@"_toast._tcp." inDomain:@""];
    
    isDiscovering = true;
    statusLabel.setText("🔍 Discovering TOAST devices...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
}

void BonjourDiscovery::stopDiscovery()
{
    if (!isDiscovering) return;
    
    if (serviceBrowser) {
        [serviceBrowser stop];
        [serviceBrowser release];
        serviceBrowser = nullptr;
    }
    
    isDiscovering = false;
    statusLabel.setText("Discovery stopped", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
}

std::vector<BonjourDiscovery::DiscoveredDevice> BonjourDiscovery::getDiscoveredDevices() const
{
    return discoveredDevices;
}

bool BonjourDiscovery::connectToSelectedDevice()
{
    int selectedId = deviceDropdown.getSelectedId();
    if (selectedId <= 0 || selectedId > static_cast<int>(discoveredDevices.size())) {
        return false;
    }
    
    auto& device = discoveredDevices[selectedId - 1];
    
    statusLabel.setText("🔗 Connecting to " + device.name + "...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
    
    // Notify listeners
    for (auto* listener : listeners) {
        listener->deviceConnected(device);
    }
    
    // This would trigger automatic sync in the main app
    statusLabel.setText("✅ Connected! Sync enabled automatically", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    
    return true;
}

void BonjourDiscovery::addListener(Listener* listener)
{
    if (listener && std::find(listeners.begin(), listeners.end(), listener) == listeners.end()) {
        listeners.push_back(listener);
    }
}

void BonjourDiscovery::removeListener(Listener* listener)
{
    listeners.erase(std::remove(listeners.begin(), listeners.end(), listener), listeners.end());
}

void BonjourDiscovery::onServiceFound(const std::string& name, const std::string& hostname, int port)
{
    // Check if device already exists
    for (auto& device : discoveredDevices) {
        if (device.name == name) {
            device.hostname = hostname;
            device.port = port;
            device.isAvailable = true;
            
            // Classify connection type based on hostname/IP
            if (hostname.find("169.254.") != std::string::npos) {
                device.connectionType = "USB4/Thunderbolt-Bonjour";
                device.isLinkLocal = true;
                device.isDHCP = false;
            } else if (hostname.find("192.168.") != std::string::npos || 
                      hostname.find("10.") != std::string::npos) {
                device.connectionType = "Network-DHCP-Bonjour";
                device.isLinkLocal = false;
                device.isDHCP = true;
            } else {
                device.connectionType = "Network-Bonjour";
                device.isLinkLocal = false;
                device.isDHCP = false;
            }
            
            updateDeviceDropdown();
            return;
        }
    }
    
    // Add new device
    DiscoveredDevice device;
    device.name = name;
    device.hostname = hostname;
    device.port = port;
    device.serviceType = "_toast._tcp.";
    device.isAvailable = true;
    
    // Classify connection type based on hostname/IP
    if (hostname.find("169.254.") != std::string::npos) {
        device.connectionType = "USB4/Thunderbolt-Bonjour";
        device.isLinkLocal = true;
        device.isDHCP = false;
    } else if (hostname.find("192.168.") != std::string::npos || 
              hostname.find("10.") != std::string::npos) {
        device.connectionType = "Network-DHCP-Bonjour";
        device.isLinkLocal = false;
        device.isDHCP = true;
    } else {
        device.connectionType = "Network-Bonjour";
        device.isLinkLocal = false;
        device.isDHCP = false;
    }
    
    discoveredDevices.push_back(device);
    updateDeviceDropdown();
    
    // Notify listeners
    for (auto* listener : listeners) {
        listener->deviceDiscovered(device);
    }
    
    // Show appropriate status based on connection type
    if (device.isLinkLocal) {
        statusLabel.setText("� Found USB4/Thunderbolt device: " + name, juce::dontSendNotification);
    } else if (device.isDHCP) {
        statusLabel.setText("📶 Found DHCP device: " + name, juce::dontSendNotification);
    } else {
        statusLabel.setText("�📱 Found device: " + name, juce::dontSendNotification);
    }
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
}

void BonjourDiscovery::onServiceLost(const std::string& name)
{
    // Mark device as unavailable
    for (auto& device : discoveredDevices) {
        if (device.name == name) {
            device.isAvailable = false;
            break;
        }
    }
    
    updateDeviceDropdown();
    
    // Notify listeners
    for (auto* listener : listeners) {
        listener->deviceLost(name);
    }
    
    statusLabel.setText("📱 Lost device: " + name, juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
}

void BonjourDiscovery::updateDeviceDropdown()
{
    juce::MessageManager::callAsync([this]() {
        deviceDropdown.clear();
        
        int id = 1;
        for (const auto& device : discoveredDevices) {
            if (device.isAvailable) {
                std::string icon = "📱"; // Default
                if (device.connectionType.find("USB4") != std::string::npos) {
                    icon = "🔗"; // USB4/Thunderbolt
                } else if (device.connectionType.find("DHCP") != std::string::npos) {
                    icon = "📶"; // DHCP network
                } else if (device.isLinkLocal) {
                    icon = "🔗"; // Link-local
                }
                
                std::string displayName = icon + " " + device.name + 
                                        " (" + device.connectionType + 
                                        " @ " + device.hostname + ":" + std::to_string(device.port) + ")";
                deviceDropdown.addItem(displayName, id++);
            }
        }
        
        if (deviceDropdown.getNumItems() > 0) {
            deviceDropdown.setTextWhenNothingSelected("Select a TOAST device");
            connectButton.setEnabled(true);
        } else {
            deviceDropdown.setTextWhenNothingSelected("🔍 No devices found...");
            connectButton.setEnabled(false);
        }
    });
}

void BonjourDiscovery::publishOurService()
{
    // Create a service to publish ourselves as discoverable
    publishedService = [[NSNetService alloc] initWithDomain:@"" 
                                                       type:@"_toast._tcp." 
                                                       name:@"TOASTer" 
                                                       port:8080];
    
    if (publishedService) {
        publishedService.delegate = bonjourDelegate;
        [publishedService publish];
        
        statusLabel.setText("📡 Broadcasting as discoverable TOAST device", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
    }
}

void BonjourDiscovery::unpublishOurService()
{
    if (publishedService) {
        [publishedService stop];
        [publishedService release];
        publishedService = nullptr;
    }
}
