#include "ConnectionDiscovery.h"
#include <sstream>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <ifaddrs.h>

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

ConnectionDiscovery::ConnectionDiscovery()
    : refreshButton("🔄 Refresh"), autoConnectButton("🚀 Auto Connect")
{
    // Set up title
    titleLabel.setText("🔍 TOAST Device Discovery", juce::dontSendNotification);
    titleLabel.setFont(getEmojiCompatibleFont(16.0f));
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);
    
    // Set up status
    statusLabel.setText("⏹️ Discovery stopped", juce::dontSendNotification);
    statusLabel.setFont(getEmojiCompatibleFont(12.0f));
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
    addAndMakeVisible(statusLabel);
    
    // Set up device list
    addAndMakeVisible(deviceListBox);
    
    // Set up buttons
    refreshButton.onClick = [this] { refreshDiscovery(); };
    addAndMakeVisible(refreshButton);
    
    autoConnectButton.onClick = [this] { 
        if (isDiscovering) {
            stopDiscovery();
            autoConnectButton.setButtonText("🚀 Auto Connect");
        } else {
            startDiscovery();
            autoConnectButton.setButtonText("⏹️ Stop Discovery");
        }
    };
    addAndMakeVisible(autoConnectButton);
    
    // Start timer for UI updates
    startTimer(2000); // Update every 2 seconds
}

ConnectionDiscovery::~ConnectionDiscovery()
{
    stopDiscovery();
    stopTimer();
}

void ConnectionDiscovery::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::darkgrey.darker(0.3f));
    g.setColour(juce::Colours::grey);
    g.drawRect(getLocalBounds(), 1);
}

void ConnectionDiscovery::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    titleLabel.setBounds(bounds.removeFromTop(25));
    bounds.removeFromTop(5);
    
    statusLabel.setBounds(bounds.removeFromTop(20));
    bounds.removeFromTop(5);
    
    // Buttons row
    auto buttonRow = bounds.removeFromTop(30);
    refreshButton.setBounds(buttonRow.removeFromLeft(100));
    buttonRow.removeFromLeft(10);
    autoConnectButton.setBounds(buttonRow.removeFromLeft(150));
    
    bounds.removeFromTop(5);
    
    // Device list takes remaining space
    deviceListBox.setBounds(bounds);
}

void ConnectionDiscovery::timerCallback()
{
    updateDeviceList();
}

void ConnectionDiscovery::startDiscovery()
{
    if (isDiscovering) return;
    
    isDiscovering = true;
    statusLabel.setText("🔍 Scanning for TOAST devices...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
    
    // Start discovery in background thread
    discoveryThread = std::thread([this] {
        while (isDiscovering) {
            scanNetwork();
            scanUSBConnections();
            broadcastPresence();
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    });
}

void ConnectionDiscovery::stopDiscovery()
{
    isDiscovering = false;
    
    if (discoveryThread.joinable()) {
        discoveryThread.join();
    }
    
    statusLabel.setText("⏹️ Discovery stopped", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
}

void ConnectionDiscovery::refreshDiscovery()
{
    discoveredDevices.clear();
    if (isDiscovering) {
        scanNetwork();
        scanUSBConnections();
    }
    updateDeviceList();
}

void ConnectionDiscovery::scanNetwork()
{
    // Get local network interfaces
    struct ifaddrs *ifaddrs_ptr;
    if (getifaddrs(&ifaddrs_ptr) == -1) return;
    
    std::vector<std::string> localNetworks;
    
    for (struct ifaddrs *ifa = ifaddrs_ptr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;
            char ip_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &(addr_in->sin_addr), ip_str, INET_ADDRSTRLEN);
            
            std::string ip = ip_str;
            if (ip.find("192.168.") == 0 || ip.find("10.") == 0 || ip.find("172.") == 0) {
                // Extract network base (e.g., 192.168.1. from 192.168.1.188)
                size_t lastDot = ip.find_last_of('.');
                if (lastDot != std::string::npos) {
                    localNetworks.push_back(ip.substr(0, lastDot + 1));
                }
            }
        }
    }
    freeifaddrs(ifaddrs_ptr);
    
    // Scan common IP ranges for TOAST servers
    for (const auto& networkBase : localNetworks) {
        for (int i = 1; i < 255; ++i) {
            if (!isDiscovering) break;
            
            std::string targetIP = networkBase + std::to_string(i);
            
            // Quick TCP port check on 8080
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) continue;
            
            struct timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 500000; // 500ms timeout
            setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            
            struct sockaddr_in target_addr;
            target_addr.sin_family = AF_INET;
            target_addr.sin_port = htons(8080);
            inet_pton(AF_INET, targetIP.c_str(), &target_addr.sin_addr);
            
            if (connect(sock, (struct sockaddr*)&target_addr, sizeof(target_addr)) == 0) {
                // Found a TOAST server!
                DiscoveredDevice device;
                device.name = "TOAST Device @ " + targetIP;
                device.ipAddress = targetIP;
                device.port = 8080;
                device.connectionType = "WiFi/Ethernet";
                device.isAvailable = true;
                device.lastSeen = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                
                // Check if device already exists
                bool found = false;
                for (auto& existing : discoveredDevices) {
                    if (existing.ipAddress == targetIP) {
                        existing.lastSeen = device.lastSeen;
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    discoveredDevices.push_back(device);
                    // Notify listeners
                    for (auto* listener : listeners) {
                        listener->deviceDiscovered(device);
                    }
                }
            }
            
            close(sock);
        }
    }
}

void ConnectionDiscovery::scanUSBConnections()
{
    // Check for USB4/Thunderbolt connections
    // This is a simplified check - in practice you'd use system APIs
    
    // For now, simulate USB4 detection by checking if we have a direct connection
    // to another computer via high-speed interface
    
    DiscoveredDevice usbDevice;
    usbDevice.name = "USB4 Connected Device";
    usbDevice.ipAddress = "169.254.0.1"; // Link-local address
    usbDevice.port = 8080;
    usbDevice.connectionType = "USB4/Thunderbolt";
    usbDevice.isAvailable = false; // Will be true if detected
    
    // TODO: Implement actual USB4/Thunderbolt device detection
    // For now, this is a placeholder
}

void ConnectionDiscovery::broadcastPresence()
{
    // Broadcast our presence on the network so other TOAST instances can find us
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return;
    
    int broadcast = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
    
    struct sockaddr_in broadcast_addr;
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_port = htons(8082); // Discovery port
    broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;
    
    std::string message = "{\"type\":\"toast_discovery\",\"name\":\"TOASTer\",\"port\":8080}";
    sendto(sock, message.c_str(), message.length(), 0, 
           (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    
    close(sock);
}

void ConnectionDiscovery::updateDeviceList()
{
    // Remove stale devices (not seen for 30 seconds)
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    discoveredDevices.erase(
        std::remove_if(discoveredDevices.begin(), discoveredDevices.end(),
            [now](const DiscoveredDevice& device) {
                return (now - device.lastSeen) > 30000; // 30 seconds
            }),
        discoveredDevices.end()
    );
    
    // Update status
    if (isDiscovering) {
        std::string status = "🔍 Found " + std::to_string(discoveredDevices.size()) + " device(s)";
        statusLabel.setText(status, juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, 
                             discoveredDevices.empty() ? juce::Colours::orange : juce::Colours::lightgreen);
    }
}

std::vector<ConnectionDiscovery::DiscoveredDevice> ConnectionDiscovery::getDiscoveredDevices() const
{
    return discoveredDevices;
}

bool ConnectionDiscovery::connectToDevice(const std::string& deviceId)
{
    // Find device and initiate connection
    for (const auto& device : discoveredDevices) {
        if (device.ipAddress == deviceId) {
            // TODO: Trigger connection in NetworkConnectionPanel
            return true;
        }
    }
    return false;
}

void ConnectionDiscovery::addListener(Listener* listener)
{
    listeners.push_back(listener);
}

void ConnectionDiscovery::removeListener(Listener* listener)
{
    listeners.erase(std::remove(listeners.begin(), listeners.end(), listener), listeners.end());
}
