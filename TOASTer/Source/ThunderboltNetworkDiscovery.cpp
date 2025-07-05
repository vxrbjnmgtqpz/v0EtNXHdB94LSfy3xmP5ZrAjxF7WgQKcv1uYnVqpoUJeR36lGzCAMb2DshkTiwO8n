#include "ThunderboltNetworkDiscovery.h"
#include <thread>
#include <chrono>
#include <iostream>

#ifdef __APPLE__
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#elif _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

ThunderboltNetworkDiscovery::ThunderboltNetworkDiscovery()
    : titleLabel("ThunderboltTitle", "🔗 Thunderbolt Bridge Discovery"),
      scanButton("🔍 Scan Network"),
      connectButton("🚀 Connect"),
      statusLabel("Status", "Ready to scan"),
      customIPEditor()
{
    // Title
    titleLabel.setFont(juce::Font(juce::FontOptions().withHeight(16.0f).withStyle("bold")));
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);
    
    // Device dropdown
    deviceDropdown.setTextWhenNothingSelected("No devices found");
    deviceDropdown.onChange = [this]() {
        connectButton.setEnabled(deviceDropdown.getSelectedId() > 0);
    };
    addAndMakeVisible(deviceDropdown);
    
    // Custom IP editor
    customIPEditor.setTextToShowWhenEmpty("Enter IP (e.g., 169.254.212.92)", juce::Colours::grey);
    customIPEditor.setText("169.254.212.92"); // Default from screenshot
    addAndMakeVisible(customIPEditor);
    
    // Scan button
    scanButton.onClick = [this]() {
        if (discovering_) {
            stopDiscovery();
        } else {
            startDiscovery();
        }
    };
    addAndMakeVisible(scanButton);
    
    // Connect button
    connectButton.onClick = [this]() {
        // Try custom IP first
        auto customIP = customIPEditor.getText().toStdString();
        if (!customIP.empty() && connectToIP(customIP)) {
            return;
        }
        
        // Try selected device
        int selectedId = deviceDropdown.getSelectedId();
        if (selectedId > 0 && selectedId <= static_cast<int>(discovered_devices_.size())) {
            auto& device = discovered_devices_[selectedId - 1];
            connectToIP(device.ip_address, device.port);
        }
    };
    connectButton.setEnabled(false);
    addAndMakeVisible(connectButton);
    
    // Status label
    statusLabel.setFont(juce::Font(juce::FontOptions().withHeight(12.0f)));
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(statusLabel);
    
    // Start with an initial scan
    startTimer(100);
}

ThunderboltNetworkDiscovery::~ThunderboltNetworkDiscovery()
{
    stopDiscovery();
}

void ThunderboltNetworkDiscovery::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::grey);
    g.drawRect(getLocalBounds(), 1);
}

void ThunderboltNetworkDiscovery::resized()
{
    auto bounds = getLocalBounds().reduced(8);
    
    titleLabel.setBounds(bounds.removeFromTop(20));
    bounds.removeFromTop(5);
    
    auto topRow = bounds.removeFromTop(25);
    customIPEditor.setBounds(topRow.removeFromLeft(150));
    topRow.removeFromLeft(5);
    scanButton.setBounds(topRow.removeFromLeft(80));
    topRow.removeFromLeft(5);
    connectButton.setBounds(topRow.removeFromLeft(80));
    
    bounds.removeFromTop(5);
    deviceDropdown.setBounds(bounds.removeFromTop(25));
    
    bounds.removeFromTop(5);
    statusLabel.setBounds(bounds.removeFromTop(15));
}

void ThunderboltNetworkDiscovery::timerCallback()
{
    if (discovering_) {
        // Perform incremental scanning to avoid blocking UI
        if (scan_cycle_ < 7) { // Number of predefined IPs
            pingDevice(THUNDERBOLT_IPS[scan_cycle_], 7777);
            scan_cycle_++;
        } else {
            // Reset scan cycle
            scan_cycle_ = 0;
            updateDeviceUI();
        }
    }
}

void ThunderboltNetworkDiscovery::startDiscovery()
{
    if (discovering_) return;
    
    std::cout << "🔍 Starting Thunderbolt network discovery..." << std::endl;
    discovering_ = true;
    scan_cycle_ = 0;
    discovered_devices_.clear();
    
    scanButton.setButtonText("⏹ Stop Scan");
    statusLabel.setText("🔍 Scanning Thunderbolt network...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
    
    // Clear dropdown
    deviceDropdown.clear();
    deviceDropdown.setTextWhenNothingSelected("Scanning...");
    connectButton.setEnabled(false);
    
    startTimer(200); // Scan every 200ms
}

void ThunderboltNetworkDiscovery::stopDiscovery()
{
    if (!discovering_) return;
    
    std::cout << "⏹ Stopping discovery..." << std::endl;
    discovering_ = false;
    
    scanButton.setButtonText("🔍 Scan Network");
    statusLabel.setText("Discovery stopped", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    
    stopTimer();
}

bool ThunderboltNetworkDiscovery::connectToIP(const std::string& ip_address, int port)
{
    std::cout << "🚀 Attempting connection to " << ip_address << ":" << port << std::endl;
    
    statusLabel.setText("🔗 Connecting to " + ip_address + "...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
    
    if (testDirectConnection(ip_address, port)) {
        statusLabel.setText("✅ Connected to " + ip_address, juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        
        // Create device info and notify listeners
        PeerDevice device;
        device.name = "TOAST Device";
        device.ip_address = ip_address;
        device.port = port;
        device.is_thunderbolt = isThunderboltIP(ip_address);
        device.is_responsive = true;
        device.latency_ms = 0.0; // Could measure this
        
        for (auto* listener : listeners_) {
            listener->connectionEstablished(device);
        }
        
        return true;
    } else {
        statusLabel.setText("❌ Failed to connect to " + ip_address, juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
        return false;
    }
}

bool ThunderboltNetworkDiscovery::testDirectConnection(const std::string& ip_address, int port)
{
    std::cout << "🧪 Testing connection to " << ip_address << ":" << port << std::endl;
    
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return false;
    }
#endif
    
    int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        std::cerr << "❌ Failed to create test socket" << std::endl;
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
    
    // Set non-blocking mode for timeout
#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(socket_fd, FIONBIO, &mode);
#else
    int flags = fcntl(socket_fd, F_GETFL, 0);
    fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK);
#endif
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip_address.c_str(), &server_addr.sin_addr);
    
    // Try to connect
    int result = connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    
    bool connected = false;
    if (result == 0) {
        // Immediate success
        connected = true;
    } else {
        // Check with select for timeout
        fd_set write_fds;
        FD_ZERO(&write_fds);
        FD_SET(socket_fd, &write_fds);
        
        struct timeval timeout;
        timeout.tv_sec = 2;  // 2 second timeout
        timeout.tv_usec = 0;
        
        int select_result = select(socket_fd + 1, nullptr, &write_fds, nullptr, &timeout);
        if (select_result > 0 && FD_ISSET(socket_fd, &write_fds)) {
            // Check if connection actually succeeded
            int error = 0;
            socklen_t len = sizeof(error);
            getsockopt(socket_fd, SOL_SOCKET, SO_ERROR, (char*)&error, &len);
            connected = (error == 0);
        }
    }
    
#ifdef _WIN32
    closesocket(socket_fd);
    WSACleanup();
#else
    close(socket_fd);
#endif
    
    if (connected) {
        std::cout << "✅ Connection test successful to " << ip_address << std::endl;
    } else {
        std::cout << "❌ Connection test failed to " << ip_address << std::endl;
    }
    
    return connected;
}

void ThunderboltNetworkDiscovery::pingDevice(const std::string& ip_address, int port)
{
    if (testDirectConnection(ip_address, port)) {
        // Check if we already have this device
        for (auto& device : discovered_devices_) {
            if (device.ip_address == ip_address) {
                device.is_responsive = true;
                return;
            }
        }
        
        // Add new device
        PeerDevice device;
        device.name = "TOAST Device @ " + ip_address;
        device.ip_address = ip_address;
        device.port = port;
        device.is_thunderbolt = isThunderboltIP(ip_address);
        device.is_responsive = true;
        device.latency_ms = 0.0;
        
        discovered_devices_.push_back(device);
        
        // Notify listeners
        for (auto* listener : listeners_) {
            listener->peerDiscovered(device);
        }
        
        std::cout << "✅ Discovered device at " << ip_address << std::endl;
    }
}

bool ThunderboltNetworkDiscovery::isThunderboltIP(const std::string& ip_address)
{
    return ip_address.substr(0, 8) == "169.254.";
}

void ThunderboltNetworkDiscovery::updateDeviceUI()
{
    juce::MessageManager::callAsync([this]() {
        deviceDropdown.clear();
        
        if (discovered_devices_.empty()) {
            deviceDropdown.setTextWhenNothingSelected("No devices found");
            connectButton.setEnabled(false);
            statusLabel.setText("No devices discovered", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
        } else {
            int id = 1;
            for (const auto& device : discovered_devices_) {
                std::string icon = device.is_thunderbolt ? "🔗" : "📶";
                std::string displayName = icon + " " + device.name + " (" + device.ip_address + ":" + std::to_string(device.port) + ")";
                deviceDropdown.addItem(displayName, id++);
            }
            
            deviceDropdown.setTextWhenNothingSelected("Select device to connect");
            connectButton.setEnabled(true);
            
            statusLabel.setText("Found " + juce::String(discovered_devices_.size()) + " device(s)", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        }
    });
}

void ThunderboltNetworkDiscovery::addListener(Listener* listener)
{
    if (listener && std::find(listeners_.begin(), listeners_.end(), listener) == listeners_.end()) {
        listeners_.push_back(listener);
    }
}

void ThunderboltNetworkDiscovery::removeListener(Listener* listener)
{
    listeners_.erase(std::remove(listeners_.begin(), listeners_.end(), listener), listeners_.end());
}
