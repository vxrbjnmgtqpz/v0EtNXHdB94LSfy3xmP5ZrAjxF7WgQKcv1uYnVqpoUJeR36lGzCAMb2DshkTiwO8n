#include "WiFiNetworkDiscovery.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>

#ifdef __APPLE__
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <ifaddrs.h>
#elif _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#endif

WiFiNetworkDiscovery::WiFiNetworkDiscovery()
    : titleLabel("WiFiTitle", "📶 Wi-Fi Network Discovery"),
      scanButton("🔍 Scan Wi-Fi"),
      connectButton("🚀 Connect"),
      statusLabel("Status", "Ready to scan"),
      currentIPLabel("CurrentIP", "Detecting network..."),
      customIPEditor(),
      discovering_(false),
      scan_index_(0)
{
    // Title
    titleLabel.setFont(juce::Font(juce::FontOptions().withHeight(16.0f).withStyle("bold")));
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);
    
    // Current network info
    currentIPLabel.setFont(juce::Font(juce::FontOptions().withHeight(12.0f)));
    currentIPLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
    currentIPLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(currentIPLabel);
    
    // Device dropdown
    deviceDropdown.setTextWhenNothingSelected("No devices found");
    deviceDropdown.onChange = [this]() {
        connectButton.setEnabled(deviceDropdown.getSelectedId() > 0);
    };
    addAndMakeVisible(deviceDropdown);
    
    // Custom IP editor
    customIPEditor.setTextToShowWhenEmpty("Enter IP (e.g., 192.168.1.100)", juce::Colours::grey);
    customIPEditor.setText("192.168.1.100"); // Default Wi-Fi IP
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
        std::string target_ip;
        
        if (deviceDropdown.getSelectedId() > 0) {
            // Use selected device
            int index = deviceDropdown.getSelectedId() - 1;
            if (index < discovered_devices_.size()) {
                target_ip = discovered_devices_[index].ip_address;
            }
        } else {
            // Use manual IP
            target_ip = customIPEditor.getText().toStdString();
        }
        
        if (!target_ip.empty()) {
            std::cout << "🚀 Connecting to: " << target_ip << std::endl;
            // Notify listeners about connection attempt
            for (auto* listener : listeners_) {
                WiFiPeer peer(target_ip);
                listener->deviceDiscovered(peer);
            }
        }
    };
    connectButton.setEnabled(false);
    addAndMakeVisible(connectButton);
    
    // Status label
    statusLabel.setFont(juce::Font(juce::FontOptions().withHeight(12.0f)));
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    statusLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(statusLabel);
    
    // Detect current network
    current_network_base_ = getCurrentNetworkBase();
    currentIPLabel.setText("Network: " + current_network_base_ + ".x", juce::dontSendNotification);
}

WiFiNetworkDiscovery::~WiFiNetworkDiscovery()
{
    stopDiscovery();
}

void WiFiNetworkDiscovery::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xFF2A2A2A));
    
    // Draw border
    g.setColour(juce::Colours::lightgreen.withAlpha(0.3f));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(2), 5.0f, 2.0f);
}

void WiFiNetworkDiscovery::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    // Title
    titleLabel.setBounds(bounds.removeFromTop(20));
    bounds.removeFromTop(5);
    
    // Current network info
    currentIPLabel.setBounds(bounds.removeFromTop(15));
    bounds.removeFromTop(5);
    
    // Custom IP editor
    customIPEditor.setBounds(bounds.removeFromTop(25));
    bounds.removeFromTop(5);
    
    // Buttons row
    auto topRow = bounds.removeFromTop(25);
    topRow.removeFromLeft(5);
    scanButton.setBounds(topRow.removeFromLeft(100));
    topRow.removeFromLeft(5);
    connectButton.setBounds(topRow.removeFromLeft(80));
    
    bounds.removeFromTop(5);
    deviceDropdown.setBounds(bounds.removeFromTop(25));
    
    bounds.removeFromTop(5);
    statusLabel.setBounds(bounds.removeFromTop(15));
}

void WiFiNetworkDiscovery::timerCallback()
{
    if (!discovering_ || scan_ips_.empty()) return;
    
    // Scan a few IPs per timer cycle to avoid blocking
    int scans_per_cycle = 3;
    
    for (int i = 0; i < scans_per_cycle && scan_index_ < scan_ips_.size(); i++) {
        std::string ip = scan_ips_[scan_index_];
        
        // Update status
        statusLabel.setText("🔍 Scanning " + ip + " (" + 
                          std::to_string(scan_index_ + 1) + "/" + 
                          std::to_string(scan_ips_.size()) + ")", 
                          juce::dontSendNotification);
        
        if (pingDevice(ip, 7777)) {
            WiFiPeer peer(ip);
            peer.device_name = "TOASTer-" + ip.substr(ip.find_last_of('.') + 1);
            peer.responded = true;
            discovered_devices_.push_back(peer);
            
            std::cout << "✅ Found TOASTer peer at: " << ip << std::endl;
        }
        
        scan_index_++;
    }
    
    // Check if scan complete
    if (scan_index_ >= scan_ips_.size()) {
        std::cout << "🔍 Wi-Fi scan complete. Found " << discovered_devices_.size() << " peers." << std::endl;
        stopDiscovery();
        updateDeviceUI();
        notifyListeners();
    }
}

void WiFiNetworkDiscovery::startDiscovery()
{
    if (discovering_) return;
    
    std::cout << "🔍 Starting Wi-Fi network discovery..." << std::endl;
    discovering_ = true;
    scan_index_ = 0;
    discovered_devices_.clear();
    
    // Update current network
    current_network_base_ = getCurrentNetworkBase();
    currentIPLabel.setText("Scanning: " + current_network_base_ + ".x", juce::dontSendNotification);
    scan_ips_ = generateScanIPs();
    
    scanButton.setButtonText("⏹ Stop Scan");
    statusLabel.setText("🔍 Starting Wi-Fi scan...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
    
    // Clear dropdown
    deviceDropdown.clear();
    deviceDropdown.setTextWhenNothingSelected("Scanning...");
    connectButton.setEnabled(false);
    
    startTimer(100); // Scan every 100ms
}

void WiFiNetworkDiscovery::stopDiscovery()
{
    if (!discovering_) return;
    
    std::cout << "⏹ Stopping Wi-Fi discovery" << std::endl;
    discovering_ = false;
    stopTimer();
    
    scanButton.setButtonText("🔍 Scan Wi-Fi");
    statusLabel.setText("✅ Scan complete", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    
    currentIPLabel.setText("Network: " + current_network_base_ + ".x", juce::dontSendNotification);
}

std::string WiFiNetworkDiscovery::getSelectedDeviceIP() const
{
    if (deviceDropdown.getSelectedId() > 0) {
        int index = deviceDropdown.getSelectedId() - 1;
        if (index < discovered_devices_.size()) {
            return discovered_devices_[index].ip_address;
        }
    }
    return customIPEditor.getText().toStdString();
}

bool WiFiNetworkDiscovery::pingDevice(const std::string& ip_address, int port)
{
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cout << "❌ WSAStartup failed for " << ip_address << ":" << port << std::endl;
        return false;
    }
#endif
    
    int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd < 0) {
        std::cout << "❌ Socket creation failed for " << ip_address << ":" << port 
                  << " - " << strerror(errno) << std::endl;
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
    
    // CRITICAL FIX: Set socket reuse options
    int reuse = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        std::cout << "⚠️  SO_REUSEADDR failed for " << ip_address << ":" << port 
                  << " - " << strerror(errno) << std::endl;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, ip_address.c_str(), &server_addr.sin_addr) <= 0) {
        std::cout << "❌ inet_pton failed for " << ip_address << ":" << port << std::endl;
#ifdef _WIN32
        closesocket(socket_fd);
        WSACleanup();
#else
        close(socket_fd);
#endif
        return false;
    }
    
    // Send UDP ping message
    const char* ping_message = "TOAST_PING";
    ssize_t bytes_sent = sendto(socket_fd, ping_message, strlen(ping_message), 0,
                               (struct sockaddr*)&server_addr, sizeof(server_addr));
    
    bool connected = false;
    if (bytes_sent > 0) {
        // Try to receive response with timeout
        char response_buffer[256];
        struct sockaddr_in from_addr;
        socklen_t from_len = sizeof(from_addr);
        
        // Set receive timeout
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 500000;  // 0.5 second timeout
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        
        ssize_t bytes_received = recvfrom(socket_fd, response_buffer, sizeof(response_buffer) - 1, 0,
                                         (struct sockaddr*)&from_addr, &from_len);
        
        if (bytes_received > 0) {
            response_buffer[bytes_received] = '\0';
            if (strstr(response_buffer, "TOAST") != nullptr) {
                connected = true;
                std::cout << "✅ UDP ping successful to " << ip_address << ":" << port 
                          << " - Response: " << response_buffer << std::endl;
            } else {
                std::cout << "⚠️  UDP response from " << ip_address << ":" << port 
                          << " but not TOAST protocol" << std::endl;
            }
        } else {
            std::cout << "⏱️  UDP ping timeout to " << ip_address << ":" << port << std::endl;
        }
    } else {
        std::cout << "❌ UDP sendto failed for " << ip_address << ":" << port 
                  << " - " << strerror(errno) << std::endl;
    }
    
#ifdef _WIN32
    closesocket(socket_fd);
    WSACleanup();
#else
    close(socket_fd);
#endif
    
    return connected;
}

void WiFiNetworkDiscovery::updateDeviceUI()
{
    deviceDropdown.clear();
    
    if (discovered_devices_.empty()) {
        deviceDropdown.setTextWhenNothingSelected("No TOASTer peers found");
        statusLabel.setText("❌ No devices found", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
    } else {
        deviceDropdown.setTextWhenNothingSelected("Select a device...");
        
        for (size_t i = 0; i < discovered_devices_.size(); i++) {
            const auto& device = discovered_devices_[i];
            std::string display_name = "📱 " + device.device_name + " (" + device.ip_address + ")";
            deviceDropdown.addItem(display_name, static_cast<int>(i + 1));
        }
        
        statusLabel.setText("✅ Found " + std::to_string(discovered_devices_.size()) + " device(s)", 
                          juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    }
    
    // Enable connect button if we have devices or manual IP
    connectButton.setEnabled(!discovered_devices_.empty() || !customIPEditor.getText().isEmpty());
}

void WiFiNetworkDiscovery::notifyListeners()
{
    for (auto* listener : listeners_) {
        listener->discoveryCompleted();
    }
}

std::string WiFiNetworkDiscovery::getCurrentNetworkBase()
{
#ifdef __APPLE__
    struct ifaddrs *ifap, *ifa;
    struct sockaddr_in *sa;
    char addr_str[INET_ADDRSTRLEN];
    
    if (getifaddrs(&ifap) == -1) {
        return "192.168.1"; // Default fallback
    }
    
    for (ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) {
            // Look for Wi-Fi interface (usually en0 on Mac)
            if (strncmp(ifa->ifa_name, "en0", 3) == 0) {
                sa = (struct sockaddr_in *) ifa->ifa_addr;
                inet_ntop(AF_INET, &(sa->sin_addr), addr_str, INET_ADDRSTRLEN);
                
                std::string ip(addr_str);
                std::size_t last_dot = ip.find_last_of('.');
                if (last_dot != std::string::npos) {
                    freeifaddrs(ifap);
                    return ip.substr(0, last_dot);
                }
            }
        }
    }
    
    freeifaddrs(ifap);
#endif
    
    return "192.168.1"; // Default fallback
}

std::vector<std::string> WiFiNetworkDiscovery::generateScanIPs()
{
    std::vector<std::string> ips;
    
    // Generate priority IPs (common router/device ranges)
    std::vector<int> priority_hosts = {
        1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50, 
        100, 101, 102, 103, 104, 105, 110, 111, 112, 150, 200, 
        201, 202, 203, 254
    };
    
    // Add priority IPs first
    for (int host : priority_hosts) {
        ips.push_back(current_network_base_ + "." + std::to_string(host));
    }
    
    // Add remaining IPs in ranges
    for (int i = 2; i <= 253; i++) {
        std::string ip = current_network_base_ + "." + std::to_string(i);
        
        // Skip if already in priority list
        if (std::find(ips.begin(), ips.end(), ip) == ips.end()) {
            ips.push_back(ip);
        }
    }
    
    return ips;
}
