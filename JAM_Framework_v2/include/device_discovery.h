#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <map>

namespace jam {

/**
 * Enhanced device discovery system that works with USB4, Ethernet, and WiFi
 * Fixes "Discover TOAST devices" not working by testing all network interfaces
 */
class DeviceDiscovery {
public:
    struct NetworkInterface {
        std::string name;           // eth0, en0, usb0, bridge100, etc.
        std::string display_name;   // "USB4 Thunderbolt", "WiFi", "Ethernet"
        std::string ip_address;
        std::string subnet_mask;
        std::string broadcast_addr;
        bool is_active = false;
        bool is_multicast_capable = false;
        bool is_usb4 = false;
        bool is_thunderbolt = false;
        bool is_wifi = false;
        bool is_ethernet = false;
    };
    
    struct DiscoveredDevice {
        std::string device_id;
        std::string device_name;
        std::string ip_address;
        int port = 7777;
        std::string interface_name;
        std::string connection_type;
        uint64_t last_seen_ms = 0;
        double latency_ms = -1.0;
        bool is_responding = false;
    };
    
    using DeviceFoundCallback = std::function<void(const DiscoveredDevice&)>;
    using DeviceLostCallback = std::function<void(const std::string& device_id)>;
    using InterfaceFoundCallback = std::function<void(const NetworkInterface&)>;
    
    DeviceDiscovery();
    ~DeviceDiscovery();
    
    // Start/stop discovery on all network interfaces
    bool startDiscovery(const std::string& multicast_addr = "239.255.77.77", uint16_t port = 7777);
    void stopDiscovery();
    
    // Callbacks
    void setDeviceFoundCallback(DeviceFoundCallback callback) { device_found_callback_ = callback; }
    void setDeviceLostCallback(DeviceLostCallback callback) { device_lost_callback_ = callback; }
    void setInterfaceFoundCallback(InterfaceFoundCallback callback) { interface_found_callback_ = callback; }
    
    // Get current state
    std::vector<NetworkInterface> getAvailableInterfaces() const;
    std::vector<DiscoveredDevice> getDiscoveredDevices() const;
    
    // Force discovery on specific interface (useful for USB4 debugging)
    bool testInterface(const std::string& interface_name, const std::string& multicast_addr, uint16_t port);
    
    // Test if a specific device is reachable
    bool pingDevice(const std::string& ip_address, uint16_t port, uint32_t timeout_ms = 1000);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // Callbacks
    DeviceFoundCallback device_found_callback_;
    DeviceLostCallback device_lost_callback_;
    InterfaceFoundCallback interface_found_callback_;
    
    // State
    mutable std::atomic<bool> discovering_{false};
    std::thread discovery_thread_;
    std::string multicast_address_;
    uint16_t discovery_port_ = 7777;
    
    // Internal methods
    void discoveryLoop();
    void scanNetworkInterfaces();
    void sendDiscoveryPing(const NetworkInterface& interface);
    void processReceivedPackets();
    NetworkInterface identifyInterface(const std::string& interface_name);
    bool isUSB4Interface(const std::string& interface_name);
    bool isThunderboltInterface(const std::string& interface_name);
};

} // namespace jam
