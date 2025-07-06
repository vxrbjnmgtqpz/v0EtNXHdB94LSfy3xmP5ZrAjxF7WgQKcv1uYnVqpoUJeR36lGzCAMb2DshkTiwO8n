#include "Source/WiFiNetworkDiscovery.h"
#include <iostream>
#include <thread>
#include <chrono>

class TestWiFiListener : public WiFiNetworkDiscovery::Listener {
public:
    void deviceDiscovered(const WiFiPeer& device) override {
        std::cout << "Device discovered: " << device.ip_address 
                  << " (port " << device.port << ")" << std::endl;
    }
    
    void discoveryCompleted() override {
        std::cout << "WiFi discovery completed" << std::endl;
        discovery_completed = true;
    }
    
    bool discovery_completed = false;
};

int main() {
    std::cout << "Testing WiFi Network Discovery..." << std::endl;
    
    WiFiNetworkDiscovery discovery;
    TestWiFiListener listener;
    discovery.addListener(&listener);
    
    std::cout << "Starting WiFi scan..." << std::endl;
    discovery.startDiscovery();
    
    // Wait for discovery to complete or timeout
    auto start_time = std::chrono::steady_clock::now();
    while (!listener.discovery_completed && 
           std::chrono::steady_clock::now() - start_time < std::chrono::seconds(30)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (listener.discovery_completed) {
        std::cout << "Discovery completed successfully!" << std::endl;
        auto devices = discovery.getDiscoveredDevices();
        std::cout << "Found " << devices.size() << " devices:" << std::endl;
        for (const auto& device : devices) {
            std::cout << "  - " << device.ip_address << ":" << device.port;
            if (device.responded) {
                std::cout << " (responded)";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Discovery timed out or failed" << std::endl;
    }
    
    discovery.stopDiscovery();
    return 0;
}
