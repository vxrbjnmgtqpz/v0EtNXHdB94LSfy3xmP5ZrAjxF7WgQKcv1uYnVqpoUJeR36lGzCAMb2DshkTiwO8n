#pragma once

#include <string>
#include <functional>
#include <memory>
#include <atomic>
#include <thread>

namespace jam {

/**
 * Real-time network state detection and monitoring
 * Prevents false positive "connected" status before actual network availability
 */
class NetworkStateDetector {
public:
    enum class NetworkState {
        UNKNOWN,
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        ERROR
    };
    
    struct NetworkInfo {
        NetworkState state = NetworkState::UNKNOWN;
        std::string interface_name;
        std::string ip_address;
        bool has_internet = false;
        bool multicast_capable = false;
        double latency_ms = -1.0;
        std::string error_message;
    };
    
    using StateChangeCallback = std::function<void(const NetworkInfo&)>;
    
    NetworkStateDetector();
    ~NetworkStateDetector();
    
    // Test actual network connectivity with real packets
    bool testUDPConnectivity(const std::string& multicast_addr, uint16_t port);
    
    // Test if multicast works by sending discovery packet to ourselves
    bool testMulticastCapability(const std::string& multicast_addr, uint16_t port, uint32_t timeout_ms = 2000);
    
    // Get current network state (cached, updated every 1000ms)
    NetworkInfo getCurrentState() const;
    
    // Start continuous monitoring
    void startMonitoring(StateChangeCallback callback);
    void stopMonitoring();
    
    // Test if OS network permission is granted (macOS specific)
    bool hasNetworkPermission();
    
    // Check if DHCP/network interface is actually ready
    bool isNetworkInterfaceReady();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    mutable std::atomic<NetworkState> cached_state_{NetworkState::UNKNOWN};
    mutable NetworkInfo cached_info_;
    std::atomic<bool> monitoring_{false};
    std::thread monitor_thread_;
    
    void monitorLoop(StateChangeCallback callback);
    void updateNetworkState();
    bool testBasicConnectivity();
    bool pingGateway();
};

} // namespace jam
