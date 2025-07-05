#include "../include/network_state_detector.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cstring>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#elif _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#else
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace jam {

struct NetworkStateDetector::Impl {
    std::atomic<bool> running{false};
    std::chrono::steady_clock::time_point last_update;
    
#ifdef _WIN32
    WSADATA wsa_data;
    bool wsa_initialized = false;
    
    bool init_wsa() {
        if (!wsa_initialized) {
            int result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
            wsa_initialized = (result == 0);
        }
        return wsa_initialized;
    }
    
    ~Impl() {
        if (wsa_initialized) {
            WSACleanup();
        }
    }
#endif
};

NetworkStateDetector::NetworkStateDetector() 
    : impl_(std::make_unique<Impl>()) {
    
#ifdef _WIN32
    impl_->init_wsa();
#endif
}

NetworkStateDetector::~NetworkStateDetector() {
    stopMonitoring();
}

bool NetworkStateDetector::testUDPConnectivity(const std::string& multicast_addr, uint16_t port) {
    // Create test socket
    int test_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (test_socket < 0) {
        std::cerr << "❌ UDP test: Failed to create socket" << std::endl;
        return false;
    }
    
    // Set short timeout for quick test
    struct timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(test_socket, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
    setsockopt(test_socket, SOL_SOCKET, SO_SNDTIMEO, (char*)&timeout, sizeof(timeout));
    
    // Setup multicast address
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, multicast_addr.c_str(), &addr.sin_addr);
    
    // Try to send a small test packet
    const char test_data[] = "UDP_TEST";
    ssize_t sent = sendto(test_socket, test_data, sizeof(test_data), 0, 
                         (struct sockaddr*)&addr, sizeof(addr));
    
#ifdef _WIN32
    closesocket(test_socket);
#else
    close(test_socket);
#endif
    
    if (sent < 0) {
        std::cerr << "❌ UDP test: Failed to send test packet (errno: " << errno << ")" << std::endl;
        return false;
    }
    
    std::cout << "✅ UDP test: Successfully sent test packet" << std::endl;
    return true;
}

bool NetworkStateDetector::testMulticastCapability(const std::string& multicast_addr, uint16_t port, uint32_t timeout_ms) {
    std::cout << "🔍 Testing multicast capability..." << std::endl;
    
    // Create socket for multicast test
    int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd < 0) {
        std::cerr << "❌ Multicast test: Failed to create socket" << std::endl;
        return false;
    }
    
    try {
        // Set socket options
        int reuse = 1;
        setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));
        
        // Setup multicast address
        struct sockaddr_in multicast_sockaddr;
        memset(&multicast_sockaddr, 0, sizeof(multicast_sockaddr));
        multicast_sockaddr.sin_family = AF_INET;
        multicast_sockaddr.sin_port = htons(port);
        inet_pton(AF_INET, multicast_addr.c_str(), &multicast_sockaddr.sin_addr);
        
        // Join multicast group
        struct ip_mreq mreq;
        mreq.imr_multiaddr = multicast_sockaddr.sin_addr;
        mreq.imr_interface.s_addr = INADDR_ANY;
        
        if (setsockopt(socket_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) < 0) {
            std::cerr << "❌ Multicast test: Failed to join multicast group (errno: " << errno << ")" << std::endl;
#ifdef _WIN32
            closesocket(socket_fd);
#else
            close(socket_fd);
#endif
            return false;
        }
        
        // Bind to receive
        struct sockaddr_in bind_addr;
        memset(&bind_addr, 0, sizeof(bind_addr));
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_port = htons(port);
        bind_addr.sin_addr.s_addr = INADDR_ANY;
        
        if (bind(socket_fd, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
            std::cerr << "❌ Multicast test: Failed to bind socket (errno: " << errno << ")" << std::endl;
#ifdef _WIN32
            closesocket(socket_fd);
#else
            close(socket_fd);
#endif
            return false;
        }
        
        // Set receive timeout
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, sizeof(tv));
        
        // Send test message to ourselves
        const char test_msg[] = "MULTICAST_DISCOVERY_TEST";
        ssize_t sent = sendto(socket_fd, test_msg, sizeof(test_msg), 0,
                             (struct sockaddr*)&multicast_sockaddr, sizeof(multicast_sockaddr));
        
        if (sent < 0) {
            std::cerr << "❌ Multicast test: Failed to send test message" << std::endl;
#ifdef _WIN32
            closesocket(socket_fd);
#else
            close(socket_fd);
#endif
            return false;
        }
        
        // Try to receive our own message
        char buffer[1024];
        struct sockaddr_in sender_addr;
        socklen_t addr_len = sizeof(sender_addr);
        
        ssize_t received = recvfrom(socket_fd, buffer, sizeof(buffer), 0,
                                   (struct sockaddr*)&sender_addr, &addr_len);
        
#ifdef _WIN32
        closesocket(socket_fd);
#else
        close(socket_fd);
#endif
        
        if (received > 0 && strncmp(buffer, test_msg, sizeof(test_msg)) == 0) {
            std::cout << "✅ Multicast test: Successfully received our own message" << std::endl;
            return true;
        } else {
            std::cerr << "❌ Multicast test: Did not receive test message (timeout or error)" << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
#ifdef _WIN32
        closesocket(socket_fd);
#else
        close(socket_fd);
#endif
        std::cerr << "❌ Multicast test: Exception - " << e.what() << std::endl;
        return false;
    }
}

bool NetworkStateDetector::hasNetworkPermission() {
#ifdef __APPLE__
    // On macOS, we can test network permission by trying to create a socket
    // and checking if we get permission denied
    int test_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (test_socket < 0) {
        if (errno == EPERM || errno == EACCES) {
            std::cerr << "❌ Network permission: Denied by system" << std::endl;
            return false;
        }
        std::cerr << "❌ Network permission: Socket creation failed (errno: " << errno << ")" << std::endl;
        return false;
    }
    
    close(test_socket);
    std::cout << "✅ Network permission: Granted" << std::endl;
    return true;
#else
    // On other platforms, assume permission is granted if we can create a socket
    int test_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (test_socket < 0) {
        return false;
    }
    
#ifdef _WIN32
    closesocket(test_socket);
#else
    close(test_socket);
#endif
    return true;
#endif
}

bool NetworkStateDetector::isNetworkInterfaceReady() {
    struct ifaddrs *ifap, *ifa;
    bool has_active_interface = false;
    
    if (getifaddrs(&ifap) == -1) {
        std::cerr << "❌ Interface check: Failed to get interface list" << std::endl;
        return false;
    }
    
    for (ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        
        // Skip loopback interface
        if (strcmp(ifa->ifa_name, "lo") == 0 || strcmp(ifa->ifa_name, "lo0") == 0) {
            continue;
        }
        
        if (ifa->ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;
            char ip_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &(addr_in->sin_addr), ip_str, INET_ADDRSTRLEN);
            
            // Check if it's not a link-local address (169.254.x.x)
            if (strncmp(ip_str, "169.254.", 8) != 0 && 
                strncmp(ip_str, "127.", 4) != 0) {
                std::cout << "✅ Active interface found: " << ifa->ifa_name << " (" << ip_str << ")" << std::endl;
                has_active_interface = true;
                break;
            }
        }
    }
    
    freeifaddrs(ifap);
    
    if (!has_active_interface) {
        std::cerr << "❌ No active network interfaces found" << std::endl;
    }
    
    return has_active_interface;
}

NetworkStateDetector::NetworkInfo NetworkStateDetector::getCurrentState() const {
    return cached_info_;
}

void NetworkStateDetector::startMonitoring(StateChangeCallback callback) {
    if (monitoring_.load()) {
        return;
    }
    
    monitoring_ = true;
    monitor_thread_ = std::thread(&NetworkStateDetector::monitorLoop, this, callback);
}

void NetworkStateDetector::stopMonitoring() {
    monitoring_ = false;
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void NetworkStateDetector::monitorLoop(StateChangeCallback callback) {
    impl_->running = true;
    
    while (monitoring_.load()) {
        updateNetworkState();
        
        if (callback) {
            callback(cached_info_);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    impl_->running = false;
}

void NetworkStateDetector::updateNetworkState() {
    NetworkInfo info;
    info.state = NetworkState::DISCONNECTED;
    
    // Step 1: Check network permission
    if (!hasNetworkPermission()) {
        info.state = NetworkState::ERROR;
        info.error_message = "Network permission denied by system";
        cached_info_ = info;
        cached_state_ = info.state;
        return;
    }
    
    // Step 2: Check if network interface is ready
    if (!isNetworkInterfaceReady()) {
        info.state = NetworkState::DISCONNECTED;
        info.error_message = "No active network interfaces";
        cached_info_ = info;
        cached_state_ = info.state;
        return;
    }
    
    // Step 3: Test basic UDP connectivity
    if (!testUDPConnectivity("239.255.77.77", 7777)) {
        info.state = NetworkState::ERROR;
        info.error_message = "UDP connectivity test failed";
        cached_info_ = info;
        cached_state_ = info.state;
        return;
    }
    
    // Step 4: Test multicast capability
    if (!testMulticastCapability("239.255.77.77", 7777, 1000)) {
        info.state = NetworkState::ERROR;
        info.error_message = "Multicast test failed";
        cached_info_ = info;
        cached_state_ = info.state;
        return;
    }
    
    // All tests passed
    info.state = NetworkState::CONNECTED;
    info.multicast_capable = true;
    cached_info_ = info;
    cached_state_ = info.state;
}

bool NetworkStateDetector::testBasicConnectivity() {
    return testUDPConnectivity("239.255.77.77", 7777);
}

bool NetworkStateDetector::pingGateway() {
    // This would require implementing ICMP ping
    // For now, we rely on UDP connectivity test
    return testBasicConnectivity();
}

} // namespace jam
