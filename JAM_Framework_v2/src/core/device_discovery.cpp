#include "../include/device_discovery.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <mutex>
#include <map>

#ifdef __APPLE__
#include <ifaddrs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <unistd.h>
#include <fcntl.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/network/IOEthernetInterface.h>
#include <IOKit/network/IONetworkInterface.h>
#elif _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#else
#include <ifaddrs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace jam {

struct DeviceDiscovery::Impl {
    std::vector<NetworkInterface> interfaces;
    std::map<std::string, DiscoveredDevice> discovered_devices;
    std::map<std::string, int> interface_sockets;
    std::atomic<bool> running{false};
    mutable std::mutex devices_mutex;
    mutable std::mutex interfaces_mutex;
    
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

DeviceDiscovery::DeviceDiscovery() : impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
    impl_->init_wsa();
#endif
}

DeviceDiscovery::~DeviceDiscovery() {
    stopDiscovery();
}

bool DeviceDiscovery::startDiscovery(const std::string& multicast_addr, uint16_t port) {
    if (discovering_.load()) {
        return true;
    }
    
    multicast_address_ = multicast_addr;
    discovery_port_ = port;
    
    std::cout << "🔍 Starting enhanced device discovery on all interfaces..." << std::endl;
    std::cout << "   Multicast: " << multicast_addr << ":" << port << std::endl;
    
    // First scan available network interfaces
    scanNetworkInterfaces();
    
    auto interfaces = getAvailableInterfaces();
    if (interfaces.empty()) {
        std::cerr << "❌ No network interfaces found" << std::endl;
        return false;
    }
    
    std::cout << "✅ Found " << interfaces.size() << " network interfaces:" << std::endl;
    for (const auto& iface : interfaces) {
        std::cout << "   - " << iface.display_name << " (" << iface.name << ") " << iface.ip_address;
        if (iface.is_usb4) std::cout << " [USB4]";
        if (iface.is_thunderbolt) std::cout << " [Thunderbolt]";
        if (iface.is_wifi) std::cout << " [WiFi]";
        if (iface.is_ethernet) std::cout << " [Ethernet]";
        std::cout << std::endl;
    }
    
    discovering_ = true;
    discovery_thread_ = std::thread(&DeviceDiscovery::discoveryLoop, this);
    
    return true;
}

void DeviceDiscovery::stopDiscovery() {
    discovering_ = false;
    if (discovery_thread_.joinable()) {
        discovery_thread_.join();
    }
    
    // Close all interface sockets
    for (auto& [interface_name, socket_fd] : impl_->interface_sockets) {
        if (socket_fd >= 0) {
#ifdef _WIN32
            closesocket(socket_fd);
#else
            close(socket_fd);
#endif
        }
    }
    impl_->interface_sockets.clear();
}

void DeviceDiscovery::discoveryLoop() {
    impl_->running = true;
    
    while (discovering_.load()) {
        // Rescan interfaces periodically in case USB4 devices are connected
        scanNetworkInterfaces();
        
        // Send discovery pings on all active interfaces
        auto interfaces = getAvailableInterfaces();
        for (const auto& interface : interfaces) {
            if (interface.is_active && interface.is_multicast_capable) {
                sendDiscoveryPing(interface);
            }
        }
        
        // Process any received responses
        processReceivedPackets();
        
        // Clean up stale devices (not seen for 10 seconds)
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        {
            std::lock_guard<std::mutex> lock(impl_->devices_mutex);
            auto it = impl_->discovered_devices.begin();
            while (it != impl_->discovered_devices.end()) {
                if (now - it->second.last_seen_ms > 10000) {
                    std::cout << "🔍 Device lost: " << it->second.device_name << std::endl;
                    if (device_lost_callback_) {
                        device_lost_callback_(it->first);
                    }
                    it = impl_->discovered_devices.erase(it);
                } else {
                    ++it;
                }
            }
        }
        
        // Discovery interval
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
    
    impl_->running = false;
}

void DeviceDiscovery::scanNetworkInterfaces() {
    std::vector<NetworkInterface> new_interfaces;
    
    struct ifaddrs *ifap, *ifa;
    if (getifaddrs(&ifap) == -1) {
        std::cerr << "❌ Failed to get network interfaces" << std::endl;
        return;
    }
    
    for (ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        
        // Only process IPv4 interfaces
        if (ifa->ifa_addr->sa_family != AF_INET) continue;
        
        // Skip loopback
        if (strcmp(ifa->ifa_name, "lo") == 0 || strcmp(ifa->ifa_name, "lo0") == 0) {
            continue;
        }
        
        NetworkInterface interface = identifyInterface(ifa->ifa_name);
        
        struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(addr_in->sin_addr), ip_str, INET_ADDRSTRLEN);
        interface.ip_address = ip_str;
        
        // Check if it's a real IP (not link-local)
        if (strncmp(ip_str, "169.254.", 8) != 0 && strncmp(ip_str, "127.", 4) != 0) {
            interface.is_active = true;
        }
        
        // Test multicast capability
        if (interface.is_active) {
            interface.is_multicast_capable = testInterface(interface.name, multicast_address_, discovery_port_);
        }
        
        new_interfaces.push_back(interface);
        
        // Notify about new interface
        if (interface_found_callback_) {
            interface_found_callback_(interface);
        }
    }
    
    freeifaddrs(ifap);
    
    {
        std::lock_guard<std::mutex> lock(impl_->interfaces_mutex);
        impl_->interfaces = new_interfaces;
    }
}

DeviceDiscovery::NetworkInterface DeviceDiscovery::identifyInterface(const std::string& interface_name) {
    NetworkInterface interface;
    interface.name = interface_name;
    
    // Identify interface type based on name patterns
    if (interface_name.find("en") == 0) {
        if (isUSB4Interface(interface_name) || isThunderboltInterface(interface_name)) {
            interface.display_name = "USB4/Thunderbolt";
            interface.is_usb4 = true;
            interface.is_thunderbolt = true;
        } else {
            interface.display_name = "Ethernet";
            interface.is_ethernet = true;
        }
    } else if (interface_name.find("bridge") == 0) {
        interface.display_name = "Thunderbolt Bridge";
        interface.is_thunderbolt = true;
        interface.is_usb4 = true;
    } else if (interface_name.find("usb") == 0) {
        interface.display_name = "USB Ethernet";
        interface.is_usb4 = true;
    } else if (interface_name.find("wifi") == 0 || interface_name.find("wlan") == 0) {
        interface.display_name = "WiFi";
        interface.is_wifi = true;
    } else {
        interface.display_name = interface_name;
    }
    
    return interface;
}

bool DeviceDiscovery::isUSB4Interface(const std::string& interface_name) {
#ifdef __APPLE__
    // On macOS, USB4/Thunderbolt interfaces often appear as en1, en2, etc.
    // We need to check IOKit to see if it's actually USB4/Thunderbolt
    
    io_iterator_t iterator;
    io_object_t service;
    
    kern_return_t result = IOServiceGetMatchingServices(kIOMasterPortDefault,
                                                        IOServiceMatching(kIOEthernetInterfaceClass),
                                                        &iterator);
    
    if (result != KERN_SUCCESS) {
        return false;
    }
    
    bool is_usb4 = false;
    while ((service = IOIteratorNext(iterator)) != IO_OBJECT_NULL) {
        CFStringRef bsd_name_ref = (CFStringRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("BSD Name"), kCFAllocatorDefault, 0);
        
        if (bsd_name_ref) {
            char bsd_name[256];
            CFStringGetCString(bsd_name_ref, bsd_name, sizeof(bsd_name), kCFStringEncodingUTF8);
            
            if (interface_name == bsd_name) {
                // Check if parent is USB or Thunderbolt
                io_object_t parent;
                if (IORegistryEntryGetParentEntry(service, kIOServicePlane, &parent) == KERN_SUCCESS) {
                    CFStringRef class_name = (CFStringRef)IOObjectCopyClass(parent);
                    if (class_name) {
                        char class_str[256];
                        CFStringGetCString(class_name, class_str, sizeof(class_str), kCFStringEncodingUTF8);
                        
                        if (strstr(class_str, "USB") || strstr(class_str, "Thunderbolt") || 
                            strstr(class_str, "AppleThunderbolt")) {
                            is_usb4 = true;
                        }
                        
                        CFRelease(class_name);
                    }
                    IOObjectRelease(parent);
                }
            }
            
            CFRelease(bsd_name_ref);
        }
        
        IOObjectRelease(service);
        
        if (is_usb4) break;
    }
    
    IOObjectRelease(iterator);
    return is_usb4;
#else
    // On other platforms, check for USB in interface name or path
    return interface_name.find("usb") != std::string::npos ||
           interface_name.find("thunderbolt") != std::string::npos;
#endif
}

bool DeviceDiscovery::isThunderboltInterface(const std::string& interface_name) {
    return isUSB4Interface(interface_name); // USB4 and Thunderbolt overlap on modern systems
}

bool DeviceDiscovery::testInterface(const std::string& interface_name, const std::string& multicast_addr, uint16_t port) {
    // Create test socket for this interface
    int test_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (test_socket < 0) {
        return false;
    }
    
    // Set socket options
    int reuse = 1;
    setsockopt(test_socket, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));
    
    // Bind to specific interface if possible
    struct sockaddr_in bind_addr;
    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons(port + 1); // Use different port for testing
    bind_addr.sin_addr.s_addr = INADDR_ANY;
    
    if (bind(test_socket, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
#ifdef _WIN32
        closesocket(test_socket);
#else
        close(test_socket);
#endif
        return false;
    }
    
    // Test multicast send
    struct sockaddr_in multicast_sockaddr;
    memset(&multicast_sockaddr, 0, sizeof(multicast_sockaddr));
    multicast_sockaddr.sin_family = AF_INET;
    multicast_sockaddr.sin_port = htons(port);
    inet_pton(AF_INET, multicast_addr.c_str(), &multicast_sockaddr.sin_addr);
    
    const char test_msg[] = "INTERFACE_TEST";
    ssize_t sent = sendto(test_socket, test_msg, sizeof(test_msg), 0,
                         (struct sockaddr*)&multicast_sockaddr, sizeof(multicast_sockaddr));
    
#ifdef _WIN32
    closesocket(test_socket);
#else
    close(test_socket);
#endif
    
    return sent > 0;
}

void DeviceDiscovery::sendDiscoveryPing(const NetworkInterface& interface) {
    // Get or create socket for this interface
    int socket_fd = -1;
    auto it = impl_->interface_sockets.find(interface.name);
    if (it != impl_->interface_sockets.end()) {
        socket_fd = it->second;
    } else {
        // Create new socket for this interface
        socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd < 0) {
            return;
        }
        
        int reuse = 1;
        setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));
        
        impl_->interface_sockets[interface.name] = socket_fd;
    }
    
    // Create discovery message
    std::string discovery_msg = "TOAST_DISCOVERY:" + interface.name + ":" + interface.ip_address;
    
    // Send to multicast address
    struct sockaddr_in multicast_addr;
    memset(&multicast_addr, 0, sizeof(multicast_addr));
    multicast_addr.sin_family = AF_INET;
    multicast_addr.sin_port = htons(discovery_port_);
    inet_pton(AF_INET, multicast_address_.c_str(), &multicast_addr.sin_addr);
    
    ssize_t sent = sendto(socket_fd, discovery_msg.c_str(), discovery_msg.length(), 0,
                         (struct sockaddr*)&multicast_addr, sizeof(multicast_addr));
    
    if (sent > 0) {
        std::cout << "🔍 Sent discovery ping on " << interface.display_name << " (" << interface.name << ")" << std::endl;
    }
}

void DeviceDiscovery::processReceivedPackets() {
    // Process packets on all interface sockets
    for (auto& [interface_name, socket_fd] : impl_->interface_sockets) {
        if (socket_fd < 0) continue;
        
        // Set non-blocking mode
        int flags = fcntl(socket_fd, F_GETFL, 0);
        fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK);
        
        char buffer[1024];
        struct sockaddr_in sender_addr;
        socklen_t addr_len = sizeof(sender_addr);
        
        ssize_t received = recvfrom(socket_fd, buffer, sizeof(buffer) - 1, 0,
                                   (struct sockaddr*)&sender_addr, &addr_len);
        
        if (received > 0) {
            buffer[received] = '\0';
            
            // Parse discovery response
            if (strncmp(buffer, "TOAST_DISCOVERY:", 16) == 0) {
                char sender_ip[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &(sender_addr.sin_addr), sender_ip, INET_ADDRSTRLEN);
                
                DiscoveredDevice device;
                device.device_id = sender_ip;
                device.device_name = "TOASTer Device";
                device.ip_address = sender_ip;
                device.port = discovery_port_;
                device.interface_name = interface_name;
                device.connection_type = "UDP Multicast";
                device.last_seen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                device.is_responding = true;
                
                {
                    std::lock_guard<std::mutex> lock(impl_->devices_mutex);
                    bool is_new = impl_->discovered_devices.find(device.device_id) == impl_->discovered_devices.end();
                    impl_->discovered_devices[device.device_id] = device;
                    
                    if (is_new) {
                        std::cout << "🔍 New device discovered: " << device.device_name << " at " << device.ip_address << std::endl;
                        if (device_found_callback_) {
                            device_found_callback_(device);
                        }
                    }
                }
            }
        }
        
        // Restore blocking mode
        fcntl(socket_fd, F_SETFL, flags);
    }
}

bool DeviceDiscovery::pingDevice(const std::string& ip_address, uint16_t port, uint32_t timeout_ms) {
    int test_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (test_socket < 0) {
        return false;
    }
    
    // Set timeout
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(test_socket, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, sizeof(tv));
    setsockopt(test_socket, SOL_SOCKET, SO_SNDTIMEO, (char*)&tv, sizeof(tv));
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip_address.c_str(), &addr.sin_addr);
    
    const char ping_msg[] = "PING";
    ssize_t sent = sendto(test_socket, ping_msg, sizeof(ping_msg), 0,
                         (struct sockaddr*)&addr, sizeof(addr));
    
#ifdef _WIN32
    closesocket(test_socket);
#else
    close(test_socket);
#endif
    
    return sent > 0;
}

std::vector<DeviceDiscovery::NetworkInterface> DeviceDiscovery::getAvailableInterfaces() const {
    std::lock_guard<std::mutex> lock(impl_->interfaces_mutex);
    return impl_->interfaces;
}

std::vector<DeviceDiscovery::DiscoveredDevice> DeviceDiscovery::getDiscoveredDevices() const {
    std::lock_guard<std::mutex> lock(impl_->devices_mutex);
    std::vector<DiscoveredDevice> devices;
    for (const auto& [id, device] : impl_->discovered_devices) {
        devices.push_back(device);
    }
    return devices;
}

} // namespace jam
