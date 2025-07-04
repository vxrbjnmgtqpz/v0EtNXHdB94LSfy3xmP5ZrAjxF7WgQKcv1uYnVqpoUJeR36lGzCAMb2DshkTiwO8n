#include "../../include/jam_toast.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <sstream>
#include <map>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace jam {

// TOASTDiscovery internal implementation
class TOASTDiscovery::Impl {
public:
    int socket_fd = -1;
    struct sockaddr_in multicast_addr;
    std::atomic<bool> running{false};
    std::thread discovery_thread;
    std::string local_name;
    std::vector<PeerInfo> discovered_peers;
    std::mutex peers_mutex;
    
    bool initialize_socket(const std::string& addr, uint16_t port) {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            return false;
        }
#endif
        
        socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd < 0) {
            return false;
        }
        
        // Set socket options
        int reuse = 1;
        setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));
        
        // Enable broadcast for discovery
        int broadcast = 1;
        setsockopt(socket_fd, SOL_SOCKET, SO_BROADCAST, (char*)&broadcast, sizeof(broadcast));
        
        // Setup multicast address
        memset(&multicast_addr, 0, sizeof(multicast_addr));
        multicast_addr.sin_family = AF_INET;
        multicast_addr.sin_port = htons(port);
        inet_pton(AF_INET, addr.c_str(), &multicast_addr.sin_addr);
        
        // Join multicast group for receiving
        struct ip_mreq mreq;
        mreq.imr_multiaddr = multicast_addr.sin_addr;
        mreq.imr_interface.s_addr = INADDR_ANY;
        
        if (setsockopt(socket_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) < 0) {
            close(socket_fd);
            return false;
        }
        
        // Bind to receive discovery messages
        struct sockaddr_in bind_addr;
        memset(&bind_addr, 0, sizeof(bind_addr));
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_port = htons(port);
        bind_addr.sin_addr.s_addr = INADDR_ANY;
        
        if (bind(socket_fd, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
            close(socket_fd);
            return false;
        }
        
        return true;
    }
    
    void shutdown_socket() {
        if (socket_fd >= 0) {
#ifdef _WIN32
            closesocket(socket_fd);
            WSACleanup();
#else
            close(socket_fd);
#endif
            socket_fd = -1;
        }
    }
    
    bool send_discovery_message() {
        // Simple text-based discovery message format
        std::ostringstream msg;
        msg << "TOAST_DISCOVERY|" << local_name << "|" 
            << std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count();
        
        std::string message = msg.str();
        
        ssize_t sent = sendto(socket_fd, message.c_str(), message.length(), 0,
                             (struct sockaddr*)&multicast_addr, sizeof(multicast_addr));
        
        return sent == static_cast<ssize_t>(message.length());
    }
    
    void discovery_loop(TOASTDiscovery* discovery) {
        uint8_t buffer[1024];
        
        // Send initial announcement
        send_discovery_message();
        
        auto last_announce = std::chrono::steady_clock::now();
        
        while (running) {
            // Send periodic announcements (every 5 seconds)
            auto now = std::chrono::steady_clock::now();
            if (now - last_announce > std::chrono::seconds(5)) {
                send_discovery_message();
                last_announce = now;
            }
            
            // Listen for discovery messages with timeout
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(socket_fd, &read_fds);
            
            struct timeval timeout;
            timeout.tv_sec = 1;
            timeout.tv_usec = 0;
            
            int activity = select(socket_fd + 1, &read_fds, nullptr, nullptr, &timeout);
            
            if (activity > 0 && FD_ISSET(socket_fd, &read_fds)) {
                struct sockaddr_in sender_addr;
                socklen_t addr_len = sizeof(sender_addr);
                
                ssize_t received = recvfrom(socket_fd, (char*)buffer, sizeof(buffer) - 1, 0,
                                          (struct sockaddr*)&sender_addr, &addr_len);
                
                if (received > 0) {
                    buffer[received] = '\0';
                    std::string message(reinterpret_cast<char*>(buffer));
                    
                    // Parse simple discovery message: "TOAST_DISCOVERY|name|timestamp"
                    if (message.find("TOAST_DISCOVERY|") == 0) {
                        size_t first_pipe = message.find('|');
                        size_t second_pipe = message.find('|', first_pipe + 1);
                        
                        if (first_pipe != std::string::npos && second_pipe != std::string::npos) {
                            std::string peer_name = message.substr(first_pipe + 1, second_pipe - first_pipe - 1);
                            
                            if (peer_name != local_name) {
                                PeerInfo peer;
                                peer.address = inet_ntoa(sender_addr.sin_addr);
                                peer.port = ntohs(sender_addr.sin_port);
                                peer.name = peer_name;
                                peer.session_id = 0; // Not implemented yet
                                peer.last_seen_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch()
                                ).count();
                                
                                discovery->add_peer(peer);
                            }
                        }
                    }
                }
            }
        }
    }
};

// TOASTDiscovery implementation
TOASTDiscovery::TOASTDiscovery() : impl_(std::make_unique<Impl>()) {}

TOASTDiscovery::~TOASTDiscovery() {
    stop_discovery();
}

bool TOASTDiscovery::start_discovery(const std::string& multicast_addr, uint16_t port,
                                    const std::string& local_name) {
    impl_->local_name = local_name;
    
    if (!impl_->initialize_socket(multicast_addr, port)) {
        return false;
    }
    
    impl_->running = true;
    impl_->discovery_thread = std::thread(&Impl::discovery_loop, impl_.get(), this);
    
    return true;
}

void TOASTDiscovery::stop_discovery() {
    impl_->running = false;
    if (impl_->discovery_thread.joinable()) {
        impl_->discovery_thread.join();
    }
    impl_->shutdown_socket();
}

std::vector<TOASTDiscovery::PeerInfo> TOASTDiscovery::get_peers() const {
    std::lock_guard<std::mutex> lock(impl_->peers_mutex);
    return impl_->discovered_peers;
}

void TOASTDiscovery::announce() {
    impl_->send_discovery_message();
}

// Internal method to add discovered peer
void TOASTDiscovery::add_peer(const PeerInfo& peer) {
    std::lock_guard<std::mutex> lock(impl_->peers_mutex);
    
    // Update existing peer or add new one
    auto it = std::find_if(impl_->discovered_peers.begin(), impl_->discovered_peers.end(),
                          [&peer](const PeerInfo& existing) {
                              return existing.address == peer.address && existing.name == peer.name;
                          });
    
    if (it != impl_->discovered_peers.end()) {
        *it = peer; // Update existing
    } else {
        impl_->discovered_peers.push_back(peer); // Add new
    }
    
    // Call callback if set
    if (peer_callback_) {
        peer_callback_(peer);
    }
    
    // Clean up old peers (older than 30 seconds)
    auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    impl_->discovered_peers.erase(
        std::remove_if(impl_->discovered_peers.begin(), impl_->discovered_peers.end(),
                      [now](const PeerInfo& p) {
                          return (now - p.last_seen_us) > 30000000; // 30 seconds
                      }),
        impl_->discovered_peers.end()
    );
}

} // namespace jam
