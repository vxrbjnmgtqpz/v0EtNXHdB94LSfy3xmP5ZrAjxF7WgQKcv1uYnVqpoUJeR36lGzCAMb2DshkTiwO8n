/**
 * JAMNet Network Server - Fixes the missing server socket issue
 * 
 * This addresses the Technical Audit finding that no service was listening
 * on port 8888, causing all connection attempts to fail with ECONNREFUSED.
 */

#pragma once

#include <JuceHeader.h>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>

#ifdef __APPLE__
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#endif

class JAMNetworkServer : public juce::Thread {
public:
    struct ServerConfig {
        int port = 8888;
        std::string multicast_group = "224.0.2.60";
        bool enable_tcp_server = true;
        bool enable_udp_multicast = true;
        bool verbose_logging = true;
    };
    
    class Listener {
    public:
        virtual ~Listener() = default;
        virtual void onClientConnected(const std::string& client_ip) = 0;
        virtual void onClientDisconnected(const std::string& client_ip) = 0;
        virtual void onMessageReceived(const std::string& message, const std::string& from_ip) = 0;
        virtual void onMulticastMessageReceived(const std::string& message) = 0;
    };
    
    JAMNetworkServer(const ServerConfig& config = ServerConfig());
    ~JAMNetworkServer();
    
    // Server control
    bool startServer();
    void stopServer();
    bool isRunning() const { return is_running_; }
    
    // Send messages
    void sendToClient(const std::string& client_ip, const std::string& message);
    void sendMulticast(const std::string& message);
    void broadcastToAllClients(const std::string& message);
    
    // Listener management
    void addListener(Listener* listener);
    void removeListener(Listener* listener);
    
    // Status
    std::vector<std::string> getConnectedClients() const;
    ServerConfig getConfig() const { return config_; }
    
    // Thread implementation
    void run() override;
    
private:
    ServerConfig config_;
    std::atomic<bool> is_running_{false};
    
    // TCP Server
    int tcp_server_fd_ = -1;
    std::vector<int> client_sockets_;
    std::vector<std::string> client_ips_;
    
    // UDP Multicast
    int udp_multicast_fd_ = -1;
    
    // Listeners
    std::vector<Listener*> listeners_;
    mutable std::mutex listeners_mutex_;
    mutable std::mutex clients_mutex_;
    
    // Server methods
    bool setupTCPServer();
    bool setupUDPMulticast();
    void handleTCPConnections();
    void handleUDPMulticast();
    void acceptNewTCPClients();
    void handleExistingTCPClients();
    void notifyListeners(const std::function<void(Listener*)>& callback);
    void logMessage(const std::string& message);
    void cleanup();
};

// Implementation
JAMNetworkServer::JAMNetworkServer(const ServerConfig& config) 
    : Thread("JAMNetworkServer"), config_(config) {
    logMessage("JAMNetworkServer created with port " + std::to_string(config_.port));
}

JAMNetworkServer::~JAMNetworkServer() {
    stopServer();
}

bool JAMNetworkServer::startServer() {
    if (is_running_) {
        logMessage("Server already running");
        return true;
    }
    
    logMessage("🚀 Starting JAMNet Network Server...");
    
    // Setup TCP server if enabled
    if (config_.enable_tcp_server && !setupTCPServer()) {
        logMessage("❌ Failed to setup TCP server");
        return false;
    }
    
    // Setup UDP multicast if enabled
    if (config_.enable_udp_multicast && !setupUDPMulticast()) {
        logMessage("❌ Failed to setup UDP multicast");
        cleanup();
        return false;
    }
    
    is_running_ = true;
    startThread();
    
    logMessage("✅ JAMNet Network Server started successfully");
    logMessage("   📡 TCP Server: " + (config_.enable_tcp_server ? "ENABLED on port " + std::to_string(config_.port) : "DISABLED"));
    logMessage("   📻 UDP Multicast: " + (config_.enable_udp_multicast ? "ENABLED on " + config_.multicast_group : "DISABLED"));
    
    return true;
}

void JAMNetworkServer::stopServer() {
    if (!is_running_) return;
    
    logMessage("🛑 Stopping JAMNet Network Server...");
    is_running_ = false;
    
    // Signal thread to stop
    signalThreadShouldExit();
    waitForThreadToExit(5000);
    
    cleanup();
    logMessage("✅ JAMNet Network Server stopped");
}

bool JAMNetworkServer::setupTCPServer() {
    logMessage("🔧 Setting up TCP server on port " + std::to_string(config_.port));
    
#ifdef __APPLE__
    tcp_server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_server_fd_ < 0) {
        logMessage("❌ TCP socket creation failed: " + std::string(strerror(errno)));
        return false;
    }
    
    // CRITICAL FIX: Enable socket reuse
    int reuse = 1;
    if (setsockopt(tcp_server_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        logMessage("⚠️  SO_REUSEADDR failed: " + std::string(strerror(errno)));
    }
    
    if (setsockopt(tcp_server_fd_, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse)) < 0) {
        logMessage("⚠️  SO_REUSEPORT failed: " + std::string(strerror(errno)));
    }
    
    // Set non-blocking
    int flags = fcntl(tcp_server_fd_, F_GETFL, 0);
    if (fcntl(tcp_server_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
        logMessage("⚠️  fcntl non-blocking failed: " + std::string(strerror(errno)));
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(config_.port);
    
    if (bind(tcp_server_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        logMessage("❌ TCP bind failed: " + std::string(strerror(errno)));
        close(tcp_server_fd_);
        tcp_server_fd_ = -1;
        return false;
    }
    
    if (listen(tcp_server_fd_, 10) < 0) {
        logMessage("❌ TCP listen failed: " + std::string(strerror(errno)));
        close(tcp_server_fd_);
        tcp_server_fd_ = -1;
        return false;
    }
    
    logMessage("✅ TCP server listening on port " + std::to_string(config_.port));
    return true;
#else
    logMessage("⚠️  TCP server not implemented for this platform");
    return false;
#endif
}

bool JAMNetworkServer::setupUDPMulticast() {
    logMessage("🔧 Setting up UDP multicast on " + config_.multicast_group + ":" + std::to_string(config_.port));
    
#ifdef __APPLE__
    udp_multicast_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_multicast_fd_ < 0) {
        logMessage("❌ UDP socket creation failed: " + std::string(strerror(errno)));
        return false;
    }
    
    // CRITICAL FIX: Enable socket reuse
    int reuse = 1;
    if (setsockopt(udp_multicast_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        logMessage("⚠️  SO_REUSEADDR failed: " + std::string(strerror(errno)));
    }
    
    if (setsockopt(udp_multicast_fd_, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse)) < 0) {
        logMessage("⚠️  SO_REUSEPORT failed: " + std::string(strerror(errno)));
    }
    
    struct sockaddr_in multicast_addr;
    memset(&multicast_addr, 0, sizeof(multicast_addr));
    multicast_addr.sin_family = AF_INET;
    multicast_addr.sin_addr.s_addr = INADDR_ANY;
    multicast_addr.sin_port = htons(config_.port);
    
    if (bind(udp_multicast_fd_, (struct sockaddr*)&multicast_addr, sizeof(multicast_addr)) < 0) {
        logMessage("❌ UDP bind failed: " + std::string(strerror(errno)));
        close(udp_multicast_fd_);
        udp_multicast_fd_ = -1;
        return false;
    }
    
    // Join multicast group
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr(config_.multicast_group.c_str());
    mreq.imr_interface.s_addr = INADDR_ANY;
    
    if (setsockopt(udp_multicast_fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        logMessage("❌ Multicast group join failed: " + std::string(strerror(errno)));
        close(udp_multicast_fd_);
        udp_multicast_fd_ = -1;
        return false;
    }
    
    logMessage("✅ UDP multicast listening on " + config_.multicast_group + ":" + std::to_string(config_.port));
    return true;
#else
    logMessage("⚠️  UDP multicast not implemented for this platform");
    return false;
#endif
}

void JAMNetworkServer::run() {
    logMessage("🔄 Network server thread started");
    
    while (!threadShouldExit() && is_running_) {
        if (config_.enable_tcp_server && tcp_server_fd_ >= 0) {
            handleTCPConnections();
        }
        
        if (config_.enable_udp_multicast && udp_multicast_fd_ >= 0) {
            handleUDPMulticast();
        }
        
        // Small sleep to prevent busy waiting
        Thread::sleep(10);
    }
    
    logMessage("🔄 Network server thread stopped");
}

void JAMNetworkServer::handleTCPConnections() {
    acceptNewTCPClients();
    handleExistingTCPClients();
}

void JAMNetworkServer::acceptNewTCPClients() {
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(tcp_server_fd_, &read_fds);
    
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 1000; // 1ms timeout
    
    int result = select(tcp_server_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);
    if (result > 0 && FD_ISSET(tcp_server_fd_, &read_fds)) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_fd = accept(tcp_server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd >= 0) {
            char client_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
            
            std::lock_guard<std::mutex> lock(clients_mutex_);
            client_sockets_.push_back(client_fd);
            client_ips_.push_back(std::string(client_ip));
            
            logMessage("✅ New TCP client connected: " + std::string(client_ip));
            
            notifyListeners([client_ip](Listener* l) {
                l->onClientConnected(std::string(client_ip));
            });
        }
    }
}

void JAMNetworkServer::handleExistingTCPClients() {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    
    for (size_t i = 0; i < client_sockets_.size(); ) {
        int client_fd = client_sockets_[i];
        std::string client_ip = client_ips_[i];
        
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(client_fd, &read_fds);
        
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 1000;
        
        int result = select(client_fd + 1, &read_fds, nullptr, nullptr, &timeout);
        if (result > 0 && FD_ISSET(client_fd, &read_fds)) {
            char buffer[1024];
            ssize_t bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
            
            if (bytes > 0) {
                buffer[bytes] = '\0';
                std::string message(buffer);
                
                logMessage("📨 Received from " + client_ip + ": " + message);
                
                notifyListeners([message, client_ip](Listener* l) {
                    l->onMessageReceived(message, client_ip);
                });
                
                ++i;
            } else {
                // Client disconnected
                logMessage("❌ TCP client disconnected: " + client_ip);
                
                close(client_fd);
                client_sockets_.erase(client_sockets_.begin() + i);
                client_ips_.erase(client_ips_.begin() + i);
                
                notifyListeners([client_ip](Listener* l) {
                    l->onClientDisconnected(client_ip);
                });
            }
        } else {
            ++i;
        }
    }
}

void JAMNetworkServer::handleUDPMulticast() {
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(udp_multicast_fd_, &read_fds);
    
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 1000;
    
    int result = select(udp_multicast_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);
    if (result > 0 && FD_ISSET(udp_multicast_fd_, &read_fds)) {
        char buffer[1024];
        struct sockaddr_in sender_addr;
        socklen_t sender_len = sizeof(sender_addr);
        
        ssize_t bytes = recvfrom(udp_multicast_fd_, buffer, sizeof(buffer) - 1, 0,
                                (struct sockaddr*)&sender_addr, &sender_len);
        
        if (bytes > 0) {
            buffer[bytes] = '\0';
            std::string message(buffer);
            
            char sender_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &sender_addr.sin_addr, sender_ip, INET_ADDRSTRLEN);
            
            logMessage("📡 Multicast received from " + std::string(sender_ip) + ": " + message);
            
            notifyListeners([message](Listener* l) {
                l->onMulticastMessageReceived(message);
            });
        }
    }
}

void JAMNetworkServer::sendMulticast(const std::string& message) {
    if (udp_multicast_fd_ < 0) return;
    
    struct sockaddr_in multicast_addr;
    memset(&multicast_addr, 0, sizeof(multicast_addr));
    multicast_addr.sin_family = AF_INET;
    multicast_addr.sin_port = htons(config_.port);
    inet_pton(AF_INET, config_.multicast_group.c_str(), &multicast_addr.sin_addr);
    
    ssize_t sent = sendto(udp_multicast_fd_, message.c_str(), message.length(), 0,
                         (struct sockaddr*)&multicast_addr, sizeof(multicast_addr));
    
    if (sent >= 0) {
        logMessage("📡 Multicast sent: " + message);
    } else {
        logMessage("❌ Multicast send failed: " + std::string(strerror(errno)));
    }
}

void JAMNetworkServer::addListener(Listener* listener) {
    std::lock_guard<std::mutex> lock(listeners_mutex_);
    listeners_.push_back(listener);
}

void JAMNetworkServer::removeListener(Listener* listener) {
    std::lock_guard<std::mutex> lock(listeners_mutex_);
    listeners_.erase(std::remove(listeners_.begin(), listeners_.end(), listener), listeners_.end());
}

void JAMNetworkServer::notifyListeners(const std::function<void(Listener*)>& callback) {
    std::lock_guard<std::mutex> lock(listeners_mutex_);
    for (auto* listener : listeners_) {
        callback(listener);
    }
}

void JAMNetworkServer::logMessage(const std::string& message) {
    if (config_.verbose_logging) {
        std::cout << "[JAMNetworkServer] " << message << std::endl;
    }
}

void JAMNetworkServer::cleanup() {
    if (tcp_server_fd_ >= 0) {
        close(tcp_server_fd_);
        tcp_server_fd_ = -1;
    }
    
    if (udp_multicast_fd_ >= 0) {
        close(udp_multicast_fd_);
        udp_multicast_fd_ = -1;
    }
    
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (int fd : client_sockets_) {
        close(fd);
    }
    client_sockets_.clear();
    client_ips_.clear();
}
