#include "../../include/jam_toast.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>

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

/**
 * Simple multicast utilities for TOAST protocol
 */
class MulticastHelper {
public:
    static bool setup_multicast_socket(int& socket_fd, const std::string& multicast_addr, uint16_t port, bool sender = false) {
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
        
        // Set socket reuse
        int reuse = 1;
        setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));
        
        if (!sender) {
            // Receiver: join multicast group and bind
            struct sockaddr_in bind_addr;
            memset(&bind_addr, 0, sizeof(bind_addr));
            bind_addr.sin_family = AF_INET;
            bind_addr.sin_port = htons(port);
            bind_addr.sin_addr.s_addr = INADDR_ANY;
            
            if (bind(socket_fd, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
                close(socket_fd);
                return false;
            }
            
            // Join multicast group
            struct ip_mreq mreq;
            inet_pton(AF_INET, multicast_addr.c_str(), &mreq.imr_multiaddr);
            mreq.imr_interface.s_addr = INADDR_ANY;
            
            if (setsockopt(socket_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) < 0) {
                close(socket_fd);
                return false;
            }
        } else {
            // Sender: configure TTL
            unsigned char ttl = 32;
            setsockopt(socket_fd, IPPROTO_IP, IP_MULTICAST_TTL, (char*)&ttl, sizeof(ttl));
        }
        
        return true;
    }
    
    static void close_socket(int socket_fd) {
        if (socket_fd >= 0) {
#ifdef _WIN32
            closesocket(socket_fd);
            WSACleanup();
#else
            close(socket_fd);
#endif
        }
    }
    
    static bool send_multicast(int socket_fd, const std::string& multicast_addr, uint16_t port, 
                              const void* data, size_t size) {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, multicast_addr.c_str(), &addr.sin_addr);
        
        ssize_t sent = sendto(socket_fd, (const char*)data, size, 0, 
                             (struct sockaddr*)&addr, sizeof(addr));
        
        return sent == static_cast<ssize_t>(size);
    }
    
    static ssize_t receive_multicast(int socket_fd, void* buffer, size_t buffer_size, 
                                    struct sockaddr_in* sender_addr = nullptr) {
        socklen_t addr_len = sizeof(struct sockaddr_in);
        
        if (sender_addr) {
            return recvfrom(socket_fd, (char*)buffer, buffer_size, 0, 
                           (struct sockaddr*)sender_addr, &addr_len);
        } else {
            return recv(socket_fd, (char*)buffer, buffer_size, 0);
        }
    }
};

/**
 * TOAST multicast session manager
 */
class TOASTMulticast {
private:
    int socket_fd = -1;
    std::string multicast_addr;
    uint16_t port;
    std::atomic<bool> running{false};
    std::thread receiver_thread;
    
    std::function<void(const std::vector<uint8_t>&, const std::string&)> message_callback;
    
public:
    TOASTMulticast() = default;
    
    ~TOASTMulticast() {
        stop();
    }
    
    bool start(const std::string& addr, uint16_t p, 
               std::function<void(const std::vector<uint8_t>&, const std::string&)> callback) {
        multicast_addr = addr;
        port = p;
        message_callback = std::move(callback);
        
        if (!MulticastHelper::setup_multicast_socket(socket_fd, multicast_addr, port, false)) {
            return false;
        }
        
        running = true;
        receiver_thread = std::thread(&TOASTMulticast::receiver_loop, this);
        
        return true;
    }
    
    void stop() {
        running = false;
        if (receiver_thread.joinable()) {
            receiver_thread.join();
        }
        MulticastHelper::close_socket(socket_fd);
        socket_fd = -1;
    }
    
    bool send(const std::vector<uint8_t>& data) {
        if (socket_fd < 0) return false;
        
        return MulticastHelper::send_multicast(socket_fd, multicast_addr, port, 
                                              data.data(), data.size());
    }
    
private:
    void receiver_loop() {
        uint8_t buffer[65536];
        
        while (running) {
            struct sockaddr_in sender_addr;
            ssize_t received = MulticastHelper::receive_multicast(socket_fd, buffer, sizeof(buffer), &sender_addr);
            
            if (received > 0 && message_callback) {
                std::vector<uint8_t> data(buffer, buffer + received);
                std::string sender_ip = inet_ntoa(sender_addr.sin_addr);
                message_callback(data, sender_ip);
            }
        }
    }
};

} // namespace jam
