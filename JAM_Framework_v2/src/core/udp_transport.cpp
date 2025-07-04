/**
 * JAM Framework v2: Pure UDP Transport Implementation
 * 
 * NO TCP/HTTP - UDP multicast only
 * Fire-and-forget, stateless messaging
 */

#include "jam_transport.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>

namespace jam {

class UDPTransportImpl : public UDPTransport {
private:
    std::string multicast_group_;
    uint16_t port_;
    std::string interface_ip_;
    
    int send_socket_ = -1;
    int recv_socket_ = -1;
    
    std::atomic<bool> receiving_{false};
    std::thread recv_thread_;
    
    std::function<void(std::span<const uint8_t>)> receive_callback_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    TransportStats stats_{};
    
    // Sequence number generation
    std::atomic<uint32_t> sequence_counter_{1};

public:
    UDPTransportImpl(const std::string& multicast_group, uint16_t port, const std::string& interface_ip)
        : multicast_group_(multicast_group), port_(port), interface_ip_(interface_ip) {
        
        // Initialize UDP sockets immediately
        init_sockets();
    }
    
    ~UDPTransportImpl() {
        stop_receiving();
        cleanup_sockets();
    }
    
    bool join_multicast() override {
        if (recv_socket_ < 0) return false;
        
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(multicast_group_.c_str());
        mreq.imr_interface.s_addr = interface_ip_.empty() ? INADDR_ANY : inet_addr(interface_ip_.c_str());
        
        if (setsockopt(recv_socket_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
            return false;
        }
        
        return true;
    }
    
    void leave_multicast() override {
        if (recv_socket_ < 0) return;
        
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(multicast_group_.c_str());
        mreq.imr_interface.s_addr = interface_ip_.empty() ? INADDR_ANY : inet_addr(interface_ip_.c_str());
        
        setsockopt(recv_socket_, IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq));
    }
    
    uint64_t send_immediate(const void* data, size_t size) override {
        if (send_socket_ < 0 || !data || size == 0) return 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create UDP packet with JAM header
        std::vector<uint8_t> packet_data(sizeof(UDPPacketHeader) + size);
        
        UDPPacketHeader* header = reinterpret_cast<UDPPacketHeader*>(packet_data.data());
        *header = udp_utils::create_header(
            PacketType::JSONL,
            static_cast<uint16_t>(size),
            udp_utils::next_sequence_number()
        );
        
        // Copy payload
        std::memcpy(packet_data.data() + sizeof(UDPPacketHeader), data, size);
        
        // Calculate checksum over header + payload
        header->checksum = udp_utils::calculate_checksum(packet_data.data(), packet_data.size());
        
        // Send via UDP multicast
        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr(multicast_group_.c_str());
        addr.sin_port = htons(port_);
        
        ssize_t sent = sendto(send_socket_, packet_data.data(), packet_data.size(), 0,
                              reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t send_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (sent > 0) {
                stats_.packets_sent++;
                stats_.bytes_sent += sent;
                stats_.avg_send_time_us = (stats_.avg_send_time_us + send_time_us) / 2;
            } else {
                stats_.send_failures++;
            }
        }
        
        return sent > 0 ? send_time_us : 0;
    }
    
    uint64_t send_burst(const void* data, size_t size, uint8_t burst_count, uint16_t burst_interval_us) override {
        if (burst_count == 0 || burst_count > 5) burst_count = 1;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        uint16_t burst_id = static_cast<uint16_t>(sequence_counter_.fetch_add(1));
        uint64_t total_send_time = 0;
        
        for (uint8_t i = 0; i < burst_count; i++) {
            // Create packet with burst info
            std::vector<uint8_t> packet_data(sizeof(UDPPacketHeader) + size);
            
            UDPPacketHeader* header = reinterpret_cast<UDPPacketHeader*>(packet_data.data());
            *header = udp_utils::create_header(
                PacketType::JSONL,
                static_cast<uint16_t>(size),
                udp_utils::next_sequence_number(),
                burst_id,
                i
            );
            
            // Copy payload
            std::memcpy(packet_data.data() + sizeof(UDPPacketHeader), data, size);
            
            // Calculate checksum
            header->checksum = udp_utils::calculate_checksum(packet_data.data(), packet_data.size());
            
            // Send packet
            struct sockaddr_in addr;
            std::memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = inet_addr(multicast_group_.c_str());
            addr.sin_port = htons(port_);
            
            ssize_t sent = sendto(send_socket_, packet_data.data(), packet_data.size(), 0,
                                  reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
            
            if (sent > 0) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.packets_sent++;
                stats_.bytes_sent += sent;
            } else {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.send_failures++;
            }
            
            // Wait between burst packets (except for last one)
            if (i < burst_count - 1 && burst_interval_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(burst_interval_us));
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        total_send_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        return total_send_time;
    }
    
    void set_receive_callback(std::function<void(std::span<const uint8_t>)> callback) override {
        receive_callback_ = callback;
    }
    
    bool start_receiving() override {
        if (receiving_.load() || recv_socket_ < 0) return false;
        
        if (!join_multicast()) return false;
        
        receiving_.store(true);
        recv_thread_ = std::thread(&UDPTransportImpl::receive_loop, this);
        
        return true;
    }
    
    void stop_receiving() override {
        receiving_.store(false);
        
        if (recv_thread_.joinable()) {
            recv_thread_.join();
        }
        
        leave_multicast();
    }
    
    TransportStats get_stats() const override {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }
    
    std::string get_multicast_group() const override {
        return multicast_group_;
    }
    
    uint16_t get_port() const override {
        return port_;
    }

private:
    void init_sockets() {
        // Create send socket
        send_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (send_socket_ >= 0) {
            // Enable multicast
            int ttl = 1;
            setsockopt(send_socket_, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));
            
            // Bind to interface if specified
            if (!interface_ip_.empty()) {
                struct in_addr addr;
                addr.s_addr = inet_addr(interface_ip_.c_str());
                setsockopt(send_socket_, IPPROTO_IP, IP_MULTICAST_IF, &addr, sizeof(addr));
            }
        }
        
        // Create receive socket
        recv_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (recv_socket_ >= 0) {
            // Allow address reuse
            int reuse = 1;
            setsockopt(recv_socket_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
            
            // Bind to multicast port
            struct sockaddr_in addr;
            std::memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = INADDR_ANY;
            addr.sin_port = htons(port_);
            
            if (bind(recv_socket_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
                close(recv_socket_);
                recv_socket_ = -1;
            }
        }
    }
    
    void cleanup_sockets() {
        if (send_socket_ >= 0) {
            close(send_socket_);
            send_socket_ = -1;
        }
        
        if (recv_socket_ >= 0) {
            close(recv_socket_);
            recv_socket_ = -1;
        }
    }
    
    void receive_loop() {
        std::vector<uint8_t> buffer(65536); // Max UDP packet size
        
        while (receiving_.load()) {
            struct sockaddr_in sender_addr;
            socklen_t addr_len = sizeof(sender_addr);
            
            ssize_t received = recvfrom(recv_socket_, buffer.data(), buffer.size(), 0,
                                        reinterpret_cast<struct sockaddr*>(&sender_addr), &addr_len);
            
            if (received > 0) {
                // Update statistics
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.packets_received++;
                    stats_.bytes_received += received;
                }
                
                // Validate packet
                if (received >= sizeof(UDPPacketHeader)) {
                    const UDPPacketHeader* header = reinterpret_cast<const UDPPacketHeader*>(buffer.data());
                    
                    if (udp_utils::validate_header(*header, received)) {
                        // Extract payload
                        const uint8_t* payload = buffer.data() + sizeof(UDPPacketHeader);
                        size_t payload_size = received - sizeof(UDPPacketHeader);
                        
                        // Call callback if set
                        if (receive_callback_) {
                            receive_callback_(std::span<const uint8_t>(payload, payload_size));
                        }
                    }
                }
            }
        }
    }
};

// Static factory method
std::unique_ptr<UDPTransport> UDPTransport::create(
    const std::string& multicast_group,
    uint16_t port,
    const std::string& interface_ip
) {
    return std::make_unique<UDPTransportImpl>(multicast_group, port, interface_ip);
}

// UDP utilities implementation
namespace udp_utils {
    uint32_t calculate_checksum(const void* data, size_t size) {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        uint32_t checksum = 0;
        
        for (size_t i = 0; i < size; i++) {
            checksum = checksum * 31 + bytes[i];
        }
        
        return checksum;
    }
    
    bool validate_header(const UDPPacketHeader& header, size_t total_packet_size) {
        // Check magic number
        if (header.magic != 0x4A414D32) return false; // "JAM2"
        
        // Check version
        if (header.version != 0x0001) return false;
        
        // Check size consistency
        if (sizeof(UDPPacketHeader) + header.payload_size != total_packet_size) return false;
        
        // Additional validation could include checksum verification
        // For now, basic structure validation
        
        return true;
    }
    
    UDPPacketHeader create_header(
        PacketType type,
        uint16_t payload_size,
        uint32_t sequence_number,
        uint16_t burst_id,
        uint8_t burst_index
    ) {
        UDPPacketHeader header;
        std::memset(&header, 0, sizeof(header));
        
        header.magic = 0x4A414D32; // "JAM2"
        header.version = 0x0001;
        header.payload_size = payload_size;
        header.timestamp_us = get_timestamp_us();
        header.sequence_number = sequence_number;
        header.burst_id = burst_id;
        header.burst_index = burst_index;
        header.packet_type = static_cast<uint8_t>(type);
        header.checksum = 0; // Will be calculated later
        
        return header;
    }
    
    uint64_t get_timestamp_us() {
        auto now = std::chrono::high_resolution_clock::now();
        auto epoch = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
    }
    
    static std::atomic<uint32_t> sequence_number_{1};
    
    uint32_t next_sequence_number() {
        return sequence_number_.fetch_add(1);
    }
}

} // namespace jam
