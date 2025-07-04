#include "../../include/jam_toast.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <random>
#include <cstring>

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

// CRC16 checksum calculation
uint16_t calculate_crc16(const uint8_t* data, size_t length) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

// TOASTFrame implementation
void TOASTFrame::calculate_checksum() {
    header.checksum = 0; // Reset checksum field
    
    // Calculate checksum over header and payload
    std::vector<uint8_t> data;
    data.resize(sizeof(TOASTFrameHeader) + payload.size());
    
    std::memcpy(data.data(), &header, sizeof(TOASTFrameHeader));
    if (!payload.empty()) {
        std::memcpy(data.data() + sizeof(TOASTFrameHeader), payload.data(), payload.size());
    }
    
    header.checksum = calculate_crc16(data.data(), data.size());
}

bool TOASTFrame::validate_checksum() const {
    uint16_t stored_checksum = header.checksum;
    
    // Create copy and recalculate
    TOASTFrame temp = *this;
    temp.calculate_checksum();
    
    return temp.header.checksum == stored_checksum;
}

std::vector<uint8_t> TOASTFrame::serialize() const {
    std::vector<uint8_t> data;
    data.resize(sizeof(TOASTFrameHeader) + payload.size());
    
    std::memcpy(data.data(), &header, sizeof(TOASTFrameHeader));
    if (!payload.empty()) {
        std::memcpy(data.data() + sizeof(TOASTFrameHeader), payload.data(), payload.size());
    }
    
    return data;
}

std::unique_ptr<TOASTFrame> TOASTFrame::deserialize(const uint8_t* data, size_t size) {
    if (size < sizeof(TOASTFrameHeader)) {
        return nullptr;
    }
    
    auto frame = std::make_unique<TOASTFrame>();
    std::memcpy(&frame->header, data, sizeof(TOASTFrameHeader));
    
    // Validate header
    if (frame->header.magic != 0x54534F54 || frame->header.version != 2) {
        return nullptr;
    }
    
    // Extract payload
    size_t payload_size = frame->header.payload_size;
    if (size < sizeof(TOASTFrameHeader) + payload_size) {
        return nullptr;
    }
    
    if (payload_size > 0) {
        frame->payload.resize(payload_size);
        std::memcpy(frame->payload.data(), data + sizeof(TOASTFrameHeader), payload_size);
    }
    
    // Validate checksum
    if (!frame->validate_checksum()) {
        return nullptr;
    }
    
    return frame;
}

// TOASTv2Protocol internal implementation
class TOASTv2Protocol::Impl {
public:
    int socket_fd = -1;
    struct sockaddr_in multicast_addr;
    std::atomic<bool> running{false};
    std::thread receiver_thread;
    std::mutex send_mutex;
    std::unordered_map<uint32_t, std::vector<std::unique_ptr<TOASTFrame>>> burst_cache;
    
    std::random_device rd;
    std::mt19937 rng{rd()};
    
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
        
        // Set socket options for multicast
        int reuse = 1;
        setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));
        
        // Setup multicast address
        memset(&multicast_addr, 0, sizeof(multicast_addr));
        multicast_addr.sin_family = AF_INET;
        multicast_addr.sin_port = htons(port);
        inet_pton(AF_INET, addr.c_str(), &multicast_addr.sin_addr);
        
        // Join multicast group
        struct ip_mreq mreq;
        mreq.imr_multiaddr = multicast_addr.sin_addr;
        mreq.imr_interface.s_addr = INADDR_ANY;
        
        if (setsockopt(socket_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&mreq, sizeof(mreq)) < 0) {
            close(socket_fd);
            return false;
        }
        
        // Bind to receive
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
    
    bool send_packet(const std::vector<uint8_t>& data) {
        std::lock_guard<std::mutex> lock(send_mutex);
        
        ssize_t sent = sendto(socket_fd, (char*)data.data(), data.size(), 0,
                             (struct sockaddr*)&multicast_addr, sizeof(multicast_addr));
        
        return sent == static_cast<ssize_t>(data.size());
    }
    
    void receiver_loop(TOASTv2Protocol* protocol) {
        uint8_t buffer[65536];
        
        while (running) {
            struct sockaddr_in sender_addr;
            socklen_t addr_len = sizeof(sender_addr);
            
            ssize_t received = recvfrom(socket_fd, (char*)buffer, sizeof(buffer), 0,
                                      (struct sockaddr*)&sender_addr, &addr_len);
            
            if (received > 0) {
                auto frame = TOASTFrame::deserialize(buffer, received);
                if (frame) {
                    protocol->handle_received_frame(*frame);
                }
            }
        }
    }
};

// TOASTv2Protocol implementation
TOASTv2Protocol::TOASTv2Protocol() : impl_(std::make_unique<Impl>()) {}

TOASTv2Protocol::~TOASTv2Protocol() {
    shutdown();
}

bool TOASTv2Protocol::initialize(const std::string& multicast_addr, uint16_t port, uint32_t session_id) {
    session_id_ = session_id;
    return impl_->initialize_socket(multicast_addr, port);
}

void TOASTv2Protocol::shutdown() {
    stop_processing();
    impl_->shutdown_socket();
}

bool TOASTv2Protocol::send_frame(const TOASTFrame& frame, bool use_burst) {
    if (use_burst) {
        return send_burst(frame);
    }
    
    auto data = frame.serialize();
    bool success = impl_->send_packet(data);
    
    if (success) {
        update_stats_sent(frame);
    }
    
    return success;
}

bool TOASTv2Protocol::send_midi(const std::vector<uint8_t>& midi_data, uint64_t timestamp_us, bool use_burst) {
    TOASTFrame frame;
    frame.header.frame_type = TOASTFrameType::MIDI;
    frame.header.sequence_number = next_sequence_++;
    frame.header.timestamp_us = static_cast<uint32_t>(timestamp_us);
    frame.header.session_id = session_id_;
    frame.header.payload_size = midi_data.size();
    frame.payload = midi_data;
    
    frame.calculate_checksum();
    
    return send_frame(frame, use_burst);
}

bool TOASTv2Protocol::send_audio(const std::vector<float>& audio_data, uint64_t timestamp_us,
                                uint32_t sample_rate, uint8_t channels) {
    TOASTFrame frame;
    frame.header.frame_type = TOASTFrameType::AUDIO;
    frame.header.sequence_number = next_sequence_++;
    frame.header.timestamp_us = static_cast<uint32_t>(timestamp_us);
    frame.header.session_id = session_id_;
    
    // Convert float audio to bytes (simple conversion)
    frame.payload.resize(audio_data.size() * sizeof(float) + 8); // +8 for metadata
    
    // Add metadata: sample_rate (4 bytes) + channels (1 byte) + reserved (3 bytes)
    *reinterpret_cast<uint32_t*>(frame.payload.data()) = sample_rate;
    frame.payload[4] = channels;
    frame.payload[5] = frame.payload[6] = frame.payload[7] = 0; // reserved
    
    // Copy audio data
    std::memcpy(frame.payload.data() + 8, audio_data.data(), audio_data.size() * sizeof(float));
    frame.header.payload_size = frame.payload.size();
    
    frame.calculate_checksum();
    
    return send_frame(frame, false); // Audio typically doesn't use burst
}

bool TOASTv2Protocol::send_video(const std::vector<uint8_t>& frame_data, uint64_t timestamp_us,
                                uint16_t width, uint16_t height, uint8_t format) {
    TOASTFrame frame;
    frame.header.frame_type = TOASTFrameType::VIDEO;
    frame.header.sequence_number = next_sequence_++;
    frame.header.timestamp_us = static_cast<uint32_t>(timestamp_us);
    frame.header.session_id = session_id_;
    
    // Add video metadata: width (2 bytes) + height (2 bytes) + format (1 byte) + reserved (3 bytes)
    frame.payload.resize(frame_data.size() + 8);
    *reinterpret_cast<uint16_t*>(frame.payload.data()) = width;
    *reinterpret_cast<uint16_t*>(frame.payload.data() + 2) = height;
    frame.payload[4] = format;
    frame.payload[5] = frame.payload[6] = frame.payload[7] = 0; // reserved
    
    // Copy frame data
    std::memcpy(frame.payload.data() + 8, frame_data.data(), frame_data.size());
    frame.header.payload_size = frame.payload.size();
    
    frame.calculate_checksum();
    
    return send_frame(frame, false); // Video typically doesn't use burst
}

bool TOASTv2Protocol::send_sync(uint64_t sync_timestamp) {
    TOASTFrame frame;
    frame.header.frame_type = TOASTFrameType::SYNC;
    frame.header.sequence_number = next_sequence_++;
    frame.header.timestamp_us = static_cast<uint32_t>(sync_timestamp);
    frame.header.session_id = session_id_;
    
    // Sync payload contains the full 64-bit timestamp
    frame.payload.resize(8);
    *reinterpret_cast<uint64_t*>(frame.payload.data()) = sync_timestamp;
    frame.header.payload_size = 8;
    
    frame.calculate_checksum();
    
    return send_frame(frame, true); // Sync uses burst for reliability
}

bool TOASTv2Protocol::start_processing() {
    if (impl_->running) return true;
    
    impl_->running = true;
    impl_->receiver_thread = std::thread(&Impl::receiver_loop, impl_.get(), this);
    
    return true;
}

void TOASTv2Protocol::stop_processing() {
    impl_->running = false;
    if (impl_->receiver_thread.joinable()) {
        impl_->receiver_thread.join();
    }
}

uint32_t TOASTv2Protocol::generate_burst_id() {
    return impl_->rng();
}

bool TOASTv2Protocol::send_burst(const TOASTFrame& frame) {
    uint32_t burst_id = generate_burst_id();
    
    for (uint8_t i = 0; i < burst_config_.burst_size; ++i) {
        TOASTFrame burst_frame = frame;
        burst_frame.header.burst_id = burst_id;
        burst_frame.header.burst_index = i;
        burst_frame.header.burst_total = burst_config_.burst_size;
        
        // Recalculate checksum
        burst_frame.calculate_checksum();
        
        auto data = burst_frame.serialize();
        bool success = impl_->send_packet(data);
        
        if (!success) {
            return false;
        }
        
        // Add jitter between burst packets
        if (i < burst_config_.burst_size - 1) {
            std::uniform_int_distribution<uint16_t> jitter_dist(0, burst_config_.jitter_window_us);
            auto jitter = std::chrono::microseconds(jitter_dist(impl_->rng));
            std::this_thread::sleep_for(jitter);
        }
    }
    
    stats_.burst_packets_sent += burst_config_.burst_size;
    update_stats_sent(frame);
    
    return true;
}

void TOASTv2Protocol::handle_received_frame(const TOASTFrame& frame) {
    // Check if this is our own session
    if (frame.header.session_id == session_id_) {
        return; // Ignore our own messages
    }
    
    // Handle burst deduplication
    if (frame.header.burst_id != 0) {
        auto& cache = impl_->burst_cache[frame.header.burst_id];
        
        // Check if we already have this burst
        for (const auto& cached : cache) {
            if (cached->header.burst_index == frame.header.burst_index) {
                stats_.duplicate_packets_received++;
                return; // Duplicate
            }
        }
        
        // Add to cache
        cache.push_back(std::make_unique<TOASTFrame>(frame));
        
        // Clean up old burst entries (simple cleanup)
        if (impl_->burst_cache.size() > 1000) {
            impl_->burst_cache.clear();
        }
    }
    
    update_stats_received(frame);
    
    // Route to appropriate callback
    switch (frame.header.frame_type) {
        case TOASTFrameType::MIDI:
            if (midi_callback_) midi_callback_(frame);
            break;
        case TOASTFrameType::AUDIO:
            if (audio_callback_) audio_callback_(frame);
            break;
        case TOASTFrameType::VIDEO:
            if (video_callback_) video_callback_(frame);
            break;
        case TOASTFrameType::SYNC:
            if (sync_callback_) sync_callback_(frame);
            break;
        default:
            break;
    }
}

void TOASTv2Protocol::update_stats_sent(const TOASTFrame& frame) {
    stats_.frames_sent++;
    stats_.bytes_sent += sizeof(TOASTFrameHeader) + frame.payload.size();
}

void TOASTv2Protocol::update_stats_received(const TOASTFrame& frame) {
    stats_.frames_received++;
    stats_.bytes_received += sizeof(TOASTFrameHeader) + frame.payload.size();
}

} // namespace jam
