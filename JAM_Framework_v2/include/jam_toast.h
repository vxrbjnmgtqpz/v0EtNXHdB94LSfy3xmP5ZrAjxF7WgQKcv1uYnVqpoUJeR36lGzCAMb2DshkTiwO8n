#pragma once

/**
 * JAM Framework v2: TOAST v2 Protocol
 * 
 * Transport Oriented Audio Sync Tunnel v2
 * Pure UDP implementation with GPU NATIVE processing
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace jam {

/**
 * TOAST v2 Frame Types
 */
enum class TOASTFrameType : uint8_t {
    MIDI = 0x01,        // MIDI event data
    AUDIO = 0x02,       // PCM audio data  
    VIDEO = 0x03,       // Video frame data
    SYNC = 0x04,        // Timing synchronization
    TRANSPORT = 0x05,   // Transport commands (play/stop/position/bpm)
    DISCOVERY = 0x06,   // Peer discovery  
    HEARTBEAT = 0x07,   // Keep-alive
    BURST_HEADER = 0x08 // Burst transmission header
};

/**
 * TOAST v2 Frame Header (32 bytes fixed)
 */
struct TOASTFrameHeader {
    uint32_t magic = 0x54534F54;        // "TOST" magic number (4 bytes)
    uint8_t version = 2;                // TOAST version 2 (1 byte)
    uint8_t frame_type;                 // Frame type (1 byte, changed from enum)
    uint16_t flags = 0;                 // Control flags (2 bytes)
    uint32_t sequence_number = 0;       // Sequence number (4 bytes)
    uint32_t timestamp_us = 0;          // Microsecond timestamp (4 bytes)
    uint32_t payload_size = 0;          // Payload size in bytes (4 bytes)
    uint32_t burst_id = 0;              // Burst identifier (4 bytes)
    uint8_t burst_index = 0;            // Index in burst (1 byte)
    uint8_t burst_total = 1;            // Total packets in burst (1 byte)
    uint16_t checksum = 0;              // Header + payload checksum (2 bytes)
    uint32_t session_id = 0;            // Session identifier (4 bytes)
    // Total: 32 bytes exactly
} __attribute__((packed));

static_assert(sizeof(TOASTFrameHeader) == 32, "TOASTFrameHeader must be 32 bytes");

/**
 * TOAST v2 Frame
 */
struct TOASTFrame {
    TOASTFrameHeader header;
    std::vector<uint8_t> payload;
    
    // Calculate and set checksum
    void calculate_checksum();
    
    // Validate checksum
    bool validate_checksum() const;
    
    // Serialize to bytes
    std::vector<uint8_t> serialize() const;
    
    // Deserialize from bytes
    static std::unique_ptr<TOASTFrame> deserialize(const uint8_t* data, size_t size);
};

/**
 * Burst Transmission Configuration
 */
struct BurstConfig {
    uint8_t burst_size = 3;             // Number of packets per burst
    uint16_t jitter_window_us = 500;    // Jitter window in microseconds
    bool enable_redundancy = true;      // Enable burst redundancy
    uint8_t max_retries = 0;           // Retries (0 = fire-and-forget)
};

/**
 * TOAST v2 Protocol Handler
 * 
 * Pure UDP implementation with GPU NATIVE processing support
 */
class TOASTv2Protocol {
public:
    // Frame callback function types
    using FrameCallback = std::function<void(const TOASTFrame& frame)>;
    using ErrorCallback = std::function<void(const std::string& error)>;
    
    TOASTv2Protocol();
    ~TOASTv2Protocol();
    
    /**
     * Initialize TOAST v2 protocol
     * 
     * @param multicast_addr Multicast address (e.g., "239.255.77.77")
     * @param port UDP port number
     * @param session_id Session identifier
     * @return true if successful
     */
    bool initialize(const std::string& multicast_addr, uint16_t port, uint32_t session_id);
    
    /**
     * Shutdown protocol
     */
    void shutdown();
    
    /**
     * Send TOAST frame
     * 
     * @param frame Frame to send
     * @param use_burst Whether to use burst transmission
     * @return true if successful
     */
    bool send_frame(const TOASTFrame& frame, bool use_burst = false);
    
    /**
     * Send MIDI event
     * 
     * @param midi_data MIDI event data
     * @param timestamp_us Timestamp in microseconds
     * @param use_burst Whether to use burst transmission
     * @return true if successful
     */
    bool send_midi(const std::vector<uint8_t>& midi_data, uint64_t timestamp_us, bool use_burst = true);
    
    /**
     * Send audio chunk
     * 
     * @param audio_data PCM audio data
     * @param timestamp_us Timestamp in microseconds
     * @param sample_rate Sample rate
     * @param channels Number of channels
     * @return true if successful
     */
    bool send_audio(const std::vector<float>& audio_data, uint64_t timestamp_us, 
                   uint32_t sample_rate, uint8_t channels);
    
    /**
     * Send video frame
     * 
     * @param frame_data Video frame data
     * @param timestamp_us Timestamp in microseconds
     * @param width Frame width
     * @param height Frame height
     * @param format Pixel format
     * @return true if successful
     */
    bool send_video(const std::vector<uint8_t>& frame_data, uint64_t timestamp_us,
                   uint16_t width, uint16_t height, uint8_t format);
    
    /**
     * Send synchronization frame
     * 
     * @param sync_timestamp Master synchronization timestamp
     * @return true if successful
     */
    bool send_sync(uint64_t sync_timestamp);
    
    /**
     * Send discovery message to announce presence
     * 
     * @return true if successful
     */
    bool send_discovery();
    
    /**
     * Send heartbeat to maintain presence
     * 
     * @return true if successful  
     */
    bool send_heartbeat();
    
    /**
     * Set frame callbacks
     */
    void set_midi_callback(FrameCallback callback) { midi_callback_ = std::move(callback); }
    void set_audio_callback(FrameCallback callback) { audio_callback_ = std::move(callback); }
    void set_video_callback(FrameCallback callback) { video_callback_ = std::move(callback); }
    void set_sync_callback(FrameCallback callback) { sync_callback_ = std::move(callback); }
    void set_discovery_callback(FrameCallback callback) { discovery_callback_ = std::move(callback); }
    void set_heartbeat_callback(FrameCallback callback) { heartbeat_callback_ = std::move(callback); }
    void set_error_callback(ErrorCallback callback) { error_callback_ = std::move(callback); }
    
    /**
     * Configure burst transmission
     */
    void set_burst_config(const BurstConfig& config) { burst_config_ = config; }
    
    /**
     * Start/stop processing
     */
    bool start_processing();
    void stop_processing();
    
    /**
     * Get protocol statistics
     */
    struct TOASTStats {
        uint64_t frames_sent = 0;
        uint64_t frames_received = 0;
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t burst_packets_sent = 0;
        uint64_t duplicate_packets_received = 0;
        uint64_t invalid_packets = 0;
        uint64_t checksum_errors = 0;
        uint32_t active_peers = 0;
        double average_latency_us = 0.0;
    };
    
    TOASTStats get_stats() const { return stats_; }
    
    /**
     * Get session ID
     */
    uint32_t get_session_id() const { return session_id_; }

private:
    // Internal implementation
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // Configuration
    uint32_t session_id_ = 0;
    uint32_t next_sequence_ = 1;
    BurstConfig burst_config_;
    
    // Callbacks
    FrameCallback midi_callback_;
    FrameCallback audio_callback_;
    FrameCallback video_callback_;
    FrameCallback sync_callback_;
    FrameCallback discovery_callback_;
    FrameCallback heartbeat_callback_;
    ErrorCallback error_callback_;
    
    // Statistics
    mutable TOASTStats stats_;
    
    // Helper methods
    uint32_t generate_burst_id();
    bool send_burst(const TOASTFrame& frame);
    void handle_received_frame(const TOASTFrame& frame);
    void update_stats_sent(const TOASTFrame& frame);
    void update_stats_received(const TOASTFrame& frame);
};

/**
 * TOAST v2 Multicast Discovery
 */
class TOASTDiscovery {
public:
    struct PeerInfo {
        std::string address;
        uint16_t port;
        uint32_t session_id;
        std::string name;
        uint64_t last_seen_us;
    };
    
    using PeerCallback = std::function<void(const PeerInfo& peer)>;
    
    TOASTDiscovery();
    ~TOASTDiscovery();
    
    /**
     * Start peer discovery
     * 
     * @param multicast_addr Discovery multicast address
     * @param port Discovery port
     * @param local_name Local peer name
     * @return true if successful
     */
    bool start_discovery(const std::string& multicast_addr, uint16_t port, 
                        const std::string& local_name);
    
    /**
     * Stop discovery
     */
    void stop_discovery();
    
    /**
     * Set peer discovery callback
     */
    void set_peer_callback(PeerCallback callback) { peer_callback_ = std::move(callback); }
    
    /**
     * Get discovered peers
     */
    std::vector<PeerInfo> get_peers() const;
    
    /**
     * Manually announce presence
     */
    void announce();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    PeerCallback peer_callback_;
    
    // Internal method for adding discovered peers
    void add_peer(const PeerInfo& peer);
};

} // namespace jam
