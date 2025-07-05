#pragma once

/**
 * JAM Framework v2: UDP Transport Layer
 * 
 * Pure UDP multicast - NO TCP/HTTP anywhere
 * Fire-and-forget, stateless messaging
 */

#include <cstdint>
#include <string>
#include <functional>
#include <span>
#include <memory>

namespace jam {

/**
 * UDPTransport - Pure UDP multicast implementation
 * 
 * PRINCIPLES:
 * - Fire-and-forget sending (no acknowledgments)
 * - Multicast for 1-to-many efficiency
 * - No connection state or session management
 * - Immediate sending (no queuing or buffering)
 */
class UDPTransport {
public:
    virtual ~UDPTransport() = default;

    /**
     * Create UDP multicast transport
     * 
     * @param multicast_group Multicast IP (239.0.0.0 - 239.255.255.255)
     * @param port UDP port number
     * @param interface_ip Local interface IP (empty for default)
     */
    static std::unique_ptr<UDPTransport> create(
        const std::string& multicast_group,
        uint16_t port,
        const std::string& interface_ip = ""
    );

    /**
     * Join multicast group for receiving
     * 
     * @return true if successful
     */
    virtual bool join_multicast() = 0;

    /**
     * Leave multicast group
     */
    virtual void leave_multicast() = 0;

    /**
     * Send data immediately via UDP multicast
     * 
     * Fire-and-forget - no confirmation or retransmission
     * 
     * @param data Payload to send
     * @param size Payload size in bytes
     * @return Microseconds to send, or 0 if failed
     */
    virtual uint64_t send_immediate(const void* data, size_t size) = 0;

    /**
     * Send burst of duplicate packets
     * 
     * For reliability without retransmission
     * 
     * @param data Payload to send
     * @param size Payload size
     * @param burst_count Number of duplicates (1-5)
     * @param burst_interval_us Microseconds between duplicates
     * @return Total time for burst in microseconds
     */
    virtual uint64_t send_burst(
        const void* data, 
        size_t size, 
        uint8_t burst_count,
        uint16_t burst_interval_us = 50
    ) = 0;

    /**
     * Set receive callback for incoming UDP packets
     * 
     * Called immediately upon packet arrival
     * May be called from multiple threads concurrently
     * 
     * @param callback Function to handle received data
     */
    virtual void set_receive_callback(std::function<void(const uint8_t* data, size_t size)> callback) = 0;

    /**
     * Start UDP receive processing
     * 
     * Non-blocking - starts background receive thread
     */
    virtual bool start_receiving() = 0;

    /**
     * Stop UDP receive processing
     */
    virtual void stop_receiving() = 0;

    /**
     * Get transport statistics
     */
    struct TransportStats {
        uint64_t packets_sent = 0;
        uint64_t packets_received = 0;
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t send_failures = 0;
        uint64_t avg_send_time_us = 0;
        uint32_t estimated_packet_loss_percent = 0;
    };

    virtual TransportStats get_stats() const = 0;

    /**
     * Get multicast group info
     */
    virtual std::string get_multicast_group() const = 0;
    virtual uint16_t get_port() const = 0;
};

/**
 * UDP packet structure for JAMNet
 * 
 * Fixed header for all UDP packets
 */
struct UDPPacketHeader {
    uint32_t magic;             // 0x4A414D32 ("JAM2")
    uint16_t version;           // Protocol version (0x0001)
    uint16_t payload_size;      // Size of payload following header
    uint64_t timestamp_us;      // Microsecond timestamp
    uint32_t sequence_number;   // For deduplication and ordering
    uint16_t burst_id;          // Burst group identifier
    uint8_t burst_index;        // Index within burst (0-4)
    uint8_t packet_type;        // JSONL=0, Binary=1, Control=2
    uint32_t checksum;          // CRC32 of header + payload
    uint8_t reserved[4];        // Future use, must be zero
} __attribute__((packed));

static_assert(sizeof(UDPPacketHeader) == 32, "Header must be exactly 32 bytes");

/**
 * Packet type constants
 */
enum class PacketType : uint8_t {
    JSONL = 0,          // JSON Lines message
    Binary = 1,         // Binary data
    Control = 2,        // Control messages (discovery, etc.)
    Heartbeat = 3       // Keep-alive packets
};

/**
 * Control message types
 */
enum class ControlType : uint8_t {
    Discovery = 0,      // Peer discovery
    Join = 1,           // Join session
    Leave = 2,          // Leave session
    Sync = 3            // Time synchronization
};

/**
 * Utility functions for UDP packet handling
 */
namespace udp_utils {
    /**
     * Calculate CRC32 checksum
     */
    uint32_t calculate_checksum(const void* data, size_t size);

    /**
     * Validate packet header
     */
    bool validate_header(const UDPPacketHeader& header, size_t total_packet_size);

    /**
     * Create packet header
     */
    UDPPacketHeader create_header(
        PacketType type,
        uint16_t payload_size,
        uint32_t sequence_number,
        uint16_t burst_id = 0,
        uint8_t burst_index = 0
    );

    /**
     * Get current timestamp in microseconds
     */
    uint64_t get_timestamp_us();

    /**
     * Generate next sequence number (thread-safe)
     */
    uint32_t next_sequence_number();
}

} // namespace jam
