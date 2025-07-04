#pragma once

/**
 * JAM Framework v2: Core UDP GPU Architecture
 * 
 * Pure UDP, GPU-accelerated, stateless multimedia streaming
 * NO TCP/HTTP DEPENDENCIES - UDP multicast only
 */

#include <cstdint>
#include <memory>
#include <functional>
#include <span>

namespace jam {

// Forward declarations
class UDPTransport;
class GPUProcessor; 
class MemoryMapper;
class MessageRouter;

/**
 * JAMCore - The heart of the UDP revolution
 * 
 * DESIGN PRINCIPLES:
 * - NO TCP/HTTP anywhere
 * - NO connection state
 * - NO acknowledgments or retransmission
 * - GPU-first processing
 * - Fire-and-forget messaging
 */
class JAMCore {
public:
    /**
     * Initialize JAM Framework with pure UDP architecture
     * 
     * @param multicast_group UDP multicast group address (e.g. "239.255.77.77")
     * @param port UDP port for JAMNet communication (e.g. 7777)
     * @param gpu_backend "metal" for macOS, "vulkan" for Linux
     */
    static std::unique_ptr<JAMCore> create(
        const std::string& multicast_group,
        uint16_t port,
        const std::string& gpu_backend
    );

    virtual ~JAMCore() = default;

    /**
     * Send JSONL message via fire-and-forget UDP
     * 
     * NO acknowledgment, NO retransmission, NO waiting
     * Message is immediately sent to multicast group
     * 
     * @param jsonl_message Single line of JSON (no newlines)
     * @param burst_count Number of duplicate packets (1-5 for reliability)
     */
    virtual void send_jsonl(const std::string& jsonl_message, uint8_t burst_count = 1) = 0;

    /**
     * Send raw binary data via fire-and-forget UDP
     * 
     * @param data Binary payload
     * @param format_type Type identifier for receiver processing
     * @param burst_count Number of duplicate packets
     */
    virtual void send_binary(const std::vector<uint8_t>& data, const std::string& format_type, uint8_t burst_count = 1) = 0;

    /**
     * Set message receiver callback
     * 
     * Called for each received message after GPU processing
     * Messages may arrive out of order - design accordingly
     * 
     * @param callback Function to handle received messages
     */
    virtual void set_message_callback(std::function<void(const std::string& jsonl)> callback) = 0;

    /**
     * Set binary data receiver callback
     * 
     * @param callback Function to handle received binary data
     */
    virtual void set_binary_callback(std::function<void(const std::vector<uint8_t>& data, const std::string& format_type)> callback) = 0;

    /**
     * Start UDP multicast listening and GPU processing
     * 
     * Non-blocking - starts background threads and GPU compute pipeline
     */
    virtual void start() = 0;

    /**
     * Stop all processing and release resources
     */
    virtual void stop() = 0;

    /**
     * Get current performance statistics
     */
    struct Statistics {
        uint64_t messages_sent = 0;
        uint64_t messages_received = 0;
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t gpu_process_time_us = 0;  // Average GPU processing time
        uint64_t udp_send_time_us = 0;     // Average UDP send time
        uint32_t packet_loss_percent = 0;   // Estimated packet loss
        uint32_t duplicate_packets = 0;     // Burst duplicates received
    };

    virtual Statistics get_statistics() const = 0;

    /**
     * Force GPU pipeline flush (for testing/debugging)
     */
    virtual void flush_gpu_pipeline() = 0;
};

/**
 * Message structure for internal processing
 * 
 * Self-contained - no external dependencies or state
 */
struct JAMMessage {
    uint64_t timestamp_us;      // Microsecond timestamp
    uint32_t sequence_number;   // For deduplication
    uint16_t burst_id;          // Burst group identifier
    uint8_t burst_index;        // Index within burst (0-4)
    uint8_t format_type;        // JSONL=0, Binary=1, etc.
    std::vector<uint8_t> payload;
    
    // Self-contained validation
    bool is_valid() const;
    uint32_t checksum() const;
};

/**
 * GPU processing interface
 * 
 * All message processing happens on GPU compute shaders
 */
class GPUProcessor {
public:
    virtual ~GPUProcessor() = default;
    
    /**
     * Process batch of messages on GPU
     * 
     * @param messages Input message batch
     * @param deduplicated Output deduplicated messages
     * @return Processing time in microseconds
     */
    virtual uint64_t process_batch(
        const std::vector<JAMMessage>& messages,
        std::vector<JAMMessage>& deduplicated
    ) = 0;
    
    /**
     * Get GPU memory usage statistics
     */
    virtual size_t get_gpu_memory_used() const = 0;
};

/**
 * Memory-mapped buffer interface
 * 
 * Zero-copy data flow from network to GPU
 */
class MemoryMapper {
public:
    virtual ~MemoryMapper() = default;
    
    /**
     * Map network buffer for GPU access
     * 
     * @param buffer Network receive buffer
     * @param size Buffer size in bytes
     * @return GPU-accessible memory handle
     */
    virtual void* map_for_gpu(void* buffer, size_t size) = 0;
    
    /**
     * Unmap GPU buffer
     */
    virtual void unmap_gpu_buffer(void* gpu_handle) = 0;
    
    /**
     * Synchronize GPU writes back to CPU
     */
    virtual void sync_to_cpu(void* gpu_handle) = 0;
};

} // namespace jam
