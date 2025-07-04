#pragma once

/**
 * JAM Framework v2: GPU Compute Interface
 * 
 * GPU-accelerated message processing pipeline
 * All parsing and processing on GPU compute shaders
 */

#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <string>

namespace jam {

/**
 * GPU Backend types
 */
enum class GPUBackend {
    Metal,      // macOS Metal
    Vulkan,     // Linux Vulkan  
    OpenGL      // Fallback (not recommended)
};

/**
 * GPU Buffer handle for memory-mapped operations
 */
struct GPUBuffer {
    void* gpu_ptr = nullptr;    // GPU memory pointer
    void* cpu_ptr = nullptr;    // CPU memory pointer (if mapped)
    size_t size = 0;            // Buffer size in bytes
    uint32_t buffer_id = 0;     // GPU buffer identifier
    bool is_mapped = false;     // CPU mapping status
};

/**
 * GPU Compute Pipeline for JAMNet processing
 * 
 * PRINCIPLES:
 * - All message processing on GPU
 * - Memory-mapped buffers for zero-copy
 * - Parallel processing of message batches
 * - No CPU-side parsing or processing
 */
class GPUCompute {
public:
    virtual ~GPUCompute() = default;

    /**
     * Create GPU compute pipeline
     * 
     * @param backend GPU backend to use
     * @param shader_path Path to compiled shader directory
     */
    static std::unique_ptr<GPUCompute> create(
        GPUBackend backend,
        const std::string& shader_path
    );

    /**
     * Initialize GPU pipeline
     * 
     * @return true if successful
     */
    virtual bool initialize() = 0;

    /**
     * Shutdown GPU pipeline and release resources
     */
    virtual void shutdown() = 0;

    /**
     * Create GPU buffer for zero-copy operations
     * 
     * @param size Buffer size in bytes
     * @param cpu_accessible Whether CPU needs to access buffer
     * @return GPU buffer handle
     */
    virtual GPUBuffer create_buffer(size_t size, bool cpu_accessible = true) = 0;

    /**
     * Release GPU buffer
     */
    virtual void release_buffer(const GPUBuffer& buffer) = 0;

    /**
     * Map GPU buffer for CPU access (zero-copy)
     * 
     * @param buffer GPU buffer to map
     * @return CPU-accessible pointer
     */
    virtual void* map_buffer(GPUBuffer& buffer) = 0;

    /**
     * Unmap GPU buffer
     */
    virtual void unmap_buffer(GPUBuffer& buffer) = 0;

    /**
     * Process JSONL messages on GPU
     * 
     * Takes raw UDP packet data and processes in parallel:
     * 1. Parse JSONL format
     * 2. Deduplicate burst packets
     * 3. Extract message content
     * 4. Validate checksums
     * 
     * @param input_buffer GPU buffer containing raw packets
     * @param packet_count Number of packets in buffer
     * @param output_buffer GPU buffer for processed messages
     * @return Processing time in microseconds
     */
    virtual uint64_t process_jsonl_batch(
        const GPUBuffer& input_buffer,
        uint32_t packet_count,
        const GPUBuffer& output_buffer
    ) = 0;

    /**
     * Deduplicate burst packets on GPU
     * 
     * Removes duplicate packets from burst transmission
     * 
     * @param input_buffer GPU buffer with potentially duplicate packets
     * @param packet_count Input packet count
     * @param output_buffer GPU buffer for deduplicated packets
     * @param output_count Reference to store output packet count
     * @return Processing time in microseconds
     */
    virtual uint64_t deduplicate_bursts(
        const GPUBuffer& input_buffer,
        uint32_t packet_count,
        const GPUBuffer& output_buffer,
        uint32_t& output_count
    ) = 0;

    /**
     * Route messages by type on GPU
     * 
     * Separates MIDI, audio, video messages into different buffers
     * 
     * @param input_buffer Mixed message buffer
     * @param message_count Input message count
     * @param midi_buffer Output buffer for MIDI messages
     * @param audio_buffer Output buffer for audio messages
     * @param video_buffer Output buffer for video messages
     * @param counts Array to receive count per type [midi, audio, video]
     * @return Processing time in microseconds
     */
    virtual uint64_t route_by_type(
        const GPUBuffer& input_buffer,
        uint32_t message_count,
        const GPUBuffer& midi_buffer,
        const GPUBuffer& audio_buffer,
        const GPUBuffer& video_buffer,
        uint32_t counts[3]
    ) = 0;

    /**
     * Validate message checksums on GPU
     * 
     * @param input_buffer Buffer with messages to validate
     * @param message_count Number of messages
     * @param valid_mask Output buffer for validation results (1 bit per message)
     * @return Processing time in microseconds
     */
    virtual uint64_t validate_checksums(
        const GPUBuffer& input_buffer,
        uint32_t message_count,
        const GPUBuffer& valid_mask
    ) = 0;

    /**
     * Synchronize GPU operations
     * 
     * Ensures all GPU operations complete before returning
     */
    virtual void sync() = 0;

    /**
     * Get GPU performance statistics
     */
    struct GPUStats {
        uint64_t total_operations = 0;
        uint64_t total_process_time_us = 0;
        uint64_t avg_process_time_us = 0;
        size_t gpu_memory_used = 0;
        size_t gpu_memory_total = 0;
        uint32_t active_buffers = 0;
        uint32_t failed_operations = 0;
    };

    virtual GPUStats get_stats() const = 0;

    /**
     * Get GPU backend type
     */
    virtual GPUBackend get_backend() const = 0;

    /**
     * Check if GPU backend is available
     */
    static bool is_backend_available(GPUBackend backend);
};

/**
 * GPU shader types for JAMNet processing
 */
enum class ShaderType {
    JSONLParser,        // Parse JSON Lines format
    BurstDeduplicator,  // Remove duplicate packets
    MessageRouter,      // Route by message type
    ChecksumValidator,  // Validate message integrity
    BufferCopy,         // Memory operations
    AudioProcessor,     // Audio-specific processing
    VideoProcessor,     // Video-specific processing
    MIDIProcessor       // MIDI-specific processing
};

/**
 * Shader compilation and management
 */
class ShaderManager {
public:
    virtual ~ShaderManager() = default;

    /**
     * Load and compile shader
     * 
     * @param type Shader type to load
     * @param source_path Path to shader source file
     * @return true if successful
     */
    virtual bool load_shader(ShaderType type, const std::string& source_path) = 0;

    /**
     * Get compiled shader handle
     */
    virtual uint32_t get_shader_handle(ShaderType type) const = 0;

    /**
     * Check if shader is loaded
     */
    virtual bool is_shader_loaded(ShaderType type) const = 0;

    /**
     * Reload all shaders (for development)
     */
    virtual bool reload_all_shaders() = 0;
};

} // namespace jam
