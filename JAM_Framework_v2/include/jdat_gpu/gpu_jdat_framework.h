#pragma once

#include "../gpu_native/gpu_timebase.h"
#include "../gpu_native/gpu_shared_timeline.h"
#include <vector>
#include <atomic>
#include <memory>
#include <cstdint>

namespace jam::jdat {

/**
 * @brief GPU-Native Audio Data Transport Framework
 * 
 * Transforms traditional CPU-based audio buffering to GPU-native scheduling
 * where the GPU timebase controls all audio event timing and processing.
 */
class GPUJDATFramework {
public:
    /**
     * @brief GPU-native audio frame event
     */
    struct GPUAudioEvent {
        gpu_native::EventType type = gpu_native::EventType::AUDIO_FRAME;
        uint64_t gpu_timestamp_ns = 0;      // GPU timebase timestamp
        uint64_t frame_id = 0;              // Sequential frame identifier
        uint32_t channel_mask = 0;          // Bitfield for active channels
        uint32_t sample_rate = 96000;       // Samples per second
        uint32_t frame_size_samples = 0;    // Number of samples in this frame
        uint32_t gpu_buffer_offset = 0;     // Offset in GPU audio buffer
        uint16_t bit_depth = 32;            // Bits per sample (16, 24, 32)
        uint8_t num_channels = 2;           // Number of audio channels
        uint8_t format = 0;                 // Audio format flags
        bool is_realtime = true;            // Real-time priority flag
        bool needs_processing = false;      // Requires GPU audio processing
    };

    /**
     * @brief GPU audio buffer configuration
     */
    struct GPUAudioConfig {
        uint32_t max_concurrent_frames = 64;    // GPU buffer depth
        uint32_t sample_rate = 96000;           // Target sample rate
        uint32_t frame_size_ms = 10;            // Frame duration in milliseconds
        uint32_t max_channels = 16;             // Maximum audio channels
        uint32_t gpu_buffer_size_mb = 8;        // GPU memory allocation
        bool enable_gpu_dsp = true;             // Enable GPU audio processing
        bool enable_realtime_priority = true;   // GPU real-time scheduling
        float max_latency_ms = 5.0f;            // Maximum acceptable latency
    };

    /**
     * @brief GPU audio processing statistics
     */
    struct GPUAudioStats {
        uint64_t frames_scheduled = 0;          // Total frames sent to GPU
        uint64_t frames_processed = 0;          // Frames processed by GPU
        uint64_t gpu_overruns = 0;              // GPU buffer overruns
        uint64_t gpu_underruns = 0;             // GPU buffer underruns
        double avg_gpu_latency_us = 0.0;        // Average GPU processing latency
        double max_gpu_latency_us = 0.0;        // Maximum GPU processing latency
        uint32_t current_gpu_load_percent = 0;  // GPU audio load percentage
        uint64_t last_gpu_timestamp_ns = 0;     // Last GPU timeline update
    };

private:
    GPUAudioConfig config_;
    std::atomic<bool> is_initialized_{false};
    std::atomic<bool> is_running_{false};
    
    // GPU memory management
    void* gpu_audio_buffer_ = nullptr;
    size_t gpu_buffer_size_bytes_ = 0;
    std::atomic<uint32_t> gpu_write_offset_{0};
    std::atomic<uint32_t> gpu_read_offset_{0};
    
    // Event scheduling
    std::atomic<uint64_t> next_frame_id_{1};
    std::atomic<uint64_t> scheduled_events_{0};
    
    // Statistics
    mutable GPUAudioStats stats_;
    std::atomic<uint64_t> stats_update_timestamp_{0};

public:
    /**
     * @brief Constructor
     * @param config GPU audio configuration
     */
    GPUJDATFramework(const GPUAudioConfig& config);

    /**
     * @brief Destructor
     */
    ~GPUJDATFramework();

    // Core GPU-native operations
    bool initialize();
    bool start();
    bool stop();
    void shutdown();

    // GPU audio event scheduling
    bool scheduleAudioFrame(const GPUAudioEvent& event);
    bool scheduleAudioFrameAt(const GPUAudioEvent& event, uint64_t gpu_timestamp_ns);
    bool cancelScheduledFrame(uint64_t frame_id);
    
    // GPU buffer management
    bool allocateGPUAudioBuffer(size_t size_bytes);
    bool writeAudioToGPU(const float* audio_data, size_t sample_count, uint32_t channels);
    bool readAudioFromGPU(float* output_buffer, size_t sample_count, uint32_t channels);
    void flushGPUAudioBuffer();

    // Real-time GPU audio processing
    bool processAudioFrameOnGPU(uint64_t frame_id);
    bool applyGPUAudioEffects(uint64_t frame_id, const std::vector<uint32_t>& effect_ids);
    
    // GPU timeline integration
    uint64_t getCurrentGPUTimestamp() const;
    uint64_t predictNextFrameTimestamp() const;
    bool synchronizeWithGPUTimeline();
    
    // Legacy CPU bridge compatibility
    bool bridgeFromCPUAudioBuffer(const std::vector<float>& cpu_samples, uint32_t channels);
    bool bridgeToCPUAudioBuffer(std::vector<float>& output_samples, uint32_t channels);
    
    // Monitoring and diagnostics
    GPUAudioStats getGPUAudioStatistics() const;
    bool checkGPUAudioHealth() const;
    void resetGPUAudioStatistics();
    
    // Configuration
    const GPUAudioConfig& getConfig() const { return config_; }
    bool updateConfig(const GPUAudioConfig& new_config);
    
    // State queries
    bool isInitialized() const { return is_initialized_.load(); }
    bool isRunning() const { return is_running_.load(); }
    size_t getGPUBufferUtilization() const;
    uint32_t getActiveFrameCount() const;
};

/**
 * @brief GPU-Native JELLIE Audio Encoder
 * 
 * Hardware NATIVE audio encoding using GPU compute shaders
 */
class GPUJELLIEEncoder {
public:
    struct GPUEncodeParams {
        uint32_t target_bitrate_kbps = 128;     // Target encoding bitrate
        uint32_t sample_rate = 96000;           // Input sample rate
        uint8_t num_channels = 2;               // Number of channels
        uint8_t bit_depth = 24;                 // Bits per sample
        bool enable_gpu_acceleration = true;    // Use GPU compute shaders
        bool enable_realtime_encode = true;     // Real-time encoding mode
        float quality_factor = 0.9f;            // Encoding quality (0-1)
    };

private:
    std::shared_ptr<GPUJDATFramework> jdat_framework_;
    GPUEncodeParams params_;
    void* gpu_encode_buffer_ = nullptr;
    std::atomic<bool> is_encoding_{false};

public:
    explicit GPUJELLIEEncoder(std::shared_ptr<GPUJDATFramework> jdat_framework,
                              const GPUEncodeParams& params);
    ~GPUJELLIEEncoder();

    bool initialize();
    bool encodeAudioFrameOnGPU(uint64_t frame_id, std::vector<uint8_t>& output_encoded);
    bool setEncodeParameters(const GPUEncodeParams& new_params);
    void shutdown();
};

/**
 * @brief GPU-Native JELLIE Audio Decoder
 * 
 * Hardware NATIVE audio decoding using GPU compute shaders
 */
class GPUJELLIEDecoder {
public:
    struct GPUDecodeParams {
        uint32_t expected_sample_rate = 96000;  // Expected output sample rate
        uint8_t expected_channels = 2;          // Expected number of channels
        bool enable_gpu_acceleration = true;    // Use GPU compute shaders
        bool enable_realtime_decode = true;     // Real-time decoding mode
        uint32_t max_decode_latency_ms = 5;     // Maximum decode latency
    };

private:
    std::shared_ptr<GPUJDATFramework> jdat_framework_;
    GPUDecodeParams params_;
    void* gpu_decode_buffer_ = nullptr;
    std::atomic<bool> is_decoding_{false};

public:
    explicit GPUJELLIEDecoder(std::shared_ptr<GPUJDATFramework> jdat_framework,
                              const GPUDecodeParams& params);
    ~GPUJELLIEDecoder();

    bool initialize();
    bool decodeAudioFrameOnGPU(const std::vector<uint8_t>& encoded_data, uint64_t frame_id);
    bool setDecodeParameters(const GPUDecodeParams& new_params);
    void shutdown();
};

} // namespace jam::jdat
