#pragma once

#include "../gpu_native/gpu_timebase.h"
#include "../gpu_native/gpu_shared_timeline.h"
#include <vector>
#include <atomic>
#include <memory>
#include <cstdint>

namespace jam::jvid {

/**
 * @brief GPU-Native Video Data Transport Framework
 * 
 * Transforms traditional CPU-based video buffering to GPU-native scheduling
 * where the GPU timebase controls all video frame timing and processing.
 */
class GPUJVIDFramework {
public:
    /**
     * @brief GPU-native video frame event
     */
    struct GPUVideoEvent {
        gpu_native::EventType type = gpu_native::EventType::VIDEO_FRAME;
        uint64_t gpu_timestamp_ns = 0;      // GPU timebase timestamp
        uint64_t frame_id = 0;              // Sequential frame identifier
        uint32_t width = 640;               // Frame width in pixels
        uint32_t height = 360;              // Frame height in pixels
        uint32_t stride = 0;                // Row stride in bytes
        uint32_t gpu_texture_id = 0;        // GPU texture/buffer identifier
        uint8_t channels = 3;               // Color channels (RGB=3, RGBA=4)
        uint8_t bit_depth = 8;              // Bits per channel (8, 10, 12, 16)
        uint8_t color_space = 0;            // Color space identifier
        uint8_t codec_hint = 0;             // Codec preference hint
        bool is_keyframe = false;           // Keyframe flag
        bool needs_gpu_processing = false;  // Requires GPU video processing
        bool is_realtime = true;            // Real-time priority flag
        float target_fps = 30.0f;           // Target frame rate
    };

    /**
     * @brief GPU video configuration
     */
    struct GPUVideoConfig {
        uint32_t max_concurrent_frames = 32;    // GPU buffer depth
        uint32_t max_width = 1920;              // Maximum frame width
        uint32_t max_height = 1080;             // Maximum frame height
        uint32_t max_fps = 60;                  // Maximum frame rate
        uint32_t gpu_texture_pool_size = 16;    // Number of GPU textures
        uint32_t gpu_buffer_size_mb = 64;       // GPU memory allocation
        bool enable_gpu_processing = true;      // Enable GPU video processing
        bool enable_realtime_priority = true;   // GPU real-time scheduling
        bool enable_zero_copy = true;           // Zero-copy GPU operations
        float max_frame_latency_ms = 16.67f;    // Maximum frame latency (60fps = 16.67ms)
        uint8_t preferred_color_depth = 8;      // Preferred bit depth
    };

    /**
     * @brief GPU video processing statistics
     */
    struct GPUVideoStats {
        uint64_t frames_scheduled = 0;          // Total frames sent to GPU
        uint64_t frames_processed = 0;          // Frames processed by GPU
        uint64_t gpu_texture_overruns = 0;      // GPU texture pool overruns
        uint64_t gpu_frame_drops = 0;           // Dropped frames due to timing
        double avg_gpu_frame_time_us = 0.0;     // Average GPU frame processing time
        double max_gpu_frame_time_us = 0.0;     // Maximum GPU frame processing time
        uint32_t current_gpu_load_percent = 0;  // GPU video load percentage
        uint64_t last_gpu_timestamp_ns = 0;     // Last GPU timeline update
        uint32_t active_texture_count = 0;      // Currently active GPU textures
        double current_fps = 0.0;               // Current actual frame rate
    };

private:
    GPUVideoConfig config_;
    std::atomic<bool> is_initialized_{false};
    std::atomic<bool> is_running_{false};
    
    // GPU texture management
    struct GPUTexture {
        void* gpu_texture_handle = nullptr;
        uint32_t texture_id = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint8_t channels = 0;
        bool is_available = true;
        uint64_t last_used_timestamp = 0;
    };
    
    std::vector<GPUTexture> gpu_texture_pool_;
    std::atomic<uint32_t> next_texture_id_{1};
    std::atomic<uint32_t> available_textures_{0};
    
    // Frame scheduling
    std::atomic<uint64_t> next_frame_id_{1};
    std::atomic<uint64_t> scheduled_frames_{0};
    std::atomic<double> current_fps_{0.0};
    
    // Statistics
    mutable GPUVideoStats stats_;
    std::atomic<uint64_t> stats_update_timestamp_{0};
    std::atomic<uint64_t> last_frame_timestamp_{0};

public:
    /**
     * @brief Constructor
     * @param config GPU video configuration
     */
    GPUJVIDFramework(const GPUVideoConfig& config);

    /**
     * @brief Destructor
     */
    ~GPUJVIDFramework();

    // Core GPU-native operations
    bool initialize();
    bool start();
    bool stop();
    void shutdown();

    // GPU video frame scheduling
    bool scheduleVideoFrame(const GPUVideoEvent& event);
    bool scheduleVideoFrameAt(const GPUVideoEvent& event, uint64_t gpu_timestamp_ns);
    bool cancelScheduledFrame(uint64_t frame_id);
    
    // GPU texture management
    uint32_t allocateGPUTexture(uint32_t width, uint32_t height, uint8_t channels);
    bool releaseGPUTexture(uint32_t texture_id);
    bool uploadFrameToGPUTexture(uint32_t texture_id, const uint8_t* frame_data, size_t data_size);
    bool downloadFrameFromGPUTexture(uint32_t texture_id, uint8_t* output_buffer, size_t buffer_size);
    void flushGPUTextures();

    // Real-time GPU video processing
    bool processVideoFrameOnGPU(uint64_t frame_id);
    bool applyGPUVideoEffects(uint64_t frame_id, const std::vector<uint32_t>& effect_ids);
    bool resizeFrameOnGPU(uint32_t source_texture_id, uint32_t target_texture_id, 
                          uint32_t new_width, uint32_t new_height);
    
    // GPU timeline integration
    uint64_t getCurrentGPUTimestamp() const;
    uint64_t predictNextFrameTimestamp() const;
    bool synchronizeWithGPUTimeline();
    
    // Frame rate control
    bool setTargetFrameRate(float fps);
    float getCurrentFrameRate() const;
    bool enableAdaptiveFrameRate(bool enable);
    
    // Legacy CPU bridge compatibility
    bool bridgeFromCPUFrameBuffer(const std::vector<uint8_t>& cpu_frame_data, 
                                  uint32_t width, uint32_t height, uint8_t channels);
    bool bridgeToCPUFrameBuffer(std::vector<uint8_t>& output_frame_data, 
                                uint32_t& width, uint32_t& height, uint8_t& channels);
    
    // Monitoring and diagnostics
    GPUVideoStats getGPUVideoStatistics() const;
    bool checkGPUVideoHealth() const;
    void resetGPUVideoStatistics();
    
    // Configuration
    const GPUVideoConfig& getConfig() const { return config_; }
    bool updateConfig(const GPUVideoConfig& new_config);
    
    // State queries
    bool isInitialized() const { return is_initialized_.load(); }
    bool isRunning() const { return is_running_.load(); }
    uint32_t getGPUTextureUtilization() const;
    uint32_t getActiveFrameCount() const;
};

/**
 * @brief GPU-Native JAMCam Video Encoder
 * 
 * Hardware NATIVE video encoding using GPU compute shaders
 */
class GPUJAMCamEncoder {
public:
    struct GPUEncodeParams {
        uint32_t target_bitrate_kbps = 2000;    // Target encoding bitrate
        uint32_t width = 640;                   // Video width
        uint32_t height = 360;                  // Video height
        float target_fps = 30.0f;               // Target frame rate
        uint8_t quality_preset = 2;             // Quality preset (0=fastest, 4=best)
        bool enable_gpu_acceleration = true;    // Use GPU compute shaders
        bool enable_realtime_encode = true;     // Real-time encoding mode
        bool enable_keyframe_insertion = true;  // Automatic keyframe insertion
        uint32_t keyframe_interval = 30;        // Keyframes every N frames
    };

private:
    std::shared_ptr<GPUJVIDFramework> jvid_framework_;
    GPUEncodeParams params_;
    void* gpu_encode_context_ = nullptr;
    std::atomic<bool> is_encoding_{false};

public:
    explicit GPUJAMCamEncoder(std::shared_ptr<GPUJVIDFramework> jvid_framework,
                              const GPUEncodeParams& params);
    ~GPUJAMCamEncoder();

    bool initialize();
    bool encodeVideoFrameOnGPU(uint32_t source_texture_id, uint64_t frame_id, 
                               std::vector<uint8_t>& output_encoded);
    bool insertKeyframeOnGPU(uint32_t source_texture_id, uint64_t frame_id);
    bool setEncodeParameters(const GPUEncodeParams& new_params);
    void shutdown();
};

/**
 * @brief GPU-Native JAMCam Video Decoder
 * 
 * Hardware NATIVE video decoding using GPU compute shaders
 */
class GPUJAMCamDecoder {
public:
    struct GPUDecodeParams {
        uint32_t expected_width = 640;          // Expected video width
        uint32_t expected_height = 360;         // Expected video height
        float expected_fps = 30.0f;             // Expected frame rate
        bool enable_gpu_acceleration = true;    // Use GPU compute shaders
        bool enable_realtime_decode = true;     // Real-time decoding mode
        bool enable_frame_interpolation = false; // GPU frame interpolation
        uint32_t max_decode_latency_ms = 33;    // Maximum decode latency
    };

private:
    std::shared_ptr<GPUJVIDFramework> jvid_framework_;
    GPUDecodeParams params_;
    void* gpu_decode_context_ = nullptr;
    std::atomic<bool> is_decoding_{false};

public:
    explicit GPUJAMCamDecoder(std::shared_ptr<GPUJVIDFramework> jvid_framework,
                              const GPUDecodeParams& params);
    ~GPUJAMCamDecoder();

    bool initialize();
    bool decodeVideoFrameOnGPU(const std::vector<uint8_t>& encoded_data, 
                               uint32_t target_texture_id, uint64_t frame_id);
    bool setDecodeParameters(const GPUDecodeParams& new_params);
    void shutdown();
};

/**
 * @brief GPU-Native Frame Predictor
 * 
 * AI-powered frame prediction and interpolation using GPU neural networks
 */
class GPUFramePredictor {
public:
    struct GPUPredictorConfig {
        bool enable_motion_prediction = true;   // Enable motion vector prediction
        bool enable_frame_interpolation = true; // Enable frame interpolation
        bool enable_ai_upscaling = false;       // Enable AI-based upscaling
        uint8_t prediction_quality = 2;         // Prediction quality (0-4)
        uint32_t max_prediction_frames = 4;     // Maximum frames to predict ahead
    };

private:
    std::shared_ptr<GPUJVIDFramework> jvid_framework_;
    GPUPredictorConfig config_;
    void* gpu_ai_context_ = nullptr;
    std::atomic<bool> is_predicting_{false};

public:
    explicit GPUFramePredictor(std::shared_ptr<GPUJVIDFramework> jvid_framework,
                               const GPUPredictorConfig& config);
    ~GPUFramePredictor();

    bool initialize();
    bool predictNextFrameOnGPU(uint32_t reference_texture_id, uint32_t output_texture_id);
    bool interpolateFramesOnGPU(uint32_t frame1_texture_id, uint32_t frame2_texture_id,
                                uint32_t output_texture_id, float interpolation_factor);
    bool setConfig(const GPUPredictorConfig& new_config);
    void shutdown();
};

} // namespace jam::jvid
