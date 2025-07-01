#pragma once

#include "JSONVIDMessage.h"
#include "VideoBufferManager.h"
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>

namespace jsonvid {

/**
 * @brief JAMCam video encoder for ultra-low latency video streaming
 * 
 * Captures video frames from camera/screen and encodes them into JSONVID messages
 * with target latencies <300μs for video transmission over TOAST transport.
 */
class JAMCamEncoder {
public:
    /**
     * @brief Encoder configuration
     */
    struct Config {
        VideoResolution resolution = VideoResolution::LOW_144P;
        VideoQuality quality = VideoQuality::FAST;
        FrameFormat format = FrameFormat::BASE64_JPEG;
        uint32_t target_fps = 15;            // Target framerate
        uint32_t target_latency_us = 300;    // Target encode latency (300μs)
        bool enable_face_detection = false;  // JAMCam face detection
        bool enable_auto_framing = false;    // Automatic face framing
        bool enable_lighting_norm = false;   // Lighting normalization
        uint32_t keyframe_interval = 30;     // I-frame every N frames
        uint8_t stream_id = 0;               // Stream identifier for multi-cam
        
        // Ultra-low latency optimizations
        bool enable_gpu_encoding = true;     // Use GPU acceleration if available
        bool enable_zero_copy = true;        // Zero-copy frame processing
        bool enable_frame_dropping = true;   // Drop frames under heavy load
        uint32_t max_encode_time_us = 500;   // Drop frames taking longer than this
        
        // Quality vs latency tuning
        uint32_t jpeg_quality = 60;          // JPEG quality 0-100
        bool enable_chroma_subsampling = true; // 4:2:0 chroma subsampling
        bool enable_motion_estimation = false; // Motion vectors (higher latency)
        
        // Buffer management
        uint32_t frame_buffer_size = 3;      // Number of frames to buffer
        bool adaptive_quality = true;        // Adjust quality based on latency
    };
    
    /**
     * @brief Video frame callback signature
     * Called when encoded frame is ready for transmission
     */
    using FrameCallback = std::function<void(const JSONVIDMessage&)>;
    
    /**
     * @brief Frame drop callback signature  
     * Called when frame is dropped due to latency constraints
     */
    using FrameDropCallback = std::function<void(uint64_t dropped_frame_seq, uint32_t encode_time_us)>;
    
    /**
     * @brief Encoding statistics callback
     * Called periodically with performance metrics
     */
    using StatsCallback = std::function<void(const VideoStreamStats&)>;

private:
    Config config_;
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_capturing_{false};
    
    // Frame capture and processing
    std::unique_ptr<VideoBufferManager> buffer_manager_;
    std::thread capture_thread_;
    std::thread encode_thread_;
    
    // Callbacks
    FrameCallback frame_callback_;
    FrameDropCallback drop_callback_;
    StatsCallback stats_callback_;
    
    // Timing and sequence management
    std::atomic<uint64_t> frame_sequence_{0};
    std::atomic<uint64_t> frames_encoded_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    uint64_t start_time_us_;
    
    // Video processing state
    struct CaptureState {
        uint32_t actual_width = 0;
        uint32_t actual_height = 0;
        uint32_t capture_fps = 0;
        bool camera_available = false;
        std::string device_id;
        
        // Face detection state
        bool face_detected = false;
        uint16_t face_x = 0, face_y = 0;
        uint16_t face_width = 0, face_height = 0;
        
        // Auto-framing state
        uint16_t crop_x = 0, crop_y = 0;
        uint16_t crop_width = 0, crop_height = 0;
        
        // Lighting state
        float current_brightness = 1.0f;
        float current_contrast = 1.0f;
        float current_gamma = 1.0f;
    };
    
    CaptureState capture_state_;
    mutable std::mutex state_mutex_;

public:
    /**
     * @brief Constructor
     * @param config Encoder configuration
     */
    explicit JAMCamEncoder(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~JAMCamEncoder();
    
    /**
     * @brief Set frame output callback
     * @param callback Function to call when frame is encoded
     */
    void setFrameCallback(FrameCallback callback);
    
    /**
     * @brief Set frame drop callback
     * @param callback Function to call when frame is dropped
     */
    void setFrameDropCallback(FrameDropCallback callback);
    
    /**
     * @brief Set statistics callback
     * @param callback Function to call with performance stats
     */
    void setStatsCallback(StatsCallback callback);
    
    /**
     * @brief Start video capture and encoding
     * @return True if started successfully
     */
    bool start();
    
    /**
     * @brief Stop video capture and encoding
     */
    void stop();
    
    /**
     * @brief Check if encoder is running
     * @return True if running
     */
    bool isRunning() const { return is_running_.load(); }
    
    /**
     * @brief Update encoder configuration
     * @param config New configuration
     * @return True if updated successfully
     */
    bool updateConfig(const Config& config);
    
    /**
     * @brief Get current configuration
     * @return Current encoder configuration
     */
    const Config& getConfig() const { return config_; }
    
    /**
     * @brief Manually encode a frame (for testing/external capture)
     * @param frame_data Raw frame data (RGB/YUV)
     * @param width Frame width
     * @param height Frame height
     * @param timestamp_us Capture timestamp
     * @return Encoded message or nullptr if encoding failed
     */
    std::unique_ptr<JSONVIDMessage> encodeFrame(
        const std::vector<uint8_t>& frame_data,
        uint32_t width,
        uint32_t height,
        uint64_t timestamp_us
    );
    
    /**
     * @brief Force generation of keyframe
     */
    void forceKeyframe();
    
    /**
     * @brief Get encoding statistics
     */
    struct Statistics {
        uint64_t frames_captured = 0;
        uint64_t frames_encoded = 0;
        uint64_t frames_dropped = 0;
        uint64_t keyframes_generated = 0;
        
        double average_encode_time_us = 0.0;
        double max_encode_time_us = 0.0;
        double average_frame_size_kb = 0.0;
        double current_fps = 0.0;
        double target_fps = 15.0;
        
        double cpu_usage_percent = 0.0;
        double memory_usage_mb = 0.0;
        
        // Quality metrics
        double average_compression_ratio = 0.0;
        uint32_t current_bitrate_kbps = 0;
        double quality_score = 0.0;          // 0-1 subjective quality
        
        // Timing analysis
        double capture_jitter_us = 0.0;
        double encode_jitter_us = 0.0;
        uint32_t missed_frame_deadlines = 0;
        
        // JAMCam features
        uint32_t faces_detected = 0;
        bool auto_framing_active = false;
        bool lighting_normalized = false;
    };
    
    /**
     * @brief Get current statistics
     * @return Current encoding statistics
     */
    Statistics getStatistics() const;
    
    /**
     * @brief Reset statistics counters
     */
    void resetStatistics();
    
    /**
     * @brief Get available camera devices
     * @return List of camera device IDs and names
     */
    static std::vector<std::pair<std::string, std::string>> getAvailableCameras();
    
    /**
     * @brief Get optimal encoder settings for target latency
     * @param target_latency_us Desired latency in microseconds
     * @return Recommended configuration
     */
    static Config getOptimalConfig(uint32_t target_latency_us);
    
    /**
     * @brief Test camera capture capabilities
     * @param device_id Camera device identifier
     * @return True if camera supports required features
     */
    static bool testCamera(const std::string& device_id);

private:
    /**
     * @brief Main capture thread function
     */
    void captureThreadFunction();
    
    /**
     * @brief Main encoding thread function
     */
    void encodeThreadFunction();
    
    /**
     * @brief Initialize camera capture
     * @return True if camera initialized successfully
     */
    bool initializeCapture();
    
    /**
     * @brief Capture single frame from camera
     * @param frame_data Output buffer for frame data
     * @param timestamp_us Capture timestamp
     * @return True if frame captured successfully
     */
    bool captureFrame(std::vector<uint8_t>& frame_data, uint64_t& timestamp_us);
    
    /**
     * @brief Apply JAMCam processing (face detection, auto-framing, lighting)
     * @param frame_data Frame data to process (in-place)
     * @param width Frame width
     * @param height Frame height
     */
    void applyJAMCamProcessing(std::vector<uint8_t>& frame_data, uint32_t width, uint32_t height);
    
    /**
     * @brief Detect faces in frame
     * @param frame_data Frame data
     * @param width Frame width  
     * @param height Frame height
     * @return Vector of detected face bounding boxes
     */
    std::vector<JSONVIDMessage::JAMCamFeatures::FaceBBox> detectFaces(
        const std::vector<uint8_t>& frame_data,
        uint32_t width,
        uint32_t height
    );
    
    /**
     * @brief Apply automatic framing based on detected faces
     * @param frame_data Frame data to crop (in-place)
     * @param width Frame width (updated)
     * @param height Frame height (updated)
     * @param faces Detected faces
     */
    void applyAutoFraming(
        std::vector<uint8_t>& frame_data,
        uint32_t& width,
        uint32_t& height,
        const std::vector<JSONVIDMessage::JAMCamFeatures::FaceBBox>& faces
    );
    
    /**
     * @brief Apply lighting normalization
     * @param frame_data Frame data to adjust (in-place)
     * @param width Frame width
     * @param height Frame height
     */
    void applyLightingNormalization(std::vector<uint8_t>& frame_data, uint32_t width, uint32_t height);
    
    /**
     * @brief Encode frame to specified format
     * @param frame_data Raw frame data
     * @param width Frame width
     * @param height Frame height
     * @param format Target format
     * @param quality Encoding quality
     * @return Encoded frame data as base64 or raw bytes
     */
    std::pair<std::string, std::vector<uint8_t>> encodeFrameData(
        const std::vector<uint8_t>& frame_data,
        uint32_t width,
        uint32_t height,
        FrameFormat format,
        VideoQuality quality
    );
    
    /**
     * @brief Check if frame should be dropped due to timing constraints
     * @param encode_start_time_us When encoding started
     * @return True if frame should be dropped
     */
    bool shouldDropFrame(uint64_t encode_start_time_us) const;
    
    /**
     * @brief Update encoding statistics
     * @param encode_time_us Time taken for encoding
     * @param frame_size_bytes Size of encoded frame
     * @param was_keyframe True if this was a keyframe
     */
    void updateStatistics(uint32_t encode_time_us, uint32_t frame_size_bytes, bool was_keyframe);
    
    /**
     * @brief Adapt quality based on current performance
     */
    void adaptQuality();
    
    // Statistics tracking
    mutable std::mutex stats_mutex_;
    Statistics stats_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
};

/**
 * @brief Factory function to create JAMCam encoder with ultra-low latency settings
 * @param resolution Target resolution
 * @return Configured encoder instance
 */
std::unique_ptr<JAMCamEncoder> createUltraLowLatencyEncoder(VideoResolution resolution = VideoResolution::ULTRA_LOW_72P);

/**
 * @brief Factory function to create JAMCam encoder with balanced settings
 * @param resolution Target resolution
 * @return Configured encoder instance
 */
std::unique_ptr<JAMCamEncoder> createBalancedEncoder(VideoResolution resolution = VideoResolution::LOW_144P);

/**
 * @brief Factory function to create JAMCam encoder with high quality settings
 * @param resolution Target resolution  
 * @return Configured encoder instance
 */
std::unique_ptr<JAMCamEncoder> createHighQualityEncoder(VideoResolution resolution = VideoResolution::HIGH_360P);

} // namespace jsonvid 