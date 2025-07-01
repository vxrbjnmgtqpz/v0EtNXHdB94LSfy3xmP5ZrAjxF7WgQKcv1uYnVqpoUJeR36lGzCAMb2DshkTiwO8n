#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <memory>
#include <array>
#include <mutex>
#include <nlohmann/json.hpp>

namespace jsonvid {

/**
 * @brief Video resolution presets optimized for ultra-low latency streaming
 */
enum class VideoResolution : uint8_t {
    ULTRA_LOW_72P = 0,    // 128x72 (ultra-low latency, ~1KB frames)
    LOW_144P = 1,         // 256x144 (low latency, ~4KB frames)  
    MEDIUM_240P = 2,      // 426x240 (balanced, ~8KB frames)
    HIGH_360P = 3         // 640x360 (higher quality, ~16KB frames)
};

/**
 * @brief Video compression quality levels
 */
enum class VideoQuality : uint8_t {
    ULTRA_FAST = 0,       // Maximum compression, minimum quality (~200μs encoding)
    FAST = 1,             // Balanced compression/quality (~500μs encoding)
    BALANCED = 2,         // Better quality, moderate latency (~1ms encoding)
    HIGH_QUALITY = 3      // Best quality, higher latency (~2ms encoding)
};

/**
 * @brief Frame encoding formats for JSON transmission
 */
enum class FrameFormat : uint8_t {
    BASE64_JPEG = 0,      // JPEG compressed, base64 encoded
    BASE64_WEBP = 1,      // WebP compressed, base64 encoded
    PIXEL_ARRAY = 2,      // Raw pixel data as JSON array (for tiny frames)
    DIFFERENTIAL = 3      // Difference from previous frame (motion vectors)
};

/**
 * @brief Core JSONVID message structure for frame-by-frame video streaming
 */
struct JSONVIDMessage {
    // Message metadata
    uint64_t timestamp_us = 0;           // Microsecond timestamp for sync
    uint64_t sequence_number = 0;        // Frame sequence for ordering
    std::string session_id;              // Session identifier
    
    // Video stream information
    struct VideoInfo {
        VideoResolution resolution = VideoResolution::LOW_144P;
        VideoQuality quality = VideoQuality::FAST;
        FrameFormat format = FrameFormat::BASE64_JPEG;
        uint32_t frame_width = 256;      // Actual frame width in pixels
        uint32_t frame_height = 144;     // Actual frame height in pixels
        uint32_t fps_target = 15;        // Target framerate
        bool is_keyframe = false;        // True for I-frames, false for P-frames
        uint8_t stream_id = 0;           // For multi-camera setups
    } video_info;
    
    // Frame data
    struct FrameData {
        std::string frame_base64;        // Base64-encoded compressed frame
        std::vector<uint8_t> pixel_data; // Raw pixel data (for PIXEL_ARRAY format)
        uint32_t compressed_size = 0;    // Size of compressed data in bytes
        uint32_t original_size = 0;      // Uncompressed frame size
        double compression_ratio = 0.0;   // Achieved compression ratio
        
        // Motion vector data for differential encoding
        struct MotionVector {
            int16_t dx = 0;              // Horizontal displacement
            int16_t dy = 0;              // Vertical displacement
            uint16_t block_x = 0;        // Block position X
            uint16_t block_y = 0;        // Block position Y
        };
        std::vector<MotionVector> motion_vectors;
    } frame_data;
    
    // Timing and synchronization
    struct TimingInfo {
        uint64_t capture_timestamp_us = 0;   // When frame was captured
        uint64_t encode_timestamp_us = 0;    // When encoding started
        uint64_t send_timestamp_us = 0;      // When frame was sent
        uint32_t encode_duration_us = 0;     // Time taken to encode
        uint32_t expected_decode_us = 200;   // Expected decode time
        uint64_t audio_sync_timestamp = 0;   // Corresponding audio timestamp
    } timing_info;
    
    // Error detection and recovery
    struct FrameIntegrity {
        uint32_t checksum = 0;           // CRC32 checksum of frame data
        bool is_predicted = false;       // True if frame was PNTBTR predicted
        uint8_t prediction_confidence = 0; // 0-255 confidence in prediction
        uint64_t reference_frame_seq = 0;  // Sequence of reference frame used
    } integrity;
    
    // JAMCam specific features
    struct JAMCamFeatures {
        bool face_detection_enabled = false;
        bool auto_framing_active = false;
        bool lighting_normalized = false;
        uint8_t face_count = 0;
        
        // Face bounding boxes (if detected)
        struct FaceBBox {
            uint16_t x = 0, y = 0;       // Top-left corner
            uint16_t width = 0, height = 0; // Bounding box size
            uint8_t confidence = 0;       // Detection confidence 0-255
        };
        std::vector<FaceBBox> detected_faces;
        
        // Lighting adjustment parameters
        struct LightingParams {
            float brightness_adjust = 1.0f;
            float contrast_adjust = 1.0f;
            float gamma_adjust = 1.0f;
            bool auto_exposure = true;
        } lighting_params;
    } jamcam_features;
    
    /**
     * @brief Convert message to JSON string
     * @return JSON representation of the message
     */
    std::string toJSON() const;
    
    /**
     * @brief Create message from JSON string
     * @param json_string JSON string to parse
     * @return Parsed message object
     */
    static JSONVIDMessage fromJSON(const std::string& json_string);
    
    /**
     * @brief Create optimized compact JSON (minimal field names)
     * @return Compact JSON string for ultra-low latency
     */
    std::string toCompactJSON() const;
    
    /**
     * @brief Parse compact JSON format
     * @param compact_json Compact JSON string
     * @return Parsed message object
     */
    static JSONVIDMessage fromCompactJSON(const std::string& compact_json);
    
    /**
     * @brief Get message size in bytes (for bandwidth calculation)
     * @return Estimated size including JSON overhead
     */
    size_t getMessageSize() const;
    
    /**
     * @brief Validate message integrity
     * @return True if message is valid and complete
     */
    bool validate() const;
    
    /**
     * @brief Generate checksum for frame data
     */
    void calculateChecksum();
    
    /**
     * @brief Get resolution dimensions
     * @param resolution Video resolution enum
     * @return Pair of (width, height)
     */
    static std::pair<uint32_t, uint32_t> getResolutionDimensions(VideoResolution resolution);
    
    /**
     * @brief Get estimated encoding time for quality level
     * @param quality Video quality enum
     * @param frame_size Size of frame in pixels
     * @return Estimated encoding time in microseconds
     */
    static uint32_t getEstimatedEncodeTime(VideoQuality quality, uint32_t frame_size);
    
    /**
     * @brief Calculate target bitrate for resolution/quality combination
     * @param resolution Video resolution
     * @param quality Video quality
     * @param fps Frames per second
     * @return Target bitrate in bits per second
     */
    static uint32_t calculateTargetBitrate(VideoResolution resolution, VideoQuality quality, uint32_t fps);
};

/**
 * @brief Video stream statistics for monitoring performance
 */
struct VideoStreamStats {
    uint64_t frames_sent = 0;
    uint64_t frames_received = 0;
    uint64_t frames_dropped = 0;
    uint64_t frames_predicted = 0;          // PNTBTR recovery count
    
    double average_frame_size_kb = 0.0;
    double average_encode_time_us = 0.0;
    double average_decode_time_us = 0.0;
    double average_compression_ratio = 0.0;
    
    uint32_t current_fps = 0;
    uint32_t target_fps = 15;
    
    double bandwidth_kbps = 0.0;
    double packet_loss_rate = 0.0;
    
    uint64_t total_bytes_sent = 0;
    uint64_t total_bytes_received = 0;
    
    // Latency measurements
    double average_end_to_end_latency_us = 0.0;
    double min_latency_us = 0.0;
    double max_latency_us = 0.0;
    double jitter_us = 0.0;
    
    // Quality metrics
    double average_psnr = 0.0;              // Peak Signal-to-Noise Ratio
    double average_ssim = 0.0;              // Structural Similarity Index
    uint8_t subjective_quality = 0;         // 0-255 subjective quality rating
};

/**
 * @brief Message type constants for TOAST transport integration
 */
namespace MessageTypes {
    constexpr uint8_t VIDEO_FRAME = 0x10;
    constexpr uint8_t VIDEO_KEYFRAME = 0x11;
    constexpr uint8_t VIDEO_CONTROL = 0x12;
    constexpr uint8_t VIDEO_SYNC = 0x13;
}

/**
 * @brief Factory functions for common message types
 */

/**
 * @brief Create a standard video frame message
 * @param frame_data Base64-encoded frame data
 * @param width Frame width in pixels
 * @param height Frame height in pixels
 * @param timestamp_us Capture timestamp
 * @return Configured video message
 */
JSONVIDMessage createVideoFrame(
    const std::string& frame_data,
    uint32_t width,
    uint32_t height,
    uint64_t timestamp_us
);

/**
 * @brief Create a keyframe message
 * @param frame_data Base64-encoded keyframe data
 * @param width Frame width in pixels
 * @param height Frame height in pixels
 * @param timestamp_us Capture timestamp
 * @return Configured keyframe message
 */
JSONVIDMessage createKeyFrame(
    const std::string& frame_data,
    uint32_t width,
    uint32_t height,
    uint64_t timestamp_us
);

/**
 * @brief Create a minimal ultra-low latency frame (72p ULTRA_FAST)
 * @param pixel_data Raw pixel data
 * @param timestamp_us Capture timestamp
 * @return Minimal latency video message
 */
JSONVIDMessage createUltraLowLatencyFrame(
    const std::vector<uint8_t>& pixel_data,
    uint64_t timestamp_us
);

/**
 * @brief Utility functions for video processing
 */
namespace VideoUtils {
    /**
     * @brief Calculate CRC32 checksum for frame data
     * @param data Frame data bytes
     * @return CRC32 checksum
     */
    uint32_t calculateCRC32(const std::vector<uint8_t>& data);
    
    /**
     * @brief Estimate bandwidth usage for video stream
     * @param resolution Video resolution
     * @param quality Video quality  
     * @param fps Frames per second
     * @return Estimated bandwidth in KB/s
     */
    double estimateBandwidth(VideoResolution resolution, VideoQuality quality, uint32_t fps);
    
    /**
     * @brief Get optimal settings for target latency
     * @param target_latency_us Target end-to-end latency in microseconds
     * @return Recommended resolution and quality settings
     */
    std::pair<VideoResolution, VideoQuality> getOptimalSettings(uint32_t target_latency_us);
}

} // namespace jsonvid 