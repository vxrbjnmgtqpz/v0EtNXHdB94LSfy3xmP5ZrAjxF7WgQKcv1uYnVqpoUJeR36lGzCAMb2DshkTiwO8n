#pragma once

#include "JVIDMessage.h"
#include "FramePredictor.h"
#include "VideoBufferManager.h"
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace jvid {

/**
 * @brief JAMCam video decoder for ultra-low latency video rendering
 * 
 * Decodes JVID messages into video frames and renders them with
 * target latencies <300μs for real-time video display over TOAST transport.
 */
class JAMCamDecoder {
public:
    /**
     * @brief Decoder configuration
     */
    struct Config {
        uint32_t target_latency_us = 300;    // Target decode latency (300μs)
        uint32_t max_frame_age_us = 33333;   // Drop frames older than this (~30fps)
        bool enable_frame_prediction = true; // PNTBTR frame recovery
        bool enable_gpu_decoding = true;     // Use GPU acceleration if available
        bool enable_zero_copy = true;        // Zero-copy frame processing
        bool enable_adaptive_quality = true; // Adjust quality based on performance
        
        // Display configuration
        uint32_t display_width = 256;        // Target display width
        uint32_t display_height = 144;       // Target display height
        bool maintain_aspect_ratio = true;   // Preserve original aspect ratio
        bool enable_upscaling = true;        // Allow upscaling to display size
        
        // Buffer management
        uint32_t decode_buffer_size = 4;     // Number of decode buffers
        uint32_t display_buffer_size = 2;    // Number of display buffers
        
        // Quality and performance
        uint32_t max_decode_time_us = 500;   // Drop frames taking longer than this
        bool enable_frame_dropping = true;   // Drop late frames
        bool enable_interpolation = true;    // Interpolate missing frames
        uint8_t prediction_confidence_threshold = 128; // Min confidence for predicted frames
        
        // Synchronization
        bool enable_audio_sync = false;      // Sync with audio timestamps
        uint32_t sync_tolerance_us = 1000;   // Sync tolerance window
    };
    
    /**
     * @brief Frame ready callback signature
     * Called when decoded frame is ready for display
     */
    using FrameReadyCallback = std::function<void(const VideoBufferManager::FrameBuffer*)>;
    
    /**
     * @brief Frame drop callback signature
     * Called when frame is dropped due to latency or errors
     */
    using FrameDropCallback = std::function<void(const JVIDMessage&, const std::string& reason)>;
    
    /**
     * @brief Statistics callback signature
     * Called periodically with decoder performance metrics
     */
    using StatsCallback = std::function<void(const VideoStreamStats&)>;

private:
    Config config_;
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_decoding_{false};
    
    // Frame processing components
    std::unique_ptr<VideoBufferManager> buffer_manager_;
    std::unique_ptr<FramePredictor> frame_predictor_;
    
    // Processing threads
    std::thread decode_thread_;
    std::thread display_thread_;
    
    // Input frame queue
    LockFreeQueue<JVIDMessage> input_queue_;
    LockFreeQueue<VideoBufferManager::FrameBuffer*> ready_queue_;
    
    // Callbacks
    FrameReadyCallback frame_ready_callback_;
    FrameDropCallback frame_drop_callback_;
    StatsCallback stats_callback_;
    
    // Synchronization
    std::mutex state_mutex_;
    std::condition_variable frame_available_;
    
    // Statistics and monitoring
    std::atomic<uint64_t> frames_received_{0};
    std::atomic<uint64_t> frames_decoded_{0};
    std::atomic<uint64_t> frames_dropped_{0};
    std::atomic<uint64_t> frames_predicted_{0};
    uint64_t start_time_us_;
    
    // Decoder state
    struct DecoderState {
        uint64_t last_sequence_number = 0;
        uint64_t expected_sequence = 0;
        std::vector<uint64_t> missing_sequences;
        
        // Timing tracking
        uint64_t last_frame_timestamp = 0;
        uint32_t current_fps = 0;
        double average_decode_time_us = 0.0;
        
        // Quality tracking  
        double current_quality_score = 1.0;
        uint32_t consecutive_drops = 0;
        bool quality_adaptation_active = false;
        
        // Sync state
        int64_t audio_video_offset_us = 0;   // Video ahead of audio (positive)
        bool sync_established = false;
    };
    
    DecoderState decoder_state_;
    mutable std::mutex state_lock_;

public:
    /**
     * @brief Constructor
     * @param config Decoder configuration
     */
    explicit JAMCamDecoder(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~JAMCamDecoder();
    
    /**
     * @brief Set frame ready callback
     * @param callback Function to call when frame is ready for display
     */
    void setFrameReadyCallback(FrameReadyCallback callback);
    
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
     * @brief Start video decoding
     * @return True if started successfully
     */
    bool start();
    
    /**
     * @brief Stop video decoding
     */
    void stop();
    
    /**
     * @brief Check if decoder is running
     * @return True if running
     */
    bool isRunning() const { return is_running_.load(); }
    
    /**
     * @brief Process incoming video message
     * @param message Video message to decode
     * @return True if message was queued for processing
     */
    bool processMessage(const JVIDMessage& message);
    
    /**
     * @brief Update decoder configuration
     * @param config New configuration
     * @return True if updated successfully
     */
    bool updateConfig(const Config& config);
    
    /**
     * @brief Get current configuration
     * @return Current decoder configuration
     */
    const Config& getConfig() const { return config_; }
    
    /**
     * @brief Manually decode a frame (for testing)
     * @param message Video message to decode
     * @return Decoded frame buffer or nullptr if decoding failed
     */
    std::unique_ptr<VideoBufferManager::FrameBuffer> decodeFrame(const JVIDMessage& message);
    
    /**
     * @brief Get decoder statistics
     */
    struct Statistics {
        uint64_t frames_received = 0;
        uint64_t frames_decoded = 0;
        uint64_t frames_dropped = 0;
        uint64_t frames_predicted = 0;
        uint64_t frames_interpolated = 0;
        
        double average_decode_time_us = 0.0;
        double max_decode_time_us = 0.0;
        double average_frame_latency_us = 0.0;
        double current_fps = 0.0;
        double target_fps = 15.0;
        
        // Quality metrics
        double average_quality_score = 1.0;
        double packet_loss_rate = 0.0;
        uint32_t out_of_order_frames = 0;
        uint32_t duplicate_frames = 0;
        
        // Performance metrics
        double cpu_usage_percent = 0.0;
        double memory_usage_mb = 0.0;
        double decode_queue_depth = 0.0;
        
        // Timing analysis
        double decode_jitter_us = 0.0;
        uint32_t missed_display_deadlines = 0;
        double frame_presentation_accuracy = 1.0;
        
        // Synchronization
        int64_t audio_video_sync_offset_us = 0;
        bool sync_established = false;
        double sync_accuracy_percent = 100.0;
        
        // Error recovery
        uint32_t prediction_successes = 0;
        uint32_t prediction_failures = 0;
        double prediction_accuracy = 1.0;
    };
    
    /**
     * @brief Get current statistics
     * @return Current decoder statistics
     */
    Statistics getStatistics() const;
    
    /**
     * @brief Reset statistics counters
     */
    void resetStatistics();
    
    /**
     * @brief Flush all pending frames and reset decoder state
     */
    void flush();
    
    /**
     * @brief Set audio synchronization timestamp
     * @param audio_timestamp_us Current audio timestamp for A/V sync
     */
    void setAudioTimestamp(uint64_t audio_timestamp_us);
    
    /**
     * @brief Get optimal decoder settings for target latency
     * @param target_latency_us Desired latency in microseconds
     * @return Recommended configuration
     */
    static Config getOptimalConfig(uint32_t target_latency_us);

private:
    /**
     * @brief Main decode thread function
     */
    void decodeThreadFunction();
    
    /**
     * @brief Main display thread function
     */
    void displayThreadFunction();
    
    /**
     * @brief Decode a single video message
     * @param message Message to decode
     * @param output_buffer Buffer to store decoded frame
     * @return True if decoding successful
     */
    bool decodeSingleFrame(const JVIDMessage& message, VideoBufferManager::FrameBuffer* output_buffer);
    
    /**
     * @brief Decode base64-encoded frame data
     * @param base64_data Base64-encoded frame
     * @param format Frame format
     * @param output_buffer Buffer to store decoded pixels
     * @return True if decoding successful
     */
    bool decodeFrameData(
        const std::string& base64_data,
        FrameFormat format,
        VideoBufferManager::FrameBuffer* output_buffer
    );
    
    /**
     * @brief Check if frame should be dropped due to timing
     * @param message Frame message to check
     * @return True if frame should be dropped
     */
    bool shouldDropFrame(const JVIDMessage& message) const;
    
    /**
     * @brief Handle missing frame via prediction
     * @param missing_sequence Sequence number of missing frame
     * @return Predicted frame buffer or nullptr if prediction failed
     */
    VideoBufferManager::FrameBuffer* handleMissingFrame(uint64_t missing_sequence);
    
    /**
     * @brief Update sequence tracking and detect missing frames
     * @param sequence_number Current frame sequence number
     */
    void updateSequenceTracking(uint64_t sequence_number);
    
    /**
     * @brief Apply quality adaptation based on performance
     */
    void adaptQuality();
    
    /**
     * @brief Update synchronization with audio
     * @param video_timestamp Video frame timestamp
     */
    void updateAudioVideoSync(uint64_t video_timestamp);
    
    /**
     * @brief Scale frame to display size
     * @param source_buffer Source frame buffer
     * @param target_buffer Target frame buffer
     * @return True if scaling successful
     */
    bool scaleFrameToDisplay(
        const VideoBufferManager::FrameBuffer* source_buffer,
        VideoBufferManager::FrameBuffer* target_buffer
    );
    
    /**
     * @brief Update decoder statistics
     * @param decode_time_us Time taken for decoding
     * @param was_predicted True if frame was predicted
     * @param quality_score Frame quality score 0-1
     */
    void updateStatistics(uint32_t decode_time_us, bool was_predicted, double quality_score);
    
    /**
     * @brief Cleanup expired frames and perform maintenance
     */
    void performMaintenance();
    
    // Statistics tracking
    mutable std::mutex stats_mutex_;
    Statistics stats_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
};

/**
 * @brief Factory function to create JAMCam decoder with ultra-low latency settings
 * @param display_resolution Target display resolution
 * @return Configured decoder instance
 */
std::unique_ptr<JAMCamDecoder> createUltraLowLatencyDecoder(
    std::pair<uint32_t, uint32_t> display_resolution = {256, 144}
);

/**
 * @brief Factory function to create JAMCam decoder with balanced settings
 * @param display_resolution Target display resolution
 * @return Configured decoder instance
 */
std::unique_ptr<JAMCamDecoder> createBalancedDecoder(
    std::pair<uint32_t, uint32_t> display_resolution = {426, 240}
);

/**
 * @brief Factory function to create JAMCam decoder with high quality settings
 * @param display_resolution Target display resolution
 * @return Configured decoder instance
 */
std::unique_ptr<JAMCamDecoder> createHighQualityDecoder(
    std::pair<uint32_t, uint32_t> display_resolution = {640, 360}
);

} // namespace jvid 