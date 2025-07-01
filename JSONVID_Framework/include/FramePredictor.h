#pragma once

#include "JSONVIDMessage.h"
#include "VideoBufferManager.h"
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>

namespace jsonvid {

/**
 * @brief PNTBTR video frame predictor for ultra-low latency packet recovery
 * 
 * Predicts missing video frames using temporal motion estimation and
 * frame interpolation for seamless video streaming with packet loss.
 */
class FramePredictor {
public:
    /**
     * @brief Frame prediction configuration
     */
    struct Config {
        uint32_t max_reference_frames = 4;     // Number of reference frames to keep
        uint32_t prediction_window_frames = 8; // Max frames to predict ahead
        uint32_t max_prediction_age_ms = 100;  // Max age for predictions
        
        // Prediction quality vs speed
        bool enable_motion_estimation = true;   // Use motion vectors for prediction
        bool enable_temporal_smoothing = true; // Apply temporal filtering
        bool enable_adaptive_prediction = true; // Adapt based on motion complexity
        
        // Motion estimation parameters
        uint32_t block_size = 8;               // Motion estimation block size
        uint32_t search_range = 16;            // Motion vector search range
        uint8_t subpixel_precision = 2;        // Sub-pixel motion precision (1/4 pixel)
        
        // Quality thresholds
        uint8_t min_confidence_threshold = 64; // Minimum prediction confidence (0-255)
        uint8_t max_prediction_distance = 3;   // Max frames to predict forward
        float motion_threshold = 0.1f;         // Threshold for detecting significant motion
        
        // Performance tuning
        uint32_t max_prediction_time_us = 200; // Max time allowed for prediction
        bool enable_parallel_prediction = true; // Use multiple threads for prediction
        bool enable_gpu_prediction = false;    // Use GPU acceleration (experimental)
    };
    
    /**
     * @brief Motion vector structure for frame prediction
     */
    struct MotionVector {
        int16_t dx = 0;                        // Horizontal displacement (1/4 pixel units)
        int16_t dy = 0;                        // Vertical displacement (1/4 pixel units)
        uint16_t block_x = 0;                  // Block position X
        uint16_t block_y = 0;                  // Block position Y
        uint8_t confidence = 0;                // Motion vector confidence (0-255)
        uint16_t sad = 0;                      // Sum of Absolute Differences
    };
    
    /**
     * @brief Frame reference for prediction
     */
    struct FrameReference {
        uint64_t sequence_number = 0;
        uint64_t timestamp_us = 0;
        std::unique_ptr<VideoBufferManager::FrameBuffer> frame_data;
        std::vector<MotionVector> motion_vectors;
        
        // Frame metadata
        uint32_t width = 0;
        uint32_t height = 0;
        double quality_score = 1.0;
        bool is_keyframe = false;
        
        // Prediction state
        uint32_t prediction_generation = 0;    // 0 = original, >0 = predicted
        uint8_t prediction_confidence = 255;   // Overall frame confidence
        std::chrono::high_resolution_clock::time_point creation_time;
    };

private:
    Config config_;
    
    // Reference frame storage
    std::vector<std::unique_ptr<FrameReference>> reference_frames_;
    std::mutex references_mutex_;
    
    // Prediction state
    std::atomic<uint64_t> predictions_made_{0};
    std::atomic<uint64_t> predictions_validated_{0};
    std::atomic<uint64_t> prediction_hits_{0};
    std::atomic<uint64_t> prediction_misses_{0};
    
    // Performance tracking
    mutable std::mutex stats_mutex_;
    double average_prediction_time_us_ = 0.0;
    double max_prediction_time_us_ = 0.0;
    uint32_t prediction_count_ = 0;
    
    // Motion analysis state
    struct MotionAnalysis {
        double average_motion_magnitude = 0.0;
        double motion_variance = 0.0;
        bool high_motion_detected = false;
        uint32_t static_blocks = 0;
        uint32_t moving_blocks = 0;
        double scene_complexity = 0.5;         // 0 = simple, 1 = complex
    };
    
    MotionAnalysis motion_analysis_;

public:
    /**
     * @brief Constructor
     * @param config Predictor configuration
     */
    explicit FramePredictor(const Config& config = Config{});
    
    /**
     * @brief Destructor
     */
    ~FramePredictor();
    
    /**
     * @brief Add reference frame for future predictions
     * @param frame Frame to add as reference
     * @param sequence_number Frame sequence number
     * @param timestamp_us Frame timestamp
     */
    void addReferenceFrame(
        std::unique_ptr<VideoBufferManager::FrameBuffer> frame,
        uint64_t sequence_number,
        uint64_t timestamp_us
    );
    
    /**
     * @brief Predict missing frame
     * @param target_sequence Target sequence number to predict
     * @param target_timestamp Target timestamp for prediction
     * @return Predicted frame or nullptr if prediction failed
     */
    std::unique_ptr<VideoBufferManager::FrameBuffer> predictFrame(
        uint64_t target_sequence,
        uint64_t target_timestamp
    );
    
    /**
     * @brief Check if frame can be predicted with acceptable confidence
     * @param target_sequence Target sequence number
     * @param target_timestamp Target timestamp
     * @return Prediction confidence (0-255), 0 = cannot predict
     */
    uint8_t getPredictionConfidence(uint64_t target_sequence, uint64_t target_timestamp) const;
    
    /**
     * @brief Validate a prediction against actual received frame
     * @param predicted_frame Previously predicted frame
     * @param actual_frame Actually received frame
     * @return Validation score (0-1), 1 = perfect prediction
     */
    double validatePrediction(
        const VideoBufferManager::FrameBuffer* predicted_frame,
        const VideoBufferManager::FrameBuffer* actual_frame
    );
    
    /**
     * @brief Update predictor configuration
     * @param config New configuration
     */
    void updateConfig(const Config& config);
    
    /**
     * @brief Get current configuration
     * @return Current predictor configuration
     */
    const Config& getConfig() const { return config_; }
    
    /**
     * @brief Clear all reference frames and reset predictor state
     */
    void reset();
    
    /**
     * @brief Get predictor statistics
     */
    struct Statistics {
        uint64_t predictions_made = 0;
        uint64_t predictions_validated = 0;
        uint64_t prediction_hits = 0;
        uint64_t prediction_misses = 0;
        
        double prediction_accuracy = 0.0;      // Hit rate percentage
        double average_prediction_time_us = 0.0;
        double max_prediction_time_us = 0.0;
        
        // Quality metrics
        double average_prediction_confidence = 0.0;
        double average_motion_magnitude = 0.0;
        double scene_complexity = 0.5;
        
        // Reference frame status
        uint32_t reference_frames_count = 0;
        uint32_t oldest_reference_age_ms = 0;
        uint32_t memory_usage_mb = 0;
        
        // Performance analysis
        uint32_t predictions_dropped_timeout = 0;
        uint32_t predictions_dropped_confidence = 0;
        double cpu_usage_percent = 0.0;
    };
    
    /**
     * @brief Get current statistics
     * @return Current predictor statistics
     */
    Statistics getStatistics() const;
    
    /**
     * @brief Reset statistics counters
     */
    void resetStatistics();
    
    /**
     * @brief Cleanup expired reference frames
     */
    void performMaintenance();
    
    /**
     * @brief Get optimal predictor settings for video characteristics
     * @param resolution Video resolution
     * @param fps Frames per second
     * @param motion_level Expected motion level (0=static, 1=high motion)
     * @return Recommended configuration
     */
    static Config getOptimalConfig(VideoResolution resolution, uint32_t fps, float motion_level);

private:
    /**
     * @brief Estimate motion vectors between two frames
     * @param reference_frame Reference frame
     * @param target_timestamp Target timestamp for motion estimation
     * @return Vector of motion vectors
     */
    std::vector<MotionVector> estimateMotion(
        const FrameReference* reference_frame,
        uint64_t target_timestamp
    );
    
    /**
     * @brief Apply motion compensation to predict frame
     * @param reference_frame Source reference frame
     * @param motion_vectors Motion vectors to apply
     * @param target_frame Output predicted frame
     * @return True if motion compensation successful
     */
    bool applyMotionCompensation(
        const FrameReference* reference_frame,
        const std::vector<MotionVector>& motion_vectors,
        VideoBufferManager::FrameBuffer* target_frame
    );
    
    /**
     * @brief Find best reference frame for prediction
     * @param target_sequence Target sequence number
     * @param target_timestamp Target timestamp
     * @return Pointer to best reference frame or nullptr
     */
    const FrameReference* findBestReference(uint64_t target_sequence, uint64_t target_timestamp) const;
    
    /**
     * @brief Calculate temporal interpolation weight
     * @param ref_timestamp Reference frame timestamp
     * @param target_timestamp Target timestamp
     * @param next_timestamp Next frame timestamp (optional)
     * @return Interpolation weight (0-1)
     */
    float calculateInterpolationWeight(
        uint64_t ref_timestamp,
        uint64_t target_timestamp,
        uint64_t next_timestamp = 0
    ) const;
    
    /**
     * @brief Analyze motion complexity in frame
     * @param frame Frame to analyze
     * @param motion_vectors Motion vectors (if available)
     * @return Motion analysis results
     */
    MotionAnalysis analyzeMotion(
        const VideoBufferManager::FrameBuffer* frame,
        const std::vector<MotionVector>& motion_vectors = {}
    );
    
    /**
     * @brief Apply temporal smoothing to predicted frame
     * @param predicted_frame Frame to smooth
     * @param reference_frames Reference frames for smoothing
     */
    void applyTemporalSmoothing(
        VideoBufferManager::FrameBuffer* predicted_frame,
        const std::vector<const FrameReference*>& reference_frames
    );
    
    /**
     * @brief Calculate Sum of Absolute Differences between blocks
     * @param frame1 First frame
     * @param frame2 Second frame
     * @param block_x1 Block X position in frame1
     * @param block_y1 Block Y position in frame1
     * @param block_x2 Block X position in frame2
     * @param block_y2 Block Y position in frame2
     * @return SAD value
     */
    uint32_t calculateSAD(
        const VideoBufferManager::FrameBuffer* frame1,
        const VideoBufferManager::FrameBuffer* frame2,
        uint32_t block_x1, uint32_t block_y1,
        uint32_t block_x2, uint32_t block_y2
    ) const;
    
    /**
     * @brief Cleanup expired reference frames
     */
    void cleanupExpiredReferences();
    
    /**
     * @brief Update prediction statistics
     * @param prediction_time_us Time taken for prediction
     * @param confidence Prediction confidence
     * @param success True if prediction was successful
     */
    void updatePredictionStats(uint32_t prediction_time_us, uint8_t confidence, bool success);
};

/**
 * @brief Utility functions for frame prediction
 */
namespace FramePredictionUtils {
    /**
     * @brief Calculate optimal block size for motion estimation
     * @param frame_width Frame width in pixels
     * @param frame_height Frame height in pixels
     * @param motion_level Expected motion level (0-1)
     * @return Recommended block size
     */
    uint32_t calculateOptimalBlockSize(uint32_t frame_width, uint32_t frame_height, float motion_level);
    
    /**
     * @brief Estimate motion complexity from frame differences
     * @param frame1 First frame
     * @param frame2 Second frame
     * @return Motion complexity score (0-1)
     */
    float estimateMotionComplexity(
        const VideoBufferManager::FrameBuffer* frame1,
        const VideoBufferManager::FrameBuffer* frame2
    );
    
    /**
     * @brief Calculate frame similarity score
     * @param frame1 First frame
     * @param frame2 Second frame
     * @return Similarity score (0-1), 1 = identical
     */
    float calculateFrameSimilarity(
        const VideoBufferManager::FrameBuffer* frame1,
        const VideoBufferManager::FrameBuffer* frame2
    );
    
    /**
     * @brief Check if frame is suitable for prediction reference
     * @param frame Frame to check
     * @param quality_threshold Minimum quality threshold
     * @return True if frame is suitable as reference
     */
    bool isSuitableReference(const VideoBufferManager::FrameBuffer* frame, float quality_threshold = 0.5f);
}

} // namespace jsonvid 