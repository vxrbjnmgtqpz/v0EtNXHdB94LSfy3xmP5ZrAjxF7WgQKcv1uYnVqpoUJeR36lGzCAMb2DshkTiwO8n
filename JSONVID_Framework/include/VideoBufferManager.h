#pragma once

#include "LockFreeQueue.h"
#include <vector>
#include <atomic>
#include <memory>
#include <chrono>
#include <mutex>

namespace jsonvid {

/**
 * @brief Lock-free video buffer manager for real-time video processing
 * 
 * Handles buffering of video frame data with minimal latency and thread-safe access
 * optimized for ultra-low latency video streaming applications.
 */
class VideoBufferManager {
public:
    /**
     * @brief Video frame buffer structure
     */
    struct FrameBuffer {
        std::vector<uint8_t> data;           // Frame pixel data
        uint64_t timestamp_us = 0;           // Capture timestamp
        uint32_t width = 0;                  // Frame width in pixels
        uint32_t height = 0;                 // Frame height in pixels
        uint32_t stride = 0;                 // Row stride in bytes
        uint8_t channels = 3;                // Number of color channels (RGB=3, RGBA=4)
        uint64_t sequence_number = 0;        // Frame sequence number
        bool is_keyframe = false;            // True for keyframes
        
        // Buffer management
        bool is_available = true;            // True if buffer can be used
        std::atomic<bool> is_locked{false};  // True if buffer is being processed
        uint64_t allocation_time_us = 0;     // When buffer was allocated
        
        /**
         * @brief Get frame size in bytes
         * @return Total size of frame data
         */
        size_t getFrameSize() const {
            return width * height * channels;
        }
        
        /**
         * @brief Check if frame buffer is valid
         * @return True if buffer contains valid frame data
         */
        bool isValid() const {
            return width > 0 && height > 0 && channels > 0 && 
                   data.size() >= getFrameSize();
        }
    };
    
    /**
     * @brief Buffer pool configuration
     */
    struct Config {
        uint32_t max_buffers = 8;            // Maximum number of frame buffers
        uint32_t max_width = 640;            // Maximum frame width supported
        uint32_t max_height = 360;           // Maximum frame height supported
        uint8_t max_channels = 4;            // Maximum channels (RGBA)
        uint32_t buffer_timeout_ms = 100;    // Timeout for unused buffers
        bool enable_zero_copy = true;        // Use zero-copy operations where possible
        bool enable_memory_pool = true;      // Pre-allocate memory pool
        size_t memory_pool_size_mb = 32;     // Memory pool size in MB
    };

private:
    Config config_;
    
    // Buffer pool
    std::vector<std::unique_ptr<FrameBuffer>> buffer_pool_;
    LockFreeQueue<FrameBuffer*> available_buffers_;
    LockFreeQueue<FrameBuffer*> ready_buffers_;
    
    // Memory management
    std::unique_ptr<uint8_t[]> memory_pool_;
    std::atomic<size_t> memory_pool_offset_{0};
    size_t memory_pool_size_;
    
    // Statistics and monitoring
    std::atomic<uint64_t> frames_allocated_{0};
    std::atomic<uint64_t> frames_released_{0};
    std::atomic<uint64_t> buffer_overruns_{0};
    std::atomic<uint64_t> allocation_failures_{0};
    
    mutable std::mutex stats_mutex_;
    std::chrono::high_resolution_clock::time_point start_time_;

public:
    /**
     * @brief Constructor
     * @param config Buffer manager configuration
     */
    explicit VideoBufferManager(const Config& config = Config{});
    
    /**
     * @brief Destructor
     */
    ~VideoBufferManager();
    
    /**
     * @brief Initialize buffer manager
     * @return True if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Shutdown buffer manager and release resources
     */
    void shutdown();
    
    /**
     * @brief Allocate a frame buffer
     * @param width Frame width in pixels
     * @param height Frame height in pixels
     * @param channels Number of color channels
     * @return Pointer to allocated buffer or nullptr if allocation failed
     */
    FrameBuffer* allocateBuffer(uint32_t width, uint32_t height, uint8_t channels = 3);
    
    /**
     * @brief Release a frame buffer back to the pool
     * @param buffer Buffer to release
     */
    void releaseBuffer(FrameBuffer* buffer);
    
    /**
     * @brief Get next ready frame buffer (non-blocking)
     * @return Pointer to ready buffer or nullptr if none available
     */
    FrameBuffer* getReadyBuffer();
    
    /**
     * @brief Mark buffer as ready for processing
     * @param buffer Buffer to mark as ready
     */
    void markBufferReady(FrameBuffer* buffer);
    
    /**
     * @brief Check if buffers are available for allocation
     * @return True if buffers are available
     */
    bool hasAvailableBuffers() const;
    
    /**
     * @brief Check if ready buffers are available for processing
     * @return True if ready buffers are available
     */
    bool hasReadyBuffers() const;
    
    /**
     * @brief Get number of available buffers
     * @return Count of available buffers
     */
    size_t getAvailableBufferCount() const;
    
    /**
     * @brief Get number of ready buffers
     * @return Count of ready buffers
     */
    size_t getReadyBufferCount() const;
    
    /**
     * @brief Copy frame data into buffer (zero-copy if possible)
     * @param buffer Target buffer
     * @param frame_data Source frame data
     * @param width Frame width
     * @param height Frame height
     * @param channels Number of channels
     * @param timestamp_us Frame timestamp
     * @param sequence_number Frame sequence number
     * @return True if copy successful
     */
    bool copyFrameData(
        FrameBuffer* buffer,
        const std::vector<uint8_t>& frame_data,
        uint32_t width,
        uint32_t height,
        uint8_t channels,
        uint64_t timestamp_us,
        uint64_t sequence_number
    );
    
    /**
     * @brief Buffer manager statistics
     */
    struct Statistics {
        uint64_t frames_allocated = 0;
        uint64_t frames_released = 0;
        uint64_t buffer_overruns = 0;
        uint64_t allocation_failures = 0;
        
        size_t available_buffers = 0;
        size_t ready_buffers = 0;
        size_t total_buffers = 0;
        
        double memory_usage_mb = 0.0;
        double memory_pool_usage_percent = 0.0;
        
        double average_allocation_time_us = 0.0;
        double average_release_time_us = 0.0;
        
        uint64_t peak_buffer_usage = 0;
        double buffer_turnover_rate = 0.0;       // Buffers per second
    };
    
    /**
     * @brief Get buffer manager statistics
     * @return Current statistics
     */
    Statistics getStatistics() const;
    
    /**
     * @brief Reset statistics counters
     */
    void resetStatistics();
    
    /**
     * @brief Configure buffer manager
     * @param config New configuration
     * @return True if configuration applied successfully
     */
    bool configure(const Config& config);
    
    /**
     * @brief Get current configuration
     * @return Current buffer manager configuration
     */
    const Config& getConfig() const { return config_; }
    
    /**
     * @brief Perform maintenance tasks (cleanup unused buffers, etc.)
     */
    void performMaintenance();
    
    /**
     * @brief Estimate memory usage for given configuration
     * @param config Configuration to estimate
     * @return Estimated memory usage in bytes
     */
    static size_t estimateMemoryUsage(const Config& config);
    
    /**
     * @brief Get optimal configuration for target performance
     * @param target_fps Target frames per second
     * @param max_resolution Maximum expected resolution
     * @return Recommended configuration
     */
    static Config getOptimalConfig(uint32_t target_fps, std::pair<uint32_t, uint32_t> max_resolution);

private:
    /**
     * @brief Initialize memory pool
     * @return True if memory pool initialized successfully
     */
    bool initializeMemoryPool();
    
    /**
     * @brief Allocate memory from pool
     * @param size Size to allocate in bytes
     * @return Pointer to allocated memory or nullptr if failed
     */
    uint8_t* allocateFromPool(size_t size);
    
    /**
     * @brief Create new frame buffer
     * @return Pointer to new buffer or nullptr if creation failed
     */
    std::unique_ptr<FrameBuffer> createFrameBuffer();
    
    /**
     * @brief Cleanup expired buffers
     */
    void cleanupExpiredBuffers();
    
    /**
     * @brief Update performance statistics
     */
    void updateStatistics();
    
    /**
     * @brief Check if buffer pool needs expansion
     * @return True if more buffers should be allocated
     */
    bool needsExpansion() const;
    
    /**
     * @brief Expand buffer pool if possible
     * @return Number of new buffers added
     */
    size_t expandBufferPool();
};

/**
 * @brief Utility functions for video buffer management
 */
namespace VideoBufferUtils {
    /**
     * @brief Calculate optimal buffer count for given parameters
     * @param fps Target frames per second
     * @param latency_budget_ms Latency budget in milliseconds
     * @return Recommended buffer count
     */
    uint32_t calculateOptimalBufferCount(uint32_t fps, uint32_t latency_budget_ms);
    
    /**
     * @brief Estimate frame data size
     * @param width Frame width
     * @param height Frame height
     * @param channels Number of channels
     * @return Frame size in bytes
     */
    size_t estimateFrameSize(uint32_t width, uint32_t height, uint8_t channels);
    
    /**
     * @brief Convert frame format between RGB/BGR/YUV (in-place)
     * @param data Frame data to convert
     * @param width Frame width
     * @param height Frame height
     * @param from_format Source format
     * @param to_format Target format
     * @return True if conversion successful
     */
    bool convertFrameFormat(
        std::vector<uint8_t>& data,
        uint32_t width,
        uint32_t height,
        const std::string& from_format,
        const std::string& to_format
    );
    
    /**
     * @brief Validate frame data integrity
     * @param buffer Frame buffer to validate
     * @return True if frame data appears valid
     */
    bool validateFrameData(const VideoBufferManager::FrameBuffer* buffer);
}

} // namespace jsonvid 