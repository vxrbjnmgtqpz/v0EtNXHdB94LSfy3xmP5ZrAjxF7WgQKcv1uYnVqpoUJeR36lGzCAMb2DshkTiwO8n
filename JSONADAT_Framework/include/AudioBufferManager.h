#pragma once

#include "LockFreeQueue.h"
#include <vector>
#include <atomic>
#include <memory>
#include <chrono>
#include <mutex>

namespace jsonadat {

/**
 * @brief Lock-free audio buffer manager for real-time audio processing
 * 
 * Handles buffering of audio data with minimal latency and thread-safe access
 * for both encoding and decoding operations.
 */
class AudioBufferManager {
public:
    /**
     * @brief Audio buffer configuration
     */
    struct Config {
        uint32_t sample_rate = 96000;
        uint32_t buffer_size_ms = 20;
        uint32_t max_buffers = 64;  // Maximum number of buffers to allocate
        bool enable_overflow_protection = true;
        bool enable_underrun_protection = true;
    };

    /**
     * @brief Audio frame structure
     */
    struct AudioFrame {
        std::vector<float> samples;
        uint64_t timestamp_us = 0;
        uint32_t frame_id = 0;
        uint8_t channel = 0;
        bool is_valid = true;
        
        AudioFrame() = default;
        AudioFrame(const std::vector<float>& s, uint64_t ts, uint32_t id = 0, uint8_t ch = 0)
            : samples(s), timestamp_us(ts), frame_id(id), channel(ch) {}
    };

    /**
     * @brief Buffer statistics
     */
    struct Statistics {
        uint64_t frames_written = 0;
        uint64_t frames_read = 0;
        uint64_t overruns = 0;
        uint64_t underruns = 0;
        uint32_t current_fill_level = 0;
        uint32_t max_fill_level = 0;
        double average_write_time_us = 0.0;
        double average_read_time_us = 0.0;
        uint32_t buffer_utilization_percent = 0;
    };

private:
    Config config_;
    std::unique_ptr<LockFreeQueue<AudioFrame>> frame_queue_;
    
    std::atomic<uint32_t> write_frame_id_{0};
    std::atomic<uint32_t> read_frame_id_{0};
    std::atomic<bool> is_running_{false};
    
    // Pre-allocated frame pool to avoid dynamic allocation
    std::vector<AudioFrame> frame_pool_;
    std::atomic<uint32_t> pool_index_{0};
    
    // Statistics tracking
    mutable std::mutex stats_mutex_;
    Statistics stats_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
    
    // Buffer management
    uint32_t frame_size_samples_;
    uint64_t frame_duration_us_;

public:
    /**
     * @brief Constructor
     * @param config Buffer configuration
     */
    explicit AudioBufferManager(const Config& config);

    /**
     * @brief Destructor
     */
    ~AudioBufferManager();

    /**
     * @brief Initialize the buffer manager
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Shutdown the buffer manager
     */
    void shutdown();

    /**
     * @brief Check if buffer manager is running
     * @return True if running
     */
    bool isRunning() const { return is_running_.load(); }

    /**
     * @brief Write audio frame to buffer
     * @param frame Audio frame to write
     * @return True if write successful
     */
    bool writeFrame(const AudioFrame& frame);

    /**
     * @brief Write audio samples to buffer (creates frame automatically)
     * @param samples Audio samples
     * @param timestamp_us Timestamp in microseconds
     * @param channel Channel identifier
     * @return True if write successful
     */
    bool writeSamples(const std::vector<float>& samples, 
                     uint64_t timestamp_us = 0, 
                     uint8_t channel = 0);

    /**
     * @brief Read audio frame from buffer
     * @param frame Output frame
     * @return True if read successful
     */
    bool readFrame(AudioFrame& frame);

    /**
     * @brief Read audio samples from buffer
     * @param samples Output samples vector
     * @param timestamp_us Output timestamp
     * @param channel Output channel
     * @return True if read successful
     */
    bool readSamples(std::vector<float>& samples, 
                    uint64_t& timestamp_us, 
                    uint8_t& channel);

    /**
     * @brief Try to write frame without blocking
     * @param frame Audio frame to write
     * @return True if write successful, false if buffer full
     */
    bool tryWriteFrame(const AudioFrame& frame);

    /**
     * @brief Try to read frame without blocking
     * @param frame Output frame
     * @return True if read successful, false if buffer empty
     */
    bool tryReadFrame(AudioFrame& frame);

    /**
     * @brief Get number of frames available for reading
     * @return Number of available frames
     */
    uint32_t getAvailableFrames() const;

    /**
     * @brief Get free space in buffer (frames)
     * @return Number of free frame slots
     */
    uint32_t getFreeSpace() const;

    /**
     * @brief Get current buffer fill level (0-100%)
     * @return Fill level percentage
     */
    uint32_t getFillLevel() const;

    /**
     * @brief Check if buffer is empty
     * @return True if empty
     */
    bool isEmpty() const;

    /**
     * @brief Check if buffer is full
     * @return True if full
     */
    bool isFull() const;

    /**
     * @brief Clear all buffered data
     */
    void clear();

    /**
     * @brief Flush buffer (wait for all data to be processed)
     * @param timeout_ms Timeout in milliseconds
     * @return True if flushed successfully
     */
    bool flush(uint32_t timeout_ms = 1000);

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Update configuration (may require restart)
     * @param config New configuration
     * @return True if updated successfully
     */
    bool updateConfig(const Config& config);

    /**
     * @brief Get current statistics
     * @return Current buffer statistics
     */
    Statistics getStatistics() const;

    /**
     * @brief Reset statistics counters
     */
    void resetStatistics();

    /**
     * @brief Get expected frame size in samples
     * @return Frame size in samples
     */
    uint32_t getFrameSizeSamples() const { return frame_size_samples_; }

    /**
     * @brief Get frame duration in microseconds
     * @return Frame duration
     */
    uint64_t getFrameDuration() const { return frame_duration_us_; }

    /**
     * @brief Calculate latency based on current buffer state
     * @return Estimated latency in milliseconds
     */
    double getEstimatedLatency() const;

    /**
     * @brief Enable/disable overflow protection
     * @param enable True to enable protection
     */
    void setOverflowProtection(bool enable);

    /**
     * @brief Enable/disable underrun protection
     * @param enable True to enable protection
     */
    void setUnderrunProtection(bool enable);

    /**
     * @brief Set buffer size dynamically
     * @param size_ms New buffer size in milliseconds
     * @return True if size changed successfully
     */
    bool setBufferSize(uint32_t size_ms);

    /**
     * @brief Get current timestamp in microseconds
     * @return Current timestamp
     */
    static uint64_t getCurrentTimestamp();

    /**
     * @brief Calculate buffer size in frames for given duration
     * @param sample_rate Sample rate in Hz
     * @param duration_ms Duration in milliseconds
     * @param frame_size_samples Frame size in samples
     * @return Buffer size in frames
     */
    static uint32_t calculateBufferSizeFrames(uint32_t sample_rate, 
                                             uint32_t duration_ms, 
                                             uint32_t frame_size_samples);

private:
    /**
     * @brief Initialize frame pool
     */
    void initializeFramePool();

    /**
     * @brief Get next available frame from pool
     * @return Pointer to available frame or nullptr
     */
    AudioFrame* getPoolFrame();

    /**
     * @brief Return frame to pool
     * @param frame Frame to return
     */
    void returnPoolFrame(AudioFrame* frame);

    /**
     * @brief Update statistics after operation
     * @param operation_time_us Time taken for operation
     * @param is_write True if write operation, false if read
     */
    void updateStatistics(uint64_t operation_time_us, bool is_write);

    /**
     * @brief Calculate frame size based on configuration
     */
    void calculateFrameParameters();

    /**
     * @brief Handle buffer overflow condition
     */
    void handleOverflow();

    /**
     * @brief Handle buffer underrun condition
     */
    void handleUnderrun();
};

/**
 * @brief Create a default audio buffer manager
 * @param sample_rate Sample rate in Hz
 * @param buffer_size_ms Buffer size in milliseconds
 * @return Unique pointer to AudioBufferManager
 */
std::unique_ptr<AudioBufferManager> createDefaultBufferManager(
    uint32_t sample_rate = 96000, 
    uint32_t buffer_size_ms = 20);

/**
 * @brief Create a low-latency audio buffer manager
 * @param sample_rate Sample rate in Hz
 * @return Unique pointer to AudioBufferManager configured for low latency
 */
std::unique_ptr<AudioBufferManager> createLowLatencyBufferManager(uint32_t sample_rate = 96000);

} // namespace jsonadat 