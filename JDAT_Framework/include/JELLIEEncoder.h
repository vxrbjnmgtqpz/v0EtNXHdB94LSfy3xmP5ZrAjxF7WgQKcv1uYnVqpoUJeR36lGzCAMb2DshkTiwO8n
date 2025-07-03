#pragma once

#include "JDATMessage.h"
#include "AudioBufferManager.h"
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>

namespace jdat {

/**
 * @brief JELLIE (JAM Embedded Low Latency Instrument Encoding) audio encoder
 * 
 * This class handles the conversion of real-time audio streams into JDAT messages
 * for transmission over the network using TOAST protocol.
 */
class JELLIEEncoder {
public:
    /**
     * @brief Encoding configuration
     */
    struct Config {
        SampleRate sample_rate = SampleRate::SR_96000;
        AudioQuality quality = AudioQuality::HIGH_PRECISION;
        uint32_t frame_size_samples = 480;  // 10ms at 48kHz
        uint8_t redundancy_level = 1;
        bool enable_adat_mapping = false;
        bool enable_192k_mode = false;
        uint32_t buffer_size_ms = 20;
        std::string session_id;
    };

    /**
     * @brief Audio input callback signature
     * Called when new audio data is available for encoding
     */
    using AudioCallback = std::function<void(const std::vector<float>&, uint64_t timestamp)>;

    /**
     * @brief Message output callback signature
     * Called when a JDAT message is ready for transmission
     */
    using MessageCallback = std::function<void(const JDATMessage&)>;

private:
    Config config_;
    std::unique_ptr<AudioBufferManager> buffer_manager_;
    MessageCallback message_callback_;
    
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_encoding_{false};
    std::atomic<uint64_t> sequence_number_{0};
    
    std::thread encoding_thread_;
    std::vector<std::thread> stream_threads_;
    
    // For 192k mode: dual streams with interleaving
    struct StreamContext {
        uint8_t stream_id;
        uint32_t offset_samples;
        std::vector<float> buffer;
        bool is_interleaved;
        ADATChannel adat_channel;
    };
    
    std::vector<StreamContext> stream_contexts_;
    uint64_t start_time_us_;

public:
    /**
     * @brief Constructor
     * @param config Encoder configuration
     */
    explicit JELLIEEncoder(const Config& config);

    /**
     * @brief Destructor
     */
    ~JELLIEEncoder();

    /**
     * @brief Set the message output callback
     * @param callback Function to call when messages are ready
     */
    void setMessageCallback(MessageCallback callback);

    /**
     * @brief Start the encoder
     * @return True if started successfully
     */
    bool start();

    /**
     * @brief Stop the encoder
     */
    void stop();

    /**
     * @brief Check if encoder is running
     * @return True if running
     */
    bool isRunning() const { return is_running_.load(); }

    /**
     * @brief Process audio input samples
     * @param samples Audio samples (mono, 32-bit float)
     * @param timestamp_us Timestamp in microseconds
     * @return True if processed successfully
     */
    bool processAudio(const std::vector<float>& samples, uint64_t timestamp_us);

    /**
     * @brief Process audio input samples with automatic timestamping
     * @param samples Audio samples (mono, 32-bit float)
     * @return True if processed successfully
     */
    bool processAudio(const std::vector<float>& samples);

    /**
     * @brief Update encoder configuration
     * @param config New configuration
     * @return True if updated successfully (may require restart)
     */
    bool updateConfig(const Config& config);

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Get current sequence number
     * @return Current sequence number
     */
    uint64_t getSequenceNumber() const { return sequence_number_.load(); }

    /**
     * @brief Get encoding statistics
     */
    struct Statistics {
        uint64_t messages_sent = 0;
        uint64_t samples_processed = 0;
        uint64_t encoding_errors = 0;
        double average_encoding_time_us = 0.0;
        double cpu_usage_percent = 0.0;
        uint32_t buffer_fill_percent = 0;
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
     * @brief Enable/disable 192kHz mode
     * @param enable True to enable 192k mode
     * @return True if mode change was successful
     */
    bool set192kMode(bool enable);

    /**
     * @brief Enable/disable ADAT channel mapping
     * @param enable True to enable ADAT mapping
     * @return True if mapping change was successful
     */
    bool setADATMapping(bool enable);

    /**
     * @brief Set redundancy level
     * @param level Redundancy level (1-4)
     * @return True if level was set successfully
     */
    bool setRedundancyLevel(uint8_t level);

    /**
     * @brief Flush any pending audio data
     * Forces encoding of partial buffers
     */
    void flush();

    /**
     * @brief Create a stream info message for current configuration
     * @return Stream info message
     */
    JDATMessage createStreamInfoMessage() const;

private:
    /**
     * @brief Initialize stream contexts based on configuration
     */
    void initializeStreamContexts();

    /**
     * @brief Main encoding thread function
     */
    void encodingThreadFunction();

    /**
     * @brief Encode a single frame of audio data
     * @param stream_context Stream context for this frame
     * @param samples Audio samples
     * @param timestamp_us Timestamp
     */
    void encodeFrame(const StreamContext& stream_context,
                    const std::vector<float>& samples,
                    uint64_t timestamp_us);

    /**
     * @brief Prepare samples for 192k mode (interleaving)
     * @param input_samples Input audio samples
     * @param stream_id Stream identifier
     * @return Processed samples for this stream
     */
    std::vector<float> prepare192kSamples(const std::vector<float>& input_samples,
                                         uint8_t stream_id);

    /**
     * @brief Apply audio quality settings to samples
     * @param samples Audio samples to process
     * @param quality Quality level to apply
     * @return Processed samples
     */
    std::vector<float> applyQualitySettings(const std::vector<float>& samples,
                                          AudioQuality quality);

    /**
     * @brief Generate redundant streams
     * @param primary_samples Primary audio samples
     * @param redundancy_level Number of redundant streams to create
     * @return Vector of redundant sample arrays
     */
    std::vector<std::vector<float>> generateRedundantStreams(
        const std::vector<float>& primary_samples,
        uint8_t redundancy_level);

    /**
     * @brief Update encoding statistics
     * @param encoding_time_us Time taken for encoding
     * @param samples_count Number of samples processed
     */
    void updateStatistics(uint64_t encoding_time_us, uint32_t samples_count);

    // Statistics tracking
    mutable std::mutex stats_mutex_;
    Statistics stats_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
};

/**
 * @brief Factory function to create a JELLIE encoder with default configuration
 * @param sample_rate Desired sample rate
 * @param session_id Session identifier
 * @return Unique pointer to JELLIEEncoder
 */
std::unique_ptr<JELLIEEncoder> createDefaultEncoder(
    SampleRate sample_rate = SampleRate::SR_96000,
    const std::string& session_id = "");

/**
 * @brief Factory function to create a JELLIE encoder for 192k mode
 * @param session_id Session identifier
 * @return Unique pointer to JELLIEEncoder configured for 192k mode
 */
std::unique_ptr<JELLIEEncoder> create192kEncoder(const std::string& session_id = "");

} // namespace jdat 