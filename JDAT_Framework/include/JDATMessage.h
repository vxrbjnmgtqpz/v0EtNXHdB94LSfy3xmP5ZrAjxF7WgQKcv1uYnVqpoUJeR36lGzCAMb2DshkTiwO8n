#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <memory>
#include <array>
#include <mutex>
#include <nlohmann/json.hpp>

namespace jdat {

/**
 * @brief Audio sample rate constants
 */
enum class SampleRate : uint32_t {
    SR_48000 = 48000,
    SR_96000 = 96000,
    SR_192000 = 192000
};

/**
 * @brief ADAT channel mapping for redundancy strategy
 */
enum class ADATChannel : uint8_t {
    STREAM_1 = 0,  // Primary stream
    STREAM_2 = 1,  // Interleaved offset stream
    STREAM_3 = 2,  // Parity/redundancy stream
    STREAM_4 = 3   // Additional redundancy/prediction stream
};

/**
 * @brief Audio quality levels for adaptive streaming
 */
enum class AudioQuality : uint8_t {
    HIGH_PRECISION = 0,  // 32-bit float samples
    STANDARD = 1,        // 24-bit samples
    COMPRESSED = 2       // 16-bit samples with compression
};

/**
 * @brief Core JDAT message structure for audio streaming
 */
class JDATMessage {
public:
    /**
     * @brief Message types for JDAT protocol
     */
    enum class MessageType {
        AUDIO_DATA,
        HANDSHAKE,
        HEARTBEAT,
        ERROR,
        CONTROL,
        STREAM_INFO,
        ADAT_SYNC
    };

    /**
     * @brief Audio data payload structure
     */
    struct AudioData {
        std::vector<float> samples;     // 32-bit float audio samples
        uint32_t sample_rate;           // Sample rate (48k, 96k, 192k)
        uint8_t channel;                // Channel/stream identifier
        uint64_t timestamp_us;          // Microsecond timestamp
        uint32_t frame_size;            // Number of samples in this frame
        uint8_t redundancy_level;       // Redundancy factor (1-4)
        bool is_interleaved;            // True if this is an offset stream
        uint32_t offset_samples;        // Sample offset for interleaving
    };

    /**
     * @brief Stream information payload
     */
    struct StreamInfo {
        uint32_t stream_id;
        SampleRate sample_rate;
        AudioQuality quality;
        uint8_t total_streams;
        bool use_adat_mapping;
        uint32_t buffer_size_ms;
    };

    /**
     * @brief Control message payload
     */
    struct ControlData {
        std::string command;
        nlohmann::json parameters;
    };

private:
    MessageType type_;
    uint64_t sequence_number_;
    uint64_t timestamp_us_;
    std::string session_id_;
    std::string message_id_;
    
    // Payload data (only one will be active based on type)
    std::unique_ptr<AudioData> audio_data_;
    std::unique_ptr<StreamInfo> stream_info_;
    std::unique_ptr<ControlData> control_data_;
    std::string error_message_;

public:
    /**
     * @brief Default constructor
     */
    JDATMessage();

    /**
     * @brief Copy constructor
     */
    JDATMessage(const JDATMessage& other);

    /**
     * @brief Move constructor
     */
    JDATMessage(JDATMessage&& other) noexcept;

    /**
     * @brief Copy assignment operator
     */
    JDATMessage& operator=(const JDATMessage& other);

    /**
     * @brief Move assignment operator
     */
    JDATMessage& operator=(JDATMessage&& other) noexcept;

    /**
     * @brief Destructor
     */
    ~JDATMessage() = default;

    /**
     * @brief Create an audio data message
     * @param samples Audio sample data
     * @param sample_rate Sample rate
     * @param channel Channel/stream identifier
     * @param redundancy_level Redundancy factor
     * @param is_interleaved Whether this is an offset stream
     * @param offset_samples Sample offset for interleaving
     * @return JDATMessage instance
     */
    static JDATMessage createAudioMessage(
        const std::vector<float>& samples,
        uint32_t sample_rate,
        uint8_t channel = 0,
        uint8_t redundancy_level = 1,
        bool is_interleaved = false,
        uint32_t offset_samples = 0
    );

    /**
     * @brief Create a stream info message
     * @param stream_id Stream identifier
     * @param sample_rate Sample rate
     * @param quality Audio quality level
     * @param total_streams Number of streams
     * @param use_adat_mapping Whether to use ADAT channel mapping
     * @param buffer_size_ms Buffer size in milliseconds
     * @return JDATMessage instance
     */
    static JDATMessage createStreamInfoMessage(
        uint32_t stream_id,
        SampleRate sample_rate,
        AudioQuality quality,
        uint8_t total_streams = 1,
        bool use_adat_mapping = false,
        uint32_t buffer_size_ms = 10
    );

    /**
     * @brief Create a control message
     * @param command Control command string
     * @param parameters Command parameters
     * @return JDATMessage instance
     */
    static JDATMessage createControlMessage(
        const std::string& command,
        const nlohmann::json& parameters = nlohmann::json::object()
    );

    /**
     * @brief Create an error message
     * @param error_msg Error description
     * @return JDATMessage instance
     */
    static JDATMessage createErrorMessage(const std::string& error_msg);

    /**
     * @brief Create a heartbeat message
     * @return JDATMessage instance
     */
    static JDATMessage createHeartbeatMessage();

    /**
     * @brief Serialize message to JSON
     * @return JSON representation
     */
    nlohmann::json toJSON() const;

    /**
     * @brief Deserialize message from JSON
     * @param json JSON data
     * @return True if successful
     */
    bool fromJSON(const nlohmann::json& json);

    /**
     * @brief Serialize message to JSON string
     * @return JSON string
     */
    std::string toString() const;

    /**
     * @brief Deserialize message from JSON string
     * @param json_str JSON string
     * @return True if successful
     */
    bool fromString(const std::string& json_str);

    /**
     * @brief Get message type
     * @return Message type
     */
    MessageType getType() const { return type_; }

    /**
     * @brief Get sequence number
     * @return Sequence number
     */
    uint64_t getSequenceNumber() const { return sequence_number_; }

    /**
     * @brief Set sequence number
     * @param seq Sequence number
     */
    void setSequenceNumber(uint64_t seq) { sequence_number_ = seq; }

    /**
     * @brief Get timestamp
     * @return Timestamp in microseconds
     */
    uint64_t getTimestamp() const { return timestamp_us_; }

    /**
     * @brief Set timestamp
     * @param timestamp_us Timestamp in microseconds
     */
    void setTimestamp(uint64_t timestamp_us) { timestamp_us_ = timestamp_us; }

    /**
     * @brief Get session ID
     * @return Session ID
     */
    const std::string& getSessionId() const { return session_id_; }

    /**
     * @brief Set session ID
     * @param session_id Session ID
     */
    void setSessionId(const std::string& session_id) { session_id_ = session_id; }

    /**
     * @brief Get message ID
     * @return Message ID
     */
    const std::string& getMessageId() const { return message_id_; }

    /**
     * @brief Get audio data (if message is audio type)
     * @return Audio data pointer or nullptr
     */
    const AudioData* getAudioData() const { return audio_data_.get(); }

    /**
     * @brief Get stream info (if message is stream info type)
     * @return Stream info pointer or nullptr
     */
    const StreamInfo* getStreamInfo() const { return stream_info_.get(); }

    /**
     * @brief Get control data (if message is control type)
     * @return Control data pointer or nullptr
     */
    const ControlData* getControlData() const { return control_data_.get(); }

    /**
     * @brief Get error message (if message is error type)
     * @return Error message
     */
    const std::string& getErrorMessage() const { return error_message_; }

    /**
     * @brief Validate message integrity
     * @return True if message is valid
     */
    bool isValid() const;

    /**
     * @brief Get estimated message size in bytes
     * @return Estimated size
     */
    size_t getEstimatedSize() const;

    /**
     * @brief Check if message contains audio data
     * @return True if audio data message
     */
    bool hasAudioData() const { return audio_data_ != nullptr; }

    /**
     * @brief Generate unique message ID
     * @return Unique message ID
     */
    static std::string generateMessageId();

    /**
     * @brief Get current timestamp in microseconds
     * @return Current timestamp
     */
    static uint64_t getCurrentTimestamp();

private:
    /**
     * @brief Initialize message with common fields
     */
    void initializeMessage(MessageType type);

    /**
     * @brief Copy audio data from another message
     */
    void copyAudioData(const AudioData& other);

    /**
     * @brief Copy stream info from another message
     */
    void copyStreamInfo(const StreamInfo& other);

    /**
     * @brief Copy control data from another message
     */
    void copyControlData(const ControlData& other);
};

/**
 * @brief Utility functions for JDAT messages
 */
namespace utils {
    /**
     * @brief Convert sample rate enum to integer
     */
    uint32_t sampleRateToInt(SampleRate rate);

    /**
     * @brief Convert integer to sample rate enum
     */
    SampleRate intToSampleRate(uint32_t rate);

    /**
     * @brief Convert ADAT channel enum to integer
     */
    uint8_t adatChannelToInt(ADATChannel channel);

    /**
     * @brief Convert integer to ADAT channel enum
     */
    ADATChannel intToADATChannel(uint8_t channel);

    /**
     * @brief Calculate buffer size in samples for given duration
     */
    uint32_t calculateBufferSize(SampleRate rate, uint32_t duration_ms);

    /**
     * @brief Calculate interleaved offset for 192k simulation
     */
    uint32_t calculateInterleavedOffset(SampleRate base_rate, uint8_t stream_index);
}

} // namespace jdat 