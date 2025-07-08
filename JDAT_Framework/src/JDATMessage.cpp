#include "../include/JDATMessage.h"
#include <sstream>
#include <random>
#include <iomanip>
#include <chrono>
#include <iostream>

namespace jdat {

// JDATMessage Implementation

JDATMessage::JDATMessage() 
    : type_(MessageType::AUDIO_DATA)
    , sequence_number_(0)
    , timestamp_us_(0) {
    initializeMessage(MessageType::AUDIO_DATA);
}

JDATMessage::JDATMessage(const JDATMessage& other) 
    : type_(other.type_)
    , sequence_number_(other.sequence_number_)
    , timestamp_us_(other.timestamp_us_)
    , session_id_(other.session_id_)
    , message_id_(other.message_id_)
    , error_message_(other.error_message_) {
    
    // Deep copy payload data based on type
    if (other.audio_data_) {
        copyAudioData(*other.audio_data_);
    }
    if (other.stream_info_) {
        copyStreamInfo(*other.stream_info_);
    }
    if (other.control_data_) {
        copyControlData(*other.control_data_);
    }
}

JDATMessage::JDATMessage(JDATMessage&& other) noexcept
    : type_(other.type_)
    , sequence_number_(other.sequence_number_)
    , timestamp_us_(other.timestamp_us_)
    , session_id_(std::move(other.session_id_))
    , message_id_(std::move(other.message_id_))
    , audio_data_(std::move(other.audio_data_))
    , stream_info_(std::move(other.stream_info_))
    , control_data_(std::move(other.control_data_))
    , error_message_(std::move(other.error_message_)) {
}

JDATMessage& JDATMessage::operator=(const JDATMessage& other) {
    if (this == &other) {
        return *this;
    }
    
    type_ = other.type_;
    sequence_number_ = other.sequence_number_;
    timestamp_us_ = other.timestamp_us_;
    session_id_ = other.session_id_;
    message_id_ = other.message_id_;
    error_message_ = other.error_message_;
    
    // Reset existing payload data
    audio_data_.reset();
    stream_info_.reset();
    control_data_.reset();
    
    // Deep copy payload data based on type
    if (other.audio_data_) {
        copyAudioData(*other.audio_data_);
    }
    if (other.stream_info_) {
        copyStreamInfo(*other.stream_info_);
    }
    if (other.control_data_) {
        copyControlData(*other.control_data_);
    }
    
    return *this;
}

JDATMessage& JDATMessage::operator=(JDATMessage&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    
    type_ = other.type_;
    sequence_number_ = other.sequence_number_;
    timestamp_us_ = other.timestamp_us_;
    session_id_ = std::move(other.session_id_);
    message_id_ = std::move(other.message_id_);
    audio_data_ = std::move(other.audio_data_);
    stream_info_ = std::move(other.stream_info_);
    control_data_ = std::move(other.control_data_);
    error_message_ = std::move(other.error_message_);
    
    return *this;
}

// Static factory methods

JDATMessage JDATMessage::createAudioMessage(
    const std::vector<float>& samples,
    uint32_t sample_rate,
    uint8_t channel,
    uint8_t redundancy_level,
    bool is_interleaved,
    uint32_t offset_samples) {
    
    JDATMessage message;
    message.initializeMessage(MessageType::AUDIO_DATA);
    
    message.audio_data_ = std::make_unique<AudioData>();
    message.audio_data_->samples = samples;
    message.audio_data_->sample_rate = sample_rate;
    message.audio_data_->channel = channel;
    message.audio_data_->timestamp_us = getCurrentTimestamp();
    message.audio_data_->frame_size = static_cast<uint32_t>(samples.size());
    message.audio_data_->redundancy_level = redundancy_level;
    message.audio_data_->is_interleaved = is_interleaved;
    message.audio_data_->offset_samples = offset_samples;
    
    return message;
}

JDATMessage JDATMessage::createStreamInfoMessage(
    uint32_t stream_id,
    SampleRate sample_rate,
    AudioQuality quality,
    uint8_t total_streams,
    bool use_adat_mapping,
    uint32_t buffer_size_ms) {
    
    JDATMessage message;
    message.initializeMessage(MessageType::STREAM_INFO);
    
    message.stream_info_ = std::make_unique<StreamInfo>();
    message.stream_info_->stream_id = stream_id;
    message.stream_info_->sample_rate = sample_rate;
    message.stream_info_->quality = quality;
    message.stream_info_->total_streams = total_streams;
    message.stream_info_->use_adat_mapping = use_adat_mapping;
    message.stream_info_->buffer_size_ms = buffer_size_ms;
    
    return message;
}

JDATMessage JDATMessage::createControlMessage(
    const std::string& command,
    const nlohmann::json& parameters) {
    
    JDATMessage message;
    message.initializeMessage(MessageType::CONTROL);
    
    message.control_data_ = std::make_unique<ControlData>();
    message.control_data_->command = command;
    message.control_data_->parameters = parameters;
    
    return message;
}

JDATMessage JDATMessage::createErrorMessage(const std::string& error_msg) {
    JDATMessage message;
    message.initializeMessage(MessageType::ERROR);
    message.error_message_ = error_msg;
    return message;
}

JDATMessage JDATMessage::createHeartbeatMessage() {
    JDATMessage message;
    message.initializeMessage(MessageType::HEARTBEAT);
    return message;
}

// JSON serialization methods

nlohmann::json JDATMessage::toJSON() const {
    nlohmann::json json;
    
    // Common fields
    json["type"] = static_cast<int>(type_);
    json["sequence_number"] = sequence_number_;
    json["timestamp_us"] = timestamp_us_;
    json["session_id"] = session_id_;
    json["message_id"] = message_id_;
    
    // Type-specific payload
    switch (type_) {
        case MessageType::AUDIO_DATA:
            if (audio_data_) {
                json["audio_data"] = {
                    {"samples", audio_data_->samples},
                    {"sample_rate", audio_data_->sample_rate},
                    {"channel", audio_data_->channel},
                    {"timestamp_us", audio_data_->timestamp_us},
                    {"frame_size", audio_data_->frame_size},
                    {"redundancy_level", audio_data_->redundancy_level},
                    {"is_interleaved", audio_data_->is_interleaved},
                    {"offset_samples", audio_data_->offset_samples}
                };
            }
            break;
            
        case MessageType::STREAM_INFO:
            if (stream_info_) {
                json["stream_info"] = {
                    {"stream_id", stream_info_->stream_id},
                    {"sample_rate", static_cast<uint32_t>(stream_info_->sample_rate)},
                    {"quality", static_cast<uint8_t>(stream_info_->quality)},
                    {"total_streams", stream_info_->total_streams},
                    {"use_adat_mapping", stream_info_->use_adat_mapping},
                    {"buffer_size_ms", stream_info_->buffer_size_ms}
                };
            }
            break;
            
        case MessageType::CONTROL:
            if (control_data_) {
                json["control_data"] = {
                    {"command", control_data_->command},
                    {"parameters", control_data_->parameters}
                };
            }
            break;
            
        case MessageType::ERROR:
            json["error_message"] = error_message_;
            break;
            
        case MessageType::HEARTBEAT:
        case MessageType::HANDSHAKE:
        case MessageType::ADAT_SYNC:
            // No additional payload for these types
            break;
    }
    
    return json;
}

std::string JDATMessage::toString() const {
    return toJSON().dump();
}

bool JDATMessage::fromJSON(const nlohmann::json& json) {
    try {
        // Parse common fields
        type_ = static_cast<MessageType>(json["type"].get<int>());
        sequence_number_ = json["sequence_number"].get<uint64_t>();
        timestamp_us_ = json["timestamp_us"].get<uint64_t>();
        session_id_ = json["session_id"].get<std::string>();
        message_id_ = json["message_id"].get<std::string>();
        
        // Reset existing payload data
        audio_data_.reset();
        stream_info_.reset();
        control_data_.reset();
        error_message_.clear();
        
        // Parse type-specific payload
        switch (type_) {
            case MessageType::AUDIO_DATA:
                if (json.contains("audio_data")) {
                    const auto& audio_json = json["audio_data"];
                    audio_data_ = std::make_unique<AudioData>();
                    audio_data_->samples = audio_json["samples"].get<std::vector<float>>();
                    audio_data_->sample_rate = audio_json["sample_rate"].get<uint32_t>();
                    audio_data_->channel = audio_json["channel"].get<uint8_t>();
                    audio_data_->timestamp_us = audio_json["timestamp_us"].get<uint64_t>();
                    audio_data_->frame_size = audio_json["frame_size"].get<uint32_t>();
                    audio_data_->redundancy_level = audio_json["redundancy_level"].get<uint8_t>();
                    audio_data_->is_interleaved = audio_json["is_interleaved"].get<bool>();
                    audio_data_->offset_samples = audio_json["offset_samples"].get<uint32_t>();
                }
                break;
                
            case MessageType::STREAM_INFO:
                if (json.contains("stream_info")) {
                    const auto& stream_json = json["stream_info"];
                    stream_info_ = std::make_unique<StreamInfo>();
                    stream_info_->stream_id = stream_json["stream_id"].get<uint32_t>();
                    stream_info_->sample_rate = static_cast<SampleRate>(stream_json["sample_rate"].get<uint32_t>());
                    stream_info_->quality = static_cast<AudioQuality>(stream_json["quality"].get<uint8_t>());
                    stream_info_->total_streams = stream_json["total_streams"].get<uint8_t>();
                    stream_info_->use_adat_mapping = stream_json["use_adat_mapping"].get<bool>();
                    stream_info_->buffer_size_ms = stream_json["buffer_size_ms"].get<uint32_t>();
                }
                break;
                
            case MessageType::CONTROL:
                if (json.contains("control_data")) {
                    const auto& control_json = json["control_data"];
                    control_data_ = std::make_unique<ControlData>();
                    control_data_->command = control_json["command"].get<std::string>();
                    control_data_->parameters = control_json["parameters"];
                }
                break;
                
            case MessageType::ERROR:
                if (json.contains("error_message")) {
                    error_message_ = json["error_message"].get<std::string>();
                }
                break;
                
            default:
                // No additional payload for other types
                break;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

bool JDATMessage::fromString(const std::string& json_str) {
    try {
        nlohmann::json json = nlohmann::json::parse(json_str);
        return fromJSON(json);
    } catch (const std::exception& e) {
        return false;
    }
}

// Note: getAudioData(), getStreamInfo(), getControlData() are inline in header

bool JDATMessage::isValid() const {
    if (message_id_.empty()) {
        return false;
    }
    
    // Validate type-specific requirements
    switch (type_) {
        case MessageType::AUDIO_DATA:
            return audio_data_ != nullptr && !audio_data_->samples.empty();
            
        case MessageType::STREAM_INFO:
            return stream_info_ != nullptr;
            
        case MessageType::CONTROL:
            return control_data_ != nullptr && !control_data_->command.empty();
            
        case MessageType::ERROR:
            return !error_message_.empty();
            
        default:
            return true; // HEARTBEAT, HANDSHAKE, ADAT_SYNC require no specific payload
    }
}

size_t JDATMessage::getEstimatedSize() const {
    size_t size = sizeof(JDATMessage);
    
    // Add variable-size components
    size += session_id_.size();
    size += message_id_.size();
    size += error_message_.size();
    
    if (audio_data_) {
        size += audio_data_->samples.size() * sizeof(float);
    }
    
    return size;
}

std::string JDATMessage::generateMessageId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    std::stringstream ss;
    ss << std::hex;
    
    for (int i = 0; i < 32; ++i) {
        ss << dis(gen);
        if (i == 7 || i == 11 || i == 15 || i == 19) {
            ss << "-";
        }
    }
    
    return ss.str();
}

uint64_t JDATMessage::getCurrentTimestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// Private helper methods

void JDATMessage::initializeMessage(MessageType type) {
    type_ = type;
    timestamp_us_ = getCurrentTimestamp();
    message_id_ = generateMessageId();
    sequence_number_ = 0;
}

void JDATMessage::copyAudioData(const AudioData& other) {
    audio_data_ = std::make_unique<AudioData>();
    *audio_data_ = other;
}

void JDATMessage::copyStreamInfo(const StreamInfo& other) {
    stream_info_ = std::make_unique<StreamInfo>();
    *stream_info_ = other;
}

void JDATMessage::copyControlData(const ControlData& other) {
    control_data_ = std::make_unique<ControlData>();
    *control_data_ = other;
}

// Utility functions

namespace utils {

uint32_t sampleRateToInt(SampleRate rate) {
    return static_cast<uint32_t>(rate);
}

SampleRate intToSampleRate(uint32_t rate) {
    switch (rate) {
        case 48000: return SampleRate::SR_48000;
        case 96000: return SampleRate::SR_96000;
        case 192000: return SampleRate::SR_192000;
        default: return SampleRate::SR_48000; // Default fallback
    }
}

uint8_t adatChannelToInt(ADATChannel channel) {
    return static_cast<uint8_t>(channel);
}

ADATChannel intToADATChannel(uint8_t channel) {
    if (channel > 3) channel = 0; // Clamp to valid range
    return static_cast<ADATChannel>(channel);
}

uint32_t calculateBufferSize(SampleRate rate, uint32_t duration_ms) {
    return (static_cast<uint32_t>(rate) * duration_ms) / 1000;
}

uint32_t calculateInterleavedOffset(SampleRate base_rate, uint8_t stream_index) {
    // For 192kHz simulation using ADAT strategy
    // Stream 0: samples 0, 4, 8, ... (even samples)
    // Stream 1: samples 2, 6, 10, ... (odd samples, offset)
    // Streams 2-3: Redundancy and parity data
    
    switch (stream_index) {
        case 0: return 0;        // Even samples
        case 1: return 1;        // Odd samples
        case 2: return 2;        // Redundancy A
        case 3: return 3;        // Redundancy B
        default: return 0;
    }
}

} // namespace utils

} // namespace jdat 