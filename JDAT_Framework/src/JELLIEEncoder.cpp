#include "../include/JELLIEEncoder.h"
#include "../include/WaveformPredictor.h"
#include <cstring>
#include <algorithm>
#include <random>
#include <iostream>
#include <chrono>

namespace jdat {

JELLIEEncoder::JELLIEEncoder(const Config& config) 
    : config_(config)
    , start_time_us_(JDATMessage::getCurrentTimestamp()) {
    
    // Initialize buffer manager
    AudioBufferManager::Config buffer_config;
    buffer_config.sample_rate = static_cast<uint32_t>(config_.sample_rate);
    buffer_config.buffer_size_ms = config_.buffer_size_ms;
    
    buffer_manager_ = std::make_unique<AudioBufferManager>(buffer_config);
    
    // Generate session UUID if not provided
    if (config_.session_id.empty()) {
        // Use const_cast to modify the config - this is safe in constructor
        const_cast<Config&>(config_).session_id = JDATMessage::generateMessageId();
    }
    
    std::cout << "JELLIE Encoder initialized:\n";
    std::cout << "  Sample Rate: " << static_cast<uint32_t>(config_.sample_rate) << " Hz\n";
    std::cout << "  Frame Size: " << config_.frame_size_samples << " samples\n";
    std::cout << "  Redundancy: " << static_cast<int>(config_.redundancy_level) << " streams\n";
    std::cout << "  Session ID: " << config_.session_id << "\n";
}

JELLIEEncoder::~JELLIEEncoder() {
    stop();
}

bool JELLIEEncoder::start() {
    if (is_running_) {
        return false;
    }
    
    is_running_ = true;
    
    // Start encoding thread using the header-defined member name
    encoding_thread_ = std::thread([this]() {
        encodingThreadFunction();
    });
    
    return true;
}

void JELLIEEncoder::stop() {
    if (!is_running_) {
        return;
    }
    
    is_running_ = false;
    
    if (encoding_thread_.joinable()) {
        encoding_thread_.join();
    }
}

bool JELLIEEncoder::processAudio(const std::vector<float>& samples, uint64_t timestamp_us) {
    if (!is_running_) {
        return false;
    }
    
    if (samples.size() != config_.frame_size_samples) {
        return false; // Frame size mismatch
    }
    
    // Store samples in buffer manager for processing
    return buffer_manager_->writeSamples(samples, timestamp_us);
}

bool JELLIEEncoder::processAudio(const std::vector<float>& samples) {
    return processAudio(samples, JDATMessage::getCurrentTimestamp());
}

void JELLIEEncoder::setMessageCallback(MessageCallback callback) {
    message_callback_ = std::move(callback);
}

JDATMessage JELLIEEncoder::createStreamInfoMessage() const {
    return JDATMessage::createStreamInfoMessage(
        0, // stream_id
        config_.sample_rate,
        config_.quality,
        config_.redundancy_level,
        config_.enable_adat_mapping,
        config_.buffer_size_ms
    );
}

void JELLIEEncoder::encodingThreadFunction() {
    const auto frame_duration = std::chrono::microseconds(
        (config_.frame_size_samples * 1000000) / static_cast<uint32_t>(config_.sample_rate)
    );
    
    while (is_running_) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Try to read audio from buffer manager
        std::vector<float> samples;
        uint64_t timestamp_us;
        uint8_t channel;
        
        if (buffer_manager_->readSamples(samples, timestamp_us, channel)) {
            try {
                // Generate ADAT-inspired streams for redundancy
                auto streams = generateRedundantStreams(samples, config_.redundancy_level);
                
                // Create JDAT messages for each stream
                for (size_t stream_idx = 0; stream_idx < streams.size(); ++stream_idx) {
                    JDATMessage message = JDATMessage::createAudioMessage(
                        streams[stream_idx],
                        static_cast<uint32_t>(config_.sample_rate),
                        static_cast<uint8_t>(stream_idx),
                        config_.redundancy_level,
                        stream_idx % 2 == 1,  // is_interleaved for odd streams
                        stream_idx * config_.frame_size_samples / 4  // offset for ADAT strategy
                    );
                    
                    // Set sequence number and session
                    message.setSequenceNumber(sequence_number_.fetch_add(1));
                    message.setSessionId(config_.session_id);
                    message.setTimestamp(timestamp_us);
                    
                    // Send message via callback
                    if (message_callback_) {
                        message_callback_(message);
                    }
                }
                
            } catch (const std::exception& e) {
                std::cerr << "JELLIE encoding error: " << e.what() << std::endl;
            }
        }
        
        // Maintain real-time scheduling
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = end_time - start_time;
        
        if (elapsed < frame_duration) {
            std::this_thread::sleep_for(frame_duration - elapsed);
        }
    }
}

std::vector<std::vector<float>> JELLIEEncoder::generateRedundantStreams(
    const std::vector<float>& primary_samples,
    uint8_t redundancy_level) {
    
    std::vector<std::vector<float>> streams;
    
    // ADAT-inspired 4-stream strategy
    const size_t samples_per_stream = primary_samples.size() / 2;
    
    // Stream 0: Even samples (0, 2, 4, ...)
    std::vector<float> stream0;
    stream0.reserve(samples_per_stream);
    for (size_t i = 0; i < primary_samples.size(); i += 2) {
        stream0.push_back(primary_samples[i]);
    }
    streams.push_back(stream0);
    
    // Stream 1: Odd samples (1, 3, 5, ...)
    std::vector<float> stream1;
    stream1.reserve(samples_per_stream);
    for (size_t i = 1; i < primary_samples.size(); i += 2) {
        stream1.push_back(primary_samples[i]);
    }
    streams.push_back(stream1);
    
    // Generate redundancy streams if requested
    if (redundancy_level >= 2) {
        // Stream 2: Redundancy A (weighted combination)
        std::vector<float> stream2;
        stream2.reserve(samples_per_stream);
        for (size_t i = 0; i < std::min(stream0.size(), stream1.size()); ++i) {
            float redundancy_sample = (stream0[i] + stream1[i]) * 0.5f + (stream0[i] - stream1[i]) * 0.3f;
            stream2.push_back(redundancy_sample);
        }
        streams.push_back(stream2);
    }
    
    if (redundancy_level >= 3) {
        // Stream 3: Redundancy B (alternative parity)
        std::vector<float> stream3;
        stream3.reserve(samples_per_stream);
        for (size_t i = 0; i < std::min(stream0.size(), stream1.size()); ++i) {
            float redundancy_sample = (stream0[i] - stream1[i]) * 0.7f + (stream0[i] + stream1[i]) * 0.1f;
            stream3.push_back(redundancy_sample);
        }
        streams.push_back(stream3);
    }
    
    return streams;
}

JELLIEEncoder::Statistics JELLIEEncoder::getStatistics() const {
    Statistics stats;
    stats.messages_sent = sequence_number_.load();
    stats.samples_processed = stats.messages_sent * config_.frame_size_samples;
    // Additional statistics would be calculated here
    return stats;
}

void JELLIEEncoder::resetStatistics() {
    sequence_number_.store(0);
}

bool JELLIEEncoder::updateConfig(const Config& config) {
    // For now, require restart to change config
    if (is_running_) {
        return false;
    }
    
    config_ = config;
    return true;
}

bool JELLIEEncoder::set192kMode(bool enable) {
    config_.enable_192k_mode = enable;
    return true;
}

bool JELLIEEncoder::setADATMapping(bool enable) {
    config_.enable_adat_mapping = enable;
    return true;
}

bool JELLIEEncoder::setRedundancyLevel(uint8_t level) {
    if (level > 4) level = 4; // Clamp to maximum
    config_.redundancy_level = level;
    return true;
}

void JELLIEEncoder::flush() {
    // Force processing of any pending audio data
    if (buffer_manager_) {
        // This would flush the buffer manager if it had that capability
        // For now, just wait a short time for pending data to process
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

} // namespace jdat
