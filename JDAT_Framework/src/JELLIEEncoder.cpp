#include "JELLIEEncoder.h"
#include "WaveformPredictor.h"
#include <cstring>
#include <algorithm>
#include <random>

namespace jdat {

JELLIEEncoder::JELLIEEncoder(const Config& config) 
    : config_(config)
    , is_running_(false)
    , frame_counter_(0)
    , timestamp_offset_(0) {
    
    // Initialize buffer manager
    AudioBufferManager::Config buffer_config;
    buffer_config.sample_rate = static_cast<uint32_t>(config_.sample_rate);
    buffer_config.buffer_size_ms = config_.buffer_size_ms;
    buffer_config.frame_size_samples = config_.frame_size_samples;
    
    buffer_manager_ = std::make_unique<AudioBufferManager>(buffer_config);
    
    // Initialize ADAT simulator for 4-stream redundancy
    adat_simulator_ = std::make_unique<ADATSimulator>(config_.redundancy_level);
    
    // Generate session UUID if not provided
    if (config_.session_id.empty()) {
        generateSessionId();
    }
    
    // Initialize timing
    resetTiming();
    
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
    
    // Start real-time processing thread
    processing_thread_ = std::thread([this]() {
        processingLoop();
    });
    
    return true;
}

void JELLIEEncoder::stop() {
    if (!is_running_) {
        return;
    }
    
    is_running_ = false;
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

std::vector<JDATMessage> JELLIEEncoder::encodeFrame(const std::vector<float>& audio_samples) {
    if (audio_samples.size() != config_.frame_size_samples) {
        throw std::invalid_argument("Audio frame size mismatch");
    }
    
    std::vector<JDATMessage> messages;
    messages.reserve(4); // ADAT-inspired 4 streams
    
    // Get current timestamp
    uint64_t timestamp = getCurrentTimestamp();
    
    // Split audio into ADAT-inspired streams
    auto streams = adat_simulator_->splitToStreams(audio_samples);
    
    // Create JDAT messages for each stream
    for (size_t stream_idx = 0; stream_idx < streams.size(); ++stream_idx) {
        JDATMessage message;
        message.header.version = 2;
        message.header.message_type = MessageType::AUDIO_DATA;
        message.header.stream_id = static_cast<uint8_t>(stream_idx);
        message.header.frame_number = frame_counter_;
        message.header.timestamp_us = timestamp;
        message.header.sample_rate = config_.sample_rate;
        message.header.quality = config_.quality;
        message.header.redundancy = config_.redundancy_level;
        
        // Session identification
        std::strncpy(message.header.session_id, config_.session_id.c_str(), 
                    sizeof(message.header.session_id) - 1);
        
        // Encode audio data
        encodeAudioData(streams[stream_idx], message);
        
        // Calculate checksum
        message.header.checksum = calculateChecksum(message);
        
        messages.push_back(std::move(message));
    }
    
    frame_counter_++;
    
    // Call output callback if set
    if (message_callback_) {
        for (const auto& msg : messages) {
            message_callback_(msg);
        }
    }
    
    return messages;
}

void JELLIEEncoder::setAudioCallback(AudioCallback callback) {
    audio_callback_ = std::move(callback);
}

void JELLIEEncoder::setMessageCallback(MessageCallback callback) {
    message_callback_ = std::move(callback);
}

void JELLIEEncoder::enableReferenceRecording(bool enable) {
    reference_recording_enabled_ = enable;
    if (enable) {
        reference_samples_.clear();
        reference_samples_.reserve(static_cast<uint32_t>(config_.sample_rate) * 60); // 1 minute buffer
    }
}

bool JELLIEEncoder::enableGPUAcceleration(const std::string& shader_path) {
    // GPU acceleration implementation would go here
    // For now, return false to indicate CPU processing
    gpu_enabled_ = false;
    return false;
}

void JELLIEEncoder::processingLoop() {
    const auto frame_duration = std::chrono::microseconds(
        (config_.frame_size_samples * 1000000) / static_cast<uint32_t>(config_.sample_rate)
    );
    
    while (is_running_) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Get audio from buffer manager
        auto audio_data = buffer_manager_->getNextFrame();
        if (!audio_data.empty()) {
            // Call audio callback if set
            if (audio_callback_) {
                audio_callback_(audio_data, getCurrentTimestamp());
            }
            
            // Encode frame
            try {
                auto messages = encodeFrame(audio_data);
                
                // Store reference data if enabled
                if (reference_recording_enabled_) {
                    std::lock_guard<std::mutex> lock(reference_mutex_);
                    reference_samples_.insert(reference_samples_.end(), 
                                            audio_data.begin(), audio_data.end());
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

void JELLIEEncoder::encodeAudioData(const std::vector<float>& samples, JDATMessage& message) {
    // Clear previous data
    message.audio_data.samples.clear();
    message.audio_data.samples.reserve(samples.size());
    
    // Convert float samples to 24-bit integers for transmission
    for (float sample : samples) {
        // Clamp to [-1.0, 1.0] range
        sample = std::max(-1.0f, std::min(1.0f, sample));
        
        // Convert to 24-bit integer
        int32_t sample_24bit = static_cast<int32_t>(sample * 8388607.0f); // 2^23 - 1
        
        // Ensure 24-bit range
        sample_24bit = std::max(-8388608, std::min(8388607, sample_24bit));
        
        message.audio_data.samples.push_back(sample_24bit);
    }
    
    // Set metadata
    message.audio_data.num_samples = static_cast<uint32_t>(samples.size());
    message.audio_data.bit_depth = 24;
    message.audio_data.is_compressed = false;
    message.audio_data.compression_ratio = 1.0f;
}

uint32_t JELLIEEncoder::calculateChecksum(const JDATMessage& message) const {
    // Simple CRC32-like checksum
    uint32_t checksum = 0;
    
    // Include header data (excluding checksum field)
    const uint8_t* header_data = reinterpret_cast<const uint8_t*>(&message.header);
    for (size_t i = 0; i < sizeof(JDATHeader) - sizeof(uint32_t); ++i) {
        checksum = ((checksum << 1) | (checksum >> 31)) ^ header_data[i];
    }
    
    // Include audio data
    for (int32_t sample : message.audio_data.samples) {
        const uint8_t* sample_data = reinterpret_cast<const uint8_t*>(&sample);
        for (size_t i = 0; i < sizeof(int32_t); ++i) {
            checksum = ((checksum << 1) | (checksum >> 31)) ^ sample_data[i];
        }
    }
    
    return checksum;
}

uint64_t JELLIEEncoder::getCurrentTimestamp() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() + timestamp_offset_;
}

void JELLIEEncoder::generateSessionId() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    std::string session_id = "jellie-";
    for (int i = 0; i < 8; ++i) {
        session_id += "0123456789abcdef"[dis(gen)];
    }
    
    config_.session_id = session_id;
}

void JELLIEEncoder::resetTiming() {
    frame_counter_ = 0;
    timestamp_offset_ = 0;
}

} // namespace jdat
