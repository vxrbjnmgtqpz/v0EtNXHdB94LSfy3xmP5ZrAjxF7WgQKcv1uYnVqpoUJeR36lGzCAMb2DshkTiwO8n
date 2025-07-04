#include "JELLIEDecoder.h"
#include "WaveformPredictor.h"
#include <algorithm>
#include <cstring>

namespace jdat {

JELLIEDecoder::JELLIEDecoder(const Config& config)
    : config_(config)
    , is_running_(false)
    , expected_frame_number_(0)
    , last_valid_timestamp_(0) {
    
    // Initialize buffer manager for output
    AudioBufferManager::Config buffer_config;
    buffer_config.sample_rate = static_cast<uint32_t>(config_.expected_sample_rate);
    buffer_config.buffer_size_ms = config_.buffer_size_ms;
    buffer_config.frame_size_samples = config_.expected_frame_size;
    
    buffer_manager_ = std::make_unique<AudioBufferManager>(buffer_config);
    
    // Initialize ADAT simulator for 4-stream reconstruction
    adat_simulator_ = std::make_unique<ADATSimulator>(config_.redundancy_level);
    
    // Initialize PNBTR predictor if enabled
    if (config_.enable_pnbtr) {
        WaveformPredictor::Config predictor_config;
        predictor_config.sample_rate = static_cast<uint32_t>(config_.expected_sample_rate);
        predictor_config.prediction_window_ms = 50; // 50ms lookahead
        predictor_config.learning_rate = 0.001f;
        
        pnbtr_predictor_ = std::make_unique<WaveformPredictor>(predictor_config);
    }
    
    // Initialize stream tracking
    received_streams_.resize(4, false);
    stream_messages_.resize(4);
    
    std::cout << "JELLIE Decoder initialized:\n";
    std::cout << "  Expected Sample Rate: " << static_cast<uint32_t>(config_.expected_sample_rate) << " Hz\n";
    std::cout << "  Frame Size: " << config_.expected_frame_size << " samples\n";
    std::cout << "  PNBTR Enabled: " << (config_.enable_pnbtr ? "Yes" : "No") << "\n";
}

JELLIEDecoder::~JELLIEDecoder() {
    stop();
}

bool JELLIEDecoder::start() {
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

void JELLIEDecoder::stop() {
    if (!is_running_) {
        return;
    }
    
    is_running_ = false;
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

std::vector<float> JELLIEDecoder::decodeMessages(const std::vector<JDATMessage>& messages) {
    // Reset stream tracking for this frame
    std::fill(received_streams_.begin(), received_streams_.end(), false);
    
    // Process each message
    for (const auto& message : messages) {
        if (!validateMessage(message)) {
            continue;
        }
        
        // Store message for stream reconstruction
        uint8_t stream_id = message.header.stream_id;
        if (stream_id < stream_messages_.size()) {
            stream_messages_[stream_id] = message;
            received_streams_[stream_id] = true;
        }
    }
    
    // Reconstruct audio from available streams
    return reconstructAudio();
}

bool JELLIEDecoder::processMessage(const JDATMessage& message) {
    std::lock_guard<std::mutex> lock(processing_mutex_);
    
    if (!validateMessage(message)) {
        return false;
    }
    
    // Add to processing queue
    message_queue_.push(message);
    
    return true;
}

void JELLIEDecoder::setOutputCallback(OutputCallback callback) {
    output_callback_ = std::move(callback);
}

bool JELLIEDecoder::enableGPUAcceleration(const std::string& shader_path) {
    // GPU acceleration implementation would go here
    // For now, return false to indicate CPU processing
    gpu_enabled_ = false;
    return false;
}

void JELLIEDecoder::processingLoop() {
    const auto frame_duration = std::chrono::microseconds(
        (config_.expected_frame_size * 1000000) / static_cast<uint32_t>(config_.expected_sample_rate)
    );
    
    while (is_running_) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process queued messages
        std::vector<JDATMessage> frame_messages;
        {
            std::lock_guard<std::mutex> lock(processing_mutex_);
            
            // Collect all messages for current frame
            while (!message_queue_.empty()) {
                auto msg = message_queue_.front();
                message_queue_.pop();
                
                if (msg.header.frame_number == expected_frame_number_) {
                    frame_messages.push_back(msg);
                }
            }
        }
        
        // Decode frame if we have messages
        if (!frame_messages.empty()) {
            try {
                auto audio_output = decodeMessages(frame_messages);
                
                // Call output callback if set
                if (output_callback_ && !audio_output.empty()) {
                    output_callback_(audio_output, getCurrentTimestamp());
                }
                
                // Store in buffer manager
                buffer_manager_->addFrame(audio_output);
                
            } catch (const std::exception& e) {
                std::cerr << "JELLIE decoding error: " << e.what() << std::endl;
            }
        }
        
        expected_frame_number_++;
        
        // Maintain real-time scheduling
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = end_time - start_time;
        
        if (elapsed < frame_duration) {
            std::this_thread::sleep_for(frame_duration - elapsed);
        }
    }
}

std::vector<float> JELLIEDecoder::reconstructAudio() {
    // Check which streams we received
    int received_count = std::count(received_streams_.begin(), received_streams_.end(), true);
    
    if (received_count == 0) {
        // No streams received - use PNBTR prediction if available
        return handleCompletePacketLoss();
    }
    
    if (received_count >= 2) {
        // We have enough streams for redundancy reconstruction
        return reconstructFromRedundancy();
    }
    
    // Partial loss - try to reconstruct what we can
    return reconstructPartial();
}

std::vector<float> JELLIEDecoder::reconstructFromRedundancy() {
    std::vector<std::vector<float>> streams;
    streams.reserve(4);
    
    // Decode available streams
    for (size_t i = 0; i < stream_messages_.size(); ++i) {
        if (received_streams_[i]) {
            auto stream_samples = decodeAudioData(stream_messages_[i]);
            streams.push_back(std::move(stream_samples));
        } else {
            // Create empty stream for missing data
            streams.emplace_back(config_.expected_frame_size / 2, 0.0f);
        }
    }
    
    // Use ADAT simulator to reconstruct original audio
    return adat_simulator_->reconstructFromStreams(streams);
}

std::vector<float> JELLIEDecoder::reconstructPartial() {
    // Try to reconstruct from whatever streams we have
    std::vector<std::vector<float>> available_streams;
    
    for (size_t i = 0; i < stream_messages_.size(); ++i) {
        if (received_streams_[i]) {
            auto stream_samples = decodeAudioData(stream_messages_[i]);
            available_streams.push_back(std::move(stream_samples));
        }
    }
    
    if (available_streams.empty()) {
        return handleCompletePacketLoss();
    }
    
    // Use first available stream as base, fill gaps with PNBTR
    auto base_audio = available_streams[0];
    
    if (config_.enable_pnbtr && pnbtr_predictor_) {
        // Use PNBTR to predict missing samples
        base_audio = pnbtr_predictor_->predictMissingSamples(base_audio, last_valid_samples_);
    }
    
    // Store for next prediction
    last_valid_samples_ = base_audio;
    
    return base_audio;
}

std::vector<float> JELLIEDecoder::handleCompletePacketLoss() {
    std::vector<float> predicted_audio(config_.expected_frame_size, 0.0f);
    
    if (config_.enable_pnbtr && pnbtr_predictor_ && !last_valid_samples_.empty()) {
        // Use PNBTR to predict entire frame
        predicted_audio = pnbtr_predictor_->predictNextFrame(last_valid_samples_);
        
        // Update statistics
        prediction_stats_.total_predictions++;
        prediction_stats_.complete_frame_predictions++;
    } else {
        // Fallback: repeat last frame with fade
        if (!last_valid_samples_.empty()) {
            predicted_audio = last_valid_samples_;
            
            // Apply fade to avoid clicks
            for (size_t i = 0; i < predicted_audio.size(); ++i) {
                float fade = 1.0f - (static_cast<float>(i) / predicted_audio.size()) * 0.5f;
                predicted_audio[i] *= fade;
            }
        }
    }
    
    return predicted_audio;
}

std::vector<float> JELLIEDecoder::decodeAudioData(const JDATMessage& message) {
    std::vector<float> samples;
    samples.reserve(message.audio_data.num_samples);
    
    // Convert 24-bit integers back to float
    for (int32_t sample_24bit : message.audio_data.samples) {
        // Convert from 24-bit to float [-1.0, 1.0]
        float sample = static_cast<float>(sample_24bit) / 8388607.0f; // 2^23 - 1
        
        // Clamp to valid range
        sample = std::max(-1.0f, std::min(1.0f, sample));
        
        samples.push_back(sample);
    }
    
    return samples;
}

bool JELLIEDecoder::validateMessage(const JDATMessage& message) const {
    // Check version
    if (message.header.version != 2) {
        return false;
    }
    
    // Check message type
    if (message.header.message_type != MessageType::AUDIO_DATA) {
        return false;
    }
    
    // Check stream ID
    if (message.header.stream_id >= 4) {
        return false;
    }
    
    // Check sample rate compatibility
    if (message.header.sample_rate != config_.expected_sample_rate) {
        return false;
    }
    
    // Verify checksum
    uint32_t calculated_checksum = calculateChecksum(message);
    if (calculated_checksum != message.header.checksum) {
        return false;
    }
    
    return true;
}

uint32_t JELLIEDecoder::calculateChecksum(const JDATMessage& message) const {
    // Same checksum algorithm as encoder
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

uint64_t JELLIEDecoder::getCurrentTimestamp() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

} // namespace jdat
