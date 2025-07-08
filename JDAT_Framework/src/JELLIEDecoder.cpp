#include "../include/JELLIEDecoder.h"
#include "../include/WaveformPredictor.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <chrono>
#include <queue>
#include <map>

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
    
    buffer_manager_ = std::make_unique<AudioBufferManager>(buffer_config);
    
    // Initialize PNBTR predictor if enabled
    if (config_.enable_pnbtr) {
        WaveformPredictor::Config predictor_config;
        predictor_config.sample_rate = static_cast<uint32_t>(config_.expected_sample_rate);
        predictor_config.prediction_window_ms = 50; // 50ms lookahead
        predictor_config.learning_rate = 0.001f;
        
        pnbtr_predictor_ = std::make_unique<WaveformPredictor>(predictor_config);
    }
    
    std::cout << "JELLIE Decoder initialized:\n";
    std::cout << "  Expected Sample Rate: " << static_cast<uint32_t>(config_.expected_sample_rate) << " Hz\n";
    std::cout << "  PNBTR Enabled: " << (config_.enable_pnbtr ? "Yes" : "No") << "\n";
    std::cout << "  Max Recovery Gap: " << config_.max_recovery_gap_ms << " ms\n";
}

JELLIEDecoder::~JELLIEDecoder() {
    stop();
}

bool JELLIEDecoder::start() {
    if (is_running_) {
        return false;
    }
    
    is_running_ = true;
    
    // Start decoding thread 
    decoding_thread_ = std::thread([this]() {
        decodingThreadFunction();
    });
    
    return true;
}

void JELLIEDecoder::stop() {
    if (!is_running_) {
        return;
    }
    
    is_running_ = false;
    
    // Notify condition variable to wake up processing thread
    queue_condition_.notify_all();
    
    if (decoding_thread_.joinable()) {
        decoding_thread_.join();
    }
}

bool JELLIEDecoder::processMessage(const JDATMessage& message) {
    if (!message.isValid() || message.getType() != JDATMessage::MessageType::AUDIO_DATA) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Create queued message with receive timestamp
    QueuedMessage queued_msg;
    queued_msg.message = message;
    queued_msg.receive_time_us = JDATMessage::getCurrentTimestamp();
    
    // Add to processing queue (simulated with simple storage)
    message_queue_.push(queued_msg);
    
    // Notify processing thread
    queue_condition_.notify_one();
    
    return true;
}

void JELLIEDecoder::setAudioOutputCallback(AudioOutputCallback callback) {
    audio_output_callback_ = std::move(callback);
}

void JELLIEDecoder::setPacketLossCallback(PacketLossCallback callback) {
    packet_loss_callback_ = std::move(callback);
}

void JELLIEDecoder::setRecoveryCallback(RecoveryCallback callback) {
    recovery_callback_ = std::move(callback);
}

void JELLIEDecoder::decodingThreadFunction() {
    std::map<uint64_t, std::vector<QueuedMessage>> frame_groups;
    uint64_t current_sequence = 0;
    
    while (is_running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for messages or stop signal
        queue_condition_.wait(lock, [this]() { 
            return !message_queue_.empty() || !is_running_; 
        });
        
        if (!is_running_) {
            break;
        }
        
        // Collect messages by sequence number
        while (!message_queue_.empty()) {
            QueuedMessage queued_msg = message_queue_.front();
            message_queue_.pop();
            
            uint64_t seq = queued_msg.message.getSequenceNumber();
            frame_groups[seq].push_back(queued_msg);
        }
        
        lock.unlock();
        
        // Process complete frames
        auto it = frame_groups.begin();
        while (it != frame_groups.end()) {
            uint64_t seq = it->first;
            auto& messages = it->second;
            
            // Check if we should process this frame
            if (seq <= current_sequence + 10) { // Allow some reordering
                try {
                    auto decoded_audio = reconstructAudioFromStreams(messages);
                    
                    if (!decoded_audio.empty()) {
                        // Send to callback
                        if (audio_output_callback_) {
                            audio_output_callback_(decoded_audio, JDATMessage::getCurrentTimestamp());
                        }
                        
                        // Store in buffer
                        buffer_manager_->writeSamples(decoded_audio);
                    }
                    
                    current_sequence = std::max(current_sequence, seq);
                    
                } catch (const std::exception& e) {
                    std::cerr << "JELLIE decoding error: " << e.what() << std::endl;
                }
                
                it = frame_groups.erase(it);
            } else {
                ++it;
            }
        }
        
        // Clean up old incomplete frames
        auto cleanup_it = frame_groups.begin();
        while (cleanup_it != frame_groups.end()) {
            if (cleanup_it->first < current_sequence - 5) {
                cleanup_it = frame_groups.erase(cleanup_it);
            } else {
                ++cleanup_it;
            }
        }
    }
}

std::vector<float> JELLIEDecoder::reconstructAudioFromStreams(
    const std::vector<QueuedMessage>& stream_messages) {
    
    if (stream_messages.empty()) {
        return handleCompletePacketLoss();
    }
    
    // Organize streams by channel ID
    std::map<uint8_t, std::vector<float>> streams;
    std::map<uint8_t, bool> received_streams;
    
    for (const auto& queued_msg : stream_messages) {
        const auto* audio_data = queued_msg.message.getAudioData();
        if (audio_data) {
            streams[audio_data->channel] = audio_data->samples;
            received_streams[audio_data->channel] = true;
        }
    }
    
    // Check what streams we have (ADAT 4-stream strategy)
    bool have_stream0 = received_streams[0]; // Even samples
    bool have_stream1 = received_streams[1]; // Odd samples  
    bool have_stream2 = received_streams[2]; // Redundancy A
    bool have_stream3 = received_streams[3]; // Redundancy B
    
    std::vector<float> reconstructed_audio;
    
    if (have_stream0 && have_stream1) {
        // Perfect case: we have both primary streams
        reconstructed_audio = combineEvenOddStreams(streams[0], streams[1]);
        
    } else if (have_stream0 && (have_stream2 || have_stream3)) {
        // Reconstruct odd stream from redundancy
        auto reconstructed_odd = reconstructOddFromRedundancy(
            streams[0], 
            have_stream2 ? streams[2] : streams[3],
            have_stream2 ? 2 : 3
        );
        reconstructed_audio = combineEvenOddStreams(streams[0], reconstructed_odd);
        
    } else if (have_stream1 && (have_stream2 || have_stream3)) {
        // Reconstruct even stream from redundancy
        auto reconstructed_even = reconstructEvenFromRedundancy(
            streams[1],
            have_stream2 ? streams[2] : streams[3],
            have_stream2 ? 2 : 3
        );
        reconstructed_audio = combineEvenOddStreams(reconstructed_even, streams[1]);
        
    } else if (have_stream2 && have_stream3) {
        // Reconstruct from both redundancy streams
        reconstructed_audio = reconstructFromRedundancyPair(streams[2], streams[3]);
        
    } else {
        // Partial reconstruction or prediction
        reconstructed_audio = handlePartialReconstruction(streams, received_streams);
    }
    
    return reconstructed_audio;
}

std::vector<float> JELLIEDecoder::combineEvenOddStreams(
    const std::vector<float>& even_samples,
    const std::vector<float>& odd_samples) {
    
    size_t total_samples = even_samples.size() + odd_samples.size();
    std::vector<float> combined;
    combined.reserve(total_samples);
    
    size_t max_pairs = std::min(even_samples.size(), odd_samples.size());
    
    // Interleave even and odd samples
    for (size_t i = 0; i < max_pairs; ++i) {
        combined.push_back(even_samples[i]);  // Even position
        combined.push_back(odd_samples[i]);   // Odd position
    }
    
    // Handle remaining samples if sizes differ
    if (even_samples.size() > max_pairs) {
        for (size_t i = max_pairs; i < even_samples.size(); ++i) {
            combined.push_back(even_samples[i]);
        }
    }
    if (odd_samples.size() > max_pairs) {
        for (size_t i = max_pairs; i < odd_samples.size(); ++i) {
            combined.push_back(odd_samples[i]);
        }
    }
    
    return combined;
}

std::vector<float> JELLIEDecoder::reconstructOddFromRedundancy(
    const std::vector<float>& even_stream,
    const std::vector<float>& redundancy_stream,
    uint8_t redundancy_type) {
    
    std::vector<float> odd_stream;
    odd_stream.reserve(even_stream.size());
    
    for (size_t i = 0; i < std::min(even_stream.size(), redundancy_stream.size()); ++i) {
        float even_sample = even_stream[i];
        float redundancy_sample = redundancy_stream[i];
        
        // Reverse the redundancy calculation
        float odd_sample;
        if (redundancy_type == 2) {
            // Redundancy A: redundancy = (even + odd) * 0.5 + (even - odd) * 0.3
            // Solve for odd: odd = ((redundancy - even * 0.8) / 0.2)
            odd_sample = (redundancy_sample - even_sample * 0.8f) / 0.2f;
        } else {
            // Redundancy B: redundancy = (even - odd) * 0.7 + (even + odd) * 0.1
            // Solve for odd: odd = (even * 0.8 - redundancy) / 0.6
            odd_sample = (even_sample * 0.8f - redundancy_sample) / 0.6f;
        }
        
        odd_stream.push_back(odd_sample);
    }
    
    return odd_stream;
}

std::vector<float> JELLIEDecoder::reconstructEvenFromRedundancy(
    const std::vector<float>& odd_stream,
    const std::vector<float>& redundancy_stream,
    uint8_t redundancy_type) {
    
    std::vector<float> even_stream;
    even_stream.reserve(odd_stream.size());
    
    for (size_t i = 0; i < std::min(odd_stream.size(), redundancy_stream.size()); ++i) {
        float odd_sample = odd_stream[i];
        float redundancy_sample = redundancy_stream[i];
        
        // Reverse the redundancy calculation
        float even_sample;
        if (redundancy_type == 2) {
            // Redundancy A: redundancy = (even + odd) * 0.5 + (even - odd) * 0.3
            // Solve for even: even = ((redundancy + odd * 0.2) / 0.8)
            even_sample = (redundancy_sample + odd_sample * 0.2f) / 0.8f;
        } else {
            // Redundancy B: redundancy = (even - odd) * 0.7 + (even + odd) * 0.1
            // Solve for even: even = (redundancy + odd * 0.6) / 0.8
            even_sample = (redundancy_sample + odd_sample * 0.6f) / 0.8f;
        }
        
        even_stream.push_back(even_sample);
    }
    
    return even_stream;
}

std::vector<float> JELLIEDecoder::reconstructFromRedundancyPair(
    const std::vector<float>& redundancy_a,
    const std::vector<float>& redundancy_b) {
    
    // This is more complex - need to solve system of equations
    // For now, use weighted average approach
    std::vector<float> reconstructed;
    reconstructed.reserve(redundancy_a.size() * 2);
    
    for (size_t i = 0; i < std::min(redundancy_a.size(), redundancy_b.size()); ++i) {
        // Simple weighted reconstruction - can be improved with PNBTR
        float avg_redundancy = (redundancy_a[i] + redundancy_b[i]) * 0.5f;
        
        // Generate even and odd samples from average
        float even_sample = avg_redundancy;
        float odd_sample = avg_redundancy * 0.9f; // Slight variation
        
        reconstructed.push_back(even_sample);
        reconstructed.push_back(odd_sample);
    }
    
    return reconstructed;
}

std::vector<float> JELLIEDecoder::handlePartialReconstruction(
    const std::map<uint8_t, std::vector<float>>& available_streams,
    const std::map<uint8_t, bool>& received_streams) {
    
    // Use PNBTR prediction if available
    if (config_.enable_pnbtr && pnbtr_predictor_) {
        
        // Find the best available stream
        std::vector<float> reference_stream;
        for (const auto& stream_pair : available_streams) {
            if (!stream_pair.second.empty()) {
                reference_stream = stream_pair.second;
                break;
            }
        }
        
        if (!reference_stream.empty()) {
            // Use PNBTR to predict the missing audio content
            return predictMissingAudio(reference_stream);
        }
    }
    
    return handleCompletePacketLoss();
}

std::vector<float> JELLIEDecoder::handleCompletePacketLoss() {
    // Complete packet loss - use PNBTR if available
    if (config_.enable_pnbtr && pnbtr_predictor_ && !last_valid_audio_.empty()) {
        
        // Predict audio based on last known good samples
        auto predicted = pnbtr_predictor_->predict(
            last_valid_audio_, 
            config_.expected_frame_size
        );
        
        // Notify packet loss callback
        if (packet_loss_callback_) {
            packet_loss_callback_(JDATMessage::getCurrentTimestamp(), 
                                config_.expected_frame_size);
        }
        
        // Notify recovery callback
        if (recovery_callback_) {
            recovery_callback_(predicted, true); // true = PNBTR used
        }
        
        return predicted;
    }
    
    // Return silence if no prediction available
    std::vector<float> silence(config_.expected_frame_size, 0.0f);
    
    if (packet_loss_callback_) {
        packet_loss_callback_(JDATMessage::getCurrentTimestamp(), 
                            config_.expected_frame_size);
    }
    
    return silence;
}

std::vector<float> JELLIEDecoder::predictMissingAudio(
    const std::vector<float>& reference_samples) {
    
    if (!config_.enable_pnbtr || !pnbtr_predictor_) {
        return reference_samples; // Return as-is if no prediction
    }
    
    // Use PNBTR to enhance/expand the reference samples
    auto predicted = pnbtr_predictor_->predict(reference_samples, config_.expected_frame_size);
    
    if (recovery_callback_) {
        recovery_callback_(predicted, true); // true = PNBTR used
    }
    
    return predicted;
}

JELLIEDecoder::Statistics JELLIEDecoder::getStatistics() const {
    Statistics stats;
    // Implementation would collect real statistics
    return stats;
}

void JELLIEDecoder::resetStatistics() {
    // Implementation would reset counters
}

} // namespace jdat
