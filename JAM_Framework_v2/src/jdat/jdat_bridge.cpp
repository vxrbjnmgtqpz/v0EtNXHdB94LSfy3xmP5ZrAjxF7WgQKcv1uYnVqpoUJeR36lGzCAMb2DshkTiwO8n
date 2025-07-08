/**
 * JAM Framework v2 - JDAT Integration Bridge Implementation
 */

#include "jdat_bridge.h"

// JDAT Framework includes (these would be real includes in practice)
#include "JELLIEEncoder.h"
#include "JELLIEDecoder.h"
#include "JDATMessage.h"

#include <chrono>
#include <algorithm>

namespace jam {

JDATBridge::JDATBridge(const JDATConfig& config) : config_(config) {
    session_id_ = std::hash<std::string>{}(config_.session_name);
}

JDATBridge::~JDATBridge() {
    stop();
}

bool JDATBridge::initialize() {
    if (is_initialized_) return true;
    
    try {
        // Initialize JDAT encoder
        jdat::JELLIEEncoder::Config encoder_config;
        encoder_config.sample_rate = config_.sample_rate;
        encoder_config.quality = config_.quality;
        encoder_config.frame_size_samples = config_.frame_size_samples;
        encoder_config.buffer_size_ms = config_.buffer_size_ms;
        encoder_config.redundancy_level = config_.redundancy_level;
        encoder_config.session_id = config_.session_name;
        
        encoder_ = std::make_unique<jdat::JELLIEEncoder>(encoder_config);
        
        // Set JDAT message callback to forward to network
        encoder_->setMessageCallback([this](const jdat::JDATMessage& message) {
            handleJDATMessage(message);
        });
        
        // Initialize JDAT decoder  
        jdat::JELLIEDecoder::Config decoder_config;
        decoder_config.sample_rate = config_.sample_rate;
        decoder_config.quality = config_.quality;
        decoder_config.buffer_size_ms = config_.buffer_size_ms;
        decoder_config.enable_prediction = config_.enable_audio_prediction;
        
        decoder_ = std::make_unique<jdat::JELLIEDecoder>(decoder_config);
        
        // Initialize JAM Framework v2 transport
        if (config_.enable_multithreaded_transport) {
            MultiThreadConfig transport_config;
            transport_config.send_threads = config_.send_threads;
            transport_config.recv_threads = config_.recv_threads;
            transport_config.enable_redundancy = config_.redundancy_level > 1;
            transport_config.enable_gpu_burst = config_.enable_gpu_native;
            
            transport_ = std::make_unique<MultiThreadedUDPTransport>(
                config_.multicast_address, config_.udp_port, "0.0.0.0", transport_config);
        }
        
        // Initialize TOAST v2 protocol
        toast_protocol_ = std::make_unique<TOASTv2Protocol>();
        if (!toast_protocol_->initialize(config_.multicast_address, config_.udp_port, session_id_)) {
            notifyError("Failed to initialize TOAST v2 protocol");
            return false;
        }
        
        // Set up network receive callback
        toast_protocol_->set_audio_callback([this](const TOASTFrame& frame) {
            try {
                jdat::JDATMessage jdat_message = convertTOASTToJDAT(frame);
                
                // Process with decoder
                if (decoder_) {
                    // This would typically generate audio output
                    // For now, we'll create a frame from the JDAT message
                    JDATAudioFrame audio_frame;
                    audio_frame.timestamp_us = frame.header.timestamp_us * 1000;
                    audio_frame.sequence_number = frame.header.sequence_number;
                    audio_frame.stream_id = 0; // Default stream
                    
                    // Add to output buffer
                    {
                        std::lock_guard<std::mutex> lock(output_buffer_mutex_);
                        output_buffer_.push_back(audio_frame);
                        
                        // Keep buffer size reasonable
                        if (output_buffer_.size() > 100) {
                            output_buffer_.erase(output_buffer_.begin());
                        }
                    }
                    
                    // Notify callback
                    if (audio_output_callback_) {
                        audio_output_callback_(audio_frame);
                    }
                }
            } catch (const std::exception& e) {
                notifyError("Audio processing error: " + std::string(e.what()));
            }
        });
        
        // Initialize GPU pipeline if requested
        if (config_.enable_gpu_native) {
            gpu_pipeline_ = std::make_unique<ComputePipeline>();
            if (gpu_pipeline_->initialize()) {
                if (transport_) {
                    transport_->initialize_gpu_backend(gpu_pipeline_);
                }
                notifyStatus("GPU NATIVE processing enabled for JDAT", false);
            } else {
                notifyError("Failed to initialize GPU NATIVE processing");
            }
        }
        
        is_initialized_ = true;
        notifyStatus("JDAT Bridge initialized", false);
        return true;
        
    } catch (const std::exception& e) {
        notifyError("JDAT Bridge initialization failed: " + std::string(e.what()));
        return false;
    }
}

bool JDATBridge::start() {
    if (!is_initialized_) {
        if (!initialize()) {
            return false;
        }
    }
    
    try {
        // Start JDAT encoder
        if (encoder_ && !encoder_->start()) {
            notifyError("Failed to start JDAT encoder");
            return false;
        }
        
        // Start JDAT decoder
        if (decoder_ && !decoder_->start()) {
            notifyError("Failed to start JDAT decoder");
            return false;
        }
        
        // Start transport
        if (transport_) {
            if (!transport_->initialize() || !transport_->start_receiving([this](std::span<const uint8_t> data) {
                std::vector<uint8_t> data_vec(data.begin(), data.end());
                handleNetworkData(data_vec);
            })) {
                notifyError("Failed to start multi-threaded transport");
                return false;
            }
        }
        
        // Start TOAST protocol
        if (toast_protocol_ && !toast_protocol_->start_processing()) {
            notifyError("Failed to start TOAST v2 protocol");
            return false;
        }
        
        is_active_ = true;
        notifyStatus("JDAT audio streaming active", true);
        return true;
        
    } catch (const std::exception& e) {
        notifyError("Failed to start JDAT bridge: " + std::string(e.what()));
        return false;
    }
}

void JDATBridge::stop() {
    is_active_ = false;
    
    if (encoder_) {
        encoder_->stop();
    }
    
    if (decoder_) {
        decoder_->stop();
    }
    
    if (transport_) {
        transport_->stop_receiving();
    }
    
    if (toast_protocol_) {
        toast_protocol_->stop_processing();
    }
    
    notifyStatus("JDAT audio streaming stopped", false);
}

void JDATBridge::processInputAudio(const float* samples, int numSamples, uint32_t sampleRate) {
    if (!is_active_ || !encoder_ || !samples || numSamples <= 0) {
        return;
    }
    
    try {
        // Convert to vector for JDAT encoder
        std::vector<float> audio_data(samples, samples + numSamples);
        uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        // Process with JDAT encoder (this will trigger the message callback)
        encoder_->processAudio(audio_data, timestamp);
        
        // Update metrics
        current_metrics_.active_streams = 1; // For input stream
        
    } catch (const std::exception& e) {
        notifyError("Input audio processing error: " + std::string(e.what()));
    }
}

bool JDATBridge::getOutputAudio(JDATAudioFrame& frame) {
    std::lock_guard<std::mutex> lock(output_buffer_mutex_);
    
    if (output_buffer_.empty()) {
        return false;
    }
    
    // Get oldest frame
    frame = output_buffer_.front();
    output_buffer_.erase(output_buffer_.begin());
    
    return true;
}

void JDATBridge::handleJDATMessage(const jdat::JDATMessage& message) {
    if (!toast_protocol_ || !is_active_) {
        return;
    }
    
    try {
        // Convert JDAT message to TOAST frame
        TOASTFrame toast_frame = convertJDATToTOAST(message);
        
        // Send via TOAST protocol
        bool use_burst = config_.redundancy_level > 1;
        toast_protocol_->send_frame(toast_frame, use_burst);
        
        // Update metrics
        current_metrics_.throughput_mbps += (toast_frame.payload.size() * 8.0) / 1000000.0;
        
    } catch (const std::exception& e) {
        notifyError("JDAT message transmission error: " + std::string(e.what()));
    }
}

void JDATBridge::handleNetworkData(const std::vector<uint8_t>& data) {
    // This would typically be handled by the TOAST protocol callbacks
    // but we can add additional processing here if needed
}

TOASTFrame JDATBridge::convertJDATToTOAST(const jdat::JDATMessage& jdat_message) {
    TOASTFrame frame;
    
    // Set frame header
    frame.header.frame_type = static_cast<uint8_t>(TOASTFrameType::AUDIO);
    frame.header.timestamp_us = static_cast<uint32_t>(jdat_message.getTimestamp() / 1000);
    frame.header.sequence_number = jdat_message.getSequenceNumber();
    frame.header.session_id = session_id_;
    
    // Serialize JDAT message to JSON and put in payload
    std::string json_data = jdat_message.toJSON();
    frame.payload.assign(json_data.begin(), json_data.end());
    frame.header.payload_size = frame.payload.size();
    
    return frame;
}

jdat::JDATMessage JDATBridge::convertTOASTToJDAT(const TOASTFrame& toast_frame) {
    // Extract JSON from payload
    std::string json_data(toast_frame.payload.begin(), toast_frame.payload.end());
    
    // Parse JDAT message from JSON
    jdat::JDATMessage message;
    message.fromJSON(json_data);
    
    return message;
}

void JDATBridge::enableAudioPrediction(bool enable) {
    config_.enable_audio_prediction = enable;
    if (decoder_) {
        decoder_->enablePrediction(enable);
    }
}

void JDATBridge::enableGPUNative(bool enable) {
    config_.enable_gpu_native = enable;
    if (transport_) {
        transport_->enable_gpu_processing(enable);
    }
}

JDATBridge::PerformanceMetrics JDATBridge::getPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

void JDATBridge::notifyStatus(const std::string& status, bool connected) {
    if (status_callback_) {
        status_callback_(status, connected);
    }
}

void JDATBridge::notifyError(const std::string& error) {
    if (error_callback_) {
        error_callback_(error);
    }
}

// TOASTerJDATIntegration implementation

TOASTerJDATIntegration::TOASTerJDATIntegration() {
    // Set up default JDAT configuration for TOASTer
    default_config_.sample_rate = jdat::SampleRate::SR_48000;
    default_config_.quality = jdat::AudioQuality::HIGH_PRECISION;
    default_config_.frame_size_samples = 480; // 10ms at 48kHz
    default_config_.buffer_size_ms = 20;
    default_config_.enable_audio_prediction = true;
    default_config_.enable_gpu_native = true;
    default_config_.redundancy_level = 2;
    default_config_.enable_multithreaded_transport = true;
    default_config_.send_threads = 2;
    default_config_.recv_threads = 2;
}

TOASTerJDATIntegration::~TOASTerJDATIntegration() {
    stopAudioTransmission();
}

bool TOASTerJDATIntegration::initializeAudioStreaming(const std::string& session_name) {
    default_config_.session_name = session_name;
    
    jdat_bridge_ = std::make_unique<JDATBridge>(default_config_);
    
    // Set up callbacks
    jdat_bridge_->setStatusCallback([this](const std::string& status, bool connected) {
        onJDATStatus(status, connected);
    });
    
    jdat_bridge_->setAudioOutputCallback([this](const JDATAudioFrame& frame) {
        onJDATAudioOutput(frame);
    });
    
    return jdat_bridge_->initialize();
}

bool TOASTerJDATIntegration::startAudioTransmission() {
    if (jdat_bridge_) {
        return jdat_bridge_->start();
    }
    return false;
}

void TOASTerJDATIntegration::stopAudioTransmission() {
    if (jdat_bridge_) {
        jdat_bridge_->stop();
    }
}

void TOASTerJDATIntegration::sendAudioData(const float* samples, int numSamples, int sampleRate) {
    if (jdat_bridge_) {
        jdat_bridge_->processInputAudio(samples, numSamples, sampleRate);
    }
}

bool TOASTerJDATIntegration::receiveAudioData(float* samples, int maxSamples, int& actualSamples, int& sampleRate) {
    if (!jdat_bridge_ || !samples) {
        actualSamples = 0;
        return false;
    }
    
    JDATAudioFrame frame;
    if (jdat_bridge_->getOutputAudio(frame)) {
        actualSamples = std::min(maxSamples, static_cast<int>(frame.samples.size()));
        sampleRate = frame.sample_rate;
        
        std::copy(frame.samples.begin(), frame.samples.begin() + actualSamples, samples);
        return true;
    }
    
    actualSamples = 0;
    return false;
}

bool TOASTerJDATIntegration::isAudioConnected() const {
    return jdat_bridge_ && jdat_bridge_->isActive();
}

int TOASTerJDATIntegration::getActiveAudioPeers() const {
    if (jdat_bridge_) {
        return jdat_bridge_->getActiveStreamCount();
    }
    return 0;
}

double TOASTerJDATIntegration::getAudioLatency() const {
    if (jdat_bridge_) {
        return jdat_bridge_->getCurrentLatency();
    }
    return 0.0;
}

void TOASTerJDATIntegration::onJDATStatus(const std::string& status, bool connected) {
    if (audio_status_callback_) {
        int peers = getActiveAudioPeers();
        audio_status_callback_(status, connected, peers);
    }
}

void TOASTerJDATIntegration::onJDATAudioOutput(const JDATAudioFrame& frame) {
    // Audio frame received - could be forwarded to audio output system
}

} // namespace jam
