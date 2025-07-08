#include "../include/PNBTR_JDAT_Bridge.h"
#include "../../PNBTR_Framework/include/pnbtr_framework.h"
#include "../include/JELLIEDecoder.h"
#include "../include/JELLIEEncoder.h"
#include "../include/WaveformPredictor.h"

#include <iostream>
#include <chrono>
#include <algorithm>

namespace jdat {

class PNBTR_JDAT_Bridge::Impl {
public:
    std::unique_ptr<pnbtr::PNBTRFramework> pnbtr_framework;
    pnbtr::PNBTRConfig pnbtr_config;
    
    // Integration state
    bool pnbtr_enabled = true;
    bool learning_enabled = true;
    std::string session_id;
    
    // Audio context tracking for PNBTR
    pnbtr::AudioContext current_context;
    std::vector<float> reference_buffer;
    
    // Performance metrics
    struct PerfMetrics {
        uint64_t total_predictions = 0;
        uint64_t successful_predictions = 0;
        uint64_t packet_loss_events = 0;
        double average_prediction_time_ms = 0.0;
        double prediction_accuracy = 0.0;
    } metrics;
};

PNBTR_JDAT_Bridge::PNBTR_JDAT_Bridge() 
    : pImpl(std::make_unique<Impl>()) {
    
    // Initialize PNBTR configuration for audio streaming
    pImpl->pnbtr_config.sample_rate = 96000;  // High-quality audio
    pImpl->pnbtr_config.bit_depth = 24;      // Professional audio depth
    pImpl->pnbtr_config.channels = 2;        // Stereo
    pImpl->pnbtr_config.prediction_window_ms = 50.0f;  // 50ms lookahead
    pImpl->pnbtr_config.enable_neural_inference = true;
    pImpl->pnbtr_config.enable_continuous_learning = true;
    pImpl->pnbtr_config.gpu_backend = pnbtr::PNBTRConfig::PNBTR_GPU_AUTO;
    
    // Initialize PNBTR framework
    pImpl->pnbtr_framework = std::make_unique<pnbtr::PNBTRFramework>();
    
    std::cout << "PNBTR-JDAT Bridge initialized for VST3 plugin development\n";
}

PNBTR_JDAT_Bridge::~PNBTR_JDAT_Bridge() = default;

bool PNBTR_JDAT_Bridge::initialize() {
    if (!pImpl->pnbtr_framework->initialize(pImpl->pnbtr_config)) {
        std::cerr << "Failed to initialize PNBTR framework\n";
        return false;
    }
    
    // Generate session ID for learning
    pImpl->session_id = generateSessionId();
    
    std::cout << "PNBTR-JDAT Bridge ready for audio processing\n";
    std::cout << "  Session ID: " << pImpl->session_id << "\n";
    std::cout << "  GPU Backend: " << (pImpl->pnbtr_config.gpu_backend == pnbtr::PNBTRConfig::PNBTR_GPU_METAL ? "Metal" : "Vulkan") << "\n";
    std::cout << "  Neural Inference: " << (pImpl->pnbtr_config.enable_neural_inference ? "Enabled" : "Disabled") << "\n";
    
    return true;
}

void PNBTR_JDAT_Bridge::shutdown() {
    if (pImpl->pnbtr_framework) {
        pImpl->pnbtr_framework->shutdown();
    }
}

std::vector<float> PNBTR_JDAT_Bridge::enhanceJELLIEDecoding(
    const std::vector<float>& decoded_audio,
    const std::map<uint8_t, bool>& stream_availability,
    uint32_t sample_rate,
    PacketLossInfo loss_info) {
    
    if (!pImpl->pnbtr_enabled || !pImpl->pnbtr_framework) {
        return decoded_audio;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert to PNBTR format
    pnbtr::AudioBuffer pnbtr_buffer;
    pnbtr_buffer.samples = decoded_audio;
    pnbtr_buffer.sample_rate = sample_rate;
    pnbtr_buffer.channels = 2;
    pnbtr_buffer.bit_depth = 24;
    pnbtr_buffer.timestamp_ns = getCurrentTimestamp() * 1000;
    
    // Update audio context based on stream availability
    updateAudioContext(pnbtr_buffer, stream_availability, loss_info);
    
    std::vector<float> enhanced_audio;
    
    if (loss_info.has_packet_loss) {
        // Use PNBTR for packet loss recovery
        enhanced_audio = handlePacketLossRecovery(pnbtr_buffer, loss_info);
        pImpl->metrics.packet_loss_events++;
    } else {
        // Use PNBTR for quality enhancement (LSB reconstruction)
        enhanced_audio = enhanceAudioQuality(pnbtr_buffer);
    }
    
    // Store reference for learning
    if (pImpl->learning_enabled) {
        pImpl->reference_buffer = decoded_audio;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    pImpl->metrics.average_prediction_time_ms = duration.count() / 1000.0;
    pImpl->metrics.total_predictions++;
    
    return enhanced_audio;
}

std::vector<float> PNBTR_JDAT_Bridge::handlePacketLossRecovery(
    const pnbtr::AudioBuffer& audio_buffer,
    const PacketLossInfo& loss_info) {
    
    // Calculate prediction samples needed
    uint32_t prediction_samples = static_cast<uint32_t>(
        loss_info.gap_duration_ms * audio_buffer.sample_rate / 1000.0f);
    
    // Use PNBTR neural analog extrapolation for gap filling
    auto extrapolated = pImpl->pnbtr_framework->extrapolate_analog_signal(
        audio_buffer, 
        pImpl->current_context,
        prediction_samples
    );
    
    // Combine available audio with predicted audio
    std::vector<float> recovered_audio;
    recovered_audio.reserve(audio_buffer.samples.size() + prediction_samples);
    
    // Add available audio
    recovered_audio.insert(recovered_audio.end(), 
                          audio_buffer.samples.begin(), 
                          audio_buffer.samples.end());
    
    // Add predicted audio for gap
    recovered_audio.insert(recovered_audio.end(),
                          extrapolated.samples.begin(),
                          extrapolated.samples.end());
    
    std::cout << "PNBTR recovered " << prediction_samples << " samples (" 
              << loss_info.gap_duration_ms << "ms) during packet loss\n";
    
    return recovered_audio;
}

std::vector<float> PNBTR_JDAT_Bridge::enhanceAudioQuality(
    const pnbtr::AudioBuffer& audio_buffer) {
    
    // Use PNBTR for LSB reconstruction (zero-noise dither replacement)
    auto enhanced = pImpl->pnbtr_framework->reconstruct_lsb_mathematically(
        audio_buffer,
        pImpl->current_context
    );
    
    return enhanced.samples;
}

void PNBTR_JDAT_Bridge::updateAudioContext(
    const pnbtr::AudioBuffer& audio_buffer,
    const std::map<uint8_t, bool>& stream_availability,
    const PacketLossInfo& loss_info) {
    
    // Analyze audio characteristics for PNBTR context
    if (!audio_buffer.samples.empty()) {
        // Estimate fundamental frequency using autocorrelation
        pImpl->current_context.fundamental_frequency = estimateFundamentalFrequency(
            audio_buffer.samples, audio_buffer.sample_rate);
        
        // Calculate pitch confidence based on stream availability
        float availability_ratio = 0.0f;
        for (const auto& stream : stream_availability) {
            if (stream.second) availability_ratio += 0.25f; // 4 streams max
        }
        pImpl->current_context.pitch_confidence = availability_ratio;
        
        // Extract envelope information
        extractEnvelopeContext(audio_buffer.samples);
        
        // Update spectral context
        updateSpectralContext(audio_buffer.samples, audio_buffer.sample_rate);
    }
    
    // Adjust context based on packet loss
    if (loss_info.has_packet_loss) {
        // Reduce confidence for prediction
        pImpl->current_context.pitch_confidence *= 0.7f;
    }
}

float PNBTR_JDAT_Bridge::estimateFundamentalFrequency(
    const std::vector<float>& samples, 
    uint32_t sample_rate) {
    
    if (samples.size() < 512) return 0.0f;
    
    // Simple autocorrelation-based pitch detection
    float max_correlation = 0.0f;
    int best_lag = 1;
    int min_lag = sample_rate / 2000; // 2kHz max
    int max_lag = sample_rate / 60;   // 60Hz min
    
    for (int lag = min_lag; lag < std::min(max_lag, (int)samples.size() / 2); ++lag) {
        float correlation = 0.0f;
        int count = 0;
        
        for (size_t i = 0; i < samples.size() - lag; ++i) {
            correlation += samples[i] * samples[i + lag];
            count++;
        }
        
        if (count > 0) {
            correlation /= count;
            if (correlation > max_correlation) {
                max_correlation = correlation;
                best_lag = lag;
            }
        }
    }
    
    return (max_correlation > 0.3f) ? static_cast<float>(sample_rate) / best_lag : 0.0f;
}

void PNBTR_JDAT_Bridge::extractEnvelopeContext(const std::vector<float>& samples) {
    if (samples.empty()) return;
    
    // Simple envelope extraction
    float max_amplitude = 0.0f;
    float rms_sum = 0.0f;
    
    for (float sample : samples) {
        float abs_sample = std::abs(sample);
        max_amplitude = std::max(max_amplitude, abs_sample);
        rms_sum += sample * sample;
    }
    
    float rms = std::sqrt(rms_sum / samples.size());
    
    // Simple ADSR estimation (placeholder - could be more sophisticated)
    pImpl->current_context.attack_time_ms = 10.0f;
    pImpl->current_context.decay_time_ms = 50.0f;
    pImpl->current_context.sustain_level = rms / max_amplitude;
    pImpl->current_context.release_time_ms = 100.0f;
}

void PNBTR_JDAT_Bridge::updateSpectralContext(
    const std::vector<float>& samples, 
    uint32_t sample_rate) {
    
    // Placeholder for spectral analysis
    // In full implementation, would use FFT to analyze:
    // - Spectral centroid (brightness)
    // - Spectral rolloff (high-frequency content)
    // - MFCC coefficients (timbral characteristics)
    
    pImpl->current_context.spectral_centroid.clear();
    pImpl->current_context.spectral_rolloff.clear();
    pImpl->current_context.mfcc_coefficients.clear();
    
    // Simple spectral centroid estimation
    float spectral_centroid = static_cast<float>(sample_rate) * 0.3f; // Placeholder
    pImpl->current_context.spectral_centroid.push_back(spectral_centroid);
}

void PNBTR_JDAT_Bridge::submitLearningData(
    const std::vector<float>& pnbtr_output,
    const std::vector<float>& reference_audio) {
    
    if (!pImpl->learning_enabled || !pImpl->pnbtr_framework) {
        return;
    }
    
    // Convert to PNBTR format
    pnbtr::AudioBuffer pnbtr_buffer;
    pnbtr_buffer.samples = pnbtr_output;
    pnbtr_buffer.sample_rate = pImpl->pnbtr_config.sample_rate;
    pnbtr_buffer.channels = pImpl->pnbtr_config.channels;
    pnbtr_buffer.bit_depth = pImpl->pnbtr_config.bit_depth;
    pnbtr_buffer.pnbtr_processed = true;
    
    pnbtr::AudioBuffer reference_buffer;
    reference_buffer.samples = reference_audio;
    reference_buffer.sample_rate = pImpl->pnbtr_config.sample_rate;
    reference_buffer.channels = pImpl->pnbtr_config.channels;
    reference_buffer.bit_depth = pImpl->pnbtr_config.bit_depth;
    
    // Submit to PNBTR learning system
    pImpl->pnbtr_framework->submit_learning_pair(
        pnbtr_buffer,
        reference_buffer,
        pImpl->current_context
    );
}

PNBTR_JDAT_Bridge::Statistics PNBTR_JDAT_Bridge::getStatistics() const {
    Statistics stats;
    stats.total_predictions = pImpl->metrics.total_predictions;
    stats.successful_predictions = pImpl->metrics.successful_predictions;
    stats.packet_loss_events = pImpl->metrics.packet_loss_events;
    stats.average_prediction_time_ms = pImpl->metrics.average_prediction_time_ms;
    stats.prediction_accuracy = pImpl->metrics.prediction_accuracy;
    
    // Calculate packet loss recovery rate
    stats.packet_loss_recovery_rate = (pImpl->metrics.packet_loss_events > 0) ?
        static_cast<double>(pImpl->metrics.successful_predictions) / pImpl->metrics.packet_loss_events :
        0.0;
    
    return stats;
}

bool PNBTR_JDAT_Bridge::enablePNBTR(bool enable) {
    pImpl->pnbtr_enabled = enable;
    return true;
}

bool PNBTR_JDAT_Bridge::enableLearning(bool enable) {
    pImpl->learning_enabled = enable;
    return true;
}

std::string PNBTR_JDAT_Bridge::generateSessionId() {
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
    
    return "pnbtr-jdat-" + std::to_string(timestamp);
}

uint64_t PNBTR_JDAT_Bridge::getCurrentTimestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

// Factory functions for VST3 plugin integration
std::unique_ptr<PNBTR_JDAT_Bridge> createPNBTRJDATBridge() {
    auto bridge = std::make_unique<PNBTR_JDAT_Bridge>();
    if (!bridge->initialize()) {
        return nullptr;
    }
    return bridge;
}

std::unique_ptr<PNBTR_JDAT_Bridge> createPNBTRJDATBridgeForVST3(
    uint32_t sample_rate,
    uint16_t bit_depth,
    uint16_t channels) {
    
    auto bridge = std::make_unique<PNBTR_JDAT_Bridge>();
    
    // Configure for VST3 requirements
    bridge->pImpl->pnbtr_config.sample_rate = sample_rate;
    bridge->pImpl->pnbtr_config.bit_depth = bit_depth;
    bridge->pImpl->pnbtr_config.channels = channels;
    
    if (!bridge->initialize()) {
        return nullptr;
    }
    
    return bridge;
}

} // namespace jdat 