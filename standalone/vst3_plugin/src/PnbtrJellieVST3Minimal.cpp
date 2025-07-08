#include "../include/PnbtrJellieVST3Minimal.h"
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cstdio>

// Constructor
PnbtrJellieVST3Minimal::PnbtrJellieVST3Minimal() {
    // Create dual engines (TX and RX)
    tx_engine_ = std::make_unique<pnbtr_jellie::PnbtrJellieEngine>();
    rx_engine_ = std::make_unique<pnbtr_jellie::PnbtrJellieEngine>();
    
    printf("[VST3-MINIMAL] PNBTR+JELLIE plugin created (sub-50μs target)\n");
}

// Destructor
PnbtrJellieVST3Minimal::~PnbtrJellieVST3Minimal() {
    if (initialized_.load()) {
        shutdown();
    }
    printf("[VST3-MINIMAL] PNBTR+JELLIE plugin destroyed\n");
}

// Initialize plugin
bool PnbtrJellieVST3Minimal::initialize(double sample_rate, int block_size, int num_channels) {
    if (initialized_.load()) {
        return true;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    sample_rate_ = sample_rate;
    block_size_ = block_size;
    num_channels_ = num_channels;
    
    // Initialize both engines
    bool tx_success = tx_engine_->initialize(static_cast<uint32_t>(sample_rate), 
                                           static_cast<uint16_t>(block_size));
    bool rx_success = rx_engine_->initialize(static_cast<uint32_t>(sample_rate), 
                                           static_cast<uint16_t>(block_size));
    
    if (!tx_success || !rx_success) {
        printf("[VST3-MINIMAL] Failed to initialize engines\n");
        return false;
    }
    
    // Configure engines
    tx_engine_->setPluginMode(pnbtr_jellie::PnbtrJellieEngine::PluginMode::TX_MODE);
    rx_engine_->setPluginMode(pnbtr_jellie::PnbtrJellieEngine::PluginMode::RX_MODE);
    
    // Allocate processing buffers
    int max_samples = block_size * 2; // Allow some headroom
    interleaved_input_buffer_.resize(max_samples * num_channels);
    interleaved_output_buffer_.resize(max_samples * num_channels);
    
    // Apply initial configuration
    updateEngineConfiguration();
    
    initialized_.store(true);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double init_time_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    
    printf("[VST3-MINIMAL] Plugin initialized in %.2f μs (SR: %.0f, Block: %d, Channels: %d)\n", 
           init_time_us, sample_rate, block_size, num_channels);
    
    return true;
}

// Shutdown plugin
void PnbtrJellieVST3Minimal::shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    tx_engine_->terminate();
    rx_engine_->terminate();
    initialized_.store(false);
    
    printf("[VST3-MINIMAL] Plugin shutdown complete\n");
}

// Main audio processing function
bool PnbtrJellieVST3Minimal::processAudio(const float* const* input_channels, 
                                         float* const* output_channels, 
                                         int num_samples, 
                                         int num_channels) {
    if (!initialized_.load()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Update engine configuration if parameters changed
    updateEngineConfiguration();
    
    // Convert channel-based audio to interleaved
    convertChannelsToInterleaved(input_channels, interleaved_input_buffer_.data(), 
                               num_samples, num_channels);
    
    // Process with appropriate engine based on mode
    bool success = false;
    if (current_mode_.load() == PluginMode::TX_MODE) {
        success = tx_engine_->processAudio(
            interleaved_input_buffer_.data(),
            interleaved_output_buffer_.data(),
            static_cast<uint32_t>(num_samples),
            static_cast<uint32_t>(num_channels)
        );
    } else {
        success = rx_engine_->processAudio(
            interleaved_input_buffer_.data(),
            interleaved_output_buffer_.data(),
            static_cast<uint32_t>(num_samples),
            static_cast<uint32_t>(num_channels)
        );
    }
    
    // Convert interleaved audio back to channel-based
    convertInterleavedToChannels(interleaved_output_buffer_.data(), output_channels, 
                               num_samples, num_channels);
    
    // Update performance statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    updatePerformanceStats(processing_time_us);
    
    performance_stats_.frames_processed.fetch_add(1);
    
    return success;
}

// Set plugin mode
void PnbtrJellieVST3Minimal::setPluginMode(PluginMode mode) {
    current_mode_.store(mode);
    parameters_.plugin_mode.store(mode == PluginMode::TX_MODE ? 0.0f : 1.0f);
    
    printf("[VST3-MINIMAL] Mode set to %s\n", 
           mode == PluginMode::TX_MODE ? "TX (Transmit)" : "RX (Receive)");
}

// Get plugin mode
PnbtrJellieVST3Minimal::PluginMode PnbtrJellieVST3Minimal::getPluginMode() const {
    return current_mode_.load();
}

// Update engine configuration based on parameters
void PnbtrJellieVST3Minimal::updateEngineConfiguration() {
    // Update plugin mode
    float mode_param = parameters_.plugin_mode.load();
    PluginMode new_mode = (mode_param < 0.5f) ? PluginMode::TX_MODE : PluginMode::RX_MODE;
    if (new_mode != current_mode_.load()) {
        setPluginMode(new_mode);
    }
    
    // Update network configuration
    pnbtr_jellie::PnbtrJellieEngine::NetworkConfig network_config;
    network_config.target_port = static_cast<uint16_t>(parameters_.network_port.load());
    network_config.network_quality = parameters_.network_quality.load();
    
    tx_engine_->setNetworkConfig(network_config);
    rx_engine_->setNetworkConfig(network_config);
    
    // Update JELLIE configuration
    pnbtr_jellie::PnbtrJellieEngine::JellieConfig jellie_config;
    jellie_config.sample_rate = static_cast<uint32_t>(sample_rate_);
    jellie_config.channels = static_cast<uint8_t>(num_channels_);
    jellie_config.encoding_quality = parameters_.jellie_quality.load();
    
    tx_engine_->setJellieConfig(jellie_config);
    rx_engine_->setJellieConfig(jellie_config);
    
    // Update PNBTR configuration
    pnbtr_jellie::PnbtrJellieEngine::PnbtrConfig pnbtr_config;
    pnbtr_config.prediction_strength = parameters_.pnbtr_strength.load();
    pnbtr_config.prediction_window_ms = static_cast<uint32_t>(parameters_.pnbtr_window.load());
    pnbtr_config.enable_reconstruction = true;
    pnbtr_config.enable_zero_noise_dither = true;
    
    tx_engine_->setPnbtrConfig(pnbtr_config);
    rx_engine_->setPnbtrConfig(pnbtr_config);
    
    // Update test configuration
    pnbtr_jellie::PnbtrJellieEngine::TestConfig test_config;
    test_config.enable_sine_generator = (parameters_.test_sine_enable.load() > 0.5f);
    test_config.sine_frequency_hz = parameters_.test_sine_freq.load();
    test_config.sine_amplitude = parameters_.test_sine_amplitude.load();
    test_config.enable_packet_loss_simulation = (parameters_.test_packet_loss_enable.load() > 0.5f);
    test_config.packet_loss_percentage = parameters_.test_packet_loss_percent.load();
    
    tx_engine_->setTestConfig(test_config);
    rx_engine_->setTestConfig(test_config);
}

// Update performance statistics
void PnbtrJellieVST3Minimal::updatePerformanceStats(double processing_time_us) {
    performance_stats_.current_latency_us.store(processing_time_us);
    
    // Update max latency
    double current_max = performance_stats_.max_latency_us.load();
    if (processing_time_us > current_max) {
        performance_stats_.max_latency_us.store(processing_time_us);
    }
    
    // Update running average (exponential moving average)
    double current_avg = performance_stats_.avg_latency_us.load();
    double new_avg = current_avg * 0.95 + processing_time_us * 0.05;
    performance_stats_.avg_latency_us.store(new_avg);
    
    // Get engine statistics
    const auto* current_engine = (current_mode_.load() == PluginMode::TX_MODE) ? 
                                 tx_engine_.get() : rx_engine_.get();
    
    if (current_engine) {
        const auto& engine_stats = current_engine->getPerformanceStats();
        performance_stats_.packets_sent.store(engine_stats.packets_sent.load());
        performance_stats_.packets_received.store(engine_stats.packets_received.load());
        performance_stats_.packets_lost.store(engine_stats.packets_lost.load());
        performance_stats_.current_snr_db.store(engine_stats.current_snr_db.load());
    }
}

// Reset performance statistics
void PnbtrJellieVST3Minimal::resetPerformanceStats() {
    performance_stats_.current_latency_us.store(0.0);
    performance_stats_.max_latency_us.store(0.0);
    performance_stats_.avg_latency_us.store(0.0);
    performance_stats_.current_snr_db.store(0.0f);
    performance_stats_.frames_processed.store(0);
    performance_stats_.packets_sent.store(0);
    performance_stats_.packets_received.store(0);
    performance_stats_.packets_lost.store(0);
    
    if (tx_engine_) tx_engine_->resetPerformanceStats();
    if (rx_engine_) rx_engine_->resetPerformanceStats();
}

// Convert channels to interleaved
void PnbtrJellieVST3Minimal::convertChannelsToInterleaved(const float* const* channels, 
                                                         float* interleaved, 
                                                         int num_samples, 
                                                         int num_channels) {
    for (int sample = 0; sample < num_samples; ++sample) {
        for (int channel = 0; channel < num_channels; ++channel) {
            interleaved[sample * num_channels + channel] = channels[channel][sample];
        }
    }
}

// Convert interleaved to channels
void PnbtrJellieVST3Minimal::convertInterleavedToChannels(const float* interleaved, 
                                                         float* const* channels, 
                                                         int num_samples, 
                                                         int num_channels) {
    for (int sample = 0; sample < num_samples; ++sample) {
        for (int channel = 0; channel < num_channels; ++channel) {
            channels[channel][sample] = interleaved[sample * num_channels + channel];
        }
    }
}

// C-style interface implementation
extern "C" {

PnbtrJellieVST3Minimal* createPnbtrJelliePlugin() {
    return new PnbtrJellieVST3Minimal();
}

void destroyPnbtrJelliePlugin(PnbtrJellieVST3Minimal* plugin) {
    delete plugin;
}

int processAudioBlock(PnbtrJellieVST3Minimal* plugin,
                     const float* const* input_channels,
                     float* const* output_channels,
                     int num_samples,
                     int num_channels) {
    if (!plugin) return 0;
    
    return plugin->processAudio(input_channels, output_channels, 
                               num_samples, num_channels) ? 1 : 0;
}

void setPluginParameter(PnbtrJellieVST3Minimal* plugin, int param_id, float value) {
    if (!plugin) return;
    
    auto& params = plugin->getParameters();
    
    switch (param_id) {
        case PARAM_PLUGIN_MODE:
            params.plugin_mode.store(value);
            break;
        case PARAM_NETWORK_PORT:
            params.network_port.store(value);
            break;
        case PARAM_NETWORK_QUALITY:
            params.network_quality.store(value);
            break;
        case PARAM_JELLIE_QUALITY:
            params.jellie_quality.store(value);
            break;
        case PARAM_PNBTR_STRENGTH:
            params.pnbtr_strength.store(value);
            break;
        case PARAM_PNBTR_WINDOW:
            params.pnbtr_window.store(value);
            break;
        case PARAM_TEST_SINE_ENABLE:
            params.test_sine_enable.store(value);
            break;
        case PARAM_TEST_SINE_FREQ:
            params.test_sine_freq.store(value);
            break;
        case PARAM_TEST_SINE_AMPLITUDE:
            params.test_sine_amplitude.store(value);
            break;
        case PARAM_TEST_PACKET_LOSS_ENABLE:
            params.test_packet_loss_enable.store(value);
            break;
        case PARAM_TEST_PACKET_LOSS_PERCENT:
            params.test_packet_loss_percent.store(value);
            break;
        default:
            break;
    }
}

float getPluginParameter(PnbtrJellieVST3Minimal* plugin, int param_id) {
    if (!plugin) return 0.0f;
    
    const auto& params = plugin->getParameters();
    
    switch (param_id) {
        case PARAM_PLUGIN_MODE:
            return params.plugin_mode.load();
        case PARAM_NETWORK_PORT:
            return params.network_port.load();
        case PARAM_NETWORK_QUALITY:
            return params.network_quality.load();
        case PARAM_JELLIE_QUALITY:
            return params.jellie_quality.load();
        case PARAM_PNBTR_STRENGTH:
            return params.pnbtr_strength.load();
        case PARAM_PNBTR_WINDOW:
            return params.pnbtr_window.load();
        case PARAM_TEST_SINE_ENABLE:
            return params.test_sine_enable.load();
        case PARAM_TEST_SINE_FREQ:
            return params.test_sine_freq.load();
        case PARAM_TEST_SINE_AMPLITUDE:
            return params.test_sine_amplitude.load();
        case PARAM_TEST_PACKET_LOSS_ENABLE:
            return params.test_packet_loss_enable.load();
        case PARAM_TEST_PACKET_LOSS_PERCENT:
            return params.test_packet_loss_percent.load();
        default:
            return 0.0f;
    }
}

const char* getPluginInfo(int info_type) {
    switch (info_type) {
        case 0: return PnbtrJellieVST3Minimal::getPluginName();
        case 1: return PnbtrJellieVST3Minimal::getPluginVersion();
        case 2: return PnbtrJellieVST3Minimal::getPluginVendor();
        case 3: return PnbtrJellieVST3Minimal::getPluginDescription();
        default: return "Unknown";
    }
}

} // extern "C" 