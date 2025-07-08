/*
 * PNBTR+JELLIE VST3 Plugin - Engine Implementation
 * Simplified implementation for GUI demonstration
 */

#include "../include/PnbtrJelliePlugin.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

namespace pnbtr_jellie {

PnbtrJellieEngine::PnbtrJellieEngine() 
    : m_mode(PluginMode::TX_MODE)
    , m_sample_rate(48000)
    , m_buffer_size(512)
    , m_initialized(false)
    , m_sine_phase(0.0) 
{
    // Initialize configurations with defaults
    m_network_config = NetworkConfig{};
    m_jellie_config = JellieConfig{};
    m_pnbtr_config = PnbtrConfig{};
    m_test_config = TestConfig{};
    
    // Initialize performance stats
    m_stats.avg_latency_us = 0.0;
    m_stats.max_latency_us = 0.0;
    m_stats.snr_improvement_db = 0.0;
    m_stats.packets_sent = 0;
    m_stats.packets_received = 0;
    m_stats.packets_lost = 0;
    m_stats.reconstruction_success_rate = 0.0;
}

PnbtrJellieEngine::~PnbtrJellieEngine() {
    shutdown();
}

bool PnbtrJellieEngine::initialize(int sample_rate, int buffer_size) {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    
    if (m_initialized) {
        return true;
    }
    
    m_sample_rate = sample_rate;
    m_buffer_size = buffer_size;
    m_sine_phase = 0.0;
    
    // Initialize configurations
    m_jellie_config.sample_rate = sample_rate;
    
    // Reserve space for latency history
    m_latency_history.reserve(1000);
    
    m_initialized = true;
    return true;
}

void PnbtrJellieEngine::shutdown() {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    m_initialized = false;
}

void PnbtrJellieEngine::setPluginMode(PluginMode mode) {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    m_mode = mode;
}

void PnbtrJellieEngine::setNetworkConfig(const NetworkConfig& config) {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    m_network_config = config;
}

void PnbtrJellieEngine::setJellieConfig(const JellieConfig& config) {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    m_jellie_config = config;
}

void PnbtrJellieEngine::setPnbtrConfig(const PnbtrConfig& config) {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    m_pnbtr_config = config;
}

void PnbtrJellieEngine::setTestConfig(const TestConfig& config) {
    std::lock_guard<std::mutex> lock(m_config_mutex);
    m_test_config = config;
}

void PnbtrJellieEngine::processAudio(const float* input_buffer, float* output_buffer, int num_samples) {
    if (!m_initialized) {
        // Pass through if not initialized
        for (int i = 0; i < num_samples; ++i) {
            output_buffer[i] = input_buffer[i];
        }
        return;
    }
    
    m_process_start = std::chrono::high_resolution_clock::now();
    
    if (m_mode == PluginMode::TX_MODE) {
        processTransmitMode(input_buffer, output_buffer, num_samples);
    } else {
        processReceiveMode(input_buffer, output_buffer, num_samples);
    }
    
    // Update performance stats
    auto process_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(process_end - m_process_start);
    updatePerformanceStats(duration.count());
}

const PnbtrJellieEngine::PerformanceStats& PnbtrJellieEngine::getPerformanceStats() const {
    return m_stats;
}

void PnbtrJellieEngine::resetPerformanceStats() {
    std::lock_guard<std::mutex> lock(m_stats_mutex);
    m_stats.avg_latency_us = 0.0;
    m_stats.max_latency_us = 0.0;
    m_stats.snr_improvement_db = 0.0;
    m_stats.packets_sent = 0;
    m_stats.packets_received = 0;
    m_stats.packets_lost = 0;
    m_stats.reconstruction_success_rate = 0.0;
    m_latency_history.clear();
}

void PnbtrJellieEngine::simulateNetworkTransmission() {
    // Simulate network activity
    m_stats.packets_sent++;
    
    // Simulate occasional packet loss
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 100.0);
    
    if (dis(gen) < m_test_config.packet_loss_percentage) {
        m_stats.packets_lost++;
    } else {
        m_stats.packets_received++;
    }
    
    // Update reconstruction success rate
    int total_packets = m_stats.packets_sent.load();
    int successful_packets = m_stats.packets_received.load();
    if (total_packets > 0) {
        m_stats.reconstruction_success_rate = (double(successful_packets) / total_packets) * 100.0;
    }
}

void PnbtrJellieEngine::simulatePacketLoss() {
    simulateNetworkTransmission();
}

void PnbtrJellieEngine::generateSineWave(float* buffer, int num_samples, double frequency) {
    // SINE WAVE GENERATOR DISABLED - ALWAYS USE REAL MICROPHONE INPUT
    // Fill with silence instead of sine wave
    for (int i = 0; i < num_samples; ++i) {
        buffer[i] = 0.0f; // Silent - no sine wave allowed!
    }
    
    // Reset phase to prevent any accumulated state
    m_sine_phase = 0.0;
}

double PnbtrJellieEngine::calculateSNR(const float* original, const float* processed, int num_samples) {
    double signal_power = 0.0;
    double noise_power = 0.0;
    
    for (int i = 0; i < num_samples; ++i) {
        signal_power += original[i] * original[i];
        double noise = processed[i] - original[i];
        noise_power += noise * noise;
    }
    
    if (noise_power < 1e-10) {
        return 100.0; // Very high SNR
    }
    
    return 10.0 * log10(signal_power / noise_power);
}

// Private methods
void PnbtrJellieEngine::processTransmitMode(const float* input, float* output, int num_samples) {
    // Simulate TX processing: Input -> JELLIE encoding -> Network simulation
    
    // ALWAYS use real microphone input - NO SINE WAVE GENERATOR
    for (int i = 0; i < num_samples; ++i) {
        output[i] = input[i]; // Pass through real microphone input
    }
    
    // Simulate network transmission
    simulateNetworkTransmission();
    
    // Apply gain and basic processing
    for (int i = 0; i < num_samples; ++i) {
        output[i] *= 1.0f; // Unity gain for real input
    }
    
    // Update SNR stats with realistic values for mic input
    m_stats.snr_improvement_db = 5.0 + (rand() % 30) / 10.0; // 5-8 dB improvement
}

void PnbtrJellieEngine::processReceiveMode(const float* input, float* output, int num_samples) {
    // Simulate RX processing: Network -> JELLIE decoding -> PNBTR reconstruction
    
    // ALWAYS use real microphone input - NO SINE WAVE GENERATOR
    for (int i = 0; i < num_samples; ++i) {
        output[i] = input[i]; // Start with real microphone input
    }
    
    // Simulate packet loss and PNBTR reconstruction
    simulatePacketLoss();
    applyPnbtrReconstruction(output, num_samples);
    
    // Update SNR stats with PNBTR improvement
    m_stats.snr_improvement_db = 7.0 + (rand() % 20) / 10.0; // 7-9 dB PNBTR improvement
}

void PnbtrJellieEngine::simulateJellieEncoding(const float* input, float* encoded, int num_samples) {
    // Simulate JELLIE 8-channel encoding with slight processing
    for (int i = 0; i < num_samples; ++i) {
        encoded[i] = input[i] * 0.95f; // Slight attenuation to simulate encoding
    }
}

void PnbtrJellieEngine::simulateJellieDecoding(const float* encoded, float* decoded, int num_samples) {
    // Simulate JELLIE decoding with packet loss recovery
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 100.0);
    
    for (int i = 0; i < num_samples; ++i) {
        if (dis(gen) < m_test_config.packet_loss_percentage) {
            // Simulate packet loss - use previous sample
            decoded[i] = (i > 0) ? decoded[i-1] : 0.0f;
        } else {
            decoded[i] = encoded[i];
        }
    }
}

void PnbtrJellieEngine::applyPnbtrReconstruction(float* audio, int num_samples) {
    // Simulate PNBTR reconstruction with quality improvement
    for (int i = 0; i < num_samples; ++i) {
        // Apply subtle smoothing to simulate neural reconstruction
        if (i > 0 && i < num_samples - 1) {
            audio[i] = audio[i] * 0.6f + audio[i-1] * 0.2f + audio[i+1] * 0.2f;
        }
    }
    
    // Update SNR improvement
    m_stats.snr_improvement_db = 7.0 + 2.0 * sin(m_sine_phase * 0.1);
}

void PnbtrJellieEngine::updatePerformanceStats(double latency_us) {
    std::lock_guard<std::mutex> lock(m_stats_mutex);
    
    // Update latency history
    m_latency_history.push_back(latency_us);
    if (m_latency_history.size() > 100) {
        m_latency_history.erase(m_latency_history.begin());
    }
    
    // Calculate average latency
    double sum = 0.0;
    for (double lat : m_latency_history) {
        sum += lat;
    }
    m_stats.avg_latency_us = sum / m_latency_history.size();
    
    // Update max latency
    if (latency_us > m_stats.max_latency_us.load()) {
        m_stats.max_latency_us = latency_us;
    }
}

} // namespace pnbtr_jellie 