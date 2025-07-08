/*
 * PNBTR+JELLIE VST3 Plugin - Main Header
 * Defines the core engine class and configuration structures
 */

#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>

namespace pnbtr_jellie {

class PnbtrJellieEngine {
public:
    enum class PluginMode {
        TX_MODE,  // Transmit - Audio input → JELLIE encoding → network
        RX_MODE   // Receive - Network → JELLIE decoding → PNBTR reconstruction
    };

    struct NetworkConfig {
        std::string multicast_address = "239.255.0.1";
        int port = 8888;
        int buffer_size_kb = 64;
        bool enable_multicast = true;
    };

    struct JellieConfig {
        int sample_rate = 48000;
        int bit_depth = 24;
        int num_channels = 2;
        bool enable_8_channel_redundancy = true;
        bool enable_adat_mode = false;
    };

    struct PnbtrConfig {
        double prediction_strength = 0.75;
        bool enable_reconstruction = true;
        int prediction_window_ms = 50;
        bool enable_zero_noise_dither = true;
        bool enable_neural_extrapolation = true;
    };

    struct TestConfig {
        bool enable_sine_generator = true;
        double sine_frequency_hz = 440.0;
        bool enable_packet_loss_simulation = true;
        double packet_loss_percentage = 5.0;
        bool enable_quality_analysis = true;
    };

    struct PerformanceStats {
        std::atomic<double> avg_latency_us{0.0};
        std::atomic<double> max_latency_us{0.0};
        std::atomic<double> snr_improvement_db{0.0};
        std::atomic<int> packets_sent{0};
        std::atomic<int> packets_received{0};
        std::atomic<int> packets_lost{0};
        std::atomic<double> reconstruction_success_rate{0.0};
    };

    PnbtrJellieEngine();
    ~PnbtrJellieEngine();

    // Core initialization
    bool initialize(int sample_rate, int buffer_size);
    void shutdown();

    // Configuration
    void setPluginMode(PluginMode mode);
    void setNetworkConfig(const NetworkConfig& config);
    void setJellieConfig(const JellieConfig& config);
    void setPnbtrConfig(const PnbtrConfig& config);
    void setTestConfig(const TestConfig& config);

    // Audio processing
    void processAudio(const float* input_buffer, float* output_buffer, int num_samples);

    // Performance monitoring
    const PerformanceStats& getPerformanceStats() const;
    void resetPerformanceStats();

    // Network simulation
    void simulateNetworkTransmission();
    void simulatePacketLoss();

    // Test utilities
    void generateSineWave(float* buffer, int num_samples, double frequency);
    double calculateSNR(const float* original, const float* processed, int num_samples);

private:
    PluginMode m_mode;
    NetworkConfig m_network_config;
    JellieConfig m_jellie_config;
    PnbtrConfig m_pnbtr_config;
    TestConfig m_test_config;
    PerformanceStats m_stats;

    // Audio processing state
    int m_sample_rate;
    int m_buffer_size;
    bool m_initialized;
    
    // Sine wave generator state
    double m_sine_phase;
    
    // Performance timing
    std::chrono::high_resolution_clock::time_point m_process_start;
    std::vector<double> m_latency_history;
    
    // Thread safety
    mutable std::mutex m_config_mutex;
    mutable std::mutex m_stats_mutex;

    // Internal processing methods
    void processTransmitMode(const float* input, float* output, int num_samples);
    void processReceiveMode(const float* input, float* output, int num_samples);
    void simulateJellieEncoding(const float* input, float* encoded, int num_samples);
    void simulateJellieDecoding(const float* encoded, float* decoded, int num_samples);
    void applyPnbtrReconstruction(float* audio, int num_samples);
    void updatePerformanceStats(double latency_us);
};

} // namespace pnbtr_jellie 