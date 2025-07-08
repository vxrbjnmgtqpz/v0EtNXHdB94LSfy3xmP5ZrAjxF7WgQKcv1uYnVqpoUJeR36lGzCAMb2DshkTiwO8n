#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

namespace pnbtr_jellie {

/**
 * PNBTR+JELLIE DSP Standalone Engine
 * 
 * Dual-mode network audio interface:
 * - TX Mode: Audio Input → JELLIE encoding → Network Output
 * - RX Mode: Network Input → JELLIE decoding → PNBTR reconstruction → Audio Output
 * 
 * Performance Target: <100μs end-to-end processing
 */
class PnbtrJellieEngine {
public:
    // Construction/Destruction
    PnbtrJellieEngine();
    virtual ~PnbtrJellieEngine();

    // Core Engine Interface
    bool initialize(uint32_t sample_rate, uint16_t block_size);
    void terminate();
    bool isInitialized() const { return initialized_.load(); }
    
    // Audio Processing Interface (main processing call)
    bool processAudio(const float* input, float* output, uint32_t sample_count, uint32_t channels);
    
    // Plugin Mode Management
    enum class PluginMode {
        TX_MODE,    // Transmit: Audio Input → Network Output
        RX_MODE     // Receive: Network Input → Audio Output
    };
    
    void setPluginMode(PluginMode mode);
    PluginMode getPluginMode() const { return current_mode_.load(); }

    // Network Configuration
    struct NetworkConfig {
        std::string target_ip = "239.255.0.1";
        uint16_t target_port = 7777;
        bool enable_multicast = true;
        uint32_t packet_size = 1024;
        float network_quality = 1.0f;
        uint8_t redundancy_level = 4;  // ADAT 4-stream redundancy
    };
    
    void setNetworkConfig(const NetworkConfig& config);
    const NetworkConfig& getNetworkConfig() const { return network_config_; }

    // JELLIE Processing Configuration
    struct JellieConfig {
        uint32_t sample_rate = 48000;
        uint16_t bit_depth = 24;
        uint8_t channels = 2;
        bool enable_redundancy = true;
        float encoding_quality = 1.0f;
        uint32_t frame_size_samples = 64;
    };
    
    void setJellieConfig(const JellieConfig& config);
    const JellieConfig& getJellieConfig() const { return jellie_config_; }

    // PNBTR Configuration
    struct PnbtrConfig {
        bool enable_reconstruction = true;
        float prediction_strength = 0.8f;
        uint32_t prediction_window_ms = 50;
        bool enable_zero_noise_dither = true;
        float quality_threshold = 0.95f;
    };
    
    void setPnbtrConfig(const PnbtrConfig& config);
    const PnbtrConfig& getPnbtrConfig() const { return pnbtr_config_; }

    // Testing & Monitoring
    struct TestConfig {
        bool enable_sine_generator = false;
        float sine_frequency_hz = 440.0f;
        float sine_amplitude = 0.5f;
        bool enable_packet_loss_simulation = false;
        float packet_loss_percentage = 5.0f;
        bool enable_latency_monitoring = true;
    };
    
    void setTestConfig(const TestConfig& config);
    const TestConfig& getTestConfig() const { return test_config_; }

    // Performance Monitoring
    struct PerformanceStats {
        std::atomic<uint64_t> frames_processed{0};
        std::atomic<uint64_t> packets_sent{0};
        std::atomic<uint64_t> packets_received{0};
        std::atomic<uint64_t> packets_lost{0};
        std::atomic<double> avg_processing_time_us{0.0};
        std::atomic<double> max_processing_time_us{0.0};
        std::atomic<double> current_latency_us{0.0};
        std::atomic<float> current_snr_db{0.0f};
    };
    
    const PerformanceStats& getPerformanceStats() const { return stats_; }
    void resetPerformanceStats();

private:
    // Core Processing Functions
    bool processTXMode(const float* input, float* output, uint32_t sample_count, uint32_t channels);
    bool processRXMode(const float* input, float* output, uint32_t sample_count, uint32_t channels);
    
    // JELLIE Processing
    bool initializeJellieEncoder();
    bool initializeJellieDecoder();
    void encodeAudioToJellie(const float* input, uint32_t sample_count, uint32_t channels);
    void decodeJellieToAudio(float* output, uint32_t sample_count, uint32_t channels);
    
    // PNBTR Processing
    bool initializePnbtrEngine();
    void reconstructAudioWithPnbtr(float* audio, uint32_t sample_count, uint32_t channels);
    
    // Network Processing
    bool initializeNetworking();
    void sendJelliePackets(const std::vector<uint8_t>& packet_data);
    std::vector<uint8_t> receiveJelliePackets();
    
    // Test Signal Generation
    void generateSineWave(float* output, uint32_t sample_count, uint32_t channels);
    void simulatePacketLoss(std::vector<uint8_t>& packet_data);
    
    // Performance Monitoring
    void updatePerformanceStats(double processing_time_us);
    std::chrono::high_resolution_clock::time_point getHighResTime() const;
    
    // Configuration
    std::atomic<PluginMode> current_mode_{PluginMode::TX_MODE};
    NetworkConfig network_config_;
    JellieConfig jellie_config_;
    PnbtrConfig pnbtr_config_;
    TestConfig test_config_;
    
    // Engine State
    std::atomic<bool> initialized_{false};
    std::atomic<bool> is_processing_{false};
    uint32_t sample_rate_ = 48000;
    uint16_t block_size_ = 512;
    
    // Performance Monitoring
    mutable PerformanceStats stats_;
    
    // Processing Buffers
    std::vector<float> temp_audio_buffer_;
    std::vector<uint8_t> jellie_packet_buffer_;
    std::vector<float> pnbtr_reconstruction_buffer_;
    
    // Sine Wave Generator State
    std::atomic<double> sine_phase_{0.0};
    
    // Network State
    std::atomic<bool> network_initialized_{false};
    std::atomic<bool> is_connected_{false};
};

// Simplified Parameter System for Testing
struct PnbtrJellieParams {
    static constexpr int kPluginMode = 0;
    static constexpr int kNetworkTargetPort = 1;
    static constexpr int kNetworkQuality = 2;
    static constexpr int kJellieEncodingQuality = 3;
    static constexpr int kPnbtrReconstructionStrength = 4;
    static constexpr int kPnbtrPredictionWindow = 5;
    static constexpr int kTestSineEnable = 6;
    static constexpr int kTestSineFrequency = 7;
    static constexpr int kTestSineAmplitude = 8;
    static constexpr int kTestPacketLossEnable = 9;
    static constexpr int kTestPacketLossPercentage = 10;
    static constexpr int kMonitorLatency = 11;
    static constexpr int kMonitorSNR = 12;
    static constexpr int kMonitorPacketLoss = 13;
    
    static constexpr int kNumParameters = 14;
};

} // namespace pnbtr_jellie 