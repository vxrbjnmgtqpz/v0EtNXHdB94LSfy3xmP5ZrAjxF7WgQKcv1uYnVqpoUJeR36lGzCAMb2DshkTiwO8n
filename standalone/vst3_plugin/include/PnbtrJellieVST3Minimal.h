#pragma once

#include "PnbtrJelliePlugin.h"
#include <memory>
#include <string>
#include <atomic>

/**
 * Minimal VST3 Plugin Wrapper (No JUCE dependencies)
 * 
 * Lightweight wrapper around our proven PnbtrJellieEngine core
 * Can be loaded by DAWs that support minimal VST3 interfaces
 */
class PnbtrJellieVST3Minimal {
public:
    // Constructor/Destructor
    PnbtrJellieVST3Minimal();
    ~PnbtrJellieVST3Minimal();

    // Core plugin interface
    bool initialize(double sample_rate, int block_size, int num_channels);
    void shutdown();
    bool isInitialized() const { return initialized_.load(); }

    // Audio processing (main entry point)
    bool processAudio(const float* const* input_channels, 
                     float* const* output_channels, 
                     int num_samples, 
                     int num_channels);

    // Plugin mode management
    enum class PluginMode {
        TX_MODE = 0,    // Transmit: Audio Input → Network Output
        RX_MODE = 1     // Receive: Network Input → Audio Output
    };
    
    void setPluginMode(PluginMode mode);
    PluginMode getPluginMode() const;

    // Parameter management (simplified)
    struct Parameters {
        std::atomic<float> plugin_mode{0.0f};          // 0.0 = TX, 1.0 = RX
        std::atomic<float> network_port{7777.0f};      // Network port
        std::atomic<float> network_quality{1.0f};      // 0.0-1.0
        std::atomic<float> jellie_quality{1.0f};       // 0.0-1.0
        std::atomic<float> pnbtr_strength{0.8f};       // 0.0-1.0
        std::atomic<float> pnbtr_window{50.0f};        // 1-100ms
        std::atomic<float> test_sine_enable{0.0f};     // 0.0 = off, 1.0 = on
        std::atomic<float> test_sine_freq{440.0f};     // Hz
        std::atomic<float> test_sine_amplitude{0.5f};  // 0.0-1.0
        std::atomic<float> test_packet_loss_enable{0.0f}; // 0.0 = off, 1.0 = on
        std::atomic<float> test_packet_loss_percent{5.0f}; // 0-100%
    };
    
    Parameters& getParameters() { return parameters_; }
    const Parameters& getParameters() const { return parameters_; }

    // Performance monitoring
    struct PerformanceStats {
        std::atomic<double> current_latency_us{0.0};
        std::atomic<double> max_latency_us{0.0};
        std::atomic<double> avg_latency_us{0.0};
        std::atomic<float> current_snr_db{0.0f};
        std::atomic<uint64_t> frames_processed{0};
        std::atomic<uint64_t> packets_sent{0};
        std::atomic<uint64_t> packets_received{0};
        std::atomic<uint64_t> packets_lost{0};
    };
    
    const PerformanceStats& getPerformanceStats() const { return performance_stats_; }
    void resetPerformanceStats();

    // Plugin information
    static const char* getPluginName() { return "PNBTR+JELLIE Network Audio"; }
    static const char* getPluginVersion() { return "1.0.0"; }
    static const char* getPluginVendor() { return "JAMNet"; }
    static const char* getPluginDescription() { return "Revolutionary network audio interface with sub-50μs latency"; }
    
    // VST3 Plugin identification
    static constexpr uint32_t getPluginUID1() { return 0x4A414D6E; } // 'JAMn'
    static constexpr uint32_t getPluginUID2() { return 0x506E4A6C; } // 'PnJl'

    // Parameter count for VST3 host
    static constexpr int getNumParameters() { return 11; }

private:
    // Core engines (our proven performers!)
    std::unique_ptr<pnbtr_jellie::PnbtrJellieEngine> tx_engine_;
    std::unique_ptr<pnbtr_jellie::PnbtrJellieEngine> rx_engine_;
    
    // Plugin state
    std::atomic<bool> initialized_{false};
    std::atomic<PluginMode> current_mode_{PluginMode::TX_MODE};
    
    // Audio configuration
    double sample_rate_ = 44100.0;
    int block_size_ = 512;
    int num_channels_ = 2;
    
    // Parameters and monitoring
    Parameters parameters_;
    mutable PerformanceStats performance_stats_;
    
    // Processing buffers
    std::vector<float> interleaved_input_buffer_;
    std::vector<float> interleaved_output_buffer_;
    
    // Helper methods
    void updateEngineConfiguration();
    void updatePerformanceStats(double processing_time_us);
    void convertInterleavedToChannels(const float* interleaved, 
                                     float* const* channels, 
                                     int num_samples, 
                                     int num_channels);
    void convertChannelsToInterleaved(const float* const* channels, 
                                     float* interleaved, 
                                     int num_samples, 
                                     int num_channels);
};

// C-style interface for VST3 host integration
extern "C" {
    // Factory function for VST3 host
    PnbtrJellieVST3Minimal* createPnbtrJelliePlugin();
    void destroyPnbtrJelliePlugin(PnbtrJellieVST3Minimal* plugin);
    
    // Basic VST3 interface functions
    int processAudioBlock(PnbtrJellieVST3Minimal* plugin,
                         const float* const* input_channels,
                         float* const* output_channels,
                         int num_samples,
                         int num_channels);
    
    void setPluginParameter(PnbtrJellieVST3Minimal* plugin, int param_id, float value);
    float getPluginParameter(PnbtrJellieVST3Minimal* plugin, int param_id);
    
    const char* getPluginInfo(int info_type); // 0=name, 1=version, 2=vendor, 3=description
}

// Parameter IDs for minimal VST3 wrapper
enum MinimalParameterIDs {
    PARAM_PLUGIN_MODE = 0,
    PARAM_NETWORK_PORT = 1,
    PARAM_NETWORK_QUALITY = 2,
    PARAM_JELLIE_QUALITY = 3,
    PARAM_PNBTR_STRENGTH = 4,
    PARAM_PNBTR_WINDOW = 5,
    PARAM_TEST_SINE_ENABLE = 6,
    PARAM_TEST_SINE_FREQ = 7,
    PARAM_TEST_SINE_AMPLITUDE = 8,
    PARAM_TEST_PACKET_LOSS_ENABLE = 9,
    PARAM_TEST_PACKET_LOSS_PERCENT = 10
}; 