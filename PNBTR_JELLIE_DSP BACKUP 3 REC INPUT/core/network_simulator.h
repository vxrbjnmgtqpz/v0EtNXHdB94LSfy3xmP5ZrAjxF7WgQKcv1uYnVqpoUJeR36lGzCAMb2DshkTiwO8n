/*
 * PNBTR+JELLIE DSP - Network Simulation Engine
 * Phase 1: Simulated Network Environment Setup
 * 
 * Creates controlled network simulation to recreate delays, jitter, and packet loss
 * that occur in real networks for testing and training PNBTR without external network
 */

#pragma once

#include <vector>
#include <queue>
#include <map>
#include <random>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <functional>

namespace pnbtr_jellie {

struct NetworkConditions {
    // Base latency configuration (20-100ms typical)
    double base_latency_ms = 50.0;
    double jitter_variance_ms = 10.0;
    
    // Packet loss configuration (1-5% typical, up to 10% stress test)
    double packet_loss_percentage = 2.0;
    bool enable_burst_loss = true;
    double burst_duration_ms = 20.0;
    
    // Bandwidth simulation
    uint32_t bandwidth_kbps = 1000;  // 1 Mbps default
    bool enable_bandwidth_limiting = false;
    
    // Advanced network characteristics
    bool enable_out_of_order_delivery = true;
    double reorder_probability = 0.5;
    uint32_t max_reorder_distance = 3;
    
    // Continuous operation settings
    bool continuous_mode = true;
    uint64_t simulation_duration_hours = 24;  // Can run overnight
};

struct NetworkPacket {
    std::vector<uint8_t> data;
    uint64_t sequence_number;
    std::chrono::high_resolution_clock::time_point timestamp;
    std::chrono::high_resolution_clock::time_point scheduled_delivery;
    bool is_lost = false;
    bool is_reordered = false;
    uint32_t reorder_delay_samples = 0;
};

class NetworkSimulator {
public:
    NetworkSimulator();
    ~NetworkSimulator();
    
    // Lifecycle
    bool initialize(const NetworkConditions& conditions);
    void shutdown();
    bool isRunning() const { return is_running_.load(); }
    
    // Configuration
    void updateNetworkConditions(const NetworkConditions& conditions);
    const NetworkConditions& getCurrentConditions() const { return conditions_; }
    
    // Packet simulation
    bool sendPacket(const std::vector<uint8_t>& data, uint64_t sequence_number);
    bool receivePacket(NetworkPacket& packet);
    
    // Statistics and monitoring
    struct NetworkStats {
        std::atomic<uint64_t> packets_sent{0};
        std::atomic<uint64_t> packets_delivered{0};
        std::atomic<uint64_t> packets_lost{0};
        std::atomic<uint64_t> packets_reordered{0};
        std::atomic<double> average_latency_ms{0.0};
        std::atomic<double> jitter_ms{0.0};
        std::atomic<double> current_bandwidth_kbps{0.0};
    };
    
    const NetworkStats& getStats() const { return stats_; }
    void resetStats();
    
    // Event logging for training correlation
    struct NetworkEvent {
        std::chrono::high_resolution_clock::time_point timestamp;
        enum Type { PACKET_SENT, PACKET_DELIVERED, PACKET_LOST, LATENCY_SPIKE, JITTER_BURST } type;
        uint64_t sequence_number;
        double latency_ms;
        std::string description;
    };
    
    void enableEventLogging(bool enable) { log_events_.store(enable); }
    std::vector<NetworkEvent> getRecentEvents(size_t max_events = 1000);
    
    // Scenario presets for reproducible testing
    static NetworkConditions createLowLatencyScenario();    // 20ms, 1% loss
    static NetworkConditions createTypicalScenario();       // 50ms, 2% loss  
    static NetworkConditions createStressScenario();        // 100ms, 10% loss
    static NetworkConditions createJitterScenario();        // High jitter variance
    static NetworkConditions createBurstLossScenario();     // Periodic loss bursts

private:
    // Core simulation engine
    void simulationThread();
    void processPacketQueue();
    
    // Network impairment functions
    bool shouldDropPacket();
    double calculateLatency();
    bool shouldReorderPacket();
    void applyBandwidthLimiting();
    
    // Statistics tracking
    void updateLatencyStats(double latency_ms);
    void logNetworkEvent(NetworkEvent::Type type, uint64_t seq_num, 
                        double latency_ms = 0.0, const std::string& desc = "");
    
    // Configuration and state
    NetworkConditions conditions_;
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_initialized_{false};
    std::atomic<bool> log_events_{false};
    
    // Threading and synchronization
    std::thread simulation_thread_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    mutable std::mutex events_mutex_;
    
    // Packet queues
    std::queue<NetworkPacket> incoming_queue_;
    std::queue<NetworkPacket> outgoing_queue_;
    std::priority_queue<NetworkPacket> delayed_packets_;  // Sorted by delivery time
    
    // Statistics and monitoring
    NetworkStats stats_;
    std::vector<NetworkEvent> event_log_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
    
    // Random number generation for realistic impairments
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;
    std::exponential_distribution<double> exponential_dist_;
    
    // Bandwidth limiting
    std::chrono::high_resolution_clock::time_point last_bandwidth_check_;
    uint64_t bytes_sent_this_second_;
    
    // Burst loss state
    bool in_loss_burst_ = false;
    std::chrono::high_resolution_clock::time_point burst_start_time_;
    
    // Sequence tracking for reordering
    uint64_t expected_sequence_ = 0;
    std::map<uint64_t, NetworkPacket> reorder_buffer_;
};

// Utility functions for integration with PNBTR learning system
class NetworkConditionGenerator {
public:
    // Generate varied conditions for training data collection
    static std::vector<NetworkConditions> generateTrainingScenarios(size_t count = 100);
    
    // Create gradual transitions between network states for realistic testing
    static std::vector<NetworkConditions> createTransitionSequence(
        const NetworkConditions& start, 
        const NetworkConditions& end, 
        size_t steps = 10);
        
    // Load conditions from configuration file for reproducible experiments
    static bool loadConditionsFromFile(const std::string& filename, 
                                     std::vector<NetworkConditions>& conditions);
    static bool saveConditionsToFile(const std::string& filename, 
                                    const std::vector<NetworkConditions>& conditions);
};

// Integration helper for PNBTR+JELLIE engine
class NetworkSimulatorBridge {
public:
    NetworkSimulatorBridge(NetworkSimulator* simulator);
    
    // Audio packet simulation specifically for JELLIE encoding/decoding
    bool sendAudioPacket(const std::vector<float>& audio_data, 
                        uint32_t sample_rate, uint32_t channels);
    bool receiveAudioPacket(std::vector<float>& audio_data, 
                           uint32_t& sample_rate, uint32_t& channels);
    
    // Integration with PNBTR training pipeline
    void enablePnbtrTraining(bool enable) { training_mode_.store(enable); }
    bool isTrainingMode() const { return training_mode_.load(); }
    
    // Collect data for PNBTR learning
    struct AudioPacketEvent {
        std::chrono::high_resolution_clock::time_point timestamp;
        std::vector<float> original_audio;
        std::vector<float> received_audio;
        bool was_lost;
        double latency_ms;
        uint32_t sequence_number;
    };
    
    std::vector<AudioPacketEvent> getTrainingData(size_t max_events = 10000);
    void clearTrainingData();

private:
    NetworkSimulator* simulator_;
    std::atomic<bool> training_mode_{false};
    std::vector<AudioPacketEvent> training_events_;
    mutable std::mutex training_mutex_;
    uint32_t audio_sequence_counter_ = 0;
};

} // namespace pnbtr_jellie
