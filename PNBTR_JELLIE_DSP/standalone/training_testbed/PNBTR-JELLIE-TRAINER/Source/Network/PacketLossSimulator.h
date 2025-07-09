/*
  ==============================================================================

    PacketLossSimulator.h
    Created: Packet loss simulation for network training

    Generates realistic packet loss patterns for PNBTR training:
    - Configurable loss percentage and jitter
    - Burst loss patterns (realistic network behavior)
    - Integration with Metal GPU buffers

  ==============================================================================
*/

#pragma once

#include <atomic>
#include <vector>
#include <memory>
#include <random>
#include <functional>

// TOAST v2 Simulation Types (for training - not real network)
struct SimulatedTOASTFrame {
    std::vector<uint8_t> payload;
    uint64_t timestamp_us;
    uint32_t sequence_number;
};

struct SimulatedBurstConfig {
    uint8_t burst_size = 3;
    uint16_t jitter_window_us = 500;
    bool enable_redundancy = true;
    uint8_t max_retries = 0;
};

/**
 * TOAST-Based Network Simulator
 * 
 * Replaces the old stubbed PacketLossSimulator with real TOAST v2 protocol integration.
 * Provides actual UDP multicast transmission with configurable packet loss and jitter.
 */
class PacketLossSimulator
{
public:
    PacketLossSimulator();
    ~PacketLossSimulator();

    // Initialize TOAST protocol for real network simulation
    bool initialize(const std::string& multicast_addr = "239.255.77.88", 
                   uint16_t port = 9988);
    
    // Shutdown TOAST protocol
    void shutdown();

    // Configuration methods
    void prepare(double sampleRate, int samplesPerBlock);
    void generateLossMap(float lossPercentage, float jitterMs);
    void setPacketLossPercentage(float percentage);
    void setJitterAmount(float jitterMs);
    
    // TOAST-based audio transmission with loss simulation
    bool transmitAudioChunk(const std::vector<float>& audioData, 
                           uint64_t timestamp_us,
                           uint32_t sample_rate = 48000,
                           uint8_t channels = 2);
    
    // Receive processed audio from TOAST (after loss/jitter simulation)
    bool receiveAudioChunk(std::vector<float>& audioData);
    
    // Network statistics  
    struct NetworkStats {
        uint64_t packets_sent = 0;
        uint64_t packets_received = 0;
        uint64_t packets_lost = 0;
        uint64_t burst_packets_sent = 0;
        double actual_loss_rate = 0.0;
        double average_jitter_ms = 0.0;
        uint32_t active_sessions = 0;
    };
    
    NetworkStats getStats() const;
    void resetStats();
    
    // Legacy compatibility methods (for existing PNBTRTrainer code)
    void getLossMap(int* dest, int n) const {
        // Convert modern loss tracking to legacy format
        for (int i = 0; i < n; ++i) {
            dest[i] = (i < static_cast<int>(lossMap_.size())) ? (lossMap_[i] ? 1 : 0) : 0;
        }
    }

private:
    // TOAST v2 simulation state (for training)
    bool toast_initialized_ = false;
    std::vector<SimulatedTOASTFrame> simulated_network_buffer_;
    
    // Network simulation parameters
    std::atomic<float> packetLossPercentage_{2.0f};
    std::atomic<float> jitterAmountMs_{1.0f};
    std::atomic<bool> networkActive_{false};
    
    // Session configuration
    uint32_t sessionId_;
    std::string multicastAddress_;
    uint16_t port_;
    
    // Audio processing parameters
    double currentSampleRate_ = 48000.0;
    int currentBlockSize_ = 512;
    
    // Loss simulation state
    std::vector<bool> lossMap_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> lossDist_;
    std::uniform_real_distribution<float> jitterDist_;
    
    // Network statistics
    mutable NetworkStats stats_;
    
    // Audio buffer management for TOAST transmission/reception
    std::vector<float> transmitBuffer_;
    std::vector<float> receiveBuffer_;
    std::atomic<bool> hasNewAudio_{false};
    
    // TOAST simulation callbacks
    void handleAudioFrame(const SimulatedTOASTFrame& frame);
    void handleRealAudioFrame(const SimulatedTOASTFrame& frame);
    void handleNetworkError(const std::string& error);
    
    // Loss simulation logic
    bool shouldDropPacket();
    uint32_t calculateJitterDelay();
    void updateLossMap(int blockSize);
    
    // Utility methods
    uint64_t getCurrentTimestamp();
    void initializeSession();
    
    // Burst transmission configuration
    SimulatedBurstConfig createBurstConfig() const;
}; 