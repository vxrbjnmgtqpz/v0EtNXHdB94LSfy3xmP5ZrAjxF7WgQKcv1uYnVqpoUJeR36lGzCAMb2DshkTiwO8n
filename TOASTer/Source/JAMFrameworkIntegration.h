#pragma once

/**
 * JAM Framework v2 Integration for TOASTer
 * 
 * This module provides the bridge between JUCE-based TOASTer application
 * and the JAM Framework v2 UDP-native TOAST protocol implementation.
 */

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include "../../JAM_Framework_v2/include/gpu_native/gpu_timebase.h"  // GPU-native timebase
#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>

// JAM Framework v2 and PNBTR integration
#include "PNBTRManager.h"

// Forward declarations for JAM Framework v2
namespace jam {
    class TOASTv2Protocol;
    struct TOASTFrame;
    struct BurstConfig;
    class ComputePipeline;
    class GPUManager;
    class NetworkStateDetector;
}

/**
 * JAM Framework v2 Integration Manager
 * 
 * Provides high-level interface for TOASTer to use JAM Framework v2
 * features including UDP multicast, GPU acceleration, and PNBTR prediction.
 */
class JAMFrameworkIntegration : public juce::Timer {
public:
    
    // Callback function types
    using MIDICallback = std::function<void(uint8_t status, uint8_t data1, uint8_t data2, uint32_t timestamp)>;
    using AudioCallback = std::function<void(const float* samples, int numSamples, uint32_t timestamp)>;
    using VideoCallback = std::function<void(const uint8_t* frameData, int width, int height, uint32_t timestamp)>;
    using StatusCallback = std::function<void(const std::string& status, bool isConnected)>;
    using PerformanceCallback = std::function<void(double latency_us, double throughput_mbps, int active_peers)>;
    using TransportCallback = std::function<void(const std::string& command, uint64_t timestamp, double position, double bpm)>;
    
    JAMFrameworkIntegration();
    ~JAMFrameworkIntegration();
    
    // === Network Management ===
    
    /**
     * Initialize JAM Framework v2 with UDP multicast
     * 
     * @param multicast_addr Multicast address (default: "239.255.77.77")
     * @param port UDP port (default: 7777)
     * @param session_name Session identifier
     * @return true if initialization successful
     */
    bool initialize(const std::string& multicast_addr = "239.255.77.77", 
                   int port = 7777, 
                   const std::string& session_name = "TOASTer_Session");
    
    /**
     * Start UDP multicast networking
     */
    bool startNetwork();
    
    /**
     * Start network with bypassed connectivity tests (for direct connections)
     */
    bool startNetworkDirect();
    
    /**
     * Stop networking and cleanup
     */
    void stopNetwork();
    
    /**
     * Get connection status
     */
    bool isConnected() const { return networkActive; }
    
    /**
     * Get number of active peers
     */
    int getActivePeerCount() const { return activePeers; }
    
    // === GPU Backend Management ===
    
    /**
     * Check if GPU backend is available (always true in GPU-native architecture)
     */
    bool isGPUAvailable() const { return jam::gpu_native::GPUTimebase::is_initialized(); }
    
    // === MIDI Transmission ===
    
    /**
     * Send MIDI event with burst transmission
     * 
     * @param status MIDI status byte
     * @param data1 MIDI data byte 1
     * @param data2 MIDI data byte 2
     * @param use_burst Enable burst transmission (3-5 packets)
     */
    void sendMIDIEvent(uint8_t status, uint8_t data1, uint8_t data2, bool use_burst = true);
    
    /**
     * Send raw MIDI data buffer
     */
    void sendMIDIData(const uint8_t* data, size_t size, bool use_burst = true);
    
    // === Audio Streaming ===
    
    /**
     * Send audio data with PNBTR prediction
     * 
     * @param samples PCM audio samples (32-bit float)
     * @param numSamples Number of samples
     * @param sampleRate Sample rate (e.g., 44100)
     * @param enablePrediction Use PNBTR audio prediction
     */
    void sendAudioData(const float* samples, int numSamples, int sampleRate, bool enablePrediction = true);
    
    // === Video Streaming ===
    
    /**
     * Send video frame with PNBTR-JVID prediction
     * 
     * @param frameData Raw video frame data
     * @param width Frame width
     * @param height Frame height
     * @param format Pixel format (RGB, RGBA, etc.)
     * @param enablePrediction Use PNBTR-JVID video prediction
     */
    void sendVideoFrame(const uint8_t* frameData, int width, int height, 
                       const std::string& format = "RGB24", bool enablePrediction = true);
    
    // === PNBTR Prediction ===
    
    /**
     * Enable/disable PNBTR audio prediction
     */
    void setPNBTRAudioPrediction(bool enabled) { 
        pnbtrAudioEnabled = enabled; 
        if (pnbtrManager) {
            pnbtrManager->setAudioPredictionEnabled(enabled);
        }
    }
    
    /**
     * Enable/disable PNBTR-JVID video prediction  
     */
    void setPNBTRVideoPrediction(bool enabled) { 
        pnbtrVideoEnabled = enabled; 
        if (pnbtrManager) {
            pnbtrManager->setVideoPredictionEnabled(enabled);
        }
    }
    
    /**
     * Get PNBTR prediction confidence (0.0 - 1.0)
     */
    double getPredictionConfidence() const { 
        if (pnbtrManager) {
            auto stats = pnbtrManager->getStatistics();
            return (stats.averageAudioConfidence + stats.averageVideoConfidence) / 2.0;
        }
        return predictionConfidence; 
    }
    
    // === Callback Registration ===
    
    void setMIDICallback(MIDICallback callback) { midiCallback = callback; }
    void setAudioCallback(AudioCallback callback) { audioCallback = callback; }
    void setVideoCallback(VideoCallback callback) { videoCallback = callback; }
    void setStatusCallback(StatusCallback callback) { statusCallback = callback; }
    void setPerformanceCallback(PerformanceCallback callback) { performanceCallback = callback; }
    void setTransportCallback(TransportCallback callback) { transportCallback = callback; }
    
    // === Configuration ===
    
    /**
     * Configure burst transmission settings
     */
    void setBurstConfig(int burstSize, int jitterWindow_us, bool enableRedundancy);
    
    /**
     * Set prediction buffer size (affects latency vs accuracy)
     */
    void setPredictionBufferSize(int samples) { predictionBufferSize = samples; }
    
    /**
     * Get current network performance metrics
     */
    struct PerformanceMetrics {
        double latency_us = 0.0;
        double throughput_mbps = 0.0;
        double packet_loss_rate = 0.0;
        int active_peers = 0;
        double prediction_accuracy = 0.0;
    };
    
    PerformanceMetrics getPerformanceMetrics() const { return currentMetrics; }
    
    // === Transport Control ===
    
    /**
     * Send transport command (play/stop/position/bpm) to all peers
     * 
     * @param command Transport command ("PLAY", "STOP", "POSITION", "BPM")
     * @param timestamp Current timestamp in microseconds
     * @param position Current playback position (optional)
     * @param bpm Current BPM (optional)
     */
    void sendTransportCommand(const std::string& command, uint64_t timestamp, 
                             double position = 0.0, double bpm = 120.0);
    
    /**
     * Handle incoming transport command from network
     */
    void handleTransportCommand(const std::string& command, uint64_t timestamp, 
                               double position, double bpm);
    
    // === Auto-Discovery and Auto-Connection ===
    
    /**
     * Enable automatic peer discovery and connection
     * When enabled, will automatically connect to discovered peers
     */
    void enableAutoConnection(bool enable) { auto_connection_enabled_ = enable; }
    
    /**
     * Set minimum peers for session start (auto-starts when reached)
     */
    void setMinimumPeers(int min_peers) { minimum_peers_ = min_peers; }
    
    /**
     * Get discovered but not yet connected peers
     */
    std::vector<std::string> getDiscoveredPeers() const;
    
private:
    // JAM Framework v2 components
    std::unique_ptr<jam::TOASTv2Protocol> toastProtocol;
    std::unique_ptr<jam::ComputePipeline> gpuPipeline;
    std::unique_ptr<jam::GPUManager> gpuManager;
    std::unique_ptr<jam::NetworkStateDetector> networkStateDetector;
    
    // PNBTR Prediction Manager
    std::unique_ptr<PNBTRManager> pnbtrManager;
    
    // State management
    bool networkActive = false;
    bool pnbtrAudioEnabled = true;
    bool pnbtrVideoEnabled = true;
    std::atomic<bool> auto_connection_enabled_{true}; // Always auto-connect for seamless experience
    int activePeers = 0;
    int minimum_peers_ = 1;  // Start session as soon as one peer is found
    int predictionBufferSize = 256;
    double predictionConfidence = 0.0;
    
    // Performance tracking
    PerformanceMetrics currentMetrics;
    
    // Session info
    std::string sessionName;
    std::string multicastAddress;
    int udpPort = 7777;
    
    // Callbacks
    MIDICallback midiCallback;
    AudioCallback audioCallback;
    VideoCallback videoCallback;
    StatusCallback statusCallback;
    PerformanceCallback performanceCallback;
    TransportCallback transportCallback;
    
    // Auto-discovery and connection
    std::vector<std::string> discovered_peers_;
    mutable std::mutex discovered_peers_mutex_;

    // Internal methods
    void handleIncomingFrame(const jam::TOASTFrame& frame);
    void updatePerformanceMetrics();
    void notifyStatusChange(const std::string& status, bool connected);
    
    // Timer callback for periodic updates
    void timerCallback() override;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(JAMFrameworkIntegration)
};
