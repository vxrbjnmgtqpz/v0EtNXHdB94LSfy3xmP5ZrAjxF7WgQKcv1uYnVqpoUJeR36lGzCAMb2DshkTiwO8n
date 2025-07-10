#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>

// Forward declarations for JAM Framework integration
class JAMFrameworkIntegration;

/**
 * TOAST Network Oscilloscope Component
 * 
 * Visualizes TOAST protocol network activity with:
 * - Real-time packet transmission visualization
 * - Network performance metrics (latency, packet loss, jitter)
 * - UDP multicast peer connectivity status
 * - Transport command synchronization indicators
 * 
 * Integrates with JAM Framework v2 TOAST protocol implementation.
 */
class TOASTNetworkOscilloscope : public juce::Component, public juce::Timer
{
public:
    struct NetworkMetrics {
        double latency_us = 0.0;
        double throughput_mbps = 0.0;
        double packet_loss_rate = 0.0;
        double jitter_ms = 0.0;
        int active_peers = 0;
        bool is_connected = false;
        std::string session_name = "";
        std::string multicast_address = "";
        int udp_port = 0;
    };

    TOASTNetworkOscilloscope();
    ~TOASTNetworkOscilloscope() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

    // Network integration
    void setJAMFrameworkIntegration(JAMFrameworkIntegration* integration);
    void startNetworkVisualization();
    void stopNetworkVisualization();
    
    // Configuration
    void setSessionName(const std::string& sessionName);
    void setMulticastAddress(const std::string& address, int port);
    
    // Metrics access
    NetworkMetrics getCurrentMetrics() const { 
        std::lock_guard<std::mutex> lock(metricsMutex);
        return currentMetrics; 
    }
    
    // Packet visualization
    void addPacketEvent(const std::string& eventType, uint32_t timestamp, bool isOutgoing);
    void addTransportEvent(const std::string& command, uint64_t timestamp);

private:
    // JAM Framework integration
    JAMFrameworkIntegration* jamIntegration = nullptr;
    
    // Network metrics (thread-safe with mutex since NetworkMetrics has strings)
    NetworkMetrics currentMetrics;
    mutable std::mutex metricsMutex;
    
    // Visualization data
    struct PacketEvent {
        std::string type;
        uint32_t timestamp;
        bool isOutgoing;
        juce::Colour colour;
        float intensity;
    };
    
    struct TransportEvent {
        std::string command;
        uint64_t timestamp;
        juce::Colour colour;
    };
    
    std::vector<PacketEvent> recentPackets;
    std::vector<TransportEvent> recentTransportEvents;
    static constexpr int MAX_PACKET_HISTORY = 100;
    static constexpr int MAX_TRANSPORT_HISTORY = 20;
    
    // Visual components
    juce::Rectangle<int> headerArea;
    juce::Rectangle<int> waveformArea;
    juce::Rectangle<int> metricsArea;
    
    // Update methods
    void updateNetworkMetrics();
    void drawNetworkWaveform(juce::Graphics& g, juce::Rectangle<int> area);
    void drawMetricsDisplay(juce::Graphics& g, juce::Rectangle<int> area);
    void drawConnectionStatus(juce::Graphics& g, juce::Rectangle<int> area);
    void drawPacketActivity(juce::Graphics& g, juce::Rectangle<int> area);
    void drawTransportSync(juce::Graphics& g, juce::Rectangle<int> area);
    
    // Callback handlers
    void onNetworkStatusChanged(const std::string& status, bool connected);
    void onPerformanceUpdated(double latency_us, double throughput_mbps, int active_peers);
    void onTransportCommand(const std::string& command, uint64_t timestamp, double position, double bpm);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TOASTNetworkOscilloscope)
}; 