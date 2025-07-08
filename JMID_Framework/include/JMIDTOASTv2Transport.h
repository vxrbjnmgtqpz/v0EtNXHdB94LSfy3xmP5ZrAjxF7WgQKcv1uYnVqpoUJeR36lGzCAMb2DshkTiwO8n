#pragma once

/**
 * JMID TOAST v2 Transport - JAM Framework v2 Integration
 * 
 * Implements JMIDTransportInterface using TOAST v2 protocol from JAM Framework v2.
 * Enables JMID to participate in unified JAMNet ecosystem while preserving
 * all performance achievements from JMID modernization.
 * 
 * Features:
 * - Pure UDP multicast transport via TOAST v2
 * - Burst transmission with TOAST v2 burst fields
 * - Compact JMID format embedded in TOAST payloads
 * - Universal message routing integration
 * - Peer discovery and session management
 * - Performance monitoring and validation
 */

#include "JMIDTransportInterface.h"
#include "CompactJMIDFormat.h"
#include "SIMDJMIDParser.h"

// JAM Framework v2 includes
#include "../../JAM_Framework_v2/include/jam_toast.h"
#include "../../JAM_Framework_v2/include/message_router.h"

#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace JMID {

/**
 * JMID TOAST v2 Transport Implementation
 * 
 * Wraps JAM Framework v2 TOAST v2 protocol to provide JMID-compatible interface
 * while enabling participation in unified JAMNet transport layer.
 */
class JMIDTOASTv2Transport : public JMIDTransportInterface {
public:
    JMIDTOASTv2Transport();
    virtual ~JMIDTOASTv2Transport();
    
    // JMIDTransportInterface implementation
    bool initialize(const TransportConfig& config) override;
    void shutdown() override;
    bool isInitialized() const override;
    bool isRunning() const override;
    
    // Message transmission
    bool sendMessage(const std::string& compactJson, bool useBurst = true) override;
    bool sendMIDIMessage(std::unique_ptr<JMID::MIDIMessage> message, bool useBurst = true) override;
    bool sendRawMessage(const std::string& jsonMessage, bool useBurst = true) override;
    
    // Message reception
    void setMessageHandler(MessageHandler handler) override;
    void setErrorHandler(ErrorHandler handler) override;
    void setPeerCallback(PeerCallback callback) override;
    
    // Transport control
    bool startProcessing() override;
    void stopProcessing() override;
    
    // Configuration
    void setTransportConfig(const TransportConfig& config) override;
    TransportConfig getTransportConfig() const override;
    
    // Statistics and monitoring
    TransportStats getStats() const override;
    void resetStats() override;
    
    // Session management
    bool joinSession(uint32_t sessionId) override;
    void leaveSession() override;
    uint32_t getCurrentSessionId() const override;
    std::vector<std::string> getActivePeers() const override;
    
    // Performance tuning
    void setBurstConfig(int burstCount, int delayMicros) override;
    void setLatencyTarget(double targetMicros) override;
    void enablePrecisionTiming(bool enabled) override;
    
    // Transport information
    std::string getTransportType() const override;
    std::string getTransportVersion() const override;
    bool supportsFeature(const std::string& feature) const override;

private:
    // Core components
    std::unique_ptr<jam::TOASTv2Protocol> toast_;
    std::unique_ptr<jam::JAMMessageRouter> messageRouter_;
    std::unique_ptr<SIMDJMIDParser> parser_;
    
    // Configuration and state
    TransportConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    std::atomic<uint32_t> sessionId_{0};
    std::atomic<uint64_t> sequenceCounter_{0};
    
    // Callbacks
    MessageHandler messageHandler_;
    ErrorHandler errorHandler_;
    PeerCallback peerCallback_;
    
    // Statistics tracking
    mutable std::mutex statsMutex_;
    TransportStats stats_;
    std::chrono::high_resolution_clock::time_point startTime_;
    
    // Performance monitoring
    std::vector<double> latencyHistory_;
    std::vector<uint64_t> throughputHistory_;
    size_t maxHistorySize_ = 1000;
    
    // Peer management
    mutable std::mutex peersMutex_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> activePeers_;
    
    // Internal methods
    
    /**
     * TOAST v2 frame callbacks
     */
    void handleMIDIFrame(const jam::TOASTFrame& frame);
    void handleDiscoveryFrame(const jam::TOASTFrame& frame);
    void handleHeartbeatFrame(const jam::TOASTFrame& frame);
    void handleErrorFrame(const std::string& error);
    
    /**
     * Message processing
     */
    bool processIncomingFrame(const jam::TOASTFrame& frame);
    std::unique_ptr<JMID::MIDIMessage> parseCompactMessage(const std::string& compactJson);
    
    /**
     * TOAST v2 frame construction
     */
    jam::TOASTFrame createMIDIFrame(const std::string& compactJson, bool useBurst);
    void configureFrameHeader(jam::TOASTFrame& frame, bool useBurst);
    
    /**
     * Burst transmission management
     */
    bool sendWithBurst(const jam::TOASTFrame& frame);
    uint32_t generateBurstId();
    
    /**
     * Statistics management
     */
    void updateSentStats(const jam::TOASTFrame& frame);
    void updateReceivedStats(const jam::TOASTFrame& frame);
    void updateLatencyStats(double latencyMicros);
    void updateThroughputStats();
    
    /**
     * Performance monitoring
     */
    void recordLatency(double latencyMicros);
    void recordThroughput(uint64_t messagesPerSecond);
    bool validatePerformanceTargets() const;
    
    /**
     * Peer management
     */
    void addPeer(const std::string& peerId, const std::string& address);
    void removePeer(const std::string& peerId);
    void updatePeerHeartbeat(const std::string& peerId);
    void cleanupStalePeers();
    
    /**
     * Message routing integration
     */
    void setupMessageRouting();
    void handleRoutedMessage(const nlohmann::json& message);
    void sendToMessageRouter(const nlohmann::json& message);
    
    /**
     * Error handling
     */
    void handleInternalError(const std::string& error, int errorCode = 0);
    
    /**
     * Utility functions
     */
    uint64_t getCurrentMicroseconds() const;
    std::string generateUniqueId() const;
    
    // Constants
    static constexpr uint32_t PEER_TIMEOUT_MS = 30000;           // 30 second peer timeout
    static constexpr uint32_t STATS_UPDATE_INTERVAL_MS = 1000;  // 1 second stats update
    static constexpr double MAX_ACCEPTABLE_LATENCY_US = 50.0;    // 50μs latency target
    static constexpr uint64_t MIN_ACCEPTABLE_THROUGHPUT = 100000; // 100K msg/sec minimum
};

/**
 * TOAST v2 Transport Factory Implementation
 * 
 * Provides factory method for creating TOAST v2 transport instances
 */
class TOASTv2TransportFactory {
public:
    /**
     * Create TOAST v2 transport instance
     */
    static std::unique_ptr<JMIDTransportInterface> create(const TransportConfig& config);
    
    /**
     * Check if TOAST v2 is available
     */
    static bool isAvailable();
    
    /**
     * Get TOAST v2 feature capabilities
     */
    static std::vector<std::string> getSupportedFeatures();
    
    /**
     * Validate configuration for TOAST v2
     */
    static bool validateConfig(const TransportConfig& config, std::string& errorMessage);
};

/**
 * Performance Validation for TOAST v2 Integration
 * 
 * Ensures JMID performance targets are maintained with TOAST v2 transport
 */
class TOASTv2PerformanceValidator {
public:
    struct ValidationResult {
        bool latencyTargetMet = false;        // <50μs total latency
        bool throughputTargetMet = false;     // >100K msg/sec
        bool packetLossTargetMet = false;     // >66% tolerance
        bool compressionTargetMet = false;    // 67% compression preserved
        bool parseTargetMet = false;          // <0.095μs parse time preserved
        
        double actualLatencyMicros = 0.0;
        uint64_t actualThroughput = 0;
        double actualPacketLossRate = 0.0;
        double actualCompressionRatio = 0.0;
        double actualParseTimeMicros = 0.0;
        
        std::string summary;
        std::vector<std::string> issues;
        std::vector<std::string> recommendations;
    };
    
    /**
     * Run comprehensive performance validation
     */
    static ValidationResult validatePerformance(JMIDTOASTv2Transport& transport, 
                                               size_t testMessageCount = 10000);
    
    /**
     * Run latency-specific validation
     */
    static ValidationResult validateLatency(JMIDTOASTv2Transport& transport,
                                          size_t testMessageCount = 1000);
    
    /**
     * Run throughput-specific validation
     */
    static ValidationResult validateThroughput(JMIDTOASTv2Transport& transport,
                                             std::chrono::seconds testDuration = std::chrono::seconds(10));
    
    /**
     * Run packet loss validation
     */
    static ValidationResult validatePacketLoss(JMIDTOASTv2Transport& transport,
                                              double simulatedLossRate = 0.66);
    
private:
    static void generateTestMessages(std::vector<std::string>& messages, size_t count);
    static double measureLatency(JMIDTOASTv2Transport& transport, const std::string& message);
    static uint64_t measureThroughput(JMIDTOASTv2Transport& transport, 
                                     const std::vector<std::string>& messages,
                                     std::chrono::seconds duration);
};

} // namespace JMID 