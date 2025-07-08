#pragma once

/**
 * JMID Transport Interface - Universal Transport Abstraction
 * 
 * Provides unified interface for JMID to work with multiple transport layers:
 * - Current UDP Burst Transport (fire-and-forget)
 * - TOAST v2 Protocol (universal JAMNet transport)
 * - Future transport implementations
 * 
 * Enables seamless migration from custom UDP to TOAST v2 while preserving
 * all performance achievements from JMID modernization.
 */

#include "JMIDMessage.h"
#include <string>
#include <memory>
#include <functional>
#include <cstdint>
#include <vector>

namespace JMID {

/**
 * Transport Configuration
 */
struct TransportConfig {
    // Network settings
    std::string multicastGroup = "239.255.77.77";
    uint16_t port = 7777;
    uint32_t sessionId = 0;
    
    // Performance settings
    bool enableBurstTransmission = true;
    int burstCount = 3;
    int burstDelayMicros = 10;
    int maxPacketSize = 1024;
    
    // TOAST v2 specific settings
    bool enableDiscovery = true;
    bool enableHeartbeat = true;
    uint32_t heartbeatIntervalMs = 5000;
    
    // Timing settings
    bool enablePrecisionTiming = true;
    bool enableLatencyMeasurement = true;
};

/**
 * Transport Statistics
 */
struct TransportStats {
    // Message counters
    uint64_t messagesSent = 0;
    uint64_t messagesReceived = 0;
    uint64_t duplicatesReceived = 0;
    uint64_t burstPacketsSent = 0;
    uint64_t checksumErrors = 0;
    uint64_t invalidPackets = 0;
    
    // Performance metrics
    double averageLatencyMicros = 0.0;
    double minLatencyMicros = 0.0;
    double maxLatencyMicros = 0.0;
    uint64_t messagesPerSecond = 0;
    uint64_t bytesPerSecond = 0;
    
    // Network health
    double packetLossRate = 0.0;
    uint32_t activePeers = 0;
    bool isConnected = false;
    
    // JMID-specific metrics
    double compressionRatio = 0.0;
    double parseTimeAvgMicros = 0.0;
    uint64_t deduplicationCount = 0;
};

/**
 * Message Handler Callback
 */
using MessageHandler = std::function<void(std::unique_ptr<JMID::MIDIMessage>)>;

/**
 * Error Handler Callback
 */
using ErrorHandler = std::function<void(const std::string& error, int errorCode)>;

/**
 * Peer Discovery Callback
 */
using PeerCallback = std::function<void(const std::string& peerId, const std::string& address, bool connected)>;

/**
 * Universal JMID Transport Interface
 * 
 * Provides transport-agnostic interface for JMID framework to send/receive
 * MIDI messages over various transport layers while preserving performance.
 */
class JMIDTransportInterface {
public:
    virtual ~JMIDTransportInterface() = default;
    
    /**
     * Lifecycle Management
     */
    virtual bool initialize(const TransportConfig& config) = 0;
    virtual void shutdown() = 0;
    virtual bool isInitialized() const = 0;
    virtual bool isRunning() const = 0;
    
    /**
     * Message Transmission
     * 
     * @param compactJson Ultra-compact JMID format message
     * @param useBurst Whether to use burst transmission for reliability
     * @return true if message was sent successfully
     */
    virtual bool sendMessage(const std::string& compactJson, bool useBurst = true) = 0;
    
    /**
     * Convenience methods for common MIDI messages
     */
    virtual bool sendMIDIMessage(std::unique_ptr<JMID::MIDIMessage> message, bool useBurst = true) = 0;
    virtual bool sendRawMessage(const std::string& jsonMessage, bool useBurst = true) = 0;
    
    /**
     * Message Reception
     */
    virtual void setMessageHandler(MessageHandler handler) = 0;
    virtual void setErrorHandler(ErrorHandler handler) = 0;
    virtual void setPeerCallback(PeerCallback callback) = 0;
    
    /**
     * Transport Control
     */
    virtual bool startProcessing() = 0;
    virtual void stopProcessing() = 0;
    
    /**
     * Configuration
     */
    virtual void setTransportConfig(const TransportConfig& config) = 0;
    virtual TransportConfig getTransportConfig() const = 0;
    
    /**
     * Statistics and Monitoring
     */
    virtual TransportStats getStats() const = 0;
    virtual void resetStats() = 0;
    
    /**
     * Session Management
     */
    virtual bool joinSession(uint32_t sessionId) = 0;
    virtual void leaveSession() = 0;
    virtual uint32_t getCurrentSessionId() const = 0;
    virtual std::vector<std::string> getActivePeers() const = 0;
    
    /**
     * Performance Tuning
     */
    virtual void setBurstConfig(int burstCount, int delayMicros) = 0;
    virtual void setLatencyTarget(double targetMicros) = 0;
    virtual void enablePrecisionTiming(bool enabled) = 0;
    
    /**
     * Transport Information
     */
    virtual std::string getTransportType() const = 0;
    virtual std::string getTransportVersion() const = 0;
    virtual bool supportsFeature(const std::string& feature) const = 0;
};

/**
 * Transport Factory - Creates appropriate transport implementation
 */
class JMIDTransportFactory {
public:
    enum class TransportType {
        UDP_BURST,      // Current JMID UDP implementation
        TOAST_V2,       // JAM Framework v2 TOAST v2 protocol
        AUTO_SELECT     // Automatically select best available
    };
    
    /**
     * Create transport instance
     * 
     * @param type Desired transport type
     * @param config Initial configuration
     * @return Transport instance or nullptr if unavailable
     */
    static std::unique_ptr<JMIDTransportInterface> create(
        TransportType type, 
        const TransportConfig& config = TransportConfig{}
    );
    
    /**
     * Get available transport types
     */
    static std::vector<TransportType> getAvailableTransports();
    
    /**
     * Get recommended transport for current environment
     */
    static TransportType getRecommendedTransport();
    
    /**
     * Transport type utilities
     */
    static std::string transportTypeToString(TransportType type);
    static TransportType stringToTransportType(const std::string& str);
};

/**
 * Transport Performance Monitor
 * 
 * Tracks transport performance and provides recommendations
 */
class TransportPerformanceMonitor {
private:
    std::vector<double> latencyHistory_;
    std::vector<uint64_t> throughputHistory_;
    size_t maxHistorySize_ = 1000;
    
public:
    /**
     * Update performance metrics
     */
    void recordLatency(double latencyMicros);
    void recordThroughput(uint64_t messagesPerSecond);
    
    /**
     * Performance analysis
     */
    double getAverageLatency() const;
    double getLatencyJitter() const;
    uint64_t getAverageThroughput() const;
    bool isPerformanceAcceptable() const;
    
    /**
     * Performance recommendations
     */
    struct PerformanceRecommendation {
        bool adjustBurstCount = false;
        int recommendedBurstCount = 3;
        bool adjustBurstDelay = false;
        int recommendedDelayMicros = 10;
        bool switchTransport = false;
        JMIDTransportFactory::TransportType recommendedTransport;
        std::string reasoning;
    };
    
    PerformanceRecommendation getRecommendation() const;
    
    /**
     * Reset monitoring data
     */
    void reset();
};

/**
 * Transport Event Types
 */
enum class TransportEvent {
    CONNECTED,
    DISCONNECTED,
    PEER_JOINED,
    PEER_LEFT,
    SESSION_STARTED,
    SESSION_ENDED,
    ERROR_OCCURRED,
    PERFORMANCE_WARNING
};

/**
 * Transport Event Callback
 */
using TransportEventCallback = std::function<void(TransportEvent event, const std::string& details)>;

} // namespace JMID 