#pragma once

#include "JMIDMessage.h"
#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <mutex>
#include <cstdint>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
#endif

namespace JMID {

/**
 * UDP Burst Transport - Fire-and-Forget MIDI Transmission
 * 
 * Sends each MIDI message 3-5 times in rapid succession for reliability
 * without retransmission. Handles packet loss through burst redundancy.
 */
class UDPBurstTransport {
public:
    struct BurstConfig {
        int burstCount;           // Number of duplicate packets per message
        int burstDelayMicros;     // Delay between burst packets (μs)
        int maxPacketSize;        // Maximum UDP packet size
        std::string multicastGroup; // JMID multicast group
        uint16_t port;            // JMID default port
        
        BurstConfig() 
            : burstCount(3)
            , burstDelayMicros(10)
            , maxPacketSize(1024)
            , multicastGroup("239.255.77.77")
            , port(7777) {}
    };

    /**
     * Message handler for received MIDI messages
     */
    using MessageHandler = std::function<void(std::unique_ptr<JMID::MIDIMessage>)>;

public:
    explicit UDPBurstTransport(const BurstConfig& config = BurstConfig{});
    ~UDPBurstTransport();

    // Server operations
    bool startServer(uint16_t port = 0); // 0 = use config port
    bool startClient();
    void stop();

    // Message handling
    void setMessageHandler(MessageHandler handler) { messageHandler_ = handler; }

    // Send MIDI message with burst reliability
    bool sendMIDIMessage(std::unique_ptr<JMID::MIDIMessage> message);
    bool sendRawMessage(const std::string& jsonMessage);

    // Configuration
    void setBurstCount(int count) { config_.burstCount = count; }
    void setBurstDelay(int micros) { config_.burstDelayMicros = micros; }
    void setMulticastGroup(const std::string& group) { config_.multicastGroup = group; }

    // Statistics
    struct Stats {
        uint64_t messagesSent = 0;
        uint64_t messagesReceived = 0;
        uint64_t duplicatesReceived = 0;
        uint64_t burstPacketsSent = 0;
        double averageLatencyMicros = 0.0;
    };
    Stats getStats() const { return stats_; }
    void resetStats();

    // State
    bool isRunning() const { return running_.load(); }
    bool isServer() const { return isServer_; }

private:
    // Core networking
    bool initializeSocket();
    void cleanupSocket();
    bool joinMulticastGroup();
    
    // Message processing
    void receiveLoop();
    bool sendUDPPacket(const std::string& data);
    void processBurstMessage(const std::string& jsonMessage);
    
    // Sequence numbering for deduplication
    std::string addSequenceNumber(const std::string& jsonMessage);
    uint64_t extractSequenceNumber(const std::string& jsonMessage);
    bool isValidSequence(uint64_t sequence);
    
    // Timing utilities
    void microDelay(int microseconds);
    uint64_t getCurrentMicroseconds();

private:
    BurstConfig config_;
    MessageHandler messageHandler_;
    
    // Networking
    int socket_ = -1;
    struct sockaddr_in serverAddr_;
    struct sockaddr_in multicastAddr_;
    bool isServer_ = false;
    
    // Threading
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> receiveThread_;
    
    // Sequence tracking for deduplication
    std::atomic<uint64_t> sequenceCounter_{0};
    std::unordered_set<uint64_t> seenSequences_;
    mutable std::mutex sequenceMutex_;
    
    // Statistics
    mutable Stats stats_;
    mutable std::mutex statsMutex_;
    
    // Constants
    static constexpr int SEQUENCE_CACHE_SIZE = 10000;
    static constexpr uint64_t SEQUENCE_TIMEOUT_MICROS = 5000000; // 5 seconds
};

/**
 * Burst Deduplicator - Handles duplicate detection and timeline reconstruction
 */
class BurstDeduplicator {
public:
    /**
     * Process incoming message and return true if it's new (not a duplicate)
     */
    bool processMessage(const std::string& jsonMessage);
    
    /**
     * Get messages in chronological order
     */
    std::vector<std::string> getTimelineMessages(uint64_t fromTimestamp = 0);
    
    /**
     * Clear old sequences to prevent memory growth
     */
    void cleanupOldSequences(uint64_t olderThanMicros = 5000000);
    
    /**
     * Statistics
     */
    struct DeduplicationStats {
        uint64_t totalMessages = 0;
        uint64_t uniqueMessages = 0;
        uint64_t duplicatesFiltered = 0;
        double deduplicationRate = 0.0;
    };
    DeduplicationStats getStats() const { return stats_; }

private:
    struct MessageInfo {
        uint64_t sequence;
        uint64_t timestamp;
        std::string message;
    };
    
    std::unordered_set<uint64_t> seenSequences_;
    std::vector<MessageInfo> timeline_;
    mutable std::mutex mutex_;
    DeduplicationStats stats_;
    
    uint64_t extractSequenceNumber(const std::string& jsonMessage);
    uint64_t extractTimestamp(const std::string& jsonMessage);
};

} // namespace JMID 