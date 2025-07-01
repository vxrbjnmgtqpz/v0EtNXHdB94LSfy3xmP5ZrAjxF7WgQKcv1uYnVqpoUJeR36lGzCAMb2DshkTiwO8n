#pragma once

/*
 * TOAST: Transport Oriented Audio Synchronization Tunnel
 * 
 * Ultra-low-latency TCP-based transport protocol for real-time MIDI streaming
 * with distributed clock synchronization and sub-10ms network latency.
 */

#include "JSONMIDIMessage.h"
#include "JSONMIDIParser.h"
#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

namespace TOAST {

/**
 * TOAST Message Types
 */
enum class MessageType : uint8_t {
    MIDI = 0x01,
    CLOCK_SYNC = 0x02,
    SESSION_CONTROL = 0x03,
    HEARTBEAT = 0x04,
    CONNECTION_HANDSHAKE = 0x05,
    ERROR = 0x06,
    METADATA = 0x07
};

/**
 * TOAST Protocol Frame Structure
 */
struct FrameHeader {
    uint32_t frameLength;      // Total frame size including header
    MessageType messageType;   // Message type identifier
    uint8_t reserved[3];       // Reserved for future use
    uint64_t masterTimestamp;  // Master clock timestamp (microseconds)
    uint32_t sequenceNumber;   // Message sequence number
    uint32_t checksum;         // CRC32 checksum of payload
    
    static constexpr size_t HEADER_SIZE = 24;
};

/**
 * Connection State
 */
enum class ConnectionState : uint8_t {
    DISCONNECTED = 0,
    CONNECTING = 1,
    HANDSHAKING = 2,
    CONNECTED = 3,
    SYNCHRONIZING = 4,
    ACTIVE = 5,
    ERROR_STATE = 6
};

/**
 * Client Information
 */
struct ClientInfo {
    std::string clientId;
    std::string name;
    std::string version;
    std::vector<std::string> capabilities;
    uint64_t connectTime;
};

/**
 * Session Control Actions
 */
enum class SessionAction : uint8_t {
    JOIN = 0,
    LEAVE = 1,
    START = 2,
    STOP = 3,
    PAUSE = 4,
    RESUME = 5
};

/**
 * TOAST Transport Message
 */
class TransportMessage {
public:
    TransportMessage(MessageType type, const std::string& payload, 
                    uint64_t timestamp = 0, uint32_t sequenceNumber = 0);
    
    MessageType getType() const { return header_.messageType; }
    uint64_t getTimestamp() const { return header_.masterTimestamp; }
    uint32_t getSequenceNumber() const { return header_.sequenceNumber; }
    const std::string& getPayload() const { return payload_; }
    
    void setTimestamp(uint64_t timestamp) { header_.masterTimestamp = timestamp; }
    void setSequenceNumber(uint32_t seq) { header_.sequenceNumber = seq; }
    
    // Serialize to wire format
    std::vector<uint8_t> serialize() const;
    
    // Deserialize from wire format
    static std::unique_ptr<TransportMessage> deserialize(const std::vector<uint8_t>& data);
    
    // Validate checksum
    bool validateChecksum() const;

private:
    FrameHeader header_;
    std::string payload_;
    
    uint32_t calculateChecksum() const;
};

/**
 * Connection Manager
 */
class ConnectionManager {
public:
    ConnectionManager();
    ~ConnectionManager();
    
    // Connection management
    bool startServer(uint16_t port);
    bool connectToServer(const std::string& hostname, uint16_t port);
    void disconnect();
    
    ConnectionState getState() const { return state_; }
    std::string getSessionId() const { return sessionId_; }
    
    // Message handling
    using MessageHandler = std::function<void(std::unique_ptr<TransportMessage>)>;
    void setMessageHandler(MessageHandler handler);
    
    // Send message
    bool sendMessage(std::unique_ptr<TransportMessage> message);
    
    // Client management
    std::vector<ClientInfo> getConnectedClients() const;
    bool isClient(const std::string& clientId) const;
    
    // Statistics
    struct NetworkStats {
        uint64_t bytesSent = 0;
        uint64_t bytesReceived = 0;
        uint64_t messagesSent = 0;
        uint64_t messagesReceived = 0;
        uint64_t connectionsAccepted = 0;
        double averageLatency = 0.0;
        uint32_t packetLoss = 0;
    };
    
    NetworkStats getNetworkStats() const { return stats_; }
    void resetNetworkStats();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    std::atomic<ConnectionState> state_{ConnectionState::DISCONNECTED};
    std::string sessionId_;
    MessageHandler messageHandler_;
    mutable NetworkStats stats_;
};

/**
 * Forward declaration of ClockDriftArbiter
 * Full implementation is in ClockDriftArbiter.h
 */
class ClockDriftArbiter;

/**
 * TOAST Protocol Handler
 */
class ProtocolHandler {
public:
    ProtocolHandler(ConnectionManager& connectionManager, ClockDriftArbiter& clockArbiter);
    ~ProtocolHandler();
    
    // Protocol operations
    void start();
    void stop();
    
    // MIDI message handling
    void sendMIDIMessage(std::unique_ptr<JSONMIDI::MIDIMessage> message);
    using MIDIMessageHandler = std::function<void(std::unique_ptr<JSONMIDI::MIDIMessage>)>;
    void setMIDIMessageHandler(MIDIMessageHandler handler) { midiHandler_ = handler; }
    
    // Session control
    void joinSession(const std::string& sessionId, const ClientInfo& clientInfo);
    void leaveSession();
    void startSession();
    void stopSession();
    
    // Heartbeat and keepalive
    void enableHeartbeat(bool enabled, uint32_t intervalMs = 1000);
    
    // Error handling
    using ErrorHandler = std::function<void(const std::string& error)>;
    void setErrorHandler(ErrorHandler handler) { errorHandler_ = handler; }

private:
    ConnectionManager& connectionManager_;
    ClockDriftArbiter& clockArbiter_;
    
    MIDIMessageHandler midiHandler_;
    ErrorHandler errorHandler_;
    
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> processingThread_;
    std::atomic<uint32_t> sequenceNumber_{0};
    
    // Message processing
    void processMessage(std::unique_ptr<TransportMessage> message);
    void handleMIDIMessage(const TransportMessage& message);
    void handleClockSyncMessage(const TransportMessage& message);
    void handleSessionControlMessage(const TransportMessage& message);
    void handleHeartbeatMessage(const TransportMessage& message);
    
    // Utility methods
    uint32_t getNextSequenceNumber() { return sequenceNumber_++; }
    uint64_t getCurrentTimestamp() const;
};

/**
 * TOAST Session Manager
 */
class SessionManager {
public:
    SessionManager();
    ~SessionManager();
    
    // Session lifecycle
    std::string createSession(const std::string& sessionName);
    bool joinSession(const std::string& sessionId, const ClientInfo& clientInfo);
    void leaveSession(const std::string& clientId);
    void destroySession(const std::string& sessionId);
    
    // Session state
    struct SessionInfo {
        std::string sessionId;
        std::string name;
        std::vector<ClientInfo> clients;
        uint64_t createdTime;
        bool isActive;
    };
    
    std::vector<SessionInfo> getActiveSessions() const;
    std::optional<SessionInfo> getSession(const std::string& sessionId) const;
    
    // MIDI routing
    void routeMIDIMessage(const std::string& sessionId, 
                         std::unique_ptr<JSONMIDI::MIDIMessage> message,
                         const std::string& sourceClientId = "");

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace TOAST
