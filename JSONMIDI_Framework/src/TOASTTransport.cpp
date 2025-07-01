// TOAST Transport implementation
// Phase 2.2: TCP Tunnel Implementation

#include "TOASTTransport.h"
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>

namespace TOAST {

// Placeholder implementations
TransportMessage::TransportMessage(MessageType type, const std::string& payload, 
                                 uint64_t timestamp, uint32_t sequenceNumber) 
    : payload_(payload) {
    header_.messageType = type;
    header_.masterTimestamp = timestamp;
    header_.sequenceNumber = sequenceNumber;
    header_.frameLength = FrameHeader::HEADER_SIZE + payload.size();
    header_.checksum = calculateChecksum();
}

std::vector<uint8_t> TransportMessage::serialize() const {
    std::vector<uint8_t> data;
    data.reserve(header_.frameLength);
    
    // Serialize frame header in correct order
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&header_.frameLength), 
                reinterpret_cast<const uint8_t*>(&header_.frameLength) + sizeof(header_.frameLength));
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&header_.messageType), 
                reinterpret_cast<const uint8_t*>(&header_.messageType) + sizeof(header_.messageType));
    
    // Add reserved bytes (padding)
    uint8_t reserved[3] = {0, 0, 0};
    data.insert(data.end(), reserved, reserved + 3);
    
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&header_.masterTimestamp), 
                reinterpret_cast<const uint8_t*>(&header_.masterTimestamp) + sizeof(header_.masterTimestamp));
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&header_.sequenceNumber), 
                reinterpret_cast<const uint8_t*>(&header_.sequenceNumber) + sizeof(header_.sequenceNumber));
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&header_.checksum), 
                reinterpret_cast<const uint8_t*>(&header_.checksum) + sizeof(header_.checksum));
    
    // Serialize payload
    data.insert(data.end(), payload_.begin(), payload_.end());
    
    return data;
}

std::unique_ptr<TransportMessage> TransportMessage::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < FrameHeader::HEADER_SIZE) {
        return nullptr; // Insufficient data
    }
    
    // Parse header in correct order
    FrameHeader header;
    size_t offset = 0;
    
    std::memcpy(&header.frameLength, data.data() + offset, sizeof(header.frameLength));
    offset += sizeof(header.frameLength);
    
    std::memcpy(&header.messageType, data.data() + offset, sizeof(header.messageType));
    offset += sizeof(header.messageType);
    
    // Skip reserved bytes
    offset += 3;
    
    std::memcpy(&header.masterTimestamp, data.data() + offset, sizeof(header.masterTimestamp));
    offset += sizeof(header.masterTimestamp);
    
    std::memcpy(&header.sequenceNumber, data.data() + offset, sizeof(header.sequenceNumber));
    offset += sizeof(header.sequenceNumber);
    
    std::memcpy(&header.checksum, data.data() + offset, sizeof(header.checksum));
    offset += sizeof(header.checksum);
    
    // Validate frame length
    if (data.size() < header.frameLength) {
        return nullptr; // Incomplete frame
    }
    
    // Extract payload (after header)
    size_t payloadSize = header.frameLength - FrameHeader::HEADER_SIZE;
    if (payloadSize > 0 && data.size() >= FrameHeader::HEADER_SIZE + payloadSize) {
        std::string payload(data.begin() + FrameHeader::HEADER_SIZE, 
                           data.begin() + FrameHeader::HEADER_SIZE + payloadSize);
        
        // Create message
        auto message = std::make_unique<TransportMessage>(
            static_cast<MessageType>(header.messageType), 
            payload, 
            header.masterTimestamp, 
            header.sequenceNumber
        );
        
        // Set the received header data
        message->header_ = header;
        
        if (!message->validateChecksum()) {
            std::cerr << "⚠️ Checksum validation failed" << std::endl;
            return nullptr; // Checksum validation failed
        }
        
        return message;
    }
    
    return nullptr;
}

bool TransportMessage::validateChecksum() const {
    uint32_t calculatedChecksum = calculateChecksum();
    return calculatedChecksum == header_.checksum;
}

uint32_t TransportMessage::calculateChecksum() const {
    // Simple CRC32 implementation for now
    // In production, use a proper CRC32 library
    uint32_t crc = 0xFFFFFFFF;
    
    // Hash header (excluding checksum field)
    const uint8_t* headerBytes = reinterpret_cast<const uint8_t*>(&header_);
    for (size_t i = 0; i < FrameHeader::HEADER_SIZE - sizeof(header_.checksum); ++i) {
        crc ^= headerBytes[i];
        for (int j = 0; j < 8; ++j) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    
    // Hash payload
    for (char byte : payload_) {
        crc ^= static_cast<uint8_t>(byte);
        for (int j = 0; j < 8; ++j) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    
    return crc ^ 0xFFFFFFFF;
}

// ConnectionManager implementation
class ConnectionManager::Impl {
public:
    std::atomic<bool> running{false};
    int serverSocket = -1;
    uint16_t serverPort = 0;
    std::thread acceptThread;
    std::unordered_map<std::string, int> clientSockets;
    std::unordered_map<std::string, std::thread> readerThreads;
    std::mutex clientsMutex;
    std::function<void(std::unique_ptr<TransportMessage>)> messageHandler;
    
    bool startServer(uint16_t port) {
        if (running.load()) {
            return false; // Already running
        }
        
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket < 0) {
            std::cerr << "❌ Failed to create server socket" << std::endl;
            return false;
        }
        
        // Set socket options
        int opt = 1;
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        struct sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(port);
        
        if (bind(serverSocket, reinterpret_cast<struct sockaddr*>(&serverAddr), sizeof(serverAddr)) < 0) {
            std::cerr << "❌ Failed to bind server socket to port " << port << std::endl;
            close(serverSocket);
            return false;
        }
        
        if (listen(serverSocket, 16) < 0) {  // 16+ concurrent clients as per roadmap
            std::cerr << "❌ Failed to listen on server socket" << std::endl;
            close(serverSocket);
            return false;
        }
        
        serverPort = port;
        running.store(true);
        
        // Start accept thread
        acceptThread = std::thread([this]() {
            while (running.load()) {
                struct sockaddr_in clientAddr{};
                socklen_t clientLen = sizeof(clientAddr);
                
                int clientSocket = accept(serverSocket, reinterpret_cast<struct sockaddr*>(&clientAddr), &clientLen);
                if (clientSocket >= 0) {
                    std::string clientId = std::string(inet_ntoa(clientAddr.sin_addr)) + ":" + std::to_string(ntohs(clientAddr.sin_port));
                    
                    std::lock_guard<std::mutex> lock(clientsMutex);
                    clientSockets[clientId] = clientSocket;
                    
                    // Start reader thread for this client
                    readerThreads[clientId] = std::thread([this, clientSocket, clientId]() {
                        readMessagesFromClient(clientSocket, clientId);
                    });
                    
                    std::cout << "🔗 TOAST client connected: " << clientId << std::endl;
                }
            }
        });
        
        std::cout << "🚀 TOAST server started on port " << port << std::endl;
        return true;
    }
    
    bool connectToServer(const std::string& address, uint16_t port, const std::string& clientId) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            return false;
        }
        
        struct sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        inet_pton(AF_INET, address.c_str(), &serverAddr.sin_addr);
        
        if (connect(sock, reinterpret_cast<struct sockaddr*>(&serverAddr), sizeof(serverAddr)) < 0) {
            close(sock);
            return false;
        }
        
        std::lock_guard<std::mutex> lock(clientsMutex);
        clientSockets[clientId] = sock;
        
        // Start reader thread for client connection
        readerThreads[clientId] = std::thread([this, sock, clientId]() {
            readMessagesFromClient(sock, clientId);
        });
        
        std::cout << "🔗 Connected to TOAST server: " << address << ":" << port << std::endl;
        return true;
    }
    
    void shutdown() {
        running.store(false);
        
        if (serverSocket >= 0) {
            close(serverSocket);
            serverSocket = -1;
        }
        
        if (acceptThread.joinable()) {
            acceptThread.join();
        }
        
        // Join all reader threads
        for (auto& [clientId, thread] : readerThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        readerThreads.clear();
        
        std::lock_guard<std::mutex> lock(clientsMutex);
        for (auto& [clientId, socket] : clientSockets) {
            close(socket);
        }
        clientSockets.clear();
        
        std::cout << "🛑 TOAST ConnectionManager shutdown complete" << std::endl;
    }
    
    void readMessagesFromClient(int socket, const std::string& clientId) {
        std::vector<uint8_t> buffer(4096);
        std::vector<uint8_t> messageBuffer;
        
        while (running.load()) {
            ssize_t bytesRead = recv(socket, buffer.data(), buffer.size(), 0);
            if (bytesRead <= 0) {
                // Connection closed or error
                std::cout << "🔌 Client disconnected: " << clientId << std::endl;
                break;
            }
            
            // Add received data to message buffer
            messageBuffer.insert(messageBuffer.end(), buffer.begin(), buffer.begin() + bytesRead);
            
            // Try to parse complete messages
            while (messageBuffer.size() >= FrameHeader::HEADER_SIZE) {
                // Read frame length from the first 4 bytes
                uint32_t frameLength;
                std::memcpy(&frameLength, messageBuffer.data(), sizeof(frameLength));
                
                if (messageBuffer.size() >= frameLength) {
                    // We have a complete message
                    std::vector<uint8_t> messageData(messageBuffer.begin(), messageBuffer.begin() + frameLength);
                    messageBuffer.erase(messageBuffer.begin(), messageBuffer.begin() + frameLength);
                    
                    // Deserialize and handle the message
                    auto message = TransportMessage::deserialize(messageData);
                    if (message && messageHandler) {
                        messageHandler(std::move(message));
                    }
                } else {
                    // Wait for more data
                    break;
                }
            }
        }
    }
    
    bool sendToSocket(int socket, const std::vector<uint8_t>& data) {
        size_t totalSent = 0;
        while (totalSent < data.size()) {
            ssize_t sent = send(socket, data.data() + totalSent, data.size() - totalSent, 0);
            if (sent <= 0) {
                return false;
            }
            totalSent += sent;
        }
        return true;
    }
};

ConnectionManager::ConnectionManager() : impl_(std::make_unique<Impl>()) {}
ConnectionManager::~ConnectionManager() {
    impl_->shutdown();
}

bool ConnectionManager::startServer(uint16_t port) {
    return impl_->startServer(port);
}

bool ConnectionManager::connectToServer(const std::string& hostname, uint16_t port) {
    return impl_->connectToServer(hostname, port, "default-client");
}

void ConnectionManager::disconnect() {
    impl_->shutdown();
}

void ConnectionManager::setMessageHandler(MessageHandler handler) {
    impl_->messageHandler = handler;
}

bool ConnectionManager::sendMessage(std::unique_ptr<TransportMessage> message) {
    auto serialized = message->serialize();
    bool success = true;
    
    std::lock_guard<std::mutex> lock(impl_->clientsMutex);
    for (const auto& [clientId, socket] : impl_->clientSockets) {
        if (!impl_->sendToSocket(socket, serialized)) {
            std::cout << "⚠️ Failed to send message to client: " << clientId << std::endl;
            success = false;
        }
    }
    
    return success;
}

std::vector<ClientInfo> ConnectionManager::getConnectedClients() const {
    std::vector<ClientInfo> clients;
    std::lock_guard<std::mutex> lock(impl_->clientsMutex);
    
    for (const auto& [clientId, socket] : impl_->clientSockets) {
        ClientInfo info;
        info.clientId = clientId;
        info.name = "TOAST Client";
        info.version = "1.0.0";
        info.connectTime = 0; // TODO: Track actual connect time
        clients.push_back(info);
    }
    
    return clients;
}

bool ConnectionManager::isClient(const std::string& clientId) const {
    std::lock_guard<std::mutex> lock(impl_->clientsMutex);
    return impl_->clientSockets.find(clientId) != impl_->clientSockets.end();
}

void ConnectionManager::resetNetworkStats() {
    // TODO: Implement network statistics tracking and reset
    std::cout << "📊 Network statistics reset" << std::endl;
}

// ProtocolHandler implementation
ProtocolHandler::ProtocolHandler(ConnectionManager& connectionManager, ClockDriftArbiter& clockArbiter)
    : connectionManager_(connectionManager), clockArbiter_(clockArbiter) {}

ProtocolHandler::~ProtocolHandler() = default;

void ProtocolHandler::start() {
    // TODO: Implement protocol handler startup
}

void ProtocolHandler::stop() {
    // TODO: Implement protocol handler shutdown
}

void ProtocolHandler::sendMIDIMessage(std::unique_ptr<JSONMIDI::MIDIMessage> message) {
    // TODO: Implement MIDI message sending
}

void ProtocolHandler::joinSession(const std::string& sessionId, const ClientInfo& clientInfo) {
    // TODO: Implement session joining
}

void ProtocolHandler::leaveSession() {
    // TODO: Implement session leaving
}

void ProtocolHandler::startSession() {
    // TODO: Implement session start
}

void ProtocolHandler::stopSession() {
    // TODO: Implement session stop
}

void ProtocolHandler::enableHeartbeat(bool enabled, uint32_t intervalMs) {
    // TODO: Implement heartbeat mechanism
}

uint64_t ProtocolHandler::getCurrentTimestamp() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

// SessionManager placeholder
class SessionManager::Impl {
public:
    // TODO: Implement session management
};

SessionManager::SessionManager() : impl_(std::make_unique<Impl>()) {}
SessionManager::~SessionManager() = default;

std::string SessionManager::createSession(const std::string& sessionName) {
    // TODO: Implement session creation
    return "";
}

bool SessionManager::joinSession(const std::string& sessionId, const ClientInfo& clientInfo) {
    // TODO: Implement session joining
    return false;
}

void SessionManager::leaveSession(const std::string& clientId) {
    // TODO: Implement session leaving
}

void SessionManager::destroySession(const std::string& sessionId) {
    // TODO: Implement session destruction
}

std::vector<SessionManager::SessionInfo> SessionManager::getActiveSessions() const {
    // TODO: Implement session enumeration
    return {};
}

std::optional<SessionManager::SessionInfo> SessionManager::getSession(const std::string& sessionId) const {
    // TODO: Implement session lookup
    return std::nullopt;
}

void SessionManager::routeMIDIMessage(const std::string& sessionId, 
                                    std::unique_ptr<JSONMIDI::MIDIMessage> message,
                                    const std::string& sourceClientId) {
    // TODO: Implement MIDI message routing
}

} // namespace TOAST
