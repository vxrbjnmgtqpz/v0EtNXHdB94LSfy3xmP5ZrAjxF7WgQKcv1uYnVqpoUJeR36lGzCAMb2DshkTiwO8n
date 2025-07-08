#include "UDPBurstTransport.h"
#include "JMIDParser.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <nlohmann/json.hpp>

#ifdef _WIN32
    #pragma comment(lib, "ws2_32.lib")
#endif

namespace JMID {

//=============================================================================
// UDPBurstTransport Implementation
//=============================================================================

UDPBurstTransport::UDPBurstTransport(const BurstConfig& config) 
    : config_(config) {
    
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
    
    std::cout << "🚀 JMID UDP Burst Transport initialized" << std::endl;
    std::cout << "   📡 Multicast Group: " << config_.multicastGroup << ":" << config_.port << std::endl;
    std::cout << "   💥 Burst Count: " << config_.burstCount << " packets" << std::endl;
    std::cout << "   ⏱️  Burst Delay: " << config_.burstDelayMicros << "μs" << std::endl;
}

UDPBurstTransport::~UDPBurstTransport() {
    stop();
    cleanupSocket();
    
#ifdef _WIN32
    WSACleanup();
#endif
}

bool UDPBurstTransport::startServer(uint16_t port) {
    if (running_.load()) {
        std::cerr << "❌ Transport already running" << std::endl;
        return false;
    }
    
    if (port != 0) {
        config_.port = port;
    }
    
    isServer_ = true;
    
    if (!initializeSocket()) {
        std::cerr << "❌ Failed to initialize UDP socket" << std::endl;
        return false;
    }
    
    if (!joinMulticastGroup()) {
        std::cerr << "❌ Failed to join multicast group" << std::endl;
        cleanupSocket();
        return false;
    }
    
    running_.store(true);
    receiveThread_ = std::make_unique<std::thread>(&UDPBurstTransport::receiveLoop, this);
    
    std::cout << "✅ JMID UDP Server started on " << config_.multicastGroup << ":" << config_.port << std::endl;
    return true;
}

bool UDPBurstTransport::startClient() {
    if (running_.load()) {
        std::cerr << "❌ Transport already running" << std::endl;
        return false;
    }
    
    isServer_ = false;
    
    if (!initializeSocket()) {
        std::cerr << "❌ Failed to initialize UDP socket" << std::endl;
        return false;
    }
    
    if (!joinMulticastGroup()) {
        std::cerr << "❌ Failed to join multicast group" << std::endl;
        cleanupSocket();
        return false;
    }
    
    running_.store(true);
    receiveThread_ = std::make_unique<std::thread>(&UDPBurstTransport::receiveLoop, this);
    
    std::cout << "✅ JMID UDP Client connected to " << config_.multicastGroup << ":" << config_.port << std::endl;
    return true;
}

void UDPBurstTransport::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    if (receiveThread_ && receiveThread_->joinable()) {
        receiveThread_->join();
    }
    
    cleanupSocket();
    std::cout << "🛑 JMID UDP Transport stopped" << std::endl;
}

bool UDPBurstTransport::sendMIDIMessage(std::unique_ptr<JMID::MIDIMessage> message) {
    if (!message) {
        return false;
    }
    
    // Convert MIDI message to compact JSON
    std::string jsonMessage = message->toJSON();
    return sendRawMessage(jsonMessage);
}

bool UDPBurstTransport::sendRawMessage(const std::string& jsonMessage) {
    if (!running_.load()) {
        std::cerr << "❌ Transport not running" << std::endl;
        return false;
    }
    
    // Add sequence number for deduplication
    std::string sequencedMessage = addSequenceNumber(jsonMessage);
    
    // Send burst packets
    bool success = true;
    for (int i = 0; i < config_.burstCount; ++i) {
        if (!sendUDPPacket(sequencedMessage)) {
            success = false;
            std::cerr << "⚠️ Failed to send burst packet " << (i + 1) << "/" << config_.burstCount << std::endl;
        }
        
        // Add delay between burst packets (except last one)
        if (i < config_.burstCount - 1) {
            microDelay(config_.burstDelayMicros);
        }
    }
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.messagesSent++;
        stats_.burstPacketsSent += config_.burstCount;
    }
    
    if (success) {
        std::cout << "📤 Sent MIDI burst: " << jsonMessage.substr(0, 50) << "..." << std::endl;
    }
    
    return success;
}

bool UDPBurstTransport::initializeSocket() {
    socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ < 0) {
        std::cerr << "❌ Failed to create socket" << std::endl;
        return false;
    }
    
    // Set socket options for multicast
    int opt = 1;
    if (setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        std::cerr << "❌ Failed to set SO_REUSEADDR" << std::endl;
        return false;
    }
    
#ifdef SO_REUSEPORT
    if (setsockopt(socket_, SOL_SOCKET, SO_REUSEPORT, reinterpret_cast<const char*>(&opt), sizeof(opt)) < 0) {
        std::cerr << "⚠️ Failed to set SO_REUSEPORT (non-critical)" << std::endl;
    }
#endif
    
    // Setup server address
    std::memset(&serverAddr_, 0, sizeof(serverAddr_));
    serverAddr_.sin_family = AF_INET;
    serverAddr_.sin_addr.s_addr = INADDR_ANY;
    serverAddr_.sin_port = htons(config_.port);
    
    // Setup multicast address
    std::memset(&multicastAddr_, 0, sizeof(multicastAddr_));
    multicastAddr_.sin_family = AF_INET;
    multicastAddr_.sin_port = htons(config_.port);
    if (inet_pton(AF_INET, config_.multicastGroup.c_str(), &multicastAddr_.sin_addr) <= 0) {
        std::cerr << "❌ Invalid multicast address: " << config_.multicastGroup << std::endl;
        return false;
    }
    
    // Bind socket
    if (bind(socket_, reinterpret_cast<struct sockaddr*>(&serverAddr_), sizeof(serverAddr_)) < 0) {
        std::cerr << "❌ Failed to bind socket to port " << config_.port << std::endl;
        return false;
    }
    
    return true;
}

void UDPBurstTransport::cleanupSocket() {
    if (socket_ >= 0) {
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
        socket_ = -1;
    }
}

bool UDPBurstTransport::joinMulticastGroup() {
    struct ip_mreq mreq;
    mreq.imr_multiaddr = multicastAddr_.sin_addr;
    mreq.imr_interface.s_addr = INADDR_ANY;
    
    if (setsockopt(socket_, IPPROTO_IP, IP_ADD_MEMBERSHIP, reinterpret_cast<const char*>(&mreq), sizeof(mreq)) < 0) {
        std::cerr << "❌ Failed to join multicast group" << std::endl;
        return false;
    }
    
    std::cout << "📡 Joined multicast group: " << config_.multicastGroup << std::endl;
    return true;
}

void UDPBurstTransport::receiveLoop() {
    char buffer[4096];
    struct sockaddr_in senderAddr;
    socklen_t senderLen = sizeof(senderAddr);
    
    std::cout << "👂 JMID UDP receive loop started" << std::endl;
    
    while (running_.load()) {
        ssize_t bytesReceived = recvfrom(socket_, buffer, sizeof(buffer) - 1, 0,
                                       reinterpret_cast<struct sockaddr*>(&senderAddr), &senderLen);
        
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            std::string jsonMessage(buffer, bytesReceived);
            
            // Process burst message (includes deduplication)
            processBurstMessage(jsonMessage);
            
            // Update statistics
            {
                std::lock_guard<std::mutex> lock(statsMutex_);
                stats_.messagesReceived++;
            }
        } else if (bytesReceived < 0 && running_.load()) {
            std::cerr << "⚠️ UDP receive error" << std::endl;
        }
    }
    
    std::cout << "🛑 JMID UDP receive loop stopped" << std::endl;
}

bool UDPBurstTransport::sendUDPPacket(const std::string& data) {
    if (data.size() > static_cast<size_t>(config_.maxPacketSize)) {
        std::cerr << "❌ Message too large: " << data.size() << " bytes" << std::endl;
        return false;
    }
    
    ssize_t bytesSent = sendto(socket_, data.c_str(), data.size(), 0,
                              reinterpret_cast<const struct sockaddr*>(&multicastAddr_),
                              sizeof(multicastAddr_));
    
    return bytesSent == static_cast<ssize_t>(data.size());
}

void UDPBurstTransport::processBurstMessage(const std::string& jsonMessage) {
    // Extract sequence number for deduplication
    uint64_t sequence = extractSequenceNumber(jsonMessage);
    
    // Check if we've seen this sequence before
    {
        std::lock_guard<std::mutex> lock(sequenceMutex_);
        if (seenSequences_.find(sequence) != seenSequences_.end()) {
            // Duplicate message - discard
            std::lock_guard<std::mutex> statsLock(statsMutex_);
            stats_.duplicatesReceived++;
            return;
        }
        
        // New message - remember this sequence
        seenSequences_.insert(sequence);
        
        // Cleanup old sequences to prevent memory growth
        if (seenSequences_.size() > SEQUENCE_CACHE_SIZE) {
            // Simple cleanup: remove some old entries
            auto it = seenSequences_.begin();
            std::advance(it, SEQUENCE_CACHE_SIZE / 4);
            seenSequences_.erase(seenSequences_.begin(), it);
        }
    }
    
    // Parse and handle the message
    if (messageHandler_) {
        try {
            nlohmann::json doc = nlohmann::json::parse(jsonMessage);
            // TODO: Convert JSON to MIDIMessage object
            // For now, just log the received message
            std::cout << "📥 Received unique MIDI: " << jsonMessage.substr(0, 50) << "..." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "⚠️ Failed to parse JSON: " << e.what() << std::endl;
        }
    }
}

std::string UDPBurstTransport::addSequenceNumber(const std::string& jsonMessage) {
    try {
        nlohmann::json doc = nlohmann::json::parse(jsonMessage);
        
        // Add sequence number and timestamp
        doc["seq"] = sequenceCounter_.fetch_add(1);
        doc["ts"] = getCurrentMicroseconds();
        
        return doc.dump(-1); // Compact JSON
    } catch (const std::exception& e) {
        std::cerr << "⚠️ Failed to add sequence number: " << e.what() << std::endl;
        return jsonMessage; // Return original if parsing fails
    }
}

uint64_t UDPBurstTransport::extractSequenceNumber(const std::string& jsonMessage) {
    try {
        nlohmann::json doc = nlohmann::json::parse(jsonMessage);
        if (doc.contains("seq")) {
            return doc["seq"].get<uint64_t>();
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠️ Failed to extract sequence number: " << e.what() << std::endl;
    }
    return 0; // Default if extraction fails
}

bool UDPBurstTransport::isValidSequence(uint64_t sequence) {
    std::lock_guard<std::mutex> lock(sequenceMutex_);
    return seenSequences_.find(sequence) == seenSequences_.end();
}

void UDPBurstTransport::microDelay(int microseconds) {
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
}

uint64_t UDPBurstTransport::getCurrentMicroseconds() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

void UDPBurstTransport::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_ = Stats{};
}

//=============================================================================
// BurstDeduplicator Implementation
//=============================================================================

bool BurstDeduplicator::processMessage(const std::string& jsonMessage) {
    uint64_t sequence = extractSequenceNumber(jsonMessage);
    uint64_t timestamp = extractTimestamp(jsonMessage);
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update statistics
    stats_.totalMessages++;
    
    // Check if we've seen this sequence before
    if (seenSequences_.find(sequence) != seenSequences_.end()) {
        stats_.duplicatesFiltered++;
        stats_.deduplicationRate = static_cast<double>(stats_.duplicatesFiltered) / stats_.totalMessages;
        return false; // Duplicate
    }
    
    // New message - add to timeline
    seenSequences_.insert(sequence);
    timeline_.push_back({sequence, timestamp, jsonMessage});
    
    // Keep timeline sorted by timestamp
    std::sort(timeline_.begin(), timeline_.end(),
              [](const MessageInfo& a, const MessageInfo& b) {
                  return a.timestamp < b.timestamp;
              });
    
    stats_.uniqueMessages++;
    stats_.deduplicationRate = static_cast<double>(stats_.duplicatesFiltered) / stats_.totalMessages;
    
    return true; // Process this message
}

std::vector<std::string> BurstDeduplicator::getTimelineMessages(uint64_t fromTimestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> messages;
    for (const auto& info : timeline_) {
        if (info.timestamp >= fromTimestamp) {
            messages.push_back(info.message);
        }
    }
    
    return messages;
}

void BurstDeduplicator::cleanupOldSequences(uint64_t olderThanMicros) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint64_t cutoffTime = getCurrentMicroseconds() - olderThanMicros;
    
    // Remove old timeline entries
    timeline_.erase(
        std::remove_if(timeline_.begin(), timeline_.end(),
                      [cutoffTime](const MessageInfo& info) {
                          return info.timestamp < cutoffTime;
                      }),
        timeline_.end());
    
    // Rebuild seen sequences from remaining timeline
    seenSequences_.clear();
    for (const auto& info : timeline_) {
        seenSequences_.insert(info.sequence);
    }
}

uint64_t BurstDeduplicator::extractSequenceNumber(const std::string& jsonMessage) {
    try {
        nlohmann::json doc = nlohmann::json::parse(jsonMessage);
        if (doc.contains("seq")) {
            return doc["seq"].get<uint64_t>();
        }
    } catch (const std::exception& e) {
        // Ignore parsing errors for deduplicator
    }
    return 0;
}

uint64_t BurstDeduplicator::extractTimestamp(const std::string& jsonMessage) {
    try {
        nlohmann::json doc = nlohmann::json::parse(jsonMessage);
        if (doc.contains("ts")) {
            return doc["ts"].get<uint64_t>();
        }
    } catch (const std::exception& e) {
        // Ignore parsing errors for deduplicator
    }
    return getCurrentMicroseconds(); // Use current time as fallback
}

uint64_t getCurrentMicroseconds() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

} // namespace JMID
 