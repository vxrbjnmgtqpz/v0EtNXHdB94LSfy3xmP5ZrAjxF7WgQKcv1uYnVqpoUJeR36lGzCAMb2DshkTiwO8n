/**
 * JMID TOAST v2 Transport Implementation
 * 
 * Pure UDP fire-and-forget transport using TOAST v2 protocol.
 * Preserves JMID's 11.77μs latency achievement while enabling
 * unified JAMNet ecosystem participation.
 */

#include "../include/JMIDTOASTv2Transport.h"
#include "../../JAM_Framework_v2/include/jam_toast.h"
#include <chrono>
#include <thread>
#include <random>
#include <nlohmann/json.hpp>

namespace JMID {

JMIDTOASTv2Transport::JMIDTOASTv2Transport()
    : startTime_(std::chrono::high_resolution_clock::now()) {
    
    // Initialize core components
    parser_ = std::make_unique<SIMDJMIDParser>();
    
    // Reserve space for performance history
    latencyHistory_.reserve(maxHistorySize_);
    throughputHistory_.reserve(maxHistorySize_);
}

JMIDTOASTv2Transport::~JMIDTOASTv2Transport() {
    shutdown();
}

bool JMIDTOASTv2Transport::initialize(const TransportConfig& config) {
    if (initialized_.load()) {
        return true;
    }
    
    config_ = config;
    
    try {
        // Initialize TOAST v2 protocol - PURE UDP MULTICAST
        toast_ = std::make_unique<jam::TOASTv2Protocol>();
        if (!toast_->initialize(config.multicastGroup, config.port, config.sessionId)) {
            handleInternalError("Failed to initialize TOAST v2 protocol", 1001);
            return false;
        }
        
        // Set up TOAST v2 frame callbacks
        toast_->set_midi_callback([this](const jam::TOASTFrame& frame) { 
            handleMIDIFrame(frame); 
        });
        
        toast_->set_error_callback([this](const std::string& error) { 
            handleErrorFrame(error); 
        });
        
        // Configure burst transmission
        jam::BurstConfig burstConfig;
        burstConfig.burst_size = config.burstCount;
        burstConfig.jitter_window_us = config.burstDelayMicros;
        burstConfig.enable_redundancy = config.enableBurstTransmission;
        burstConfig.max_retries = 0;  // Fire-and-forget
        toast_->set_burst_config(burstConfig);
        
        initialized_.store(true);
        sessionId_.store(config.sessionId);
        
        resetStats();
        
        return true;
        
    } catch (const std::exception& e) {
        handleInternalError("Initialization failed: " + std::string(e.what()), 1002);
        return false;
    }
}

void JMIDTOASTv2Transport::shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    stopProcessing();
    
    if (toast_) {
        toast_->shutdown();
        toast_.reset();
    }
    
    initialized_.store(false);
    running_.store(false);
}

bool JMIDTOASTv2Transport::isInitialized() const {
    return initialized_.load();
}

bool JMIDTOASTv2Transport::isRunning() const {
    return running_.load();
}

bool JMIDTOASTv2Transport::sendMessage(const std::string& compactJson, bool useBurst) {
    if (!running_.load()) {
        handleInternalError("Transport not running", 2001);
        return false;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Create TOAST v2 MIDI frame with compact JSON payload
        jam::TOASTFrame frame = createMIDIFrame(compactJson, useBurst);
        
        // Send with burst if requested (fire-and-forget UDP)
        bool success = toast_->send_frame(frame, useBurst);
        
        if (success) {
            updateSentStats(frame);
            
            // Record latency for performance monitoring
            auto endTime = std::chrono::high_resolution_clock::now();
            double latencyMicros = std::chrono::duration<double, std::micro>(endTime - startTime).count();
            updateLatencyStats(latencyMicros);
        }
        
        return success;
        
    } catch (const std::exception& e) {
        handleInternalError("Send failed: " + std::string(e.what()), 2002);
        return false;
    }
}

bool JMIDTOASTv2Transport::sendRawMessage(const std::string& jsonMessage, bool useBurst) {
    try {
        // For now, just treat raw message as compact JSON
        // TODO: Parse and convert to compact format
        return sendMessage(jsonMessage, useBurst);
        
    } catch (const std::exception& e) {
        handleInternalError("Raw message processing failed: " + std::string(e.what()), 2006);
        return false;
    }
}

jam::TOASTFrame JMIDTOASTv2Transport::createMIDIFrame(const std::string& compactJson, bool useBurst) {
    jam::TOASTFrame frame;
    
    // Configure frame header
    configureFrameHeader(frame, useBurst);
    frame.header.frame_type = static_cast<uint8_t>(jam::TOASTFrameType::MIDI);
    
    // Embed compact JSON in payload
    frame.payload.assign(compactJson.begin(), compactJson.end());
    frame.header.payload_size = frame.payload.size();
    
    // Calculate checksum for frame validation
    frame.calculate_checksum();
    
    return frame;
}

void JMIDTOASTv2Transport::configureFrameHeader(jam::TOASTFrame& frame, bool useBurst) {
    auto now = getCurrentMicroseconds();
    
    frame.header.magic = 0x54534F54;  // "TOST"
    frame.header.version = 2;
    frame.header.flags = 0;
    frame.header.sequence_number = sequenceCounter_.fetch_add(1);
    frame.header.timestamp_us = static_cast<uint32_t>(now & 0xFFFFFFFF);
    frame.header.session_id = sessionId_.load();
    
    // Configure burst fields for fire-and-forget reliability
    if (useBurst) {
        frame.header.burst_id = generateBurstId();
        frame.header.burst_index = 0;  // Will be set per burst packet
        frame.header.burst_total = config_.burstCount;
        frame.header.flags |= 0x01;  // Burst flag
    } else {
        frame.header.burst_id = 0;
        frame.header.burst_index = 0;
        frame.header.burst_total = 1;
    }
}

uint32_t JMIDTOASTv2Transport::generateBurstId() {
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_int_distribution<uint32_t> dist;
    return dist(gen);
}

void JMIDTOASTv2Transport::handleMIDIFrame(const jam::TOASTFrame& frame) {
    if (!running_.load() || !messageHandler_) {
        return;
    }
    
    try {
        // Extract compact JSON from payload
        std::string compactJson(frame.payload.begin(), frame.payload.end());
        
        // Parse with SIMD-optimized parser (0.095μs target)
        auto message = parseCompactMessage(compactJson);
        if (message) {
            updateReceivedStats(frame);
            
            // Call user handler
            messageHandler_(std::move(message));
        }
        
    } catch (const std::exception& e) {
        handleInternalError("MIDI frame handling failed: " + std::string(e.what()), 3001);
    }
}

std::unique_ptr<JMID::MIDIMessage> JMIDTOASTv2Transport::parseCompactMessage(const std::string& compactJson) {
    auto parseStart = std::chrono::high_resolution_clock::now();
    
    auto message = parser_->parseMessage(compactJson);
    
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseTimeMicros = std::chrono::duration<double, std::micro>(parseEnd - parseStart).count();
    
    // Update parse performance stats
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        if (stats_.parseTimeAvgMicros == 0.0) {
            stats_.parseTimeAvgMicros = parseTimeMicros;
        } else {
            stats_.parseTimeAvgMicros = (stats_.parseTimeAvgMicros * 0.95) + (parseTimeMicros * 0.05);
        }
    }
    
    return message;
}

bool JMIDTOASTv2Transport::startProcessing() {
    if (!initialized_.load()) {
        handleInternalError("Transport not initialized", 4001);
        return false;
    }
    
    if (running_.load()) {
        return true;
    }
    
    try {
        // Start TOAST v2 protocol processing
        if (!toast_->start_processing()) {
            handleInternalError("Failed to start TOAST v2 protocol", 4002);
            return false;
        }
        
        running_.store(true);
        
        return true;
        
    } catch (const std::exception& e) {
        handleInternalError("Start processing failed: " + std::string(e.what()), 4004);
        return false;
    }
}

void JMIDTOASTv2Transport::stopProcessing() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    if (toast_) {
        toast_->stop_processing();
    }
}

TransportStats JMIDTOASTv2Transport::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    // Update real-time stats
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsedSeconds = std::chrono::duration<double>(now - startTime_).count();
    
    TransportStats currentStats = stats_;
    
    if (elapsedSeconds > 0) {
        currentStats.messagesPerSecond = static_cast<uint64_t>(stats_.messagesReceived / elapsedSeconds);
        currentStats.bytesPerSecond = currentStats.messagesPerSecond * 50; // Estimate
    }
    
    currentStats.isConnected = running_.load();
    currentStats.activePeers = getActivePeers().size();
    
    return currentStats;
}

void JMIDTOASTv2Transport::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_ = TransportStats{};
    startTime_ = std::chrono::high_resolution_clock::now();
}

void JMIDTOASTv2Transport::updateSentStats(const jam::TOASTFrame& frame) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.messagesSent++;
    stats_.burstPacketsSent += (frame.header.flags & 0x01) ? frame.header.burst_total : 1;
}

void JMIDTOASTv2Transport::updateReceivedStats(const jam::TOASTFrame& frame) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.messagesReceived++;
}

void JMIDTOASTv2Transport::updateLatencyStats(double latencyMicros) {
    recordLatency(latencyMicros);
    
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    if (stats_.minLatencyMicros == 0.0 || latencyMicros < stats_.minLatencyMicros) {
        stats_.minLatencyMicros = latencyMicros;
    }
    
    if (latencyMicros > stats_.maxLatencyMicros) {
        stats_.maxLatencyMicros = latencyMicros;
    }
    
    // Update rolling average
    if (stats_.averageLatencyMicros == 0.0) {
        stats_.averageLatencyMicros = latencyMicros;
    } else {
        stats_.averageLatencyMicros = (stats_.averageLatencyMicros * 0.95) + (latencyMicros * 0.05);
    }
}

void JMIDTOASTv2Transport::recordLatency(double latencyMicros) {
    if (latencyHistory_.size() >= maxHistorySize_) {
        latencyHistory_.erase(latencyHistory_.begin());
    }
    latencyHistory_.push_back(latencyMicros);
}

std::string JMIDTOASTv2Transport::getTransportType() const {
    return "TOAST_V2_UDP";
}

std::string JMIDTOASTv2Transport::getTransportVersion() const {
    return "2.0.0";
}

std::vector<std::string> JMIDTOASTv2Transport::getActivePeers() const {
    std::lock_guard<std::mutex> lock(peersMutex_);
    std::vector<std::string> peers;
    for (const auto& peer : activePeers_) {
        peers.push_back(peer.first);
    }
    return peers;
}

uint64_t JMIDTOASTv2Transport::getCurrentMicroseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

void JMIDTOASTv2Transport::handleErrorFrame(const std::string& error) {
    handleInternalError("TOAST v2 error: " + error, 5001);
}

void JMIDTOASTv2Transport::handleInternalError(const std::string& error, int errorCode) {
    if (errorHandler_) {
        errorHandler_(error, errorCode);
    }
}

// Stub implementations for required interface methods
void JMIDTOASTv2Transport::setMessageHandler(MessageHandler handler) {
    messageHandler_ = std::move(handler);
}

void JMIDTOASTv2Transport::setErrorHandler(ErrorHandler handler) {
    errorHandler_ = std::move(handler);
}

void JMIDTOASTv2Transport::setPeerCallback(PeerCallback callback) {
    peerCallback_ = std::move(callback);
}

void JMIDTOASTv2Transport::setTransportConfig(const TransportConfig& config) {
    config_ = config;
}

TransportConfig JMIDTOASTv2Transport::getTransportConfig() const {
    return config_;
}

bool JMIDTOASTv2Transport::joinSession(uint32_t sessionId) {
    sessionId_.store(sessionId);
    return true;
}

void JMIDTOASTv2Transport::leaveSession() {
    sessionId_.store(0);
}

uint32_t JMIDTOASTv2Transport::getCurrentSessionId() const {
    return sessionId_.load();
}

void JMIDTOASTv2Transport::setBurstConfig(int burstCount, int delayMicros) {
    config_.burstCount = burstCount;
    config_.burstDelayMicros = delayMicros;
}

void JMIDTOASTv2Transport::setLatencyTarget(double targetMicros) {
    // TODO: Implement latency target configuration
}

void JMIDTOASTv2Transport::enablePrecisionTiming(bool enabled) {
    config_.enablePrecisionTiming = enabled;
}

bool JMIDTOASTv2Transport::supportsFeature(const std::string& feature) const {
    // Basic feature support
    return (feature == "burst_transmission" || 
            feature == "udp_multicast" || 
            feature == "compact_format" ||
            feature == "simd_parsing");
}

} // namespace JMID
 