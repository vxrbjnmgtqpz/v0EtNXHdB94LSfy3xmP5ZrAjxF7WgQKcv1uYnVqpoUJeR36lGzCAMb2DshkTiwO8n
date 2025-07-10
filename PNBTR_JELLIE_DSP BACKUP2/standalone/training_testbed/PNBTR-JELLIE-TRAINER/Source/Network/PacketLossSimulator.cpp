/*
  ==============================================================================

    PacketLossSimulator.cpp
    Network Simulation with TOAST v2 Integration (Stub + Working Implementation)

    This implementation provides a working network simulator that can be extended
    with full TOAST protocol integration once JAM_Framework_v2 is properly linked.

  ==============================================================================
*/

#include "PacketLossSimulator.h"
#include <chrono>
#include <thread>
#include <random>
#include <iostream>
#include <algorithm>
#include <cstring>

// Utility function to replace std::clamp (C++17)
template<typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

//==============================================================================
PacketLossSimulator::PacketLossSimulator() 
    : rng_(std::random_device{}())
    , lossDist_(0.0f, 100.0f)
    , jitterDist_(0.0f, 50.0f)
    , sessionId_(0)
    , port_(9988)
    , multicastAddress_("239.255.77.88")
{
    lossMap_.reserve(2048); // Pre-allocate for typical block sizes
    transmitBuffer_.reserve(2048);
    receiveBuffer_.reserve(2048);
    
    std::cout << "[PacketLossSimulator] Initialized - Ready for TOAST v2 integration" << std::endl;
}

PacketLossSimulator::~PacketLossSimulator()
{
    shutdown();
}

//==============================================================================
bool PacketLossSimulator::initialize(const std::string& multicast_addr, uint16_t port)
{
    if (networkActive_.load()) {
        return true; // Already initialized
    }
    
    multicastAddress_ = multicast_addr;
    port_ = port;
    
    // Generate unique session ID
    sessionId_ = static_cast<uint32_t>(std::chrono::high_resolution_clock::now()
                .time_since_epoch().count() & 0xFFFFFFFF);
    
    std::cout << "[PacketLossSimulator] Initializing network simulation..." << std::endl;
    std::cout << "  Target multicast: " << multicast_addr << ":" << port << std::endl;
    std::cout << "  Session ID: " << sessionId_ << std::endl;
    
    // Simulate already established TOAST v2 connection
    // In a real deployment, this would be a live UDP multicast connection
    // For training purposes, we simulate the connection being already active
    
    std::cout << "[PacketLossSimulator] Simulating established TOAST v2 connection..." << std::endl;
    std::cout << "  Simulated peers: 3 active nodes on network" << std::endl;
    std::cout << "  Connection quality: Excellent (simulated)" << std::endl;
    std::cout << "  Burst transmission: 3-packet redundancy active" << std::endl;
    
    // Simulate connection establishment delay
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    toast_initialized_ = true;
    
    networkActive_.store(true);
    resetStats();
    
    std::cout << "[PacketLossSimulator] Network simulation initialized (TOAST integration pending)" << std::endl;
    return true;
}

void PacketLossSimulator::shutdown()
{
    if (!networkActive_.load()) {
        return;
    }
    
    std::cout << "[PacketLossSimulator] Shutting down network simulation..." << std::endl;
    
    networkActive_.store(false);
    
    // Simulate TOAST connection shutdown
    std::cout << "[PacketLossSimulator] Simulating TOAST connection shutdown..." << std::endl;
    std::cout << "  Disconnecting from 3 simulated peers" << std::endl;
    std::cout << "  Cleaning up UDP multicast simulation" << std::endl;
    
    toast_initialized_ = false;
    
    // Display final network statistics
    auto finalStats = getStats();
    std::cout << "[PacketLossSimulator] Final Network Statistics:" << std::endl;
    std::cout << "  Packets sent: " << finalStats.packets_sent << std::endl;
    std::cout << "  Packets received: " << finalStats.packets_received << std::endl;
    std::cout << "  Actual loss rate: " << finalStats.actual_loss_rate << "%" << std::endl;
    std::cout << "  Average jitter: " << finalStats.average_jitter_ms << "ms" << std::endl;
}

//==============================================================================
void PacketLossSimulator::prepare(double sampleRate, int samplesPerBlock)
{
    currentSampleRate_ = sampleRate;
    currentBlockSize_ = samplesPerBlock;
    
    // Resize buffers for this block size (stereo)
    transmitBuffer_.resize(samplesPerBlock * 2);
    receiveBuffer_.resize(samplesPerBlock * 2);
    lossMap_.resize(samplesPerBlock);
    
    std::cout << "[PacketLossSimulator] Prepared for " << samplesPerBlock 
              << " samples at " << sampleRate << "Hz" << std::endl;
}

void PacketLossSimulator::generateLossMap(float lossPercentage, float jitterMs)
{
    setPacketLossPercentage(lossPercentage);
    setJitterAmount(jitterMs);
    updateLossMap(currentBlockSize_);
}

void PacketLossSimulator::setPacketLossPercentage(float percentage)
{
    packetLossPercentage_.store(clamp(percentage, 0.0f, 100.0f));
}

void PacketLossSimulator::setJitterAmount(float jitterMs)
{
    jitterAmountMs_.store(clamp(jitterMs, 0.0f, 100.0f));
}

//==============================================================================
bool PacketLossSimulator::transmitAudioChunk(const std::vector<float>& audioData, 
                                            uint64_t timestamp_us,
                                            uint32_t sample_rate,
                                            uint8_t channels)
{
    if (!networkActive_.load()) {
        return false;
    }
    
    // Apply loss simulation at the packet level
    if (shouldDropPacket()) {
        stats_.packets_lost++;
        // Simulate packet loss
        std::fill(lossMap_.begin(), lossMap_.end(), true); // Mark as lost
        return false; // Simulate packet drop
    }
    
    // Calculate jitter delay
    uint32_t jitterDelay = calculateJitterDelay();
    uint64_t adjustedTimestamp = timestamp_us + jitterDelay * 1000; // Convert ms to μs
    
    // Simulate TOAST v2 transmission over established connection
    // In real deployment: toast_->send_audio(audioData, adjustedTimestamp, sample_rate, channels);
    // For training: simulate successful transmission with realistic timing
    bool success = true; // Simulate established connection always succeeds
    
    if (success) {
        stats_.packets_sent++;
        std::fill(lossMap_.begin(), lossMap_.end(), false); // Mark as transmitted
        
        // Copy to receive buffer to simulate loopback
        receiveBuffer_ = audioData;
        hasNewAudio_.store(true);
    } else {
        stats_.packets_lost++;
        std::fill(lossMap_.begin(), lossMap_.end(), true); // Mark as lost
    }
    
    return success;
}

bool PacketLossSimulator::receiveAudioChunk(std::vector<float>& audioData)
{
    if (!hasNewAudio_.load()) {
        return false;
    }
    
    // Copy received buffer
    audioData = receiveBuffer_;
    hasNewAudio_.store(false);
    
    stats_.packets_received++;
    return true;
}

//==============================================================================
PacketLossSimulator::NetworkStats PacketLossSimulator::getStats() const
{
    NetworkStats currentStats = stats_;
    
    // Calculate real-time loss rate
    if (currentStats.packets_sent > 0) {
        currentStats.actual_loss_rate = 
            (static_cast<double>(currentStats.packets_lost) / currentStats.packets_sent) * 100.0;
    }
    
    // TODO: Get TOAST protocol statistics when available
    // if (toast_) {
    //     auto toastStats = toast_->get_stats();
    //     currentStats.burst_packets_sent = toastStats.burst_packets_sent;
    //     currentStats.active_sessions = toastStats.active_peers;
    // }
    
    currentStats.average_jitter_ms = jitterAmountMs_.load();
    
    return currentStats;
}

void PacketLossSimulator::resetStats()
{
    stats_ = NetworkStats{};
}

//==============================================================================
// Private implementation methods

bool PacketLossSimulator::shouldDropPacket()
{
    float currentLossRate = packetLossPercentage_.load();
    return lossDist_(rng_) < currentLossRate;
}

uint32_t PacketLossSimulator::calculateJitterDelay()
{
    float maxJitter = jitterAmountMs_.load();
    return static_cast<uint32_t>(jitterDist_(rng_) * maxJitter / 50.0f); // Scale to range
}

void PacketLossSimulator::updateLossMap(int blockSize)
{
    lossMap_.resize(blockSize);
    
    // Generate loss pattern for this block
    for (int i = 0; i < blockSize; ++i) {
        lossMap_[i] = shouldDropPacket();
    }
}

uint64_t PacketLossSimulator::getCurrentTimestamp()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

//==============================================================================
// TOAST Integration Stubs (to be implemented when JAM_Framework_v2 is linked)

void PacketLossSimulator::handleAudioFrame(const SimulatedTOASTFrame& frame)
{
    std::cout << "[PacketLossSimulator] Simulated TOAST audio frame received" << std::endl;
}

void PacketLossSimulator::handleRealAudioFrame(const SimulatedTOASTFrame& frame)
{
    // Simulate extracting audio data from TOAST frame
    // This simulates what would happen with real UDP multicast reception
    try {
        // Extract audio samples from simulated frame payload
        const auto& payload = frame.payload;
        if (payload.size() >= sizeof(float)) {
            receiveBuffer_.clear();
            receiveBuffer_.resize(payload.size() / sizeof(float));
            std::memcpy(receiveBuffer_.data(), payload.data(), payload.size());
            hasNewAudio_.store(true);
            stats_.packets_received++;
            
            std::cout << "[PacketLossSimulator] Simulated TOAST audio frame processed: " 
                      << receiveBuffer_.size() << " samples" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[PacketLossSimulator] Error processing simulated frame: " << e.what() << std::endl;
    }
}

void PacketLossSimulator::handleNetworkError(const std::string& error)
{
    std::cerr << "[PacketLossSimulator] Network error: " << error << std::endl;
}

SimulatedBurstConfig PacketLossSimulator::createBurstConfig() const
{
    SimulatedBurstConfig config;
    config.burst_size = 3; // Triple transmission for reliability
    config.jitter_window_us = static_cast<uint16_t>(jitterAmountMs_.load() * 1000);
    config.enable_redundancy = true;
    config.max_retries = 0; // Fire-and-forget for real-time audio
    return config;
}
