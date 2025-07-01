#pragma once

/*
 * ClockDriftArbiter: Network Clock Synchronization for TOAST
 * 
 * Provides distributed timing synchronization with sub-10ms accuracy
 * for real-time MIDI streaming over network connections.
 */

#include <chrono>
#include <atomic>
#include <memory>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace TOAST {

using HighResClock = std::chrono::high_resolution_clock;
using TimePoint = HighResClock::time_point;
using Duration = std::chrono::nanoseconds;
using MicrosecondClock = std::chrono::duration<uint64_t, std::micro>;

/**
 * Network timing measurement sample
 */
struct TimingSample {
    uint64_t localTimestamp;     // Local time when message sent (microseconds)
    uint64_t remoteTimestamp;    // Remote time when message received
    uint64_t roundTripTime;      // Full round-trip latency
    uint64_t networkLatency;     // One-way estimated latency
    double clockOffset;          // Remote clock offset from local
    double quality;              // Sample quality metric (0.0-1.0)
    
    TimingSample() : localTimestamp(0), remoteTimestamp(0), roundTripTime(0), 
                    networkLatency(0), clockOffset(0.0), quality(0.0) {}
};

/**
 * Clock synchronization role in the network
 */
enum class ClockRole {
    UNINITIALIZED,
    MASTER,           // Timing master for the session
    SLAVE,            // Synchronized to master
    CANDIDATE         // Eligible to become master
};

/**
 * Network connection state for timing
 */
struct NetworkPeer {
    std::string peerId;
    std::string ipAddress;
    uint16_t port;
    ClockRole role;
    
    // Timing statistics
    std::vector<TimingSample> samples;
    double averageLatency;
    double latencyVariance;
    double clockDrift;
    uint64_t lastSyncTime;
    
    // Connection quality
    uint32_t packetsLost;
    uint32_t packetsSent;
    double connectionQuality;
    
    NetworkPeer() : port(0), role(ClockRole::UNINITIALIZED), 
                   averageLatency(0.0), latencyVariance(0.0), 
                   clockDrift(0.0), lastSyncTime(0),
                   packetsLost(0), packetsSent(0), connectionQuality(0.0) {}
};

/**
 * Master clock election criteria
 */
struct MasterCandidate {
    std::string peerId;
    double networkStability;     // Lower latency variance = higher stability
    double clockPrecision;       // Platform clock resolution
    uint32_t sessionPriority;    // User-defined priority
    bool isManualOverride;       // Manually designated master
    
    MasterCandidate() : networkStability(0.0), clockPrecision(0.0), 
                       sessionPriority(0), isManualOverride(false) {}
};

/**
 * Clock synchronization state
 */
struct SyncState {
    ClockRole currentRole;
    std::string masterId;
    uint64_t masterEpoch;        // Master clock reference time
    double driftCompensation;    // Current drift adjustment
    double syncQuality;          // Overall synchronization quality
    uint64_t lastMasterSync;     // Last successful master sync
    
    SyncState() : currentRole(ClockRole::UNINITIALIZED), masterEpoch(0),
                 driftCompensation(0.0), syncQuality(0.0), lastMasterSync(0) {}
};

/**
 * ClockDriftArbiter: Core network timing synchronization
 */
class ClockDriftArbiter {
public:
    ClockDriftArbiter();
    ~ClockDriftArbiter();
    
    // Initialization and lifecycle
    bool initialize(const std::string& peerId, bool allowMasterRole = true);
    void shutdown();
    bool isRunning() const { return running_.load(); }
    
    // Master clock election
    void startMasterElection();
    void nominateForMaster(uint32_t priority = 100);
    void forceMasterRole();  // Manual override
    ClockRole getCurrentRole() const { return syncState_.currentRole; }
    std::string getMasterId() const { return syncState_.masterId; }
    
    // Network synchronization
    void addPeer(const std::string& peerId, const std::string& ipAddress, uint16_t port);
    void removePeer(const std::string& peerId);
    void synchronizeWithNetwork();
    
    // Timestamp compensation
    uint64_t getCurrentMasterTime() const;
    uint64_t compensateTimestamp(uint64_t localTimestamp) const;
    uint64_t getLocalTimestamp() const;
    
    // Network timing measurement
    void sendTimingPing(const std::string& peerId);
    void receivePingResponse(const std::string& fromPeer, uint64_t originalTimestamp, 
                           uint64_t remoteTimestamp, uint64_t returnTimestamp);
    
    // Quality and statistics
    double getSyncQuality() const { return syncState_.syncQuality; }
    double getNetworkLatency(const std::string& peerId) const;
    double getClockDrift(const std::string& peerId) const;
    std::vector<std::string> getConnectedPeers() const;
    
    // Network failure handling
    void handleConnectionLoss(const std::string& peerId);
    void handleMasterFailure();
    void recoverFromNetworkJitter();
    
    // Adaptive buffer management
    uint32_t getRecommendedBufferSize() const;
    uint32_t getMinimumBufferSize() const;
    void updateBufferRecommendations();
    
    // Configuration
    void setMaxClockDrift(double maxDriftMs) { maxAllowedDrift_ = maxDriftMs; }
    void setSyncInterval(uint32_t intervalMs) { syncIntervalMs_ = intervalMs; }
    void setElectionTimeout(uint32_t timeoutMs) { electionTimeoutMs_ = timeoutMs; }
    
    // Callbacks for network events
    std::function<void(const std::string&, const std::vector<uint8_t>&)> onSendMessage;
    std::function<void(ClockRole, const std::string&)> onRoleChanged;
    std::function<void(const std::string&)> onMasterElected;
    std::function<void(double)> onSyncQualityChanged;

private:
    // Core state
    std::atomic<bool> running_;
    std::string localPeerId_;
    bool allowMasterRole_;
    
    // Synchronization state
    SyncState syncState_;
    mutable std::mutex syncMutex_;
    
    // Network peers
    std::unordered_map<std::string, std::unique_ptr<NetworkPeer>> peers_;
    mutable std::mutex peersMutex_;
    
    // Timing and synchronization
    std::thread syncThread_;
    std::condition_variable syncCondition_;
    uint32_t syncIntervalMs_;
    uint32_t electionTimeoutMs_;
    double maxAllowedDrift_;
    
    // Master election
    std::vector<MasterCandidate> masterCandidates_;
    std::mutex electionMutex_;
    
    // Internal methods
    void synchronizationLoop();
    void performMasterElection();
    void updatePeerStatistics(const std::string& peerId, const TimingSample& sample);
    void calculateDriftCompensation();
    void validateSyncQuality();
    
    // Master role methods
    void becomeMaster();
    void broadcastMasterAnnouncement();
    void handleSlaveSync(const std::string& peerId, uint64_t slaveTimestamp);
    
    // Slave role methods
    void becomeSlave(const std::string& masterId);
    void requestMasterSync();
    void processMasterSync(uint64_t masterTimestamp, uint64_t localReceiveTime);
    
    // Timing utilities
    uint64_t getCurrentTimeMicros() const;
    double calculateClockOffset(const TimingSample& sample) const;
    double calculateSampleQuality(const TimingSample& sample) const;
    void pruneOldSamples(NetworkPeer& peer);
    
    // Network quality assessment
    void assessNetworkQuality();
    bool isPeerSuitable(const NetworkPeer& peer) const;
    double calculateNetworkStability(const NetworkPeer& peer) const;
};

} // namespace TOAST
