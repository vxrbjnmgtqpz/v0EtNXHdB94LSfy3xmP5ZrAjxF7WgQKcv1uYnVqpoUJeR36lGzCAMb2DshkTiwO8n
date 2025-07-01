#include "ClockDriftArbiter.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

namespace TOAST {

ClockDriftArbiter::ClockDriftArbiter()
    : running_(false)
    , allowMasterRole_(true)
    , syncIntervalMs_(100)  // 100ms sync interval
    , electionTimeoutMs_(5000)  // 5 second election timeout
    , maxAllowedDrift_(10.0)  // 10ms max drift
{
    syncState_.currentRole = ClockRole::UNINITIALIZED;
}

ClockDriftArbiter::~ClockDriftArbiter() {
    shutdown();
}

bool ClockDriftArbiter::initialize(const std::string& peerId, bool allowMasterRole) {
    if (running_.load()) {
        return false;
    }
    
    localPeerId_ = peerId;
    allowMasterRole_ = allowMasterRole;
    
    syncState_.currentRole = ClockRole::CANDIDATE;
    syncState_.masterEpoch = getCurrentTimeMicros();
    syncState_.driftCompensation = 0.0;
    syncState_.syncQuality = 0.0;
    syncState_.lastMasterSync = 0;
    
    running_.store(true);
    
    // Start synchronization thread
    syncThread_ = std::thread(&ClockDriftArbiter::synchronizationLoop, this);
    
    std::cout << "🕒 ClockDriftArbiter initialized for peer: " << peerId << std::endl;
    return true;
}

void ClockDriftArbiter::shutdown() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    syncCondition_.notify_all();
    
    if (syncThread_.joinable()) {
        syncThread_.join();
    }
    
    std::lock_guard<std::mutex> lock(peersMutex_);
    peers_.clear();
    
    std::cout << "🕒 ClockDriftArbiter shutdown complete" << std::endl;
}

void ClockDriftArbiter::startMasterElection() {
    if (!allowMasterRole_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(electionMutex_);
    
    // Add self as candidate
    MasterCandidate selfCandidate;
    selfCandidate.peerId = localPeerId_;
    selfCandidate.clockPrecision = 1.0;  // Assume 1 microsecond precision
    selfCandidate.sessionPriority = 100;  // Default priority
    selfCandidate.isManualOverride = false;
    
    // Calculate network stability (lower variance = better)
    double totalVariance = 0.0;
    int peerCount = 0;
    {
        std::lock_guard<std::mutex> peerLock(peersMutex_);
        for (const auto& [peerId, peer] : peers_) {
            totalVariance += peer->latencyVariance;
            peerCount++;
        }
    }
    
    selfCandidate.networkStability = peerCount > 0 ? 1.0 / (1.0 + totalVariance / peerCount) : 1.0;
    
    masterCandidates_.clear();
    masterCandidates_.push_back(selfCandidate);
    
    // Trigger election after short delay to collect other candidates
    std::cout << "🗳️ Starting master election..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    performMasterElection();
}

void ClockDriftArbiter::performMasterElection() {
    std::lock_guard<std::mutex> lock(electionMutex_);
    
    if (masterCandidates_.empty()) {
        return;
    }
    
    // Sort candidates by suitability (manual override > stability > precision > priority)
    std::sort(masterCandidates_.begin(), masterCandidates_.end(), 
        [](const MasterCandidate& a, const MasterCandidate& b) {
            if (a.isManualOverride != b.isManualOverride) {
                return a.isManualOverride > b.isManualOverride;
            }
            if (std::abs(a.networkStability - b.networkStability) > 0.1) {
                return a.networkStability > b.networkStability;
            }
            if (std::abs(a.clockPrecision - b.clockPrecision) > 0.1) {
                return a.clockPrecision > b.clockPrecision;
            }
            return a.sessionPriority > b.sessionPriority;
        });
    
    const auto& winner = masterCandidates_[0];
    
    if (winner.peerId == localPeerId_) {
        becomeMaster();
    } else {
        becomeSlave(winner.peerId);
    }
    
    if (onMasterElected) {
        onMasterElected(winner.peerId);
    }
}

void ClockDriftArbiter::becomeMaster() {
    std::lock_guard<std::mutex> lock(syncMutex_);
    
    syncState_.currentRole = ClockRole::MASTER;
    syncState_.masterId = localPeerId_;
    syncState_.masterEpoch = getCurrentTimeMicros();
    syncState_.driftCompensation = 0.0;  // Master has no drift compensation
    syncState_.syncQuality = 1.0;  // Perfect quality as master
    
    std::cout << "👑 Became timing master" << std::endl;
    
    if (onRoleChanged) {
        onRoleChanged(ClockRole::MASTER, localPeerId_);
    }
    
    broadcastMasterAnnouncement();
}

void ClockDriftArbiter::becomeSlave(const std::string& masterId) {
    std::lock_guard<std::mutex> lock(syncMutex_);
    
    syncState_.currentRole = ClockRole::SLAVE;
    syncState_.masterId = masterId;
    syncState_.lastMasterSync = getCurrentTimeMicros();
    
    std::cout << "⏰ Became slave to master: " << masterId << std::endl;
    
    if (onRoleChanged) {
        onRoleChanged(ClockRole::SLAVE, masterId);
    }
}

void ClockDriftArbiter::forceMasterRole() {
    if (!allowMasterRole_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(electionMutex_);
    
    MasterCandidate manualCandidate;
    manualCandidate.peerId = localPeerId_;
    manualCandidate.isManualOverride = true;
    manualCandidate.sessionPriority = 1000;
    
    masterCandidates_.clear();
    masterCandidates_.push_back(manualCandidate);
    
    becomeMaster();
}

void ClockDriftArbiter::addPeer(const std::string& peerId, const std::string& ipAddress, uint16_t port) {
    std::lock_guard<std::mutex> lock(peersMutex_);
    
    auto peer = std::make_unique<NetworkPeer>();
    peer->peerId = peerId;
    peer->ipAddress = ipAddress;
    peer->port = port;
    peer->role = ClockRole::UNINITIALIZED;
    peer->averageLatency = 0.0;
    peer->latencyVariance = 0.0;
    peer->clockDrift = 0.0;
    peer->lastSyncTime = getCurrentTimeMicros();
    
    peers_[peerId] = std::move(peer);
    
    std::cout << "🔗 Added peer: " << peerId << " (" << ipAddress << ":" << port << ")" << std::endl;
}

void ClockDriftArbiter::removePeer(const std::string& peerId) {
    std::lock_guard<std::mutex> lock(peersMutex_);
    
    auto it = peers_.find(peerId);
    if (it != peers_.end()) {
        std::cout << "❌ Removed peer: " << peerId << std::endl;
        peers_.erase(it);
    }
    
    // If this was our master, trigger new election
    if (syncState_.currentRole == ClockRole::SLAVE && syncState_.masterId == peerId) {
        std::cout << "⚠️ Master lost, triggering re-election" << std::endl;
        handleMasterFailure();
    }
}

uint64_t ClockDriftArbiter::getCurrentMasterTime() const {
    std::lock_guard<std::mutex> lock(syncMutex_);
    
    uint64_t localTime = getCurrentTimeMicros();
    
    if (syncState_.currentRole == ClockRole::MASTER) {
        return localTime;
    } else {
        // Apply drift compensation for slaves
        return localTime + static_cast<uint64_t>(syncState_.driftCompensation);
    }
}

uint64_t ClockDriftArbiter::compensateTimestamp(uint64_t localTimestamp) const {
    if (syncState_.currentRole == ClockRole::MASTER) {
        return localTimestamp;
    }
    
    return localTimestamp + static_cast<uint64_t>(syncState_.driftCompensation);
}

uint64_t ClockDriftArbiter::getLocalTimestamp() const {
    return getCurrentTimeMicros();
}

void ClockDriftArbiter::sendTimingPing(const std::string& peerId) {
    uint64_t timestamp = getCurrentTimeMicros();
    
    // Create timing ping message (this would be sent via network)
    if (onSendMessage) {
        std::vector<uint8_t> pingData;
        // Serialize timing ping with timestamp
        pingData.resize(8);
        *reinterpret_cast<uint64_t*>(pingData.data()) = timestamp;
        onSendMessage(peerId, pingData);
    }
}

void ClockDriftArbiter::receivePingResponse(const std::string& fromPeer, 
                                          uint64_t originalTimestamp, 
                                          uint64_t remoteTimestamp, 
                                          uint64_t returnTimestamp) {
    uint64_t currentTime = getCurrentTimeMicros();
    
    TimingSample sample;
    sample.localTimestamp = originalTimestamp;
    sample.remoteTimestamp = remoteTimestamp;
    sample.roundTripTime = currentTime - originalTimestamp;
    sample.networkLatency = sample.roundTripTime / 2;  // Rough estimate
    sample.clockOffset = calculateClockOffset(sample);
    sample.quality = calculateSampleQuality(sample);
    
    updatePeerStatistics(fromPeer, sample);
}

void ClockDriftArbiter::updatePeerStatistics(const std::string& peerId, const TimingSample& sample) {
    std::lock_guard<std::mutex> lock(peersMutex_);
    
    auto it = peers_.find(peerId);
    if (it == peers_.end()) {
        return;
    }
    
    NetworkPeer& peer = *it->second;
    peer.samples.push_back(sample);
    
    // Keep only recent samples (last 100)
    if (peer.samples.size() > 100) {
        peer.samples.erase(peer.samples.begin());
    }
    
    // Update statistics
    if (!peer.samples.empty()) {
        double totalLatency = 0.0;
        for (const auto& s : peer.samples) {
            totalLatency += s.networkLatency;
        }
        peer.averageLatency = totalLatency / peer.samples.size();
        
        // Calculate variance
        double varianceSum = 0.0;
        for (const auto& s : peer.samples) {
            double diff = s.networkLatency - peer.averageLatency;
            varianceSum += diff * diff;
        }
        peer.latencyVariance = varianceSum / peer.samples.size();
        
        // Update connection quality (inverse of variance)
        peer.connectionQuality = 1.0 / (1.0 + peer.latencyVariance);
    }
}

void ClockDriftArbiter::synchronizationLoop() {
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(syncMutex_);
        
        // Wait for sync interval
        syncCondition_.wait_for(lock, std::chrono::milliseconds(syncIntervalMs_));
        
        if (!running_.load()) {
            break;
        }
        
        // Perform synchronization based on role
        if (syncState_.currentRole == ClockRole::MASTER) {
            // Master broadcasts timing to slaves
            broadcastMasterAnnouncement();
        } else if (syncState_.currentRole == ClockRole::SLAVE) {
            // Slave requests sync from master
            requestMasterSync();
        }
        
        // Update synchronization quality
        validateSyncQuality();
        
        // Send timing pings to all peers
        {
            std::lock_guard<std::mutex> peerLock(peersMutex_);
            for (const auto& [peerId, peer] : peers_) {
                sendTimingPing(peerId);
            }
        }
    }
}

void ClockDriftArbiter::broadcastMasterAnnouncement() {
    if (syncState_.currentRole != ClockRole::MASTER) {
        return;
    }
    
    uint64_t masterTime = getCurrentTimeMicros();
    
    // Broadcast master time to all peers (implementation would use network)
    std::cout << "📡 Broadcasting master time: " << masterTime << std::endl;
}

void ClockDriftArbiter::requestMasterSync() {
    if (syncState_.currentRole != ClockRole::SLAVE) {
        return;
    }
    
    // Request sync from master (implementation would use network)
    std::cout << "🔄 Requesting sync from master: " << syncState_.masterId << std::endl;
}

void ClockDriftArbiter::handleMasterFailure() {
    if (allowMasterRole_) {
        std::cout << "🚨 Master failure detected, starting election" << std::endl;
        syncState_.currentRole = ClockRole::CANDIDATE;
        startMasterElection();
    }
}

void ClockDriftArbiter::validateSyncQuality() {
    double quality = 0.0;
    
    if (syncState_.currentRole == ClockRole::MASTER) {
        quality = 1.0;  // Master always has perfect quality
    } else if (syncState_.currentRole == ClockRole::SLAVE) {
        uint64_t timeSinceSync = getCurrentTimeMicros() - syncState_.lastMasterSync;
        // Quality degrades over time without sync
        quality = std::max(0.0, 1.0 - (timeSinceSync / 1000000.0) / 10.0);  // 10 second decay
    }
    
    if (std::abs(syncState_.syncQuality - quality) > 0.1 && onSyncQualityChanged) {
        onSyncQualityChanged(quality);
    }
    
    syncState_.syncQuality = quality;
}

uint32_t ClockDriftArbiter::getRecommendedBufferSize() const {
    std::lock_guard<std::mutex> lock(peersMutex_);
    
    double maxLatency = 0.0;
    for (const auto& [peerId, peer] : peers_) {
        maxLatency = std::max(maxLatency, peer->averageLatency + 2.0 * std::sqrt(peer->latencyVariance));
    }
    
    // Buffer size in microseconds, minimum 1ms
    return std::max(1000U, static_cast<uint32_t>(maxLatency * 2.0));
}

uint32_t ClockDriftArbiter::getMinimumBufferSize() const {
    return 1000;  // 1ms minimum buffer
}

uint64_t ClockDriftArbiter::getCurrentTimeMicros() const {
    auto now = HighResClock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

double ClockDriftArbiter::calculateClockOffset(const TimingSample& sample) const {
    // Simple clock offset calculation
    // Real implementation would use more sophisticated algorithms
    return static_cast<double>(sample.remoteTimestamp) - 
           static_cast<double>(sample.localTimestamp + sample.networkLatency);
}

double ClockDriftArbiter::calculateSampleQuality(const TimingSample& sample) const {
    // Quality based on round-trip time consistency
    // Lower RTT variance = higher quality
    return 1.0 / (1.0 + sample.roundTripTime / 1000.0);  // Normalize by 1ms
}

std::vector<std::string> ClockDriftArbiter::getConnectedPeers() const {
    std::lock_guard<std::mutex> lock(peersMutex_);
    
    std::vector<std::string> peerIds;
    for (const auto& [peerId, peer] : peers_) {
        peerIds.push_back(peerId);
    }
    
    return peerIds;
}

double ClockDriftArbiter::getNetworkLatency(const std::string& peerId) const {
    std::lock_guard<std::mutex> lock(peersMutex_);
    
    auto it = peers_.find(peerId);
    if (it != peers_.end()) {
        return it->second->averageLatency;
    }
    
    return 0.0;
}

double ClockDriftArbiter::getClockDrift(const std::string& peerId) const {
    std::lock_guard<std::mutex> lock(peersMutex_);
    
    auto it = peers_.find(peerId);
    if (it != peers_.end()) {
        return it->second->clockDrift;
    }
    
    return 0.0;
}

} // namespace TOAST
