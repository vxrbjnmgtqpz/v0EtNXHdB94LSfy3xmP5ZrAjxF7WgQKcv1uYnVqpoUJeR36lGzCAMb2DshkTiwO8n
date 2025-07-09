#include "network_simulator.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>

namespace pnbtr_jellie {

NetworkSimulator::NetworkSimulator() : rng_(std::random_device{}()), 
                                       uniform_dist_(0.0, 1.0),
                                       normal_dist_(0.0, 1.0),
                                       exponential_dist_(1.0) {
    last_stats_update_ = std::chrono::high_resolution_clock::now();
    last_bandwidth_check_ = std::chrono::high_resolution_clock::now();
    bytes_sent_this_second_ = 0;
}

NetworkSimulator::~NetworkSimulator() {
    shutdown();
}

bool NetworkSimulator::initialize(const NetworkConditions& conditions) {
    if (is_running_.load()) {
        return false;
    }
    
    conditions_ = conditions;
    is_initialized_.store(true);
    is_running_.store(true);
    
    // Start simulation thread if continuous mode is enabled
    if (conditions_.continuous_mode) {
        simulation_thread_ = std::thread(&NetworkSimulator::simulationThread, this);
    }
    
    return true;
}

void NetworkSimulator::shutdown() {
    if (is_running_.load()) {
        is_running_.store(false);
        if (simulation_thread_.joinable()) {
            simulation_thread_.join();
        }
    }
    is_initialized_.store(false);
}

void NetworkSimulator::updateNetworkConditions(const NetworkConditions& conditions) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    conditions_ = conditions;
}

bool NetworkSimulator::sendPacket(const std::vector<uint8_t>& data, uint64_t sequence_number) {
    if (!is_running_.load()) {
        return false;
    }
    
    stats_.packets_sent++;
    
    // Check if packet should be dropped
    if (shouldDropPacket()) {
        stats_.packets_lost++;
        if (log_events_.load()) {
            logNetworkEvent(NetworkEvent::PACKET_LOST, sequence_number);
        }
        return false;
    }
    
    NetworkPacket packet;
    packet.data = data;
    packet.sequence_number = sequence_number;
    packet.timestamp = std::chrono::high_resolution_clock::now();
    
    // Calculate latency
    double latency_ms = calculateLatency();
    packet.scheduled_delivery = packet.timestamp + 
        std::chrono::milliseconds(static_cast<int64_t>(latency_ms));
    
    // Check for reordering
    if (shouldReorderPacket()) {
        packet.is_reordered = true;
        packet.reorder_delay_samples = static_cast<uint32_t>(uniform_dist_(rng_) * conditions_.max_reorder_distance);
        stats_.packets_reordered++;
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        incoming_queue_.push(packet);
    }
    
    updateLatencyStats(latency_ms);
    
    if (log_events_.load()) {
        logNetworkEvent(NetworkEvent::PACKET_SENT, sequence_number, latency_ms);
    }
    
    return true;
}

bool NetworkSimulator::receivePacket(NetworkPacket& packet) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    auto now = std::chrono::high_resolution_clock::now();
    
    // Process delayed packets
    while (!delayed_packets_.empty()) {
        // Note: priority_queue doesn't have top() that returns non-const reference
        // This is a simplified implementation
        break;
    }
    
    if (outgoing_queue_.empty()) {
        return false;
    }
    
    packet = outgoing_queue_.front();
    outgoing_queue_.pop();
    
    stats_.packets_delivered++;
    
    if (log_events_.load()) {
        logNetworkEvent(NetworkEvent::PACKET_DELIVERED, packet.sequence_number);
    }
    
    return true;
}

void NetworkSimulator::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.packets_sent.store(0);
    stats_.packets_delivered.store(0);
    stats_.packets_lost.store(0);
    stats_.packets_reordered.store(0);
    stats_.average_latency_ms.store(0.0);
    stats_.jitter_ms.store(0.0);
    stats_.current_bandwidth_kbps.store(0.0);
}

std::vector<NetworkSimulator::NetworkEvent> NetworkSimulator::getRecentEvents(size_t max_events) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    if (event_log_.size() <= max_events) {
        return event_log_;
    }
    
    return std::vector<NetworkEvent>(event_log_.end() - max_events, event_log_.end());
}

// Static preset methods
NetworkConditions NetworkSimulator::createLowLatencyScenario() {
    NetworkConditions conditions;
    conditions.base_latency_ms = 20.0;
    conditions.jitter_variance_ms = 2.0;
    conditions.packet_loss_percentage = 1.0;
    conditions.bandwidth_kbps = 5000;  // 5 Mbps
    conditions.enable_bandwidth_limiting = false;
    return conditions;
}

NetworkConditions NetworkSimulator::createTypicalScenario() {
    NetworkConditions conditions;
    conditions.base_latency_ms = 50.0;
    conditions.jitter_variance_ms = 10.0;
    conditions.packet_loss_percentage = 2.0;
    conditions.bandwidth_kbps = 1000;  // 1 Mbps
    conditions.enable_bandwidth_limiting = true;
    return conditions;
}

NetworkConditions NetworkSimulator::createStressScenario() {
    NetworkConditions conditions;
    conditions.base_latency_ms = 100.0;
    conditions.jitter_variance_ms = 30.0;
    conditions.packet_loss_percentage = 10.0;
    conditions.bandwidth_kbps = 256;   // 256 kbps
    conditions.enable_bandwidth_limiting = true;
    conditions.enable_burst_loss = true;
    return conditions;
}

NetworkConditions NetworkSimulator::createJitterScenario() {
    NetworkConditions conditions;
    conditions.base_latency_ms = 50.0;
    conditions.jitter_variance_ms = 50.0;  // High jitter
    conditions.packet_loss_percentage = 2.0;
    conditions.bandwidth_kbps = 1000;
    return conditions;
}

NetworkConditions NetworkSimulator::createBurstLossScenario() {
    NetworkConditions conditions;
    conditions.base_latency_ms = 50.0;
    conditions.jitter_variance_ms = 10.0;
    conditions.packet_loss_percentage = 5.0;
    conditions.bandwidth_kbps = 1000;
    conditions.enable_burst_loss = true;
    conditions.burst_duration_ms = 50.0;
    return conditions;
}

// Private methods
void NetworkSimulator::simulationThread() {
    while (is_running_.load()) {
        processPacketQueue();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void NetworkSimulator::processPacketQueue() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    auto now = std::chrono::high_resolution_clock::now();
    
    // Move packets from incoming to outgoing when their delivery time arrives
    while (!incoming_queue_.empty()) {
        NetworkPacket& packet = incoming_queue_.front();
        
        if (now >= packet.scheduled_delivery) {
            outgoing_queue_.push(packet);
            incoming_queue_.pop();
        } else {
            break;
        }
    }
}

bool NetworkSimulator::shouldDropPacket() {
    double loss_rate = conditions_.packet_loss_percentage / 100.0;
    
    if (conditions_.enable_burst_loss && in_loss_burst_) {
        auto now = std::chrono::high_resolution_clock::now();
        auto burst_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - burst_start_time_).count();
        
        if (burst_duration > conditions_.burst_duration_ms) {
            in_loss_burst_ = false;
        } else {
            return true; // Drop packet during burst
        }
    }
    
    if (uniform_dist_(rng_) < loss_rate) {
        if (conditions_.enable_burst_loss && !in_loss_burst_) {
            in_loss_burst_ = true;
            burst_start_time_ = std::chrono::high_resolution_clock::now();
        }
        return true;
    }
    
    return false;
}

double NetworkSimulator::calculateLatency() {
    normal_dist_ = std::normal_distribution<double>(
        conditions_.base_latency_ms, 
        conditions_.jitter_variance_ms);
    
    double latency = normal_dist_(rng_);
    return std::max(0.0, latency);
}

bool NetworkSimulator::shouldReorderPacket() {
    if (!conditions_.enable_out_of_order_delivery) {
        return false;
    }
    
    return uniform_dist_(rng_) < (conditions_.reorder_probability / 100.0);
}

void NetworkSimulator::applyBandwidthLimiting() {
    // Implementation for bandwidth limiting
    // This would track bytes sent per second and add delays if needed
}

void NetworkSimulator::updateLatencyStats(double latency_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    double current_avg = stats_.average_latency_ms.load();
    uint64_t packet_count = stats_.packets_delivered.load() + 1;
    
    double new_avg = (current_avg * (packet_count - 1) + latency_ms) / packet_count;
    stats_.average_latency_ms.store(new_avg);
    
    // Simple jitter calculation (could be improved)
    double jitter = std::abs(latency_ms - current_avg);
    stats_.jitter_ms.store(jitter);
}

void NetworkSimulator::logNetworkEvent(NetworkEvent::Type type, uint64_t seq_num, 
                                      double latency_ms, const std::string& desc) {
    if (!log_events_.load()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    NetworkEvent event;
    event.timestamp = std::chrono::high_resolution_clock::now();
    event.type = type;
    event.sequence_number = seq_num;
    event.latency_ms = latency_ms;
    event.description = desc;
    
    event_log_.push_back(event);
    
    // Keep event log size manageable
    if (event_log_.size() > 10000) {
        event_log_.erase(event_log_.begin(), event_log_.begin() + 1000);
    }
}

// NetworkSimulatorBridge implementation (stubs for linking)
NetworkSimulatorBridge::NetworkSimulatorBridge(NetworkSimulator* simulator) 
    : simulator_(simulator) {
    std::cout << "NetworkSimulatorBridge initialized\n";
}

bool NetworkSimulatorBridge::sendAudioPacket(const std::vector<float>& audio_data, 
                                            uint32_t sample_rate, uint32_t channels) {
    // Stub implementation for now
    (void)audio_data; (void)sample_rate; (void)channels;
    return true;
}

bool NetworkSimulatorBridge::receiveAudioPacket(std::vector<float>& audio_data, 
                                               uint32_t& sample_rate, uint32_t& channels) {
    // Stub implementation for now
    (void)audio_data; (void)sample_rate; (void)channels;
    return false;
}

std::vector<NetworkSimulatorBridge::AudioPacketEvent> NetworkSimulatorBridge::getTrainingData(size_t max_events) {
    // Stub implementation
    (void)max_events;
    return {};
}

void NetworkSimulatorBridge::clearTrainingData() {
    // Stub implementation
}

} // namespace pnbtr_jellie
