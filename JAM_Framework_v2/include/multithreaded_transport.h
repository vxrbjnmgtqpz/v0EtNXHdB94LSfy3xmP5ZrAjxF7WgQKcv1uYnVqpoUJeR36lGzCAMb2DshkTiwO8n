/**
 * JAM Framework v2: Multi-threaded UDP Transport with GPU Acceleration
 * 
 * Features:
 * - Multiple send/receive worker threads
 * - GPU-accelerated burst processing
 * - Redundant transmission paths
 * - Automatic load balancing
 */

#pragma once

#include "jam_transport.h"
#include "compute_pipeline.h"
#include <thread>
#include <vector>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace jam {

/**
 * Multi-threaded UDP Transport Configuration
 */
struct MultiThreadConfig {
    int send_threads = 4;           // Number of send worker threads
    int recv_threads = 2;           // Number of receive worker threads  
    int gpu_threads = 1;            // Number of GPU processing threads
    bool enable_redundancy = true;  // Send packets on multiple paths
    bool enable_gpu_burst = true;   // Use GPU for burst processing
    int max_queue_size = 1000;      // Max queued packets per thread
};

/**
 * Packet for multi-threaded processing
 */
struct PacketJob {
    std::vector<uint8_t> data;
    std::string destination_ip;
    uint16_t destination_port;
    bool use_burst = false;
    int burst_count = 1;
    uint32_t priority = 0;          // 0 = highest priority
    std::chrono::high_resolution_clock::time_point timestamp;
};

/**
 * Multi-threaded UDP Transport with GPU acceleration
 */
class MultiThreadedUDPTransport : public UDPTransport {
public:
    MultiThreadedUDPTransport(const std::string& multicast_group, 
                             uint16_t port, 
                             const std::string& interface_ip,
                             const MultiThreadConfig& config = {});
    
    ~MultiThreadedUDPTransport() override;
    
    // Transport interface
    bool initialize() override;
    bool start_receiving(std::function<void(std::span<const uint8_t>)> callback) override;
    void stop_receiving() override;
    bool send(std::span<const uint8_t> data) override;
    TransportStats get_stats() const override;
    uint32_t generate_sequence_number() override;
    
    // Multi-threaded operations
    bool send_with_priority(std::span<const uint8_t> data, uint32_t priority = 0);
    bool send_burst_gpu(std::span<const uint8_t> data, int burst_count = 3);
    bool send_redundant(std::span<const uint8_t> data, int path_count = 2);
    
    // GPU acceleration
    bool initialize_gpu_backend(std::shared_ptr<ComputePipeline> gpu_pipeline);
    void enable_gpu_processing(bool enable) { gpu_processing_enabled_ = enable; }
    
    // Performance monitoring
    struct ThreadStats {
        std::atomic<uint64_t> packets_sent{0};
        std::atomic<uint64_t> packets_received{0};
        std::atomic<uint64_t> queue_overflows{0};
        std::atomic<double> avg_queue_latency_us{0.0};
    };
    
    ThreadStats get_thread_stats(int thread_id) const;
    int get_active_thread_count() const;

private:
    MultiThreadConfig config_;
    std::string multicast_group_;
    uint16_t port_;
    std::string interface_ip_;
    
    // Socket management
    std::vector<int> send_sockets_;
    std::vector<int> recv_sockets_;
    bool sockets_initialized_ = false;
    
    // Threading
    std::atomic<bool> running_{false};
    std::vector<std::thread> send_workers_;
    std::vector<std::thread> recv_workers_;
    std::thread gpu_worker_;
    
    // Send queues (one per thread)
    std::vector<std::queue<PacketJob>> send_queues_;
    std::vector<std::mutex> send_queue_mutexes_;
    std::vector<std::condition_variable> send_queue_cvs_;
    
    // GPU processing queue
    std::queue<PacketJob> gpu_queue_;
    std::mutex gpu_queue_mutex_;
    std::condition_variable gpu_queue_cv_;
    
    // Statistics per thread
    mutable std::vector<ThreadStats> thread_stats_;
    mutable std::mutex stats_mutex_;
    TransportStats global_stats_{};
    
    // Sequence numbers
    std::atomic<uint32_t> sequence_counter_{1};
    
    // GPU backend
    std::shared_ptr<ComputePipeline> gpu_pipeline_;
    std::atomic<bool> gpu_processing_enabled_{false};
    
    // Receive callback
    std::function<void(std::span<const uint8_t>)> receive_callback_;
    
    // Load balancing
    std::atomic<int> next_send_thread_{0};
    
    // Private methods
    bool init_sockets();
    void cleanup_sockets();
    
    // Worker thread functions
    void send_worker_loop(int worker_id);
    void recv_worker_loop(int worker_id);
    void gpu_worker_loop();
    
    // Load balancing
    int select_send_thread();
    void distribute_packet(const PacketJob& job);
    
    // GPU processing
    std::vector<uint8_t> process_burst_gpu(const std::vector<uint8_t>& data, int burst_count);
    void apply_gpu_prediction(std::vector<uint8_t>& data);
};

} // namespace jam
