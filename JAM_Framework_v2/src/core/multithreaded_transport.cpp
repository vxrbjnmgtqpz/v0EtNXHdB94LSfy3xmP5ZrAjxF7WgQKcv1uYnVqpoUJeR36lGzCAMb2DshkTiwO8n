/**
 * JAM Framework v2: Multi-threaded UDP Transport Implementation
 */

#include "multithreaded_transport.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <random>

namespace jam {

MultiThreadedUDPTransport::MultiThreadedUDPTransport(const std::string& multicast_group, 
                                                    uint16_t port, 
                                                    const std::string& interface_ip,
                                                    const MultiThreadConfig& config)
    : config_(config), multicast_group_(multicast_group), port_(port), interface_ip_(interface_ip) {
    
    // Initialize thread-local storage
    send_queues_.resize(config_.send_threads);
    send_queue_mutexes_.resize(config_.send_threads);
    send_queue_cvs_.resize(config_.send_threads);
    thread_stats_.resize(config_.send_threads + config_.recv_threads);
    
    // Initialize sockets
    send_sockets_.resize(config_.send_threads);
    recv_sockets_.resize(config_.recv_threads);
}

MultiThreadedUDPTransport::~MultiThreadedUDPTransport() {
    stop_receiving();
    cleanup_sockets();
}

bool MultiThreadedUDPTransport::initialize() {
    return init_sockets();
}

bool MultiThreadedUDPTransport::init_sockets() {
    // Create send sockets (one per send thread for parallel transmission)
    for (int i = 0; i < config_.send_threads; ++i) {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            cleanup_sockets();
            return false;
        }
        
        // Enable broadcast/multicast
        int broadcast = 1;
        if (setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) < 0) {
            cleanup_sockets();
            return false;
        }
        
        // Set multicast TTL
        int ttl = 1;
        if (setsockopt(sock, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) < 0) {
            cleanup_sockets();
            return false;
        }
        
        send_sockets_[i] = sock;
    }
    
    // Create receive sockets (one per receive thread for parallel processing)
    for (int i = 0; i < config_.recv_threads; ++i) {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            cleanup_sockets();
            return false;
        }
        
        // Enable address reuse for multiple threads
        int reuse = 1;
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
            cleanup_sockets();
            return false;
        }
        
        #ifdef SO_REUSEPORT
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse)) < 0) {
            cleanup_sockets();
            return false;
        }
        #endif
        
        // Bind to multicast address
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);
        
        if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            cleanup_sockets();
            return false;
        }
        
        // Join multicast group
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(multicast_group_.c_str());
        mreq.imr_interface.s_addr = INADDR_ANY;
        
        if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
            cleanup_sockets();
            return false;
        }
        
        recv_sockets_[i] = sock;
    }
    
    sockets_initialized_ = true;
    return true;
}

void MultiThreadedUDPTransport::cleanup_sockets() {
    for (int sock : send_sockets_) {
        if (sock >= 0) close(sock);
    }
    for (int sock : recv_sockets_) {
        if (sock >= 0) close(sock);
    }
    send_sockets_.clear();
    recv_sockets_.clear();
    sockets_initialized_ = false;
}

bool MultiThreadedUDPTransport::start_receiving(std::function<void(std::span<const uint8_t>)> callback) {
    if (!sockets_initialized_) return false;
    
    receive_callback_ = callback;
    running_ = true;
    
    // Start send worker threads
    for (int i = 0; i < config_.send_threads; ++i) {
        send_workers_.emplace_back(&MultiThreadedUDPTransport::send_worker_loop, this, i);
    }
    
    // Start receive worker threads
    for (int i = 0; i < config_.recv_threads; ++i) {
        recv_workers_.emplace_back(&MultiThreadedUDPTransport::recv_worker_loop, this, i);
    }
    
    // Start GPU worker thread if enabled
    if (config_.enable_gpu_burst && gpu_pipeline_) {
        gpu_worker_ = std::thread(&MultiThreadedUDPTransport::gpu_worker_loop, this);
    }
    
    return true;
}

void MultiThreadedUDPTransport::stop_receiving() {
    running_ = false;
    
    // Wake up all worker threads
    for (auto& cv : send_queue_cvs_) {
        cv.notify_all();
    }
    gpu_queue_cv_.notify_all();
    
    // Join all threads
    for (auto& worker : send_workers_) {
        if (worker.joinable()) worker.join();
    }
    for (auto& worker : recv_workers_) {
        if (worker.joinable()) worker.join();
    }
    if (gpu_worker_.joinable()) {
        gpu_worker_.join();
    }
    
    send_workers_.clear();
    recv_workers_.clear();
}

bool MultiThreadedUDPTransport::send(std::span<const uint8_t> data) {
    return send_with_priority(data, 0); // Default priority
}

bool MultiThreadedUDPTransport::send_with_priority(std::span<const uint8_t> data, uint32_t priority) {
    if (!running_ || data.empty()) return false;
    
    PacketJob job;
    job.data.assign(data.begin(), data.end());
    job.destination_ip = multicast_group_;
    job.destination_port = port_;
    job.priority = priority;
    job.timestamp = std::chrono::high_resolution_clock::now();
    
    distribute_packet(job);
    return true;
}

bool MultiThreadedUDPTransport::send_burst_gpu(std::span<const uint8_t> data, int burst_count) {
    if (!running_ || data.empty()) return false;
    
    if (config_.enable_gpu_burst && gpu_pipeline_ && gpu_processing_enabled_) {
        // Queue for GPU processing
        PacketJob job;
        job.data.assign(data.begin(), data.end());
        job.destination_ip = multicast_group_;
        job.destination_port = port_;
        job.use_burst = true;
        job.burst_count = burst_count;
        job.timestamp = std::chrono::high_resolution_clock::now();
        
        {
            std::lock_guard<std::mutex> lock(gpu_queue_mutex_);
            if (gpu_queue_.size() < config_.max_queue_size) {
                gpu_queue_.push(job);
                gpu_queue_cv_.notify_one();
                return true;
            }
        }
        return false; // Queue full
    } else {
        // Fallback to CPU burst
        for (int i = 0; i < burst_count; ++i) {
            send_with_priority(data, 0); // High priority for burst
        }
        return true;
    }
}

bool MultiThreadedUDPTransport::send_redundant(std::span<const uint8_t> data, int path_count) {
    if (!config_.enable_redundancy || !running_) return false;
    
    path_count = std::min(path_count, config_.send_threads);
    
    for (int i = 0; i < path_count; ++i) {
        PacketJob job;
        job.data.assign(data.begin(), data.end());
        job.destination_ip = multicast_group_;
        job.destination_port = port_;
        job.priority = 0; // High priority for redundant packets
        job.timestamp = std::chrono::high_resolution_clock::now();
        
        // Send on specific thread for path diversity
        int thread_id = i % config_.send_threads;
        {
            std::lock_guard<std::mutex> lock(send_queue_mutexes_[thread_id]);
            if (send_queues_[thread_id].size() < config_.max_queue_size) {
                send_queues_[thread_id].push(job);
                send_queue_cvs_[thread_id].notify_one();
            }
        }
    }
    
    return true;
}

void MultiThreadedUDPTransport::distribute_packet(const PacketJob& job) {
    int thread_id = select_send_thread();
    
    {
        std::lock_guard<std::mutex> lock(send_queue_mutexes_[thread_id]);
        if (send_queues_[thread_id].size() < config_.max_queue_size) {
            send_queues_[thread_id].push(job);
            send_queue_cvs_[thread_id].notify_one();
        } else {
            thread_stats_[thread_id].queue_overflows++;
        }
    }
}

int MultiThreadedUDPTransport::select_send_thread() {
    // Round-robin load balancing
    return next_send_thread_.fetch_add(1) % config_.send_threads;
}

void MultiThreadedUDPTransport::send_worker_loop(int worker_id) {
    auto& queue = send_queues_[worker_id];
    auto& queue_mutex = send_queue_mutexes_[worker_id];
    auto& queue_cv = send_queue_cvs_[worker_id];
    auto& stats = thread_stats_[worker_id];
    
    int sock = send_sockets_[worker_id];
    
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        // Wait for packets or shutdown
        queue_cv.wait(lock, [&]() { return !queue.empty() || !running_; });
        
        while (!queue.empty() && running_) {
            PacketJob job = queue.front();
            queue.pop();
            lock.unlock();
            
            // Calculate queue latency
            auto now = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::microseconds>(now - job.timestamp);
            stats.avg_queue_latency_us = latency.count();
            
            // Send packet
            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = inet_addr(job.destination_ip.c_str());
            addr.sin_port = htons(job.destination_port);
            
            ssize_t sent = sendto(sock, job.data.data(), job.data.size(), 0,
                                (struct sockaddr*)&addr, sizeof(addr));
            
            if (sent > 0) {
                stats.packets_sent++;
            }
            
            lock.lock();
        }
    }
}

void MultiThreadedUDPTransport::recv_worker_loop(int worker_id) {
    auto& stats = thread_stats_[config_.send_threads + worker_id];
    int sock = recv_sockets_[worker_id];
    
    std::vector<uint8_t> buffer(65536); // Max UDP packet size
    
    while (running_) {
        struct sockaddr_in sender_addr;
        socklen_t sender_len = sizeof(sender_addr);
        
        ssize_t received = recvfrom(sock, buffer.data(), buffer.size(), 0,
                                  (struct sockaddr*)&sender_addr, &sender_len);
        
        if (received > 0 && receive_callback_) {
            stats.packets_received++;
            
            // Call callback with received data
            std::span<const uint8_t> data(buffer.data(), received);
            receive_callback_(data);
        }
    }
}

void MultiThreadedUDPTransport::gpu_worker_loop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(gpu_queue_mutex_);
        
        // Wait for GPU jobs or shutdown
        gpu_queue_cv_.wait(lock, [&]() { return !gpu_queue_.empty() || !running_; });
        
        while (!gpu_queue_.empty() && running_) {
            PacketJob job = gpu_queue_.front();
            gpu_queue_.pop();
            lock.unlock();
            
            // Process burst with GPU acceleration
            if (job.use_burst && gpu_pipeline_) {
                auto processed_data = process_burst_gpu(job.data, job.burst_count);
                
                // Send processed burst packets
                for (int i = 0; i < job.burst_count; ++i) {
                    std::span<const uint8_t> burst_data(processed_data.data(), processed_data.size());
                    send_with_priority(burst_data, 0);
                }
            }
            
            lock.lock();
        }
    }
}

std::vector<uint8_t> MultiThreadedUDPTransport::process_burst_gpu(const std::vector<uint8_t>& data, int burst_count) {
    // This would use the GPU pipeline to process and optimize burst transmission
    // For now, return the original data with GPU prediction applied
    std::vector<uint8_t> processed_data = data;
    apply_gpu_prediction(processed_data);
    return processed_data;
}

void MultiThreadedUDPTransport::apply_gpu_prediction(std::vector<uint8_t>& data) {
    // Apply GPU-based prediction/correction to the data
    // This would use the PNBTR system for predictive transmission
    if (gpu_pipeline_) {
        // TODO: Implement GPU prediction pipeline
        // gpu_pipeline_->predict_missing_data(data);
    }
}

bool MultiThreadedUDPTransport::initialize_gpu_backend(std::shared_ptr<ComputePipeline> gpu_pipeline) {
    gpu_pipeline_ = gpu_pipeline;
    gpu_processing_enabled_ = true;
    return gpu_pipeline_ != nullptr;
}

TransportStats MultiThreadedUDPTransport::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    TransportStats stats = global_stats_;
    
    // Aggregate thread statistics
    for (const auto& thread_stat : thread_stats_) {
        stats.packets_sent += thread_stat.packets_sent.load();
        stats.packets_received += thread_stat.packets_received.load();
    }
    
    return stats;
}

MultiThreadedUDPTransport::ThreadStats MultiThreadedUDPTransport::get_thread_stats(int thread_id) const {
    if (thread_id >= 0 && thread_id < thread_stats_.size()) {
        return thread_stats_[thread_id];
    }
    return {};
}

int MultiThreadedUDPTransport::get_active_thread_count() const {
    return send_workers_.size() + recv_workers_.size() + (gpu_worker_.joinable() ? 1 : 0);
}

uint32_t MultiThreadedUDPTransport::generate_sequence_number() {
    return sequence_counter_.fetch_add(1);
}

} // namespace jam
