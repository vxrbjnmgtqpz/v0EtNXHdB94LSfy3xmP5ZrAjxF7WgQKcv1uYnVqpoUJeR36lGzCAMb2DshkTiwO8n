#include "jam_framework.h"
#include "jam_parser.h"
#include "jam_multicast.h"
#include "jam_gpu.h"
#include "jam_toast.h"
#include "jam_session.h"

#include <chrono>
#include <thread>
#include <mutex>
#include <unordered_map>

namespace jam {

class JAMFramework::Impl {
public:
    JAMConfig config;
    std::unique_ptr<JAMParser> parser;
    std::unique_ptr<JAMMulticast> multicast;
    std::unique_ptr<JAMGPU> gpu;
    std::unique_ptr<JAMToast> toast;
    std::unique_ptr<JAMSession> session_manager;
    
    // Callbacks
    AudioCallback audio_callback;
    MIDICallback midi_callback;
    VideoCallback video_callback;
    SessionCallback session_callback;
    
    // State
    bool initialized = false;
    bool running = false;
    std::mutex state_mutex;
    std::thread processing_thread;
    
    // Statistics
    mutable std::mutex stats_mutex;
    Statistics stats;
    
    void processing_loop();
    void update_statistics();
};

JAMFramework::JAMFramework() : pImpl(std::make_unique<Impl>()) {}

JAMFramework::~JAMFramework() {
    shutdown();
}

bool JAMFramework::initialize(const JAMConfig& config) {
    std::lock_guard<std::mutex> lock(pImpl->state_mutex);
    
    if (pImpl->initialized) {
        return false; // Already initialized
    }
    
    pImpl->config = config;
    
    // Initialize components in dependency order
    pImpl->gpu = std::make_unique<JAMGPU>(config);
    if (!pImpl->gpu->initialize()) {
        return false;
    }
    
    pImpl->toast = std::make_unique<JAMToast>(config);
    if (!pImpl->toast->initialize()) {
        return false;
    }
    
    pImpl->parser = std::make_unique<JAMParser>(config, *pImpl->gpu, *pImpl->toast);
    if (!pImpl->parser->initialize()) {
        return false;
    }
    
    pImpl->multicast = std::make_unique<JAMMulticast>(config);
    if (!pImpl->multicast->initialize()) {
        return false;
    }
    
    pImpl->session_manager = std::make_unique<JAMSession>(config);
    if (!pImpl->session_manager->initialize()) {
        return false;
    }
    
    pImpl->initialized = true;
    return true;
}

bool JAMFramework::start_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(pImpl->state_mutex);
    
    if (!pImpl->initialized) {
        return false;
    }
    
    if (!pImpl->running) {
        pImpl->running = true;
        pImpl->processing_thread = std::thread(&JAMFramework::Impl::processing_loop, pImpl.get());
    }
    
    return pImpl->session_manager->join_session(session_id);
}

bool JAMFramework::stop_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(pImpl->state_mutex);
    
    if (!pImpl->initialized) {
        return false;
    }
    
    return pImpl->session_manager->leave_session(session_id);
}

void JAMFramework::shutdown() {
    {
        std::lock_guard<std::mutex> lock(pImpl->state_mutex);
        pImpl->running = false;
    }
    
    if (pImpl->processing_thread.joinable()) {
        pImpl->processing_thread.join();
    }
    
    // Cleanup components in reverse order
    pImpl->session_manager.reset();
    pImpl->multicast.reset();
    pImpl->parser.reset();
    pImpl->toast.reset();
    pImpl->gpu.reset();
    
    pImpl->initialized = false;
}

void JAMFramework::on_audio_stream(AudioCallback callback) {
    pImpl->audio_callback = callback;
}

void JAMFramework::on_midi_stream(MIDICallback callback) {
    pImpl->midi_callback = callback;
}

void JAMFramework::on_video_stream(VideoCallback callback) {
    pImpl->video_callback = callback;
}

void JAMFramework::on_session_event(SessionCallback callback) {
    pImpl->session_callback = callback;
}

bool JAMFramework::send_audio(const JAMAudioData& data) {
    if (!pImpl->initialized || !pImpl->running) {
        return false;
    }
    
    // Process through TOAST compression and GPU optimization
    auto compressed_data = pImpl->toast->compress_audio(data);
    return pImpl->multicast->send_audio(compressed_data);
}

bool JAMFramework::send_midi(const JAMMIDIData& data) {
    if (!pImpl->initialized || !pImpl->running) {
        return false;
    }
    
    // Apply burst logic for reliability
    auto burst_data = pImpl->toast->apply_burst_logic(data);
    return pImpl->multicast->send_midi(burst_data);
}

bool JAMFramework::send_video(const JAMVideoData& data) {
    if (!pImpl->initialized || !pImpl->running) {
        return false;
    }
    
    // GPU process direct pixel arrays
    auto processed_data = pImpl->gpu->process_video(data);
    return pImpl->multicast->send_video(processed_data);
}

JAMFramework::Statistics JAMFramework::get_statistics() const {
    std::lock_guard<std::mutex> lock(pImpl->stats_mutex);
    return pImpl->stats;
}

bool JAMFramework::is_gpu_available() const {
    return pImpl->gpu && pImpl->gpu->is_available();
}

bool JAMFramework::enable_gpu_processing(bool enable) {
    return pImpl->gpu && pImpl->gpu->set_enabled(enable);
}

void JAMFramework::Impl::processing_loop() {
    auto last_stats_update = std::chrono::steady_clock::now();
    
    while (running) {
        // Process incoming multicast data
        auto packets = multicast->receive_packets();
        
        for (const auto& packet : packets) {
            // Parse JSONL data through GPU-accelerated parser
            auto parsed_data = parser->parse_packet(packet);
            
            // Route to appropriate callbacks
            if (parsed_data.type == JAMDataType::Audio && audio_callback) {
                audio_callback(parsed_data.audio);
            } else if (parsed_data.type == JAMDataType::MIDI && midi_callback) {
                midi_callback(parsed_data.midi);
            } else if (parsed_data.type == JAMDataType::Video && video_callback) {
                video_callback(parsed_data.video);
            }
        }
        
        // Update statistics periodically
        auto now = std::chrono::steady_clock::now();
        if (now - last_stats_update > std::chrono::seconds(1)) {
            update_statistics();
            last_stats_update = now;
        }
        
        // Yield CPU briefly
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void JAMFramework::Impl::update_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex);
    
    stats.packets_received = multicast->get_packets_received();
    stats.packets_sent = multicast->get_packets_sent();
    stats.bytes_processed = parser->get_bytes_processed();
    stats.average_latency_ms = multicast->get_average_latency();
    stats.compression_ratio = toast->get_compression_ratio();
    stats.active_streams = session_manager->get_active_stream_count();
    stats.gpu_utilization = gpu->get_utilization();
}

// Utility functions
namespace utils {

std::string generate_session_id() {
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = now.time_since_epoch().count();
    return "jam_" + std::to_string(ns);
}

uint64_t get_timestamp_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint64_t>(now.time_since_epoch().count());
}

bool is_multicast_address(const std::string& address) {
    // Basic IPv4 multicast range check (224.0.0.0 to 239.255.255.255)
    return address.substr(0, 4) == "224." || 
           address.substr(0, 4) == "225." ||
           address.substr(0, 4) == "226." ||
           address.substr(0, 4) == "227." ||
           address.substr(0, 4) == "228." ||
           address.substr(0, 4) == "229." ||
           address.substr(0, 4) == "230." ||
           address.substr(0, 4) == "231." ||
           address.substr(0, 4) == "232." ||
           address.substr(0, 4) == "233." ||
           address.substr(0, 4) == "234." ||
           address.substr(0, 4) == "235." ||
           address.substr(0, 4) == "236." ||
           address.substr(0, 4) == "237." ||
           address.substr(0, 4) == "238." ||
           address.substr(0, 4) == "239.";
}

float calculate_compression_ratio(size_t original, size_t compressed) {
    if (original == 0) return 0.0f;
    return static_cast<float>(compressed) / static_cast<float>(original);
}

} // namespace utils

} // namespace jam
