#pragma once

#include <cstdint>
#include <string>
#include <functional>
#include <memory>
#include <vector>

namespace jam {

// Forward declarations
class JAMParser;
class JAMMulticast;
class JAMGPU;
class JAMToast;
class JAMSession;

// Configuration
struct JAMConfig {
    std::string multicast_group = "239.255.77.77";
    uint16_t port = 7777;
    
    enum GPUBackend {
        JAM_GPU_VULKAN,
        JAM_GPU_METAL,
        JAM_GPU_AUTO
    } gpu_backend = JAM_GPU_AUTO;
    
    enum CompressionLevel {
        JAM_TOAST_NONE,
        JAM_TOAST_BASIC,
        JAM_TOAST_OPTIMIZED,
        JAM_TOAST_MAXIMUM
    } compression_level = JAM_TOAST_OPTIMIZED;
    
    bool enable_burst_logic = true;
    uint32_t max_streams = 1000;
    float target_latency_ms = 3.0f;
};

// Data structures
struct JAMAudioData {
    std::string session_id;
    uint64_t timestamp_ns;
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bit_depth;
    std::vector<float> samples;
    
    // PNBTR metadata
    bool pnbtr_processed = false;
    float prediction_confidence = 0.0f;
};

struct JAMMIDIData {
    std::string session_id;
    uint64_t timestamp_ns;
    uint8_t status;
    uint8_t data1;
    uint8_t data2;
    
    // Burst logic metadata
    uint8_t burst_count = 1;
    uint16_t sequence_id = 0;
    bool is_duplicate = false;
};

struct JAMVideoData {
    std::string session_id;
    uint64_t timestamp_ns;
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> pixel_data; // Direct pixel array, no base64
    
    // GPU processing metadata
    bool gpu_processed = false;
    uint32_t shader_id = 0;
};

// Callback types
using AudioCallback = std::function<void(const JAMAudioData&)>;
using MIDICallback = std::function<void(const JAMMIDIData&)>;
using VideoCallback = std::function<void(const JAMVideoData&)>;
using SessionCallback = std::function<void(const std::string& session_id, bool joined)>;

// Main JAM Framework class
class JAMFramework {
public:
    JAMFramework();
    ~JAMFramework();
    
    // Core lifecycle
    bool initialize(const JAMConfig& config);
    bool start_session(const std::string& session_id);
    bool stop_session(const std::string& session_id);
    void shutdown();
    
    // Stream handling
    void on_audio_stream(AudioCallback callback);
    void on_midi_stream(MIDICallback callback);
    void on_video_stream(VideoCallback callback);
    void on_session_event(SessionCallback callback);
    
    // Stream sending
    bool send_audio(const JAMAudioData& data);
    bool send_midi(const JAMMIDIData& data);
    bool send_video(const JAMVideoData& data);
    
    // Performance monitoring
    struct Statistics {
        uint64_t packets_received = 0;
        uint64_t packets_sent = 0;
        uint64_t bytes_processed = 0;
        float average_latency_ms = 0.0f;
        float compression_ratio = 0.0f;
        uint32_t active_streams = 0;
        float gpu_utilization = 0.0f;
    };
    
    Statistics get_statistics() const;
    
    // GPU integration
    bool is_gpu_available() const;
    bool enable_gpu_processing(bool enable);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Utility functions
namespace utils {
    std::string generate_session_id();
    uint64_t get_timestamp_ns();
    bool is_multicast_address(const std::string& address);
    float calculate_compression_ratio(size_t original, size_t compressed);
}

} // namespace jam
