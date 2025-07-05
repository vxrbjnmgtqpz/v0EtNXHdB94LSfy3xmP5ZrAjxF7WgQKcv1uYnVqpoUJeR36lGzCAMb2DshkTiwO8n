#pragma once

#include <memory>
#include <functional>
#include <atomic>
#include <iostream>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#endif

// Transport state definitions (must match shader)
enum class GPUTransportState : uint32_t {
    Stopped = 0,
    Playing = 1,
    Paused = 2,
    Recording = 3
};

// Transport control buffer structure (must match shader)
struct GPUTransportControlBuffer {
    GPUTransportState current_state;
    uint64_t play_start_frame;
    uint64_t pause_frame;
    uint64_t record_start_frame;
    uint64_t current_frame;
    float bpm;
    uint32_t samples_per_beat;
    uint32_t network_sync_offset;
    uint32_t frame_counter;
    float position_seconds;
    uint32_t is_network_synced;
    uint32_t padding[5]; // Align to 64 bytes
};

// Timeline event for GPU synchronization
struct GPUTimelineEvent {
    uint64_t timestamp_ns;
    uint32_t event_type;
    uint32_t event_data;
    GPUTransportState new_state;
    uint32_t source_peer_id;
};

// GPU Bars/Beats calculation buffer (must match shader)
struct GPUBarsBeatsBuffer {
    uint32_t bars;              // Current bar number (1-based)
    uint32_t beats;             // Current beat in bar (1-based)
    uint32_t subdivisions;      // Current subdivision (0-based)
    uint32_t beats_per_bar;     // Time signature numerator
    uint32_t beat_unit;         // Time signature denominator
    uint32_t subdivision_count; // Subdivisions per beat
    float total_beats;          // Total beats since start
    float fractional_beat;      // Fractional part of current beat
    uint32_t padding[8];        // Align to 64 bytes
};

namespace jam {
namespace gpu_transport {

// Forward declaration for static flag workaround
extern bool s_gpu_transport_ready;

/**
 * GPU-Native Transport Manager
 * 
 * Manages transport state (PLAY/STOP/PAUSE/RECORD) directly on GPU compute shaders
 * with frame-accurate timing and network synchronization support.
 */
class GPUTransportManager {
public:
    static GPUTransportManager& getInstance();
    
    // Initialization
    bool initialize();
    void shutdown();
    bool isInitialized() const { 
        return s_gpu_transport_ready;
    }
    
    // Transport control (these trigger GPU compute shaders)
    void play(uint64_t start_frame = 0);
    void stop();
    void pause();
    void record(uint64_t start_frame = 0);
    
    // State queries (read from GPU buffer)
    GPUTransportState getCurrentState() const;
    uint64_t getCurrentFrame() const;
    float getPositionSeconds() const;
    bool isPlaying() const;
    bool isRecording() const;
    bool isPaused() const;
    
    // Network synchronization
    void setNetworkSyncOffset(uint32_t offset_ns);
    void executeNetworkSyncCommand(uint32_t command, uint64_t sync_frame);
    bool isNetworkSynced() const;
    
    // Tempo control
    void setBPM(float bpm);
    float getBPM() const;
    
    // Bars/Beats control and queries
    void setTimeSignature(uint32_t beatsPerBar, uint32_t beatUnit);
    void setSubdivision(uint32_t subdivisionCount);
    GPUBarsBeatsBuffer getBarsBeatsInfo() const;
    
    // Timeline events (for scheduled transport changes)
    void scheduleTransportEvent(uint64_t timestamp_ns, GPUTransportState new_state);
    
    // Callbacks for state changes
    using StateChangeCallback = std::function<void(GPUTransportState old_state, GPUTransportState new_state)>;
    void setStateChangeCallback(StateChangeCallback callback);
    
    // Update cycle (call from main thread periodically)
    void update();
    
private:
    GPUTransportManager() : instance_id_(reinterpret_cast<uint64_t>(this)), metal_resources_ready_(false) { 
        std::cout << "🏗️ GPU Transport Manager constructor called for instance " << this << " (ID: " << instance_id_ << ")" << std::endl;
    }
    ~GPUTransportManager(); // Implemented in .mm file with debugging
    
    // Platform-specific implementations
    bool initializeMetal();
    bool initializeVulkan();
    
    void executeTransportUpdate();
    void readbackTransportState();
    
    // GPU resources
#ifdef __OBJC__
    id<MTLDevice> metal_device_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
    id<MTLComputePipelineState> transport_pipeline_ = nil;
    id<MTLComputePipelineState> network_sync_pipeline_ = nil;
    id<MTLComputePipelineState> tempo_update_pipeline_ = nil;
    id<MTLComputePipelineState> bars_beats_pipeline_ = nil;  // New: bars/beats calculation
    id<MTLBuffer> transport_buffer_ = nil;
    id<MTLBuffer> timebase_buffer_ = nil;
    id<MTLBuffer> timeline_events_buffer_ = nil;
    id<MTLBuffer> network_sync_buffer_ = nil;
    id<MTLBuffer> tempo_buffer_ = nil;
    id<MTLBuffer> bars_beats_buffer_ = nil;  // New: bars/beats buffer
#endif
    
    // State management
    mutable std::atomic<bool> initialized_{false};
    mutable bool debug_initialized_{false};  // Debug flag to compare
    mutable bool metal_resources_ready_{false};  // Simple Metal resources flag
    const uint64_t instance_id_;  // Unique ID for debugging
    mutable GPUTransportControlBuffer cached_state_{};
    mutable GPUBarsBeatsBuffer cached_bars_beats_{};  // Cached bars/beats state
    StateChangeCallback state_callback_;
    GPUTransportState last_state_ = GPUTransportState::Stopped;
    
    // Timeline events
    static constexpr size_t MAX_TIMELINE_EVENTS = 256;
    std::vector<GPUTimelineEvent> pending_events_;
    
    // Thread safety
    mutable std::atomic<bool> state_dirty_{true};
    
    // Non-copyable and non-movable
    GPUTransportManager(const GPUTransportManager&) = delete;
    GPUTransportManager& operator=(const GPUTransportManager&) = delete;
    GPUTransportManager(GPUTransportManager&&) = delete;
    GPUTransportManager& operator=(GPUTransportManager&&) = delete;
};

} // namespace gpu_transport
} // namespace jam
