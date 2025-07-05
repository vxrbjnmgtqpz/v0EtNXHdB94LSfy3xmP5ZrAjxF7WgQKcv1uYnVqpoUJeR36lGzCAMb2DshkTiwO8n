#pragma once

#include "gpu_timebase.h"
#include <atomic>
#include <memory>
#include <functional>

namespace jam {
namespace gpu_native {

/**
 * GPU Event Types for timeline scheduling
 */
enum class EventType : uint32_t {
    AUDIO_FRAME = 1,
    VIDEO_FRAME = 2,
    MIDI_EVENT = 3,
    TRANSPORT_CHANGE = 4,
    NETWORK_SYNC = 5,
    CUSTOM_EVENT = 6
};

/**
 * GPU Event Priority levels
 */
enum class EventPriority : uint32_t {
    REALTIME = 0,    // Ultra-low latency
    HIGH = 1,        // High priority
    NORMAL = 2,      // Normal priority
    LOW = 3          // Background processing
};

/**
 * GPU Timeline Event structure
 */
struct alignas(16) GPUTimelineEvent {
    EventType type = EventType::CUSTOM_EVENT;
    uint64_t timestamp_ns = 0;       // GPU timeline timestamp
    uint64_t event_id = 0;           // Unique event identifier
    EventPriority priority = EventPriority::NORMAL;
    uint32_t channel = 0;            // Channel/track identifier
    uint32_t data_size = 0;          // Size of associated data
    uint64_t data_offset = 0;        // Offset to event data in GPU memory
    uint32_t flags = 0;              // Event-specific flags
    uint32_t reserved = 0;           // Reserved for future use
};

/**
 * GPUSharedTimeline - Memory-mapped timeline accessible by GPU and CPU
 * 
 * This is the heart of the GPU-native architecture. The timeline lives in
 * GPU-accessible memory and is updated by GPU compute shaders, providing
 * the master clock for all multimedia operations.
 */
class GPUSharedTimelineManager {
public:
    /**
     * Initialize the GPU shared timeline memory
     * Creates memory-mapped buffer accessible by both GPU and CPU
     */
    static bool initialize(size_t timeline_buffer_size = 4096);
    
    /**
     * Shutdown and cleanup timeline memory
     */
    static void shutdown();
    
    /**
     * Get direct access to the shared timeline
     * This is the master timeline - handle with care
     */
    static GPUSharedTimeline* get_timeline();
    static const GPUSharedTimeline* get_timeline_const();
    
    /**
     * Update timeline from GPU compute shader
     * Called by GPU, not CPU threads
     */
    static void gpu_update_timeline();
    
    /**
     * Validate timeline integrity
     * Checks for corruption or invalid state
     */
    static bool validate_timeline();
    
    /**
     * Reset timeline to initial state
     * Emergency reset if timeline becomes corrupted
     */
    static void reset_timeline();
    
    /**
     * Get timeline update counter
     * Used to detect when timeline has been updated by GPU
     */
    static uint64_t get_update_counter();
    
    /**
     * Event scheduling for GPU timeline
     */
    static bool scheduleEvent(const GPUTimelineEvent& event);
    static bool cancelEvent(uint64_t event_id);
    static bool registerEventHandler(EventType type, std::function<bool(const GPUTimelineEvent&)> handler);
    
    /**
     * Check if timeline manager is initialized
     */
    static bool isInitialized();
    
    /**
     * Lock-free timeline reader
     * Safe concurrent access to timeline data
     */
    class TimelineReader {
    public:
        TimelineReader();
        
        // Get current values atomically
        gpu_timeline_t get_master_clock_ns() const;
        GPUTransportState get_transport_state() const;
        gpu_timeline_t get_transport_position_ns() const;
        uint32_t get_bpm() const;
        gpu_timeline_t get_network_timestamp_ns() const;
        gpu_timeline_t get_audio_frame_time_ns() const;
        gpu_timeline_t get_video_frame_time_ns() const;
        
        // Check if data is valid
        bool is_valid() const;
        
        // Get snapshot of entire timeline
        GPUSharedTimeline get_snapshot() const;
        
    private:
        mutable std::atomic<uint64_t> last_read_counter_;
    };
    
    /**
     * GPU timeline writer (for GPU compute shaders)
     * Only GPU should write to timeline
     */
    class TimelineWriter {
    public:
        TimelineWriter();
        
        // Update master clock (called by GPU)
        void update_master_clock(gpu_timeline_t new_time_ns);
        
        // Update transport state
        void update_transport_state(GPUTransportState state);
        void update_transport_position(gpu_timeline_t position_ns);
        void update_bpm(uint32_t bpm);
        
        // Update timing markers
        void update_audio_frame_time(gpu_timeline_t time_ns);
        void update_video_frame_time(gpu_timeline_t time_ns);
        void update_network_heartbeat(gpu_timeline_t time_ns);
        
        // Commit all updates atomically
        void commit_updates();
        
    private:
        GPUSharedTimeline pending_updates_;
        bool has_pending_updates_;
    };
    
    /**
     * Performance monitoring
     */
    struct TimelineStats {
        uint64_t total_updates;
        uint64_t updates_per_second;
        uint64_t avg_update_latency_ns;
        uint64_t max_update_latency_ns;
        uint64_t timeline_drift_ns;
        bool timeline_stable;
    };
    
    static TimelineStats get_performance_stats();
    static void reset_performance_stats();
    
private:
    // Platform-specific implementation
    class GPUSharedTimelineImpl;
    static std::unique_ptr<GPUSharedTimelineImpl> impl_;
    
    // Prevent instantiation
    GPUSharedTimelineManager() = delete;
    ~GPUSharedTimelineManager() = delete;
};

/**
 * GPU Timeline Synchronizer - Coordinates multiple timeline readers
 * Ensures all components read consistent timeline state
 */
class GPUTimelineSynchronizer {
public:
    /**
     * Create synchronizer for a group of components
     */
    explicit GPUTimelineSynchronizer(const std::string& group_name);
    ~GPUTimelineSynchronizer();
    
    /**
     * Register component for synchronized timeline access
     */
    void register_component(const std::string& component_name);
    void unregister_component(const std::string& component_name);
    
    /**
     * Wait for all registered components to sync to current timeline
     */
    bool wait_for_sync(uint32_t timeout_ms = 100);
    
    /**
     * Force sync all components to current timeline
     */
    void force_sync();
    
    /**
     * Check if all components are synchronized
     */
    bool is_synchronized() const;
    
private:
    class SynchronizerImpl;
    std::unique_ptr<SynchronizerImpl> impl_;
};

/**
 * GPU Timeline Event Queue - GPU-native event scheduling
 * Events are scheduled and dispatched based on GPU timeline
 */
template<typename EventType>
class GPUTimelineEventQueue {
public:
    struct ScheduledEvent {
        EventType event;
        gpu_timeline_t execute_time_ns;
        uint64_t sequence_id;
    };
    
    explicit GPUTimelineEventQueue(size_t max_events = 10000);
    ~GPUTimelineEventQueue();
    
    /**
     * Schedule event for execution at specific GPU time
     */
    bool schedule_event(const EventType& event, gpu_timeline_t execute_time_ns);
    
    /**
     * Get next event ready for execution
     * Returns nullptr if no events are ready
     */
    const ScheduledEvent* get_next_ready_event();
    
    /**
     * Remove executed event from queue
     */
    void remove_executed_event(uint64_t sequence_id);
    
    /**
     * Get number of pending events
     */
    size_t get_pending_count() const;
    
    /**
     * Clear all pending events
     */
    void clear();
    
private:
    class EventQueueImpl;
    std::unique_ptr<EventQueueImpl> impl_;
};

/**
 * Common GPU timeline event types
 */
struct GPUMIDIEvent {
    uint32_t midi_data;
    uint8_t channel;
    uint8_t velocity;
    uint16_t controller;
};

struct GPUTransportEvent {
    GPUTransportState new_state;
    gpu_timeline_t position_ns;
    uint32_t bpm;
};

struct GPUNetworkEvent {
    uint32_t peer_id;
    gpu_timeline_t timestamp_ns;
    std::vector<uint8_t> data;
};

// Pre-instantiated event queues for common types
using GPUMIDIEventQueue = GPUTimelineEventQueue<GPUMIDIEvent>;
using GPUTransportEventQueue = GPUTimelineEventQueue<GPUTransportEvent>;
using GPUNetworkEventQueue = GPUTimelineEventQueue<GPUNetworkEvent>;

} // namespace gpu_native
} // namespace jam
