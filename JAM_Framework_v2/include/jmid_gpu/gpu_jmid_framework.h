#pragma once

#include "gpu_native/gpu_timebase.h"
#include "gpu_native/gpu_shared_timeline.h"
#include <memory>
#include <functional>
#include <atomic>
#include <cstdint>

namespace jam {
namespace jmid_gpu {

/**
 * GPU-Native MIDI Event Structure
 * Aligned for GPU memory access and atomic operations
 */
struct alignas(16) GPUMIDIEvent {
    uint32_t timestamp_frame;    // GPU frame timestamp
    uint8_t status;             // MIDI status byte
    uint8_t data1;              // MIDI data byte 1
    uint8_t data2;              // MIDI data byte 2
    uint8_t channel;            // MIDI channel (0-15)
    uint32_t velocity_curve;    // Extended velocity information
    uint32_t burst_id;          // Burst deduplication ID
    uint32_t source_peer_id;    // Network source identifier
};

/**
 * GPU-Native MIDI Event Queue
 * Lock-free, GPU-accessible event scheduling
 */
class GPUMIDIEventQueue {
public:
    GPUMIDIEventQueue(size_t capacity = 4096);
    ~GPUMIDIEventQueue();

    // Schedule MIDI event at specific GPU frame
    bool schedule_event(const GPUMIDIEvent& event);
    
    // Get next ready event based on current GPU timebase
    bool get_next_ready_event(GPUMIDIEvent& event);
    
    // Bulk schedule events (optimized for JSONL parsing)
    size_t schedule_events(const GPUMIDIEvent* events, size_t count);
    
    // Queue statistics
    size_t get_pending_count() const;
    size_t get_capacity() const;
    uint32_t get_current_gpu_frame() const;
    
    // GPU memory access
    void* get_gpu_buffer() const;
    size_t get_gpu_buffer_size() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * GPU-Native JSONL MIDI Parser
 * Processes JSONL MIDI streams with GPU timestamping
 */
class GPUJSONLParser {
public:
    GPUJSONLParser(gpu_native::GPUTimebase* timebase, GPUMIDIEventQueue* event_queue);
    ~GPUJSONLParser();

    // Parse JSONL chunk with GPU timing
    struct ParseResult {
        size_t events_parsed;
        size_t events_scheduled;
        uint32_t parse_time_us;
        bool has_errors;
        std::string error_message;
    };

    ParseResult parse_jsonl_chunk(const char* data, size_t length);
    
    // Streaming JSONL processing
    void feed_data(const char* data, size_t length);
    bool has_complete_line() const;
    ParseResult process_next_line();
    
    // Burst deduplication
    void enable_burst_deduplication(bool enabled);
    void set_burst_window_frames(uint32_t frames);
    uint32_t get_deduplicated_count() const;
    
    // Performance metrics
    void reset_performance_counters();
    double get_average_parse_time() const;
    size_t get_total_events_parsed() const;
    
    // GPU compute acceleration
    bool enable_gpu_parsing(bool enabled);
    bool is_gpu_parsing_available() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * GPU-Native MIDI Dispatcher
 * Dispatches MIDI events based on GPU timebase timing
 */
class GPUMIDIDispatcher {
public:
    using MIDIOutputCallback = std::function<void(const GPUMIDIEvent& event, uint32_t frame_offset)>;
    
    GPUMIDIDispatcher(gpu_native::GPUTimebase* timebase);
    ~GPUMIDIDispatcher();

    // Register MIDI output callback
    void set_output_callback(MIDIOutputCallback callback);
    
    // Connect to event queue
    void connect_event_queue(GPUMIDIEventQueue* queue);
    
    // Start/stop dispatching based on GPU transport
    void start_dispatching();
    void stop_dispatching();
    bool is_dispatching() const;
    
    // Dispatch configuration
    void set_lookahead_frames(uint32_t frames);  // How far ahead to schedule
    void set_latency_compensation_frames(int32_t frames);  // Positive = delay, negative = advance
    
    // Real-time dispatch processing
    void process_dispatch_frame();  // Called by audio thread
    
    // Statistics
    uint32_t get_events_dispatched() const;
    uint32_t get_timing_jitter_max_us() const;
    double get_average_dispatch_latency() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * GPU-Native MIDI Transport Bridge
 * Bridges GPU timebase with legacy MIDI transport systems
 */
class GPUMIDITransportBridge {
public:
    GPUMIDITransportBridge(gpu_native::GPUTimebase* timebase);
    ~GPUMIDITransportBridge();

    // Legacy transport compatibility
    void sync_to_external_transport(uint32_t external_frame, double external_bpm);
    void set_external_sync_enabled(bool enabled);
    
    // MIDI clock generation (GPU-driven)
    void enable_midi_clock_output(bool enabled);
    void set_midi_clock_ppqn(uint16_t ppqn);  // Pulses per quarter note
    
    // Transport event generation
    void send_midi_start();
    void send_midi_stop();
    void send_midi_continue();
    void send_song_position_pointer(uint16_t position);
    
    // Sync with other JMID instances over network
    void enable_network_sync(bool enabled);
    void set_sync_peer_timeout_ms(uint32_t timeout);
    
    // Quantum synchronization
    void set_quantum_frames(uint32_t frames);  // Synchronization quantum
    void wait_for_quantum_boundary();
    uint32_t get_next_quantum_frame() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * GPU-Native JMID Framework Integration
 * Main interface for GPU-native MIDI processing
 */
class GPUJMIDFramework {
public:
    GPUJMIDFramework();
    ~GPUJMIDFramework();

    // Framework initialization
    bool initialize(uint32_t sample_rate = 48000, size_t event_queue_capacity = 4096);
    void shutdown();
    bool is_initialized() const;
    
    // Component access
    gpu_native::GPUTimebase* get_timebase() const;
    GPUMIDIEventQueue* get_event_queue() const;
    GPUJSONLParser* get_parser() const;
    GPUMIDIDispatcher* get_dispatcher() const;
    GPUMIDITransportBridge* get_transport_bridge() const;
    
    // High-level MIDI processing
    void process_jsonl_stream(const char* data, size_t length);
    void set_midi_output_callback(GPUMIDIDispatcher::MIDIOutputCallback callback);
    
    // Transport control
    void start_playback();
    void stop_playback();
    void pause_playback();
    void seek_to_frame(uint32_t frame);
    void set_bpm(uint32_t bpm);
    
    // Performance monitoring
    struct PerformanceStats {
        uint32_t events_parsed_per_second;
        uint32_t events_dispatched_per_second;
        double average_parse_latency_us;
        double average_dispatch_latency_us;
        uint32_t timing_jitter_max_us;
        uint32_t queue_utilization_percent;
    };
    
    PerformanceStats get_performance_stats() const;
    void reset_performance_stats();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace jmid_gpu
} // namespace jam
